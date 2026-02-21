# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for SGLangClient (mocked, no server required)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from strands_sglang.client import NON_RETRYABLE_STATUS_CODES, SGLangClient
from strands_sglang.exceptions import (
    SGLangClientError,
    SGLangConnectionError,
    SGLangContextLengthError,
    SGLangDecodingError,
    SGLangHTTPError,
    SGLangThrottledError,
)


class TestSGLangClientInit:
    """Tests for SGLangClient initialization."""

    def test_default_config(self):
        """Default configuration values."""
        client = SGLangClient(base_url="http://localhost:30000")

        assert client.base_url == "http://localhost:30000"
        assert client.max_retries == 60
        assert client.retry_delay == 1.0

    def test_base_url_strips_trailing_slash(self):
        """Base URL trailing slash is stripped."""
        client = SGLangClient(base_url="http://localhost:30000/")
        assert client.base_url == "http://localhost:30000"

    def test_custom_config(self):
        """Custom configuration is applied."""
        client = SGLangClient(
            base_url="http://custom:9000",
            max_connections=500,
            timeout=120.0,
            max_retries=10,
            retry_delay=2.0,
        )

        assert client.base_url == "http://custom:9000"
        assert client.max_retries == 10
        assert client.retry_delay == 2.0


class TestGetSession:
    """Tests for _get_session lazy initialization and event loop detection."""

    async def test_creates_session_on_first_call(self):
        """First call to _get_session creates a new session."""
        client = SGLangClient(base_url="http://localhost:30000")
        assert client._session is None

        session = client._get_session()
        assert session is not None
        assert client._connector is not None

    async def test_reuses_session_on_subsequent_calls(self):
        """Subsequent calls return the same session."""
        client = SGLangClient(base_url="http://localhost:30000")
        session1 = client._get_session()
        session2 = client._get_session()
        assert session1 is session2

    async def test_recreates_session_when_loop_changes(self):
        """Session is recreated when the event loop has changed."""
        client = SGLangClient(base_url="http://localhost:30000")

        # Simulate a session from a different (old) loop
        mock_session = MagicMock()
        mock_session.closed = False
        old_loop = MagicMock()  # Not the current running loop

        mock_connector = MagicMock()
        mock_connector.closed = False

        client._session = mock_session
        client._connector = mock_connector
        client._loop = old_loop

        new_session = client._get_session()
        assert new_session is not mock_session  # Got a fresh session
        mock_connector._close.assert_called_once()  # Old connector cleaned up

    async def test_recreates_session_when_closed(self):
        """Session is recreated when explicitly closed."""
        client = SGLangClient(base_url="http://localhost:30000")

        mock_session = MagicMock()
        mock_session.closed = True

        mock_connector = MagicMock()
        mock_connector.closed = False

        client._session = mock_session
        client._connector = mock_connector
        client._loop = asyncio.get_running_loop()  # Same loop, but session closed

        new_session = client._get_session()
        assert new_session is not mock_session

    def test_real_session_survives_loop_change(self):
        """Real aiohttp session from loop A works in loop B via _get_session recreation.

        This reproduces the actual bug: a shared/cached client creates a session in one
        event loop, then the loop changes (pytest-asyncio, RL training restart). Without
        the loop detection fix, the stale session would raise 'Event loop is closed'.
        """
        client = SGLangClient(base_url="http://localhost:30000")

        # Loop A: create session
        async def create_in_loop_a():
            session = client._get_session()
            return id(session)

        session_a_id = asyncio.run(create_in_loop_a())
        # Loop A is now closed

        # Loop B: _get_session should detect the loop change and create a new session
        async def use_in_loop_b():
            session = client._get_session()
            # Session should be different (new loop)
            assert id(session) != session_a_id
            # Session should be usable (not bound to dead loop)
            assert not session.closed

        asyncio.run(use_in_loop_b())



class TestClassifyHTTPError:
    """Tests for _classify_http_error static method."""

    def test_400_with_context_length_pattern_returns_context_length_error(self):
        """400 with 'exceed' in body returns SGLangContextLengthError."""
        error = SGLangClient._classify_http_error(400, "Input token count exceed the max model len")
        assert isinstance(error, SGLangContextLengthError)
        assert error.status == 400

    def test_400_with_too_long_pattern_returns_context_length_error(self):
        """400 with 'too long' in body returns SGLangContextLengthError."""
        error = SGLangClient._classify_http_error(400, "Prompt is too long for this model")
        assert isinstance(error, SGLangContextLengthError)

    def test_400_without_length_pattern_returns_generic_error(self):
        """400 without length keywords returns generic SGLangHTTPError."""
        error = SGLangClient._classify_http_error(400, "Bad request: invalid parameter")
        assert type(error) is SGLangHTTPError
        assert error.status == 400

    def test_429_returns_throttled_error(self):
        """429 returns SGLangThrottledError."""
        error = SGLangClient._classify_http_error(429, "Rate limited")
        assert isinstance(error, SGLangThrottledError)
        assert error.status == 429

    def test_503_returns_throttled_error(self):
        """503 returns SGLangThrottledError."""
        error = SGLangClient._classify_http_error(503, "Service unavailable")
        assert isinstance(error, SGLangThrottledError)
        assert error.status == 503

    def test_500_returns_generic_error(self):
        """500 returns generic SGLangHTTPError."""
        error = SGLangClient._classify_http_error(500, "Internal server error")
        assert type(error) is SGLangHTTPError
        assert error.status == 500

    def test_body_preserved_on_error(self):
        """Error body is preserved for upstream consumers."""
        error = SGLangClient._classify_http_error(400, "Input token count exceed the max model len")
        assert error.body == "Input token count exceed the max model len"


class TestRetryableErrors:
    """Tests for _is_retryable_error method.

    Aligned with SLIME: retry aggressively on most errors.
    From OpenAI: 408 (Request Timeout) and 429 (Rate Limited) ARE retried.
    """

    @pytest.fixture
    def client(self):
        return SGLangClient(base_url="http://localhost:30000")

    # --- Connection errors (always retryable) ---

    def test_connection_error_is_retryable(self, client):
        """SGLangConnectionError is retryable."""
        error = SGLangConnectionError("Connection refused")
        assert client._is_retryable_error(error) is True

    def test_decoding_error_is_retryable(self, client):
        """SGLangDecodingError is retryable."""
        error = SGLangDecodingError("Invalid JSON")
        assert client._is_retryable_error(error) is True

    def test_generic_exception_is_retryable(self, client):
        """Generic exceptions are retryable (SLIME philosophy)."""
        error = ValueError("Something wrong")
        assert client._is_retryable_error(error) is True

    # --- HTTP 5xx (always retryable) ---

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504, 507, 599])
    def test_5xx_errors_are_retryable(self, client, status_code):
        """All HTTP 5xx errors are retryable."""
        error = SGLangHTTPError("Server error", status=status_code, body="")
        assert client._is_retryable_error(error) is True

    # --- HTTP 4xx retryable (from OpenAI) ---

    def test_408_request_timeout_is_retryable(self, client):
        """HTTP 408 (Request Timeout) is retryable (from OpenAI)."""
        error = SGLangHTTPError("Request timeout", status=408, body="")
        assert client._is_retryable_error(error) is True

    def test_429_rate_limit_is_retryable(self, client):
        """HTTP 429 (Rate Limited) is retryable (from OpenAI)."""
        error = SGLangThrottledError("Rate limited", status=429, body="")
        assert client._is_retryable_error(error) is True

    # --- HTTP 4xx non-retryable (client errors) ---

    @pytest.mark.parametrize("status_code", NON_RETRYABLE_STATUS_CODES)
    def test_client_errors_not_retryable(self, client, status_code):
        """Client errors (401/403/404) are not retryable."""
        error = SGLangHTTPError("Client error", status=status_code, body="")
        assert client._is_retryable_error(error) is False

    # --- Context length (non-retryable) ---

    def test_context_length_error_not_retryable(self, client):
        """SGLangContextLengthError is not retryable."""
        error = SGLangContextLengthError("Too long", status=400, body="exceed max model len")
        assert client._is_retryable_error(error) is False


class TestHealth:
    """Tests for health method."""

    @pytest.mark.asyncio
    async def test_health_returns_true_on_200(self):
        """Health returns True on 200 response."""
        client = SGLangClient(base_url="http://localhost:30000")

        # Mock _get_session to return a mock session with async context manager
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        client._get_session = MagicMock(return_value=mock_session)

        result = await client.health()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_returns_false_on_error(self):
        """Health returns False on connection error."""
        client = SGLangClient(base_url="http://localhost:30000")

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Connection refused"))
        client._get_session = MagicMock(return_value=mock_session)

        result = await client.health()
        assert result is False


def _mock_response(status: int, body: str = "", json_data: dict | None = None):
    """Create a mock aiohttp response with async context manager support."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.text = AsyncMock(return_value=body)
    if json_data is not None:
        mock_resp.json = AsyncMock(return_value=json_data)
    else:
        mock_resp.json = AsyncMock(side_effect=Exception("Not JSON"))
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)
    return mock_resp


def _client_with_mock_session(mock_resp=None, side_effect=None):
    """Create an SGLangClient with a mocked session."""
    client = SGLangClient(base_url="http://localhost:30000", max_retries=0)
    mock_session = MagicMock()
    if side_effect:
        mock_session.post = MagicMock(side_effect=side_effect)
    else:
        mock_session.post = MagicMock(return_value=mock_resp)
    client._get_session = MagicMock(return_value=mock_session)
    return client


class TestGenerateErrors:
    """Tests for generate() error handling with mocked HTTP responses.

    These tests verify that each exception type is correctly raised
    when the server returns specific HTTP responses. Complements the
    integration tests which validate against real SGLang servers.
    """

    async def test_context_length_error_raised_on_400_with_exceed(self):
        """400 with 'exceed' in body raises SGLangContextLengthError."""
        resp = _mock_response(400, body="Input token count exceed the max model len of 4096")
        client = _client_with_mock_session(resp)

        with pytest.raises(SGLangContextLengthError) as exc_info:
            await client.generate(input_ids=[1, 2, 3])

        assert exc_info.value.status == 400
        assert "exceed" in exc_info.value.body.lower()

    async def test_throttled_error_raised_on_429(self):
        """429 raises SGLangThrottledError (retryable, but non-retryable at max_retries=0)."""
        resp = _mock_response(429, body="Rate limited")
        client = _client_with_mock_session(resp)

        with pytest.raises(SGLangThrottledError) as exc_info:
            await client.generate(input_ids=[1, 2, 3])

        assert exc_info.value.status == 429

    async def test_throttled_error_raised_on_503(self):
        """503 raises SGLangThrottledError."""
        resp = _mock_response(503, body="Service unavailable")
        client = _client_with_mock_session(resp)

        with pytest.raises(SGLangThrottledError) as exc_info:
            await client.generate(input_ids=[1, 2, 3])

        assert exc_info.value.status == 503

    async def test_http_error_raised_on_401(self):
        """401 raises SGLangHTTPError (non-retryable, immediate)."""
        resp = _mock_response(401, body="Unauthorized")
        client = _client_with_mock_session(resp)

        with pytest.raises(SGLangHTTPError) as exc_info:
            await client.generate(input_ids=[1, 2, 3])

        assert exc_info.value.status == 401

    async def test_http_error_raised_on_404(self):
        """404 raises SGLangHTTPError (non-retryable, immediate)."""
        resp = _mock_response(404, body="Not found")
        client = _client_with_mock_session(resp)

        with pytest.raises(SGLangHTTPError) as exc_info:
            await client.generate(input_ids=[1, 2, 3])

        assert exc_info.value.status == 404

    async def test_connection_error_on_connector_failure(self):
        """aiohttp.ClientConnectorError is wrapped in SGLangConnectionError."""
        conn_info = MagicMock()
        conn_info.__str__ = MagicMock(return_value="localhost:30000")
        error = aiohttp.ClientConnectorError(conn_info, OSError("Connection refused"))
        client = _client_with_mock_session(side_effect=error)

        with pytest.raises(SGLangConnectionError):
            await client.generate(input_ids=[1, 2, 3])

    async def test_connection_error_on_timeout(self):
        """asyncio.TimeoutError is wrapped in SGLangConnectionError."""
        client = _client_with_mock_session(side_effect=asyncio.TimeoutError())

        with pytest.raises(SGLangConnectionError):
            await client.generate(input_ids=[1, 2, 3])

    async def test_decoding_error_on_invalid_json(self):
        """Non-JSON success response raises SGLangDecodingError."""
        resp = _mock_response(200, body="<html>not json</html>")
        # json() raises on non-JSON
        resp.json = AsyncMock(side_effect=Exception("Expecting value"))
        client = _client_with_mock_session(resp)

        with pytest.raises(SGLangDecodingError):
            await client.generate(input_ids=[1, 2, 3])

    async def test_successful_generate(self):
        """Successful response returns parsed JSON."""
        expected = {"text": "hello", "output_ids": [1, 2], "meta_info": {}}
        resp = _mock_response(200, json_data=expected)
        client = _client_with_mock_session(resp)

        result = await client.generate(input_ids=[1, 2, 3])

        assert result == expected

    @patch("strands_sglang.client.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_500_then_succeeds(self, mock_sleep):
        """500 is retried and succeeds on second attempt."""
        error_resp = _mock_response(500, body="Internal server error")
        success_resp = _mock_response(200, json_data={"text": "ok", "output_ids": [], "meta_info": {}})

        client = SGLangClient(base_url="http://localhost:30000", max_retries=1, retry_delay=0.0)
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=[error_resp, success_resp])
        client._get_session = MagicMock(return_value=mock_session)

        result = await client.generate(input_ids=[1, 2, 3])

        assert result["text"] == "ok"
        assert mock_session.post.call_count == 2

    @patch("strands_sglang.client.asyncio.sleep", new_callable=AsyncMock)
    async def test_no_retry_on_context_length_error(self, mock_sleep):
        """Context length error is NOT retried even with max_retries > 0."""
        resp = _mock_response(400, body="Input token count exceed the max model len")
        client = SGLangClient(base_url="http://localhost:30000", max_retries=5, retry_delay=0.0)
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=resp)
        client._get_session = MagicMock(return_value=mock_session)

        with pytest.raises(SGLangContextLengthError):
            await client.generate(input_ids=[1, 2, 3])

        # Should only be called once — no retries
        assert mock_session.post.call_count == 1


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """All custom exceptions inherit from SGLangClientError."""
        assert issubclass(SGLangHTTPError, SGLangClientError)
        assert issubclass(SGLangContextLengthError, SGLangClientError)
        assert issubclass(SGLangThrottledError, SGLangClientError)
        assert issubclass(SGLangConnectionError, SGLangClientError)
        assert issubclass(SGLangDecodingError, SGLangClientError)

    def test_context_length_is_http_error(self):
        """SGLangContextLengthError is an SGLangHTTPError."""
        assert issubclass(SGLangContextLengthError, SGLangHTTPError)

    def test_throttled_is_http_error(self):
        """SGLangThrottledError is an SGLangHTTPError."""
        assert issubclass(SGLangThrottledError, SGLangHTTPError)

    def test_http_error_attributes(self):
        """SGLangHTTPError stores status and body."""
        error = SGLangHTTPError("test", status=500, body="error body")
        assert error.status == 500
        assert error.body == "error body"
        assert str(error) == "test"
