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

from unittest.mock import MagicMock, patch

import httpx
import pytest

from strands_sglang.client import NON_RETRYABLE_STATUS_CODES, SGLangClient


class TestSGLangClientInit:
    """Tests for SGLangClient initialization."""

    def test_default_config(self):
        """Default configuration values."""
        client = SGLangClient("http://localhost:30000")

        assert client.base_url == "http://localhost:30000"
        assert client.max_retries == 60
        assert client.retry_delay == 1.0

    def test_base_url_strips_trailing_slash(self):
        """Base URL trailing slash is stripped."""
        client = SGLangClient("http://localhost:30000/")
        assert client.base_url == "http://localhost:30000"

    def test_custom_config(self):
        """Custom configuration is applied."""
        client = SGLangClient(
            "http://custom:9000",
            max_connections=500,
            timeout=120.0,
            max_retries=10,
            retry_delay=2.0,
        )

        assert client.base_url == "http://custom:9000"
        assert client.max_retries == 10
        assert client.retry_delay == 2.0


class TestRetryableErrors:
    """Tests for _is_retryable_error method.

    Aligned with SLIME: retry aggressively on most errors.
    From OpenAI: 408 (Request Timeout) and 429 (Rate Limited) ARE retried.
    """

    @pytest.fixture
    def client(self):
        return SGLangClient("http://localhost:30000")

    # --- Connection errors (always retryable) ---

    def test_connect_error_is_retryable(self, client):
        """ConnectError is retryable."""
        error = httpx.ConnectError("Connection refused")
        assert client._is_retryable_error(error) is True

    def test_read_timeout_is_retryable(self, client):
        """ReadTimeout is retryable."""
        error = httpx.ReadTimeout("Read timed out")
        assert client._is_retryable_error(error) is True

    def test_pool_timeout_is_retryable(self, client):
        """PoolTimeout is retryable."""
        error = httpx.PoolTimeout("Pool exhausted")
        assert client._is_retryable_error(error) is True

    def test_generic_exception_is_retryable(self, client):
        """Generic exceptions are retryable (SLIME philosophy)."""
        error = ValueError("Something wrong")
        assert client._is_retryable_error(error) is True

    # --- HTTP 5xx (always retryable) ---

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504, 507, 599])
    def test_5xx_errors_are_retryable(self, client, status_code):
        """All HTTP 5xx errors are retryable."""
        response = MagicMock()
        response.status_code = status_code
        error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
        assert client._is_retryable_error(error) is True

    # --- HTTP 4xx retryable (from OpenAI) ---

    def test_408_request_timeout_is_retryable(self, client):
        """HTTP 408 (Request Timeout) is retryable (from OpenAI)."""
        response = MagicMock()
        response.status_code = 408
        error = httpx.HTTPStatusError("Request timeout", request=MagicMock(), response=response)
        assert client._is_retryable_error(error) is True

    def test_429_rate_limit_is_retryable(self, client):
        """HTTP 429 (Rate Limited) is retryable (from OpenAI)."""
        response = MagicMock()
        response.status_code = 429
        error = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=response)
        assert client._is_retryable_error(error) is True

    # --- HTTP 4xx non-retryable (client errors) ---

    @pytest.mark.parametrize("status_code", NON_RETRYABLE_STATUS_CODES)
    def test_client_errors_not_retryable(self, client, status_code):
        """Client errors (4xx except 408/429) are not retryable."""
        response = MagicMock()
        response.status_code = status_code
        error = httpx.HTTPStatusError("Client error", request=MagicMock(), response=response)
        assert client._is_retryable_error(error) is False


class TestSSEParsing:
    """Tests for _iter_sse_events method."""

    @pytest.fixture
    def client(self):
        return SGLangClient("http://localhost:30000")

    @pytest.mark.asyncio
    async def test_parse_valid_sse_events(self, client):
        """Parse valid SSE events."""
        response = MagicMock()
        response.aiter_lines.return_value = AsyncIteratorMock(
            [
                'data: {"text": "Hello"}',
                'data: {"text": "Hello world"}',
                "data: [DONE]",
            ]
        )

        events = [e async for e in client._iter_sse_events(response)]

        assert len(events) == 2
        assert events[0] == {"text": "Hello"}
        assert events[1] == {"text": "Hello world"}

    @pytest.mark.asyncio
    async def test_skip_empty_lines(self, client):
        """Empty lines are skipped."""
        response = MagicMock()
        response.aiter_lines.return_value = AsyncIteratorMock(
            [
                "",
                'data: {"text": "test"}',
                "",
                "data: [DONE]",
            ]
        )

        events = [e async for e in client._iter_sse_events(response)]

        assert len(events) == 1
        assert events[0] == {"text": "test"}

    @pytest.mark.asyncio
    async def test_skip_non_data_lines(self, client):
        """Non-data lines are skipped."""
        response = MagicMock()
        response.aiter_lines.return_value = AsyncIteratorMock(
            [
                "event: message",
                'data: {"text": "test"}',
                ": comment",
                "data: [DONE]",
            ]
        )

        events = [e async for e in client._iter_sse_events(response)]

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_skip_malformed_json(self, client):
        """Malformed JSON is skipped."""
        response = MagicMock()
        response.aiter_lines.return_value = AsyncIteratorMock(
            [
                "data: not json",
                'data: {"text": "valid"}',
                "data: [DONE]",
            ]
        )

        events = [e async for e in client._iter_sse_events(response)]

        assert len(events) == 1
        assert events[0] == {"text": "valid"}


class TestHealth:
    """Tests for health method."""

    @pytest.mark.asyncio
    async def test_health_returns_true_on_200(self):
        """Health returns True on 200 response."""
        with patch.object(httpx.AsyncClient, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            client = SGLangClient("http://localhost:30000")
            result = await client.health()

            assert result is True

    @pytest.mark.asyncio
    async def test_health_returns_false_on_error(self):
        """Health returns False on HTTP error."""
        with patch.object(httpx.AsyncClient, "get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            client = SGLangClient("http://localhost:30000")
            result = await client.health()

            assert result is False


class AsyncIteratorMock:
    """Mock async iterator for testing."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item
