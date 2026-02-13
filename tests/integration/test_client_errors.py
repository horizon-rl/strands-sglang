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

"""Integration tests for SGLangClient error classification.

Validates that real SGLang server error messages are correctly classified
into custom exception types. This is critical because our CONTEXT_LENGTH_PATTERNS
must match the actual error text returned by the server.

Fixtures (model, tokenizer, sglang_base_url) are provided by conftest.py.
"""

import pytest

from strands_sglang.client import SGLangClient
from strands_sglang.exceptions import SGLangConnectionError, SGLangContextLengthError, SGLangHTTPError


class TestContextLengthError:
    """Verify SGLang's actual error messages trigger SGLangContextLengthError."""

    async def test_input_exceeding_context_length_raises_context_length_error(self, sglang_base_url):
        """Sending input_ids longer than model context should raise SGLangContextLengthError.

        This test validates that our CONTEXT_LENGTH_PATTERNS in client.py match
        the real error message returned by the SGLang server.
        """
        async with SGLangClient(base_url=sglang_base_url, max_retries=0) as client:
            # Send input longer than any typical model context window (4K-128K tokens).
            # Must stay under aiohttp's TOO_LARGE_BYTES_BODY (1MB) to avoid ResourceWarning.
            # 500K single-digit token IDs ≈ 1M bytes in JSON, so use 400K to be safe.
            oversized_input_ids = [1] * 400_000

            with pytest.raises(SGLangContextLengthError) as exc_info:
                await client.generate(input_ids=oversized_input_ids)

            # Verify error carries useful information
            assert exc_info.value.status == 400
            assert len(exc_info.value.body) > 0


class TestConnectionError:
    """Verify connection failures are wrapped in SGLangConnectionError."""

    async def test_unreachable_server_raises_connection_error(self):
        """Connecting to a dead port should raise SGLangConnectionError."""
        async with SGLangClient(
            base_url="http://localhost:1",  # Port 1 — almost certainly not running SGLang
            max_retries=0,
            connect_timeout=1.0,
        ) as client:
            with pytest.raises(SGLangConnectionError):
                await client.generate(input_ids=[1, 2, 3])


class TestHTTPError:
    """Verify non-retryable HTTP errors surface correctly."""

    async def test_404_raises_http_error(self, sglang_base_url):
        """Requesting a non-existent endpoint should raise SGLangHTTPError with status 404.

        Uses a raw aiohttp session via the client to hit a bad path,
        validating that the client wraps the error correctly.
        """
        async with SGLangClient(base_url=sglang_base_url, max_retries=0) as client:
            session = client._get_session()

            # Hit a path that doesn't exist on SGLang
            async with session.get("/nonexistent_endpoint_12345") as resp:
                if resp.status == 404:
                    body = await resp.text()
                    error = SGLangClient._classify_http_error(resp.status, body)
                    assert isinstance(error, SGLangHTTPError)
                    assert error.status == 404
                else:
                    pytest.skip(f"Server returned {resp.status} instead of 404 for unknown endpoint")
