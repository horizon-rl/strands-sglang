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

"""SGLang HTTP client with connection pooling and retry logic.

Aligned with slime's http_utils.py for RL training stability:
- Aggressive retry (60 attempts by default)
- Retries on all transient errors
- 15mins timeout by default for long generations
- Non-streaming POST for better parallelism (no SSE overhead)

Uses aiohttp for high-concurrency performance.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from .exceptions import (
    SGLangClientError,
    SGLangConnectionError,
    SGLangContextLengthError,
    SGLangDecodingError,
    SGLangHTTPError,
    SGLangThrottledError,
)

logger = logging.getLogger(__name__)

# OpenAI's default connection limit (from openai/_constants.py)
DEFAULT_MAX_CONNECTIONS = 1000

# Non-retryable HTTP status codes
#
# Reference: OpenAI Python SDK (_base_client.py) retries: 408, 409, 429, 5xx
# Reference: slime (http_utils.py) retries ALL errors for local SGLang servers
#
# Our hybrid approach for local SGLang during RL training:
# - 401/403/404: Don't retry (auth/routing errors won't self-resolve)
# - 400 with context length error: Don't retry (prompt too long won't fix itself)
# - 400 other: Retry (transient for local servers - weight reloading, memory pressure)
# - 408/409/429/5xx: Retry (same as OpenAI SDK)
# - Connection errors: Retry (same as OpenAI SDK)
NON_RETRYABLE_STATUS_CODES = {401, 403, 404}  # Auth failed, forbidden, endpoint not found

# Single-source-of-truth patterns for context length errors in SGLang responses.
# Used by _classify_http_error to detect non-retryable 400 errors.
CONTEXT_LENGTH_PATTERNS = ("exceed", "too long", "maximum length", "context length")


class SGLangClient:
    """Async HTTP client for SGLang server with connection pooling and retry.

    Designed for RL training stability with aggressive retry on transient errors.
    Aligned with slime's http_utils.py approach.

    Uses non-streaming POST requests for better parallelism in high-concurrency
    training scenarios (no SSE overhead, connections released immediately).

    Example:
        >>> async with SGLangClient(base_url="http://localhost:30000") as client:
        ...     result = await client.generate(input_ids=[1, 2, 3])
        ...     print(result["text"])

        >>> # For RL training with infinite timeout (like slime):
        >>> client = SGLangClient(base_url="http://localhost:30000", timeout=None)

        >>> # From slime training args (via cached factory):
        >>> from strands_sglang import get_client_from_slime_args
        >>> client = get_client_from_slime_args(args)
    """

    def __init__(
        self,
        base_url: str,
        *,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        timeout: float | None = 900.0,
        connect_timeout: float = 5.0,
        max_retries: int = 60,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize SGLang client.

        Args:
            base_url: SGLang server URL (e.g., "http://localhost:30000").
            max_connections: Maximum concurrent connections (default: 1000).
            timeout: Request timeout in seconds, or None for infinite (default: 900.0).
            connect_timeout: TCP connection timeout in seconds (default: 5s).
            max_retries: Maximum retry attempts on transient errors (default: 60, like slime).
            retry_delay: Delay between retries in seconds (default: 1.0).
        """
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Store config for lazy session creation (avoids aiohttp warning about creating session outside event loop)
        self._max_connections = max_connections
        self._timeout = aiohttp.ClientTimeout(total=timeout, connect=connect_timeout)
        self._session: aiohttp.ClientSession | None = None

        logger.info(
            f"SGLangClient initialized: base_url={self.base_url}, "
            f"max_connections={max_connections}, "
            f"timeout={timeout}, max_retries={max_retries}"
        )

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session (lazy initialization)."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                base_url=self.base_url,
                timeout=self._timeout,
                connector=aiohttp.TCPConnector(limit=self._max_connections),
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> SGLangClient:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        await self.close()

    @staticmethod
    def _classify_http_error(status: int, body: str) -> SGLangHTTPError:
        """Classify an HTTP error into a specific custom exception.

        This is the single source of truth for error classification. All HTTP errors
        from SGLang are mapped to custom exceptions here, so that sglang.py never
        needs to inspect raw status codes or response bodies.

        Args:
            status: HTTP status code.
            body: Response body text.

        Returns:
            Appropriate SGLangHTTPError subclass instance.
        """
        # Context length exceeded (400 + length keywords) — non-retryable
        if status == 400:
            body_lower = body.lower()
            if any(p in body_lower for p in CONTEXT_LENGTH_PATTERNS):
                return SGLangContextLengthError(f"Context length exceeded (400): {body}", status=status, body=body)

        # Rate-limited or temporarily unavailable — retryable
        if status in (429, 503):
            return SGLangThrottledError(f"Service throttled ({status}): {body}", status=status, body=body)

        # All other HTTP errors
        return SGLangHTTPError(f"HTTP {status}: {body}", status=status, body=body)

    def _is_retryable_error(self, e: Exception) -> bool:
        """Check if an error is retryable.

        Aligned with slime's philosophy: retry aggressively on most errors.
        For local SGLang servers, most 400 errors are transient (weight reloading, memory pressure).

        Non-retryable:
        - 401/403/404: Auth/routing errors that won't self-resolve
        - 400 with context length keywords: Prompt too long, retrying won't help
        """
        if isinstance(e, SGLangHTTPError):
            # Non-retryable: auth/routing errors
            if e.status in NON_RETRYABLE_STATUS_CODES:
                return False
            # Non-retryable: context length exceeded
            if isinstance(e, SGLangContextLengthError):
                return False
            # Retry everything else: 5xx, 408, 429, other 400s, etc.
            return True
        # Retry all connection/timeout/decoding errors
        return True

    async def generate(self, input_ids: list[int], **kwargs: Any) -> dict[str, Any]:
        """Generate from SGLang `/generate` endpoint.

        Args:
            input_ids: Input token IDs. Do not set `text` when `input_ids` is provided.
            **kwargs: Additional parameters passed directly to SGLang (see full list in SGLang documentation).

        Returns:
            Response dict with text, output_ids, meta_info (logprobs, finish_reason, etc.).

        Raises:
            SGLangContextLengthError: When prompt exceeds model's maximum context length.
            SGLangThrottledError: On 429 or 503 responses.
            SGLangHTTPError: For non-retryable HTTP errors (401, 403, 404) or after all retries exhausted.
            SGLangConnectionError: For connection/timeout failures after retries exhausted.
            SGLangDecodingError: When server returns non-JSON response after retries exhausted.
        """
        payload: dict[str, Any] = {
            "input_ids": input_ids,
            **kwargs,
            "stream": False,  # override kwargs to non-streaming for RL training
        }

        last_error: Exception | None = None
        session = self._get_session()

        for attempt in range(self.max_retries + 1):
            try:
                async with session.post("/generate", json=payload) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        raise self._classify_http_error(resp.status, body)

                    # Success path: parse JSON directly
                    try:
                        return await resp.json(content_type=None)
                    except Exception as e:
                        # Non-JSON response — treat as retryable error
                        raise SGLangDecodingError(f"Invalid JSON response: {e}") from e

            except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
                last_error = SGLangConnectionError(str(e))
                last_error.__cause__ = e

            except SGLangClientError as e:
                last_error = e

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    raise

            except Exception as e:
                # Unexpected errors — wrap to prevent library internals leaking
                last_error = SGLangClientError(str(e))
                last_error.__cause__ = e

            # Log and retry
            error_detail = str(last_error)
            if attempt < self.max_retries:
                logger.warning(
                    f"SGLang request failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"{type(last_error).__name__}: {error_detail}. Retrying in {self.retry_delay}s..."
                )
                await asyncio.sleep(self.retry_delay)
            else:
                logger.error(
                    f"SGLang request failed after {self.max_retries + 1} attempts: "
                    f"{type(last_error).__name__}: {error_detail}"
                )
                raise last_error

        raise RuntimeError("Unreachable: loop must return or raise")

    async def health(self) -> bool:
        """Check if SGLang server is healthy.

        Returns:
            True if server responds OK to `/health` endpoint, False otherwise.
        """
        try:
            session = self._get_session()
            async with session.get("/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def get_model_info(self) -> dict[str, Any] | None:
        """Get model information from the SGLang server.

        Returns:
            Dict containing model info from `/get_model_info` endpoint, or None on error.
            Important fields include:
            - model_path: HuggingFace model ID or local path
            - tokenizer_path: Tokenizer path (may differ from model_path)
        """
        try:
            session = self._get_session()
            async with session.get("/get_model_info") as resp:
                if resp.status >= 400:
                    return None
                return await resp.json(content_type=None)
        except Exception:
            return None
