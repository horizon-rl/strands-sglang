# Copyright 2025-2026 Strands RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGLang HTTP client with connection pooling and retry logic."""

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

    Notes:
        - Designed for RL training stability with aggressive retry on transient errors.
        Aligned with slime's `http_utils.py` approach.
        - Uses non-streaming POST requests for better parallelism in high-concurrency
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

        # Store config for lazy session creation (connector has event loop affinity)
        self._max_connections = max_connections
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._session: aiohttp.ClientSession | None = None
        self._is_multimodal: bool | None = None

        logger.info(
            "SGLangClient initialized: base_url=%s, max_connections=%s, timeout=%s, max_retries=%s",
            self.base_url,
            max_connections,
            timeout,
            max_retries,
        )

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session (lazy initialization)."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                base_url=self.base_url,
                timeout=aiohttp.ClientTimeout(total=self._timeout, connect=self._connect_timeout),
                connector=aiohttp.TCPConnector(limit=self._max_connections),
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def __del__(self) -> None:
        """Sync cleanup to prevent aiohttp 'Unclosed client session' warnings at shutdown."""
        if self._session is not None and not self._session.closed:
            if self._session.connector is not None and not self._session.connector.closed:
                self._session.connector._close()
            self._session._connector = None

    async def __aenter__(self) -> SGLangClient:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        await self.close()

    @staticmethod
    def _classify_http_error(status: int, body: str) -> SGLangHTTPError:
        """Classify an HTTP error into a specific custom exception.

        Single source of truth for error classification — `sglang.py` never
        inspects raw status codes or response bodies.
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
        """Call SGLang `/generate` endpoint with retry.

        Notes:
            Non-retryable: 401/403/404 and context-length 400s. All other errors are retried.
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
                    "SGLang request failed (attempt %d/%d): %s: %s. Retrying in %ss...",
                    attempt + 1,
                    self.max_retries + 1,
                    type(last_error).__name__,
                    error_detail,
                    self.retry_delay,
                )
                await asyncio.sleep(self.retry_delay)
            else:
                logger.error(
                    "SGLang request failed after %d attempts: %s: %s",
                    self.max_retries + 1,
                    type(last_error).__name__,
                    error_detail,
                )
                raise last_error

        raise RuntimeError("Unreachable: loop must return or raise")

    async def health(self) -> bool:
        """Check if SGLang server is healthy."""
        try:
            session = self._get_session()
            async with session.get("/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def model_info(self) -> dict[str, Any] | None:
        """Get model information from the SGLang server.

        Returns:
            Dict containing model info from `/model_info` endpoint, or None on error.
            Important fields include:
            - model_path: HuggingFace model ID or local path
            - tokenizer_path: Tokenizer path (may differ from model_path)
        """
        try:
            session = self._get_session()
            async with session.get("/model_info") as resp:
                if resp.status >= 400:
                    return None
                return await resp.json(content_type=None)
        except Exception:
            return None

    async def is_multimodal(self) -> bool:
        """Check if the server's model supports multimodal (image) input.

        Queries `/model_info` for the `has_image_understanding` field. Result is cached
        after the first successful query.
        """
        if self._is_multimodal is not None:
            return self._is_multimodal
        info = await self.model_info()
        self._is_multimodal = bool(info.get("has_image_understanding", False)) if info else False
        return self._is_multimodal
