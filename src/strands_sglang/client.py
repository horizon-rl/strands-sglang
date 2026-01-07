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

"""SGLang HTTP client with connection pooling, retry logic, and SSE parsing.

Aligned with SLIME's http_utils.py for RL training stability:
- Aggressive retry (60 attempts by default)
- Retries on all transient errors (like SLIME)
- Infinite timeout by default for long generations
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

import httpx

logger = logging.getLogger(__name__)

# OpenAI's default connection limit (from openai/_constants.py)
DEFAULT_MAX_CONNECTIONS = 1000

# Non-retryable HTTP status codes (client errors that won't self-resolve)
# Everything else is retried (aligned with SLIME's aggressive retry philosophy)
# Note: 408 (Request Timeout) and 429 (Rate Limited) ARE retried (from OpenAI)
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 405, 406, 409, 410, 411, 412, 413, 414, 415, 422}


class SGLangClient:
    """Async HTTP client for SGLang server with connection pooling and retry.

    Designed for RL training stability with aggressive retry on transient errors.
    Aligned with SLIME's http_utils.py approach.

    Example:
        >>> async with SGLangClient("http://localhost:30000") as client:
        ...     async for event in client.generate(input_ids=[1, 2, 3]):
        ...         print(event)

        >>> # For RL training with infinite timeout (like SLIME):
        >>> client = SGLangClient("http://localhost:30000", timeout=None)

        >>> # From SLIME training args:
        >>> client = SGLangClient.from_slime_args(args)
    """

    def __init__(
        self,
        base_url: str,
        *,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        timeout: float | None = None,
        connect_timeout: float = 5.0,
        max_retries: int = 60,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize SGLang client.

        Args:
            base_url: SGLang server URL (e.g., "http://localhost:30000").
            max_connections: Maximum concurrent connections (default: 1000).
            timeout: Request timeout in seconds, or None for infinite (default: None, like SLIME).
            connect_timeout: TCP connection timeout in seconds (default: 5s).
            max_retries: Maximum retry attempts on transient errors (default: 60, like SLIME).
            retry_delay: Delay between retries in seconds (default: 1.0).
        """
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # timeout=None means infinite wait (like SLIME's httpx.Timeout(None))
        http_timeout = httpx.Timeout(timeout, connect=connect_timeout) if timeout else httpx.Timeout(None)

        # FIX: Set max_keepalive_connections equal to max_connections to avoid connection churn
        # By default httpx only keeps 20 connections alive, which can cause issues with high concurrency
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=http_timeout,
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        )

        logger.info(
            f"SGLangClient initialized: base_url={self.base_url}, "
            f"max_connections={max_connections}, max_keepalive={max_connections}, "
            f"timeout={timeout}, max_retries={max_retries}"
        )

    @classmethod
    def from_slime_args(cls, args: Any, **overrides: Any) -> SGLangClient:
        """Create SGLangClient from Slime's training args.

        Matches Slime's [`init_http_client`](https://github.com/THUDM/slime/blob/main/slime/utils/http_utils.py) formula:

            max_connections = concurrency * (num_gpus / gpus_per_engine)

        where `num_gpus ÷ gpus_per_engine` is the number of SGLang server instances,
        and `concurrency` is the max concurrent requests each instance can handle.
        This ensures enough connections to fully saturate all server instances.

        Args:
            args: Slime's args namespace with required attributes:
                - sglang_router_ip: SGLang router IP address
                - sglang_router_port: SGLang router port
                - sglang_server_concurrency: Concurrency per SGLang server
                - rollout_num_gpus: Total GPUs for rollout
                - rollout_num_gpus_per_engine: GPUs per engine
            **overrides: Override any configuration values.

        Returns:
            Configured SGLangClient instance.

        Example:
            >>> client = SGLangClient.from_slime_args(args)
            >>> model = SGLangModel(tokenizer=tokenizer, client=client)
        """
        return cls(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            # Matches Slime's init_http_client formula
            max_connections=args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine,
            **overrides,
        )

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        await self._client.aclose()

    async def __aenter__(self) -> SGLangClient:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        await self.close()

    async def _iter_sse_events(self, response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
        """Parse Server-Sent Events from streaming response.

        SSE format: lines starting with "data:" contain JSON payloads.
        Stream ends with "data: [DONE]".
        """
        async for line in response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue

            data = line[5:].strip()  # len("data:") = 5
            if data == "[DONE]":
                break

            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue

    def _is_retryable_error(self, e: Exception) -> bool:
        """Check if an error is retryable.

        Aligned with SLIME's philosophy: retry aggressively on most errors.
        Only non-retryable errors are client errors (4xx) that won't self-resolve.

        From OpenAI: 408 (Request Timeout) and 429 (Rate Limited) ARE retried.
        """
        if isinstance(e, httpx.HTTPStatusError):
            status = e.response.status_code
            # Don't retry non-recoverable client errors
            if status in NON_RETRYABLE_STATUS_CODES:
                return False
            # Retry everything else: 5xx, 408, 429, etc.
            return True
        # Retry all connection/timeout errors
        return True

    async def generate(
        self,
        input_ids: list[int],
        *,
        model: str | None = None,
        sampling_params: dict[str, Any] | None = None,
        return_logprob: bool = True,
        logprob_start_len: int = 0,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream generation from SGLang `/generate` endpoint.

        Args:
            input_ids: Input token IDs.
            model: Optional model identifier.
            sampling_params: Sampling parameters (temperature, top_p, max_new_tokens, etc.).
            return_logprob: Whether to return log probabilities (default: True).
            logprob_start_len: Start position for logprob computation (default: 0).

        Yields:
            Parsed SSE events containing text, output_ids, logprobs, and metadata.

        Raises:
            httpx.HTTPStatusError: For non-retryable client errors (400, 401, 403, etc.)
                or after all retries exhausted.
            httpx.ConnectError: For connection failures after retries exhausted.
        """
        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "stream": True,
        }

        if model:
            payload["model"] = model

        if sampling_params:
            payload["sampling_params"] = sampling_params

        if return_logprob:
            payload["return_logprob"] = True
            payload["logprob_start_len"] = logprob_start_len

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                async with self._client.stream("POST", "/generate", json=payload) as response:
                    response.raise_for_status()
                    async for event in self._iter_sse_events(response):
                        yield event
                    return  # Success, exit retry loop

            except Exception as e:
                last_error = e

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    raise  # Non-retryable error (e.g., 400 Bad Request)

                # Log and retry
                response_text = e.response.text if isinstance(e, httpx.HTTPStatusError) else None
                
                # Use exponential backoff for connection/timeout errors to avoid thundering herd
                # These errors indicate resource exhaustion - need longer delays with jitter
                if isinstance(e, (httpx.PoolTimeout, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError)):
                    # Exponential backoff: 1s, 2s, 4s, 8s, ... capped at 30s
                    base_delay = min(self.retry_delay * (2 ** min(attempt, 4)), 30.0)
                    # Add jitter (±25%) to prevent synchronized retries across many requests
                    jitter = base_delay * 0.25 * (2 * random.random() - 1)  # Random between -25% and +25%
                    delay = max(0.1, base_delay + jitter)  # Ensure delay is at least 0.1s
                else:
                    # For other errors, use base delay with small jitter to prevent synchronization
                    jitter = self.retry_delay * 0.1 * (2 * random.random() - 1)  # ±10% jitter
                    delay = max(0.1, self.retry_delay + jitter)
                
                if attempt < self.max_retries:
                    logger.warning(
                        f"SGLang request failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                        f"{type(e).__name__}: {e}, response={response_text}. Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"SGLang request failed after {self.max_retries + 1} attempts: "
                        f"{type(e).__name__}: {e}, response={response_text}"
                    )
                    raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error

    async def health(self) -> bool:
        """Check if SGLang server is healthy.

        Returns:
            True if server responds to `/health` endpoint, False otherwise.
        """
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
