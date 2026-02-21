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

"""Utilities for shared client/tokenizer/etc. for RL training."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from .client import DEFAULT_MAX_CONNECTIONS, SGLangClient

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@lru_cache(maxsize=None)
def get_client(
    base_url: str,
    *,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    timeout: float | None = 900.0,
    connect_timeout: float = 5.0,
    max_retries: int = 60,
    retry_delay: float = 1.0,
) -> SGLangClient:
    """Get a shared (cached) `SGLangClient` for connection pooling."""
    return SGLangClient(
        base_url=base_url,
        max_connections=max_connections,
        timeout=timeout,
        connect_timeout=connect_timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


def get_client_from_slime_args(
    args: Any,
    *,
    timeout: float | None = 900.0,
    connect_timeout: float = 5.0,
    max_retries: int = 60,
    retry_delay: float = 1.0,
) -> SGLangClient:
    """Get a shared (cached) `SGLangClient` from `slime`'s training args.

    Matches slime's :func:`init_http_client` formula for connection pooling.
    """
    base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    max_connections = int(args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine)
    return get_client(
        base_url=base_url,
        max_connections=max_connections,
        timeout=timeout,
        connect_timeout=connect_timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


@lru_cache(maxsize=None)
def get_tokenizer(tokenizer_path: str) -> PreTrainedTokenizer:
    """Get a shared (cached) tokenizer.

    Args:
        tokenizer_path: Path or HuggingFace model ID for the tokenizer.

    Returns:
        Cached tokenizer instance.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
