# Copyright 2025-2026 Horizon RL Contributors
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

"""Utilities for shared client/tokenizer/etc. for RL training."""

from __future__ import annotations

import importlib.util
import logging
import os
from functools import cache
from typing import TYPE_CHECKING, Any

from .client import DEFAULT_MAX_CONNECTIONS, SGLangClient

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, ProcessorMixin


@cache
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

    Matches slime's `init_http_client` formula for connection pooling.
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


@cache
def get_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerBase:
    """Get a shared (cached) tokenizer.

    For DeepSeek-V3.2, attach its encoding module to the tokenizer to construct `apply_chat_template()`.

    Args:
        tokenizer_path: Path or HuggingFace model ID for the tokenizer.

    Returns:
        Cached tokenizer instance.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # Auto-detect DeepSeek-V3.2 by checking for its encoding module
    encoding_file = os.path.join(tokenizer.name_or_path, "encoding", "encoding_dsv32.py")
    if os.path.isfile(encoding_file):
        attach_dsv32_encoding(tokenizer)
    return tokenizer


def attach_dsv32_encoding(tokenizer: PreTrainedTokenizerBase) -> None:
    """Attach DeepSeek-V3.2's encoding module to a tokenizer in-place.

    Replaces `apply_chat_template()` with one that delegates to
    DeepSeek-V3.2's `encoding/encoding_dsv32.py` module.

    Call this when creating a tokenizer directly instead of using `get_tokenizer()`:

        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2")
        attach_dsv32_encoding(tokenizer)

    Args:
        tokenizer: HuggingFace tokenizer to patch.
    """
    try:
        cache_dir = tokenizer.name_or_path
        filepath = os.path.join(cache_dir, "encoding", "encoding_dsv32.py")
        spec = importlib.util.spec_from_file_location("encoding_dsv32", filepath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module spec from {filepath}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.info("Loaded DeepSeek V3.2's encoding module from %s", filepath)
    except Exception as e:
        logger.error("Failed to load DeepSeek V3.2's encoding module from %s: %s", filepath, e)
        raise

    def apply_chat_template(
        conversation: list[dict[str, Any]],
        tools: list[dict] | None = None,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        **kwargs: Any,
    ) -> str:
        """Format messages using DeepSeek-V3.2's `encode_messages()`.

        Drop-in replacement for Jinja-based `apply_chat_template()`.
        """
        if kwargs:
            logger.warning("DeepSeek V3.2 doesn't support the following kwargs: %s", kwargs)

        thinking_mode = "thinking" if enable_thinking else "chat"

        # Incremental path: fake user prefix + tool results.
        # See: https://github.com/horizon-rl/strands-sglang/issues/29
        if (
            len(conversation) >= 2
            and conversation[0].get("role") == "user"
            and all(m.get("role") == "tool" for m in conversation[1:])
        ):
            result = "\n\n<function_results>"
            for msg in conversation[1:]:
                if msg.get("role") == "tool":
                    result += "\n<result>" + msg.get("content", "") + "</result>"
            result += "\n</function_results>"
            if add_generation_prompt:
                gen = "<think>" if thinking_mode == "thinking" else "</think>"
                result += "\n\n" + gen
            return str(module.encode_messages([conversation[0]], thinking_mode=thinking_mode)) + result

        # Attach tools to system message (encoding module reads them from msg.get("tools"))
        messages = list(conversation)
        if tools:
            if messages and messages[0].get("role") == "system":
                messages[0] = {**messages[0], "tools": tools}
            else:
                messages.insert(0, {"role": "system", "content": "", "tools": tools})

        return str(module.encode_messages(messages, thinking_mode=thinking_mode))

    # attach the new apply_chat_template to the tokenizer
    tokenizer.apply_chat_template = apply_chat_template  # type: ignore[method-assign,assignment]
    tokenizer._dsv32_encoding_attached = True


@cache
def get_processor(processor_path: str) -> ProcessorMixin:
    """Get a shared (cached) multimodal processor.

    Args:
        processor_path: Path or HuggingFace model ID.

    Returns:
        Cached processor instance.
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
