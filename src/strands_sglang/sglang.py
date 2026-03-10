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

"""SGLang model provider with token-in/token-out support."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Iterator
from functools import cached_property
from typing import (
    Any,
    TypedDict,
    TypeVar,
    cast,
)

from pydantic import BaseModel
from strands.models import Model
from strands.types.content import ContentBlock, Messages, SystemContentBlock
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)
from strands.types.streaming import StopReason, StreamEvent
from strands.types.tools import ToolChoice, ToolResultContent, ToolSpec
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from typing_extensions import Unpack, override

from .client import SGLangClient
from .exceptions import SGLangContextLengthError, SGLangThrottledError
from .token import TokenManager
from .tool_parsers import HermesToolParser, ToolParser, ToolParseResult

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class SGLangModel(Model):
    """SGLang native `/generate` API provider with token-in/token-out support.

    Example:
        >>> from transformers import AutoTokenizer
        >>> from strands_sglang import SGLangClient, SGLangModel
        >>> client = SGLangClient(base_url="http://localhost:30000")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        >>> model = SGLangModel(client=client, tokenizer=tokenizer)
        >>> # After generation:
        >>> model.token_manager.token_ids    # Full token trajectory
        >>> model.token_manager.loss_mask    # Boolean mask for loss computation
        >>> model.token_manager.logprobs     # Log probabilities
    """

    class SGLangConfig(TypedDict, total=False):
        """Configuration options for SGLang generation."""

        sampling_params: dict[str, Any] | None  # Passed to /generate endpoint
        return_logprob: bool | None  # Return logprobs for all tokens (default: True)
        enable_thinking: bool | None  # Enable thinking mode for Qwen3 hybrid models

    def __init__(
        self,
        *,
        client: SGLangClient,
        tokenizer: PreTrainedTokenizerBase,
        tool_parser: ToolParser | None = None,
        **config: Unpack[SGLangConfig],
    ) -> None:
        """Initialize SGLang model provider.

        Args:
            client: `SGLangClient` for HTTP communication with the SGLang server.
            tokenizer: HuggingFace tokenizer for chat template and tokenization.
            tool_parser: `ToolParser` for tool calls (default: `HermesToolParser`).
            **config: Additional SGLang generation configuration.
        """
        self.client = client
        self.tokenizer = tokenizer
        self.tool_parser = tool_parser or HermesToolParser()
        self.config = dict(config)
        self.tool_parser.validate_tokenizer(self.tokenizer)

        # State tracking (this makes SGLangModel stateful)
        self.token_manager = TokenManager()
        self.message_count: int = 0
        self.tool_parse_errors: dict[str, int] = {}  # per-tool parse error count
        self.image_data: list[str] = []  # accumulated image data URLs (VLM only)

        logger.debug("initialized with config: %s", self.config)

    def reset(self) -> None:
        """Reset token accumulation for a new episode.

        Call this at episode start. Clears all accumulated tokens and resets
        internal state for tool tracking.
        """
        self.token_manager.reset()
        self.message_count = 0
        self.tool_parse_errors = {}
        self.image_data = []

    @cached_property
    def is_multimodal(self) -> bool:
        """Auto-detect multimodal support from the model's HuggingFace config.

        Mirrors SGLang's logic: `hasattr(hf_config, "vision_config")`.
        """
        hf_config = PretrainedConfig.from_pretrained(self.tokenizer.name_or_path)
        return hasattr(hf_config, "vision_config")

    # -------------------------------------------------------------------------
    # Model interface implementation
    # -------------------------------------------------------------------------

    @override
    def update_config(self, **model_config: Unpack[SGLangConfig]) -> None:  # type: ignore[override]
        """Update the model configuration.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> SGLangConfig:
        """Get the model configuration.

        Returns:
            The model configuration dict.
        """
        return cast(SGLangModel.SGLangConfig, self.config)

    # -------------------------------------------------------------------------
    # Chat template and message formatting
    # -------------------------------------------------------------------------

    @classmethod
    def format_content_block(
        cls, content_block: ContentBlock | ToolResultContent, is_multimodal: bool = False
    ) -> dict[str, Any] | str:
        """Convert a single Strands `ContentBlock` or `ToolResultContent` to HF chat template format."""
        # ContentBlock / ToolResultContent is a TypedDict with exactly one key set at runtime
        ((key, value),) = content_block.items()
        result: dict[str, Any] = {}
        match key, value:
            case "text", str() as text:
                result = {"type": "text", "text": text}
            case "image", dict() as image:
                mime = f"image/{image['format']}"
                encoded = base64.b64encode(image["source"]["bytes"]).decode()
                result = {"type": "image", "image": f"data:{mime};base64,{encoded}"}
            case "json", data:
                result = {"type": "text", "text": json.dumps(data)}
            # TODO: add support for other content types
            case _:
                raise TypeError(f"content_type=<{key}> | unsupported type")
        # flatten to text if not multimodal
        if not is_multimodal:
            return str(result["text"])
        return result

    @classmethod
    def format_messages(
        cls, messages: Messages, system_prompt: str | None = None, is_multimodal: bool = False
    ) -> list[dict[str, Any]]:
        """Convert Strands Messages to HF chat template format.

        When `is_multimodal=False` (default), content is flattened to a plain string.
        When `is_multimodal=True`, content is kept as a list of dicts.
        """
        result: list[dict[str, Any]] = []

        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        # Each Strands message is {"role": str, "content": [ContentBlock, ...]}
        # One Strands message maps to one HF message, except toolResult blocks
        # which each become a separate HF message with role="tool".
        for msg in messages:
            if "toolResult" in msg["content"][0]:
                # Each toolResult → its own HF message (different tool_call_id)
                for cb in msg["content"]:
                    assert "toolResult" in cb
                    tr = cb["toolResult"]
                    content = [cls.format_content_block(c, is_multimodal) for c in tr["content"]]
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr["toolUseId"],
                            "content": content if is_multimodal else content[0],
                        }
                    )
            else:
                # Non-tool content → one HF message (text, image, etc.; toolUse skipped)
                content = [cls.format_content_block(c, is_multimodal) for c in msg["content"] if "toolUse" not in c]
                result.append({"role": msg["role"], "content": content if is_multimodal else content[0]})

        return result

    def format_tool_specs(self, tool_specs: list[ToolSpec]) -> list[dict]:
        """Format strands ToolSpecs to OpenAI format for chat templates."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["inputSchema"]["json"],
                },
            }
            for spec in tool_specs
        ]

    def apply_chat_template(
        self,
        hf_messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str:
        """Apply the HuggingFace chat template to the messages."""
        return str(
            self.tokenizer.apply_chat_template(
                conversation=hf_messages,
                tools=cast(list[dict | Callable], tools),
                add_generation_prompt=add_generation_prompt,
                tokenize=False,  # never tokenize here
                enable_thinking=self.config.get("enable_thinking"),
                **kwargs,
            )
        )

    @staticmethod
    def extract_image_urls(messages: list[dict[str, Any]]) -> list[str]:
        """Extract image data URLs from HF-formatted multimodal messages."""
        urls: list[str] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, dict) and content.get("type") == "image":
                urls.append(content["image"])
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        urls.append(part["image"])
        return urls

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    @staticmethod
    def sort_tool_results(messages: Messages) -> Messages:
        """Sort tool results by ID to match original call order (IDs are sequential: call_0000, call_0001, ...)."""
        return [
            {**msg, "content": sorted(msg["content"], key=lambda c: c["toolResult"]["toolUseId"])}
            if "toolResult" in msg["content"][0]
            else msg
            for msg in messages
        ]

    def tokenize_prompt_messages(
        self,
        messages: Messages,
        system_prompt: str | None,
        tools: list[dict] | None = None,
    ) -> list[int]:
        """Tokenize prompt messages for the next generation call.

        - First call: tokenizes full prompt with system prompt and tools.
        - Subsequent calls: uses a fake prefix (system + user) for boundary formatting,
        then subtracts it to extract only incremental tokens.
        """
        # First call: full prompt with tools
        if self.message_count == 0:
            hf_messages = self.format_messages(messages, system_prompt, is_multimodal=self.is_multimodal)
            self.image_data = self.extract_image_urls(hf_messages)
            prompt = self.apply_chat_template(hf_messages, tools=tools, add_generation_prompt=True)
            return list(self.tokenizer.encode(prompt, add_special_tokens=False))

        # Incremental: fake prefix subtraction with message_separator bridge
        if len(messages) > self.message_count:
            new_hf_messages = self.format_messages(
                self.sort_tool_results(messages[self.message_count :]), is_multimodal=self.is_multimodal
            )
            self.image_data.extend(self.extract_image_urls(new_hf_messages))
            fake_messages = [
                {"role": "system", "content": [{"text": "FAKE SYSTEM PROMPT"}]},
                {"role": "user", "content": [{"text": "FAKE USER MESSAGE"}]},
            ]
            fake_hf_messages = self.format_messages(cast(Messages, fake_messages), is_multimodal=self.is_multimodal)
            full_prompt = self.apply_chat_template(fake_hf_messages + new_hf_messages, add_generation_prompt=True)
            prefix_prompt = self.apply_chat_template(fake_hf_messages, add_generation_prompt=False)
            assert full_prompt.startswith(prefix_prompt), "full prompt must start with prefix prompt"
            prompt = self.tool_parser.message_separator + full_prompt[len(prefix_prompt) :]
            return list(self.tokenizer.encode(prompt, add_special_tokens=False))

        raise RuntimeError(f"No new messages to tokenize (message_count={self.message_count}, got {len(messages)})")

    def _yield_tool_use_events(
        self,
        tool_calls: list[ToolParseResult],
    ) -> Iterator[StreamEvent]:
        """Yield toolUse stream events for parsed tool calls.

        Each tool call emits three events following the Strands streaming protocol:
        - `contentBlockStart`: begins block with toolUseId and name
        - `contentBlockDelta`: contains the tool input (delta = incremental data)
        - `contentBlockStop`: ends the block
        """
        for tool_call in tool_calls:
            if tool_call.is_error:
                logger.warning("Tool parse error for '%s': %s", tool_call.name, (tool_call.raw or "")[:100])
                # Track parse error count per tool name
                self.tool_parse_errors[tool_call.name] = self.tool_parse_errors.get(tool_call.name, 0) + 1

            yield {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": tool_call.id,
                            "name": tool_call.name,
                        }
                    }
                }
            }
            yield {
                "contentBlockDelta": {
                    "delta": {
                        "toolUse": {
                            "input": tool_call.payload,
                        }
                    }
                }
            }
            yield {"contentBlockStop": {}}

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Chat completion with SGLangModel using the `/generate` endpoint.

        The `stream` method follows Strands' protocol but actually disabled here for training-only usage.
        This means users won't see streaming behavior such as print callbacks.
        """
        # Prepare request
        config = self.get_config()
        sampling_params: dict[str, Any] = dict(config.get("sampling_params") or {})
        sampling_params.setdefault("skip_special_tokens", False)
        return_logprob = config.get("return_logprob", True)
        tools = self.format_tool_specs(tool_specs) if tool_specs else None
        new_input_ids = self.tokenize_prompt_messages(messages, system_prompt, tools=tools)
        # Tracking token IDs in token_manager to ensure the token-in feature
        input_ids = self.token_manager.token_ids + new_input_ids

        # Assistant message start
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}

        # Call SGLang's `/generate` endpoint
        try:
            response = await self.client.generate(
                input_ids=input_ids,
                sampling_params=sampling_params,
                return_logprob=return_logprob,
                logprob_start_len=max(0, len(self.token_manager.token_ids) - 1) if return_logprob else None,
                image_data=self.image_data or None,
            )

            # Extract response data
            text = response["text"]
            output_ids = response["output_ids"]
            meta_info = response["meta_info"]
            input_token_logprobs = meta_info.get("input_token_logprobs")
            output_token_logprobs = meta_info.get("output_token_logprobs")

            # Assistant message content delta (single delta for non-streaming)
            yield {"contentBlockDelta": {"delta": {"text": text}}}

        except SGLangContextLengthError as e:
            raise ContextWindowOverflowException(f"Context length exceeded: {e.body}") from e
        except SGLangThrottledError as e:
            raise ModelThrottledException(f"Service throttled (status={e.status}): {e.body}") from e

        # Update token trajectory
        self.token_manager.add_prompt(
            token_ids=new_input_ids,
            logprobs=[e[0] for e in input_token_logprobs[-len(new_input_ids) :]] if input_token_logprobs else None,
        )
        self.token_manager.add_response(
            token_ids=output_ids,
            logprobs=[e[0] for e in output_token_logprobs] if output_token_logprobs else None,
        )
        self.message_count = len(messages) + 1

        # Assistant message content stop
        yield {"contentBlockStop": {}}

        # Assistant message tool use content - start, delta, stop
        parsed_tool_calls = self.tool_parser.parse(text)
        for event in self._yield_tool_use_events(parsed_tool_calls):
            yield event

        # Assistant message stop
        stop_reason: str = "tool_use" if parsed_tool_calls else "end_turn"
        if meta_info["finish_reason"]["type"] == "length":
            stop_reason = "max_tokens"
        yield {"messageStop": {"stopReason": cast(StopReason, stop_reason)}}

        # Assistant message usage metadata
        yield {
            "metadata": {
                "usage": {
                    "inputTokens": meta_info["prompt_tokens"],
                    "outputTokens": meta_info["completion_tokens"],
                    "totalTokens": meta_info["prompt_tokens"] + meta_info["completion_tokens"],
                    "cacheReadInputTokens": meta_info["cached_tokens"],
                },
                "metrics": {"latencyMs": int(meta_info["e2e_latency"] * 1000)},
            }
        }

    @override
    async def structured_output(
        self,
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Get structured output using SGLang's constrained decoding.

        Uses SGLang's `json_schema` parameter for FSM-based constrained generation,
        guaranteeing output conforms to the Pydantic model schema.

        Note: This method does NOT update token_manager (no TITO tracking).
        Intended for inference-only use cases like LLM-as-Judge.

        Args:
            output_model: Pydantic model class defining the output schema.
            prompt: Messages to send to the model.
            system_prompt: Optional system prompt.
            **kwargs: Additional arguments (unused).

        Yields:
            Single dict with "output" key containing the parsed Pydantic model instance.

        Raises:
            ValidationError: If model output fails Pydantic validation.
            SGLangHTTPError: On non-retryable HTTP errors.
        """
        # Convert Pydantic model to JSON schema string
        json_schema = json.dumps(output_model.model_json_schema())

        # Format and tokenize prompt (no tools for structured output)
        hf_messages = self.format_messages(prompt, system_prompt, is_multimodal=self.is_multimodal)
        formatted_prompt = self.apply_chat_template(hf_messages, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)

        # Build sampling params with json_schema constraint
        config = self.get_config()
        sampling_params: dict[str, Any] = dict(config.get("sampling_params") or {})
        sampling_params["json_schema"] = json_schema

        # Call SGLang /generate endpoint
        try:
            response = await self.client.generate(
                input_ids=input_ids,
                sampling_params=sampling_params,
                return_logprob=False,  # No need for logprobs in structured output
            )
        except SGLangContextLengthError as e:
            raise ContextWindowOverflowException(f"Context length exceeded: {e.body}") from e
        except SGLangThrottledError as e:
            raise ModelThrottledException(f"Service throttled (status={e.status}): {e.body}") from e

        # Parse and validate response
        text = response["text"]
        parsed = output_model.model_validate_json(text)

        yield {"output": parsed}
