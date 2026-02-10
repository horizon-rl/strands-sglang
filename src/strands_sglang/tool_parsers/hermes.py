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

"""Hermes/Qwen tool call parser."""

from __future__ import annotations

import json
import logging
import re

from typing_extensions import override

from .base import ToolParser, ToolParseResult, register_tool_parser

logger = logging.getLogger(__name__)


@register_tool_parser("hermes")
class HermesToolParser(ToolParser):
    """Parser for Hermes/Qwen tool call format (JSON wrapped in delimiters).

    Format: <tool_call>{"name": "func", "arguments": {"arg": value}}</tool_call>

    Used by:
    - Qwen/Qwen2/Qwen3 models
    - NousResearch/Hermes models
    - Models using similar wrapped JSON tool call format

    Only handles JSONDecodeError; Strands validates arguments against tool schemas.

    Think Block Handling:
        Models with reasoning capabilities (Qwen3 with thinking, DeepSeek-R1, etc.)
        may output draft tool calls inside <think>...</think> blocks. These are
        excluded by default to avoid executing planning/reasoning tool calls.
        Set think_tokens=None to disable this behavior.

    Chat Template Notes:
        Qwen3's chat template uses newline as separator between messages:
        `<|im_start|>role\\ncontent<|im_end|>\\n<|im_start|>...`
        The message_separator property returns "\\n" to match this format.

    Attributes:
        tool_call_tokens: Start/end delimiters for tool calls.
        think_tokens: Start/end delimiters for think blocks to exclude.
    """

    DEFAULT_TOOL_CALL_TOKENS = ("<tool_call>", "</tool_call>")
    DEFAULT_THINK_TOKENS = ("<think>", "</think>")
    _NAME_PATTERN = re.compile(r'"name"\s*:\s*"([^"]+)"')

    def __init__(
        self,
        tool_call_tokens: tuple[str, str] = DEFAULT_TOOL_CALL_TOKENS,
        think_tokens: tuple[str, str] | None = DEFAULT_THINK_TOKENS,
    ) -> None:
        """Initialize the parser with optional custom tokens.

        Args:
            tool_call_tokens: (start, end) delimiters for tool calls.
            think_tokens: (start, end) delimiters for think blocks to exclude.
                Set to None to disable think block exclusion.
        """
        self.tool_call_tokens = tool_call_tokens
        self.think_tokens = think_tokens

        self._pattern = re.compile(
            rf"{re.escape(tool_call_tokens[0])}\s*(.*?)\s*{re.escape(tool_call_tokens[1])}",
            re.DOTALL,
        )

        # Pattern to remove think blocks (if configured)
        if think_tokens:
            self._think_pattern: re.Pattern[str] | None = re.compile(
                rf"{re.escape(think_tokens[0])}.*?{re.escape(think_tokens[1])}",
                re.DOTALL,
            )
        else:
            self._think_pattern = None

    @override
    @property
    def message_separator(self) -> str:
        """Separator between messages in the chat template.

        Different tokenizers use different separators between messages.
        This is used during incremental tokenization to ensure the TITO
        trajectory matches what `apply_chat_template` would produce.

        Qwen models use newline `\\n` as separator between messages.
        """
        return "\n"

    @override
    def parse(self, text: str) -> list[ToolParseResult]:
        """Parse tool calls from model output.

        Args:
            text: Model output text.

        Returns:
            List of tool call results (successful and errors).
        """
        # Remove think blocks to avoid parsing draft tool calls from reasoning
        if self._think_pattern:
            text = self._think_pattern.sub("", text)

        tool_calls: list[ToolParseResult] = []

        for i, match in enumerate(self._pattern.finditer(text)):
            raw_content = match.group(1).strip()
            tool_call_id = f"call_{i:04d}"  # Sequential IDs for sortability

            # Only handle JSONDecodeError - let Strands validate the rest
            try:
                call_json = json.loads(raw_content)
            except json.JSONDecodeError as e:
                name_match = self._NAME_PATTERN.search(raw_content)
                name = name_match.group(1) if name_match else ToolParseResult.UNKNOWN_NAME
                logger.warning(f"Tool parse error: {e}")
                tool_calls.append(ToolParseResult.from_parse_error(id=tool_call_id, raw=raw_content, name=name))
                continue

            # Extract name and arguments - be lenient, let Strands validate
            if isinstance(call_json, dict):
                name = call_json.get("name")
                arguments = call_json.get("arguments", {})
            else:
                name = None
                arguments = {}

            # Need a string name to yield toolUse event
            if not name or not isinstance(name, str):
                name_match = self._NAME_PATTERN.search(raw_content)
                extracted = name_match.group(1) if name_match else ToolParseResult.UNKNOWN_NAME
                logger.warning("Tool parse error: missing name")
                tool_calls.append(ToolParseResult.from_parse_error(id=tool_call_id, raw=raw_content, name=extracted))
                continue

            # Pass arguments as-is - Strands validates against tool schema
            tool_calls.append(
                ToolParseResult(
                    id=tool_call_id,
                    name=name,
                    input=arguments if isinstance(arguments, dict) else {},
                )
            )

        return tool_calls
