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

"""Qwen XML tool call parser."""

from __future__ import annotations

import logging
import re

from .base import UNKNOWN_TOOL_NAME, ToolParser, ToolParseResult, register_tool_parser

logger = logging.getLogger(__name__)


@register_tool_parser("qwen_xml")
class QwenXMLToolParser(ToolParser):
    """Parser for Qwen3-Coder XML tool call format.

    Format:
        <tool_call>
        <function=function_name>
        <parameter=param1>
        value1
        </parameter>
        <parameter=param2>
        value2
        </parameter>
        </function>
        </tool_call>

    Used by:
    - Qwen/Qwen3-Coder models

    This format uses attribute-style XML tags where the function name and
    parameter names are embedded in the tag itself (e.g., `<function=name>`
    and `<parameter=name>`). Parameter values can span multiple lines.

    Chat Template Notes:
        Qwen Coder's chat template uses newline as separator between messages:
        `<|im_start|>role\\ncontent<|im_end|>\\n<|im_start|>...`
        The message_separator property returns "\\n" to match this format.

    Attributes:
        tool_call_tokens: Start/end delimiters for tool calls.
        think_tokens: Start/end delimiters for think blocks to exclude.
    """

    DEFAULT_TOOL_CALL_TOKENS = ("<tool_call>", "</tool_call>")
    DEFAULT_THINK_TOKENS = ("<think>", "</think>")

    # Pattern to extract function name from <function=name>...</function>
    _FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)

    # Pattern to extract parameters from <parameter=name>value</parameter>
    _PARAMETER_PATTERN = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

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

    @property
    def message_separator(self) -> str:
        """Separator between messages in the chat template.

        Qwen Coder models use newline `\\n` as separator between messages.
        """
        return "\n"

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

            # Parse the function tag
            func_match = self._FUNCTION_PATTERN.search(raw_content)
            if not func_match:
                tool_calls.append(self._make_error_tool_call(raw_content, tool_call_id, "missing <function=...> tag"))
                continue

            func_name = func_match.group(1).strip()
            func_body = func_match.group(2)

            if not func_name:
                tool_calls.append(self._make_error_tool_call(raw_content, tool_call_id, "empty function name"))
                continue

            # Parse all parameters from the function body
            arguments: dict[str, str] = {}
            for param_match in self._PARAMETER_PATTERN.finditer(func_body):
                param_name = param_match.group(1).strip()
                param_value = param_match.group(2).strip()
                if param_name:
                    arguments[param_name] = param_value

            tool_calls.append(
                ToolParseResult(
                    id=tool_call_id,
                    name=func_name,
                    input=arguments,
                )
            )

        return tool_calls

    def _make_error_tool_call(
        self,
        raw_content: str,
        tool_call_id: str,
        error: str,
    ) -> ToolParseResult:
        """Create an error tool call for parse failures."""
        # Try to extract function name from malformed content
        func_match = self._FUNCTION_PATTERN.search(raw_content)
        name = func_match.group(1).strip() if func_match else UNKNOWN_TOOL_NAME

        logger.warning(f"Tool call parse error: {error}")

        return ToolParseResult(
            id=tool_call_id,
            name=name,
            input={},
            raw=raw_content,
        )
