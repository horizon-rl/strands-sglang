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

"""Tool call parser for GLM (ChatGLM) models.

GLM models use an XML key-value format for tool calls instead of the
JSON format used by Hermes/Qwen models::

    <tool_call>function_name
    <arg_key>key1</arg_key>
    <arg_value>value1</arg_value>
    <arg_key>key2</arg_key>
    <arg_value>value2</arg_value>
    </tool_call>

Values are either plain strings or JSON-encoded (for non-string types).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .base import UNKNOWN_TOOL_NAME, ToolParser, ToolParseResult, register_tool_parser

logger = logging.getLogger(__name__)


@register_tool_parser("glm")
class GLMToolParser(ToolParser):
    """Parser for GLM XML key-value tool call format.

    Format:
        <tool_call>function_name
        <arg_key>key1</arg_key>
        <arg_value>value1</arg_value>
        <arg_key>key2</arg_key>
        <arg_value>value2</arg_value>
        </tool_call>

    This format uses a key-value pair structure where the function name
    appears on the first line after <tool_call>, followed by alternating
    <arg_key> and <arg_value> tags. Values can be plain strings or
    JSON-encoded for non-string types.

    Think Block Handling:
        Models with reasoning capabilities may output draft tool calls
        inside <think>...</think> blocks. These are excluded by default
        to avoid executing planning/reasoning tool calls.
        Set think_tokens=None to disable this behavior.

    Chat Template Notes:
        GLM uses no explicit separator between messages.

    Attributes:
        tool_call_tokens: Start/end delimiters for tool calls.
        think_tokens: Start/end delimiters for think blocks to exclude.
    """

    DEFAULT_TOOL_CALL_TOKENS = ("<tool_call>", "</tool_call>")
    DEFAULT_THINK_TOKENS = ("<think>", "</think>")

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

        if think_tokens:
            self._think_pattern: re.Pattern[str] | None = re.compile(
                rf"{re.escape(think_tokens[0])}.*?{re.escape(think_tokens[1])}",
                re.DOTALL,
            )
        else:
            self._think_pattern = None

        self._arg_pattern = re.compile(
            r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
            re.DOTALL,
        )

    @property
    def message_separator(self) -> str:
        """Separator between messages in the chat template.

        GLM uses no explicit separator between messages.
        """
        return ""

    def parse(self, text: str) -> list[ToolParseResult]:
        """Parse tool calls from GLM model output.

        Extracts the function name from the first line after ``<tool_call>``,
        then parses ``<arg_key>``/``<arg_value>`` pairs into a dict.

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

            # Function name is on the first line
            lines = raw_content.split("\n", 1)
            name = lines[0].strip()

            # Check if name is missing or contains XML tags (indicating we picked up arg tags instead)
            if not name or "<" in name or ">" in name:
                tool_calls.append(self._make_error_tool_call(raw_content, tool_call_id, "missing function name"))
                continue

            # Parse <arg_key>/<arg_value> pairs
            arguments: dict[str, Any] = {}
            rest = lines[1] if len(lines) > 1 else ""
            for arg_match in self._arg_pattern.finditer(rest):
                key = arg_match.group(1).strip()
                value_str = arg_match.group(2).strip()
                try:
                    value = json.loads(value_str)
                except (json.JSONDecodeError, ValueError):
                    value = value_str
                arguments[key] = value

            tool_calls.append(ToolParseResult(id=tool_call_id, name=name, input=arguments))

        return tool_calls

    def _make_error_tool_call(
        self,
        raw_content: str,
        tool_call_id: str,
        error: str,
    ) -> ToolParseResult:
        """Create an error tool call for parse failures."""
        # Try to extract function name from first line
        lines = raw_content.split("\n", 1)
        name = lines[0].strip() if lines else UNKNOWN_TOOL_NAME
        # If name is empty or contains XML tags, use UNKNOWN_TOOL_NAME
        if not name or "<" in name or ">" in name:
            name = UNKNOWN_TOOL_NAME

        logger.warning(f"Tool call parse error: {error}")

        return ToolParseResult(
            id=tool_call_id,
            name=name,
            input={},
            raw=raw_content,
        )
