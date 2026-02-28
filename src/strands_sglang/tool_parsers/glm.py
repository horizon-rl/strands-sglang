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

"""Tool call parser for GLM (ChatGLM) models."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from typing_extensions import override

from .base import ToolParser, ToolParseResult, register_tool_parser

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

    Chat Template Notes:
        GLM uses no explicit separator between messages.
    """

    ARG_PATTERN = re.compile(
        r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
        re.DOTALL,
    )

    @override
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
        text = self.think_pattern.sub("", text)

        tool_calls: list[ToolParseResult] = []

        for i, match in enumerate(self.tool_pattern.finditer(text)):
            raw_content = match.group(1).strip()
            tool_call_id = f"call_{i:04d}"  # Sequential IDs for sortability

            # Function name is on the first line
            lines = raw_content.split("\n", 1)
            name = lines[0].strip()

            # Check if name is missing or contains XML tags (indicating we picked up arg tags instead)
            if not name or "<" in name or ">" in name:
                logger.warning("Tool parse error: missing function name")
                tool_calls.append(ToolParseResult.from_parse_error(id=tool_call_id, raw=raw_content))
                continue

            # Parse <arg_key>/<arg_value> pairs
            arguments: dict[str, Any] = {}
            rest = lines[1] if len(lines) > 1 else ""
            for arg_match in self.ARG_PATTERN.finditer(rest):
                key = arg_match.group(1).strip()
                value_str = arg_match.group(2).strip()
                try:
                    value = json.loads(value_str)
                except (json.JSONDecodeError, ValueError):
                    value = value_str
                arguments[key] = value

            tool_calls.append(ToolParseResult(id=tool_call_id, name=name, input=arguments))

        return tool_calls
