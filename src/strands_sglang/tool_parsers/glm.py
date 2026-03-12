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

"""GLM key-value XML tool call parser."""

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
    """Parser for GLM (4.5, 4.7, 5) key-value XML tool call format.

    Format:

        <tool_call>function_name<arg_key>key1</arg_key><arg_value>value1</arg_value></tool_call>

    Notes:
        - Function name precedes the first `<arg_key>` tag (with or without a newline).
        - Values are JSON-decoded when possible, otherwise kept as strings.
        - Think blocks are excluded to avoid parsing draft tool calls from reasoning.
    """

    ARG_PATTERN = re.compile(
        r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
        re.DOTALL,
    )

    @override
    def parse(self, text: str) -> list[ToolParseResult]:
        """Parse tool calls from model output."""
        # Remove think blocks to avoid parsing draft tool calls from reasoning
        text = self.think_pattern.sub("", text)

        tool_calls: list[ToolParseResult] = []

        for i, match in enumerate(self.tool_pattern.finditer(text)):
            raw_content = match.group(1).strip()
            tool_call_id = f"call_{i:04d}"  # Sequential IDs for sortability

            # Function name is everything before the first <arg_key> (handles both
            # GLM-4.5 newline-separated and GLM-4.7/5 inline formats)
            parts = raw_content.split("<arg_key>", 1)
            name = parts[0].strip()

            if not name:
                logger.warning("Tool parse error: missing function name")
                tool_calls.append(ToolParseResult.from_parse_error(id=tool_call_id, raw=raw_content))
                continue

            # Parse <arg_key>/<arg_value> pairs
            arguments: dict[str, Any] = {}
            for arg_match in self.ARG_PATTERN.finditer(raw_content):
                key = arg_match.group(1).strip()
                value_str = arg_match.group(2).strip()
                try:
                    value = json.loads(value_str)
                except (json.JSONDecodeError, ValueError):
                    value = value_str
                arguments[key] = value

            tool_calls.append(ToolParseResult(id=tool_call_id, name=name, input=arguments))

        return tool_calls
