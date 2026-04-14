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

"""Kimi K2 special-token tool call parser."""

from __future__ import annotations

import json
import logging
import re

from typing_extensions import override

from .base import ToolParser, ToolParseResult, register_tool_parser

logger = logging.getLogger(__name__)


@register_tool_parser("kimi_k2")
class KimiK2ToolParser(ToolParser):
    """Parser for Kimi K2/K2.5 special-token tool call format.

    Format:

        <|tool_calls_section_begin|>
        <|tool_call_begin|>functions.func_name:0
        <|tool_call_argument_begin|>{"arg": "val"}<|tool_call_end|>
        <|tool_calls_section_end|>

    Notes:
        - The raw ID (e.g. `functions.func_name:0`) is preserved as `tool_call_id`
        for correct round-trip with the chat template (`## Return of <id>`).
        - Think blocks are excluded to avoid parsing draft tool calls from reasoning.
    """

    SECTION_PATTERN = re.compile(
        r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>",
        re.DOTALL,
    )

    CALL_PATTERN = re.compile(
        r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<\s]+)\s*"
        r"<\|tool_call_argument_begin\|>\s*(?P<arguments>.*?)\s*"
        r"<\|tool_call_end\|>",
        re.DOTALL,
    )

    # Matches "functions.<name>:<index>" with optional "functions." prefix
    _ID_PATTERN = re.compile(r"^(?:functions\.)?(?P<name>[\w.\-]+?)(?::(?P<index>\d+))?$")

    @override
    def parse(self, text: str) -> list[ToolParseResult]:
        """Parse tool calls from model output."""
        # Remove think blocks to avoid parsing draft tool calls from reasoning
        text = self.think_pattern.sub("", text)

        tool_calls: list[ToolParseResult] = []

        for section_match in self.SECTION_PATTERN.finditer(text):
            section = section_match.group(1)

            for call_match in self.CALL_PATTERN.finditer(section):
                raw_id = call_match.group("tool_call_id")
                raw_args = call_match.group("arguments").strip()

                # Extract function name from "functions.func_name:index"
                id_match = self._ID_PATTERN.match(raw_id)
                if id_match:
                    name = id_match.group("name")
                else:
                    logger.warning("Unexpected tool call ID format: %s", raw_id)
                    name = raw_id

                try:
                    arguments = json.loads(raw_args)
                    if not isinstance(arguments, dict):
                        logger.warning("Tool parse error: arguments is not a dict for %s", name)
                        arguments = {}
                except json.JSONDecodeError:
                    logger.warning("Failed to parse arguments for %s: %s", name, raw_args[:200])
                    tool_calls.append(ToolParseResult.from_parse_error(id=raw_id, raw=raw_args, name=name))
                    continue

                tool_calls.append(
                    ToolParseResult(
                        id=raw_id,
                        name=name,
                        input=arguments,
                    )
                )

        return tool_calls
