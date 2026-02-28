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

"""Tool call parser for Kimi K2 / K2.5 models."""

from __future__ import annotations

import json
import logging
import re

from typing_extensions import override

from .base import ToolParser, ToolParseResult, register_tool_parser

logger = logging.getLogger(__name__)


@register_tool_parser("kimi_k2")
class KimiK2ToolParser(ToolParser):
    """Parser for Kimi K2/K2.5 special-token-based tool call format.

    Format:
        <|tool_calls_section_begin|>
        <|tool_call_begin|>functions.func_name:0<|tool_call_argument_begin|>{"arg": "val"}<|tool_call_end|>
        <|tool_calls_section_end|>

    The function name is extracted from the ``functions.<name>:<index>``
    identifier. Arguments are JSON-encoded dictionaries.

    Think Block Handling:
        Models with reasoning capabilities may output draft tool calls
        inside <think>...</think> blocks. These are excluded by default
        to avoid executing planning/reasoning tool calls.

    Chat Template Notes:
        Kimi K2 uses no explicit separator between messages.
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

    @override
    def parse(self, text: str) -> list[ToolParseResult]:
        """Parse tool calls from Kimi K2 model output.

        Finds ``<|tool_calls_section_begin|>`` sections, then extracts
        individual calls with their function names and JSON arguments.

        Args:
            text: Model output text.

        Returns:
            List of tool call results (successful and errors).
        """
        # Remove think blocks to avoid parsing draft tool calls from reasoning
        text = self.think_pattern.sub("", text)

        tool_calls: list[ToolParseResult] = []
        call_index = 0

        for section_match in self.SECTION_PATTERN.finditer(text):
            section = section_match.group(1)

            for call_match in self.CALL_PATTERN.finditer(section):
                raw_id = call_match.group("tool_call_id")
                raw_args = call_match.group("arguments").strip()
                tool_call_id = f"call_{call_index:04d}"  # Sequential IDs for sortability
                call_index += 1

                # Extract function name from "functions.func_name:index"
                # Use split(".", 1) and rsplit(":", 1) to handle dots in names
                if "." in raw_id:
                    name = raw_id.split(".", 1)[1].rsplit(":", 1)[0]
                else:
                    name = raw_id

                try:
                    arguments = json.loads(raw_args)
                    if not isinstance(arguments, dict):
                        logger.warning(f"Tool parse error: arguments is not a dict for {name}")
                        arguments = {}
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse arguments for {name}: {raw_args[:200]}")
                    tool_calls.append(ToolParseResult.from_parse_error(id=tool_call_id, raw=raw_args, name=name))
                    continue

                tool_calls.append(
                    ToolParseResult(
                        id=tool_call_id,
                        name=name,
                        input=arguments,
                    )
                )

        return tool_calls
