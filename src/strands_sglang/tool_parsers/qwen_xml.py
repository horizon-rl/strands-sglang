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

"""Qwen XML tool call parser."""

from __future__ import annotations

import logging
import re

from typing_extensions import override

from .base import ToolParser, ToolParseResult, register_tool_parser

logger = logging.getLogger(__name__)


@register_tool_parser("qwen_xml")
class QwenXMLToolParser(ToolParser):
    r"""Parser for Qwen3-Coder XML tool call format.

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
    """

    # Pattern to extract function name from <function=name>...</function>
    _FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)

    # Pattern to extract parameters from <parameter=name>value</parameter>
    _PARAMETER_PATTERN = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

    @override
    @property
    def message_separator(self) -> str:
        """Qwen Coder models use newline as separator between messages."""
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
        text = self.think_pattern.sub("", text)

        tool_calls: list[ToolParseResult] = []

        for i, match in enumerate(self.tool_pattern.finditer(text)):
            raw_content = match.group(1).strip()
            tool_call_id = f"call_{i:04d}"  # Sequential IDs for sortability

            # Parse the function tag
            func_match = self._FUNCTION_PATTERN.search(raw_content)
            if not func_match:
                logger.warning("Tool parse error: missing <function=...> tag")
                tool_calls.append(ToolParseResult.from_parse_error(id=tool_call_id, raw=raw_content))
                continue

            func_name = func_match.group(1).strip()
            func_body = func_match.group(2)

            if not func_name:
                logger.warning("Tool parse error: empty function name")
                tool_calls.append(ToolParseResult.from_parse_error(id=tool_call_id, raw=raw_content))
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
