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

"""Tool call parser for DeepSeek-V3.2 models."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from .base import ToolParser, ToolParseResult, register_tool_parser

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@register_tool_parser("deepseek_v32")
class DeepSeekV32ToolParser(ToolParser):
    """Parser for DeepSeek-V3.2 DSML-prefixed XML tool call format.

    Format::

        <｜DSML｜function_calls>
        <｜DSML｜invoke name="func_name">
        <｜DSML｜parameter name="arg" string="true">value</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜function_calls>

    Parameters include a `string` attribute indicating whether the value is a
    raw string (`string="true"`) or a JSON-encoded value (`string="false"`).

    Special Token Handling:
        DSML tokens are special tokens in the DeepSeek tokenizer. This parser
        requires `skip_special_tokens=False` so SGLang preserves them in the output.

    Chat Template Notes:
        DeepSeek-V3.2 uses `<｜end▁of▁sentence｜>` as EOS / message separator.
    """

    INVOKE_PATTERN = re.compile(
        r"<｜DSML｜invoke\s+name=\"([^\"]+)\"\s*>(.*?)</｜DSML｜invoke>",
        re.DOTALL,
    )

    PARAM_PATTERN = re.compile(
        r"<｜DSML｜parameter\s+name=\"([^\"]+)\"\s+string=\"(true|false)\"\s*>(.*?)</｜DSML｜parameter>",
        re.DOTALL,
    )

    def __init__(self) -> None:
        """Initialize a `DeepSeekV32ToolParser` instance."""
        super().__init__(
            tool_start_token="<｜DSML｜function_calls>",
            tool_end_token="</｜DSML｜function_calls>",
        )

    @override
    def validate_tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Validate that the tokenizer has DeepSeek-V3.2 encoding attached."""
        if not getattr(tokenizer, "_dsv32_encoding_attached", False):
            raise ValueError(
                "`DeepSeekV32ToolParser` requires `attach_dsv32_encoding(tokenizer)` "
                "to be called before use. See `strands_sglang.utils.attach_dsv32_encoding`."
            )

    @property
    @override
    def message_separator(self) -> str:
        """DeepSeek-V3.2 uses its EOS token as message separator."""
        return "<｜end▁of▁sentence｜>"

    @override
    def parse(self, text: str) -> list[ToolParseResult]:
        """Parse tool calls from DeepSeek-V3.2 model output.

        Finds `<｜DSML｜function_calls>` sections, then extracts
        `<｜DSML｜invoke>` blocks with their parameters.

        Args:
            text: Model output text (with special tokens preserved).

        Returns:
            List of tool call results (successful and errors).
        """
        text = self.think_pattern.sub("", text)

        tool_calls: list[ToolParseResult] = []
        call_index = 0

        for section_match in self.tool_pattern.finditer(text):
            section = section_match.group(1)

            for invoke_match in self.INVOKE_PATTERN.finditer(section):
                name = invoke_match.group(1)
                body = invoke_match.group(2)
                tool_call_id = f"call_{call_index:04d}"
                call_index += 1

                arguments: dict[str, Any] = {}
                for param_name, string_flag, param_value in self.PARAM_PATTERN.findall(body):
                    if string_flag == "true":
                        arguments[param_name] = param_value
                    else:
                        try:
                            arguments[param_name] = json.loads(param_value)
                        except (json.JSONDecodeError, ValueError):
                            arguments[param_name] = param_value

                tool_calls.append(ToolParseResult(id=tool_call_id, name=name, input=arguments))

        return tool_calls
