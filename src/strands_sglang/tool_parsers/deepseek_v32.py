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

"""Tool call parser and prompt encoder for DeepSeek-V3.2 models.

DeepSeek-V3.2 uses DSML-prefixed XML-like tags with typed parameters.
All tags use the ``｜DSML｜`` prefix (fullwidth pipes U+FF5C)::

    <｜DSML｜function_calls>
    <｜DSML｜invoke name="func_name">
    <｜DSML｜parameter name="arg1" string="true">string_value</｜DSML｜parameter>
    <｜DSML｜parameter name="arg2" string="false">123</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

The ``string`` attribute controls value interpretation: ``"true"`` keeps the
raw string, ``"false"`` parses it as JSON (number, bool, list, object).

DeepSeek-V3.2 does not ship a Jinja chat template. When ``model_name_or_path``
is provided, :meth:`~DeepSeekV32ToolParser.format_prompt` loads the model's
``encoding/encoding_dsv32.py`` and delegates to it directly.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import types
from typing import Any

from typing_extensions import override

from .base import ToolParser, ToolParseResult, register_tool_parser

logger = logging.getLogger(__name__)

# DeepSeek-V3.2 special token characters
_FP = "\uff5c"  # ｜ fullwidth pipe
_HS = "\u2581"  # ▁ half-width space

# DSML prefix used in all tool call tags
_DSML = f"{_FP}DSML{_FP}"  # ｜DSML｜

# EOS token used as message separator between assistant output and tool results
_EOS_TOKEN = f"<{_FP}end{_HS}of{_HS}sentence{_FP}>"

# Tool call section delimiters
_FC_START = f"<{_DSML}function_calls>"
_FC_END = f"</{_DSML}function_calls>"


def _load_encoding_module(model_name_or_path: str) -> types.ModuleType:
    """Load ``encoding/encoding_dsv32.py`` from a HuggingFace model repo.

    Uses ``huggingface_hub.hf_hub_download`` for remote model IDs and falls
    back to direct file-system access for local paths.

    Args:
        model_name_or_path: HuggingFace model ID or local directory path.

    Returns:
        The imported encoding module.

    Raises:
        FileNotFoundError: If the encoding file cannot be found.
        ImportError: If ``huggingface_hub`` is not installed (remote models only).
    """
    import os

    encoding_filename = os.path.join("encoding", "encoding_dsv32.py")

    # Local path
    if os.path.isdir(model_name_or_path):
        filepath = os.path.join(model_name_or_path, encoding_filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Encoding file not found: {filepath}")
    else:
        # Remote HuggingFace model ID
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to download encoding_dsv32.py "
                f"from remote model {model_name_or_path!r}. "
                "Install it with: pip install huggingface_hub"
            ) from exc
        filepath = hf_hub_download(model_name_or_path, encoding_filename)

    spec = importlib.util.spec_from_file_location("encoding_dsv32", filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {filepath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info("Loaded encoding module from %s", filepath)
    return module


def _parse_param_value(value: str, is_string: bool) -> Any:
    """Parse a parameter value based on its string attribute.

    Args:
        value: Raw parameter value text.
        is_string: If True, return as-is. If False, attempt JSON decode.

    Returns:
        Parsed value (string, number, bool, list, or dict).
    """
    if is_string:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


@register_tool_parser("deepseek_v32")
class DeepSeekV32ToolParser(ToolParser):
    """Parser for DeepSeek-V3.2 DSML-prefixed tool call format.

    Format:
        <｜DSML｜function_calls>
        <｜DSML｜invoke name="func_name">
        <｜DSML｜parameter name="arg1" string="true">string_value</｜DSML｜parameter>
        <｜DSML｜parameter name="arg2" string="false">123</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜function_calls>

    Parameters have a ``string`` attribute that controls value parsing:
    ``"true"`` preserves the raw string, ``"false"`` attempts JSON decode
    for non-string types (numbers, booleans, lists, objects).

    When no ``<｜DSML｜parameter>`` tags are found, the invoke body is tried
    as raw JSON as a fallback.

    Think Block Handling:
        Models with reasoning capabilities may output draft tool calls
        inside <think>...</think> blocks. These are excluded by default
        to avoid executing planning/reasoning tool calls.

    Chat Template Notes:
        DeepSeek-V3.2 does not ship a Jinja chat template. Pass
        ``model_name_or_path`` to load the model's own
        ``encoding/encoding_dsv32.py`` for prompt formatting via
        :meth:`format_prompt`.
    """

    skip_special_tokens: bool = False

    INVOKE_PATTERN = re.compile(
        rf"<{re.escape(_DSML)}invoke\s+name=\"([^\"]+)\"\s*>(.*?)</{re.escape(_DSML)}invoke>",
        re.DOTALL,
    )

    PARAM_PATTERN = re.compile(
        rf"<{re.escape(_DSML)}parameter\s+name=\"([^\"]+)\"\s+string=\"(true|false)\"\s*>"
        rf"(.*?)</{re.escape(_DSML)}parameter>",
        re.DOTALL,
    )

    def __init__(self, *, model_name_or_path: str = "deepseek-ai/DeepSeek-V3.2", **kwargs: Any) -> None:
        """Initialize the DeepSeek-V3.2 tool parser.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
                Defaults to ``"deepseek-ai/DeepSeek-V3.2"`` which auto-downloads
                ``encoding/encoding_dsv32.py`` via ``hf_hub_download``.
            **kwargs: Forwarded to :class:`ToolParser` (custom token overrides).
        """
        kwargs.setdefault("tool_start_token", _FC_START)
        kwargs.setdefault("tool_end_token", _FC_END)
        super().__init__(**kwargs)
        self._encoding_module = _load_encoding_module(model_name_or_path)

    @override
    @property
    def message_separator(self) -> str:
        """DeepSeek-V3.2 EOS token to terminate assistant turn before tool results."""
        return _EOS_TOKEN

    @override
    def format_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> str | None:
        """Format messages using the model's ``encoding_dsv32.encode_messages()``.

        For the first call (full conversation), delegates to the encoding module.
        For incremental calls (tool-result-only messages), uses
        :meth:`_format_incremental` to avoid ``encode_messages()`` crashing
        when it looks backward for a preceding assistant message.
        """
        thinking_mode = "thinking" if enable_thinking else "chat"

        # Incremental path: only tool results, no system/user context
        if all(m.get("role") == "tool" for m in messages):
            return self._format_incremental(messages, thinking_mode, add_generation_prompt)

        # The reference render_message reads tools from msg.get("tools"),
        # so attach them to the system message for encode_messages to handle.
        if tools:
            messages = [dict(m) for m in messages]  # shallow copy to avoid mutation
            if messages and messages[0].get("role") == "system":
                messages[0] = {**messages[0], "tools": tools}
            else:
                messages.insert(0, {"role": "system", "content": "", "tools": tools})

        return self._encoding_module.encode_messages(messages, thinking_mode=thinking_mode)

    def _format_incremental(
        self, messages: list[dict[str, Any]], thinking_mode: str, add_generation_prompt: bool
    ) -> str:
        """Format tool result messages for incremental tokenization.

        ``encode_messages()`` expects the full conversation and looks backward
        for the preceding assistant message with tool calls. On subsequent turns
        only new messages (tool results) are passed, causing an AssertionError.

        This method formats tool results directly and appends the generation
        prompt so the model knows to start its next reasoning turn.
        """
        result = "\n\n<function_results>"
        for msg in messages:
            if msg.get("role") == "tool":
                result += "\n<result>" + msg.get("content", "") + "</result>"
        result += "\n</function_results>"
        if add_generation_prompt:
            gen = "<think>" if thinking_mode == "thinking" else "</think>"
            result += "\n\n" + gen
        return result

    # ── Parsing ───────────────────────────────────────────────────────────

    @override
    def parse(self, text: str) -> list[ToolParseResult]:
        """Parse tool calls from DeepSeek-V3.2 model output.

        Extracts ``<｜DSML｜invoke>`` blocks from within
        ``<｜DSML｜function_calls>`` sections, then parses typed
        ``<｜DSML｜parameter>`` tags into arguments.

        Args:
            text: Model output text.

        Returns:
            List of tool call results (successful and errors).
        """
        # Remove think blocks to avoid parsing draft tool calls from reasoning
        text = self.think_pattern.sub("", text)

        tool_calls: list[ToolParseResult] = []
        call_index = 0

        for fc_match in self.tool_pattern.finditer(text):
            fc_content = fc_match.group(1)

            for invoke_match in self.INVOKE_PATTERN.finditer(fc_content):
                func_name = invoke_match.group(1)
                invoke_body = invoke_match.group(2)
                tool_call_id = f"call_{call_index:04d}"  # Sequential IDs for sortability
                call_index += 1

                # Parse typed <｜DSML｜parameter> tags
                arguments: dict[str, Any] = {}
                for param_match in self.PARAM_PATTERN.finditer(invoke_body):
                    param_name = param_match.group(1)
                    is_string = param_match.group(2) == "true"
                    raw_value = param_match.group(3)
                    arguments[param_name] = _parse_param_value(raw_value, is_string)

                if not arguments:
                    # Fallback: try parsing invoke body as raw JSON
                    body = invoke_body.strip()
                    if body:
                        try:
                            arguments = json.loads(body)
                            if not isinstance(arguments, dict):
                                logger.warning("Tool parse error: arguments is not a dict for %s", func_name)
                                arguments = {}
                        except json.JSONDecodeError:
                            logger.warning("Tool parse error: no params and body not JSON for %s", func_name)
                            tool_calls.append(
                                ToolParseResult.from_parse_error(id=tool_call_id, raw=invoke_body, name=func_name)
                            )
                            continue

                tool_calls.append(
                    ToolParseResult(
                        id=tool_call_id,
                        name=func_name,
                        input=arguments,
                    )
                )

        return tool_calls
