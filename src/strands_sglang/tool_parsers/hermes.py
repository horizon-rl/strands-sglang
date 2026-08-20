from __future__ import annotations

import json
import logging
import re
import uuid
from typing import override

from .base import ToolParser, ToolParseResult, register_tool_parser

logger = logging.getLogger(__name__)


@register_tool_parser("hermes")
class HermesToolParser(ToolParser):
    r"""Parser for Hermes/Qwen JSON tool call format.

    Format:

        <tool_call>{"name": "func", "arguments": {"arg": "value"}}</tool_call>

    Used by Qwen2.5/Qwen3 and NousResearch/Hermes models.
    """

    _NAME_PATTERN = re.compile(r'"name"\s*:\s*"([^"]+)"')

    @override
    def parse(self, text: str) -> list[ToolParseResult]:
        """Parse tool calls from model output."""
        # Remove think blocks to avoid parsing draft tool calls from reasoning
        text = self.think_pattern.sub("", text)

        tool_calls: list[ToolParseResult] = []
        prefix = uuid.uuid4().hex[:8]

        for i, match in enumerate(self.tool_pattern.finditer(text)):
            raw_content = match.group(1).strip()
            tool_call_id = f"call_{prefix}{i:04d}"  # Unique prefix + sequential index

            # Only handle JSONDecodeError - let Strands validate the rest
            try:
                call_json = json.loads(raw_content)
            except json.JSONDecodeError as e:
                name_match = self._NAME_PATTERN.search(raw_content)
                name = name_match.group(1) if name_match else ToolParseResult.UNKNOWN_NAME
                logger.warning("Tool parse error: %s", e)
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
