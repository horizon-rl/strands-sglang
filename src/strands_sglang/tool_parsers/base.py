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

"""Base classes for tool call parsing."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

# Parser registry - populated by @register_tool_parser decorator
TOOL_PARSER_REGISTRY: dict[str, type[ToolParser]] = {}

T = TypeVar("T", bound="ToolParser")


@dataclass(frozen=True)
class ToolParseResult:
    """A parsed tool call request.

    Notes:
        - For successful parses: name and input are populated, raw is `None`.
        - For parse errors: name is extracted or `UNKNOWN_NAME`, raw contains the unparsable content.
    """

    UNKNOWN_NAME: ClassVar[str] = "unknown_tool"

    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    raw: str | None = None

    @classmethod
    def from_parse_error(cls, id: str, raw: str, name: str | None = None) -> ToolParseResult:
        """Create an error result with unparsable `raw` content for model self-correction."""
        return cls(id=id, name=name if name is not None else cls.UNKNOWN_NAME, input={}, raw=raw)

    @property
    def is_error(self) -> bool:
        """Check if this represents a parse error."""
        return self.raw is not None

    @property
    def payload(self) -> str:
        """JSON-encoded input on success, raw content on error."""
        if self.is_error:
            return self.raw or ""
        return json.dumps(self.input)


class ToolParser(ABC):
    """Base class for tool call parsers.

    Notes:
        - Subclasses implement `parse` to extract tool calls from model output:
            - `<think>` blocks should be excluded to avoid parsing draft tool calls from reasoning.
            - Best-effort extracted tool name is used for error handling.
        - Only `json.JSONDecodeError` is handled; arguments are passed as-is
          and validated by Strands downstream.
    """

    DEFAULT_TOOL_START_TOKEN = "<tool_call>"
    DEFAULT_TOOL_END_TOKEN = "</tool_call>"
    DEFAULT_THINK_START_TOKEN = "<think>"
    DEFAULT_THINK_END_TOKEN = "</think>"

    def __init__(
        self,
        tool_start_token: str = DEFAULT_TOOL_START_TOKEN,
        tool_end_token: str = DEFAULT_TOOL_END_TOKEN,
        think_start_token: str = DEFAULT_THINK_START_TOKEN,
        think_end_token: str = DEFAULT_THINK_END_TOKEN,
    ) -> None:
        """Initialize a `ToolParser` instance."""
        self.tool_start_token = tool_start_token
        self.tool_end_token = tool_end_token
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token

        # Pattern to extract tool call content (with whitespace trimming)
        self.tool_pattern = re.compile(
            rf"{re.escape(tool_start_token)}\s*(.*?)\s*{re.escape(tool_end_token)}",
            re.DOTALL,
        )
        # Pattern to remove think blocks (no capture needed)
        self.think_pattern = re.compile(
            rf"{re.escape(think_start_token)}.*?{re.escape(think_end_token)}",
            re.DOTALL,
        )

    @abstractmethod
    def parse(self, text: str) -> list[ToolParseResult]:
        """Parse tool calls from model output text."""
        ...


def register_tool_parser(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a tool parser class under `name`."""

    def decorator(cls: type[T]) -> type[T]:
        TOOL_PARSER_REGISTRY[name] = cls
        return cls

    return decorator


def get_tool_parser(name: str, **kwargs: Any) -> ToolParser:
    """Get a registered tool parser by name."""
    if name not in TOOL_PARSER_REGISTRY:
        available = ", ".join(sorted(TOOL_PARSER_REGISTRY.keys()))
        raise KeyError(f"Unknown tool parser: {name!r}. Available: {available}")
    return TOOL_PARSER_REGISTRY[name](**kwargs)
