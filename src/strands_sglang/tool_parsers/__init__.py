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

"""Tool call parsers for different model formats.

This module provides parsers that extract tool calls from model outputs.
Different models use different formats for tool calls in their chat templates.

Design for RL Training:
- Only handle `JSONDecodeError` (can't extract anything from malformed JSON)
- Let Strands validate arguments against tool schemas
- Parse errors become tool calls with error info for model feedback

Adding a new parser:
    1. Create a new file (e.g., `my_parser.py`)
    2. Decorate the class with `@register_tool_parser("my_parser")`
    3. Import the module here to trigger registration
"""

from .base import TOOL_PARSER_REGISTRY, UNKNOWN_TOOL_NAME, ToolParser, ToolParseResult, get_tool_parser

# Import parsers to trigger registration via @register_tool_parser decorator
from .hermes import HermesToolParser
from .qwen_xml import QwenXMLToolParser

__all__ = [
    # Base
    "ToolParseResult",
    "ToolParser",
    "UNKNOWN_TOOL_NAME",
    # Parsers
    "HermesToolParser",
    "QwenXMLToolParser",
    # Registry
    "TOOL_PARSER_REGISTRY",
    "get_tool_parser",
]
