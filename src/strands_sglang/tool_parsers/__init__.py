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

"""Tool call parsers for different model chat templates."""

from .base import TOOL_PARSER_REGISTRY, ToolParser, ToolParseResult, get_tool_parser

# Import parsers to trigger registration via @register_tool_parser decorator
from .glm import GLMToolParser
from .hermes import HermesToolParser
from .kimi_k2 import KimiK2ToolParser
from .qwen_xml import QwenXMLToolParser

__all__ = [
    # Base
    "ToolParseResult",
    "ToolParser",
    # Parsers
    "GLMToolParser",
    "HermesToolParser",
    "KimiK2ToolParser",
    "QwenXMLToolParser",
    # Registry
    "TOOL_PARSER_REGISTRY",
    "get_tool_parser",
]
