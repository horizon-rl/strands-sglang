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
