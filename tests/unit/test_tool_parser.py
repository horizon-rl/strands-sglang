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

"""Unit tests for tool_parser module."""

import pytest

from strands_sglang.tool_parsers import (
    GLMToolParser,
    HermesToolParser,
    KimiK2ToolParser,
    QwenXMLToolParser,
    ToolParseResult,
)


class TestToolParseResult:
    """Tests for ToolParseResult dataclass."""

    def test_successful_parse_result(self):
        """Successful parse: raw=None, payload is JSON-encoded input."""
        result = ToolParseResult(id="call_123", name="my_tool", input={"arg": "value"})
        assert result.name == "my_tool"
        assert result.input == {"arg": "value"}
        assert result.is_error is False
        assert result.payload == '{"arg": "value"}'

    def test_error_parse_result(self):
        """Error parse: raw set, payload returns raw content."""
        result = ToolParseResult(id="call_456", name="unknown_tool", input={}, raw='{"malformed": json}')
        assert result.is_error is True
        assert result.payload == '{"malformed": json}'

    def test_from_parse_error_defaults(self):
        """from_parse_error creates error with UNKNOWN_NAME and empty input."""
        result = ToolParseResult.from_parse_error(id="call_0", raw="bad data")
        assert result.is_error is True
        assert result.name == ToolParseResult.UNKNOWN_NAME
        assert result.input == {}

    def test_immutability(self):
        """ToolParseResult is frozen."""
        result = ToolParseResult(id="call_123", name="tool", input={})
        with pytest.raises(AttributeError):
            result.name = "other_tool"


class TestHermesToolParser:
    """Tests for HermesToolParser."""

    @pytest.fixture
    def parser(self):
        return HermesToolParser()

    def test_parse_single_tool_call(self, parser):
        text = '<tool_call>{"name": "calculator", "arguments": {"x": 1, "y": 2}}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "calculator"
        assert results[0].input == {"x": 1, "y": 2}
        assert results[0].is_error is False

    def test_parse_multiple_tool_calls_with_sequential_ids(self, parser):
        """Multiple calls get sequential call_NNNN IDs."""
        text = """
        <tool_call>{"name": "tool_a", "arguments": {"a": 1}}</tool_call>
        Some text in between
        <tool_call>{"name": "tool_b", "arguments": {"b": 2}}</tool_call>
        <tool_call>{"name": "tool_c", "arguments": {}}</tool_call>
        """
        results = parser.parse(text)

        assert len(results) == 3
        assert [r.name for r in results] == ["tool_a", "tool_b", "tool_c"]
        assert [r.id for r in results] == ["call_0000", "call_0001", "call_0002"]

    def test_parse_no_tool_calls(self, parser):
        assert parser.parse("Just some regular text.") == []
        assert parser.parse("") == []

    def test_missing_arguments_defaults_to_empty_dict(self, parser):
        text = '<tool_call>{"name": "no_args"}</tool_call>'
        results = parser.parse(text)
        assert results[0].input == {}
        assert results[0].is_error is False

    def test_parse_complex_arguments(self, parser):
        text = """<tool_call>{"name": "complex", "arguments": {
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [{"x": 1}, {"y": 2}]
        }}</tool_call>"""
        results = parser.parse(text)
        assert results[0].input["nested"] == {"a": 1, "b": [1, 2, 3]}
        assert results[0].input["list"] == [{"x": 1}, {"y": 2}]

    @pytest.mark.parametrize(
        "content",
        [
            '{"arguments": {"x": 1}}',  # missing name
            '{"name": "", "arguments": {}}',  # empty name
            '{"name": 123, "arguments": {}}',  # non-string name
            "[1, 2, 3]",  # non-dict JSON
            "null",  # null JSON
            "{completely broken}",  # broken JSON
        ],
    )
    def test_error_cases(self, parser, content):
        results = parser.parse(f"<tool_call>{content}</tool_call>")
        assert results[0].is_error is True

    def test_name_extraction_from_malformed_json(self, parser):
        """Regex extracts name even when JSON is malformed."""
        results = parser.parse('<tool_call>{"name": "broken", "arguments": {bad}}</tool_call>')
        assert results[0].is_error is True
        assert results[0].name == "broken"

    def test_mixed_success_and_errors(self, parser):
        text = """
        <tool_call>{"name": "first", "arguments": {}}</tool_call>
        <tool_call>{broken}</tool_call>
        <tool_call>{"name": "second", "arguments": {"x": 1}}</tool_call>
        """
        results = parser.parse(text)

        assert len(results) == 3
        successful = [r for r in results if not r.is_error]
        assert [r.name for r in successful] == ["first", "second"]

    def test_exclude_tool_calls_inside_think_block(self, parser):
        text = """
        <think>
        <tool_call>{"name": "draft", "arguments": {}}</tool_call>
        </think>
        <tool_call>{"name": "actual", "arguments": {"y": 2}}</tool_call>
        """
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "actual"

    def test_custom_tokens(self):
        parser = HermesToolParser(tool_start_token="<function>", tool_end_token="</function>")
        assert parser.parse('<function>{"name": "custom", "arguments": {}}</function>')[0].name == "custom"
        assert parser.parse('<tool_call>{"name": "ignored", "arguments": {}}</tool_call>') == []

    def test_custom_think_tokens(self):
        parser = HermesToolParser(think_start_token="<reasoning>", think_end_token="</reasoning>")
        text = """
        <reasoning><tool_call>{"name": "draft", "arguments": {}}</tool_call></reasoning>
        <tool_call>{"name": "actual", "arguments": {}}</tool_call>
        """
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "actual"


class TestQwenXMLToolParser:
    """Tests for QwenXMLToolParser (Qwen3-Coder format)."""

    @pytest.fixture
    def parser(self):
        return QwenXMLToolParser()

    def test_parse_single_tool_call(self, parser):
        text = """<tool_call>
<function=calculator>
<parameter=x>1</parameter>
<parameter=y>2</parameter>
</function>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "calculator"
        assert results[0].input == {"x": "1", "y": "2"}
        assert results[0].is_error is False

    def test_parse_multiple_tool_calls_with_sequential_ids(self, parser):
        """Multiple calls get sequential call_NNNN IDs."""
        text = """
<tool_call><function=tool_a><parameter=a>1</parameter></function></tool_call>
<tool_call><function=tool_b><parameter=b>2</parameter></function></tool_call>
<tool_call><function=tool_c></function></tool_call>
"""
        results = parser.parse(text)

        assert len(results) == 3
        assert [r.name for r in results] == ["tool_a", "tool_b", "tool_c"]
        assert [r.id for r in results] == ["call_0000", "call_0001", "call_0002"]

    def test_parse_no_tool_calls(self, parser):
        assert parser.parse("Just some regular text.") == []
        assert parser.parse("") == []

    def test_parse_multiline_parameter_value(self, parser):
        text = """<tool_call>
<function=write_file>
<parameter=path>/tmp/test.py</parameter>
<parameter=content>
def hello():
    print("Hello, World!")
</parameter>
</function>
</tool_call>"""
        results = parser.parse(text)

        assert results[0].name == "write_file"
        assert "def hello():" in results[0].input["content"]

    def test_parse_missing_function_tag(self, parser):
        text = """<tool_call>
<parameter=x>1</parameter>
</tool_call>"""
        results = parser.parse(text)

        assert results[0].is_error is True
        assert results[0].name == ToolParseResult.UNKNOWN_NAME

    def test_exclude_tool_calls_inside_think_block(self, parser):
        text = """
<think>
<tool_call><function=draft><parameter=x>1</parameter></function></tool_call>
</think>
<tool_call><function=actual><parameter=y>2</parameter></function></tool_call>
"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "actual"

    def test_real_world_git_status_example(self, parser):
        """Parse real-world example from Qwen3-Coder."""
        text = """I'll check the git status for you.

<tool_call>
<function=run_terminal_command>
<parameter=command>
git status
</parameter>
<parameter=waitForCompletion>
True
</parameter>
</function>
</tool_call>"""
        results = parser.parse(text)

        assert results[0].name == "run_terminal_command"
        assert results[0].input["command"] == "git status"
        assert results[0].input["waitForCompletion"] == "True"


class TestGLMToolParser:
    """Tests for GLMToolParser (GLM-4 and ChatGLM format)."""

    @pytest.fixture
    def parser(self):
        return GLMToolParser()

    def test_parse_single_tool_call(self, parser):
        text = """<tool_call>calculator
<arg_key>x</arg_key>
<arg_value>1</arg_value>
<arg_key>y</arg_key>
<arg_value>2</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "calculator"
        assert results[0].input == {"x": 1, "y": 2}
        assert results[0].is_error is False

    def test_parse_multiple_tool_calls_with_sequential_ids(self, parser):
        """Multiple calls get sequential call_NNNN IDs."""
        text = """
<tool_call>tool_a
<arg_key>a</arg_key>
<arg_value>1</arg_value>
</tool_call>
<tool_call>tool_b
<arg_key>b</arg_key>
<arg_value>2</arg_value>
</tool_call>
<tool_call>tool_c
</tool_call>
"""
        results = parser.parse(text)

        assert len(results) == 3
        assert [r.name for r in results] == ["tool_a", "tool_b", "tool_c"]
        assert [r.id for r in results] == ["call_0000", "call_0001", "call_0002"]

    def test_parse_no_tool_calls(self, parser):
        assert parser.parse("Just some regular text.") == []
        assert parser.parse("") == []

    def test_parse_value_types(self, parser):
        """JSON-encoded values are decoded; plain strings kept as-is."""
        text = """<tool_call>tool
<arg_key>integer</arg_key>
<arg_value>42</arg_value>
<arg_key>string</arg_key>
<arg_value>"hello"</arg_value>
<arg_key>nested</arg_key>
<arg_value>{"a": [1, 2]}</arg_value>
<arg_key>plain</arg_key>
<arg_value>hello world</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert results[0].input["integer"] == 42
        assert results[0].input["string"] == "hello"
        assert results[0].input["nested"] == {"a": [1, 2]}
        assert results[0].input["plain"] == "hello world"

    def test_parse_multiline_value(self, parser):
        text = """<tool_call>write_file
<arg_key>path</arg_key>
<arg_value>/tmp/test.py</arg_value>
<arg_key>content</arg_key>
<arg_value>def hello():
    print("Hello, World!")</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert results[0].name == "write_file"
        assert "def hello():" in results[0].input["content"]

    def test_parse_missing_function_name(self, parser):
        text = """<tool_call>
<arg_key>x</arg_key>
<arg_value>1</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert results[0].is_error is True
        assert results[0].name == ToolParseResult.UNKNOWN_NAME

    def test_exclude_tool_calls_inside_think_block(self, parser):
        text = """
<think>
<tool_call>draft
<arg_key>x</arg_key>
<arg_value>1</arg_value>
</tool_call>
</think>
<tool_call>actual
<arg_key>y</arg_key>
<arg_value>2</arg_value>
</tool_call>
"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "actual"
        assert results[0].input == {"y": 2}

    def test_real_world_example(self, parser):
        """Parse real-world example from GLM-4."""
        text = """I'll search for that information.

<tool_call>web_search
<arg_key>query</arg_key>
<arg_value>Python asyncio tutorial</arg_value>
<arg_key>limit</arg_key>
<arg_value>5</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert results[0].name == "web_search"
        assert results[0].input["query"] == "Python asyncio tutorial"
        assert results[0].input["limit"] == 5


class TestKimiK2ToolParser:
    """Tests for KimiK2ToolParser (Kimi K2/K2.5 format)."""

    @pytest.fixture
    def parser(self):
        return KimiK2ToolParser()

    def test_parse_single_tool_call(self, parser):
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.calculator:0"
            '<|tool_call_argument_begin|>{"x": 1, "y": 2}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "calculator"
        assert results[0].input == {"x": 1, "y": 2}
        assert results[0].id == "functions.calculator:0"
        assert results[0].is_error is False

    def test_parse_multiple_tool_calls_preserves_raw_ids(self, parser):
        """Multiple calls preserve raw IDs for chat template round-trip."""
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.tool_a:0"
            '<|tool_call_argument_begin|>{"a": 1}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
            "Some text"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.tool_b:0"
            '<|tool_call_argument_begin|>{"b": 2}'
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.tool_c:1"
            "<|tool_call_argument_begin|>{}"
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        results = parser.parse(text)

        assert len(results) == 3
        assert [r.name for r in results] == ["tool_a", "tool_b", "tool_c"]
        assert [r.id for r in results] == ["functions.tool_a:0", "functions.tool_b:0", "functions.tool_c:1"]

    def test_parse_no_tool_calls(self, parser):
        assert parser.parse("Just some regular text.") == []
        assert parser.parse("") == []

    @pytest.mark.parametrize(
        "func_id, expected_name",
        [
            ("functions.my_tool:0", "my_tool"),  # standard dotted format
            ("functions.web_search:0", "web_search"),  # underscored
            ("functions.web-search:0", "web-search"),  # hyphenated
            ("functions.module.sub_tool:0", "module.sub_tool"),  # nested dotted
            ("calculator:0", "calculator"),  # no functions. prefix — still extracts name
        ],
    )
    def test_function_name_extraction(self, parser, func_id, expected_name):
        text = (
            "<|tool_calls_section_begin|>"
            f"<|tool_call_begin|>{func_id}"
            '<|tool_call_argument_begin|>{"x": 1}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        results = parser.parse(text)

        assert results[0].name == expected_name

    def test_parse_malformed_json(self, parser):
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.tool:0"
            "<|tool_call_argument_begin|>{malformed json}"
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        results = parser.parse(text)

        assert results[0].is_error is True
        assert results[0].name == "tool"
        assert results[0].raw == "{malformed json}"

    def test_parse_mixed_valid_and_invalid(self, parser):
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.good:0"
            '<|tool_call_argument_begin|>{"a": 1}'
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.bad:1"
            "<|tool_call_argument_begin|>{broken"
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.also_good:2"
            '<|tool_call_argument_begin|>{"b": 2}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        results = parser.parse(text)

        assert len(results) == 3
        assert results[0].is_error is False
        assert results[1].is_error is True
        assert results[2].is_error is False

    def test_parse_ignores_tool_calls_in_think_block(self, parser):
        text = (
            "<think>"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.draft:0"
            '<|tool_call_argument_begin|>{"a": 1}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
            "</think>"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.real:0"
            '<|tool_call_argument_begin|>{"b": 2}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "real"


class TestToolParserRegistry:
    """Tests for tool parser registry."""

    @pytest.mark.parametrize(
        "name, expected_cls",
        [
            ("hermes", HermesToolParser),
            ("qwen_xml", QwenXMLToolParser),
            ("glm", GLMToolParser),
            ("kimi_k2", KimiK2ToolParser),
        ],
    )
    def test_registered_parser(self, name, expected_cls):
        """Each parser is registered and instantiable by name."""
        from strands_sglang.tool_parsers import TOOL_PARSER_REGISTRY, get_tool_parser

        assert name in TOOL_PARSER_REGISTRY
        assert TOOL_PARSER_REGISTRY[name] is expected_cls
        assert isinstance(get_tool_parser(name), expected_cls)

    def test_get_parser_with_kwargs(self):
        from strands_sglang.tool_parsers import get_tool_parser

        parser = get_tool_parser("hermes", think_start_token="<reasoning>")
        assert parser.think_start_token == "<reasoning>"

    def test_unknown_parser_raises(self):
        from strands_sglang.tool_parsers import get_tool_parser

        with pytest.raises(KeyError, match="Unknown tool parser"):
            get_tool_parser("nonexistent")
