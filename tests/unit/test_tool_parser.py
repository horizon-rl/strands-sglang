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

"""Unit tests for tool_parser module."""

import pytest

from strands_sglang.tool_parsers import (
    UNKNOWN_TOOL_NAME,
    GLMToolParser,
    HermesToolParser,
    QwenXMLToolParser,
    ToolParseResult,
)


class TestToolParseResult:
    """Tests for ToolParseResult dataclass."""

    def test_successful_parse_result(self):
        """Successful parse has raw=None."""
        result = ToolParseResult(
            id="call_123",
            name="my_tool",
            input={"arg": "value"},
        )
        assert result.id == "call_123"
        assert result.name == "my_tool"
        assert result.input == {"arg": "value"}
        assert result.raw is None
        assert result.is_error is False

    def test_error_parse_result(self):
        """Error parse has raw set."""
        result = ToolParseResult(
            id="call_456",
            name="unknown_tool",
            input={},
            raw='{"malformed": json}',
        )
        assert result.is_error is True
        assert result.raw == '{"malformed": json}'

    def test_payload_success(self):
        """payload returns JSON-encoded input for successful parses."""
        result = ToolParseResult(
            id="call_123",
            name="my_tool",
            input={"arg": "value", "num": 42},
        )
        assert result.payload == '{"arg": "value", "num": 42}'

    def test_payload_empty(self):
        """payload returns empty JSON object for empty input."""
        result = ToolParseResult(id="call_123", name="my_tool", input={})
        assert result.payload == "{}"

    def test_payload_error(self):
        """payload returns raw content for error parses."""
        result = ToolParseResult(
            id="call_456",
            name="unknown_tool",
            input={},
            raw='{"malformed": json}',
        )
        assert result.payload == '{"malformed": json}'

    def test_payload_error_empty_raw(self):
        """payload returns empty string if raw is empty."""
        result = ToolParseResult(
            id="call_789",
            name="some_tool",
            input={},
            raw="",
        )
        # Note: empty raw still counts as error (raw is not None)
        assert result.is_error is True
        assert result.payload == ""

    def test_immutability(self):
        """ToolParseResult is frozen."""
        result = ToolParseResult(id="call_123", name="tool", input={})
        with pytest.raises(AttributeError):
            result.name = "other_tool"


class TestHermesToolParser:
    """Tests for HermesToolParser."""

    @pytest.fixture
    def parser(self):
        """Create a default parser."""
        return HermesToolParser()

    # --- Basic Parsing ---

    def test_parse_single_tool_call(self, parser):
        """Parse a single valid tool call."""
        text = '<tool_call>{"name": "calculator", "arguments": {"x": 1, "y": 2}}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "calculator"
        assert results[0].input == {"x": 1, "y": 2}
        assert results[0].is_error is False

    def test_parse_multiple_tool_calls(self, parser):
        """Parse multiple tool calls in one text."""
        text = """
        <tool_call>{"name": "tool_a", "arguments": {"a": 1}}</tool_call>
        Some text in between
        <tool_call>{"name": "tool_b", "arguments": {"b": 2}}</tool_call>
        """
        results = parser.parse(text)

        assert len(results) == 2
        assert results[0].name == "tool_a"
        assert results[0].input == {"a": 1}
        assert results[1].name == "tool_b"
        assert results[1].input == {"b": 2}

    def test_parse_no_tool_calls(self, parser):
        """Return empty list when no tool calls present."""
        text = "Just some regular text without any tool calls."
        results = parser.parse(text)

        assert len(results) == 0

    def test_parse_empty_string(self, parser):
        """Handle empty string."""
        results = parser.parse("")
        assert len(results) == 0

    # --- Arguments Handling ---

    def test_parse_missing_arguments(self, parser):
        """Missing arguments defaults to empty dict."""
        text = '<tool_call>{"name": "no_args_tool"}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "no_args_tool"
        assert results[0].input == {}
        assert results[0].is_error is False

    def test_parse_empty_arguments(self, parser):
        """Empty arguments object is valid."""
        text = '<tool_call>{"name": "empty_args", "arguments": {}}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].input == {}
        assert results[0].is_error is False

    def test_parse_non_dict_arguments(self, parser):
        """Non-dict arguments defaults to empty dict (Strands validates)."""
        text = '<tool_call>{"name": "bad_args", "arguments": [1, 2, 3]}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "bad_args"
        assert results[0].input == {}  # Defaults to empty
        assert results[0].is_error is False  # Not an error - let Strands validate

    def test_parse_complex_arguments(self, parser):
        """Parse complex nested arguments."""
        text = """<tool_call>{"name": "complex", "arguments": {
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [{"x": 1}, {"y": 2}],
            "string": "hello"
        }}</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].input["nested"] == {"a": 1, "b": [1, 2, 3]}
        assert results[0].input["list"] == [{"x": 1}, {"y": 2}]

    # --- Error Cases ---

    def test_parse_malformed_json(self, parser):
        """Malformed JSON creates error result."""
        text = '<tool_call>{"name": "broken", "arguments": {invalid json}}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].is_error is True
        assert results[0].name == "broken"  # Extracted via regex
        assert results[0].raw == '{"name": "broken", "arguments": {invalid json}}'

    def test_parse_missing_name(self, parser):
        """Missing name field creates error result."""
        text = '<tool_call>{"arguments": {"x": 1}}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].is_error is True
        assert results[0].name == UNKNOWN_TOOL_NAME

    def test_parse_empty_name(self, parser):
        """Empty string name creates error result."""
        text = '<tool_call>{"name": "", "arguments": {}}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].is_error is True

    def test_parse_non_string_name(self, parser):
        """Non-string name creates error result."""
        text = '<tool_call>{"name": 123, "arguments": {}}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].is_error is True

    def test_parse_non_dict_json(self, parser):
        """Non-dict JSON (e.g., array) creates error result."""
        text = "<tool_call>[1, 2, 3]</tool_call>"
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].is_error is True
        assert results[0].name == UNKNOWN_TOOL_NAME

    def test_parse_null_json(self, parser):
        """Null JSON creates error result."""
        text = "<tool_call>null</tool_call>"
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].is_error is True

    # --- Name Extraction from Malformed JSON ---

    def test_extract_name_from_malformed_json(self, parser):
        """Extract tool name via regex even when JSON is malformed."""
        text = '<tool_call>{"name": "my_tool", broken json here}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].is_error is True
        assert results[0].name == "my_tool"  # Extracted via regex!

    def test_fallback_to_unknown_when_no_name(self, parser):
        """Fall back to UNKNOWN_TOOL_NAME when name can't be extracted."""
        text = "<tool_call>{completely broken}</tool_call>"
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].is_error is True
        assert results[0].name == UNKNOWN_TOOL_NAME

    # --- Whitespace Handling ---

    def test_parse_with_whitespace(self, parser):
        """Handle whitespace around JSON."""
        text = """<tool_call>
            {
                "name": "spacy_tool",
                "arguments": {"key": "value"}
            }
        </tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "spacy_tool"
        assert results[0].is_error is False

    # --- Custom Tokens ---

    def test_custom_tokens(self):
        """Use custom tool_call_tokens."""
        parser = HermesToolParser(tool_call_tokens=("<function>", "</function>"))
        text = '<function>{"name": "custom", "arguments": {}}</function>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "custom"

    def test_custom_tokens_ignore_default(self):
        """Custom tokens ignore default format."""
        parser = HermesToolParser(tool_call_tokens=("<function>", "</function>"))
        # Default format should not be parsed
        text = '<tool_call>{"name": "ignored", "arguments": {}}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 0

    # --- Callable Interface ---

    def test_callable_returns_successful_only(self, parser):
        """__call__ returns only successful parses as dicts."""
        text = """
        <tool_call>{"name": "good", "arguments": {"x": 1}}</tool_call>
        <tool_call>{malformed json}</tool_call>
        <tool_call>{"name": "also_good", "arguments": {}}</tool_call>
        """
        results = parser(text)  # Using __call__

        assert len(results) == 2
        assert results[0]["name"] == "good"
        assert results[0]["input"] == {"x": 1}
        assert results[1]["name"] == "also_good"

    def test_callable_returns_dict_format(self, parser):
        """__call__ returns dicts with id, name, input keys."""
        text = '<tool_call>{"name": "tool", "arguments": {"a": 1}}</tool_call>'
        results = parser(text)

        assert len(results) == 1
        assert set(results[0].keys()) == {"id", "name", "input"}
        assert results[0]["name"] == "tool"
        assert results[0]["input"] == {"a": 1}
        assert results[0]["id"].startswith("call_")

    # --- Mixed Success and Error ---

    def test_mixed_success_and_errors(self, parser):
        """Parse text with both successful and error tool calls."""
        text = """
        <tool_call>{"name": "first", "arguments": {}}</tool_call>
        <tool_call>{broken}</tool_call>
        <tool_call>{"name": "second", "arguments": {"x": 1}}</tool_call>
        <tool_call>{"arguments": {}}</tool_call>
        <tool_call>{"name": "third", "arguments": {}}</tool_call>
        """
        results = parser.parse(text)

        assert len(results) == 5

        # Check successful ones
        successful = [r for r in results if not r.is_error]
        assert len(successful) == 3
        assert [r.name for r in successful] == ["first", "second", "third"]

        # Check errors
        errors = [r for r in results if r.is_error]
        assert len(errors) == 2

    # --- Edge Cases ---

    def test_tool_call_with_special_characters_in_name(self, parser):
        """Handle special characters in tool name."""
        text = '<tool_call>{"name": "my-tool_v2.0", "arguments": {}}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "my-tool_v2.0"

    def test_tool_call_with_unicode(self, parser):
        """Handle unicode in arguments."""
        text = '<tool_call>{"name": "unicode", "arguments": {"emoji": "🚀", "chinese": "你好"}}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].input["emoji"] == "🚀"
        assert results[0].input["chinese"] == "你好"

    def test_unclosed_tool_call_tag(self, parser):
        """Unclosed tag is not parsed."""
        text = '<tool_call>{"name": "unclosed", "arguments": {}}'
        results = parser.parse(text)

        assert len(results) == 0

    def test_nested_tags_not_supported(self, parser):
        """Nested tags parse the inner content only."""
        text = '<tool_call><tool_call>{"name": "inner", "arguments": {}}</tool_call></tool_call>'
        results = parser.parse(text)

        # Regex is non-greedy, so it matches the inner one
        assert len(results) == 1
        # The inner content starts with "<tool_call>" which is not valid JSON
        assert results[0].is_error is True

    def test_unique_ids_generated(self, parser):
        """Each tool call gets a unique ID."""
        text = """
        <tool_call>{"name": "a", "arguments": {}}</tool_call>
        <tool_call>{"name": "b", "arguments": {}}</tool_call>
        """
        results = parser.parse(text)

        assert len(results) == 2
        assert results[0].id != results[1].id
        assert results[0].id.startswith("call_")
        assert results[1].id.startswith("call_")

    def test_sequential_ids_are_sortable(self, parser):
        """Tool call IDs are sequential and sortable for result ordering."""
        text = """
        <tool_call>{"name": "first", "arguments": {}}</tool_call>
        <tool_call>{"name": "second", "arguments": {}}</tool_call>
        <tool_call>{"name": "third", "arguments": {}}</tool_call>
        """
        results = parser.parse(text)

        assert len(results) == 3
        # IDs should be sequential: call_0000, call_0001, call_0002
        assert results[0].id == "call_0000"
        assert results[1].id == "call_0001"
        assert results[2].id == "call_0002"
        # String comparison should preserve order
        assert results[0].id < results[1].id < results[2].id

    # --- Think Block Exclusion ---

    def test_exclude_tool_calls_inside_think_block(self, parser):
        """Tool calls inside <think> blocks are excluded."""
        text = """
        <think>
        Let me think about this...
        Maybe I should call <tool_call>{"name": "draft_tool", "arguments": {"x": 1}}</tool_call>
        No, let me reconsider...
        </think>
        <tool_call>{"name": "actual_tool", "arguments": {"y": 2}}</tool_call>
        """
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "actual_tool"
        assert results[0].input == {"y": 2}

    def test_exclude_multiple_think_blocks(self, parser):
        """Multiple <think> blocks are all excluded."""
        text = """
        <think>Draft 1: <tool_call>{"name": "draft1", "arguments": {}}</tool_call></think>
        <tool_call>{"name": "real1", "arguments": {}}</tool_call>
        <think>Draft 2: <tool_call>{"name": "draft2", "arguments": {}}</tool_call></think>
        <tool_call>{"name": "real2", "arguments": {}}</tool_call>
        """
        results = parser.parse(text)

        assert len(results) == 2
        assert results[0].name == "real1"
        assert results[1].name == "real2"

    def test_think_block_with_no_tool_calls(self, parser):
        """Think block without tool calls doesn't affect parsing."""
        text = """
        <think>Just some reasoning without tool calls...</think>
        <tool_call>{"name": "my_tool", "arguments": {}}</tool_call>
        """
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "my_tool"

    def test_no_think_blocks(self, parser):
        """Parsing works normally when no think blocks present."""
        text = '<tool_call>{"name": "tool", "arguments": {}}</tool_call>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "tool"

    def test_disable_think_block_exclusion(self):
        """Setting think_tokens=None disables exclusion."""
        parser = HermesToolParser(think_tokens=None)
        text = """
        <think>
        <tool_call>{"name": "inside_think", "arguments": {}}</tool_call>
        </think>
        <tool_call>{"name": "outside_think", "arguments": {}}</tool_call>
        """
        results = parser.parse(text)

        # Both should be parsed when exclusion is disabled
        assert len(results) == 2
        assert results[0].name == "inside_think"
        assert results[1].name == "outside_think"

    def test_custom_think_tokens(self):
        """Custom think tokens work correctly."""
        parser = HermesToolParser(think_tokens=("<reasoning>", "</reasoning>"))
        text = """
        <reasoning>
        <tool_call>{"name": "draft", "arguments": {}}</tool_call>
        </reasoning>
        <tool_call>{"name": "actual", "arguments": {}}</tool_call>
        """
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "actual"

    def test_custom_think_tokens_ignore_default(self):
        """Custom think tokens don't exclude default <think> blocks."""
        parser = HermesToolParser(think_tokens=("<reasoning>", "</reasoning>"))
        text = """
        <think>
        <tool_call>{"name": "in_think", "arguments": {}}</tool_call>
        </think>
        <tool_call>{"name": "outside", "arguments": {}}</tool_call>
        """
        results = parser.parse(text)

        # <think> is NOT excluded with custom tokens, so both are parsed
        assert len(results) == 2
        assert results[0].name == "in_think"
        assert results[1].name == "outside"

    def test_multiline_think_block(self, parser):
        """Multiline think blocks are properly excluded."""
        text = """<think>
This is a long reasoning block
that spans multiple lines.

Let me consider calling:
<tool_call>{"name": "considered_tool", "arguments": {"query": "test"}}</tool_call>

Actually, I should use a different approach.
</think>
<tool_call>{"name": "final_tool", "arguments": {"query": "real"}}</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "final_tool"
        assert results[0].input == {"query": "real"}


class TestQwenXMLToolParser:
    """Tests for QwenXMLToolParser (Qwen3-Coder format)."""

    @pytest.fixture
    def parser(self):
        """Create a default parser."""
        return QwenXMLToolParser()

    # --- Basic Parsing ---

    def test_parse_single_tool_call(self, parser):
        """Parse a single valid tool call."""
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

    def test_parse_multiple_tool_calls(self, parser):
        """Parse multiple tool calls in one text."""
        text = """
<tool_call>
<function=tool_a>
<parameter=a>1</parameter>
</function>
</tool_call>
Some text in between
<tool_call>
<function=tool_b>
<parameter=b>2</parameter>
</function>
</tool_call>
"""
        results = parser.parse(text)

        assert len(results) == 2
        assert results[0].name == "tool_a"
        assert results[0].input == {"a": "1"}
        assert results[1].name == "tool_b"
        assert results[1].input == {"b": "2"}

    def test_parse_no_tool_calls(self, parser):
        """Return empty list when no tool calls present."""
        text = "Just some regular text without any tool calls."
        results = parser.parse(text)

        assert len(results) == 0

    def test_parse_empty_string(self, parser):
        """Handle empty string."""
        results = parser.parse("")
        assert len(results) == 0

    # --- Parameter Handling ---

    def test_parse_no_parameters(self, parser):
        """Tool call with no parameters."""
        text = """<tool_call>
<function=no_args_tool>
</function>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "no_args_tool"
        assert results[0].input == {}
        assert results[0].is_error is False

    def test_parse_multiline_parameter_value(self, parser):
        """Handle multiline parameter values."""
        text = """<tool_call>
<function=write_file>
<parameter=path>/tmp/test.py</parameter>
<parameter=content>
def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
</parameter>
</function>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "write_file"
        assert results[0].input["path"] == "/tmp/test.py"
        assert "def hello():" in results[0].input["content"]
        assert 'print("Hello, World!")' in results[0].input["content"]

    def test_parse_parameter_with_special_characters(self, parser):
        """Handle special characters in parameter values."""
        text = """<tool_call>
<function=search>
<parameter=query>hello & goodbye | "quoted"</parameter>
</function>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].input["query"] == 'hello & goodbye | "quoted"'

    # --- Error Cases ---

    def test_parse_missing_function_tag(self, parser):
        """Missing function tag creates error result."""
        text = """<tool_call>
<parameter=x>1</parameter>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].is_error is True
        assert results[0].name == UNKNOWN_TOOL_NAME

    def test_parse_empty_function_name(self, parser):
        """Empty function name creates error result."""
        text = """<tool_call>
<function=>
<parameter=x>1</parameter>
</function>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].is_error is True

    # --- Whitespace Handling ---

    def test_parse_with_extra_whitespace(self, parser):
        """Handle extra whitespace around tags."""
        text = """<tool_call>
    <function=spacy_tool>
        <parameter=key>   value   </parameter>
    </function>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "spacy_tool"
        assert results[0].input["key"] == "value"
        assert results[0].is_error is False

    def test_parse_compact_format(self, parser):
        """Handle compact single-line format."""
        text = "<tool_call><function=tool><parameter=x>1</parameter></function></tool_call>"
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "tool"
        assert results[0].input == {"x": "1"}

    # --- Custom Tokens ---

    def test_custom_tokens(self):
        """Use custom tool_call_tokens."""
        parser = QwenXMLToolParser(tool_call_tokens=("<call>", "</call>"))
        text = """<call>
<function=custom>
<parameter=x>1</parameter>
</function>
</call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "custom"

    def test_custom_tokens_ignore_default(self):
        """Custom tokens ignore default format."""
        parser = QwenXMLToolParser(tool_call_tokens=("<call>", "</call>"))
        text = """<tool_call>
<function=ignored>
<parameter=x>1</parameter>
</function>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 0

    # --- Callable Interface ---

    def test_callable_returns_successful_only(self, parser):
        """__call__ returns only successful parses as dicts."""
        text = """
<tool_call>
<function=good>
<parameter=x>1</parameter>
</function>
</tool_call>
<tool_call>
no function tag here
</tool_call>
<tool_call>
<function=also_good>
<parameter=y>2</parameter>
</function>
</tool_call>
"""
        results = parser(text)  # Using __call__

        assert len(results) == 2
        assert results[0]["name"] == "good"
        assert results[0]["input"] == {"x": "1"}
        assert results[1]["name"] == "also_good"

    def test_callable_returns_dict_format(self, parser):
        """__call__ returns dicts with id, name, input keys."""
        text = """<tool_call>
<function=tool>
<parameter=a>1</parameter>
</function>
</tool_call>"""
        results = parser(text)

        assert len(results) == 1
        assert set(results[0].keys()) == {"id", "name", "input"}
        assert results[0]["name"] == "tool"
        assert results[0]["input"] == {"a": "1"}
        assert results[0]["id"].startswith("call_")

    # --- Think Block Exclusion ---

    def test_exclude_tool_calls_inside_think_block(self, parser):
        """Tool calls inside <think> blocks are excluded."""
        text = """
<think>
Let me think about this...
<tool_call>
<function=draft_tool>
<parameter=x>1</parameter>
</function>
</tool_call>
No, let me reconsider...
</think>
<tool_call>
<function=actual_tool>
<parameter=y>2</parameter>
</function>
</tool_call>
"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "actual_tool"
        assert results[0].input == {"y": "2"}

    def test_disable_think_block_exclusion(self):
        """Setting think_tokens=None disables exclusion."""
        parser = QwenXMLToolParser(think_tokens=None)
        text = """
<think>
<tool_call>
<function=inside_think>
</function>
</tool_call>
</think>
<tool_call>
<function=outside_think>
</function>
</tool_call>
"""
        results = parser.parse(text)

        # Both should be parsed when exclusion is disabled
        assert len(results) == 2
        assert results[0].name == "inside_think"
        assert results[1].name == "outside_think"

    # --- Edge Cases ---

    def test_tool_call_with_special_characters_in_name(self, parser):
        """Handle special characters in function name."""
        text = """<tool_call>
<function=my-tool_v2.0>
<parameter=x>1</parameter>
</function>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "my-tool_v2.0"

    def test_tool_call_with_unicode(self, parser):
        """Handle unicode in parameter values."""
        text = """<tool_call>
<function=unicode>
<parameter=emoji>🚀</parameter>
<parameter=chinese>你好</parameter>
</function>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].input["emoji"] == "🚀"
        assert results[0].input["chinese"] == "你好"

    def test_unclosed_tool_call_tag(self, parser):
        """Unclosed tag is not parsed."""
        text = """<tool_call>
<function=unclosed>
</function>"""
        results = parser.parse(text)

        assert len(results) == 0

    def test_unique_ids_generated(self, parser):
        """Each tool call gets a unique ID."""
        text = """
<tool_call><function=a></function></tool_call>
<tool_call><function=b></function></tool_call>
"""
        results = parser.parse(text)

        assert len(results) == 2
        assert results[0].id != results[1].id
        assert results[0].id.startswith("call_")
        assert results[1].id.startswith("call_")

    def test_sequential_ids_are_sortable(self, parser):
        """Tool call IDs are sequential and sortable for result ordering."""
        text = """
<tool_call><function=first></function></tool_call>
<tool_call><function=second></function></tool_call>
<tool_call><function=third></function></tool_call>
"""
        results = parser.parse(text)

        assert len(results) == 3
        assert results[0].id == "call_0000"
        assert results[1].id == "call_0001"
        assert results[2].id == "call_0002"
        assert results[0].id < results[1].id < results[2].id

    def test_message_separator(self, parser):
        """Message separator is newline for Qwen models."""
        assert parser.message_separator == "\n"

    # --- Real-world Example from Qwen3-Coder ---

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

        assert len(results) == 1
        assert results[0].name == "run_terminal_command"
        assert results[0].input["command"] == "git status"
        assert results[0].input["waitForCompletion"] == "True"


class TestGLMToolCallParser:
    """Tests for GLMToolCallParser (GLM-4 and ChatGLM format)."""

    @pytest.fixture
    def parser(self):
        """Create a default parser."""
        return GLMToolParser()

    # --- Basic Parsing ---

    def test_parse_single_tool_call(self, parser):
        """Parse a single valid tool call."""
        text = """<tool_call>calculator
<arg_key>x</arg_key>
<arg_value>1</arg_value>
<arg_key>y</arg_key>
<arg_value>2</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "calculator"
        assert results[0].input == {"x": 1, "y": 2}  # JSON-decoded as integers
        assert results[0].is_error is False

    def test_parse_json_encoded_values(self, parser):
        """Parse JSON-encoded values for non-string types."""
        text = """<tool_call>calculator
<arg_key>x</arg_key>
<arg_value>10</arg_value>
<arg_key>y</arg_key>
<arg_value>20</arg_value>
<arg_key>operation</arg_key>
<arg_value>"multiply"</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "calculator"
        assert results[0].input["x"] == 10  # Decoded as integer
        assert results[0].input["y"] == 20  # Decoded as integer
        assert results[0].input["operation"] == "multiply"  # Decoded as string

    def test_parse_multiple_tool_calls(self, parser):
        """Parse multiple tool calls in one text."""
        text = """
<tool_call>tool_a
<arg_key>a</arg_key>
<arg_value>1</arg_value>
</tool_call>
Some text in between
<tool_call>tool_b
<arg_key>b</arg_key>
<arg_value>2</arg_value>
</tool_call>
"""
        results = parser.parse(text)

        assert len(results) == 2
        assert results[0].name == "tool_a"
        assert results[0].input == {"a": 1}  # JSON-decoded as integer
        assert results[1].name == "tool_b"
        assert results[1].input == {"b": 2}  # JSON-decoded as integer

    def test_parse_no_tool_calls(self, parser):
        """Return empty list when no tool calls present."""
        text = "Just some regular text without any tool calls."
        results = parser.parse(text)

        assert len(results) == 0

    def test_parse_empty_string(self, parser):
        """Handle empty string."""
        results = parser.parse("")
        assert len(results) == 0

    # --- Argument Handling ---

    def test_parse_no_arguments(self, parser):
        """Tool call with no arguments."""
        text = """<tool_call>no_args_tool
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "no_args_tool"
        assert results[0].input == {}
        assert results[0].is_error is False

    def test_parse_complex_json_value(self, parser):
        """Parse complex JSON-encoded values."""
        text = """<tool_call>complex_tool
<arg_key>data</arg_key>
<arg_value>{"nested": {"a": 1, "b": [1, 2, 3]}}</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].input["data"] == {"nested": {"a": 1, "b": [1, 2, 3]}}

    def test_parse_plain_string_value(self, parser):
        """Parse plain string values (not JSON-encoded)."""
        text = """<tool_call>search
<arg_key>query</arg_key>
<arg_value>hello world</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].input["query"] == "hello world"

    def test_parse_multiline_value(self, parser):
        """Handle multiline argument values."""
        text = """<tool_call>write_file
<arg_key>path</arg_key>
<arg_value>/tmp/test.py</arg_value>
<arg_key>content</arg_key>
<arg_value>def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "write_file"
        assert results[0].input["path"] == "/tmp/test.py"
        assert "def hello():" in results[0].input["content"]
        assert 'print("Hello, World!")' in results[0].input["content"]

    # --- Error Cases ---

    def test_parse_missing_function_name(self, parser):
        """Missing function name creates error result."""
        text = """<tool_call>
<arg_key>x</arg_key>
<arg_value>1</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].is_error is True
        assert results[0].name == UNKNOWN_TOOL_NAME

    def test_parse_whitespace_only_function_name(self, parser):
        """Whitespace-only function name creates error result."""
        text = """<tool_call>
<arg_key>x</arg_key>
<arg_value>1</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].is_error is True

    # --- Whitespace Handling ---

    def test_parse_with_extra_whitespace(self, parser):
        """Handle extra whitespace around tags."""
        text = """<tool_call>   spacy_tool
<arg_key>   key   </arg_key>
<arg_value>   value   </arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "spacy_tool"
        assert results[0].input["key"] == "value"
        assert results[0].is_error is False

    def test_parse_compact_format(self, parser):
        """Handle compact single-line format."""
        text = "<tool_call>tool\n<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "tool"
        assert results[0].input == {"x": 1}  # JSON-decoded as integer

    # --- Custom Tokens ---

    def test_custom_tokens(self):
        """Use custom tool_call_tokens."""
        parser = GLMToolParser(tool_call_tokens=("<call>", "</call>"))
        text = """<call>custom
<arg_key>x</arg_key>
<arg_value>1</arg_value>
</call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "custom"
        assert results[0].input == {"x": 1}

    def test_custom_tokens_ignore_default(self):
        """Custom tokens ignore default format."""
        parser = GLMToolParser(tool_call_tokens=("<call>", "</call>"))
        text = """<tool_call>ignored
<arg_key>x</arg_key>
<arg_value>1</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 0

    # --- Callable Interface ---

    def test_callable_returns_successful_only(self, parser):
        """__call__ returns only successful parses as dicts."""
        text = """
<tool_call>good
<arg_key>x</arg_key>
<arg_value>1</arg_value>
</tool_call>
<tool_call>
<arg_key>x</arg_key>
<arg_value>1</arg_value>
</tool_call>
<tool_call>also_good
<arg_key>y</arg_key>
<arg_value>2</arg_value>
</tool_call>
"""
        results = parser(text)  # Using __call__

        assert len(results) == 2
        assert results[0]["name"] == "good"
        assert results[0]["input"] == {"x": 1}  # JSON-decoded as integer
        assert results[1]["name"] == "also_good"
        assert results[1]["input"] == {"y": 2}  # JSON-decoded as integer

    def test_callable_returns_dict_format(self, parser):
        """__call__ returns dicts with id, name, input keys."""
        text = """<tool_call>tool
<arg_key>a</arg_key>
<arg_value>1</arg_value>
</tool_call>"""
        results = parser(text)

        assert len(results) == 1
        assert set(results[0].keys()) == {"id", "name", "input"}
        assert results[0]["name"] == "tool"
        assert results[0]["input"] == {"a": 1}  # JSON-decoded as integer
        assert results[0]["id"].startswith("call_")

    # --- Think Block Exclusion ---

    def test_exclude_tool_calls_inside_think_block(self, parser):
        """Tool calls inside <think> blocks are excluded."""
        text = """
<think>
Let me think about this...
<tool_call>draft_tool
<arg_key>x</arg_key>
<arg_value>1</arg_value>
</tool_call>
No, let me reconsider...
</think>
<tool_call>actual_tool
<arg_key>y</arg_key>
<arg_value>2</arg_value>
</tool_call>
"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "actual_tool"
        assert results[0].input == {"y": 2}  # JSON-decoded as integer

    def test_disable_think_block_exclusion(self):
        """Setting think_tokens=None disables exclusion."""
        parser = GLMToolParser(think_tokens=None)
        text = """
<think>
<tool_call>inside_think
</tool_call>
</think>
<tool_call>outside_think
</tool_call>
"""
        results = parser.parse(text)

        # Both should be parsed when exclusion is disabled
        assert len(results) == 2
        assert results[0].name == "inside_think"
        assert results[1].name == "outside_think"

    # --- Edge Cases ---

    def test_tool_call_with_special_characters_in_name(self, parser):
        """Handle special characters in function name."""
        text = """<tool_call>my-tool_v2.0
<arg_key>x</arg_key>
<arg_value>1</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "my-tool_v2.0"
        assert results[0].input == {"x": 1}

    def test_tool_call_with_unicode(self, parser):
        """Handle unicode in argument values."""
        text = """<tool_call>unicode
<arg_key>emoji</arg_key>
<arg_value>🚀</arg_value>
<arg_key>chinese</arg_key>
<arg_value>你好</arg_value>
</tool_call>"""
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].input["emoji"] == "🚀"
        assert results[0].input["chinese"] == "你好"

    def test_unclosed_tool_call_tag(self, parser):
        """Unclosed tag is not parsed."""
        text = """<tool_call>unclosed
<arg_key>x</arg_key>
<arg_value>1</arg_value>"""
        results = parser.parse(text)

        assert len(results) == 0

    def test_unique_ids_generated(self, parser):
        """Each tool call gets a unique ID."""
        text = """
<tool_call>a
</tool_call>
<tool_call>b
</tool_call>
"""
        results = parser.parse(text)

        assert len(results) == 2
        assert results[0].id != results[1].id
        assert results[0].id.startswith("call_")
        assert results[1].id.startswith("call_")

    def test_sequential_ids_are_sortable(self, parser):
        """Tool call IDs are sequential and sortable for result ordering."""
        text = """
<tool_call>first
</tool_call>
<tool_call>second
</tool_call>
<tool_call>third
</tool_call>
"""
        results = parser.parse(text)

        assert len(results) == 3
        assert results[0].id == "call_0000"
        assert results[1].id == "call_0001"
        assert results[2].id == "call_0002"
        assert results[0].id < results[1].id < results[2].id

    def test_message_separator(self, parser):
        """Message separator is empty for GLM models."""
        assert parser.message_separator == ""

    # --- Real-world Example ---

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

        assert len(results) == 1
        assert results[0].name == "web_search"
        assert results[0].input["query"] == "Python asyncio tutorial"
        assert results[0].input["limit"] == 5  # JSON-decoded as integer


class TestToolParserRegistry:
    """Tests for tool parser registry."""

    def test_get_hermes_parser(self):
        """Get hermes parser by name."""
        from strands_sglang.tool_parsers import get_tool_parser

        parser = get_tool_parser("hermes")
        assert isinstance(parser, HermesToolParser)

    def test_get_parser_with_kwargs(self):
        """Get parser with custom arguments."""
        from strands_sglang.tool_parsers import get_tool_parser

        parser = get_tool_parser("hermes", think_tokens=None)
        assert parser.think_tokens is None

    def test_unknown_parser_raises(self):
        """Unknown parser name raises KeyError."""
        from strands_sglang.tool_parsers import get_tool_parser

        with pytest.raises(KeyError, match="Unknown tool parser"):
            get_tool_parser("nonexistent")

    def test_registry_contains_hermes(self):
        """Registry contains hermes parser."""
        from strands_sglang.tool_parsers import TOOL_PARSER_REGISTRY

        assert "hermes" in TOOL_PARSER_REGISTRY
        assert TOOL_PARSER_REGISTRY["hermes"] is HermesToolParser

    def test_get_qwen_xml_parser(self):
        """Get qwen_xml parser by name."""
        from strands_sglang.tool_parsers import get_tool_parser

        parser = get_tool_parser("qwen_xml")
        assert isinstance(parser, QwenXMLToolParser)

    def test_registry_contains_qwen_xml(self):
        """Registry contains qwen_xml parser."""
        from strands_sglang.tool_parsers import TOOL_PARSER_REGISTRY

        assert "qwen_xml" in TOOL_PARSER_REGISTRY
        assert TOOL_PARSER_REGISTRY["qwen_xml"] is QwenXMLToolParser

    def test_get_glm_parser(self):
        """Get glm parser by name."""
        from strands_sglang.tool_parsers import get_tool_parser

        parser = get_tool_parser("glm")
        assert isinstance(parser, GLMToolParser)

    def test_registry_contains_glm(self):
        """Registry contains glm parser."""
        from strands_sglang.tool_parsers import TOOL_PARSER_REGISTRY

        assert "glm" in TOOL_PARSER_REGISTRY
        assert TOOL_PARSER_REGISTRY["glm"] is GLMToolParser
