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

from strands_sglang import (
    UNKNOWN_TOOL_NAME,
    HermesToolCallParser,
    ToolCallParseResult,
)


class TestToolCallParseResult:
    """Tests for ToolCallParseResult dataclass."""

    def test_successful_parse_result(self):
        """Successful parse has raw=None."""
        result = ToolCallParseResult(
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
        result = ToolCallParseResult(
            id="call_456",
            name="unknown_tool",
            input={},
            raw='{"malformed": json}',
        )
        assert result.is_error is True
        assert result.raw == '{"malformed": json}'

    def test_payload_success(self):
        """payload returns JSON-encoded input for successful parses."""
        result = ToolCallParseResult(
            id="call_123",
            name="my_tool",
            input={"arg": "value", "num": 42},
        )
        assert result.payload == '{"arg": "value", "num": 42}'

    def test_payload_empty(self):
        """payload returns empty JSON object for empty input."""
        result = ToolCallParseResult(id="call_123", name="my_tool", input={})
        assert result.payload == "{}"

    def test_payload_error(self):
        """payload returns raw content for error parses."""
        result = ToolCallParseResult(
            id="call_456",
            name="unknown_tool",
            input={},
            raw='{"malformed": json}',
        )
        assert result.payload == '{"malformed": json}'

    def test_payload_error_empty_raw(self):
        """payload returns empty string if raw is empty."""
        result = ToolCallParseResult(
            id="call_789",
            name="some_tool",
            input={},
            raw="",
        )
        # Note: empty raw still counts as error (raw is not None)
        assert result.is_error is True
        assert result.payload == ""

    def test_immutability(self):
        """ToolCallParseResult is frozen."""
        result = ToolCallParseResult(id="call_123", name="tool", input={})
        with pytest.raises(AttributeError):
            result.name = "other_tool"


class TestHermesToolCallParser:
    """Tests for HermesToolCallParser."""

    @pytest.fixture
    def parser(self):
        """Create a default parser."""
        return HermesToolCallParser()

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
        text = '''<tool_call>{"name": "complex", "arguments": {
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [{"x": 1}, {"y": 2}],
            "string": "hello"
        }}</tool_call>'''
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
        """Use custom bot/eot tokens."""
        parser = HermesToolCallParser(
            bot_token="<function>",
            eot_token="</function>",
        )
        text = '<function>{"name": "custom", "arguments": {}}</function>'
        results = parser.parse(text)

        assert len(results) == 1
        assert results[0].name == "custom"

    def test_custom_tokens_ignore_default(self):
        """Custom tokens ignore default format."""
        parser = HermesToolCallParser(
            bot_token="<function>",
            eot_token="</function>",
        )
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
