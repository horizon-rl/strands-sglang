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

"""Unit tests for SGLangModel helper methods (no API calls needed)."""

from unittest.mock import MagicMock

import pytest

from strands_sglang import SGLangModel
from strands_sglang.tool_parser import ToolCallParseResult


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    return tokenizer


@pytest.fixture
def model(mock_tokenizer):
    """Create an SGLangModel with mock tokenizer."""
    return SGLangModel(tokenizer=mock_tokenizer)


class TestFormatTools:
    """Tests for _format_tools method."""

    def test_format_single_tool(self, model):
        """Format a single tool spec."""
        tool_specs = [
            {
                "name": "calculator",
                "description": "Perform calculations",
                "inputSchema": {"type": "object", "properties": {"expr": {"type": "string"}}},
            }
        ]
        result = model._format_tools(tool_specs)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "calculator"
        assert result[0]["function"]["description"] == "Perform calculations"
        assert "properties" in result[0]["function"]["parameters"]

    def test_format_multiple_tools(self, model):
        """Format multiple tool specs."""
        tool_specs = [
            {"name": "tool1", "description": "First tool", "inputSchema": {}},
            {"name": "tool2", "description": "Second tool", "inputSchema": {}},
            {"name": "tool3", "description": "Third tool", "inputSchema": {}},
        ]
        result = model._format_tools(tool_specs)

        assert len(result) == 3
        assert [t["function"]["name"] for t in result] == ["tool1", "tool2", "tool3"]

    def test_format_tool_missing_fields(self, model):
        """Format tool spec with missing optional fields."""
        tool_specs = [{"name": "minimal"}]
        result = model._format_tools(tool_specs)

        assert len(result) == 1
        assert result[0]["function"]["name"] == "minimal"
        assert result[0]["function"]["description"] == ""
        assert result[0]["function"]["parameters"] == {}

    def test_format_empty_tools(self, model):
        """Format empty tool specs list."""
        result = model._format_tools([])
        assert result == []


class TestFormatPrompt:
    """Tests for format_prompt method."""

    def test_format_simple_prompt(self, model, mock_tokenizer):
        """Format simple user message."""
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        result = model.format_prompt(messages)

        mock_tokenizer.apply_chat_template.assert_called_once()
        assert result == "formatted prompt"

    def test_format_prompt_with_system(self, model, mock_tokenizer):
        """Format prompt with system message."""
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        model.format_prompt(messages, system_prompt="You are helpful.")

        call_args = mock_tokenizer.apply_chat_template.call_args
        chat_messages = call_args[0][0]
        assert chat_messages[0]["role"] == "system"
        assert chat_messages[0]["content"] == "You are helpful."

    def test_format_prompt_with_tools(self, model, mock_tokenizer):
        """Format prompt with tools."""
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        model.format_prompt(messages, tools=tools)

        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["tools"] == tools
        assert call_kwargs["add_generation_prompt"] is True
        assert call_kwargs["tokenize"] is False


class TestBuildSglangPayload:
    """Tests for build_sglang_payload method."""

    def test_minimal_payload(self, model):
        """Build payload with minimal parameters."""
        payload = model.build_sglang_payload(input_ids=[1, 2, 3])

        assert payload["input_ids"] == [1, 2, 3]
        assert payload["stream"] is True
        assert payload["return_logprob"] is True
        assert payload["logprob_start_len"] == 0

    def test_payload_with_sampling_params(self, model):
        """Build payload with sampling parameters."""
        sampling = {"temperature": 0.7, "max_tokens": 100}
        payload = model.build_sglang_payload(input_ids=[1, 2, 3], sampling_params=sampling)

        assert payload["sampling_params"] == sampling

    def test_payload_without_logprobs(self, model):
        """Build payload without logprobs."""
        payload = model.build_sglang_payload(input_ids=[1, 2, 3], return_logprob=False)

        assert "return_logprob" not in payload
        assert "logprob_start_len" not in payload

    def test_payload_without_streaming(self, model):
        """Build payload without streaming."""
        payload = model.build_sglang_payload(input_ids=[1, 2, 3], stream=False)

        assert payload["stream"] is False

    def test_payload_with_model_id(self, mock_tokenizer):
        """Build payload with model ID from config."""
        model = SGLangModel(tokenizer=mock_tokenizer, model_id="qwen/qwen3-4b")
        payload = model.build_sglang_payload(input_ids=[1, 2, 3])

        assert payload["model"] == "qwen/qwen3-4b"


class TestTokenizePromptMessages:
    """Tests for tokenize_prompt_messages method."""

    def test_first_call_tokenizes_full_prompt(self, model, mock_tokenizer):
        """First call tokenizes full prompt with system and tools."""
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        model._current_tools = [{"type": "function", "function": {"name": "test"}}]

        result = model.tokenize_prompt_messages(messages, system_prompt="Be helpful.")

        assert result == [1, 2, 3, 4, 5]
        mock_tokenizer.encode.assert_called_once()

    def test_subsequent_call_tokenizes_new_messages(self, model, mock_tokenizer):
        """Subsequent calls tokenize only new messages."""
        # Simulate first call already processed
        model.token_manager.add_prompt([1, 2, 3])
        model._processed_message_count = 1

        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
            {"role": "user", "content": [{"text": "New message"}]},
        ]

        result = model.tokenize_prompt_messages(messages, system_prompt=None)

        assert result is not None
        # Should only process messages after _processed_message_count
        mock_tokenizer.encode.assert_called()

    def test_no_new_messages_returns_none(self, model, mock_tokenizer):
        """No new messages returns None."""
        model.token_manager.add_prompt([1, 2, 3])
        model._processed_message_count = 2

        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
        ]

        result = model.tokenize_prompt_messages(messages, system_prompt=None)

        assert result is None


class TestExtractLogprobs:
    """Tests for _extract_logprobs method."""

    def test_extract_from_meta_info(self, model):
        """Extract logprobs from meta_info."""
        event = {
            "meta_info": {
                "output_token_logprobs": [[-0.5, 100], [-0.3, 200], [-0.1, 300]]
            }
        }
        result = model._extract_logprobs(event, "output_token_logprobs")

        assert result == [-0.5, -0.3, -0.1]

    def test_extract_from_top_level(self, model):
        """Extract logprobs from top-level event."""
        event = {"input_token_logprobs": [[-1.0, 1], [-2.0, 2]]}
        result = model._extract_logprobs(event, "input_token_logprobs")

        assert result == [-1.0, -2.0]

    def test_extract_missing_key(self, model):
        """Missing key returns None."""
        event = {"other": "data"}
        result = model._extract_logprobs(event, "output_token_logprobs")

        assert result is None

    def test_extract_empty_list(self, model):
        """Empty logprobs list returns None."""
        event = {"output_token_logprobs": []}
        result = model._extract_logprobs(event, "output_token_logprobs")

        assert result is None

    def test_extract_none_value(self, model):
        """None value returns None."""
        event = {"output_token_logprobs": None}
        result = model._extract_logprobs(event, "output_token_logprobs")

        assert result is None


class TestYieldToolUseEvents:
    """Tests for _yield_tool_use_events method."""

    def test_single_tool_call(self, model):
        """Yield events for single tool call."""
        tool_calls = [
            ToolCallParseResult(id="call_123", name="calculator", input={"expr": "2+2"})
        ]
        events = list(model._yield_tool_use_events(tool_calls))

        assert len(events) == 3
        # contentBlockStart
        assert "contentBlockStart" in events[0]
        assert events[0]["contentBlockStart"]["start"]["toolUse"]["name"] == "calculator"
        assert events[0]["contentBlockStart"]["start"]["toolUse"]["toolUseId"] == "call_123"
        # contentBlockDelta
        assert "contentBlockDelta" in events[1]
        assert '"expr": "2+2"' in events[1]["contentBlockDelta"]["delta"]["toolUse"]["input"]
        # contentBlockStop
        assert events[2] == {"contentBlockStop": {}}

    def test_multiple_tool_calls(self, model):
        """Yield events for multiple tool calls."""
        tool_calls = [
            ToolCallParseResult(id="call_1", name="tool1", input={}),
            ToolCallParseResult(id="call_2", name="tool2", input={}),
        ]
        events = list(model._yield_tool_use_events(tool_calls))

        # 3 events per tool call
        assert len(events) == 6
        assert events[0]["contentBlockStart"]["start"]["toolUse"]["name"] == "tool1"
        assert events[3]["contentBlockStart"]["start"]["toolUse"]["name"] == "tool2"

    def test_empty_tool_calls(self, model):
        """No tool calls yields no events."""
        events = list(model._yield_tool_use_events([]))
        assert events == []

    def test_error_tool_call(self, model):
        """Error tool call includes raw content."""
        tool_calls = [
            ToolCallParseResult(id="call_err", name="broken", input={}, raw="invalid json")
        ]
        events = list(model._yield_tool_use_events(tool_calls))

        assert len(events) == 3
        # Error tool call uses raw content as payload
        assert events[1]["contentBlockDelta"]["delta"]["toolUse"]["input"] == "invalid json"


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_token_manager(self, model):
        """Reset clears token manager."""
        model.token_manager.add_prompt([1, 2, 3])
        model.token_manager.add_response([4, 5, 6])

        model.reset()

        assert len(model.token_manager) == 0

    def test_reset_clears_message_count(self, model):
        """Reset clears processed message count."""
        model._processed_message_count = 5

        model.reset()

        assert model._processed_message_count == 0

    def test_reset_clears_tools(self, model):
        """Reset clears current tools."""
        model._current_tools = [{"type": "function"}]

        model.reset()

        assert model._current_tools is None


class TestConfig:
    """Tests for configuration methods."""

    def test_default_config(self, mock_tokenizer):
        """Default configuration values."""
        model = SGLangModel(tokenizer=mock_tokenizer)
        config = model.get_config()

        assert config["base_url"] == "http://localhost:8000"

    def test_custom_base_url(self, mock_tokenizer):
        """Custom base URL is stored correctly."""
        model = SGLangModel(tokenizer=mock_tokenizer, base_url="http://custom:9000/")
        config = model.get_config()

        assert config["base_url"] == "http://custom:9000"

    def test_update_config(self, model):
        """Update configuration."""
        model.update_config(base_url="http://new:8080", model_id="new-model")
        config = model.get_config()

        assert config["base_url"] == "http://new:8080"
        assert config["model_id"] == "new-model"

    def test_config_with_timeout_float(self, mock_tokenizer):
        """Configuration with timeout float (connect is always 5.0 like OpenAI)."""
        model = SGLangModel(tokenizer=mock_tokenizer, timeout=300.0)
        timeout = model._client_config["timeout"]
        assert timeout.connect == 5.0  # Fixed like OpenAI
        assert timeout.read == 300.0

    def test_config_with_default_timeout(self, mock_tokenizer):
        """Configuration with default timeout (600s like OpenAI)."""
        model = SGLangModel(tokenizer=mock_tokenizer)
        timeout = model._client_config["timeout"]
        assert timeout.connect == 5.0
        assert timeout.read == 600.0  # Default 10min like OpenAI
