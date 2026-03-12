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

"""Unit tests for SGLangModel helper methods (no API calls needed)."""

from unittest.mock import MagicMock

import pytest

from strands_sglang import SGLangModel
from strands_sglang.client import SGLangClient


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.name_or_path = "/nonexistent"
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    return tokenizer


@pytest.fixture
def model(mock_tokenizer):
    """Create an SGLangModel with mock tokenizer."""
    client = SGLangClient(base_url="http://localhost:30000")
    model = SGLangModel(client=client, tokenizer=mock_tokenizer)
    model.__dict__["is_multimodal"] = False  # override cached_property (mock has no real config)
    model.__dict__["message_separator"] = ""  # override cached_property (mock has no real template)
    return model


class TestFormatTools:
    """Tests for _format_tools method."""

    def test_format_single_tool(self, model):
        """Format a single tool spec."""
        tool_specs = [
            {
                "name": "calculator",
                "description": "Perform calculations",
                "inputSchema": {"json": {"type": "object", "properties": {"expr": {"type": "string"}}}},
            }
        ]
        result = model.format_tool_specs(tool_specs)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "calculator"
        assert result[0]["function"]["description"] == "Perform calculations"
        assert "properties" in result[0]["function"]["parameters"]

    def test_format_multiple_tools(self, model):
        """Format multiple tool specs."""
        tool_specs = [
            {"name": "tool1", "description": "First tool", "inputSchema": {"json": {}}},
            {"name": "tool2", "description": "Second tool", "inputSchema": {"json": {}}},
            {"name": "tool3", "description": "Third tool", "inputSchema": {"json": {}}},
        ]
        result = model.format_tool_specs(tool_specs)

        assert len(result) == 3
        assert [t["function"]["name"] for t in result] == ["tool1", "tool2", "tool3"]

    def test_format_tool_missing_fields_raises(self, model):
        """Format tool spec with missing required fields raises KeyError."""
        tool_specs = [{"name": "minimal"}]
        with pytest.raises(KeyError):
            model.format_tool_specs(tool_specs)

    def test_format_empty_tools(self, model):
        """Format empty tool specs list."""
        result = model.format_tool_specs([])
        assert result == []


class TestFormatMessages:
    """Tests for format_messages — especially parallel tool results."""

    def test_parallel_tool_results_all_present(self):
        """All toolResult blocks in one message must produce separate HF messages."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call_0", "status": "success", "content": [{"text": "result 0"}]}},
                    {"toolResult": {"toolUseId": "call_1", "status": "success", "content": [{"text": "result 1"}]}},
                    {"toolResult": {"toolUseId": "call_2", "status": "success", "content": [{"text": "result 2"}]}},
                ],
            }
        ]
        result = SGLangModel.format_messages(messages)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert len(tool_msgs) == 3
        assert {m["tool_call_id"] for m in tool_msgs} == {"call_0", "call_1", "call_2"}

    def test_single_tool_result(self):
        """Single toolResult still works."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call_0", "status": "success", "content": [{"text": "ok"}]}},
                ],
            }
        ]
        result = SGLangModel.format_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "ok"

    def test_tooluse_skipped(self):
        """toolUse blocks are skipped — tool calls live in raw text."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"text": "<tool_call>...</tool_call>"},
                    {"toolUse": {"toolUseId": "call_0", "name": "fn", "input": {}}},
                ],
            }
        ]
        result = SGLangModel.format_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "<tool_call>...</tool_call>"


class TestTokenizePromptMessages:
    """Tests for tokenize_prompt_messages method."""

    def test_first_call_tokenizes_full_prompt(self, model, mock_tokenizer):
        """First call tokenizes full prompt with system and tools."""
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        tool_specs = [{"name": "test", "description": "A test tool", "inputSchema": {"json": {"type": "object"}}}]

        result = model.tokenize_prompt_messages(messages, system_prompt="Be helpful.", tool_specs=tool_specs)

        assert result == [1, 2, 3, 4, 5]
        mock_tokenizer.encode.assert_called_once()

    def test_subsequent_call_tokenizes_new_messages(self, model, mock_tokenizer):
        """Subsequent calls tokenize only new messages."""
        # Simulate first call already processed
        model.token_manager.add_prompt([1, 2, 3])
        model.message_count = 1

        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
            {"role": "user", "content": [{"text": "New message"}]},
        ]

        result = model.tokenize_prompt_messages(messages, system_prompt=None)

        assert result is not None
        # Should only process messages after message_count
        mock_tokenizer.encode.assert_called()

    def test_no_new_messages_raises(self, model, mock_tokenizer):
        """No new messages raises RuntimeError."""
        model.token_manager.add_prompt([1, 2, 3])
        model.message_count = 2

        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
        ]

        with pytest.raises(RuntimeError, match="No new messages to tokenize"):
            model.tokenize_prompt_messages(messages, system_prompt=None)


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
        model.message_count = 5

        model.reset()

        assert model.message_count == 0

    def test_reset_clears_parse_errors(self, model):
        """Reset clears tool parse error counts."""
        model.tool_parse_errors = {"broken_tool": 3}

        model.reset()

        assert model.tool_parse_errors == {}


class TestConfig:
    """Tests for configuration methods."""

    def test_default_config(self, mock_tokenizer):
        """Default configuration has no base_url or timeout (those belong to SGLangClient)."""
        client = SGLangClient(base_url="http://localhost:30000")
        model = SGLangModel(client=client, tokenizer=mock_tokenizer)
        config = model.get_config()

        assert "base_url" not in config
        assert "timeout" not in config

    def test_update_config(self, model):
        """Update configuration."""
        model.update_config(return_logprob=False)
        config = model.get_config()

        assert config["return_logprob"] is False

    def test_config_with_sampling_params(self, mock_tokenizer):
        """Configuration with custom sampling_params."""
        client = SGLangClient(base_url="http://localhost:30000")
        model = SGLangModel(client=client, tokenizer=mock_tokenizer, sampling_params={"max_new_tokens": 1024})
        config = model.get_config()

        assert config["sampling_params"] == {"max_new_tokens": 1024}


class TestClientSetup:
    """Tests for client setup."""

    def test_client_is_required(self, mock_tokenizer):
        """Client parameter is required."""
        with pytest.raises(TypeError):
            SGLangModel(tokenizer=mock_tokenizer)  # type: ignore[call-arg]

    def test_client_stored_as_public_attr(self, mock_tokenizer):
        """Client is stored as public attribute."""
        client = SGLangClient(base_url="http://localhost:30000")
        model = SGLangModel(client=client, tokenizer=mock_tokenizer)

        assert model.client is client

    def test_all_params_keyword_only(self, mock_tokenizer):
        """All parameters are keyword-only (no positional args)."""
        client = SGLangClient(base_url="http://localhost:30000")
        with pytest.raises(TypeError):
            SGLangModel(mock_tokenizer, client)  # type: ignore[misc]


class TestSortToolResults:
    """Tests for sort_tool_results method."""

    def test_sort_by_sequential_id(self, model):
        """Tool results are sorted by sequential ID (call_0000 < call_0001 < call_0002)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call_0002", "content": [{"text": "third"}]}},
                    {"toolResult": {"toolUseId": "call_0000", "content": [{"text": "first"}]}},
                    {"toolResult": {"toolUseId": "call_0001", "content": [{"text": "second"}]}},
                ],
            },
        ]

        sorted_msgs = model.sort_tool_results(messages)

        results = sorted_msgs[0]["content"]
        assert results[0]["toolResult"]["toolUseId"] == "call_0000"
        assert results[1]["toolResult"]["toolUseId"] == "call_0001"
        assert results[2]["toolResult"]["toolUseId"] == "call_0002"

    def test_preserves_non_tool_messages(self, model):
        """Non-tool messages pass through unchanged."""
        messages = [
            {"role": "assistant", "content": [{"text": "Hello"}]},
            {"role": "user", "content": [{"text": "Hi"}]},
        ]

        sorted_msgs = model.sort_tool_results(messages)

        assert sorted_msgs == messages

    def test_empty_messages(self, model):
        """Empty messages list returns empty."""
        assert model.sort_tool_results([]) == []

    def test_no_tool_results(self, model):
        """Messages without toolResults pass through unchanged."""
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        sorted_msgs = model.sort_tool_results(messages)

        assert sorted_msgs == messages

    def test_mixed_message_types(self, model):
        """Mixed assistant + user messages: only user tool results are sorted."""
        messages = [
            {"role": "assistant", "content": [{"text": "I'll call some tools"}]},
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call_0001", "content": [{"text": "b"}]}},
                    {"toolResult": {"toolUseId": "call_0000", "content": [{"text": "a"}]}},
                ],
            },
        ]

        sorted_msgs = model.sort_tool_results(messages)

        # Assistant message unchanged
        assert sorted_msgs[0] == messages[0]
        # User tool results sorted
        assert sorted_msgs[1]["content"][0]["toolResult"]["toolUseId"] == "call_0000"
        assert sorted_msgs[1]["content"][1]["toolResult"]["toolUseId"] == "call_0001"

    def test_user_message_with_string_content(self, model):
        """User message with string content (not list) passes through unchanged."""
        messages = [{"role": "user", "content": "plain text message"}]

        sorted_msgs = model.sort_tool_results(messages)

        assert sorted_msgs == messages


class TestStreamDefaults:
    """Tests for stream() default behavior."""

    async def test_skip_special_tokens_defaults_to_false(self, mock_tokenizer):
        """stream() passes skip_special_tokens=False to client.generate by default."""
        from unittest.mock import AsyncMock

        client = SGLangClient(base_url="http://localhost:30000")
        client.generate = AsyncMock(
            return_value={
                "text": "hello",
                "output_ids": [1, 2],
                "meta_info": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop"},
                    "e2e_latency": 0.1,
                },
            }
        )
        model = SGLangModel(client=client, tokenizer=mock_tokenizer)
        model.__dict__["is_multimodal"] = False

        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        async for _ in model.stream(messages):
            pass

        call_kwargs = client.generate.call_args
        assert call_kwargs.kwargs["sampling_params"]["skip_special_tokens"] is False
