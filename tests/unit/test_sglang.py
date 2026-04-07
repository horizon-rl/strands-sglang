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

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pybase64
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
    client._is_multimodal = False
    model = SGLangModel(client=client, tokenizer=mock_tokenizer)
    model.__dict__["message_separator"] = ""  # override cached_property (mock has no real template)
    return model


class TestFormatTools:
    """Tests for format_tool_specs method."""

    def test_format_single_tool(self, model):
        """Format a single tool spec into HF function-calling format."""
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
        """Format multiple tool specs preserving order."""
        tool_specs = [
            {"name": "tool1", "description": "First tool", "inputSchema": {"json": {}}},
            {"name": "tool2", "description": "Second tool", "inputSchema": {"json": {}}},
            {"name": "tool3", "description": "Third tool", "inputSchema": {"json": {}}},
        ]
        result = model.format_tool_specs(tool_specs)

        assert len(result) == 3
        assert [t["function"]["name"] for t in result] == ["tool1", "tool2", "tool3"]

    def test_format_tool_missing_fields_raises(self, model):
        """Missing inputSchema raises KeyError."""
        with pytest.raises(KeyError):
            model.format_tool_specs([{"name": "minimal"}])


class TestFormatMessages:
    """Tests for format_messages — especially parallel tool results."""

    def test_parallel_tool_results_split_into_separate_messages(self):
        """All toolResult blocks in one Strands message must produce separate HF messages."""
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
        """Single toolResult produces one HF tool message with flattened content."""
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
    """Tests for tokenize_prompt_messages error handling."""

    def test_no_new_messages_raises(self, model):
        """Raises RuntimeError when message_count matches input length."""
        model.token_manager.add_prompt([1, 2, 3])
        model.message_count = 2

        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
        ]

        with pytest.raises(RuntimeError, match="No new messages to tokenize"):
            model.tokenize_prompt_messages(messages, system_prompt=None)


class TestSortToolResults:
    """Tests for sort_tool_results method."""

    def test_sort_by_sequential_id(self, model):
        """Tool results are sorted by sequential ID."""
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

        assert model.sort_tool_results(messages) == messages

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


def _make_generate_response(**overrides: object) -> dict:
    """Create a standard mock generate response with optional overrides."""
    base: dict = {
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
    base.update(overrides)
    return base


def _make_model_with_mock_client(mock_tokenizer: MagicMock, generate_return: dict | None = None, **config: object):
    """Create an SGLangModel with a mocked client.generate."""
    client = SGLangClient(base_url="http://localhost:30000")
    client._is_multimodal = False
    client.generate = AsyncMock(return_value=generate_return or _make_generate_response())
    model = SGLangModel(client=client, tokenizer=mock_tokenizer, **config)
    return model, client


class TestStreamDefaults:
    """Tests for stream() default behavior."""

    async def test_skip_special_tokens_defaults_to_false(self, mock_tokenizer):
        """stream() passes skip_special_tokens=False to client.generate by default."""
        model, client = _make_model_with_mock_client(mock_tokenizer)

        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        async for _ in model.stream(messages):
            pass

        call_kwargs = client.generate.call_args
        assert call_kwargs.kwargs["sampling_params"]["skip_special_tokens"] is False


class TestStreamRoutedExperts:
    """Tests for return_routed_experts config in stream()."""

    async def test_passed_to_client(self, mock_tokenizer):
        """stream() passes return_routed_experts to client.generate when configured."""
        response = _make_generate_response()
        response["meta_info"]["routed_experts"] = pybase64.b64encode(np.zeros(1, dtype=np.int32).tobytes()).decode()

        model, client = _make_model_with_mock_client(
            mock_tokenizer, generate_return=response, return_routed_experts=True
        )

        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        async for _ in model.stream(messages):
            pass

        assert client.generate.call_args.kwargs["return_routed_experts"] is True

    async def test_defaults_to_false(self, mock_tokenizer):
        """stream() defaults return_routed_experts to False."""
        model, client = _make_model_with_mock_client(mock_tokenizer)

        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        async for _ in model.stream(messages):
            pass

        assert client.generate.call_args.kwargs["return_routed_experts"] is False

    async def test_stored_as_base64(self, mock_tokenizer):
        """stream() stores routed_experts as raw base64 string from meta_info."""
        experts_array = np.arange(36, dtype=np.int32)
        encoded = pybase64.b64encode(experts_array.tobytes()).decode("ascii")

        response = _make_generate_response()
        response["meta_info"]["routed_experts"] = encoded

        model, _ = _make_model_with_mock_client(mock_tokenizer, generate_return=response, return_routed_experts=True)

        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        async for _ in model.stream(messages):
            pass

        assert model.routed_experts == encoded

    def test_decode_routed_experts_util(self):
        """decode_routed_experts() decodes base64 to shaped numpy array."""
        from strands_sglang import decode_routed_experts

        num_layers, top_k, seq_len = 4, 2, 5
        experts = np.arange((seq_len - 1) * num_layers * top_k, dtype=np.int32)
        encoded = pybase64.b64encode(experts.tobytes()).decode("ascii")

        decoded = decode_routed_experts(encoded, seq_len=seq_len, num_layers=num_layers, top_k=top_k)
        assert decoded.shape == (seq_len - 1, num_layers, top_k)
        np.testing.assert_array_equal(decoded.ravel(), experts)

    async def test_raises_when_not_in_response(self, mock_tokenizer):
        """stream() raises KeyError when return_routed_experts=True but server omits it."""
        model, _ = _make_model_with_mock_client(mock_tokenizer, return_routed_experts=True)

        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        with pytest.raises(KeyError, match="routed_experts"):
            async for _ in model.stream(messages):
                pass
