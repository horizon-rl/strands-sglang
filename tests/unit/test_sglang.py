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

import base64
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands_sglang import SGLangModel
from strands_sglang.client import SGLangClient
from strands_sglang.tool_parsers import ToolParseResult


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
    client = SGLangClient(base_url="http://localhost:30000")
    return SGLangModel(client=client, tokenizer=mock_tokenizer)


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

        call_kwargs = mock_tokenizer.apply_chat_template.call_args.kwargs
        chat_messages = call_kwargs["conversation"]
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
        event = {"meta_info": {"output_token_logprobs": [[-0.5, 100], [-0.3, 200], [-0.1, 300]]}}
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
        tool_calls = [ToolParseResult(id="call_123", name="calculator", input={"expr": "2+2"})]
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
            ToolParseResult(id="call_1", name="tool1", input={}),
            ToolParseResult(id="call_2", name="tool2", input={}),
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
        tool_calls = [ToolParseResult(id="call_err", name="broken", input={}, raw="invalid json")]
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
    """Tests for _sort_tool_results method."""

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

        sorted_msgs = model._sort_tool_results(messages)

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

        sorted_msgs = model._sort_tool_results(messages)

        assert sorted_msgs == messages

    def test_preserves_other_content_blocks(self, model):
        """Non-toolResult blocks are preserved (moved to front)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call_0001", "content": [{"text": "b"}]}},
                    {"text": "some context"},
                    {"toolResult": {"toolUseId": "call_0000", "content": [{"text": "a"}]}},
                ],
            },
        ]

        sorted_msgs = model._sort_tool_results(messages)

        content = sorted_msgs[0]["content"]
        assert content[0] == {"text": "some context"}  # Other blocks first
        assert content[1]["toolResult"]["toolUseId"] == "call_0000"
        assert content[2]["toolResult"]["toolUseId"] == "call_0001"

    def test_empty_messages(self, model):
        """Empty messages list returns empty."""
        assert model._sort_tool_results([]) == []

    def test_no_tool_results(self, model):
        """Messages without toolResults pass through unchanged."""
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        sorted_msgs = model._sort_tool_results(messages)

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

        sorted_msgs = model._sort_tool_results(messages)

        # Assistant message unchanged
        assert sorted_msgs[0] == messages[0]
        # User tool results sorted
        assert sorted_msgs[1]["content"][0]["toolResult"]["toolUseId"] == "call_0000"
        assert sorted_msgs[1]["content"][1]["toolResult"]["toolUseId"] == "call_0001"

    def test_user_message_with_string_content(self, model):
        """User message with string content (not list) passes through unchanged."""
        messages = [{"role": "user", "content": "plain text message"}]

        sorted_msgs = model._sort_tool_results(messages)

        assert sorted_msgs == messages


# ---------------------------------------------------------------------------
# Helpers for routing replay end-to-end tests
# ---------------------------------------------------------------------------

NUM_LAYERS = 2
TOP_K = 2
EXPERTS_PER_TOKEN = NUM_LAYERS * TOP_K  # int32 values per token


def _make_routing_b64(expert_ids: list[int]) -> str:
    """Encode int32 expert IDs to base64 (matching SGLang format)."""
    return base64.b64encode(struct.pack(f"<{len(expert_ids)}i", *expert_ids)).decode("ascii")


def _decode_routing_b64(data: str) -> list[int]:
    """Decode base64 routing data back to int32 expert IDs."""
    raw = base64.b64decode(data)
    return list(struct.unpack(f"<{len(raw) // 4}i", raw))


def _make_generate_response(
    text: str,
    output_ids: list[int],
    num_input_tokens: int,
    *,
    include_routing: bool = False,
) -> dict:
    """Build a mock SGLang /generate response.

    When include_routing is True, generates deterministic per-token routing
    for the FULL sequence (position 0 to total-1), matching SGLang's actual
    behavior. Token at position i gets experts [i*10, i*10+1, i*10+2, i*10+3].
    """
    num_output = len(output_ids)
    total = num_input_tokens + num_output

    meta_info = {
        "finish_reason": {"type": "stop"},
        "prompt_tokens": num_input_tokens,
        "completion_tokens": num_output,
        "e2e_latency": 0.1,
    }

    if include_routing:
        expert_ids = []
        for pos in range(total):
            expert_ids.extend([pos * 10 + k for k in range(EXPERTS_PER_TOKEN)])
        meta_info["routed_experts"] = _make_routing_b64(expert_ids)

    input_logprobs = [[-0.1, tid] for tid in range(num_input_tokens)]
    output_logprobs = [[-0.2, tid] for tid in output_ids]

    return {
        "text": text,
        "output_ids": output_ids,
        "meta_info": meta_info,
        "input_token_logprobs": input_logprobs,
        "output_token_logprobs": output_logprobs,
    }


async def _collect_stream(stream):
    """Collect all events from an async iterable."""
    return [event async for event in stream]


class TestRoutedExpertsE2E:
    """End-to-end tests for routing replay through SGLangModel.stream()."""

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        tokenizer.encode.return_value = [10, 20, 30]
        return tokenizer

    @pytest.fixture
    def model(self, mock_tokenizer):
        client = SGLangClient(base_url="http://localhost:30000")
        return SGLangModel(client=client, tokenizer=mock_tokenizer, return_routed_experts=True)

    async def test_single_turn_routing(self, model):
        """Single turn: routing is sliced into prompt + response segments."""
        prompt_tokens = [10, 20, 30]  # 3 tokens
        output_ids = [40, 50]  # 2 tokens

        response = _make_generate_response(
            text="Hello!",
            output_ids=output_ids,
            num_input_tokens=len(prompt_tokens),
            include_routing=True,
        )

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response):
            await _collect_stream(model.stream([{"role": "user", "content": [{"text": "Hi"}]}]))

        assert model.token_manager.token_ids == prompt_tokens + output_ids

        # Routing covers all 5 tokens, sliced into 2 segments
        routing = model.token_manager.routed_experts
        assert routing is not None
        decoded = _decode_routing_b64(routing)
        assert len(decoded) == 5 * EXPERTS_PER_TOKEN

        for pos in range(5):
            chunk = decoded[pos * EXPERTS_PER_TOKEN : (pos + 1) * EXPERTS_PER_TOKEN]
            assert chunk == [pos * 10 + k for k in range(EXPERTS_PER_TOKEN)]

    async def test_multi_turn_with_tool_call(self, model, mock_tokenizer):
        """Multi-turn: full blob is sliced per-segment across turns.

        Turn 2 returns routing for the FULL sequence; SGLangModel slices out
        only the new tokens (tool result + response) and appends them.
        """
        # --- Turn 1: user prompt -> model generates tool call ---
        prompt_tokens_t1 = [10, 20, 30]
        output_ids_t1 = [40, 50]
        mock_tokenizer.encode.return_value = prompt_tokens_t1

        response_t1 = _make_generate_response(
            text='<tool_call>{"name": "calc", "arguments": {"expr": "1+1"}}</tool_call>',
            output_ids=output_ids_t1,
            num_input_tokens=len(prompt_tokens_t1),
            include_routing=True,
        )

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response_t1):
            messages_t1 = [{"role": "user", "content": [{"text": "What is 1+1?"}]}]
            await _collect_stream(model.stream(messages_t1, tool_specs=[{"name": "calc", "description": "calc"}]))

        total_after_t1 = len(model.token_manager)  # 5

        # --- Turn 2: tool result -> model generates final answer ---
        # SGLang returns routing for ALL 10 tokens (full sequence)
        tool_result_tokens = [60, 70]
        output_ids_t2 = [80, 90, 100]
        mock_tokenizer.encode.return_value = tool_result_tokens

        response_t2 = _make_generate_response(
            text="The answer is 2.",
            output_ids=output_ids_t2,
            num_input_tokens=total_after_t1 + len(tool_result_tokens),
            include_routing=True,  # full sequence routing (positions 0-9)
        )

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response_t2):
            messages_t2 = messages_t1 + [
                {
                    "role": "assistant",
                    "content": [
                        {"text": '<tool_call>{"name": "calc", "arguments": {"expr": "1+1"}}</tool_call>'},
                        {"toolUse": {"toolUseId": "call_0000", "name": "calc", "input": {"expr": "1+1"}}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"toolResult": {"toolUseId": "call_0000", "content": [{"text": "2"}]}},
                    ],
                },
            ]
            await _collect_stream(model.stream(messages_t2, tool_specs=[{"name": "calc", "description": "calc"}]))

        # Full token trajectory: 4 segments
        expected_ids = prompt_tokens_t1 + output_ids_t1 + tool_result_tokens + output_ids_t2
        assert model.token_manager.token_ids == expected_ids
        total_tokens = len(expected_ids)  # 10

        # Routing covers ALL tokens across both turns (sliced per-segment)
        routing = model.token_manager.routed_experts
        assert routing is not None
        decoded = _decode_routing_b64(routing)
        assert len(decoded) == total_tokens * EXPERTS_PER_TOKEN

        # Verify per-token expert IDs are correct and contiguous
        for pos in range(total_tokens):
            chunk = decoded[pos * EXPERTS_PER_TOKEN : (pos + 1) * EXPERTS_PER_TOKEN]
            assert chunk == [pos * 10 + k for k in range(EXPERTS_PER_TOKEN)]

    async def test_routing_aligns_with_loss_mask(self, model, mock_tokenizer):
        """Routing entries align 1:1 with token_ids and loss_mask."""
        mock_tokenizer.encode.return_value = [10, 20, 30]

        response = _make_generate_response(
            text="answer",
            output_ids=[40, 50],
            num_input_tokens=3,
            include_routing=True,
        )

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response):
            await _collect_stream(model.stream([{"role": "user", "content": [{"text": "Hi"}]}]))

        n_tokens = len(model.token_manager.token_ids)
        routing_entries = len(_decode_routing_b64(model.token_manager.routed_experts)) // EXPERTS_PER_TOKEN

        assert routing_entries == n_tokens
        assert len(model.token_manager.loss_mask) == n_tokens
        assert len(model.token_manager.logprobs) == n_tokens

    async def test_routing_disabled_by_default(self, mock_tokenizer):
        """When return_routed_experts is not set, no routing data is recorded."""
        client = SGLangClient(base_url="http://localhost:30000")
        model = SGLangModel(client=client, tokenizer=mock_tokenizer)

        mock_tokenizer.encode.return_value = [10, 20]
        response = _make_generate_response(text="hi", output_ids=[30], num_input_tokens=2)

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response) as mock_gen:
            await _collect_stream(model.stream([{"role": "user", "content": [{"text": "Hi"}]}]))
            assert mock_gen.call_args.kwargs["return_routed_experts"] is False

        assert model.token_manager.routed_experts is None

    async def test_routing_absent_from_response(self, model, mock_tokenizer):
        """If server doesn't return routing data, routed_experts stays None."""
        mock_tokenizer.encode.return_value = [10, 20]
        response = _make_generate_response(text="hi", output_ids=[30], num_input_tokens=2)

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response):
            await _collect_stream(model.stream([{"role": "user", "content": [{"text": "Hi"}]}]))

        assert model.token_manager.routed_experts is None

    async def test_reset_clears_routing(self, model, mock_tokenizer):
        """model.reset() clears accumulated routing data."""
        mock_tokenizer.encode.return_value = [10, 20]
        response = _make_generate_response(text="hi", output_ids=[30], num_input_tokens=2, include_routing=True)

        with patch.object(model.client, "generate", new_callable=AsyncMock, return_value=response):
            await _collect_stream(model.stream([{"role": "user", "content": [{"text": "Hi"}]}]))

        assert model.token_manager.routed_experts is not None
        model.reset()
        assert model.token_manager.routed_experts is None
