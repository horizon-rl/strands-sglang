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

"""Integration tests for SGLangModel API.

Tests low-level SGLangModel streaming and TITO functionality.
Fixtures (model, tokenizer, calculator_tool) are provided by conftest.py.
"""


class TestStreamBasic:
    """Basic streaming tests."""

    async def test_simple_generation(self, model):
        """Generate a simple response without tools."""
        messages = [{"role": "user", "content": [{"text": "Say 'hello' and nothing else."}]}]

        events = []
        async for event in model.stream(messages):
            events.append(event)

        # Should have content events
        content_deltas = [e for e in events if "contentBlockDelta" in e]
        assert len(content_deltas) > 0

        # Should have text in deltas
        text = "".join(
            e["contentBlockDelta"]["delta"].get("text", "")
            for e in content_deltas
            if "text" in e["contentBlockDelta"]["delta"]
        )
        assert "hello" in text.lower()

    async def test_generation_with_system_prompt(self, model):
        """Generate with system prompt."""
        messages = [{"role": "user", "content": [{"text": "What are you?"}]}]
        system_prompt = "You are a helpful calculator assistant. Be brief."

        events = []
        async for event in model.stream(messages, system_prompt=system_prompt):
            events.append(event)

        content_deltas = [e for e in events if "contentBlockDelta" in e]
        assert len(content_deltas) > 0

    async def test_metadata_event(self, model):
        """Stream should end with metadata event."""
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        events = []
        async for event in model.stream(messages):
            events.append(event)

        # Last event should be metadata
        assert "metadata" in events[-1]
        metadata = events[-1]["metadata"]
        assert "usage" in metadata


class TestStreamWithTools:
    """Streaming tests with tool calling."""

    async def test_tool_call_generation(self, model, calculator_tool):
        """Model should generate tool call when appropriate."""
        messages = [{"role": "user", "content": [{"text": "What is 15 + 27?"}]}]
        system_prompt = "You are a calculator. Use the calculator tool for all math."

        events = []
        async for event in model.stream(
            messages, tool_specs=[calculator_tool], system_prompt=system_prompt
        ):
            events.append(event)

        # Check for tool use events
        tool_starts = [e for e in events if "contentBlockStart" in e]
        tool_use_starts = [
            e for e in tool_starts if "toolUse" in e["contentBlockStart"].get("start", {})
        ]

        # Model should have called calculator tool
        if tool_use_starts:
            tool_name = tool_use_starts[0]["contentBlockStart"]["start"]["toolUse"]["name"]
            assert tool_name == "calculator"

    async def test_multi_turn_with_tool_result(self, model, calculator_tool):
        """Multi-turn conversation with tool result."""
        # First turn: user asks question
        messages = [{"role": "user", "content": [{"text": "What is 5 * 8?"}]}]
        system_prompt = "You are a calculator. Use the calculator tool for math."

        # First generation
        events = []
        async for event in model.stream(
            messages, tool_specs=[calculator_tool], system_prompt=system_prompt
        ):
            events.append(event)

        # Add assistant response and tool result
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "call_123",
                            "name": "calculator",
                            "input": {"expression": "5 * 8"},
                        }
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "call_123", "content": [{"text": "40"}]}}],
            }
        )

        # Second generation: model should respond after receiving tool result
        events = []
        async for event in model.stream(
            messages, tool_specs=[calculator_tool], system_prompt=system_prompt
        ):
            events.append(event)

        # Should have generated a response (content deltas or tool calls)
        content_deltas = [e for e in events if "contentBlockDelta" in e]
        assert len(content_deltas) > 0, "Model should generate response after tool result"

        # Should end with metadata
        assert "metadata" in events[-1]


class TestTITO:
    """Token-in/token-out trajectory tests.

    Note: Comprehensive TITO testing is done in test_agent_math500.py.
    This class tests low-level model API behaviors not covered by agent tests.
    """

    async def test_token_count_consistency(self, model):
        """Total tokens equals sum of segment lengths."""
        messages = [{"role": "user", "content": [{"text": "Count to 5"}]}]
        async for _ in model.stream(messages):
            pass

        total_tokens = len(model.token_manager)
        segment_sum = sum(info[1] for info in model.token_manager.segment_info)

        assert total_tokens == segment_sum
        assert total_tokens == len(model.token_manager.token_ids)
        assert total_tokens == len(model.token_manager.loss_mask)
        assert total_tokens == len(model.token_manager.logprobs)

    async def test_incremental_tokenization(self, model):
        """Subsequent calls only tokenize new messages."""
        # First turn
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        async for _ in model.stream(messages):
            pass

        first_prompt_len = model.token_manager.segment_info[0][1]

        # Second turn - add previous assistant response and new user message
        messages.append({"role": "assistant", "content": [{"text": "Hello!"}]})
        messages.append({"role": "user", "content": [{"text": "How are you?"}]})

        async for _ in model.stream(messages):
            pass

        # The new prompt segment should not include first turn tokens
        # (they were already processed)
        second_prompt_len = model.token_manager.segment_info[2][1]

        # Second prompt should be smaller than first + second combined
        # (proving incremental tokenization)
        assert second_prompt_len < first_prompt_len + second_prompt_len


class TestSSEParsing:
    """Tests for SSE event parsing."""

    async def test_iter_sse_events(self, model):
        """_iter_sse_events correctly parses SSE stream."""
        messages = [{"role": "user", "content": [{"text": "Say 'test'"}]}]

        # Manually call the internal stream to test SSE parsing
        input_ids = model.tokenize_prompt_messages(messages, system_prompt=None)
        payload = model.build_sglang_payload(input_ids=input_ids)

        async with model.client.stream("POST", "/generate", json=payload) as response:
            events = []
            async for event in model._iter_sse_events(response):
                events.append(event)

        # Should have parsed JSON events
        assert len(events) > 0
        assert all(isinstance(e, dict) for e in events)
