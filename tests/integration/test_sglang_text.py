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

"""Integration tests for SGLangModel streaming and token trajectory."""


async def test_stream_generation(model):
    """Single-turn generation produces correct Strands events, metadata, and token segments."""
    messages = [{"role": "user", "content": [{"text": "Say 'hello' and nothing else."}]}]

    events = []
    async for event in model.stream(messages, system_prompt="Be brief."):
        events.append(event)

    # Strands event protocol
    assert events[0] == {"messageStart": {"role": "assistant"}}
    text_deltas = [e["contentBlockDelta"]["delta"]["text"] for e in events if "contentBlockDelta" in e]
    assert len(text_deltas) > 0
    assert "hello" in "".join(text_deltas).lower()
    stop_events = [e for e in events if "messageStop" in e]
    assert stop_events[0]["messageStop"]["stopReason"] == "end_turn"

    # Metadata with usage
    metadata = next(e["metadata"] for e in events if "metadata" in e)
    usage = metadata["usage"]
    assert usage["inputTokens"] > 0
    assert usage["outputTokens"] > 0
    assert usage["totalTokens"] == usage["inputTokens"] + usage["outputTokens"]

    # Token trajectory: prompt + response segments, consistent lengths
    tm = model.token_manager
    assert len(tm) > 0
    assert len(tm.token_ids) == len(tm.loss_mask) == len(tm.logprobs)
    segments = tm.segment_info
    assert segments[0][0] is False  # prompt
    assert segments[1][0] is True  # response
    assert sum(length for _, length in segments) == len(tm)


async def test_multi_turn_tool_use(model, calculator_tool):
    """Multi-turn tool use: logprobs complete and KV cache hit on turn 2.

    Regression test for logprob_start_len — incorrect values cause None logprobs
    at segment boundaries or defeat KV cache reuse.
    """
    system_prompt = "You are a calculator. Use the calculator tool for ALL math. Never compute in your head."

    # Turn 1
    messages = [{"role": "user", "content": [{"text": "What is 5 * 8?"}]}]
    async for _ in model.stream(messages, tool_specs=[calculator_tool], system_prompt=system_prompt):
        pass

    # Inject tool result for turn 2
    messages.append(
        {
            "role": "assistant",
            "content": [
                {"text": '<tool_call>\n{"name": "calculator", "arguments": {"expression": "5 * 8"}}\n</tool_call>'},
                {"toolUse": {"toolUseId": "call_1", "name": "calculator", "input": {"expression": "5 * 8"}}},
            ],
        }
    )
    messages.append({"role": "user", "content": [{"toolResult": {"toolUseId": "call_1", "content": [{"text": "40"}]}}]})

    # Turn 2
    events = []
    async for event in model.stream(messages, tool_specs=[calculator_tool], system_prompt=system_prompt):
        events.append(event)

    content_deltas = [e for e in events if "contentBlockDelta" in e]
    assert len(content_deltas) > 0, "Model should generate response after tool result"

    # Token trajectory: ≥4 segments (prompt, response, tool-result prompt, response)
    tm = model.token_manager
    segment_info = tm.segment_info
    assert len(segment_info) >= 4, f"Expected >=4 segments, got {len(segment_info)}"
    assert sum(1 for is_resp, _ in segment_info if is_resp) >= 2

    # logprob_start_len regression: no None logprobs after segment 0
    # Segment 0 (initial prompt) may have None for BOS token — that's expected.
    # Segments 1+ must have no None logprobs.
    for i, (is_response, _) in enumerate(segment_info):
        if i == 0:
            continue
        segment_logprobs = [t.logprob for t in tm.segments[i]]
        none_count = sum(1 for lp in segment_logprobs if lp is None)
        seg_type = "Response" if is_response else "Prompt"
        assert none_count == 0, (
            f"{seg_type} segment {i} has {none_count} None logprobs out of {len(segment_logprobs)} tokens. "
            f"logprob_start_len may be incorrect."
        )

    # KV cache: turn 2 should reuse prefix from turn 1
    metadata = next(e["metadata"] for e in events if "metadata" in e)
    cached = metadata["usage"]["cacheReadInputTokens"]
    assert cached > 0, f"Expected radix cache hit on turn 2, got cacheReadInputTokens={cached}"
