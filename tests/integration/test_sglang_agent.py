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

"""Agent integration tests for SGLangModel.

Tests the full Strands Agent pipeline: tool calling, token trajectory tracking,
multi-turn accumulation, and state reset.
"""

from strands import Agent, tool
from strands.types.exceptions import MaxTokensReachedException
from strands_tools import calculator

SYSTEM_PROMPT = """You are a math assistant. You MUST use tools for ALL computations.
Never compute in your head — always use the appropriate tool."""


# ---------------------------------------------------------------------------
# Synthetic tools
# ---------------------------------------------------------------------------


@tool
def knowledge_base(query: str) -> str:
    """Look up facts from the knowledge base.

    Args:
        query: The fact to look up. Supported queries:
            - "price_apple": price of one apple in dollars
            - "price_orange": price of one orange in dollars
            - "tax_rate": sales tax rate as a decimal
            - "discount_threshold": minimum purchase for discount in dollars
            - "discount_rate": discount rate as a decimal
    """
    data = {
        "price_apple": "2.50",
        "price_orange": "3.75",
        "tax_rate": "0.08",
        "discount_threshold": "20",
        "discount_rate": "0.10",
    }
    return data.get(query.strip().lower(), f"Unknown query: {query}")


# ---------------------------------------------------------------------------
# Token trajectory assertion helper
# ---------------------------------------------------------------------------


def assert_trajectory_valid(model, min_response_segments: int = 1):
    """Assert token trajectory is valid: segments, loss_mask, logprobs, consistency."""
    tm = model.token_manager
    assert len(tm) > 0, "Token manager should have tokens"
    assert len(tm.token_ids) == len(tm.loss_mask) == len(tm.logprobs), "Array lengths must match"
    assert sum(length for _, length in tm.segment_info) == len(tm), "Segment lengths must sum to total"

    # First segment is always prompt
    assert tm.segment_info[0][0] is False, "First segment must be prompt"

    # Check response segments
    response_segments = [(i, length) for i, (is_resp, length) in enumerate(tm.segment_info) if is_resp]
    assert len(response_segments) >= min_response_segments, (
        f"Expected >= {min_response_segments} response segments, got {len(response_segments)}"
    )

    # Response tokens should have logprobs
    for seg_idx, _ in response_segments:
        segment_tokens = tm.segments[seg_idx]
        logprobs = [t.logprob for t in segment_tokens]
        assert any(lp is not None for lp in logprobs), f"Response segment {seg_idx} should have logprobs"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_sequential_tool_chain(model):
    """Multi-step word problem requiring dependent sequential tool calls.

    Each calculation depends on the previous result, forcing the agent through
    multiple model→tool→result→model cycles.
    """
    agent = Agent(model=model, tools=[calculator], system_prompt=SYSTEM_PROMPT)

    problem = (
        "A farmer sells 48 eggs at $0.75 each. "
        "He spends 40% of the revenue on feed. "
        "From what's left, he saves 25% and spends the rest. "
        "How much does he spend? Use the calculator for every step."
    )

    try:
        await agent.invoke_async(problem)
    except MaxTokensReachedException:
        pass  # Still verify trajectory

    # Should have used the tool multiple times (at least revenue + feed + savings)
    tool_uses = [
        content["toolUse"]
        for msg in agent.messages
        if msg.get("role") == "assistant"
        for content in msg.get("content", [])
        if "toolUse" in content
    ]
    assert len(tool_uses) >= 2, f"Expected >= 2 tool calls for dependent problem, got {len(tool_uses)}"

    # Token trajectory: N tool calls → N response + N prompt (tool result) segments
    assert_trajectory_valid(model, min_response_segments=2)

    # Segments should alternate: prompt, response, prompt (tool result), response, ...
    for i, (is_resp, _) in enumerate(model.token_manager.segment_info):
        if i == 0:
            assert not is_resp, "First segment must be prompt"
        # After the first, should alternate (though consecutive prompts are possible)


async def test_multi_tool_dispatch(model):
    """Problem requiring the agent to select between multiple distinct tools.

    The agent must use knowledge_base for lookups and calculator for computation —
    exercises tool selection logic and different tool result formats.
    """
    agent = Agent(
        model=model,
        tools=[calculator, knowledge_base],
        system_prompt=SYSTEM_PROMPT,
    )

    problem = (
        "I want to buy 5 apples and 3 oranges. "
        "First, use the knowledge_base tool to look up 'price_apple' and 'price_orange'. "
        "Then use the calculator to compute the total cost. "
        "Finally, look up 'tax_rate' from knowledge_base and calculate the total with tax."
    )

    try:
        await agent.invoke_async(problem)
    except MaxTokensReachedException:
        pass

    # Should have used both tools
    tool_names = {
        content["toolUse"]["name"]
        for msg in agent.messages
        if msg.get("role") == "assistant"
        for content in msg.get("content", [])
        if "toolUse" in content
    }
    assert "knowledge_base" in tool_names, f"Should have used knowledge_base, got: {tool_names}"
    assert "calculator" in tool_names, f"Should have used calculator, got: {tool_names}"

    assert_trajectory_valid(model, min_response_segments=2)


async def test_multi_turn(model):
    """Multi-turn accumulation without reset, then reset to fresh state.

    Proves: (1) token trajectory accumulates correctly across invoke_async calls,
    (2) reset produces a clean independent trajectory.
    """
    agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt="You are a calculator. Be very brief. Use the calculator tool.",
    )

    # -- Turn 1 --
    await agent.invoke_async("What is 12 * 15? Use calculator.")
    tokens_t1 = len(model.token_manager)
    segments_t1 = len(model.token_manager.segments)
    mask_t1 = model.token_manager.loss_mask.copy()
    assert_trajectory_valid(model)

    # -- Turn 2 (no reset — accumulates) --
    await agent.invoke_async("Now add 50 to that result. Use calculator.")
    tokens_t2 = len(model.token_manager)
    segments_t2 = len(model.token_manager.segments)

    # Tokens and segments must grow
    assert tokens_t2 > tokens_t1, f"Tokens should grow: {tokens_t1} -> {tokens_t2}"
    assert segments_t2 > segments_t1, f"Segments should grow: {segments_t1} -> {segments_t2}"

    # Previous token masks preserved (first N tokens unchanged)
    assert model.token_manager.loss_mask[:tokens_t1] == mask_t1, "Previous loss_mask must be preserved"

    assert_trajectory_valid(model, min_response_segments=2)

    # -- Reset + Turn 3 (fresh trajectory) --
    model.reset()
    assert len(model.token_manager) == 0
    assert model.token_manager.segments == []

    agent2 = Agent(
        model=model,
        tools=[calculator],
        system_prompt="You are a calculator. Be very brief. Use the calculator tool.",
    )
    await agent2.invoke_async("What is 99 + 1? Use calculator.")

    # Fresh trajectory — should be much smaller than accumulated
    assert len(model.token_manager) < tokens_t2, "Reset trajectory should be smaller than accumulated"
    assert_trajectory_valid(model)
