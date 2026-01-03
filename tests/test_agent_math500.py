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

"""Agent integration tests with real MATH-500 problems.

Tests the full Agent pipeline with SGLangModel using problems from the
HuggingFace MATH-500 dataset (https://huggingface.co/datasets/HuggingFaceH4/MATH-500).

Run with: pytest tests/test_agent_math500.py -v
Requires: Running SGLang server and strands_tools installed.

Configure via environment:
    SGLANG_BASE_URL: Server URL (default: http://localhost:8000)
    SGLANG_MODEL_ID: Model ID (default: Qwen/Qwen3-4B-Instruct-2507)
"""

import os

import pytest
from strands import Agent
from strands.types.exceptions import MaxTokensReachedException
from strands_tools import calculator
from transformers import AutoTokenizer

from strands_sglang import SGLangModel
from strands_sglang.tool_parser import HermesToolCallParser

# Configuration
BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://localhost:8000")
MODEL_ID = os.environ.get("SGLANG_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")

SYSTEM_PROMPT = """You are a math tutor. Always use the calculator tool to solve problems.

The calculator tool supports these modes:
- evaluate: Compute numeric values (default mode)
- solve: Find equation roots
- derive: Compute derivatives (use wrt="x")
- integrate: Compute integrals (use wrt="x")

Show your work and use the calculator for all computations."""

pytestmark = pytest.mark.integration


# =============================================================================
# Real MATH-500 Problems (from HuggingFaceH4/MATH-500)
# =============================================================================

MATH500_PROBLEMS = [
    # Index 27: Prealgebra L2 - Bake sale profit
    {
        "id": "math500_27",
        "subject": "Prealgebra",
        "level": 2,
        "problem": (
            "A math club is having a bake sale as a fundraiser. "
            "They sell 54 cookies at three for $1, and 20 cupcakes at $2 each, "
            "and 35 brownies at $1 each. If it cost the math club $15 to bake these items, "
            "what was their profit?"
        ),
        "answer": "78",
        "answer_variants": ["78", "$78", "78 dollars"],
    },
    # Index 38: Algebra L1 - Daily calories
    {
        "id": "math500_38",
        "subject": "Algebra",
        "level": 1,
        "problem": (
            "If a snack-size tin of peaches has 40 calories and is 2% of a person's "
            "daily caloric requirement, how many calories fulfill a person's daily caloric requirement?"
        ),
        "answer": "2000",
        "answer_variants": ["2000", "2,000", "2000 calories"],
    },
    # Index 3: Number Theory L3 - Divisors of 196
    {
        "id": "math500_3",
        "subject": "Number Theory",
        "level": 3,
        "problem": "How many positive whole-number divisors does 196 have?",
        "answer": "9",
        "answer_variants": ["9", "nine"],
    },
    # Index 5: Prealgebra L2 - Hexagon perimeter
    {
        "id": "math500_5",
        "subject": "Prealgebra",
        "level": 2,
        "problem": (
            "A regular hexagon can be divided into six equilateral triangles. "
            "If the perimeter of one of the triangles is 21 inches, "
            "what is the perimeter, in inches, of the regular hexagon?"
        ),
        "answer": "42",
        "answer_variants": ["42", "42 inches"],
    },
    # Index 6: Number Theory L3 - Perfect cube sum
    {
        "id": "math500_6",
        "subject": "Number Theory",
        "level": 3,
        "problem": (
            "What is the smallest positive perfect cube that can be written "
            "as the sum of three consecutive integers?"
        ),
        "answer": "27",
        "answer_variants": ["27", "3^3", "3**3"],
    },
    # Index 2: Algebra L3 - Function evaluation
    {
        "id": "math500_2",
        "subject": "Algebra",
        "level": 3,
        "problem": (
            "If f(x) = (3x-2)/(x-2), what is the value of f(-2) + f(-1) + f(0)? "
            "Express your answer as a common fraction."
        ),
        "answer": "14/3",
        "answer_variants": ["14/3", "\\frac{14}{3}", "4.666", "4.67"],
    },
    # Index 8: Algebra L3 - Distance formula
    {
        "id": "math500_8",
        "subject": "Algebra",
        "level": 3,
        "problem": (
            "What is the distance, in units, between the points (2, -6) and (-4, 3)? "
            "Express your answer in simplest radical form."
        ),
        "answer": "3*sqrt(13)",
        "answer_variants": ["3*sqrt(13)", "3√13", "3\\sqrt{13}", "sqrt(117)", "10.8"],
    },
    # Index 20: Algebra L3 - Complex numbers
    {
        "id": "math500_20",
        "subject": "Algebra",
        "level": 3,
        "problem": "Evaluate (1+2i)*6 - 3i.",
        "answer": "6+9i",
        "answer_variants": ["6+9i", "6 + 9i", "6+9*i"],
    },
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def model(tokenizer):
    """Create SGLangModel with appropriate token limits."""
    return SGLangModel(
        tokenizer=tokenizer,
        tool_call_parser=HermesToolCallParser(),
        base_url=BASE_URL,
        model_id=MODEL_ID,
        params={"max_new_tokens": 1024},  # Limit response length
    )


@pytest.fixture
def agent(model):
    """Create Agent with calculator tool."""
    model.reset()
    return Agent(
        model=model,
        tools=[calculator],
        system_prompt=SYSTEM_PROMPT,
    )


# =============================================================================
# Agent Basic Tests
# =============================================================================


class TestAgentBasic:
    """Basic Agent functionality tests."""

    async def test_agent_simple_query(self, agent, model):
        """Agent can respond to simple query."""
        await agent.invoke_async("What is 2 + 2?")

        # Should have messages
        assert len(agent.messages) > 0

        # Token manager should have trajectory
        assert len(model.token_manager) > 0

    async def test_agent_uses_calculator_tool(self, agent):
        """Agent uses calculator tool for math."""
        await agent.invoke_async("Calculate 15 * 23 using the calculator tool.")

        # Find tool use in messages
        tool_uses = []
        for msg in agent.messages:
            if msg.get("role") == "assistant":
                for content in msg.get("content", []):
                    if "toolUse" in content:
                        tool_uses.append(content["toolUse"])

        # Should have used calculator
        assert len(tool_uses) > 0
        assert any(tu.get("name") == "calculator" for tu in tool_uses)

    async def test_agent_tito_structure(self, agent, model):
        """Agent invocation creates valid TITO structure."""
        await agent.invoke_async("What is 100 / 4?")

        # Check TITO structure
        assert len(model.token_manager.segments) >= 2
        assert len(model.token_manager.token_ids) > 0
        assert len(model.token_manager.output_mask) == len(model.token_manager.token_ids)

        # Should have both prompt (False) and response (True) masks
        assert not all(model.token_manager.output_mask)
        assert any(model.token_manager.output_mask)


# =============================================================================
# MATH-500 Problem Tests
# =============================================================================


class TestMath500Problems:
    """Tests using real MATH-500 problems."""

    @pytest.mark.parametrize("problem", MATH500_PROBLEMS[:4], ids=lambda p: p["id"])
    async def test_math500_problem(self, agent, model, problem):
        """Test Agent on real MATH-500 problem."""
        model.reset()

        # Invoke agent with problem (may hit max_tokens on verbose responses)
        try:
            await agent.invoke_async(problem["problem"])
        except MaxTokensReachedException:
            # Expected for verbose problems - still verify TITO structure
            pass

        # Get final assistant response
        final_response = ""
        for msg in reversed(agent.messages):
            if msg.get("role") == "assistant":
                for content in msg.get("content", []):
                    if "text" in content:
                        final_response = content["text"]
                        break
                if final_response:
                    break

        # Log for debugging
        print(f"\nProblem: {problem['id']}")
        print(f"Expected: {problem['answer']}")
        print(f"Response snippet: {final_response[:200] if final_response else 'N/A'}...")

        # Verify TITO structure is valid regardless of completion
        assert len(model.token_manager) > 0
        assert len(model.token_manager.segments) >= 2


# =============================================================================
# Token-Text Consistency Tests
# =============================================================================


class TestTokenTextConsistency:
    """Verify decoded token trajectory matches Agent's messages."""

    async def test_user_message_in_trajectory(self, agent, model, tokenizer):
        """User message text appears in decoded trajectory."""
        model.reset()

        user_msg = "Calculate 25 plus 17."
        await agent.invoke_async(user_msg)

        decoded = tokenizer.decode(model.token_manager.token_ids)

        # User message should appear in trajectory
        # (may have slight variations due to tokenization, check key words)
        assert "25" in decoded
        assert "17" in decoded

    async def test_system_prompt_in_trajectory(self, agent, model, tokenizer):
        """System prompt appears in decoded trajectory."""
        model.reset()

        await agent.invoke_async("What is 1 + 1?")

        decoded = tokenizer.decode(model.token_manager.token_ids)

        # System prompt keywords should appear
        assert "calculator" in decoded.lower() or "math" in decoded.lower()

    async def test_tool_name_in_trajectory(self, agent, model, tokenizer):
        """Tool call name appears in decoded trajectory when tool is used."""
        model.reset()

        try:
            await agent.invoke_async("Use calculator to compute 5 * 5.")
        except MaxTokensReachedException:
            pass

        decoded = tokenizer.decode(model.token_manager.token_ids)

        # If tool was called, "calculator" should appear in trajectory
        # Check for tool_call markers or tool name
        has_tool_reference = (
            "calculator" in decoded.lower() or
            "tool_call" in decoded.lower() or
            "<tool_call>" in decoded
        )
        assert has_tool_reference, "Tool reference should appear in trajectory"

    async def test_chat_template_markers_present(self, agent, model, tokenizer):
        """Chat template markers appear in decoded trajectory."""
        model.reset()

        await agent.invoke_async("Hi")

        decoded = tokenizer.decode(model.token_manager.token_ids)

        # Qwen chat template uses these markers
        assert "<|im_start|>" in decoded, "Should have im_start marker"
        assert "<|im_end|>" in decoded, "Should have im_end marker"

        # Should have role markers
        assert "system" in decoded or "user" in decoded or "assistant" in decoded


# =============================================================================
# TITO Within Single Invocation Tests
# =============================================================================


class TestSingleInvocationTITO:
    """Test TITO structure within a single agent.invoke_async() call."""

    async def test_tool_use_creates_multiple_segments(self, agent, model):
        """Single invocation with tool use creates proper segment structure."""
        model.reset()

        try:
            await agent.invoke_async("Use the calculator to compute 7 * 8.")
        except MaxTokensReachedException:
            pass

        segment_info = model.token_manager.segment_info

        # Should have at least: initial prompt + first response
        assert len(segment_info) >= 2

        # First segment should be prompt (False)
        assert segment_info[0][0] is False, "First segment should be prompt"

        # Should have at least one response segment (True)
        response_segments = [s for s in segment_info if s[0] is True]
        assert len(response_segments) >= 1, "Should have response segment"

    async def test_tool_result_marked_as_prompt(self, agent, model):
        """Tool results within invocation are marked as prompt (loss_mask=False)."""
        model.reset()

        try:
            await agent.invoke_async("Calculate 100 / 4 using the calculator.")
        except MaxTokensReachedException:
            pass

        segment_info = model.token_manager.segment_info

        # If tool was used, we should have pattern:
        # prompt (F) -> response (T) -> tool_result prompt (F) -> final response (T)
        # Or at minimum: prompt (F) -> response (T)

        if len(segment_info) >= 4:
            # Full tool use pattern
            # After first response, tool result should be prompt
            assert segment_info[2][0] is False, "Tool result segment should be prompt"

    async def test_logprobs_only_for_responses(self, agent, model):
        """Only response tokens should have logprobs (prompt tokens may have None)."""
        model.reset()

        await agent.invoke_async("What is 6 + 6?")

        segments = model.token_manager.segments
        segment_info = model.token_manager.segment_info

        for i, (is_response, _) in enumerate(segment_info):
            segment_tokens = segments[i]

            if is_response:
                # Response tokens should have float logprobs
                for token in segment_tokens:
                    assert isinstance(token.logprob, float), \
                        f"Response token should have float logprob, got {type(token.logprob)}"
            # Prompt tokens may have None or float (depending on SGLang config)


# =============================================================================
# Multi-Turn Consistency Tests
# =============================================================================


class TestMultiTurnConsistency:
    """Test TITO consistency across multiple invoke_async() calls."""

    async def test_tokens_accumulate_across_turns(self, model, tokenizer):
        """Tokens accumulate correctly across multiple turns."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="You are a calculator. Be brief.",
        )

        # Turn 1: Simple addition
        await agent.invoke_async("2 + 3 = ?")
        tokens_after_turn1 = len(model.token_manager)
        segments_after_turn1 = len(model.token_manager.segments)

        # Turn 2: Another simple calculation
        await agent.invoke_async("4 + 5 = ?")
        tokens_after_turn2 = len(model.token_manager)
        segments_after_turn2 = len(model.token_manager.segments)

        # Tokens should increase
        assert tokens_after_turn2 > tokens_after_turn1, \
            f"Tokens should increase: {tokens_after_turn1} -> {tokens_after_turn2}"

        # Segments should increase
        assert segments_after_turn2 > segments_after_turn1, \
            f"Segments should increase: {segments_after_turn1} -> {segments_after_turn2}"

    async def test_previous_content_preserved(self, model, tokenizer):
        """Previous turn content is preserved in trajectory."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Be brief.",
        )

        # Turn 1
        await agent.invoke_async("Remember: 7")
        decoded_after_turn1 = tokenizer.decode(model.token_manager.token_ids)

        # Turn 2
        await agent.invoke_async("What number?")
        decoded_after_turn2 = tokenizer.decode(model.token_manager.token_ids)

        # Turn 1 content should still be in trajectory
        assert "7" in decoded_after_turn2, "Previous turn content should be preserved"
        assert len(decoded_after_turn2) > len(decoded_after_turn1), \
            "Trajectory should grow"

    async def test_output_mask_consistency_across_turns(self, model, tokenizer):
        """Output mask remains consistent across turns."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Brief answers only.",
        )

        await agent.invoke_async("5+5=?")
        mask_after_turn1 = model.token_manager.output_mask.copy()
        tokens_after_turn1 = len(model.token_manager)

        await agent.invoke_async("6+6=?")
        mask_after_turn2 = model.token_manager.output_mask

        # First N tokens should have same mask as before
        assert mask_after_turn2[:tokens_after_turn1] == mask_after_turn1, \
            "Previous token masks should be preserved"

    async def test_segment_info_grows_correctly(self, model, tokenizer):
        """Segment info grows correctly across turns."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Brief.",
        )

        await agent.invoke_async("1+1")
        info1 = list(model.token_manager.segment_info)

        await agent.invoke_async("2+2")
        info2 = list(model.token_manager.segment_info)

        # Previous segments should be unchanged
        assert info2[:len(info1)] == info1, "Previous segment info should be preserved"

        # New segments should be added
        assert len(info2) > len(info1), "New segments should be added"


# =============================================================================
# Edge Cases
# =============================================================================


class TestAgentEdgeCases:
    """Edge case tests for Agent with SGLangModel."""

    async def test_empty_response_handling(self, agent, model):
        """Handle case where model gives minimal response."""
        model.reset()

        # Simple query that might get short response
        await agent.invoke_async("Reply with just 'ok'.")

        # Should still have valid structure
        assert len(model.token_manager) > 0
        assert len(model.token_manager.segments) >= 2

    async def test_reset_between_problems(self, model, tokenizer):
        """Reset properly clears state between problems."""
        # First agent
        agent1 = Agent(
            model=model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
        )
        await agent1.invoke_async("What is 1 + 1?")
        first_tokens = model.token_manager.token_ids.copy()

        # Reset
        model.reset()
        assert len(model.token_manager) == 0

        # Second agent
        agent2 = Agent(
            model=model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
        )
        await agent2.invoke_async("What is 2 + 2?")
        second_tokens = model.token_manager.token_ids

        # Should be different trajectories
        assert first_tokens != second_tokens

    async def test_long_problem(self, agent, model):
        """Handle longer, multi-step problem."""
        model.reset()

        problem = (
            "A store sells apples for $2 each and oranges for $3 each. "
            "If someone buys 5 apples and 4 oranges, and pays with a $50 bill, "
            "how much change do they receive? Use the calculator for each step."
        )

        await agent.invoke_async(problem)

        # Should complete with valid trajectory
        assert len(model.token_manager) > 0

        # Should have multiple segments (likely multiple tool calls)
        assert len(model.token_manager.segments) >= 2


# =============================================================================
# Critical RL Training Tests
# =============================================================================


class TestRetokenizationDrift:
    """Tests for retokenization drift - CRITICAL for RL training correctness.

    In RL training, we compute policy gradients using token-level logprobs.
    If the TITO trajectory doesn't match what would be produced by re-encoding
    the decoded text, the gradient updates will be computed on wrong tokens,
    leading to training instability or divergence.

    The encode → decode → re-encode cycle MUST produce identical token IDs.
    """

    async def test_single_turn_no_drift(self, model, tokenizer):
        """Single turn: encode→decode→re-encode produces identical tokens."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Be brief.",
        )

        await agent.invoke_async("What is 2 + 3?")

        # Get TITO tokens
        tito_tokens = model.token_manager.token_ids

        # Decode to text
        decoded = tokenizer.decode(tito_tokens)

        # Re-encode
        re_encoded = tokenizer.encode(decoded, add_special_tokens=False)

        # CRITICAL: Must match exactly for correct RL gradients
        assert list(tito_tokens) == list(re_encoded), (
            f"Retokenization drift detected!\n"
            f"TITO tokens: {tito_tokens[:50]}...\n"
            f"Re-encoded:  {re_encoded[:50]}...\n"
            f"Decoded text sample: {decoded[:200]}..."
        )

    async def test_multi_turn_no_drift(self, model, tokenizer):
        """Multi-turn: accumulated trajectory has no retokenization drift."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Brief answers.",
        )

        # Multiple turns
        await agent.invoke_async("5 + 5 = ?")
        await agent.invoke_async("Now add 10 to that.")

        tito_tokens = model.token_manager.token_ids
        decoded = tokenizer.decode(tito_tokens)
        re_encoded = tokenizer.encode(decoded, add_special_tokens=False)

        assert list(tito_tokens) == list(re_encoded), (
            f"Multi-turn retokenization drift detected!\n"
            f"Length mismatch: {len(tito_tokens)} vs {len(re_encoded)}"
        )

    async def test_tool_use_no_drift(self, model, tokenizer):
        """Tool use trajectory has no retokenization drift."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Always use calculator.",
        )

        try:
            await agent.invoke_async("Calculate 15 * 7 using the calculator tool.")
        except MaxTokensReachedException:
            pass  # Still check drift

        tito_tokens = model.token_manager.token_ids
        decoded = tokenizer.decode(tito_tokens)
        re_encoded = tokenizer.encode(decoded, add_special_tokens=False)

        # This is the critical invariant for RL training
        assert list(tito_tokens) == list(re_encoded), (
            "Tool use retokenization drift detected! "
            "This will cause incorrect policy gradients in RL training."
        )


class TestMultipleToolCalls:
    """Tests for multiple tool calls in a single turn."""

    async def test_multiple_calculations_single_turn(self, model, tokenizer):
        """Agent handles multiple sequential calculations correctly."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Use calculator for each step. Be methodical.",
        )

        problem = (
            "Calculate step by step: "
            "First compute 10 * 5, then compute 30 + 20, "
            "finally add those two results together."
        )

        try:
            await agent.invoke_async(problem)
        except MaxTokensReachedException:
            pass

        # Count tool calls in trajectory
        decoded = tokenizer.decode(model.token_manager.token_ids)
        tool_call_count = decoded.count("<tool_call>")

        # Should have multiple tool calls
        assert tool_call_count >= 2, (
            f"Expected multiple tool calls, found {tool_call_count}.\n"
            f"Trajectory sample: {decoded[:500]}..."
        )

        # Each tool call should have corresponding segments
        segment_info = model.token_manager.segment_info
        response_segments = [s for s in segment_info if s[0] is True]
        assert len(response_segments) >= 2, "Should have multiple response segments"

    async def test_tool_results_properly_masked(self, model, tokenizer):
        """All tool results are marked as prompt (loss_mask=False)."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Use calculator. Show your work.",
        )

        try:
            await agent.invoke_async("Compute 7*8 then 9*6 using calculator.")
        except MaxTokensReachedException:
            pass

        # Verify masking pattern
        segment_info = model.token_manager.segment_info

        # Should have alternating pattern with prompts for tool results
        prompt_count = sum(1 for s in segment_info if s[0] is False)
        response_count = sum(1 for s in segment_info if s[0] is True)

        # If tools were used, we should have more than 1 prompt segment
        # (initial prompt + tool result prompts)
        if response_count >= 2:
            assert prompt_count >= 2, (
                f"Tool results should create prompt segments. "
                f"Prompts: {prompt_count}, Responses: {response_count}"
            )


class TestLongConversation:
    """Tests for long conversations (5+ turns) to detect accumulation issues."""

    async def test_five_turn_conversation(self, model, tokenizer):
        """5-turn conversation maintains TITO integrity throughout."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Just give the number, very brief.",
        )

        token_counts = []
        segment_counts = []

        questions = [
            "1+1=?",
            "2+2=?",
            "3+3=?",
            "4+4=?",
            "5+5=?",
        ]

        for q in questions:
            await agent.invoke_async(q)
            token_counts.append(len(model.token_manager))
            segment_counts.append(len(model.token_manager.segments))

        # Tokens should monotonically increase
        for i in range(1, len(token_counts)):
            assert token_counts[i] > token_counts[i - 1], (
                f"Turn {i + 1}: tokens should increase. "
                f"{token_counts[i - 1]} -> {token_counts[i]}"
            )

        # Segments should monotonically increase
        for i in range(1, len(segment_counts)):
            assert segment_counts[i] > segment_counts[i - 1], (
                f"Turn {i + 1}: segments should increase. "
                f"{segment_counts[i - 1]} -> {segment_counts[i]}"
            )

        # Final segment count should be at least 2 per turn
        assert segment_counts[-1] >= 10, (
            f"Expected at least 10 segments for 5 turns, got {segment_counts[-1]}"
        )

    async def test_long_conversation_no_drift(self, model, tokenizer):
        """Long conversation still has no retokenization drift."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Brief.",
        )

        # 5 quick turns
        for i in range(5):
            await agent.invoke_async(f"{i}+{i}=?")

        tito_tokens = model.token_manager.token_ids
        decoded = tokenizer.decode(tito_tokens)
        re_encoded = tokenizer.encode(decoded, add_special_tokens=False)

        assert list(tito_tokens) == list(re_encoded), (
            f"Long conversation drift detected at {len(tito_tokens)} tokens!"
        )

    async def test_context_preserved_across_turns(self, model, tokenizer):
        """All previous context is preserved in trajectory across 5 turns."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Remember numbers I tell you.",
        )

        numbers = ["11", "22", "33", "44", "55"]

        for num in numbers:
            await agent.invoke_async(f"Remember: {num}")

        decoded = tokenizer.decode(model.token_manager.token_ids)

        # All numbers should appear in final trajectory
        for num in numbers:
            assert num in decoded, (
                f"Number {num} not found in trajectory. "
                f"Context may not be properly preserved."
            )


class TestMessageToTokenDrift:
    """Tests for drift between agent.messages and TITO tokens.

    This tests that reconstructing tokens from agent.messages via apply_chat_template
    produces IDENTICAL tokens to what TITO recorded during generation.

    This is critical for:
    1. Offline RL from logged messages
    2. Trajectory reconstruction from saved conversations
    3. Verifying the model saw exactly what we think it saw
    """

    async def test_single_turn_message_token_match(self, model, tokenizer):
        """Single turn: formatted messages produce identical tokens to TITO."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Be brief.",
        )

        await agent.invoke_async("What is 5 + 5?")

        # Get TITO tokens
        tito_tokens = model.token_manager.token_ids

        # Reconstruct from messages using the same formatting
        openai_messages = model.format_request_messages(agent.messages, "Be brief.")
        # Use _current_tools directly (already in OpenAI format)
        tools = model._current_tools

        formatted = tokenizer.apply_chat_template(
            openai_messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )
        reconstructed_tokens = tokenizer.encode(formatted, add_special_tokens=False)

        # Strip trailing newline from reconstructed (chat template formatting, not model output)
        # The trailing \n has no logprobs and doesn't affect RL training
        if reconstructed_tokens and reconstructed_tokens[-1] == 198:  # \n token
            reconstructed_tokens = reconstructed_tokens[:-1]

        # Token IDs must match exactly
        assert list(tito_tokens) == list(reconstructed_tokens), (
            f"Message-to-token drift detected!\n"
            f"TITO: {len(tito_tokens)} tokens\n"
            f"Reconstructed: {len(reconstructed_tokens)} tokens\n"
            f"First diff at: {next((i for i, (a, b) in enumerate(zip(tito_tokens, reconstructed_tokens)) if a != b), 'length mismatch')}"
        )

    async def test_multi_turn_message_token_match(self, model, tokenizer):
        """Multi-turn: formatted messages produce identical tokens to TITO."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Brief.",
        )

        await agent.invoke_async("2+2=?")
        await agent.invoke_async("3+3=?")

        # Get TITO tokens
        tito_tokens = model.token_manager.token_ids

        openai_messages = model.format_request_messages(agent.messages, "Brief.")
        tools = model._current_tools

        formatted = tokenizer.apply_chat_template(
            openai_messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )
        reconstructed_tokens = tokenizer.encode(formatted, add_special_tokens=False)

        # Strip trailing newline (chat template formatting, not model output)
        if reconstructed_tokens and reconstructed_tokens[-1] == 198:
            reconstructed_tokens = reconstructed_tokens[:-1]

        assert list(tito_tokens) == list(reconstructed_tokens), (
            f"Multi-turn message-to-token drift!\n"
            f"TITO: {len(tito_tokens)} tokens, Reconstructed: {len(reconstructed_tokens)} tokens"
        )

    async def test_tool_use_message_token_match(self, model, tokenizer):
        """Tool use: formatted messages with tool calls produce identical tokens."""
        model.reset()

        agent = Agent(
            model=model,
            tools=[calculator],
            system_prompt="Use calculator.",
        )

        try:
            await agent.invoke_async("Calculate 7 * 8.")
        except MaxTokensReachedException:
            pass

        # Get TITO tokens
        tito_tokens = model.token_manager.token_ids

        openai_messages = model.format_request_messages(agent.messages, "Use calculator.")
        tools = model._current_tools

        formatted = tokenizer.apply_chat_template(
            openai_messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )
        reconstructed_tokens = tokenizer.encode(formatted, add_special_tokens=False)

        # Strip trailing newline (chat template formatting, not model output)
        if reconstructed_tokens and reconstructed_tokens[-1] == 198:
            reconstructed_tokens = reconstructed_tokens[:-1]

        assert list(tito_tokens) == list(reconstructed_tokens), (
            f"Tool use message-to-token drift!\n"
            f"TITO: {len(tito_tokens)} tokens, Reconstructed: {len(reconstructed_tokens)} tokens"
        )

