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

"""Integration tests for ToolLimiter.

Tests the limiter with real agent loops to verify:
1. Iteration counting is correct (one model response with tools = 1 iteration)
2. Call counting is correct (each tool call counted individually)
3. Exception is raised at the right time (after complete iterations)
4. Token trajectory is clean (no truncation needed)

Note: An "iteration" is one model response that requests tools. Multiple parallel
tool calls in one response count as a single iteration but as individual calls.
"""

import pytest
from strands import Agent
from strands.types.exceptions import EventLoopException
from strands_tools import calculator

from strands_sglang import MaxToolCallsReachedError, MaxToolIterationsReachedError, SGLangModel, ToolLimiter
from strands_sglang.client import SGLangClient
from strands_sglang.tool_parsers import HermesToolParser


def assert_max_iterations_reached(exc_info, expected_count: int | None = None):
    """Helper to verify MaxToolIterationsReachedError was raised.

    Strands wraps hook exceptions in EventLoopException, so we check the cause.
    """
    exc = exc_info.value
    # Could be raised directly or wrapped in EventLoopException
    if isinstance(exc, EventLoopException):
        cause = exc.__cause__
        assert isinstance(cause, MaxToolIterationsReachedError), (
            f"Expected MaxToolIterationsReachedError, got {type(cause)}"
        )
        if expected_count is not None:
            assert f"({expected_count})" in str(cause)
    elif isinstance(exc, MaxToolIterationsReachedError):
        if expected_count is not None:
            assert f"({expected_count})" in str(exc)
    else:
        raise AssertionError(f"Expected MaxToolIterationsReachedError or EventLoopException, got {type(exc)}")


SYSTEM_PROMPT = """You are a calculator assistant. You MUST use the calculator tool for ALL arithmetic.
Never compute in your head - always use the calculator tool."""


# Problem that requires dependent calculations (forces sequential tool use)
# The model MUST wait for the result of step 1 before computing step 2
SEQUENTIAL_PROBLEM = """
I have a secret number X. When you call the calculator with "7 * 13", it will tell you X.
Then calculate X + 100. You must use the calculator for both steps.
What is the final answer?
"""

# Simple problem - likely completes in 1 iteration (possibly with parallel calls)
SIMPLE_PROBLEM = "What is 25 * 4? Use the calculator."

# Problem that may use parallel or sequential calls
MULTI_CALC_PROBLEM = "Calculate 10+5, 20+10, and 30+15 using the calculator."


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def fresh_model(tokenizer, sglang_base_url):
    """Create a fresh SGLangModel for each test."""
    client = SGLangClient(base_url=sglang_base_url)
    yield SGLangModel(
        client=client,
        tokenizer=tokenizer,
        tool_parser=HermesToolParser(),
        sampling_params={"max_new_tokens": 32768},
    )
    await client.close()


# =============================================================================
# Tests
# =============================================================================


class TestToolLimiterBasic:
    """Basic limiter functionality tests."""

    async def test_limiter_stops_after_one_iteration(self, fresh_model):
        """Limiter with max_tool_iters=1 should stop after first tool-using response."""
        limiter = ToolLimiter(max_tool_iters=1)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        # This problem requires at least 2 iterations (get X, then compute X+100)
        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)) as exc_info:
            await agent.invoke_async(SEQUENTIAL_PROBLEM)

        # Should have exactly 1 iteration
        assert limiter.tool_iter_count == 1
        assert_max_iterations_reached(exc_info, expected_count=1)

    async def test_limiter_allows_completion_under_limit(self, fresh_model):
        """Agent should complete normally if under the limit."""
        limiter = ToolLimiter(max_tool_iters=10)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        # Simple problem should complete in few iterations
        result = await agent.invoke_async(SIMPLE_PROBLEM)

        # Should complete without exception
        assert result is not None
        assert limiter.tool_iter_count >= 1  # At least one tool use
        assert limiter.tool_iter_count <= 10  # Under limit

    async def test_limiter_no_limit_when_none(self, fresh_model):
        """Limiter should not limit when max_tool_iters is None."""
        limiter = ToolLimiter(max_tool_iters=None)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        # Should complete without exception
        result = await agent.invoke_async(SIMPLE_PROBLEM)
        assert result is not None

    async def test_parallel_tools_count_as_one_iteration(self, fresh_model):
        """Multiple parallel tool calls in one response = 1 iteration."""
        limiter = ToolLimiter(max_tool_iters=1)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        # This problem may trigger parallel tool calls (10+5, 20+10, 30+15)
        # But all in one response = 1 iteration
        # The limiter should raise after the first iteration completes
        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)) as exc_info:
            await agent.invoke_async(MULTI_CALC_PROBLEM + " Then add all three results.")

        # Even if model made 3 parallel calls, it's still 1 iteration
        assert limiter.tool_iter_count == 1
        assert_max_iterations_reached(exc_info, expected_count=1)


class TestToolLimiterTrajectory:
    """Tests for trajectory cleanliness after limiter stops."""

    async def test_iteration_count_le_tool_messages(self, fresh_model):
        """Limiter iteration count should be <= tool messages (parallel calls = 1 iteration)."""
        limiter = ToolLimiter(max_tool_iters=2)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        # Run until limit or completion
        try:
            await agent.invoke_async(SEQUENTIAL_PROBLEM)
        except (MaxToolIterationsReachedError, EventLoopException):
            pass  # Expected if limit reached

        # Get trajectory from model
        trajectory = fresh_model.format_request_messages(agent.messages, None)

        # Count tool messages in trajectory
        tool_message_count = sum(1 for msg in trajectory if msg["role"] == "tool")

        # Iteration count <= tool message count because parallel tool calls
        # in a single response count as 1 iteration but produce N tool messages
        assert limiter.tool_iter_count <= tool_message_count, (
            f"iteration_count ({limiter.tool_iter_count}) should be <= tool_message_count ({tool_message_count})"
        )
        assert limiter.tool_iter_count > 0, "Should have at least one iteration"
        assert tool_message_count > 0, "Should have at least one tool message"

    async def test_iteration_count_matches_tool_messages_on_completion(self, fresh_model):
        """Iteration count matches tool messages when agent completes normally."""
        limiter = ToolLimiter(max_tool_iters=10)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        # Should complete without hitting limit
        await agent.invoke_async(SIMPLE_PROBLEM)

        trajectory = fresh_model.format_request_messages(agent.messages, None)
        tool_message_count = sum(1 for msg in trajectory if msg["role"] == "tool")

        assert limiter.tool_iter_count == tool_message_count, (
            f"Mismatch: limiter.iteration_count={limiter.tool_iter_count}, tool_message_count={tool_message_count}"
        )

    async def test_trajectory_is_clean_after_limit(self, fresh_model):
        """Token trajectory should be clean (complete iterations only)."""
        limiter = ToolLimiter(max_tool_iters=1)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)):
            await agent.invoke_async(SEQUENTIAL_PROBLEM)

        # Token manager should have tokens
        token_manager = fresh_model.token_manager
        assert len(token_manager) > 0

        # Should have both prompt and response segments
        segment_info = token_manager.segment_info
        assert len(segment_info) >= 2  # At least 1 prompt + 1 response

        # All arrays should have same length (consistency check)
        assert len(token_manager.token_ids) == len(token_manager.loss_mask)
        assert len(token_manager.token_ids) == len(token_manager.logprobs)

        # Should have some output tokens (model responses)
        output_count = sum(1 for mask in token_manager.loss_mask if mask)
        assert output_count > 0

    async def test_response_segments_match_iterations(self, fresh_model):
        """Number of response segments should match iteration count."""
        limiter = ToolLimiter(max_tool_iters=1)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)):
            await agent.invoke_async(SEQUENTIAL_PROBLEM)

        segment_info = fresh_model.token_manager.segment_info

        # Count response segments (is_output=True)
        response_segments = sum(1 for is_output, _ in segment_info if is_output)

        # Should have response segments matching iteration count
        assert response_segments == limiter.tool_iter_count


class TestToolLimiterEdgeCases:
    """Edge case tests."""

    async def test_max_tool_iters_zero_stops_after_first(self, fresh_model):
        """max_tool_iters=0 should stop after first complete iteration.

        With the current logic:
        - Model generates toolUse -> tool_iter_count = 1
        - Tool result arrives -> check: 1 >= 0 -> True -> raise
        So we complete 1 iteration before stopping.
        """
        limiter = ToolLimiter(max_tool_iters=0)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)):
            await agent.invoke_async(SIMPLE_PROBLEM)

        # With max_iterations=0, stops after first complete iteration
        assert limiter.tool_iter_count == 1

    async def test_limiter_reset_clears_count(self, fresh_model):
        """Limiter reset should clear iteration count."""
        limiter = ToolLimiter(max_tool_iters=1)

        # First invocation with fresh agent
        agent1 = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)):
            await agent1.invoke_async(SEQUENTIAL_PROBLEM)

        first_count = limiter.tool_iter_count
        assert first_count == 1

        # Reset limiter and model
        limiter.reset()
        fresh_model.reset()

        assert limiter.tool_iter_count == 0

        # Second invocation with fresh agent (agent state is not shared)
        agent2 = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)):
            await agent2.invoke_async(SEQUENTIAL_PROBLEM)

        assert limiter.tool_iter_count == 1


class TestToolLimiterSequentialProblems:
    """Tests with problems that require sequential tool use."""

    async def test_problem_completes_with_sufficient_iterations(self, fresh_model):
        """Problem should complete when given enough iterations."""
        # Allow enough iterations to complete
        limiter = ToolLimiter(max_tool_iters=5)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        # This problem may complete in 1-2 iterations depending on model behavior
        # (model may batch parallel calls or make sequential calls)
        result = await agent.invoke_async(SEQUENTIAL_PROBLEM)

        # Should complete with at least 1 iteration
        assert limiter.tool_iter_count >= 1
        assert limiter.tool_iter_count <= 5
        assert result is not None

    async def test_limit_stops_mid_dependent_calculation(self, fresh_model):
        """Limiter should stop dependent calculation at limit."""
        limiter = ToolLimiter(max_tool_iters=1)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        # Should stop after first iteration, before second calculation
        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)):
            await agent.invoke_async(SEQUENTIAL_PROBLEM)

        assert limiter.tool_iter_count == 1

    async def test_high_limit_allows_completion(self, fresh_model):
        """High iteration limit should allow problem to complete."""
        limiter = ToolLimiter(max_tool_iters=20)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        result = await agent.invoke_async(SIMPLE_PROBLEM)

        # Should complete normally
        assert result is not None
        assert limiter.tool_iter_count >= 1
        assert limiter.tool_iter_count < 20  # Shouldn't need that many


# =============================================================================
# max_tool_calls integration tests
# =============================================================================


def assert_max_calls_reached(exc_info, expected_count: int | None = None):
    """Helper to verify MaxToolCallsReachedError was raised.

    Strands wraps hook exceptions in EventLoopException, so we check the cause.
    """
    exc = exc_info.value
    if isinstance(exc, EventLoopException):
        cause = exc.__cause__
        assert isinstance(cause, MaxToolCallsReachedError), f"Expected MaxToolCallsReachedError, got {type(cause)}"
        if expected_count is not None:
            assert f"({expected_count})" in str(cause)
    elif isinstance(exc, MaxToolCallsReachedError):
        if expected_count is not None:
            assert f"({expected_count})" in str(exc)
    else:
        raise AssertionError(f"Expected MaxToolCallsReachedError or EventLoopException, got {type(exc)}")


class TestToolLimiterMaxCalls:
    """Integration tests for max_tool_calls limit."""

    async def test_call_limit_stops_agent(self, fresh_model):
        """Agent should stop when total tool calls reach the limit."""
        limiter = ToolLimiter(max_tool_calls=1)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        with pytest.raises((MaxToolCallsReachedError, EventLoopException)) as exc_info:
            await agent.invoke_async(SEQUENTIAL_PROBLEM)

        assert limiter.tool_call_count >= 1
        assert_max_calls_reached(exc_info, expected_count=1)

    async def test_call_limit_allows_completion_under_limit(self, fresh_model):
        """Agent should complete normally if total calls stay under the limit."""
        limiter = ToolLimiter(max_tool_calls=20)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        result = await agent.invoke_async(SIMPLE_PROBLEM)

        assert result is not None
        assert limiter.tool_call_count >= 1
        assert limiter.tool_call_count <= 20

    async def test_parallel_calls_counted_individually(self, fresh_model):
        """Parallel tool calls should each count toward the call limit."""
        # Allow only 2 calls — if model makes 3 parallel calls, it should stop
        limiter = ToolLimiter(max_tool_calls=2)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        with pytest.raises((MaxToolCallsReachedError, EventLoopException)):
            await agent.invoke_async(MULTI_CALC_PROBLEM)

        # Should have counted individual calls, not just iterations
        assert limiter.tool_call_count >= limiter.tool_iter_count

    async def test_call_count_ge_iter_count(self, fresh_model):
        """Call count should always be >= iteration count."""
        limiter = ToolLimiter(max_tool_calls=10)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        try:
            await agent.invoke_async(MULTI_CALC_PROBLEM)
        except (MaxToolCallsReachedError, EventLoopException):
            pass

        assert limiter.tool_call_count >= limiter.tool_iter_count, (
            f"tool_call_count ({limiter.tool_call_count}) should be >= tool_iter_count ({limiter.tool_iter_count})"
        )


class TestToolLimiterBothLimits:
    """Integration tests with both limits set."""

    async def test_iter_limit_fires_before_call_limit(self, fresh_model):
        """When iter limit is tighter, it should fire first."""
        limiter = ToolLimiter(max_tool_iters=1, max_tool_calls=100)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)) as exc_info:
            await agent.invoke_async(SEQUENTIAL_PROBLEM)

        assert_max_iterations_reached(exc_info, expected_count=1)

    async def test_call_limit_fires_before_iter_limit(self, fresh_model):
        """When call limit is tighter, it should fire first."""
        # Allow many iterations but very few calls
        limiter = ToolLimiter(max_tool_iters=100, max_tool_calls=1)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        with pytest.raises((MaxToolCallsReachedError, EventLoopException)) as exc_info:
            await agent.invoke_async(SEQUENTIAL_PROBLEM)

        assert_max_calls_reached(exc_info, expected_count=1)
