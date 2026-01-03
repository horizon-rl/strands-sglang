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

"""Integration tests for ToolIterationLimiter.

Tests the limiter with real agent loops to verify:
1. Iteration counting is correct (one model response with tools = 1 iteration)
2. Exception is raised at the right time (after complete iterations)
3. Token trajectory is clean (no truncation needed)

Note: An "iteration" is one model response that requests tools. Multiple parallel
tool calls in one response count as a single iteration. This means if the model
generates 4 tool calls in one response, that's still 1 iteration.
"""

import pytest
from strands import Agent
from strands.types.exceptions import EventLoopException
from strands_tools import calculator

from strands_sglang import MaxToolIterationsReachedError, SGLangModel, ToolIterationLimiter
from strands_sglang.tool_parser import HermesToolCallParser


def assert_max_iterations_reached(exc_info, expected_count: int | None = None):
    """Helper to verify MaxToolIterationsReachedError was raised.

    Strands wraps hook exceptions in EventLoopException, so we check the cause.
    """
    exc = exc_info.value
    # Could be raised directly or wrapped in EventLoopException
    if isinstance(exc, EventLoopException):
        cause = exc.__cause__
        assert isinstance(cause, MaxToolIterationsReachedError), f"Expected MaxToolIterationsReachedError, got {type(cause)}"
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
def fresh_model(tokenizer, sglang_base_url, sglang_model_id):
    """Create a fresh SGLangModel for each test."""
    return SGLangModel(
        tokenizer=tokenizer,
        tool_call_parser=HermesToolCallParser(),
        base_url=sglang_base_url,
        model_id=sglang_model_id,
        params={"max_new_tokens": 16384},
    )


# =============================================================================
# Tests
# =============================================================================


class TestToolIterationLimiterBasic:
    """Basic limiter functionality tests."""

    async def test_limiter_stops_after_one_iteration(self, fresh_model):
        """Limiter with max_iterations=1 should stop after first tool-using response."""
        limiter = ToolIterationLimiter(max_iterations=1)
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
        assert limiter.iteration_count == 1
        assert_max_iterations_reached(exc_info, expected_count=1)

    async def test_limiter_allows_completion_under_limit(self, fresh_model):
        """Agent should complete normally if under the limit."""
        limiter = ToolIterationLimiter(max_iterations=10)
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
        assert limiter.iteration_count >= 1  # At least one tool use
        assert limiter.iteration_count <= 10  # Under limit

    async def test_limiter_no_limit_when_none(self, fresh_model):
        """Limiter should not limit when max_iterations is None."""
        limiter = ToolIterationLimiter(max_iterations=None)
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
        limiter = ToolIterationLimiter(max_iterations=1)
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
        assert limiter.iteration_count == 1
        assert_max_iterations_reached(exc_info, expected_count=1)


class TestToolIterationLimiterTrajectory:
    """Tests for trajectory cleanliness after limiter stops."""

    async def test_trajectory_is_clean_after_limit(self, fresh_model):
        """Token trajectory should be clean (complete iterations only)."""
        limiter = ToolIterationLimiter(max_iterations=1)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        fresh_model.reset()

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
        limiter = ToolIterationLimiter(max_iterations=1)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        fresh_model.reset()

        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)):
            await agent.invoke_async(SEQUENTIAL_PROBLEM)

        segment_info = fresh_model.token_manager.segment_info

        # Count response segments (is_output=True)
        response_segments = sum(1 for is_output, _ in segment_info if is_output)

        # Should have response segments matching iteration count
        assert response_segments == limiter.iteration_count


class TestToolIterationLimiterEdgeCases:
    """Edge case tests."""

    async def test_max_iterations_zero_stops_after_first(self, fresh_model):
        """max_iterations=0 should stop after first complete iteration.

        With the current logic:
        - Model generates toolUse -> iteration_count = 1
        - Tool result arrives -> check: 1 >= 0 -> True -> raise
        So we complete 1 iteration before stopping.
        """
        limiter = ToolIterationLimiter(max_iterations=0)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)):
            await agent.invoke_async(SIMPLE_PROBLEM)

        # With max_iterations=0, stops after first complete iteration
        assert limiter.iteration_count == 1

    async def test_limiter_reset_clears_count(self, fresh_model):
        """Limiter reset should clear iteration count."""
        limiter = ToolIterationLimiter(max_iterations=1)

        # First invocation with fresh agent
        agent1 = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)):
            await agent1.invoke_async(SEQUENTIAL_PROBLEM)

        first_count = limiter.iteration_count
        assert first_count == 1

        # Reset limiter and model
        limiter.reset()
        fresh_model.reset()

        assert limiter.iteration_count == 0

        # Second invocation with fresh agent (agent state is not shared)
        agent2 = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)):
            await agent2.invoke_async(SEQUENTIAL_PROBLEM)

        assert limiter.iteration_count == 1


class TestToolIterationLimiterSequentialProblems:
    """Tests with problems that require sequential tool use."""

    async def test_problem_completes_with_sufficient_iterations(self, fresh_model):
        """Problem should complete when given enough iterations."""
        # Allow enough iterations to complete
        limiter = ToolIterationLimiter(max_iterations=5)
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
        assert limiter.iteration_count >= 1
        assert limiter.iteration_count <= 5
        assert result is not None

    async def test_limit_stops_mid_dependent_calculation(self, fresh_model):
        """Limiter should stop dependent calculation at limit."""
        limiter = ToolIterationLimiter(max_iterations=1)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        # Should stop after first iteration, before second calculation
        with pytest.raises((MaxToolIterationsReachedError, EventLoopException)):
            await agent.invoke_async(SEQUENTIAL_PROBLEM)

        assert limiter.iteration_count == 1

    async def test_high_limit_allows_completion(self, fresh_model):
        """High iteration limit should allow problem to complete."""
        limiter = ToolIterationLimiter(max_iterations=20)
        agent = Agent(
            model=fresh_model,
            tools=[calculator],
            system_prompt=SYSTEM_PROMPT,
            hooks=[limiter],
        )

        result = await agent.invoke_async(SIMPLE_PROBLEM)

        # Should complete normally
        assert result is not None
        assert limiter.iteration_count >= 1
        assert limiter.iteration_count < 20  # Shouldn't need that many
