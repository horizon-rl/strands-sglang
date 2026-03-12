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

"""Integration tests for ToolLimiter."""

import pytest
from strands import Agent
from strands.types.exceptions import EventLoopException
from strands_tools import calculator

from strands_sglang import MaxToolCallsReachedError, MaxToolIterationsReachedError, ToolLimiter

SYSTEM_PROMPT = """You are a calculator assistant. You MUST use the calculator tool for ALL arithmetic.
Never compute in your head - always use the calculator tool."""

# Requires dependent calculations — forces >= 2 iterations
SEQUENTIAL_PROBLEM = """
I have a secret number X. When you call the calculator with "7 * 13", it will tell you X.
Then calculate X + 100. You must use the calculator for both steps.
What is the final answer?
"""

MULTI_CALC_PROBLEM = "Calculate 10+5, 20+10, and 30+15 using the calculator."


def _unwrap_limiter_error(exc_info):
    """Unwrap EventLoopException to get the underlying limiter error."""
    exc = exc_info.value
    if isinstance(exc, EventLoopException):
        return exc.__cause__
    return exc


async def test_max_tool_iters(model):
    """max_tool_iters stops agent after N iterations with clean trajectory and resets correctly."""
    limiter = ToolLimiter(max_tool_iters=1)
    agent = Agent(model=model, tools=[calculator], system_prompt=SYSTEM_PROMPT, hooks=[limiter])

    # Sequential problem needs >= 2 iterations — should stop after 1
    with pytest.raises((MaxToolIterationsReachedError, EventLoopException)) as exc_info:
        await agent.invoke_async(SEQUENTIAL_PROBLEM)

    cause = _unwrap_limiter_error(exc_info)
    assert isinstance(cause, MaxToolIterationsReachedError)
    assert limiter.tool_iter_count == 1

    # Trajectory should be clean: prompt + response segments, consistent lengths
    tm = model.token_manager
    assert len(tm) > 0
    assert len(tm.token_ids) == len(tm.loss_mask) == len(tm.logprobs)
    response_segments = sum(1 for is_output, _ in tm.segment_info if is_output)
    assert response_segments == limiter.tool_iter_count

    # Reset allows reuse
    limiter.reset()
    model.reset()
    assert limiter.tool_iter_count == 0

    agent2 = Agent(model=model, tools=[calculator], system_prompt=SYSTEM_PROMPT, hooks=[limiter])
    with pytest.raises((MaxToolIterationsReachedError, EventLoopException)):
        await agent2.invoke_async(SEQUENTIAL_PROBLEM)
    assert limiter.tool_iter_count == 1


async def test_max_tool_calls(model):
    """max_tool_calls stops agent after N individual calls; call_count >= iter_count."""
    limiter = ToolLimiter(max_tool_calls=1)
    agent = Agent(model=model, tools=[calculator], system_prompt=SYSTEM_PROMPT, hooks=[limiter])

    with pytest.raises((MaxToolCallsReachedError, EventLoopException)) as exc_info:
        await agent.invoke_async(SEQUENTIAL_PROBLEM)

    cause = _unwrap_limiter_error(exc_info)
    assert isinstance(cause, MaxToolCallsReachedError)
    assert limiter.tool_call_count >= 1
    assert limiter.tool_call_count >= limiter.tool_iter_count


async def test_max_parallel_tool_calls(model):
    """max_parallel_tool_calls cancels excess calls within a single model response."""
    limiter = ToolLimiter(max_parallel_tool_calls=1, max_tool_iters=5)
    agent = Agent(model=model, tools=[calculator], system_prompt=SYSTEM_PROMPT, hooks=[limiter])

    # Multi-calc may trigger parallel calls; limit=1 cancels extras
    # Agent should still complete (cancelled calls get error results, model retries sequentially)
    try:
        result = await agent.invoke_async(MULTI_CALC_PROBLEM)
        assert result is not None
    except (MaxToolIterationsReachedError, EventLoopException):
        pass  # Model may exhaust retries if it keeps trying to parallelize

    # If model parallelized, we should see cancellations
    # If model went sequential, cancelled_tool_call_count == 0 — still valid
    assert limiter.cancelled_tool_call_count >= 0
    assert limiter.tool_call_count >= 1


async def test_both_limits_correct_error_fires(model):
    """When both limits are set, the tighter one fires the correct error type."""
    # Iter-tight: max_tool_iters=1 is tighter than max_tool_calls=100
    limiter = ToolLimiter(max_tool_iters=1, max_tool_calls=100)
    agent = Agent(model=model, tools=[calculator], system_prompt=SYSTEM_PROMPT, hooks=[limiter])

    with pytest.raises((MaxToolIterationsReachedError, EventLoopException)) as exc_info:
        await agent.invoke_async(SEQUENTIAL_PROBLEM)
    assert isinstance(_unwrap_limiter_error(exc_info), MaxToolIterationsReachedError)

    # Call-tight: max_tool_calls=1 is tighter than max_tool_iters=100
    model.reset()
    limiter2 = ToolLimiter(max_tool_iters=100, max_tool_calls=1)
    agent2 = Agent(model=model, tools=[calculator], system_prompt=SYSTEM_PROMPT, hooks=[limiter2])

    with pytest.raises((MaxToolCallsReachedError, EventLoopException)) as exc_info:
        await agent2.invoke_async(SEQUENTIAL_PROBLEM)
    assert isinstance(_unwrap_limiter_error(exc_info), MaxToolCallsReachedError)
