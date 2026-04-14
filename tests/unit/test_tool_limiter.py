# Copyright 2025-2026 Strands RL Contributors
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

"""Unit tests for ToolLimiter.

Tests the limiter by directly feeding it MessageAddedEvent objects,
without needing a real agent or server.
"""

from unittest.mock import Mock

import pytest
from strands.hooks.events import BeforeToolCallEvent, MessageAddedEvent

from strands_sglang.tool_limiter import MaxToolCallsReachedError, MaxToolIterationsReachedError, ToolLimiter

_MOCK_AGENT = Mock()


# =============================================================================
# Helpers
# =============================================================================


def _assistant_with_tools(n: int = 1) -> MessageAddedEvent:
    """Create an assistant message with n toolUse blocks."""
    content = [{"toolUse": {"toolUseId": f"tool-{i}", "name": "calc", "input": {}}} for i in range(n)]
    return MessageAddedEvent(agent=_MOCK_AGENT, message={"role": "assistant", "content": content})


def _tool_result() -> MessageAddedEvent:
    """Create a user message with a toolResult block."""
    return MessageAddedEvent(
        agent=_MOCK_AGENT, message={"role": "user", "content": [{"toolResult": {"toolUseId": "tool-0"}}]}
    )


def _assistant_text_only() -> MessageAddedEvent:
    """Create an assistant message with only text (no tool calls)."""
    return MessageAddedEvent(agent=_MOCK_AGENT, message={"role": "assistant", "content": [{"text": "Hello!"}]})


def _user_text_only() -> MessageAddedEvent:
    """Create a user message with only text (no tool result)."""
    return MessageAddedEvent(agent=_MOCK_AGENT, message={"role": "user", "content": [{"text": "Hi"}]})


def _before_tool_call(tool_id: str = "tool-0") -> BeforeToolCallEvent:
    """Create a BeforeToolCallEvent for testing."""
    return BeforeToolCallEvent(
        agent=_MOCK_AGENT,
        selected_tool=None,
        tool_use={"toolUseId": tool_id, "name": "calc", "input": {}},
        invocation_state={},
    )


def _simulate_iteration(limiter: ToolLimiter, parallel_calls: int = 1) -> None:
    """Simulate one complete iteration: assistant with tools -> tool result.

    Raises if a limit is hit on the tool result event.
    """
    limiter._on_message_added(_assistant_with_tools(parallel_calls))
    limiter._on_message_added(_tool_result())


# =============================================================================
# Init & Reset
# =============================================================================


class TestToolLimiterInit:
    def test_defaults_are_none(self):
        limiter = ToolLimiter()
        assert limiter.max_tool_iters is None
        assert limiter.max_tool_calls is None

    def test_counters_start_at_zero(self):
        limiter = ToolLimiter(max_tool_iters=5, max_tool_calls=10)
        assert limiter.tool_iter_count == 0
        assert limiter.tool_call_count == 0

    def test_reset_clears_counters(self):
        limiter = ToolLimiter(max_tool_iters=10)
        limiter.tool_iter_count = 3
        limiter.tool_call_count = 7
        limiter.reset()
        assert limiter.tool_iter_count == 0
        assert limiter.tool_call_count == 0


# =============================================================================
# max_tool_iters
# =============================================================================


class TestMaxToolIters:
    def test_raises_after_limit(self):
        """With max_tool_iters=2: iter 1 completes, iter 2 tool result raises (2 >= 2)."""
        limiter = ToolLimiter(max_tool_iters=2)
        _simulate_iteration(limiter)  # iter 1: count=1, result check 1 >= 2 -> ok
        limiter._on_message_added(_assistant_with_tools(1))  # iter 2: count=2
        with pytest.raises(MaxToolIterationsReachedError):
            limiter._on_message_added(_tool_result())  # result check 2 >= 2 -> raise

    def test_raises_exactly_at_limit(self):
        """Limit is checked on tool result: iter_count >= max_tool_iters."""
        limiter = ToolLimiter(max_tool_iters=1)
        # assistant: iter_count becomes 1
        limiter._on_message_added(_assistant_with_tools(1))
        # tool result: check 1 >= 1 -> raise
        with pytest.raises(MaxToolIterationsReachedError):
            limiter._on_message_added(_tool_result())

    def test_allows_under_limit(self):
        limiter = ToolLimiter(max_tool_iters=5)
        _simulate_iteration(limiter)
        _simulate_iteration(limiter)
        assert limiter.tool_iter_count == 2

    def test_parallel_calls_count_as_one_iteration(self):
        limiter = ToolLimiter(max_tool_iters=1)
        # 3 parallel tool calls = 1 iteration
        limiter._on_message_added(_assistant_with_tools(3))
        assert limiter.tool_iter_count == 1
        with pytest.raises(MaxToolIterationsReachedError):
            limiter._on_message_added(_tool_result())

    def test_zero_stops_after_first_iteration(self):
        """max_tool_iters=0 raises on first tool result (1 >= 0)."""
        limiter = ToolLimiter(max_tool_iters=0)
        limiter._on_message_added(_assistant_with_tools(1))
        with pytest.raises(MaxToolIterationsReachedError):
            limiter._on_message_added(_tool_result())
        assert limiter.tool_iter_count == 1

    def test_none_means_no_limit(self):
        """max_tool_iters=None (with max_tool_calls also None) never raises but still counts."""
        limiter = ToolLimiter(max_tool_iters=None)
        for _ in range(100):
            _simulate_iteration(limiter)
        assert limiter.tool_iter_count == 100


# =============================================================================
# max_tool_calls
# =============================================================================


class TestMaxToolCalls:
    def test_raises_after_limit(self):
        """3 calls across 2 iterations, limit=2 -> raises on iter 2 result."""
        limiter = ToolLimiter(max_tool_calls=2)
        _simulate_iteration(limiter, parallel_calls=1)  # 1 call total
        limiter._on_message_added(_assistant_with_tools(2))  # 3 calls total
        with pytest.raises(MaxToolCallsReachedError):
            limiter._on_message_added(_tool_result())

    def test_raises_exactly_at_limit(self):
        limiter = ToolLimiter(max_tool_calls=1)
        limiter._on_message_added(_assistant_with_tools(1))
        with pytest.raises(MaxToolCallsReachedError):
            limiter._on_message_added(_tool_result())

    def test_parallel_calls_counted_individually(self):
        """3 parallel calls in one response should count as 3."""
        limiter = ToolLimiter(max_tool_calls=2)
        limiter._on_message_added(_assistant_with_tools(3))
        assert limiter.tool_call_count == 3
        with pytest.raises(MaxToolCallsReachedError):
            limiter._on_message_added(_tool_result())

    def test_allows_under_limit(self):
        limiter = ToolLimiter(max_tool_calls=10)
        _simulate_iteration(limiter, parallel_calls=3)  # 3 calls
        _simulate_iteration(limiter, parallel_calls=2)  # 5 calls
        assert limiter.tool_call_count == 5

    def test_none_means_no_limit(self):
        """max_tool_calls=None (with max_tool_iters also None) never raises but still counts."""
        limiter = ToolLimiter(max_tool_calls=None)
        for _ in range(50):
            _simulate_iteration(limiter, parallel_calls=3)
        assert limiter.tool_call_count == 150

    def test_zero_stops_after_first_call(self):
        """max_tool_calls=0 raises on first tool result."""
        limiter = ToolLimiter(max_tool_calls=0)
        limiter._on_message_added(_assistant_with_tools(1))
        with pytest.raises(MaxToolCallsReachedError):
            limiter._on_message_added(_tool_result())


# =============================================================================
# Both limits together
# =============================================================================


class TestBothLimits:
    def test_iter_limit_fires_first(self):
        """iter limit=1, call limit=10 -> MaxToolIterationsReachedError."""
        limiter = ToolLimiter(max_tool_iters=1, max_tool_calls=10)
        limiter._on_message_added(_assistant_with_tools(1))
        with pytest.raises(MaxToolIterationsReachedError):
            limiter._on_message_added(_tool_result())

    def test_call_limit_fires_first(self):
        """iter limit=10, call limit=2 -> MaxToolCallsReachedError on 3rd call."""
        limiter = ToolLimiter(max_tool_iters=10, max_tool_calls=2)
        _simulate_iteration(limiter, parallel_calls=1)  # 1 call
        limiter._on_message_added(_assistant_with_tools(2))  # 3 calls total
        with pytest.raises(MaxToolCallsReachedError):
            limiter._on_message_added(_tool_result())
        # iter_count is 2 which is under limit=10, so it's the call limit that fires
        assert limiter.tool_iter_count == 2
        assert limiter.tool_call_count == 3

    def test_iter_checked_before_calls(self):
        """When both limits are hit simultaneously, iter limit takes precedence."""
        limiter = ToolLimiter(max_tool_iters=1, max_tool_calls=1)
        limiter._on_message_added(_assistant_with_tools(1))
        # Both iter_count=1 >= 1 and call_count=1 >= 1, but iter is checked first
        with pytest.raises(MaxToolIterationsReachedError):
            limiter._on_message_added(_tool_result())

    def test_only_iters_set_ignores_calls(self):
        """max_tool_calls=None should not crash when only iters is set."""
        limiter = ToolLimiter(max_tool_iters=2, max_tool_calls=None)
        _simulate_iteration(limiter, parallel_calls=5)  # 5 calls, 1 iter
        # iter 2: assistant makes 5 calls, then tool result triggers check
        limiter._on_message_added(_assistant_with_tools(5))  # 10 calls, 2 iters
        # Should raise on iter limit, not crash on None call comparison
        with pytest.raises(MaxToolIterationsReachedError):
            limiter._on_message_added(_tool_result())

    def test_only_calls_set_ignores_iters(self):
        """max_tool_iters=None should not crash when only calls is set."""
        limiter = ToolLimiter(max_tool_iters=None, max_tool_calls=3)
        _simulate_iteration(limiter, parallel_calls=2)  # 2 calls, 1 iter
        limiter._on_message_added(_assistant_with_tools(2))  # 4 calls, 2 iters
        with pytest.raises(MaxToolCallsReachedError):
            limiter._on_message_added(_tool_result())


# =============================================================================
# No limits (both None)
# =============================================================================


class TestNoLimits:
    def test_both_none_never_raises(self):
        """Both limits None: never raises, but counters are still accurate."""
        limiter = ToolLimiter()
        for _ in range(100):
            _simulate_iteration(limiter, parallel_calls=3)
        assert limiter.tool_iter_count == 100
        assert limiter.tool_call_count == 300


# =============================================================================
# Message types that should be ignored
# =============================================================================


class TestIgnoredMessages:
    def test_text_only_assistant_not_counted(self):
        limiter = ToolLimiter(max_tool_iters=1)
        limiter._on_message_added(_assistant_text_only())
        assert limiter.tool_iter_count == 0
        assert limiter.tool_call_count == 0

    def test_text_only_user_not_checked(self):
        """User text message should not trigger limit check."""
        limiter = ToolLimiter(max_tool_iters=0)
        # Force count above limit
        limiter.tool_iter_count = 5
        # Text-only user message should NOT raise
        limiter._on_message_added(_user_text_only())

    def test_assistant_with_mixed_content(self):
        """Text + toolUse in same message: counts as 1 iteration with 1 call."""
        limiter = ToolLimiter(max_tool_iters=5, max_tool_calls=5)
        event = MessageAddedEvent(
            agent=_MOCK_AGENT,
            message={
                "role": "assistant",
                "content": [
                    {"text": "Let me calculate that."},
                    {"toolUse": {"toolUseId": "t1", "name": "calc", "input": {}}},
                ],
            },
        )
        limiter._on_message_added(event)
        assert limiter.tool_iter_count == 1
        assert limiter.tool_call_count == 1


# =============================================================================
# max_parallel_tool_calls
# =============================================================================


class TestMaxParallelToolCalls:
    def test_first_n_calls_proceed(self):
        """First N calls within limit should not be cancelled."""
        limiter = ToolLimiter(max_parallel_tool_calls=3)
        limiter._on_message_added(_assistant_with_tools(3))
        for i in range(3):
            event = _before_tool_call(f"tool-{i}")
            limiter._on_before_tool_call(event)
            assert event.cancel_tool is False

    def test_excess_calls_cancelled(self):
        """Calls beyond the limit should be cancelled with error message."""
        limiter = ToolLimiter(max_parallel_tool_calls=2)
        limiter._on_message_added(_assistant_with_tools(4))
        # First 2 proceed
        for i in range(2):
            event = _before_tool_call(f"tool-{i}")
            limiter._on_before_tool_call(event)
            assert event.cancel_tool is False
        # 3rd and 4th cancelled
        for i in range(2, 4):
            event = _before_tool_call(f"tool-{i}")
            limiter._on_before_tool_call(event)
            assert "Max parallel tool calls (2) reached" in event.cancel_tool

    def test_cancelled_tool_call_count(self):
        """cancelled_tool_call_count should track the number of cancelled calls."""
        limiter = ToolLimiter(max_parallel_tool_calls=1)
        limiter._on_message_added(_assistant_with_tools(3))
        for i in range(3):
            limiter._on_before_tool_call(_before_tool_call(f"tool-{i}"))
        assert limiter.cancelled_tool_call_count == 2

    def test_counter_resets_on_new_turn(self):
        """Parallel call counter resets when a new assistant message with tools arrives."""
        limiter = ToolLimiter(max_parallel_tool_calls=1)

        # Turn 1: 1 allowed, 1 cancelled
        limiter._on_message_added(_assistant_with_tools(2))
        event1 = _before_tool_call("tool-0")
        limiter._on_before_tool_call(event1)
        assert event1.cancel_tool is False
        event2 = _before_tool_call("tool-1")
        limiter._on_before_tool_call(event2)
        assert event2.cancel_tool is not False

        # Complete iteration
        limiter._on_message_added(_tool_result())

        # Turn 2: counter resets, first call should proceed
        limiter._on_message_added(_assistant_with_tools(1))
        event3 = _before_tool_call("tool-2")
        limiter._on_before_tool_call(event3)
        assert event3.cancel_tool is False

    def test_none_means_no_limit(self):
        """max_parallel_tool_calls=None should never cancel."""
        limiter = ToolLimiter(max_parallel_tool_calls=None)
        limiter._on_message_added(_assistant_with_tools(100))
        for i in range(100):
            event = _before_tool_call(f"tool-{i}")
            limiter._on_before_tool_call(event)
            assert event.cancel_tool is False
        assert limiter.cancelled_tool_call_count == 0

    def test_zero_cancels_all(self):
        """max_parallel_tool_calls=0 cancels every tool call."""
        limiter = ToolLimiter(max_parallel_tool_calls=0)
        limiter._on_message_added(_assistant_with_tools(2))
        for i in range(2):
            event = _before_tool_call(f"tool-{i}")
            limiter._on_before_tool_call(event)
            assert "Max parallel tool calls (0) reached" in event.cancel_tool
        assert limiter.cancelled_tool_call_count == 2

    def test_with_max_tool_iters(self):
        """Per-turn limit works alongside max_tool_iters."""
        limiter = ToolLimiter(max_tool_iters=2, max_parallel_tool_calls=1)

        # Turn 1: 1 allowed, 1 cancelled
        limiter._on_message_added(_assistant_with_tools(2))
        event1 = _before_tool_call("tool-0")
        limiter._on_before_tool_call(event1)
        assert event1.cancel_tool is False
        event2 = _before_tool_call("tool-1")
        limiter._on_before_tool_call(event2)
        assert event2.cancel_tool is not False
        limiter._on_message_added(_tool_result())  # iter 1 complete

        # Turn 2: iter limit should still fire
        limiter._on_message_added(_assistant_with_tools(1))
        with pytest.raises(MaxToolIterationsReachedError):
            limiter._on_message_added(_tool_result())

    def test_with_max_tool_calls(self):
        """Per-turn limit works alongside max_tool_calls."""
        limiter = ToolLimiter(max_tool_calls=4, max_parallel_tool_calls=2)

        # Turn 1: assistant requests 3, parallel limit allows 2, 1 cancelled
        # tool_call_count becomes 3 (counted from assistant message)
        limiter._on_message_added(_assistant_with_tools(3))
        for i in range(2):
            event = _before_tool_call(f"tool-{i}")
            limiter._on_before_tool_call(event)
            assert event.cancel_tool is False
        event = _before_tool_call("tool-2")
        limiter._on_before_tool_call(event)
        assert event.cancel_tool is not False
        limiter._on_message_added(_tool_result())  # tool_call_count=3 < 4, ok

        # Turn 2: assistant requests 2, total becomes 5 >= 4, should fire
        limiter._on_message_added(_assistant_with_tools(2))
        with pytest.raises(MaxToolCallsReachedError):
            limiter._on_message_added(_tool_result())

    def test_init_default_is_none(self):
        limiter = ToolLimiter()
        assert limiter.max_parallel_tool_calls is None

    def test_reset_clears_parallel_counters(self):
        limiter = ToolLimiter(max_parallel_tool_calls=1)
        limiter._parallel_call_count = 5
        limiter.cancelled_tool_call_count = 3
        limiter.reset()
        assert limiter._parallel_call_count == 0
        assert limiter.cancelled_tool_call_count == 0
