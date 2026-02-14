"""Unit tests for ToolLimiter.

Tests the limiter by directly feeding it MessageAddedEvent objects,
without needing a real agent or server.
"""

from unittest.mock import Mock

import pytest
from strands.hooks.events import MessageAddedEvent

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
        """max_tool_iters=None (with max_tool_calls also None) never raises."""
        limiter = ToolLimiter(max_tool_iters=None)
        for _ in range(100):
            _simulate_iteration(limiter)
        # Both limits None -> early return, counters not updated
        assert limiter.tool_iter_count == 0


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
        """max_tool_calls=None (with max_tool_iters also None) never raises."""
        limiter = ToolLimiter(max_tool_calls=None)
        for _ in range(50):
            _simulate_iteration(limiter, parallel_calls=3)
        # Both limits None -> early return, counters not updated
        assert limiter.tool_call_count == 0

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
        limiter = ToolLimiter()
        for _ in range(100):
            _simulate_iteration(limiter, parallel_calls=3)
        # Counters are NOT updated when both limits are None (early return)
        assert limiter.tool_iter_count == 0
        assert limiter.tool_call_count == 0


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

    def test_string_content_ignored(self):
        """Messages with string content (not list) should be ignored."""
        limiter = ToolLimiter(max_tool_iters=0)
        limiter.tool_iter_count = 5
        event = MessageAddedEvent(agent=_MOCK_AGENT, message={"role": "user", "content": "just a string"})
        limiter._on_message_added(event)  # should not raise

    def test_missing_content_ignored(self):
        limiter = ToolLimiter(max_tool_iters=0)
        limiter.tool_iter_count = 5
        event = MessageAddedEvent(agent=_MOCK_AGENT, message={"role": "user"})
        limiter._on_message_added(event)  # should not raise

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
