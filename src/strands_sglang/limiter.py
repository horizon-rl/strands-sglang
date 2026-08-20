import logging
from typing import Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent, MessageAddedEvent

logger = logging.getLogger(__name__)


class LoopLimitReachedError(Exception):
    """Base class for all limit errors raised by `LoopLimiter`."""


class MaxToolIterationsReachedError(LoopLimitReachedError):
    """Raised after the `max_tool_iters`-th iteration finishes, so the trajectory needs no truncation."""


class MaxToolCallsReachedError(LoopLimitReachedError):
    """Raised after the iteration that reached `max_tool_calls` finishes, so nothing is truncated."""


class MaxMessagesReachedError(LoopLimitReachedError):
    """Raised when a user-role message lands (initial prompt or tool result) at `max_messages`.

    Checking only there is what makes the trajectory end on a complete message boundary.
    """


class LoopLimiter(HookProvider):
    """Hook to bound the agent loop: tool iterations, tool calls, parallel calls, and message count.

    An "iteration" is one cycle: the model emits tool call(s), they execute, results come back.
    Parallel calls in a single response are one iteration but several calls. Limits raise at
    complete message boundaries — a tool result or a user message — so a trajectory never needs
    token truncation.

    Notes:
        Counters accumulate across invocations of the same agent until `reset()`, which is what lets
        `max_messages` bound a whole multi-turn conversation rather than one invoke.

    Example:
        >>> limiter = LoopLimiter(max_tool_iters=5)
        >>> agent = Agent(model=model, tools=[...], hooks=[limiter])
        >>> try:
        ...     result = agent.invoke("solve this problem")
        ... except LoopLimitReachedError:
        ...     # Trajectory is clean - ends at a complete message boundary
        ...     print(f"Stopped after {limiter.tool_iter_count} iterations")
    """

    def __init__(
        self,
        max_tool_iters: int | None = None,
        max_tool_calls: int | None = None,
        max_parallel_tool_calls: int | None = None,
        max_messages: int | None = None,
    ):
        """Initialize the limiter.

        Args:
            max_tool_iters: Maximum number of tool iterations allowed.
                One iteration = one model response with tool calls + execution.
                Parallel tool calls count as one iteration. None means no limit.
            max_tool_calls: Maximum number of individual tool calls allowed.
                Each tool call counts individually regardless of parallelism.
                Final count may exceed this limit if the last turn has multiple
                parallel tool calls. None means no limit.
            max_parallel_tool_calls: Maximum number of parallel tool calls allowed
                per model response. Excess calls are cancelled and returned to the
                model as error results. None means no limit.
            max_messages: Maximum number of messages (all roles) in the conversation.
                Every message added by the framework is counted, but the limit is only
                checked when a user-role message lands (initial prompt or tool result)
                so the loop stops at a complete message boundary. Final count may
                slightly exceed this limit. None means no limit.
        """
        self.max_tool_iters = max_tool_iters
        self.max_tool_calls = max_tool_calls
        self.max_parallel_tool_calls = max_parallel_tool_calls
        self.max_messages = max_messages
        self.reset()

    def reset(self) -> None:
        """Reset counters for a new invocation."""
        self.tool_iter_count = 0
        self.tool_call_count = 0
        self.message_count = 0
        self._parallel_call_count = 0
        self.cancelled_tool_call_count = 0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks with the strands agent."""
        registry.add_callback(MessageAddedEvent, self._on_message_added)
        registry.add_callback(BeforeToolCallEvent, self._on_before_tool_call)

    def _on_message_added(self, event: MessageAddedEvent) -> None:
        """Count messages/iterations/calls and raise when a limit is exceeded.

        Notes:
            - Counts every message toward `message_count`
            - Counts iterations/calls on assistant messages with `toolUse` (model requesting tools)
            - Raises on user messages: tool limits when a `toolResult` arrives (iteration
              complete), message limit on any user message (complete message boundary)
        """
        message = event.message
        content = message["content"]
        self.message_count += 1

        # Count when model requests tools
        if message.get("role") == "assistant":
            cur_tool_call_count = 0
            for c in content:
                if c.get("toolUse"):
                    cur_tool_call_count += 1
            if cur_tool_call_count > 0:
                self.tool_iter_count += 1
                self.tool_call_count += cur_tool_call_count
                self._parallel_call_count = 0  # Reset parallel call counter for new model response
                logger.debug(
                    "Iteration %d started (%d tool call(s), %d total calls)",
                    self.tool_iter_count,
                    cur_tool_call_count,
                    self.tool_call_count,
                )

        # Check limits when a user message arrives (complete message boundary)
        elif message.get("role") == "user":
            # Tool limits are checked on tool results (iteration complete)
            if any(c.get("toolResult") for c in content):
                if self.max_tool_iters is not None and self.tool_iter_count >= self.max_tool_iters:
                    logger.debug("Max tool iterations (%d) reached, stopping", self.max_tool_iters)
                    raise MaxToolIterationsReachedError(
                        f"Max tool iterations ({self.max_tool_iters}) reached"
                        " (parallel tool calls count as one iteration)"
                    )
                if self.max_tool_calls is not None and self.tool_call_count >= self.max_tool_calls:
                    logger.debug("Max tool calls (%d) reached, stopping", self.max_tool_calls)
                    raise MaxToolCallsReachedError(
                        f"Max tool calls ({self.max_tool_calls}) reached"
                        " (parallel tool calls count as individual calls)"
                    )
            # Message limit is checked on any user message, so it also fires before the
            # first generation of a new invocation when the budget is already exhausted
            if self.max_messages is not None and self.message_count >= self.max_messages:
                logger.debug("Max messages (%d) reached, stopping", self.max_messages)
                raise MaxMessagesReachedError(f"Max messages ({self.max_messages}) reached")

    def _on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Cancel excess tool calls when parallel call limit is reached."""
        if self.max_parallel_tool_calls is None:
            return

        self._parallel_call_count += 1
        if self._parallel_call_count > self.max_parallel_tool_calls:
            self.cancelled_tool_call_count += 1
            event.cancel_tool = (
                f"Max parallel tool calls ({self.max_parallel_tool_calls}) reached. This tool call was not executed."
            )
            logger.debug(
                "Cancelled tool call (parallel count %d, limit %s)",
                self._parallel_call_count,
                self.max_parallel_tool_calls,
            )
