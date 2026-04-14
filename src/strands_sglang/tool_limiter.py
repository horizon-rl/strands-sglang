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

"""Strands hook for limiting tool usage within a single agent invocation."""

import logging
from typing import Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent, MessageAddedEvent

logger = logging.getLogger(__name__)


class MaxToolIterationsReachedError(Exception):
    """Raised when the `max_tool_iters` limit is reached.

    Notes:
        Raised after iteration completes, ensuring a clean trajectory without truncation.
    """


class MaxToolCallsReachedError(Exception):
    """Raised when the `max_tool_calls` limit is reached.

    Notes:
        Raised after iteration completes, ensuring a clean trajectory without truncation.
    """


class ToolLimiter(HookProvider):
    """Hook to enforce tool iteration and/or tool call limits on agent tool loops.

    Notes:
        - An "iteration" is one cycle of: model generates tool call(s) -> tool(s) execute -> result(s) returned.
        - Multiple parallel tool calls in one model response count as a single iteration but as individual calls.
        - The limiter raises after the iteration completes (on tool result), ensuring a clean trajectory
          without requiring token truncation.

    Example:
        >>> limiter = ToolLimiter(max_tool_iters=5)
        >>> agent = Agent(model=model, tools=[...], hooks=[limiter])
        >>> try:
        ...     result = agent.invoke("solve this problem")
        ... except MaxToolIterationsReachedError:
        ...     # Trajectory is clean - contains exactly 5 complete iterations
        ...     print(f"Stopped after {limiter.tool_iter_count} iterations")
    """

    def __init__(
        self,
        max_tool_iters: int | None = None,
        max_tool_calls: int | None = None,
        max_parallel_tool_calls: int | None = None,
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
        """
        self.max_tool_iters = max_tool_iters
        self.max_tool_calls = max_tool_calls
        self.max_parallel_tool_calls = max_parallel_tool_calls
        self.reset()

    def reset(self) -> None:
        """Reset counters for a new invocation."""
        self.tool_iter_count = 0
        self.tool_call_count = 0
        self._parallel_call_count = 0
        self.cancelled_tool_call_count = 0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks with the strands agent."""
        registry.add_callback(MessageAddedEvent, self._on_message_added)
        registry.add_callback(BeforeToolCallEvent, self._on_before_tool_call)

    def _on_message_added(self, event: MessageAddedEvent) -> None:
        """Count iterations/calls and raise when limit exceeded.

        Notes:
            - Counts on assistant messages with `toolUse` (model requesting tools)
            - Raises on user messages with `toolResult` (iteration complete)
        """
        message = event.message
        content = message["content"]

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

        # Check limit when tool result arrives (iteration complete)
        elif message.get("role") == "user":
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
