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

"""Strands hook for limiting tool iterations within a single agent invocation.

A "tool iteration" is one model response that requests tools. Multiple parallel
tool calls in one response count as a single iteration.
"""

import logging

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import MessageAddedEvent

logger = logging.getLogger(__name__)


class MaxToolIterationsReachedError(Exception):
    """Raised when `max_tool_iterations` limit is reached.

    A "tool iteration" is a single model response that requests one or more tools.
    Multiple parallel tool calls in one response count as one iteration.
    """

    pass


class ToolIterationLimiter(HookProvider):
    """Hook to enforce `max_tool_iterations` limit.

    Counts the number of model responses that request tool calls (not individual tools).
    A single response may request multiple parallel tools - this counts as one iteration.
    This preserves context integrity by not breaking in the middle of tool execution.

    When the limit is exceeded, raises MaxToolIterationsReachedError and records the
    truncation point so the trajectory can be cleanly truncated.

    Example:
        >>> limiter = ToolIterationLimiter(max_tool_iterations=5)
        >>> agent = Agent(model=model, tools=[...], hooks=[limiter])
        >>> try:
        ...     agent.invoke("solve this problem")
        ... except MaxToolIterationsReachedError:
        ...     print(f"Stopped after {limiter.iteration_count} iterations")
    """

    def __init__(self, max_tool_iterations: int | None = None):
        """Initialize the limiter.

        Args:
            max_tool_iterations: Maximum number of tool iterations allowed.
                None means no limit.
        """
        self.max_tool_iterations = max_tool_iterations
        self.reset()

    def reset(self) -> None:
        """Reset counters for a new invocation."""
        self.iteration_count = 0
        self.truncate_after_index: int | None = None

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register hooks with the strands agent."""
        registry.add_callback(MessageAddedEvent, self._on_message_added)

    def _on_message_added(self, event: MessageAddedEvent) -> None:
        """Count tool iterations and raise when limit exceeded."""
        if self.max_tool_iterations is None:
            return

        message = event.message
        # Count assistant messages that request tools (not individual tool calls)
        if message.get("role") == "assistant":
            content = message.get("content", [])
            if isinstance(content, list) and any(c.get("toolUse") for c in content):
                logger.debug(f"Tool iteration detected: {message}")
                self.iteration_count += 1
                if self.iteration_count > self.max_tool_iterations:
                    # Record where to truncate (before this message)
                    self.truncate_after_index = len(event.agent.messages) - 1
                    raise MaxToolIterationsReachedError(
                        f"Max tool iterations ({self.max_tool_iterations}) reached"
                    )
