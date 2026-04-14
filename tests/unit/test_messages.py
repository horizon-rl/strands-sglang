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

"""Unit tests for SGLangModel message formatting.

Regression tests compare the new direct Strands→HF conversion against the
reference OpenAI-based implementation (pre-refactor) to verify no change
in behavior.
"""

from __future__ import annotations

from typing import Any

from strands.models.openai import OpenAIModel
from strands.types.content import Messages

from strands_sglang import SGLangModel

# ---------------------------------------------------------------------------
# Reference implementation: OpenAI-based message formatting (pre-refactor)
# ---------------------------------------------------------------------------


def _ref_format_message_content(message: dict[str, Any]) -> None:
    """Reference post-processing: flatten content to first text block, delete tool_calls."""
    if "content" in message and isinstance(message["content"], list):
        text_content = ""
        for block in message["content"]:
            if "text" in block:
                text_content = block["text"]
                break
        message["content"] = text_content

    if "tool_calls" in message:
        del message["tool_calls"]


def ref_format_messages(messages: Messages, system_prompt: str | None = None) -> list[dict[str, Any]]:
    """Reference implementation: OpenAI formatter + flatten + delete tool_calls."""
    result = OpenAIModel.format_request_messages(messages=messages, system_prompt=system_prompt)
    for message in result:
        _ref_format_message_content(message)
    return result


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


class TestFormatMessagesRegression:
    """Compare new format_messages against the reference OpenAI-based implementation."""

    def test_simple_user_message(self):
        messages = [{"role": "user", "content": [{"text": "Hello, world!"}]}]
        assert SGLangModel.format_messages(messages) == ref_format_messages(messages)

    def test_system_prompt(self):
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        assert SGLangModel.format_messages(messages, "Be helpful.") == ref_format_messages(messages, "Be helpful.")

    def test_multi_turn_text_only(self):
        messages = [
            {"role": "user", "content": [{"text": "What is 2+2?"}]},
            {"role": "assistant", "content": [{"text": "4"}]},
            {"role": "user", "content": [{"text": "And 3+3?"}]},
        ]
        assert SGLangModel.format_messages(messages) == ref_format_messages(messages)

    def test_tool_result(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_001",
                            "status": "success",
                            "content": [{"text": "Result: 42"}],
                        }
                    }
                ],
            }
        ]
        new = SGLangModel.format_messages(messages)
        ref = ref_format_messages(messages)

        assert len(new) == len(ref)
        assert new[0]["role"] == ref[0]["role"] == "tool"
        assert new[0]["tool_call_id"] == ref[0]["tool_call_id"] == "call_001"
        assert new[0]["content"] == ref[0]["content"]

    def test_tool_result_with_json_content(self):
        """Tool result containing JSON data (common for structured tool outputs)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_001",
                            "status": "success",
                            "content": [{"json": {"temperature": 72, "unit": "F"}}],
                        }
                    }
                ],
            }
        ]
        new = SGLangModel.format_messages(messages)
        ref = ref_format_messages(messages)

        assert new[0]["role"] == ref[0]["role"] == "tool"
        assert new[0]["content"] == ref[0]["content"]

    def test_parallel_tool_results_separate_messages(self):
        """Multiple tool results from parallel tool calls (each in separate message)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_001",
                            "status": "success",
                            "content": [{"text": "Result A"}],
                        }
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_002",
                            "status": "success",
                            "content": [{"text": "Result B"}],
                        }
                    }
                ],
            },
        ]
        new = SGLangModel.format_messages(messages)
        ref = ref_format_messages(messages)

        assert len(new) == len(ref) == 2
        for n, r in zip(new, ref, strict=False):
            assert n["role"] == r["role"] == "tool"
            assert n["tool_call_id"] == r["tool_call_id"]
            assert n["content"] == r["content"]

    def test_parallel_tool_results_batched(self):
        """Multiple tool results batched in one Strands message (real parallel tool call format).

        Strands batches all parallel tool results into a single message:
        {"role": "user", "content": [{"toolResult": ...}, {"toolResult": ...}, ...]}.
        Each toolResult must produce its own HF message with role="tool".
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_0",
                            "status": "success",
                            "content": [{"text": "Result 0"}],
                        }
                    },
                    {
                        "toolResult": {
                            "toolUseId": "call_1",
                            "status": "success",
                            "content": [{"text": "Result 1"}],
                        }
                    },
                    {
                        "toolResult": {
                            "toolUseId": "call_2",
                            "status": "success",
                            "content": [{"text": "Result 2"}],
                        }
                    },
                ],
            },
        ]
        new = SGLangModel.format_messages(messages)
        ref = ref_format_messages(messages)

        assert len(new) == len(ref) == 3
        for n, r in zip(new, ref, strict=False):
            assert n["role"] == r["role"] == "tool"
            assert n["tool_call_id"] == r["tool_call_id"]
            assert n["content"] == r["content"]

    def test_empty_messages(self):
        assert SGLangModel.format_messages([]) == ref_format_messages([])

    def test_text_with_special_characters(self):
        """Text with newlines, unicode, and XML-like markup."""
        messages = [
            {"role": "user", "content": [{"text": "Line 1\nLine 2\n\n<b>bold</b> \u2603"}]},
        ]
        assert SGLangModel.format_messages(messages) == ref_format_messages(messages)

    def test_text_with_tool_call_markup(self):
        """Text containing tool call XML tags (raw model output)."""
        messages = [
            {
                "role": "assistant",
                "content": [{"text": 'Let me check. <tool_call>{"name": "search", "arguments": {}}</tool_call>'}],
            },
        ]
        new = SGLangModel.format_messages(messages)
        ref = ref_format_messages(messages)

        assert new[0]["content"] == ref[0]["content"]
        assert "<tool_call>" in new[0]["content"]

    def test_assistant_with_tool_use_blocks(self):
        """Assistant message with text + toolUse blocks — toolUse is skipped, text preserved."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"text": '<tool_call>{"name": "calc", "arguments": {"x": 2}}</tool_call>'},
                    {"toolUse": {"toolUseId": "call_001", "name": "calc", "input": {"x": 2}}},
                ],
            },
        ]
        new = SGLangModel.format_messages(messages)
        ref = ref_format_messages(messages)

        assert len(new) == len(ref) == 1
        assert new[0]["role"] == ref[0]["role"] == "assistant"
        assert new[0]["content"] == ref[0]["content"]
        assert "tool_calls" not in new[0]
        assert "<tool_call>" in new[0]["content"]
