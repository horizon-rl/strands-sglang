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

"""Tests for DeepSeek-V3.2 encoding attachment and apply_chat_template.

Uses DeepSeek's own encoding module and golden test cases from:
https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/encoding
"""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from strands_sglang.utils import attach_dsv32_encoding

FIXTURES_DIR = Path(__file__).parent


@pytest.fixture
def encoding_module():
    """Load the real encoding_dsv32 module from fixtures."""
    spec = importlib.util.spec_from_file_location("encoding_dsv32", FIXTURES_DIR / "encoding" / "encoding_dsv32.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tokenizer():
    """Mock tokenizer with name_or_path pointing to the fixtures directory.

    attach_dsv32_encoding looks for {name_or_path}/encoding/encoding_dsv32.py,
    so we point to the parent of the encoding/ directory.
    """
    tok = MagicMock()
    tok.name_or_path = str(FIXTURES_DIR)
    return tok


@pytest.fixture
def patched_tokenizer(tokenizer):
    """Tokenizer with apply_chat_template replaced by the real encoding module."""
    attach_dsv32_encoding(tokenizer)
    return tokenizer


@pytest.fixture
def test_data():
    """Load DeepSeek's golden test input."""
    with open(FIXTURES_DIR / "test_input.json") as f:
        return json.load(f)


@pytest.fixture
def golden_output():
    """Load DeepSeek's golden expected output."""
    with open(FIXTURES_DIR / "test_output.txt") as f:
        return f.read().strip()


class TestFullConversation:
    """Tests using real encoding module and DeepSeek's golden test cases."""

    def test_matches_golden_output(self, patched_tokenizer, test_data, golden_output):
        """Full conversation with tools matches DeepSeek's golden output."""
        result = patched_tokenizer.apply_chat_template(
            conversation=test_data["messages"],
            tools=test_data["tools"],
            enable_thinking=True,
        )
        assert result == golden_output

    def test_wrapper_matches_direct_encode_messages(self, patched_tokenizer, encoding_module, test_data):
        """Wrapper output identical to calling encode_messages() directly."""
        # Direct call (how DeepSeek's own test does it)
        messages = [msg.copy() for msg in test_data["messages"]]
        messages[0]["tools"] = test_data["tools"]
        direct = encoding_module.encode_messages(messages, thinking_mode="thinking")

        # Via our wrapper
        wrapped = patched_tokenizer.apply_chat_template(
            conversation=test_data["messages"],
            tools=test_data["tools"],
            enable_thinking=True,
        )

        assert wrapped == direct

    def test_tools_attached_to_existing_system_message(self, patched_tokenizer, encoding_module, test_data):
        """Tools merged into existing system message produce correct output."""
        # Build expected: system message with tools attached
        messages = [msg.copy() for msg in test_data["messages"]]
        messages[0]["tools"] = test_data["tools"]
        expected = encoding_module.encode_messages(messages, thinking_mode="thinking")

        # Our wrapper handles tool attachment
        result = patched_tokenizer.apply_chat_template(
            conversation=test_data["messages"],
            tools=test_data["tools"],
            enable_thinking=True,
        )

        assert result == expected

    def test_synthetic_system_message_when_none_exists(self, patched_tokenizer, encoding_module, test_data):
        """Synthetic system message created when tools given but no system message."""
        # Remove system message from conversation
        messages_no_system = [m for m in test_data["messages"] if m.get("role") != "system"]

        # Our wrapper should insert a synthetic system message with tools
        result = patched_tokenizer.apply_chat_template(
            conversation=messages_no_system,
            tools=test_data["tools"],
            enable_thinking=True,
        )

        # Direct: manually add synthetic system message
        synthetic = [{"role": "system", "content": "", "tools": test_data["tools"]}] + messages_no_system
        expected = encoding_module.encode_messages(synthetic, thinking_mode="thinking")

        assert result == expected

    def test_chat_mode(self, patched_tokenizer, encoding_module, test_data):
        """enable_thinking=False passes thinking_mode='chat'."""
        messages = [msg.copy() for msg in test_data["messages"]]
        messages[0]["tools"] = test_data["tools"]
        expected = encoding_module.encode_messages(messages, thinking_mode="chat")

        result = patched_tokenizer.apply_chat_template(
            conversation=test_data["messages"],
            tools=test_data["tools"],
            enable_thinking=False,
        )

        assert result == expected

    def test_input_messages_not_mutated(self, patched_tokenizer, test_data):
        """Original messages list and dicts are not modified."""
        messages = test_data["messages"]
        original_len = len(messages)
        original_system = dict(messages[0])

        patched_tokenizer.apply_chat_template(
            conversation=messages,
            tools=test_data["tools"],
            enable_thinking=True,
        )

        assert len(messages) == original_len
        assert messages[0] == original_system
        assert "tools" not in messages[0]


class TestIncrementalPath:
    """Tests for incremental tokenization with DSV32.

    `tokenize_prompt_messages` builds [fake_system, fake_user] + new_messages and calls
    `apply_chat_template` twice (full and prefix) for subtraction. The DSV32 wrapper
    extracts `reasoning_content` and `tool_calls` from inline assistant content so that
    `encode_messages` can handle the full conversation.
    """

    # Realistic assistant content after format_messages: raw model output with
    # thinking tags and DSML tool calls inline (no separate fields).
    ASSISTANT_WITH_TOOL_CALL = {
        "role": "assistant",
        "content": (
            "<think>I need to calculate this.</think>"
            "Let me compute that."
            "\n\n<｜DSML｜function_calls>"
            '\n<｜DSML｜invoke name="calculator">'
            '\n<｜DSML｜parameter name="expression" string="true">5 * 8</｜DSML｜parameter>'
            "\n</｜DSML｜invoke>"
            "\n</｜DSML｜function_calls>"
        ),
    }

    FAKE_PREFIX = [
        {"role": "system", "content": "FAKE SYSTEM PROMPT"},
        {"role": "user", "content": "FAKE USER MESSAGE"},
    ]

    def test_prefix_subtraction_single_tool_result(self, patched_tokenizer):
        """Prefix subtraction yields tool result portion for single tool result."""
        prefix_msgs = self.FAKE_PREFIX + [self.ASSISTANT_WITH_TOOL_CALL]
        full_msgs = prefix_msgs + [{"role": "tool", "content": "42"}]

        prefix = patched_tokenizer.apply_chat_template(prefix_msgs, enable_thinking=True)
        full = patched_tokenizer.apply_chat_template(full_msgs, enable_thinking=True)

        incremental = full[len(prefix) :]
        assert "<function_results>" in incremental
        assert "<result>42</result>" in incremental
        assert incremental.endswith("<think>")

    def test_prefix_subtraction_multiple_tool_results(self, patched_tokenizer):
        """Prefix subtraction yields all tool results (one per tool call)."""
        # Assistant with TWO tool calls
        assistant = {
            "role": "assistant",
            "content": (
                "<think>I need two lookups.</think>"
                "\n\n<｜DSML｜function_calls>"
                '\n<｜DSML｜invoke name="search">'
                '\n<｜DSML｜parameter name="q" string="true">cats</｜DSML｜parameter>'
                "\n</｜DSML｜invoke>"
                '\n<｜DSML｜invoke name="search">'
                '\n<｜DSML｜parameter name="q" string="true">dogs</｜DSML｜parameter>'
                "\n</｜DSML｜invoke>"
                "\n</｜DSML｜function_calls>"
            ),
        }
        prefix_msgs = self.FAKE_PREFIX + [assistant]
        full_msgs = prefix_msgs + [
            {"role": "tool", "content": "result1"},
            {"role": "tool", "content": "result2"},
        ]

        prefix = patched_tokenizer.apply_chat_template(prefix_msgs, enable_thinking=True)
        full = patched_tokenizer.apply_chat_template(full_msgs, enable_thinking=True)

        incremental = full[len(prefix) :]
        assert "<result>result1</result>" in incremental
        assert "<result>result2</result>" in incremental

    def test_format_matches_module_templates(self, patched_tokenizer, encoding_module):
        """Tool results use the module's tool_output_template."""
        prefix_msgs = self.FAKE_PREFIX + [self.ASSISTANT_WITH_TOOL_CALL]
        full_msgs = prefix_msgs + [{"role": "tool", "content": "test content"}]

        result = patched_tokenizer.apply_chat_template(full_msgs, enable_thinking=True)

        expected_tag = encoding_module.tool_output_template.format(content="test content")
        assert expected_tag in result

    def test_thinking_mode_appends_think_start(self, patched_tokenizer):
        """Thinking mode ends with <think> for generation prompt after tool results."""
        full_msgs = self.FAKE_PREFIX + [self.ASSISTANT_WITH_TOOL_CALL, {"role": "tool", "content": "ok"}]
        result = patched_tokenizer.apply_chat_template(full_msgs, enable_thinking=True)
        assert result.endswith("<think>")

    def test_chat_mode_appends_think_end(self, patched_tokenizer):
        """Chat mode ends with </think> after tool results."""
        full_msgs = self.FAKE_PREFIX + [self.ASSISTANT_WITH_TOOL_CALL, {"role": "tool", "content": "ok"}]
        result = patched_tokenizer.apply_chat_template(full_msgs, enable_thinking=False)
        assert result.endswith("</think>")

    def test_assistant_without_think_tags(self, patched_tokenizer):
        """Assistant content without <think> tags is handled (chat mode generation)."""
        assistant = {"role": "assistant", "content": "Let me help."}
        full_msgs = self.FAKE_PREFIX + [assistant]
        # Should not crash — assistant without reasoning_content is valid in chat mode
        result = patched_tokenizer.apply_chat_template(full_msgs, enable_thinking=False)
        assert "Let me help." in result

    def test_extract_reasoning_and_tool_calls(self, patched_tokenizer, encoding_module):
        """_to_dsv32_format correctly extracts reasoning_content and tool_calls."""
        prefix_msgs = self.FAKE_PREFIX + [self.ASSISTANT_WITH_TOOL_CALL]

        # Should produce same result as manually constructed OpenAI-format messages
        openai_assistant = {
            "role": "assistant",
            "reasoning_content": "I need to calculate this.",
            "content": "Let me compute that.",
            "tool_calls": [
                {"type": "function", "function": {"name": "calculator", "arguments": '{"expression": "5 * 8"}'}}
            ],
        }
        expected_msgs = self.FAKE_PREFIX + [openai_assistant]

        result = patched_tokenizer.apply_chat_template(prefix_msgs, enable_thinking=True)
        expected = encoding_module.encode_messages(expected_msgs, thinking_mode="thinking")
        assert result == expected

    def test_mixed_roles_full_path(self, patched_tokenizer, encoding_module):
        """Non-assistant messages go through encode_messages unchanged."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]

        result = patched_tokenizer.apply_chat_template(messages, enable_thinking=True)
        expected = encoding_module.encode_messages(messages, thinking_mode="thinking")

        assert result == expected


class TestAttachDsv32Encoding:
    """Tests for attach_dsv32_encoding() glue logic."""

    def test_replaces_apply_chat_template(self, patched_tokenizer):
        """apply_chat_template is replaced with a callable."""
        assert callable(patched_tokenizer.apply_chat_template)

    def test_module_load_failure_raises(self):
        """Raises when the encoding module cannot be loaded."""
        tok = MagicMock()
        tok.name_or_path = "/nonexistent/path"

        with pytest.raises((AttributeError, FileNotFoundError)):
            attach_dsv32_encoding(tok)

    def test_kwargs_passed_through(self, patched_tokenizer):
        """Extra kwargs don't crash (silently ignored)."""
        result = patched_tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": "hi"}],
            tokenize=False,
        )
        assert result  # non-empty


class TestGetTokenizerAutoDetect:
    """Tests for get_tokenizer() auto-detection of DeepSeek-V3.2."""

    def setup_method(self):
        from strands_sglang.utils import get_tokenizer

        get_tokenizer.cache_clear()

    @patch("strands_sglang.utils.attach_dsv32_encoding")
    @patch("strands_sglang.utils.os.path.isfile", return_value=True)
    @patch("transformers.AutoTokenizer")
    def test_detects_dsv32_and_attaches(self, mock_auto, _mock_isfile, mock_attach):
        """get_tokenizer calls attach_dsv32_encoding when encoding file exists."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.name_or_path = "/models/deepseek-v32"
        mock_auto.from_pretrained.return_value = mock_tokenizer

        from strands_sglang.utils import get_tokenizer

        result = get_tokenizer("/models/deepseek-v32")

        mock_attach.assert_called_once_with(mock_tokenizer)
        assert result is mock_tokenizer

    @patch("strands_sglang.utils.attach_dsv32_encoding")
    @patch("strands_sglang.utils.os.path.isfile", return_value=False)
    @patch("transformers.AutoTokenizer")
    def test_skips_non_dsv32(self, mock_auto, _mock_isfile, mock_attach):
        """get_tokenizer skips attachment when encoding file doesn't exist."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.name_or_path = "/models/qwen3"
        mock_auto.from_pretrained.return_value = mock_tokenizer

        from strands_sglang.utils import get_tokenizer

        get_tokenizer.cache_clear()
        result = get_tokenizer("/models/qwen3")

        mock_attach.assert_not_called()
        assert result is mock_tokenizer
