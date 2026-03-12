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

"""Chat template tests against real HuggingFace tokenizers.

This file is the source of truth for verifying that `message_separator` detection
and `tokenize_prompt_messages` (incremental tokenization via prefix subtraction)
work correctly across all supported model families.

Tests require network access to download tokenizers from HuggingFace on first run
(cached afterwards). Mark: ``pytest -m chat_template``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from strands_sglang import SGLangModel
from strands_sglang.client import SGLangClient

# ---------------------------------------------------------------------------
# Model registry — add new models here
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a model to test against."""

    id: str
    separator: str
    trust_remote_code: bool = False


MODELS: list[ModelSpec] = [
    # Qwen family — all use <|im_end|>\n → separator = "\n"
    ModelSpec(id="Qwen/Qwen2.5-32B-Instruct", separator="\n"),
    ModelSpec(id="Qwen/Qwen3-8B", separator="\n"),
    ModelSpec(id="Qwen/Qwen3.5-4B", separator="\n"),
    ModelSpec(id="Qwen/Qwen3-30B-A3B-Instruct-2507", separator="\n"),
    ModelSpec(id="Qwen/Qwen3-Coder-30B-A3B-Instruct", separator="\n"),
    # DeepSeek — uses <｜end▁of▁sentence｜> as both eos and stop, no separator
    ModelSpec(id="deepseek-ai/DeepSeek-V3.1", separator="", trust_remote_code=True),
    # GLM family — no end-of-turn token, no separator
    ModelSpec(id="THUDM/GLM-4.5-Air", separator=""),
    ModelSpec(id="zai-org/GLM-4.7", separator=""),
    # zai-org/GLM-5 omitted: uses custom TokenizersBackend not loadable by transformers
    # MiniMax — uses [e~[ as eos, separator = "\n"
    # ModelSpec(id="MiniMaxAI/MiniMax-M2.5", separator="\n"),
    # Kimi — eos_token=[EOS] but template uses <|im_end|>, separator = ""
    ModelSpec(id="moonshotai/Kimi-K2.5", separator="", trust_remote_code=True),
    ModelSpec(id="moonshotai/Kimi-K2-Thinking", separator="", trust_remote_code=True),
]

MODEL_IDS = [spec.id for spec in MODELS]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tokenizers() -> dict[str, Any]:
    """Load and cache all tokenizers once per session.

    Models that fail to load (e.g. missing `tiktoken`) are stored as the
    exception string so individual tests can skip gracefully.
    """
    from transformers import AutoTokenizer

    loaded: dict[str, Any] = {}
    for spec in MODELS:
        try:
            loaded[spec.id] = AutoTokenizer.from_pretrained(spec.id, trust_remote_code=spec.trust_remote_code)
        except Exception as e:
            loaded[spec.id] = f"Cannot load tokenizer: {e}"
    return loaded


@pytest.fixture
def client() -> SGLangClient:
    return SGLangClient(base_url="http://localhost:30000")


def _get_spec(model_id: str) -> ModelSpec:
    return next(s for s in MODELS if s.id == model_id)


def _get_tokenizer(tokenizers: dict[str, Any], model_id: str) -> Any:
    """Get tokenizer for model_id, skipping if it failed to load."""
    tok = tokenizers[model_id]
    if isinstance(tok, str):
        pytest.skip(tok)
    return tok


def _make_model(client: SGLangClient, tokenizer: Any) -> SGLangModel:
    """Create an SGLangModel with real tokenizer, overriding is_multimodal."""
    model = SGLangModel(client=client, tokenizer=tokenizer)
    model.__dict__["is_multimodal"] = False
    return model


# ---------------------------------------------------------------------------
# message_separator
# ---------------------------------------------------------------------------


@pytest.mark.chat_template
class TestMessageSeparator:
    """Verify message_separator auto-detection for each model family."""

    @pytest.mark.parametrize("model_id", MODEL_IDS, ids=MODEL_IDS)
    def test_separator(self, model_id: str, client: SGLangClient, tokenizers: dict[str, Any]) -> None:
        spec = _get_spec(model_id)
        tokenizer = _get_tokenizer(tokenizers, model_id)
        model = _make_model(client, tokenizer)
        assert model.message_separator == spec.separator, (
            f"{model_id}: expected separator {spec.separator!r}, got {model.message_separator!r}"
        )


# ---------------------------------------------------------------------------
# tokenize_prompt_messages — prefix subtraction correctness
# ---------------------------------------------------------------------------

# Conversations for testing incremental tokenization
_FIRST_TURN = [{"role": "user", "content": [{"text": "What is 2+2?"}]}]

_MULTI_TURN = [
    {"role": "user", "content": [{"text": "What is 2+2?"}]},
    {"role": "assistant", "content": [{"text": "2+2 equals 4."}]},
    {"role": "user", "content": [{"text": "And 3+3?"}]},
]

_WITH_TOOL_RESULT = [
    {"role": "user", "content": [{"text": "What is 2+2?"}]},
    {
        "role": "assistant",
        "content": [
            {"text": "Let me calculate."},
            {"toolUse": {"toolUseId": "call_0001", "name": "calculator", "input": {"expr": "2+2"}}},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "call_0001",
                    "status": "success",
                    "content": [{"text": "4"}],
                }
            }
        ],
    },
]

SYSTEM_PROMPT = "You are a helpful assistant."


@pytest.mark.chat_template
class TestTokenizePromptMessages:
    """Verify tokenize_prompt_messages correctness for each model.

    The key invariant for incremental tokenization: the incremental text
    (message_separator + prefix-subtracted text) must be a suffix of the
    full conversation text. This ensures the SGLang input, when concatenated
    with previous prompt + response tokens, produces the correct full prompt.
    """

    @pytest.mark.parametrize("model_id", MODEL_IDS, ids=MODEL_IDS)
    def test_first_call_matches_full_tokenization(
        self, model_id: str, client: SGLangClient, tokenizers: dict[str, Any]
    ) -> None:
        """First call to tokenize_prompt_messages should match direct encode of apply_chat_template."""
        tokenizer = _get_tokenizer(tokenizers, model_id)
        model = _make_model(client, tokenizer)

        tokens = model.tokenize_prompt_messages(_FIRST_TURN, system_prompt=SYSTEM_PROMPT)

        # Direct tokenization for comparison
        hf_messages = model.format_messages(_FIRST_TURN, system_prompt=SYSTEM_PROMPT)
        prompt = model.tokenizer.apply_chat_template(
            tokenize=False, conversation=hf_messages, add_generation_prompt=True
        )
        expected = list(tokenizer.encode(prompt, add_special_tokens=False))

        assert tokens == expected

    @pytest.mark.parametrize("model_id", MODEL_IDS, ids=MODEL_IDS)
    def test_incremental_text_is_suffix_of_full(
        self, model_id: str, client: SGLangClient, tokenizers: dict[str, Any]
    ) -> None:
        """Incremental text (separator + subtracted) must be a suffix of the full conversation.

        After first generation, message_count = len(first_msgs) + 1. The incremental
        call picks up messages[message_count:] = [user2]. The produced text should
        match the tail of the full conversation text.
        """
        tokenizer = _get_tokenizer(tokenizers, model_id)
        model = _make_model(client, tokenizer)

        # Set up state as if first turn already happened
        first_tokens = model.tokenize_prompt_messages(_FIRST_TURN, system_prompt=SYSTEM_PROMPT)
        model.token_manager.add_prompt(first_tokens)
        model.token_manager.add_response([0])  # dummy response
        model.message_count = len(_FIRST_TURN) + 1  # +1 for assistant response

        # Compute incremental text (reproduce tokenize_prompt_messages logic, pre-encoding)
        new_messages = _MULTI_TURN[model.message_count :]
        new_hf = model.format_messages(model.sort_tool_results(new_messages))
        fake_hf = model.format_messages(
            [
                {"role": "system", "content": [{"text": "FAKE SYSTEM PROMPT"}]},
                {"role": "user", "content": [{"text": "FAKE USER MESSAGE"}]},
            ]
        )
        full_prompt = model.tokenizer.apply_chat_template(
            tokenize=False, conversation=fake_hf + new_hf, add_generation_prompt=True
        )
        prefix_prompt = model.tokenizer.apply_chat_template(
            tokenize=False, conversation=fake_hf, add_generation_prompt=False
        )
        incremental_text = model.message_separator + full_prompt[len(prefix_prompt) :]

        # Full conversation text
        hf_all = model.format_messages(_MULTI_TURN, system_prompt=SYSTEM_PROMPT)
        full_text = model.tokenizer.apply_chat_template(tokenize=False, conversation=hf_all, add_generation_prompt=True)

        assert full_text.endswith(incremental_text), (
            f"{model_id}: incremental text is not a suffix of full conversation.\n"
            f"  full ends with: {full_text[-80:]!r}\n"
            f"  incremental:    {incremental_text!r}"
        )

    @pytest.mark.parametrize(
        "model_id",
        MODEL_IDS,
        ids=MODEL_IDS,
    )
    def test_incremental_tool_result_is_suffix_of_full(
        self, model_id: str, client: SGLangClient, tokenizers: dict[str, Any]
    ) -> None:
        """Incremental text after tool use must be a suffix of the full conversation.

        Known limitation: MiniMax-M2.5's template validates that tool messages must follow
        an assistant with tool_calls. The fake prefix [sys, user] doesn't satisfy this.
        """
        if model_id == "MiniMaxAI/MiniMax-M2.5":
            pytest.xfail("MiniMax template rejects tool result without preceding assistant tool_call")
        tokenizer = _get_tokenizer(tokenizers, model_id)
        model = _make_model(client, tokenizer)

        # Set up state as if first turn (user + assistant with tool call) already happened
        first_tokens = model.tokenize_prompt_messages(_WITH_TOOL_RESULT[:1], system_prompt=SYSTEM_PROMPT)
        model.token_manager.add_prompt(first_tokens)
        model.token_manager.add_response([0])  # dummy response
        model.message_count = len(_WITH_TOOL_RESULT[:1]) + 1  # +1 for assistant response

        # Compute incremental text
        new_messages = _WITH_TOOL_RESULT[model.message_count :]
        new_hf = model.format_messages(model.sort_tool_results(new_messages))
        fake_hf = model.format_messages(
            [
                {"role": "system", "content": [{"text": "FAKE SYSTEM PROMPT"}]},
                {"role": "user", "content": [{"text": "FAKE USER MESSAGE"}]},
            ]
        )
        full_prompt = model.tokenizer.apply_chat_template(
            tokenize=False, conversation=fake_hf + new_hf, add_generation_prompt=True
        )
        prefix_prompt = model.tokenizer.apply_chat_template(
            tokenize=False, conversation=fake_hf, add_generation_prompt=False
        )
        incremental_text = model.message_separator + full_prompt[len(prefix_prompt) :]

        # Full conversation text
        hf_all = model.format_messages(_WITH_TOOL_RESULT, system_prompt=SYSTEM_PROMPT)
        full_text = model.tokenizer.apply_chat_template(tokenize=False, conversation=hf_all, add_generation_prompt=True)

        assert full_text.endswith(incremental_text), (
            f"{model_id}: tool result incremental text is not a suffix of full conversation.\n"
            f"  full ends with: {full_text[-80:]!r}\n"
            f"  incremental:    {incremental_text!r}"
        )
