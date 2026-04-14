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

"""Chat template tests against real HuggingFace tokenizers.

Source of truth for verifying `message_separator` detection and incremental
tokenization (prefix subtraction) across all supported model families.

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


def _get_tokenizer(tokenizers: dict[str, Any], model_id: str) -> Any:
    """Get tokenizer for model_id, skipping if it failed to load."""
    tok = tokenizers[model_id]
    if isinstance(tok, str):
        pytest.skip(tok)
    return tok


def _make_model(client: SGLangClient, tokenizer: Any) -> SGLangModel:
    """Create an SGLangModel with real tokenizer."""
    client._is_multimodal = False
    return SGLangModel(client=client, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# Conversations for testing incremental tokenization
# ---------------------------------------------------------------------------

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


def _compute_incremental_text(model: SGLangModel, full_messages: list, message_count: int) -> str:
    """Compute incremental text via prefix subtraction (reproduces tokenize_prompt_messages logic)."""
    new_messages = full_messages[message_count:]
    new_hf = model.format_messages(model.sort_tool_results(new_messages))
    fake_hf = model.format_messages(
        [
            {"role": "system", "content": [{"text": "FAKE SYSTEM PROMPT"}]},
            {"role": "user", "content": [{"text": "FAKE USER MESSAGE"}]},
        ]
    )
    full_prompt = model.tokenizer.apply_chat_template(
        conversation=fake_hf + new_hf, add_generation_prompt=True, **model._chat_template_kwargs
    )
    prefix_prompt = model.tokenizer.apply_chat_template(
        conversation=fake_hf, add_generation_prompt=False, **model._chat_template_kwargs
    )
    return model.message_separator + full_prompt[len(prefix_prompt) :]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.chat_template
class TestChatTemplate:
    """Verify separator detection and incremental tokenization for each model family."""

    @pytest.mark.parametrize("model_id", MODEL_IDS, ids=MODEL_IDS)
    def test_tokenization_pipeline(self, model_id: str, client: SGLangClient, tokenizers: dict[str, Any]) -> None:
        """Separator, first-call, multi-turn, and tool-result tokenization are all correct."""
        spec = next(s for s in MODELS if s.id == model_id)
        tokenizer = _get_tokenizer(tokenizers, model_id)
        model = _make_model(client, tokenizer)

        # 1. Separator detection
        assert model.message_separator == spec.separator, (
            f"{model_id}: expected separator {spec.separator!r}, got {model.message_separator!r}"
        )

        # 2. First call matches full tokenization
        tokens = model.tokenize_prompt_messages(_FIRST_TURN, system_prompt=SYSTEM_PROMPT)
        hf_messages = model.format_messages(_FIRST_TURN, system_prompt=SYSTEM_PROMPT)
        prompt = model.tokenizer.apply_chat_template(
            conversation=hf_messages, add_generation_prompt=True, **model._chat_template_kwargs
        )
        expected = list(tokenizer.encode(prompt, add_special_tokens=False))
        assert tokens == expected

        # 3. Multi-turn incremental text is suffix of full conversation
        model.token_manager.add_prompt(tokens)
        model.token_manager.add_response([0])  # dummy response
        model.message_count = len(_FIRST_TURN) + 1  # +1 for assistant response

        incremental_text = _compute_incremental_text(model, _MULTI_TURN, model.message_count)
        hf_all = model.format_messages(_MULTI_TURN, system_prompt=SYSTEM_PROMPT)
        full_text = model.tokenizer.apply_chat_template(
            conversation=hf_all, add_generation_prompt=True, **model._chat_template_kwargs
        )
        assert full_text.endswith(incremental_text), (
            f"{model_id}: multi-turn incremental text is not a suffix of full conversation.\n"
            f"  full ends with: {full_text[-80:]!r}\n"
            f"  incremental:    {incremental_text!r}"
        )

        # 4. Tool result incremental text is suffix of full conversation
        model.token_manager.reset()
        model.message_count = 0
        first_tokens = model.tokenize_prompt_messages(_WITH_TOOL_RESULT[:1], system_prompt=SYSTEM_PROMPT)
        model.token_manager.add_prompt(first_tokens)
        model.token_manager.add_response([0])
        model.message_count = len(_WITH_TOOL_RESULT[:1]) + 1

        if model_id == "MiniMaxAI/MiniMax-M2.5":
            pytest.xfail("MiniMax template rejects tool result without preceding assistant tool_call")

        incremental_text = _compute_incremental_text(model, _WITH_TOOL_RESULT, model.message_count)
        hf_all = model.format_messages(_WITH_TOOL_RESULT, system_prompt=SYSTEM_PROMPT)
        full_text = model.tokenizer.apply_chat_template(
            conversation=hf_all, add_generation_prompt=True, **model._chat_template_kwargs
        )
        assert full_text.endswith(incremental_text), (
            f"{model_id}: tool result incremental text is not a suffix of full conversation.\n"
            f"  full ends with: {full_text[-80:]!r}\n"
            f"  incremental:    {incremental_text!r}"
        )
