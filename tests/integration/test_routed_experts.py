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

"""Integration tests for MoE routed experts capture (R3)."""

import json

import numpy as np

from strands_sglang import decode_routed_experts


async def test_single_turn_base64_and_decode(routed_experts_model):
    """Single-turn: routed_experts is base64, decode_routed_experts returns correct shape."""
    model = routed_experts_model
    messages = [{"role": "user", "content": [{"text": "Say 'hello' and nothing else."}]}]

    async for _ in model.stream(messages, system_prompt="Be brief."):
        pass

    # Raw value is a base64 string
    assert isinstance(model.routed_experts, str)

    # JSON-serializable (needed for Ray actor transport)
    json.dumps({"routed_experts": model.routed_experts})

    # decode_routed_experts returns correct shape
    if model.moe_num_layers and model.moe_top_k:
        total_tokens = len(model.token_manager.token_ids)
        decoded = decode_routed_experts(
            model.routed_experts, seq_len=total_tokens, num_layers=model.moe_num_layers, top_k=model.moe_top_k
        )
        assert decoded.shape == (total_tokens - 1, model.moe_num_layers, model.moe_top_k)
        assert decoded.dtype == np.int32


async def test_multi_turn_agent_with_tools(routed_experts_model, calculator_tool):
    """Multi-turn tool use updates routed experts across turns."""
    model = routed_experts_model
    system_prompt = "You are a calculator. Use the calculator tool for ALL math."

    # Turn 1: model should produce a tool call
    messages = [{"role": "user", "content": [{"text": "What is 5 * 8?"}]}]
    async for _ in model.stream(messages, tool_specs=[calculator_tool], system_prompt=system_prompt):
        pass

    experts_turn1 = model.routed_experts
    assert experts_turn1 is not None

    # Inject tool result for turn 2
    messages.append(
        {
            "role": "assistant",
            "content": [
                {"text": '<tool_call>\n{"name": "calculator", "arguments": {"expression": "5 * 8"}}\n</tool_call>'},
                {"toolUse": {"toolUseId": "call_1", "name": "calculator", "input": {"expression": "5 * 8"}}},
            ],
        }
    )
    messages.append({"role": "user", "content": [{"toolResult": {"toolUseId": "call_1", "content": [{"text": "40"}]}}]})

    # Turn 2: model should produce final answer
    async for _ in model.stream(messages, tool_specs=[calculator_tool], system_prompt=system_prompt):
        pass

    experts_turn2 = model.routed_experts
    assert experts_turn2 is not None
    # Turn 2 covers more tokens (full sequence), so the base64 string should be longer
    assert len(experts_turn2) > len(experts_turn1)

    if model.moe_num_layers and model.moe_top_k:
        total_tokens = len(model.token_manager.token_ids)
        decoded = decode_routed_experts(
            experts_turn2, seq_len=total_tokens, num_layers=model.moe_num_layers, top_k=model.moe_top_k
        )
        assert decoded.shape == (total_tokens - 1, model.moe_num_layers, model.moe_top_k)


async def test_reset_clears(routed_experts_model):
    """reset() clears routed experts."""
    model = routed_experts_model
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    async for _ in model.stream(messages):
        pass

    assert model.routed_experts is not None
    model.reset()
    assert model.routed_experts is None
