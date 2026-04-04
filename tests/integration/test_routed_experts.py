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

"""Integration tests for MoE routed experts capture (R3)."""

import numpy as np


async def test_routed_experts_single_turn(routed_experts_model):
    """Single-turn generation captures routed experts as int32 numpy array."""
    model = routed_experts_model
    messages = [{"role": "user", "content": [{"text": "Say 'hello' and nothing else."}]}]

    async for _ in model.stream(messages, system_prompt="Be brief."):
        pass

    assert model.routed_experts is not None
    assert model.routed_experts.dtype == np.int32
    assert len(model.routed_experts) > 0

    # Shape: (total_tokens - 1) * num_layers * moe_router_topk elements
    total_tokens = len(model.token_manager.token_ids)
    assert len(model.routed_experts) % (total_tokens - 1) == 0


async def test_routed_experts_multi_turn(routed_experts_model, calculator_tool):
    """Multi-turn tool use updates routed experts on each generation call."""
    model = routed_experts_model
    system_prompt = "You are a calculator. Use the calculator tool for ALL math."

    # Turn 1
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

    # Turn 2
    async for _ in model.stream(messages, tool_specs=[calculator_tool], system_prompt=system_prompt):
        pass

    experts_turn2 = model.routed_experts
    assert experts_turn2 is not None
    # Turn 2 covers more tokens (full sequence), so the array should be larger
    assert len(experts_turn2) > len(experts_turn1)


async def test_routed_experts_reset(routed_experts_model):
    """reset() clears routed experts."""
    model = routed_experts_model
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    async for _ in model.stream(messages):
        pass

    assert model.routed_experts is not None

    model.reset()
    assert model.routed_experts is None
