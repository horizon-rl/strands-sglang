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

import json

import numpy as np
import pytest
from strands import Agent


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
        decoded = await model.decode_routed_experts(num_layers=model.moe_num_layers, top_k=model.moe_top_k)
        total_tokens = len(model.token_manager.token_ids)
        assert decoded.shape == (total_tokens - 1, model.moe_num_layers, model.moe_top_k)
        assert decoded.dtype == np.int32


async def test_multi_turn_agent_with_tools(routed_experts_model):
    """Multi-turn agent loop: routed_experts covers the full trajectory."""
    model = routed_experts_model

    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        return str(eval(expression))  # noqa: S307

    agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt="Use the calculator tool for ALL math. Be brief.",
    )
    await agent.invoke_async("What is 137 * 251?")

    assert model.routed_experts is not None
    total_tokens = len(model.token_manager.token_ids)
    assert total_tokens > 10  # sanity: multi-turn should produce many tokens

    if model.moe_num_layers and model.moe_top_k:
        decoded = await model.decode_routed_experts(num_layers=model.moe_num_layers, top_k=model.moe_top_k)
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


async def test_decode_fails_when_none(routed_experts_model):
    """decode_routed_experts raises when routed_experts is None."""
    model = routed_experts_model
    model.reset()
    with pytest.raises(AssertionError, match="routed_experts is None"):
        await model.decode_routed_experts(num_layers=1, top_k=1)
