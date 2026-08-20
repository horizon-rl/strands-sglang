import json

import numpy as np


async def test_single_turn_base64_and_decode(routed_experts_model):
    """Single-turn: routed_experts is a list of base64 slices, decode returns correct shape."""
    model = routed_experts_model
    messages = [{"role": "user", "content": [{"text": "Say 'hello' and nothing else."}]}]

    async for _ in model.stream(messages, system_prompt="Be brief."):
        pass

    # One base64 slice per turn
    assert isinstance(model.rollout.routed_experts, list)
    assert len(model.rollout.routed_experts) == 1

    # JSON-serializable (needed for Ray actor transport)
    json.dumps({"routed_experts": model.rollout.routed_experts})

    # decode_routed_experts returns correct shape
    if model.moe_num_layers and model.moe_top_k:
        total_tokens = len(model.rollout.token_ids)
        decoded = model.rollout.decode_routed_experts(num_layers=model.moe_num_layers, top_k=model.moe_top_k)
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

    experts_turn1 = list(model.rollout.routed_experts)
    assert len(experts_turn1) == 1

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

    experts_turn2 = list(model.rollout.routed_experts)
    # Turn 2 adds another per-turn slice
    assert len(experts_turn2) == 2
    assert experts_turn2[0] == experts_turn1[0]

    if model.moe_num_layers and model.moe_top_k:
        total_tokens = len(model.rollout.token_ids)
        decoded = model.rollout.decode_routed_experts(num_layers=model.moe_num_layers, top_k=model.moe_top_k)
        assert decoded.shape == (total_tokens - 1, model.moe_num_layers, model.moe_top_k)


async def test_reset_clears(routed_experts_model):
    """reset() clears routed experts."""
    model = routed_experts_model
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    async for _ in model.stream(messages):
        pass

    assert model.rollout.routed_experts
    model.reset()
    assert model.rollout.routed_experts == []
