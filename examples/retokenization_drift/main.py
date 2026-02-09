#!/usr/bin/env python3
"""Retokenization drift example: Why TITO matters for RL training.

Demonstrates: `encode(decode(tokens)) != tokens`

This happens because `tokenizer.encode()` may produce different token sequences
than what the model generated during rollout. The same text can have multiple
valid tokenizations. TITO captures exact token IDs during generation.

NOTE: Drift is rare and depends on specific tokenizer edge cases. This example
uses a complex problem with extended thinking to increase the chance of triggering
drift. Even if no drift occurs, TITO is still valuable for capturing exact tokens.

Requirements:
    python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Thinking-2507 --port 30000

Usage:
    python examples/retokenization_drift/main.py
"""

import asyncio
import os

from strands import Agent
from strands_tools import calculator
from transformers import AutoTokenizer

from strands_sglang import SGLangModel
from strands_sglang.client import SGLangClient
from strands_sglang.tool_parsers import HermesToolParser


def find_drift_index(original: list[int], re_encoded: list[int]) -> int | None:
    """Find first index where tokens diverge."""
    for i, (a, b) in enumerate(zip(original, re_encoded)):
        if a != b:
            return i
    if len(original) != len(re_encoded):
        return min(len(original), len(re_encoded))
    return None


async def main():
    model_id = os.environ.get("SGLANG_MODEL_ID", "Qwen/Qwen3-4B-Thinking-2507")
    base_url = os.environ.get("SGLANG_BASE_URL", "http://localhost:30000")

    print(f"Model: {model_id}")
    print(f"Server: {base_url}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    client = SGLangClient(base_url=base_url)
    model = SGLangModel(
        tokenizer=tokenizer,
        client=client,
        tool_parser=HermesToolParser(),
        model_id=model_id,
        params={"max_new_tokens": 32768},
    )

    # Complex multi-step problem to induce extended thinking
    problem = """
    A farmer has 3 fields. Field A is 2.5 acres, Field B is 3.75 acres, Field C is 1.8 acres.

    Crop yields per acre: Wheat=$450, Corn=$380, Soybeans=$520.
    Costs per acre: Wheat=$120, Corn=$95, Soybeans=$150.

    The farmer plants wheat in Field A, corn in Field B, and soybeans in Field C.
    There's also a 15% tax on total profit.

    Calculate: (1) Revenue per field, (2) Cost per field, (3) Profit per field,
    (4) Total profit before tax, (5) Tax amount, (6) Final profit after tax.

    Think through each step very carefully, exploring multiple approaches.
    """

    print("Running complex problem to induce extended thinking...")
    model.reset()

    agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt=(
            "You are a math expert. Think as long and thoroughly as possible, "
            "exploring multiple solution paths and verifying each step. "
            "Use the calculator for all arithmetic."
        ),
        # callback_handler=None,
    )

    await agent.invoke_async(problem)

    # Check for drift
    tito_tokens = list(model.token_manager.token_ids)
    decoded = tokenizer.decode(tito_tokens)
    re_encoded = tokenizer.encode(decoded, add_special_tokens=False)

    print(f"\nTITO tokens:  {len(tito_tokens)}")
    print(f"Re-encoded:   {len(re_encoded)}")

    drift_idx = find_drift_index(tito_tokens, re_encoded)

    if drift_idx is not None:
        print(f"\n>>> DRIFT at index {drift_idx}/{len(tito_tokens)} <<<")

        # Show context around drift with both token IDs and text
        ctx = 5
        start = max(0, drift_idx - ctx)
        end = min(len(tito_tokens), drift_idx + ctx + 1)

        print(f"\nContext (indices {start}-{end - 1}):")
        print("  Original tokens:")
        for i in range(start, end):
            marker = " -->" if i == drift_idx else "    "
            print(f"  {marker} [{i}] {tito_tokens[i]:6d} -> {repr(tokenizer.decode([tito_tokens[i]]))}")

        print("  Re-encoded tokens:")
        for i in range(start, min(end, len(re_encoded))):
            marker = " -->" if i == drift_idx else "    "
            print(f"  {marker} [{i}] {re_encoded[i]:6d} -> {repr(tokenizer.decode([re_encoded[i]]))}")

        print("\nTITO captures exact tokens - use token_ids directly for RL training.")
    else:
        print("\nNo drift detected (drift is rare). TITO still captures exact tokens.")

    # Show TITO structure
    print("\n--- TITO Data ---")
    print(f"Tokens: {len(tito_tokens)}, Outputs: {sum(model.token_manager.loss_mask)}")
    for i, (is_out, length) in enumerate(model.token_manager.segment_info):
        print(f"  Seg {i}: {length:5d} {'Response' if is_out else 'Prompt'}")


if __name__ == "__main__":
    asyncio.run(main())
