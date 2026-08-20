"""Math agent example with TITO (Token-In/Token-Out) for RL training.

This example demonstrates:
1. Setting up SGLangModel with a HuggingFace tokenizer
2. Creating a math agent with calculator tool
3. Single-turn and multi-turn conversations
4. Accessing TITO data (tokens, masks, logprobs) for RL training

Requirements:
    - SGLang server running: python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Thinking-2507 --port 30000

Usage:
    python examples/math_agent.py
"""

import asyncio
import json
import os

from strands import Agent, tool
from transformers import AutoTokenizer

from strands_sglang import SGLangModel
from strands_sglang.client import SGLangClient
from strands_sglang.tool_parsers import HermesToolParser, QwenXMLToolParser


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression.

    Args:
        expression: An arithmetic expression over numbers, e.g. "7 * 13" or "2.50 * 3 + 1".
    """
    allowed = set("0123456789.+-*/() ")
    if not expression or set(expression) - allowed:
        return f"Unsupported expression: {expression!r}"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 — alphabet is restricted above
    except Exception as e:
        return f"Could not evaluate {expression!r}: {e}"


async def main():
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------

    # Create SGLangModel with token-level trajectory tracking support
    client = SGLangClient(base_url=os.environ.get("SGLANG_BASE_URL", "http://localhost:30000"))
    model_info = await client.model_info()
    tokenizer = AutoTokenizer.from_pretrained(model_info["model_path"])
    tool_parser = QwenXMLToolParser() if model_info["model_path"].startswith("Qwen/Qwen3.5") else HermesToolParser()
    model = SGLangModel(
        client=client,
        tokenizer=tokenizer,
        tool_parser=tool_parser,
        sampling_params={"max_new_tokens": 16384},  # Limit response length
    )

    # -------------------------------------------------------------------------
    # 2. Math 500 Example
    # -------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Math 500 Example")
    print("=" * 60)

    # Reset for new episode
    model.reset()

    # Create agent with calculator tool
    agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt="You are a helpful math assistant. You must use the calculator tool for computations.",
        callback_handler=None,  # Disable print callback for cleaner output
    )

    # Invoke agent
    math_500_problem = r"Compute: $1-2+3-4+5- \dots +99-100$."
    print(f"\n[Input Problem]: {math_500_problem}")
    await agent.invoke_async(math_500_problem)
    print(f"\n[Output Trajectory]: {json.dumps(agent.messages, indent=2)}")
    if model.rollout:
        # Token trajectory
        print(f"[Output Tokens - Decoded]: {tokenizer.decode(model.rollout.token_ids)}")

    # -------------------------------------------------------------------------
    # 3. Access TITO Data
    # -------------------------------------------------------------------------

    print("\n" + "-" * 40)
    print("TITO Data (for RL training)")
    print("-" * 40)

    # Token trajectory
    token_ids = model.rollout.token_ids
    print(f"Total tokens: {len(token_ids)}")

    # Output mask (True = model output, for loss computation)
    output_mask = model.rollout.loss_mask
    n_output = sum(output_mask)
    n_prompt = len(output_mask) - n_output
    print(f"Prompt tokens: {n_prompt} (loss_mask=False)")
    print(f"Response tokens: {n_output} (loss_mask=True)")

    # Log probabilities
    logprobs = model.rollout.logprobs
    output_logprobs = [lp for lp, mask in zip(logprobs, output_mask, strict=False) if mask and lp is not None]
    if output_logprobs:
        avg_logprob = sum(output_logprobs) / len(output_logprobs)
        print(f"Average output logprob: {avg_logprob:.4f}")

    # Segment info
    segment_info = model.rollout.segment_info
    print(f"Segments: {len(segment_info)} (Note: Segment 0 includes the system prompt and the user input)")
    for i, (is_output, length) in enumerate(segment_info):
        seg_type = "Response" if is_output else "Prompt"
        print(f"  Segment {i}: {seg_type} ({length} tokens)")


if __name__ == "__main__":
    asyncio.run(main())
