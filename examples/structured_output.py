# Structured output against a running SGLang server.
#
#   python -m sglang.launch_server --model-path Qwen/Qwen3-4B --port 30000
#   python examples/structured_output.py

import asyncio
import os

from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from strands_sglang import SGLangModel
from strands_sglang.client import SGLangClient


# Define structured output schemas
class CodeReview(BaseModel):
    """LLM-as-Judge schema for code review."""

    score: int = Field(ge=1, le=5, description="Quality score from 1 (poor) to 5 (excellent)")
    correctness: bool = Field(description="Whether the code is functionally correct")
    issues: list[str] = Field(description="List of identified issues")
    suggestion: str = Field(description="Brief improvement suggestion")


class MathVerdict(BaseModel):
    """Schema for math problem verification."""

    is_correct: bool = Field(description="Whether the answer is mathematically correct")
    expected_answer: str = Field(description="The correct answer")
    explanation: str = Field(description="Brief explanation of the verification")


async def main():
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    client = SGLangClient(base_url=os.environ.get("SGLANG_BASE_URL", "http://localhost:30000"))
    model_info = await client.model_info()
    tokenizer = AutoTokenizer.from_pretrained(model_info["model_path"])
    model = SGLangModel(
        client=client,
        tokenizer=tokenizer,
        sampling_params={"max_new_tokens": 1024},
    )

    # -------------------------------------------------------------------------
    # Example 1: Code Review (LLM-as-Judge)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 1: Code Review")
    print("=" * 60)

    code_to_review = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

    prompt = [{"role": "user", "content": [{"text": f"Review this code:\n```python{code_to_review}```"}]}]
    system_prompt = "You are a code reviewer. Evaluate code quality, correctness, and suggest improvements."

    async for event in model.structured_output(CodeReview, prompt, system_prompt=system_prompt):
        if "output" in event:
            review: CodeReview = event["output"]

    print(f"Structured output: {review.model_dump_json(indent=2)}")

    # -------------------------------------------------------------------------
    # Example 2: Math Verification
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: Math Verification")
    print("=" * 60)

    prompt = [{"role": "user", "content": [{"text": "Verify: The sum of first 10 positive integers is 55."}]}]
    system_prompt = "You are a math verifier. Check if the given mathematical statement is correct."

    async for event in model.structured_output(MathVerdict, prompt, system_prompt=system_prompt):
        if "output" in event:
            verdict: MathVerdict = event["output"]

    print(f"Structured output: {verdict.model_dump_json(indent=2)}")

    # -------------------------------------------------------------------------
    # Verify: No TITO tracking
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("TITO Status")
    print("-" * 40)
    print(f"Rollout length: {len(model.rollout)} (should be 0 - no TITO for structured output)")


if __name__ == "__main__":
    asyncio.run(main())
