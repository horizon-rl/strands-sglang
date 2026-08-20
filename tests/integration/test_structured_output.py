from pydantic import BaseModel


async def test_structured_output(model):
    """Structured output returns valid Pydantic model without updating rollout."""
    initial_token_len = len(model.rollout)

    class Verdict(BaseModel):
        is_correct: bool
        explanation: str

    prompt = [{"role": "user", "content": [{"text": "Is 2+2=5?"}]}]
    system_prompt = "You are a math validator. Answer whether the equation is correct. 2+2=4, not 5."

    result = None
    async for event in model.structured_output(Verdict, prompt, system_prompt=system_prompt):
        if "output" in event:
            result = event["output"]

    assert isinstance(result, Verdict)
    assert result.is_correct is False  # 2+2 != 5
    assert len(result.explanation) > 0
    # structured_output is inference-only — no token tracking
    assert len(model.rollout) == initial_token_len
