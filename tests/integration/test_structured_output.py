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

"""Integration tests for structured output via SGLang's json_schema constrained decoding."""

from pydantic import BaseModel


async def test_structured_output(model):
    """Structured output returns valid Pydantic model without updating token_manager."""
    initial_token_len = len(model.token_manager)

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
    assert len(model.token_manager) == initial_token_len
