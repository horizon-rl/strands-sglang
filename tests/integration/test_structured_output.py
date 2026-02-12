# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for structured output via constrained decoding.

Tests both:
1. Direct model.structured_output() - uses SGLang's json_schema constrained decoding
2. Agent with structured_output_model - uses tool-based approach via stream()

Fixtures (model, tokenizer) are provided by conftest.py.
"""

import warnings

from pydantic import BaseModel
from strands import Agent


class TestModelStructuredOutput:
    """Tests for model.structured_output() with constrained decoding."""

    async def test_simple_model(self, model):
        """Model returns valid structured output matching Pydantic schema."""

        class Rating(BaseModel):
            score: int
            reason: str

        prompt = [{"role": "user", "content": [{"text": "Rate the quality of 'Hello World' code. Score 1-5."}]}]

        result = None
        async for event in model.structured_output(Rating, prompt):
            if "output" in event:
                result = event["output"]

        assert result is not None
        assert isinstance(result, Rating)
        assert 1 <= result.score <= 5
        assert len(result.reason) > 0

    async def test_with_system_prompt(self, model):
        """Structured output respects system prompt."""

        class Verdict(BaseModel):
            is_correct: bool
            explanation: str

        prompt = [{"role": "user", "content": [{"text": "Is 2+2=5?"}]}]
        system_prompt = "You are a math validator. Be precise."

        result = None
        async for event in model.structured_output(Verdict, prompt, system_prompt=system_prompt):
            if "output" in event:
                result = event["output"]

        assert result is not None
        assert isinstance(result, Verdict)
        assert result.is_correct is False  # 2+2 != 5

    async def test_does_not_update_token_manager(self, model):
        """structured_output should NOT update token_manager (inference-only)."""
        initial_len = len(model.token_manager)

        class Simple(BaseModel):
            value: str

        prompt = [{"role": "user", "content": [{"text": "Say hello"}]}]
        async for _ in model.structured_output(Simple, prompt):
            pass

        # Token manager should be unchanged
        assert len(model.token_manager) == initial_len


class TestAgentStructuredOutput:
    """Tests for Agent-level structured output (tool-based approach via stream)."""

    async def test_invoke_with_structured_output_model(self, model):
        """Agent.invoke_async with structured_output_model returns parsed result."""

        class MathAnswer(BaseModel):
            answer: int
            explanation: str

        agent = Agent(model=model, system_prompt="You are a math assistant.")
        result = await agent.invoke_async("What is 10 + 5?", structured_output_model=MathAnswer)

        assert result.structured_output is not None
        assert isinstance(result.structured_output, MathAnswer)
        assert result.structured_output.answer == 15

    async def test_deprecated_structured_output_method(self, model):
        """Deprecated Agent.structured_output_async still works (calls model.structured_output)."""

        class Greeting(BaseModel):
            message: str

        agent = Agent(model=model)
        prompt = [{"role": "user", "content": [{"text": "Say hello in JSON format"}]}]

        # Should emit deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = await agent.structured_output_async(Greeting, prompt)
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()

        assert isinstance(result, Greeting)
        assert len(result.message) > 0
