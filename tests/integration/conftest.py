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

"""Shared fixtures for integration tests.

All tests in this directory are automatically marked as integration tests
and require a running SGLang server.

Configuration (priority: CLI > env var > default):
    pytest --sglang-base-url=http://localhost:8000 --sglang-model-id=Qwen/Qwen3-4B-Instruct-2507
    SGLANG_BASE_URL=http://... SGLANG_MODEL_ID=... pytest tests/integration/
"""

import pytest
from transformers import AutoTokenizer

from strands_sglang import SGLangModel
from strands_sglang.tool_parser import HermesToolCallParser

# Mark all tests in this directory as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def sglang_base_url(request):
    """Get SGLang server URL from pytest config."""
    return request.config.getoption("--sglang-base-url")


@pytest.fixture(scope="session")
def sglang_model_id(request):
    """Get model ID from pytest config."""
    return request.config.getoption("--sglang-model-id")


@pytest.fixture(scope="module")
def tokenizer(sglang_model_id):
    """Load tokenizer for the configured model."""
    return AutoTokenizer.from_pretrained(sglang_model_id)


@pytest.fixture
def model(tokenizer, sglang_base_url, sglang_model_id):
    """Create fresh SGLangModel for each test (perfect isolation)."""
    return SGLangModel(
        tokenizer=tokenizer,
        tool_call_parser=HermesToolCallParser(),
        base_url=sglang_base_url,
        model_id=sglang_model_id,
        params={"max_new_tokens": 32768},
    )


@pytest.fixture
def calculator_tool():
    """Sample calculator tool spec for testing."""
    return {
        "name": "calculator",
        "description": "Perform arithmetic calculations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    }
