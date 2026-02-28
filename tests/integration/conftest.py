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

Usage:
    pytest tests/integration/ --sglang-base-url=http://localhost:30000

The model ID is auto-detected from the server's /get_model_info endpoint.
"""

import httpx
import pytest
from transformers import AutoProcessor, AutoTokenizer, ProcessorMixin

from strands_sglang import SGLangModel
from strands_sglang.client import SGLangClient
from strands_sglang.tool_parsers import HermesToolParser

# Mark all tests in this directory as integration tests
pytestmark = pytest.mark.integration


def _get_server_info(base_url: str, timeout: float = 5.0) -> dict:
    """Check server health and get model info.

    Returns:
        Server info dict with 'model_path' and 'tokenizer_path'.

    Raises:
        pytest.exit: If server is not reachable or unhealthy.
    """
    # Health check
    try:
        response = httpx.get(f"{base_url}/health", timeout=timeout)
        if response.status_code != 200:
            pytest.exit(f"SGLang server unhealthy: status {response.status_code}", returncode=1)
    except httpx.ConnectError:
        pytest.exit(f"Cannot connect to {base_url} - is the server running?", returncode=1)
    except httpx.TimeoutException:
        pytest.exit(f"Connection to {base_url} timed out", returncode=1)
    except Exception as e:
        pytest.exit(f"Health check failed: {e}", returncode=1)

    # Get model info
    try:
        response = httpx.get(f"{base_url}/get_model_info", timeout=timeout)
        return response.json()
    except Exception as e:
        pytest.exit(f"Failed to get model info: {e}", returncode=1)


@pytest.fixture(scope="session")
def sglang_server_info(request):
    """Get server info (includes health check and model detection)."""
    base_url = request.config.getoption("--sglang-base-url")
    info = _get_server_info(base_url)
    info["base_url"] = base_url
    return info


@pytest.fixture(scope="session")
def sglang_base_url(sglang_server_info):
    """Get SGLang server URL."""
    return sglang_server_info["base_url"]


@pytest.fixture(scope="module")
def tokenizer(sglang_server_info):
    """Load tokenizer for the configured model."""
    tokenizer_path = sglang_server_info.get("tokenizer_path") or sglang_server_info["model_path"]
    return AutoTokenizer.from_pretrained(tokenizer_path)


@pytest.fixture
async def model(tokenizer, sglang_base_url):
    """Create fresh SGLangModel for each test (perfect isolation)."""
    client = SGLangClient(base_url=sglang_base_url)
    yield SGLangModel(
        client=client,
        tokenizer=tokenizer,
        tool_parser=HermesToolParser(),
        sampling_params={"max_new_tokens": 32768},
    )
    await client.close()


@pytest.fixture(scope="module")
def processor(sglang_server_info):
    """Load processor for VLM models; None for text-only models."""
    model_path = sglang_server_info.get("tokenizer_path") or sglang_server_info["model_path"]
    try:
        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except (ImportError, ValueError):
        # ImportError: missing torch/torchvision for VLM processors
        # ValueError: processor config not found for text-only models
        return None
    if isinstance(proc, ProcessorMixin):
        return proc
    return None


@pytest.fixture
async def vlm_model(processor, sglang_base_url):
    """Create fresh SGLangModel with processor for VLM tests."""
    if processor is None:
        pytest.skip("Server is running a text-only model")
    client = SGLangClient(base_url=sglang_base_url)
    yield SGLangModel(
        client=client,
        processor=processor,
        tool_parser=HermesToolParser(),
        sampling_params={"max_new_tokens": 32768},
    )
    await client.close()


@pytest.fixture
def calculator_tool():
    """Sample calculator tool spec for testing."""
    return {
        "name": "calculator",
        "description": "Perform arithmetic calculations",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate",
                    }
                },
                "required": ["expression"],
            }
        },
    }
