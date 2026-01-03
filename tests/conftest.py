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

"""Root pytest configuration for strands_sglang tests.

Test Structure:
    tests/unit/        - Unit tests (no external dependencies)
    tests/integration/ - Integration tests (require SGLang server)

Running Tests:
    pytest tests/unit/                    # Unit tests only
    pytest tests/integration/ -v          # Integration tests only
    pytest tests/                         # All tests

Configuration:
    pytest tests/integration/ --sglang-base-url=http://localhost:8000
    pytest tests/integration/ --sglang-model-id=Qwen/Qwen3-4B-Instruct-2507

    Or via environment variables:
    SGLANG_BASE_URL=http://localhost:8000 pytest tests/integration/
"""

import os


def pytest_addoption(parser):
    """Add command-line options for SGLang configuration."""
    parser.addoption(
        "--sglang-base-url",
        action="store",
        default=os.environ.get("SGLANG_BASE_URL", "http://localhost:8000"),
        help="SGLang server URL (default: http://localhost:8000 or SGLANG_BASE_URL env var)",
    )
    parser.addoption(
        "--sglang-model-id",
        action="store",
        default=os.environ.get("SGLANG_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507"),
        help="Model ID (default: Qwen/Qwen3-4B-Instruct-2507 or SGLANG_MODEL_ID env var)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests requiring a running SGLang server",
    )
