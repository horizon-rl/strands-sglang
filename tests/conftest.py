"""Root pytest configuration for strands_sglang tests.

Test Structure:
    tests/unit/        - Unit tests (no external dependencies)
    tests/integration/ - Integration tests (require SGLang server)

Running Tests:
    pytest tests/unit/                    # Unit tests only
    pytest tests/integration/ -v          # Integration tests only
    pytest tests/                         # All tests

Configuration:
    pytest tests/integration/ --sglang-base-url=http://localhost:30000

    Or via environment variable:
    SGLANG_BASE_URL=http://localhost:30000 pytest tests/integration/

Model ID and tokenizer path are auto-detected from the server.
"""

import os


def pytest_addoption(parser):
    """Add command-line options for SGLang configuration."""
    parser.addoption(
        "--sglang-base-url",
        action="store",
        default=os.environ.get("SGLANG_BASE_URL", "http://localhost:30000"),
        help="SGLang server URL (default: http://localhost:30000 or SGLANG_BASE_URL env var)",
    )
    parser.addoption(
        "--tool-parser",
        action="store",
        default=os.environ.get("TOOL_PARSER", "hermes"),
        help="Tool parser name: hermes, qwen_xml, glm, kimi_k2 (default: hermes or TOOL_PARSER env var)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests requiring a running SGLang server",
    )
