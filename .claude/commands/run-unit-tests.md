Run unit tests in an isolated temporary venv created with uv (mimics CI).

Steps:
1. Create a temporary venv at `/tmp/strands-sglang-test-venv` using `uv venv /tmp/strands-sglang-test-venv --python 3.12 -q`
2. Install the package with dev dependencies: `uv pip install -e . --group dev --python /tmp/strands-sglang-test-venv/bin/python -q`
3. Run linting: `/tmp/strands-sglang-test-venv/bin/ruff check src/ && /tmp/strands-sglang-test-venv/bin/ruff format --check src/`
4. Run unit tests: `/tmp/strands-sglang-test-venv/bin/python -m pytest tests/unit/ -v --tb=short $ARGUMENTS`

IMPORTANT: Never use the active shell's python/pytest — it may point to a different venv. Always use the temporary venv's python.
