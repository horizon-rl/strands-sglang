Run integration tests in an isolated temporary venv created with uv.

Integration tests require a running SGLang server. Before running any commands, ask the user:

1. "What is the SGLang server base URL?" — offer these options:
   - `http://localhost:30000` (default — assumes local or SSH-tunneled server)
   - Let the user provide a custom URL

2. "Which tool parser?" — offer these options:
   - `hermes` (default — for Hermes/Qwen models)
   - `qwen_xml` (XML format for Qwen models)
   - `glm` (GLM models)

Then proceed:

1. Create a temporary venv at `/tmp/strands-sglang-test-venv` using `uv venv /tmp/strands-sglang-test-venv --python 3.12 -q`
2. Install the package with dev dependencies: `uv pip install -e ".[dev]" --python /tmp/strands-sglang-test-venv/bin/python -q`
3. Run integration tests with the confirmed URL and tool parser:
   `/tmp/strands-sglang-test-venv/bin/python -m pytest tests/integration/ -v --tb=short --sglang-base-url=<URL> --tool-parser=<PARSER> $ARGUMENTS`

VLM tests are automatically skipped if the server is running a text-only model (auto-detected via `PretrainedConfig`). No extra dependencies (torch, torchvision) are needed.

IMPORTANT: Never use the active shell's python/pytest — it may point to a different venv. Always use the temporary venv's python.
