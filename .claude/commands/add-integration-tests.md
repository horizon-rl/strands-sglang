Write integration tests for a given module or feature.

The user provides a target as $ARGUMENTS (e.g., a module name, class, or feature). If not provided, ask.

## Conventions

Follow the existing test style in `tests/integration/`:

- **License header**: Every `.py` file must start with the Apache 2.0 license header (copy from any existing source file)
- **File naming**: `test_sglang_<focus>.py` for model pipeline tests (e.g., `test_sglang_text.py`, `test_sglang_agent.py`), `test_<feature>.py` for standalone features (e.g., `test_tool_limiter.py`)
- **Location**: `tests/integration/`
- **Async tests**: Use `async def test_*` directly — `asyncio_mode = "auto"` is configured
- **Shared fixtures**: `conftest.py` provides `sglang_base_url`, `sglang_server_info`, `tokenizer`, `model`, `vlm_model`, `calculator_tool` — use these, don't redefine them
- **Assertions**: Use plain `assert`

## Design Principles

Integration tests are **single-comprehensive** — each test exercises a complete scenario end-to-end, not a single assertion. This is not unit testing.

- **One test per scenario, not per assertion**: A single test should set up a realistic scenario, run it, and verify all relevant aspects (response, token trajectory, segments, logprobs) within that scenario.
- **Test real behavior, not mechanics**: Don't test Python guarantees (dict operations, array lengths matching trivially). Test that the pipeline produces correct results with a real server.
- **Use shared fixtures**: Never redefine `model` or `tokenizer` fixtures — use the ones from `conftest.py`. This ensures tests run against the actual server configuration (including VLM auto-detection).
- **Synthesize tools when needed**: Use `@tool` from strands to create lightweight synthetic tools for multi-tool or domain-specific scenarios. Don't limit tests to just `calculator`.
- **Token trajectory checks are part of the scenario**: Don't create separate "token tracking" tests. Instead, verify segments, loss_mask, and logprobs as assertions within each scenario test.
- **Avoid jargon in test code**: Use "token trajectory," "token preservation," "loss mask" — not internal shorthand.
- **Problems should be non-trivial**: Choose problems that genuinely exercise the code path (multi-step, dependent calculations, multi-tool selection) — not "2+2".

## Steps

1. Read the module source to understand the feature being tested
2. Read `tests/integration/conftest.py` for available fixtures
3. Read existing integration tests as reference for the single-comprehensive style
4. Design distinct scenarios that exercise different code paths
5. Write tests — each test is one comprehensive scenario with all relevant assertions
6. After writing, remind the user to run `/run-integration-tests` (requires a running SGLang server)
