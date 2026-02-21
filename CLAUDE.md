# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Strands-SGLang is an SGLang model provider for the Strands Agents SDK with Token-In/Token-Out (TITO) support for on-policy agentic reinforcement learning training. It captures exact token IDs and logprobs during generation without retokenization drift, which is critical for accurate gradient computation in RL training.

## Commands

### Setup
```bash
pip install -e ".[dev]"
pre-commit install -t pre-commit -t commit-msg
```

### Linting
```bash
ruff check src/
ruff format --check src/
```

### Testing
```bash
# Unit tests (no server needed)
pytest tests/unit/ -v

# Single test file
pytest tests/unit/test_sglang.py -v

# Single test
pytest tests/unit/test_tool_parser.py::TestHermesToolParser::test_parse_single_tool_call -v

# Unit tests with coverage
pytest tests/unit/ -v --cov=src/strands_sglang --cov-report=html

# Integration tests (requires running SGLang server)
pytest tests/integration/ -v --sglang-base-url=http://localhost:30000
# Or via env var: SGLANG_BASE_URL=http://localhost:30000 pytest tests/integration/
```

## Architecture

The package lives in `src/strands_sglang/` with 7 core modules:

**SGLangModel** (`sglang.py`) - Main entry point implementing the Strands `Model` interface. Requires `client` and `tokenizer` (all keyword-only). Formats messages using HuggingFace chat templates (`apply_chat_template()`), calls SGLang's `/generate` endpoint (non-streaming by design for RL throughput), tracks TITO trajectory, and parses tool calls. Configuration via `SGLangConfig` TypedDict (sampling_params, return_logprob, enable_thinking).

**SGLangClient** (`client.py`) - Async HTTP client using aiohttp with connection pooling and aggressive retry (60 attempts by default, aligned with SLIME RL framework). All error classification is centralized in `_classify_http_error()`, which maps HTTP responses to custom exceptions (`SGLangContextLengthError`, `SGLangThrottledError`, etc.). Non-retryable errors: 401, 403, 404, context-length 400. Uses lazy session creation to avoid aiohttp's event-loop warnings.

**Utilities** (`utils.py`) - `lru_cache`-backed factories for shared client and tokenizer instances: `get_client()`, `get_client_from_slime_args()`, `get_tokenizer()`. Ensures connection pooling and tokenizer reuse across RL workers without explicit lifecycle management.

**Exceptions** (`exceptions.py`) - Custom exception hierarchy rooted at `SGLangClientError`. HTTP errors are classified into `SGLangHTTPError` (base), `SGLangContextLengthError` (400 + length patterns), and `SGLangThrottledError` (429/503). Connection failures become `SGLangConnectionError`, non-JSON responses become `SGLangDecodingError`. These exceptions form the contract between `client.py` and `sglang.py` — the model layer never inspects raw HTTP status codes.

**TokenManager** (`token.py`) - Segment-based token accumulation for TITO. Tokens organized into PROMPT segments (loss_mask=0) and RESPONSE segments (loss_mask=1) matching multi-turn conversation structure. Exposes `token_ids`, `loss_mask`, `logprobs`, and `segments` properties.

**ToolParser** (`tool_parsers/`) - Abstract base with `HermesToolParser` and `QwenXMLToolParser` implementations. Parses tool calls from model output. Strict parsing: only catches JSONDecodeError, propagates failures as tool calls with `raw` content for model feedback. Excludes tool calls inside `<think>` blocks. New parsers self-register via `@register_tool_parser` decorator.

**ToolLimiter** (`tool_limiter.py`) - Strands hook enforcing tool iteration and/or call limits per invocation. Supports `max_tool_iters` (one iteration = model response with tool calls + execution) and `max_tool_calls` (individual call count). Raises `MaxToolIterationsReachedError` or `MaxToolCallsReachedError`.

### Key Design Decisions

- **Non-streaming**: Single POST to `/generate` instead of SSE streaming for ~20x throughput in parallel RL workers
- **Incremental tokenization**: First call tokenizes full prompt; subsequent calls only tokenize new messages (tool results) with message separator prepended
- **Strict tool parsing for RL**: No heuristic repair of malformed tool calls; errors propagated to model for self-correction
- **Segment-based TITO**: Token tracking mirrors multi-turn structure (prompt=no loss, response=loss)

## Code Style

- Ruff for linting and formatting (line-length 120, rules: E, F, I, N, W)
- Conventional commits enforced by pre-commit hook (feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert)
- Python 3.10+ required
- asyncio_mode = "auto" for pytest-asyncio

## Integration Tests with Remote GPU Server

If using a remote GPU server, SSH-tunnel port 30000 instead of copying code:

```bash
# 1. Launch SGLang on the remote server (docker or native)
# 2. Tunnel the port locally
ssh -L 30000:localhost:30000 -N -f <remote-host>
# 3. Run tests locally (model ID auto-detected from server)
pytest tests/integration/ -v --sglang-base-url=http://localhost:30000
```

Test with both an instruct model (e.g., `Qwen3-4B-Instruct-2507`) and a thinking model (e.g., `Qwen3-8B`) for full coverage. Thinking models will skip `MessageToTokenDrift` tests (expected).
