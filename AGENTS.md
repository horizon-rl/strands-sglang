# AGENTS.md

Guidance for coding agents working in this repository. `CLAUDE.md` is a symlink to this file.

## Project Overview

Strands-SGLang is an SGLang model provider for the Strands Agents SDK with Token-In/Token-Out (TITO) support for on-policy agentic reinforcement learning training. It captures exact token IDs and logprobs during generation without retokenization drift, which is critical for accurate gradient computation in RL training.

## Commands

### Setup
```bash
uv sync              # installs the dev group by default
pre-commit install   # hook types come from default_install_hook_types
```

### Linting
```bash
pre-commit run --all-files   # what CI's lint job runs; the tools below are a subset
ruff check src/ tests/ examples/
ruff format --check src/ tests/ examples/
mypy src/strands_sglang
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

**SGLangModel** (`sglang.py`) - Main entry point implementing the Strands `Model` interface. Requires `client` and `tokenizer` (keyword-only). Formats messages using HuggingFace chat templates (`apply_chat_template()`), calls SGLang's `/generate` endpoint (non-streaming by design for RL throughput), tracks TITO trajectory, and parses tool calls. VLM support is auto-detected server-side via `SGLangClient.is_multimodal()` (queries `/model_info` for `has_image_understanding`, cached after the first call). When multimodal, `collect_image_data()` derives `image_data` (base64 data URLs) from the messages on every call and forwards them to SGLang — the server handles image token expansion. Configuration via `SGLangConfig` TypedDict (sampling_params, return_logprob, return_routed_experts, enable_thinking), which inherits `context_window_limit` from Strands' `BaseModelConfig` — set it, or conversation managers and `estimate_utilization()` fall back to a hardcoded 200k. `reset()` starts a new trajectory, banking the finished one in `rollout_history`.

**SGLangClient** (`client.py`) - Async HTTP client using aiohttp with connection pooling and aggressive retry (60 attempts by default, aligned with slime RL framework). All error classification is centralized in `_classify_http_error()`, which maps HTTP responses to custom exceptions (`SGLangContextLengthError`, `SGLangThrottledError`, etc.). Non-retryable errors: 401, 403, 404, context-length 400. Uses lazy session creation to avoid aiohttp's event-loop warnings.

**Utilities** (`utils.py`) - `lru_cache`-backed factories for shared client and tokenizer instances: `get_client()`, `get_client_from_slime_args()`, `get_tokenizer()`. Ensures connection pooling and tokenizer reuse across RL workers without explicit lifecycle management.

**Exceptions** (`exceptions.py`) - Custom exception hierarchy rooted at `SGLangClientError`. HTTP errors are classified into `SGLangHTTPError` (base), `SGLangContextLengthError` (400 + length patterns), and `SGLangThrottledError` (429/503). Connection failures become `SGLangConnectionError`, non-JSON responses become `SGLangDecodingError`. These exceptions form the contract between `client.py` and `sglang.py` — the model layer never inspects raw HTTP status codes.

**Rollout** (`rollout.py`) - Pydantic model for segment-based token accumulation for TITO. Tokens are appended via `add_prompt()` (loss_mask=0: system, user, tool results) and `add_response()` (loss_mask=1: model output), matching multi-turn conversation structure. `_add_segment()` is the single place that maintains the cross-field length invariant (`token_ids`/`loss_mask`/`logprobs` stay equal length). Exposes flat `token_ids`, `loss_mask`, `logprobs`, plus `routed_experts` and `image_data` lists and `segment_info` (`(is_output, length)` per segment). Also tracks routed experts (`add_routed_experts()`, `decode_routed_experts()`) and the `initial_prompt_length` property. The `SGLangModel` exposes the in-progress one as `model.rollout`; earlier ones (closed by `reset()`) live in `model.rollout_history`, so read `[*model.rollout_history, model.rollout]` for a whole episode. `image_data` is assigned from `collect_image_data()` on each successful call, not accumulated — it is derived from the messages so that re-running a turn cannot duplicate an image.

**ToolParser** (`tool_parsers/`) - Abstract base with 4 implementations: `HermesToolParser` (Hermes/Qwen JSON), `QwenXMLToolParser` (XML), `GLMToolParser` (GLM-4), and `KimiK2ToolParser` (special-token sections). Strict parsing: only catches JSONDecodeError, propagates failures as tool calls with `raw` content for model feedback. Excludes tool calls inside `<think>` blocks. New parsers self-register via `@register_tool_parser` decorator. `SGLangModel` defaults `skip_special_tokens=False` in sampling_params so special-token tool-call formats (e.g. Kimi K2) survive in response text.

**LoopLimiter** (`limiter.py`) - Strands hook bounding the agent loop. Supports `max_tool_iters` (one iteration = model response with tool calls + execution), `max_tool_calls` (individual call count), `max_parallel_tool_calls` (excess parallel calls cancelled), and `max_messages` (all roles counted, checked only on user-role messages so the loop stops at a complete message boundary; counters accumulate across invocations until `reset()`, bounding total conversation length in multi-turn env loops). Raises `MaxToolIterationsReachedError`, `MaxToolCallsReachedError`, or `MaxMessagesReachedError`, all subclasses of `LoopLimitReachedError`.

### Key Design Decisions

- **Non-streaming**: Single POST to `/generate` instead of SSE streaming for ~20x throughput in parallel RL workers
- **Incremental tokenization**: First call tokenizes full prompt; subsequent calls only tokenize new messages (tool results) with message separator prepended
- **Strict tool parsing for RL**: No heuristic repair of malformed tool calls; errors propagated to model for self-correction
- **Segment-based TITO**: Token tracking mirrors multi-turn structure (prompt=no loss, response=loss)
- **Bound episodes with `LoopLimiter`, not a trimming `ConversationManager`**: `stream()` replays the rollout as a prompt prefix (`input_ids = rollout.token_ids + new_input_ids`), so incremental tokenization requires that prefix to only grow. A `ConversationManager` that trims `agent.messages` breaks that: the tokens already in `rollout` no longer correspond to the conversation, and `processed_messages` (which only advances) ends up past the end, raising `RuntimeError: No new messages to tokenize`. That fires on message *count*, so it can happen with context usage still far below the limit — and it is not a `ContextWindowOverflowException`, so Strands has no recovery path for it. Use `NullConversationManager` plus `LoopLimiter(max_messages=...)`: the limiter stops at a clean message boundary, and a genuine overflow surfaces as `ContextWindowOverflowException` (`NullConversationManager.reduce_context()` re-raises it) rather than being silently truncated. `reset()` is the breakpoint if a trim does happen — it banks the finished rollout in `rollout_history` and rewinds the cursor, so the next call re-tokenizes the trimmed conversation in full as a new trajectory's prompt
- **toolUse blocks skipped in message formatting**: `format_messages()` drops `toolUse` content blocks because tool calls already live verbatim in the assistant's text block (`stream()` yields the full raw text including `<tool_call>` markup, then parsed toolUse blocks separately — so Strands history holds both). Rendering toolUse via the chat template's `tool_calls` field would duplicate the call in the prompt and cause retokenization drift vs. the actual generated tokens
- **VLM via server-side expansion**: Multimodal support is auto-detected from the server (`/model_info` reports `has_image_understanding`). Tokenization always uses `tokenizer.encode()` — the SGLang server handles image token expansion via `image_data`. No `torch`/`torchvision` dependencies needed

## Maintenance

When adding new modules, changing commands, or altering key design patterns, update this file to reflect those changes.

## Code Style

- Ruff for linting and formatting (line-length 120; see `[tool.ruff.lint]` for the rule set)
- Pydocstyle with Google convention, formatting and content only (`D2`/`D4`, enforced in `src/`)
- Mypy with near-strict settings (see `pyproject.toml` for full config)
- Single backticks around identifiers and endpoints — `` `token_ids` ``, not Sphinx-style double
  backticks. Applies to docstrings, comments, and Markdown.
- Use lazy `%` formatting for logging (not f-strings)
- Conventional commits (feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert)
- Python 3.12+ required (CI tests 3.12 and 3.13)
- `asyncio_mode = "auto"` for pytest-asyncio, default timeout 90s
- `pre-commit run --all-files` is what CI's lint job runs; the individual tools are a subset

### Files

No licence or copyright headers — `LICENSE` at the repo root is the whole story.

No module or package docstrings. A file starts at its first import or definition; `D100` and
`D104` are unselected for that reason. An example that needs run instructions puts them in a
comment or its own README, not a docstring.

### Docstring Style

Document what the signature can't say. A docstring that restates the identifier is worse than
none, so `D1` (must exist) is not selected — no docstring is a valid choice, and
`def token_ids(self) -> list[int]` needs nothing. `D2`/`D4` still apply to the ones that are
there.

- **`Args:` is all-or-nothing.** `D417` fails a partial section, so the question is never "does
  this parameter need a line" but "do this function's parameters need explaining at all". Default
  no — the name and the annotation carry it. Write the section when a parameter's meaning isn't in
  its type: resolution order, what `None` falls back to, units, what happens when two arguments
  disagree. One line per parameter that only restates the annotation is the signal the whole
  section should go.
- **`Notes:` is a footnote, the body is the explanation.** The test: delete the sentence. If the
  function no longer makes sense, it was body text. If all that's lost is a warning, it belongs
  under `Notes:` — a tokenizer quirk, an upstream API's behaviour, a deliberate divergence from
  the SGLang server's defaults. One `Notes:` holding three unrelated facts means none of them got
  classified; split them.
- **`Returns:`/`Yields:` earn their place against the annotation**, not the reader's curiosity.
  `-> list[int]` needs no "a list of token ids". A `-> dict` that is really a `/generate` payload
  does. The poorer the type, the more the section is worth.
- **`Raises:` only for exceptions a caller is expected to catch**, not every one that can escape.
  **`Example:`** only where usage isn't guessable from the signature. Nothing else: no
  `Attributes:`, no `References:`, no `Todo:`.
- **Token-level claims get verified, not reflowed.** This package's docstrings assert things about
  `loss_mask`, `logprobs` and retokenization that a reader will trust for training correctness. If
  a docstring names an index, an offset or an invariant, check it against the code before editing
  the sentence around it.

### Comments

One sentence. Longer only when the reasoning genuinely doesn't compress.

Comment the *why*, on the line it explains. Design rationale that needs paragraphs belongs in the
commit message, attached to the change rather than to the code forever.

### Private helpers

Extract a helper because the logic has a *name*, not because it repeats. Repetition is usually
better removed with a parameter or a loop; a helper earns its place by naming a concept. Eight to
fifteen lines is the size at which naming something starts to pay.

- **Under four lines: look for a reason to keep it.** The indirection usually costs more than it
  saves. It survives when the body carries a rule a caller would otherwise repeat.
- **Called once: the name has to say something concrete.** Single-use helpers are normal, so call
  count is not the test — nameability is. If the best name available is `_add` or `_process`, there
  is no concept to extract.
- **Private share is a smell, not a limit.** A high share is fine when each name is a domain
  concept. It is a problem when the names are stages of one procedure that was sliced up.

When a helper does survive, define it before its callers so reading top to bottom never requires
jumping ahead.

### Module-level private functions

**Default is zero per file.** Adding one means arguing why it belongs to neither a class nor its
single caller. A module-level `_helper` has no owner, which is why these accumulate: anyone can add
one and nothing says what it pairs with.

Module-level private *constants* are exempt — constants belong at module scope.

## Tests

Group into a class named for the unit under test, with the methods naming only the scenario — don't
repeat the class's subject in both. Reach for a class when there are three or more scenarios; below
that a module-level function carries the context on its own.

Docstrings on tests are optional and most are better off without one — the name is the
documentation. Write one when the *reason* the case exists isn't obvious from the name (a
regression whose trigger needs naming, a fidelity claim against upstream behaviour). If a single
assertion needs justification, a one-line comment above it beats a docstring.

## Integration Tests with Remote GPU Server

If using a remote GPU server, SSH-tunnel port 30000 instead of copying code:

```bash
# 1. Launch SGLang on the remote server (docker or native)
# 2. Tunnel the port locally
ssh -L 30000:localhost:30000 -N -f <remote-host>
# 3. Run tests locally (model ID auto-detected from server)
pytest tests/integration/ -v --sglang-base-url=http://localhost:30000
```

Test with both an instruct model (e.g., `Qwen3-4B-Instruct-2507`) and a thinking model (e.g., `Qwen3-8B`) for full coverage.

### VLM Integration Tests

VLM tests require an SGLang server running a VLM model (e.g., `Qwen/Qwen3.5-4B`). No extra dependencies (`torch`, `torchvision`) are needed — multimodal support is auto-detected from the server and image token expansion is handled server-side. Tests are automatically skipped when the server is running a text-only model.

```bash
# Run VLM tests specifically
pytest tests/integration/test_sglang_vision.py -v --sglang-base-url=http://localhost:30000
```
