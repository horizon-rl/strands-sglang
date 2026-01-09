# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.3] - 2026-01-08

### Added

- **Qwen3 Hybrid Thinking Mode Support**: Added `enable_thinking` config option for Qwen3 hybrid thinking models. This is passed to `apply_chat_template` to control whether the model uses its internal reasoning mode with `<think>` tokens. Default is `None` (not passed to template) to avoid affecting non-Qwen3 models.

  ```python
  # For Qwen3 hybrid models: enable thinking mode
  model = SGLangModel(tokenizer=tokenizer, enable_thinking=True)

  # For Qwen3 hybrid models: disable thinking for faster non-reasoning tasks
  model = SGLangModel(tokenizer=tokenizer, enable_thinking=False)

  # For non-Qwen3 models: don't set (default None, parameter not passed)
  model = SGLangModel(tokenizer=tokenizer)
  ```

- **Related Projects**: Added [strands-vllm](https://github.com/agents-community/strands-vllm) to README as a community vLLM provider inspired by this project.

### Changed

- **Simplified TokenManager API**: Removed `tokenizer` parameter from `TokenManager.__init__()` and removed the `decode()` method. The tokenizer was only used for single-token decoding which was never used in practice—all real usage calls `model.tokenizer.decode()` directly for batch decoding with options like `skip_special_tokens`.

- **Message Formatting Methods**: Converted `_format_message_content` and `format_request_messages` to `@classmethod` since they don't use instance state. This clarifies intent and allows calling without an instance.

### Fixed

- **SLIME-Aligned Retry for Local Servers**: Changed retry behavior to match SLIME's aggressive retry philosophy for local SGLang servers during RL training:
  - 400 errors are now **retried** (can be transient during weight reloading, memory pressure)
  - Only truly non-retryable: 401 (auth), 403 (forbidden), 404 (not found)
  - 400 with context length patterns still not retried (won't help)
  - References: [OpenAI Python SDK](https://github.com/openai/openai-python) retries 408/409/429/5xx; [SLIME](https://github.com/THUDM/slime) retries ALL errors

- **Improved Context Length Detection**: Expanded patterns to detect context/prompt length errors in 400 responses:
  - Now matches: "exceed", "too long", "max model len", "maximum length", "context length"
  - These are converted to `ContextWindowOverflowException` (TRUNCATED, not ABORTED)
  - Added logging for unexpected 400 errors to aid debugging

## [0.0.2] - 2026-01-07

### Added

- **`SGLangClient` Class** (`client.py`): High-level async HTTP client for SGLang server, aligned with [Slime's http_utils.py](https://github.com/THUDM/slime/blob/main/slime/utils/http_utils.py) for RL training stability:
  - Connection pooling (default 1000 max connections, with matching keepalive)
  - Aggressive retry: 60 attempts with 1s delay (like Slime)
  - Infinite timeout by default for long generations (`timeout=None`)
  - Non-streaming POST for better parallelism at scale

- **`SGLangClient.from_slime_args()` Factory Method**: Create client directly from Slime training args with auto-computed `max_connections`:

  ```python
  client = SGLangClient.from_slime_args(args)
  model = SGLangModel(tokenizer=tokenizer, client=client)
  ```

- **Slime-Aligned Retry Logic**: Aggressive retry on most errors (blacklist approach):
  - Retries all 5xx server errors
  - Retries 408 (Request Timeout) and 429 (Rate Limit) per OpenAI best practices
  - Retries connection errors (`ConnectError`, `PoolTimeout`, `ReadTimeout`)
  - Only non-retryable: permanent client errors (400, 401, 403, 404, 422, etc.)

- **Conventional Commits**: Added `commit-msg` hook via `conventional-pre-commit` to enforce [Conventional Commits](https://www.conventionalcommits.org/) format.

### Changed

- **Default Port**: Changed from 8000 to 30000 to match SGLang's default.
- **`SGLangModel` Now Uses `SGLangClient`**: The model uses `SGLangClient` for HTTP communication, providing retry logic and better error handling.
- **Improved Error Handling**: SGLang HTTP errors now properly raise `ContextWindowOverflowException` for context length errors and `ModelThrottledException` for rate limiting (429/503).
- **BREAKING: Non-Streaming Only**: `SGLangClient.generate()` now returns `dict[str, Any]` directly instead of an `AsyncGenerator`. This provides ~20x better parallelism for RL training at scale by releasing connections immediately after response.

### Removed

- **Streaming Support**: Removed all streaming/SSE code. Non-streaming POST is now the only mode, aligned with Slime's `http_utils.py` for optimal RL training performance. Streaming held connections open during generation, causing serialization at high concurrency.
- **`stream` Config Option**: Removed from `SGLangConfig` as streaming is no longer supported.

### Fixed

- Default `max_new_tokens` increased for thinking models that require longer outputs.

## [0.0.1] - 2026-01-03

### Added

- Initial release with SGLang native `/generate` API support.
- Token-In/Token-Out (TITO) tracking via `TokenManager`.
- Hermes/Qwen tool call parsing with `HermesToolCallParser`.
- `ToolIterationLimiter` hook for clean trajectory truncation.
- Integration with Strands Agents SDK.
