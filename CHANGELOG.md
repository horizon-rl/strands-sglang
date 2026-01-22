# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Think Block Exclusion**: `HermesToolCallParser` excludes tool calls inside `<think>` blocks by default, preventing parsing of draft tool calls from reasoning models (Qwen3, DeepSeek-R1). Configurable via `think_start_token`/`think_end_token`.

- **Tool Result Ordering**: Tool results are now sorted by sequential IDs (`call_0000`, `call_0001`, ...) before tokenization. Fixes ordering issues when Strands executes tools concurrently and returns results in completion order.

## [0.1.0] - 2026-01-20

First beta release. Stable for production agentic RL training.

### Added

- **Tool Parse Error Tracking**: `SGLangModel.tool_parse_errors` tracks parse errors per tool name for RL metrics.

### Changed

- **Beta Status**: API now stable for production RL workloads.

## [0.0.3] - 2026-01-08

### Added

- **Qwen3 Thinking Mode**: `enable_thinking` config for Qwen3 hybrid models.
- **Related Projects**: Added strands-vllm to README.

### Changed

- **Simplified TokenManager**: Removed unused `tokenizer` param and `decode()` method.
- **Message Formatting**: `_format_message_content` and `format_request_messages` now `@classmethod`.

### Fixed

- **Retry Logic**: 400 errors now retried (transient during weight reload). Only 401/403/404 non-retryable.
- **Context Length Detection**: Expanded patterns for context length errors in 400 responses.

## [0.0.2] - 2026-01-07

### Added

- **SGLangClient**: Async HTTP client with connection pooling, aggressive retry (60 attempts), infinite timeout.
- **`from_slime_args()` Factory**: Create client from Slime training args.
- **Conventional Commits**: Added commit-msg hook.

### Changed

- **Default Port**: 8000 → 30000 (SGLang default).
- **Non-Streaming Only**: Better parallelism (~20x) for RL training at scale.

### Removed

- **Streaming Support**: Removed for optimal RL training performance.

### Fixed

- Increased default `max_new_tokens` for thinking models.

## [0.0.1] - 2026-01-03

### Added

- Initial release with SGLang `/generate` API, TITO tracking, Hermes/Qwen tool parsing, `ToolIterationLimiter` hook.
