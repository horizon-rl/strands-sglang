Write unit tests for a given module or class.

The user provides a target as $ARGUMENTS (e.g., a file path, class name, or module). If not provided, ask.

## Conventions

Follow the existing test style in `tests/unit/`:

- **License header**: Every `.py` file must start with the Apache 2.0 license header (copy from any existing source file)
- **File naming**: `test_<module>.py` (e.g., `test_sglang.py` for `sglang.py`)
- **Location**: `tests/unit/`
- **Test classes**: Group by feature/method, named `Test<Feature>` (e.g., `TestFormatMessages`, `TestMaxToolIters`)
- **Test methods**: `test_<behavior>` in snake_case
- **Async tests**: Use `async def test_*` directly — `asyncio_mode = "auto"` is configured
- **Fixtures**: Define at the top of the file after imports
- **Mocking**: Use `unittest.mock` (`MagicMock`, `AsyncMock`, `patch`)
- **Imports**: Import from the public API (e.g., `from strands_sglang import SGLangModel`)
- **Assertions**: Use plain `assert`

## Design Principles

- **Test real behavior, not Python mechanics**: Don't test frozen dataclass immutability, dict get/set, or signature enforcement — Python guarantees these. Test that *our code* produces correct results.
- **Don't re-implement production logic in tests**: If a test copy-pastes the `if isinstance` chain from production code and asserts it returns the same thing, it's testing Python, not our code. Instead, test the real code path through the public API.
- **Mock-hidden tests are weak**: If `mock.encode.return_value = [1,2,3]` and you assert `result == [1,2,3]`, you're testing the mock, not the code. Test that the mock was called correctly or test with real inputs.
- **Each test should test ONE behavior** that could independently break
- **Do NOT over-test**: If a behavior is covered by integration tests, skip it in unit tests. Keep test-to-source ratio reasonable.
- **Focus on the public API**: Test constructors, public methods, properties. Only test private methods if they contain complex logic (e.g., `_classify_http_error`).

## Steps

1. Read the source file to understand the public interface
2. Read existing tests in `tests/unit/` for style reference
3. Write tests covering: key public methods, edge cases, error cases
4. After writing, run tests with `/run-unit-tests` to verify they pass
