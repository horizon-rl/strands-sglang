# Tests for strands-sglang

## Running Tests

### Unit Tests Only (No Server Needed)

```bash
# Fast - run during development
pytest tests/unit/
```

### Integration Tests Only (Requires SGLang Server)

Start an SGLang server first:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --port 30000 \
    --host 0.0.0.0 \
    --tp-size 8 \
    --mem-fraction-static 0.7
```

| Parameter               | Description                                         |
| ----------------------- | --------------------------------------------------- |
| `--model-path`          | HuggingFace model ID or local path                  |
| `--port`                | Port to serve on (default: 8000)                    |
| `--host`                | Host to bind to (`0.0.0.0` for external access)     |
| `--tp-size`             | Tensor parallelism size (match your GPU count)      |
| `--mem-fraction-static` | Fraction of GPU memory for KV cache (reduce if OOM) |

**Note**: `--tool-call-parser` is NOT needed - we handle tool parsing internally.

Then run integration tests:

```bash
pytest tests/integration/ -v
```

### All Tests

```bash
pytest tests/
```

## Configuration

Integration tests can be configured via **command-line options** (recommended) or environment variables.

### Command-Line Options (Recommended)

```bash
# View available options
pytest --help | grep sglang

# Configure via CLI (model ID auto-detected from server)
pytest tests/integration/ --sglang-base-url=http://localhost:30000
```

### Environment Variables

| Variable          | Default                  | Description       |
| ----------------- | ------------------------ | ----------------- |
| `SGLANG_BASE_URL` | `http://localhost:30000` | SGLang server URL |

Model ID and tokenizer path are automatically detected from the server's `/get_model_info` endpoint.

```bash
SGLANG_BASE_URL=http://my-server:my-port pytest tests/integration/
```

**Priority**: CLI options > Environment variables > Defaults

## Writing New Tests

### Unit Tests

Add to `tests/unit/`. Use mocks for external dependencies:

```python
from unittest.mock import MagicMock

def test_my_feature():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    # ... test with mock
```

### Integration Tests

Add to `tests/integration/`. Use fixtures from `conftest.py`:

```python
# Fixtures available: tokenizer, model, calculator_tool

class TestMyFeature:
    async def test_something(self, model, tokenizer):
        # model and tokenizer are real, connected to server
        async for event in model.stream(messages):
            ...
```
