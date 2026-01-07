# strands-sglang

SGLang model provider for [Strands Agents SDK](https://github.com/strands-agents/sdk-python) with Token-In/Token-Out (TITO) support for agentic RL training.

## Features

- **SGLang Native API**: Uses SGLang's native `/generate` endpoint with non-streaming POST for optimal parallelism
- **TITO Support**: Tracks complete token trajectories with logprobs for RL training - no retokenization drift
- **Tool Call Parsing**: Customizable tool parsing aligned with model chat templates (Hermes/Qwen format)
- **Iteration Limiting**: Built-in hook to limit tool iterations with clean trajectory truncation
- **RL Training Optimized**: Connection pooling, aggressive retry (60 attempts), and non-streaming design aligned with [Slime's http_utils.py](https://github.com/THUDM/slime/blob/main/slime/utils/http_utils.py)

## Requirements

- Python 3.10+
- Strands Agents SDK 1.7.0+
- SGLang server running with your model
- HuggingFace tokenizer for the model

## Installation

```bash
pip install strands-sglang strands-agents-tools
```

Or install from source with development dependencies:

```bash
git clone https://github.com/anthropics/strands-sglang.git
cd strands-sglang
pip install -e ".[dev]"
```

## Quick Start

### 1. Start SGLang Server

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --port 30000 \
    --host 0.0.0.0
```

### 2. Basic Agent

```python
import asyncio
from transformers import AutoTokenizer
from strands import Agent
from strands_tools import calculator
from strands_sglang import SGLangModel

async def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    model = SGLangModel(tokenizer=tokenizer, base_url="http://localhost:30000")
    agent = Agent(model=model, tools=[calculator])

    model.reset()  # Reset TITO state for new episode
    result = await agent.invoke_async("What is 25 * 17?")
    print(result)

    # Access TITO data for RL training
    print(f"Tokens: {model.token_manager.token_ids}")
    print(f"Loss mask: {model.token_manager.loss_mask}")
    print(f"Logprobs: {model.token_manager.logprobs}")

asyncio.run(main())
```

## Slime Training

For RL training with [Slime](https://github.com/THUDM/slime/), `SGLangModel` with TITO eliminates the retokenization step in [`generate_with_strands.py`](https://github.com/THUDM/slime/blob/main/examples/strands-agents/generate_with_strands.py) (this example is not fully ready yet):

```python
from strands import Agent
from strands_sglang import SGLangClient, SGLangModel, ToolIterationLimiter
from slime.utils.types import Sample

SYSTEM_PROMPT = "..."  # Your system prompt

async def generate(args, sample: Sample, sampling_params) -> Sample:
    ...
    state = GenerateState(args)

    # Create client and model with TITO tracking
    client = SGLangClient.from_slime_args(args)
    model = SGLangModel(
        tokenizer=state.tokenizer,
        client=client,
        params={"max_new_tokens": sampling_params["max_new_tokens"], ...},
    )
    agent = Agent(
        model=model,
        tools=[...],  # Your tools
        hooks=[ToolIterationLimiter(max_iterations=...)],
        system_prompt=SYSTEM_PROMPT,
    )

    # Run agent
    model.reset()
    try:
        await agent.invoke_async(sample.prompt)
        sample.status = Sample.Status.COMPLETED
    except Exception as e:
        # Define what truncation exceptions look like
        if _is_truncation_error(e):
            sample.status = Sample.Status.TRUNCATED
        else:
            sample.status = Sample.Status.ABORTED

    # TITO: No retokenization needed - tokens tracked during generation
    prompt_len = len(model.token_manager.segments[0])  # system + user are first segment
    sample.tokens = model.token_manager.token_ids
    sample.loss_mask = model.token_manager.loss_mask[prompt_len:]
    sample.rollout_log_probs = model.token_manager.logprobs[prompt_len:]
    sample.response_length = len(sample.tokens) - prompt_len
    sample.response = model.tokenizer.decode(sample.tokens[prompt_len:], skip_special_tokens=False)
    ...

    return sample
```

## Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires SGLang server)
pytest tests/integration/ -v --sglang-base-url=http://localhost:30000
```

## Contributing

Contributions welcome! Install pre-commit hooks for code style and commit message validation:

```bash
pip install -e ".[dev]"
pre-commit install -t pre-commit -t commit-msg
```

This project uses [Conventional Commits](https://www.conventionalcommits.org/). Commit messages must follow the format:

```
<type>(<scope>): <description>

# Examples:
feat(client): add retry backoff configuration
fix(sglang): handle empty response from server
docs: update TITO usage examples
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

## License

Apache License 2.0 - see [LICENSE](LICENSE).
