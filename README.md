# strands-sglang

SGLang model provider for [Strands Agents SDK](https://github.com/strands-agents/sdk-python) with Token-In/Token-Out (TITO) support for agentic RL training.

## Features

- **SGLang Native API**: Uses SGLang's native `/generate` endpoint for efficient token-level generation
- **TITO Support**: Tracks complete token trajectories with logprobs for RL training - no retokenization drift (see [examples/retokenization_drift/](examples/retokenization_drift/))
- **Tool Call Parsing**: Customizable tool parsing aligned with model chat templates (Hermes/Qwen format)
- **Iteration Limiting**: Built-in hook to limit tool iterations with clean trajectory truncation

## Requirements

- Python 3.10+
- Strands Agents SDK 1.7.0+
- SGLang server running with your model
- HuggingFace tokenizer for the model

## Installation

```bash
pip install strands-agents strands-sglang strands-agents-tools
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
    --model-path Qwen/Qwen3-4B-Thinking-2507 \
    --port 8000 \
    --host 0.0.0.0
```

> Tips: There's no need to load SGLang's tool parser because this is for training

### 2. Basic Agent Usage

```python
import asyncio
from transformers import AutoTokenizer
from strands import Agent
from strands_tools import calculator
from strands_sglang import SGLangModel

async def main():
    # Initialize model with tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    model = SGLangModel(
        tokenizer=tokenizer,
        base_url="http://localhost:8000",
        model_id="Qwen/Qwen3-4B-Thinking-2507",
        params={
            "max_new_tokens": 10240,
        },
    )

    # Create agent with tools
    agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt="You are a helpful math assistant. Use the calculator for all arithmetic.",
    )

    # Run episode
    model.reset()  # Reset TITO state for new episode
    result = await agent.invoke_async("What is 25 * 17?")
    print(result)

    # Access TITO data for RL training
    print(f"Trajectory: {len(model.token_manager)} tokens")
    print(f"Output tokens: {sum(model.token_manager.loss_mask)}")

asyncio.run(main())
```

## RL Training with Slime

For RL training with [Slime](https://github.com/THUDM/slime/), run async rollout:

```python
async def generate(args, sample: Sample, sampling_params) -> Sample:
    ...
    # The whole agent loop logic in a few lines
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    model = SGLangModel(tokenizer=tokenizer, base_url=url)
    limiter = ToolIterationLimiter(max_iterations=5)  # Optional: control maximum tool iteration
    agent = Agent(model=model, tools=[calculator], hooks[limiter], system_prompt="...")
    try:
        await agent.invoke_async(sample.prompt)
        sample.status = Sample.Status.COMPLETED
    except Exception as e:
        # Use exception to determine TRUNCATED or ABORTED
        ...
    # Use model.token_manager to fill in sample's attributes
    sample.tokens = model.token_manager.token_ids
    sample.loss_mask = model.token_manager.loss_mask
    sample.rollout_log_probs = model.token_manager.logprobs
    ...
```

A concrete example at Slime's repository will be available later.

## Configuration

### SGLangModel Options

```python
model = SGLangModel(
    tokenizer=tokenizer,           # Required: HuggingFace tokenizer
    base_url="http://localhost:8000",  # SGLang server URL
    model_id="Qwen/Qwen3-4B-Thinking-2507",  # Optional: model identifier
    tool_call_parser=HermesToolCallParser(),  # Tool call format parser
    params={                        # Sampling parameters
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    timeout=300.0,                  # Request timeout in seconds
    return_logprobs=True,           # Return logprobs (default: True)
)
```

> See more sampling params options at SGLang's [documentation](https://docs.sglang.io/basic_usage/sampling_params.html).

## Testing

### Unit Tests

```bash
pytest tests/unit/ -v
```

### Integration Tests

Requires a running SGLang server:

```bash
# Start server first
python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Thinking-2507 --port 8000

# Run tests
pytest tests/integration/ -v \
    --sglang-base-url=http://localhost:8000 \
    --sglang-model-id=Qwen/Qwen3-4B-Thinking-2507
```

Or configure via environment variables:

```bash
export SGLANG_BASE_URL=http://localhost:8000
export SGLANG_MODEL_ID=Qwen/Qwen3-4B-Thinking-2507
pytest tests/integration/ -v
```

## Contributing

```bash
pip install -e ".[dev]"
pre-commit install
```

Now `git commit` will auto-run linting and formatting checks.

## License

Apache 2.0
