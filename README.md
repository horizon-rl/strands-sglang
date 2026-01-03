# strands-sglang

SGLang model provider for [Strands Agents SDK](https://github.com/strands-agents/sdk-python) with Token-In/Token-Out (TITO) support for agentic RL training.

## Features

- **SGLang Native**: Utilizes SGLang's native `/generate` endpoint to integrate with Strands Agents SDK for agent loop
- **Token Management**: Manages tokens and their logprobs to ensure TITO (Token-In/Token-Out) in training.
- **Tool Parser**: Customizes tool parsing in training to align with model-specific chat template and ensure no *off-policy* post-processing

## Installation

```bash
pip install strands-sglang
```

Or with development dependencies:

```bash
git clone <repo>
cd strands-sglang
pip install -e ".[dev]"
```

## Quick Start

1. **Start SGLang server**:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B \
    --port 8000 \
    --host 0.0.0.0
```

2. **Use with Strands Agent**:

```python
import asyncio
from transformers import AutoTokenizer
from strands import Agent
from strands_tools import calculator
from strands_sglang import SGLangModel

async def main():
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = SGLangModel(
        tokenizer=tokenizer,
        base_url="http://localhost:8000",
    )

    # Create agent
    agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt="You are a helpful math assistant.",
    )

    # Run episode
    model.reset()  # Reset TITO state
    await agent.invoke_async("What is 25 * 17?")

    # Access TITO data for RL training
    token_ids = model.token_manager.token_ids      # All tokens
    output_mask = model.token_manager.output_mask  # True = model output
    logprobs = model.token_manager.logprobs        # Log probabilities

    print(f"Trajectory: {len(token_ids)} tokens")
    print(f"Output tokens: {sum(output_mask)}")

asyncio.run(main())
```

## TITO Data for RL Training

After generation, access trajectory data from `token_manager`:

```python
# Token trajectory (exactly what model saw/generated)
token_ids = model.token_manager.token_ids

# Loss mask (True = model output, for gradient computation)
output_mask = model.token_manager.output_mask

# Log probabilities (for policy gradient)
logprobs = model.token_manager.logprobs

# Segment info: [(is_output, length), ...]
segment_info = model.token_manager.segment_info
```

## Running Tests

See `tests/README.md` for detailed test configuration.

## License

Apache 2.0
