# Strands-SGLang

[![Awesome Strands Agents](https://img.shields.io/badge/Awesome-Strands%20Agents-00FF77?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjkwIiBoZWlnaHQ9IjQ2MyIgdmlld0JveD0iMCAwIDI5MCA0NjMiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik05Ny4yOTAyIDUyLjc4ODRDODUuMDY3NCA0OS4xNjY3IDcyLjIyMzQgNTYuMTM4OSA2OC42MDE3IDY4LjM2MTZDNjQuOTgwMSA4MC41ODQzIDcxLjk1MjQgOTMuNDI4MyA4NC4xNzQ5IDk3LjA1MDFMMjM1LjExNyAxMzkuNzc1QzI0NS4yMjMgMTQyLjc2OSAyNDYuMzU3IDE1Ni42MjggMjM2Ljg3NCAxNjEuMjI2TDMyLjU0NiAyNjAuMjkxQy0xNC45NDM5IDI4My4zMTYgLTkuMTYxMDcgMzUyLjc0IDQxLjQ4MzUgMzY3LjU5MUwxODkuNTUxIDQxMS4wMDlMMTkwLjEyNSA0MTEuMTY5QzIwMi4xODMgNDE0LjM3NiAyMTQuNjY1IDQwNy4zOTYgMjE4LjE5NiAzOTUuMzU1QzIyMS43ODQgMzgzLjEyMiAyMTQuNzc0IDM3MC4yOTYgMjAyLjU0MSAzNjYuNzA5TDU0LjQ3MzggMzIzLjI5MUM0NC4zNDQ3IDMyMC4zMjEgNDMuMTg3OSAzMDYuNDM2IDUyLjY4NTcgMzAxLjgzMUwyNTcuMDE0IDIwMi43NjZDMzA0LjQzMiAxNzkuNzc2IDI5OC43NTggMTEwLjQ4MyAyNDguMjMzIDk1LjUxMkw5Ny4yOTAyIDUyLjc4ODRaIiBmaWxsPSIjRkZGRkZGIi8+CjxwYXRoIGQ9Ik0yNTkuMTQ3IDAuOTgxODEyQzI3MS4zODkgLTIuNTc0OTggMjg0LjE5NyA0LjQ2NTcxIDI4Ny43NTQgMTYuNzA3NEMyOTEuMzExIDI4Ljk0OTIgMjg0LjI3IDQxLjc1NyAyNzIuMDI4IDQ1LjMxMzhMNzEuMTcyNyAxMDMuNjcxQzQwLjcxNDIgMTEyLjUyMSAzNy4xOTc2IDE1NC4yNjIgNjUuNzQ1OSAxNjguMDgzTDI0MS4zNDMgMjUzLjA5M0MzMDcuODcyIDI4NS4zMDIgMjk5Ljc5NCAzODIuNTQ2IDIyOC44NjIgNDAzLjMzNkwzMC40MDQxIDQ2MS41MDJDMTguMTcwNyA0NjUuMDg4IDUuMzQ3MDggNDU4LjA3OCAxLjc2MTUzIDQ0NS44NDRDLTEuODIzOSA0MzMuNjExIDUuMTg2MzcgNDIwLjc4NyAxNy40MTk3IDQxNy4yMDJMMjE1Ljg3OCAzNTkuMDM1QzI0Ni4yNzcgMzUwLjEyNSAyNDkuNzM5IDMwOC40NDkgMjIxLjIyNiAyOTQuNjQ1TDQ1LjYyOTcgMjA5LjYzNUMtMjAuOTgzNCAxNzcuMzg2IC0xMi43NzcyIDc5Ljk4OTMgNTguMjkyOCA1OS4zNDAyTDI1OS4xNDcgMC45ODE4MTJaIiBmaWxsPSIjRkZGRkZGIi8+Cjwvc3ZnPgo=&logoColor=white)](https://github.com/cagataycali/awesome-strands-agents)

[![CI](https://github.com/strands-rl/strands-sglang/actions/workflows/ci.yml/badge.svg)](https://github.com/strands-rl/strands-sglang/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/strands-sglang.svg)](https://pypi.org/project/strands-sglang/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/strands-rl/strands-sglang)
[![Blog](https://img.shields.io/badge/Blog-Strands--SGLang-000?logo=rss&logoColor=fff)](https://yuanhe.wiki/posts/technical/strands-sglang/)
[![Strands-Agents](https://img.shields.io/badge/Strands-Featured-111111?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjkwIiBoZWlnaHQ9IjQ2MyIgdmlld0JveD0iMCAwIDI5MCA0NjMiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI%2BCjxwYXRoIGQ9Ik05Ny4yOTAyIDUyLjc4ODRDODUuMDY3NCA0OS4xNjY3IDcyLjIyMzQgNTYuMTM4OSA2OC42MDE3IDY4LjM2MTZDNjQuOTgwMSA4MC41ODQzIDcxLjk1MjQgOTMuNDI4MyA4NC4xNzQ5IDk3LjA1MDFMMjM1LjExNyAxMzkuNzc1QzI0NS4yMjMgMTQyLjc2OSAyNDYuMzU3IDE1Ni42MjggMjM2Ljg3NCAxNjEuMjI2TDMyLjU0NiAyNjAuMjkxQy0xNC45NDM5IDI4My4zMTYgLTkuMTYxMDcgMzUyLjc0IDQxLjQ4MzUgMzY3LjU5MUwxODkuNTUxIDQxMS4wMDlMMTkwLjEyNSA0MTEuMTY5QzIwMi4xODMgNDE0LjM3NiAyMTQuNjY1IDQwNy4zOTYgMjE4LjE5NiAzOTUuMzU1QzIyMS43ODQgMzgzLjEyMiAyMTQuNzc0IDM3MC4yOTYgMjAyLjU0MSAzNjYuNzA5TDU0LjQ3MzggMzIzLjI5MUM0NC4zNDQ3IDMyMC4zMjEgNDMuMTg3OSAzMDYuNDM2IDUyLjY4NTcgMzAxLjgzMUwyNTcuMDE0IDIwMi43NjZDMzA0LjQzMiAxNzkuNzc2IDI5OC43NTggMTEwLjQ4MyAyNDguMjMzIDk1LjUxMkw5Ny4yOTAyIDUyLjc4ODRaIiBmaWxsPSIjOTg5ODk4Ii8%2BCjxwYXRoIGQ9Ik0yNTkuMTQ3IDAuOTgxODEyQzI3MS4zODkgLTIuNTc0OTggMjg0LjE5NyA0LjQ2NTcxIDI4Ny43NTQgMTYuNzA3NEMyOTEuMzExIDI4Ljk0OTIgMjg0LjI3IDQxLjc1NyAyNzIuMDI4IDQ1LjMxMzhMNzEuMTcyNyAxMDMuNjcxQzQwLjcxNDIgMTEyLjUyMSAzNy4xOTc2IDE1NC4yNjIgNjUuNzQ1OSAxNjguMDgzTDI0MS4zNDMgMjUzLjA5M0MzMDcuODcyIDI4NS4zMDIgMjk5Ljc5NCAzODIuNTQ2IDIyOC44NjIgNDAzLjMzNkwzMC40MDQxIDQ2MS41MDJDMTguMTcwNyA0NjUuMDg4IDUuMzQ3MDggNDU4LjA3OCAxLjc2MTUzIDQ0NS44NDRDLTEuODIzOSA0MzMuNjExIDUuMTg2MzcgNDIwLjc4NyAxNy40MTk3IDQxNy4yMDJMMjE1Ljg3OCAzNTkuMDM1QzI0Ni4yNzcgMzUwLjEyNSAyNDkuNzM5IDMwOC40NDkgMjIxLjIyNiAyOTQuNjQ1TDQ1LjYyOTcgMjA5LjYzNUMtMjAuOTgzNCAxNzcuMzg2IC0xMi43NzcyIDc5Ljk4OTMgNTguMjkyOCA1OS4zNDAyTDI1OS4xNDcgMC45ODE4MTJaIiBmaWxsPSIjMDBGRjc3Ii8%2BCjwvc3ZnPgo%3D&logoWidth=14)](https://strandsagents.com/latest/documentation/docs/community/model-providers/sglang/)


**Agentic RL, done correctly.**

- [Strands Agents SDK](https://github.com/strands-agents/sdk-python): a harness builder whose event-based hooks make the agent loop fully customizable.
- [SGLang](https://docs.sglang.io/): a high-performance serving framework for fast, high-concurrency rollouts that exposes per-token metadata.

**Strands-SGLang** bridges the two so multi-turn rollouts stay on-policy — token-in, token-out, no silent retokenization drift.

## Features

- **Token-In/Token-Out** rollouts: model-generated tokens carried through as-is
    - Only new messages are tokenized each turn with **incremental chat templating**
- **Strict tool-call parsing**: parsed exactly as generated, no heuristic repair
- **Harness customization**: pass tools and hooks into `Agent` to customize your harness
- **Native SGLang `/generate` endpoint**: high-throughput, non-streaming rollouts

> For RL environment integration, please refer to [`strands-env`](https://github.com/strands-rl/strands-env)

## Installation

```bash
pip install strands-sglang strands-agents-tools
```

Or install from source with development dependencies:

```bash
git clone https://github.com/strands-rl/strands-sglang.git
cd strands-sglang
pip install -e ".[dev]"
```

## Quick Start

### 1. Start SGLang Server

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B
```

### 2. Run an Agent

```python
import asyncio
from transformers import AutoTokenizer
from strands import Agent, tool
from strands_sglang import SGLangClient, SGLangModel
from strands_sglang.tool_parsers import get_tool_parser

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression."""
    return str(eval(expression))

async def main():
    model = SGLangModel(
        client=SGLangClient(base_url="http://localhost:30000"),
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B"),
        tool_parser=get_tool_parser("qwen_xml"),
    )
    agent = Agent(model=model, tools=[calculator])
    await agent.invoke_async("What is 25 * 17?")

    # SGLangModel captures the full token trajectory for on-policy RL training
    rollout = model.rollout
    print(rollout.token_ids, rollout.loss_mask, rollout.logprobs)

asyncio.run(main())
```

## RL Training

In principle, Strands-SGLang can be seen as a drop-in agentic rollout service and can be integrated with any RL training framework. A concrete example of training a math coding agent ([ReTool](https://arxiv.org/abs/2504.11536)) is available at [slime/examples/strands_sglang](https://github.com/THUDM/slime/tree/main/examples/strands_sglang).

Some key highlights of adapting Strands-SGLang to any RL framework:
- Pass tokens and token metadata from `Rollout` for on-policy rollouts
- Hook your harness with the built-in `LoopLimiter` (or your own) for controlled rollouts
- Use a shared `SGLangClient` and HF tokenizer; don't create one instance per rollout
- Classify the rollout's termination reason properly — it shapes reward and sampling


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
docs: update usage examples
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

## License

Apache License 2.0 - see [LICENSE](LICENSE).
