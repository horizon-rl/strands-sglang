# VLM Agent with Inline Image + Image-Returning Tool

This example demonstrates two ways images enter the strands-sglang VLM pipeline:

1. **Inline in user prompt** — a.jpg is embedded directly in the initial message
2. **Via tool result** — the agent calls `read_image("b.jpg")` to load the second image

## How It Works

```
User message: [a.jpg IMAGE] "This is a.jpg. Use read_image to load b.jpg. Describe both."
  │
  ▼  tokenize_prompt_messages() — first turn
  │  tokenizer.encode(text) → input_ids with unexpanded image placeholders
  │  image_data = [a_b64]
  ▼
Model generates: <tool_call>{"name": "read_image", "arguments": {"file_path": "b.jpg"}}</tool_call>
  │
  ▼  Tool reads b.jpg from disk → returns {"image": "data:image/jpeg;base64,..."}
  │
  ▼  tokenize_prompt_messages() — incremental (tool result only)
  │  tokenizer.encode(text) → input_ids for new tokens
  │  image_data = [a_b64, b_b64]  (accumulated)
  ▼
Model generates final response with both images in context
  │  SGLang payload: {input_ids: [...], image_data: [a_b64, b_b64]}
  │  Server handles image token expansion internally
  ▼
TITO state:
  segment[0]: PROMPT    — system + tools + user message with image placeholders
  segment[1]: RESPONSE  — model output with tool call
  segment[2]: PROMPT    — tool result with image placeholders
  segment[3]: RESPONSE  — final description of both images
```

## Setup

```bash
# 1. Install strands-sglang
pip install -e ".[dev]"

# 2. Download sample images
curl -Lo examples/vlm_agent/images/a.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Labrador_on_Quantock_%282175262184%29.jpg/500px-Labrador_on_Quantock_%282175262184%29.jpg"
curl -Lo examples/vlm_agent/images/b.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Orange_tabby_kitten.jpg/500px-Orange_tabby_kitten.jpg"

# 3. Launch SGLang with a VLM model
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-4B \
    --port 30000

# 4. Run the example
python examples/vlm_agent/vlm_agent.py
# Configure via env vars: SGLANG_BASE_URL, MODEL_PATH
```
