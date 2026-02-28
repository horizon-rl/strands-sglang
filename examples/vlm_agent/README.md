# VLM Agent with Inline Image + Image-Returning Tool

This example demonstrates two ways images enter the strands-sglang VLM pipeline:

1. **Inline in user prompt** — a.jpg is embedded directly in the initial message
2. **Via tool result** — the agent calls `read_image("b.jpg")` to load the second image

## How It Works

```
User message: [a.jpg IMAGE] "This is a.jpg. Use read_image to load b.jpg. Describe both."
  │
  ▼  tokenize_prompt_messages() — first turn
  │  processor(text, images=[a_pil]) → input_ids with vision tokens for a.jpg
  │  image_data = [a_b64]
  ▼
Model generates: <tool_call>{"name": "read_image", "arguments": {"file_path": "b.jpg"}}</tool_call>
  │
  ▼  Tool reads b.jpg from disk → returns {"image": "data:image/jpeg;base64,..."}
  │
  ▼  tokenize_prompt_messages() — incremental (tool result only)
  │  processor(text, images=[a_b64, b_b64]) → input_ids with vision tokens for b.jpg
  │  image_data = [a_b64, b_b64]  (accumulated)
  ▼
Model generates final response with both images in context
  │  SGLang payload: {input_ids: [...], image_data: [a_b64, b_b64]}
  ▼
TITO state:
  segment[0]: PROMPT    — system + tools + user message with a.jpg vision tokens
  segment[1]: RESPONSE  — model output with tool call
  segment[2]: PROMPT    — tool result with b.jpg vision tokens
  segment[3]: RESPONSE  — final description of both images
```

The example also shows how to extract multimodal training tensors (`pixel_values`, `image_grid_thw`) from the HF processor for RL training pipelines — this is outside strands-sglang's scope but commonly needed alongside the TITO trajectory.

## Setup

```bash
# 1. Install strands-sglang with VLM dependencies
pip install -e ".[vision]"
pip install torch torchvision  # install separately to match your CUDA version

# 2. Download sample images
curl -Lo examples/vlm_agent/images/a.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Labrador_on_Quantock_%282175262184%29.jpg/500px-Labrador_on_Quantock_%282175262184%29.jpg"
curl -Lo examples/vlm_agent/images/b.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Orange_tabby_kitten.jpg/500px-Orange_tabby_kitten.jpg"

# 3. Launch SGLang with a VLM model
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-VL-4B-Instruct \
    --port 30000

# 4. Run the example
python examples/vlm_agent/vlm_agent.py
# Configure via env vars: SGLANG_BASE_URL, MODEL_PATH
```
