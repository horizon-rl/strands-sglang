#!/usr/bin/env python3
# Copyright 2025 Horizon RL Contributors
# SPDX-License-Identifier: Apache-2.0

"""VLM agent example with inline image and image-returning tool.

Demonstrates two ways images enter the strands-sglang VLM pipeline:

1. **Inline in user prompt** — a.jpg is embedded directly in the initial message
2. **Via tool result** — the agent calls ``read_image("b.jpg")`` to load the second image

After the agent finishes, we inspect TITO trajectory and show how to extract
multimodal training tensors (pixel_values, image_grid_thw) from the HF processor
for RL training pipelines.

Usage:
    python examples/vlm_agent/vlm_agent.py
    # Configure via env vars: SGLANG_BASE_URL, MODEL_PATH
"""

import asyncio
import base64
import json
import os
from pathlib import Path

from strands import Agent, tool
from transformers import AutoProcessor

from strands_sglang import SGLangModel, ToolLimiter
from strands_sglang.client import SGLangClient
from strands_sglang.tool_parsers import HermesToolParser

IMAGE_DIR = Path(__file__).parent / "images"


@tool
def read_image(file_path: str) -> dict:
    """Read an image file and return it for visual inspection.

    Args:
        file_path: Filename of the image (e.g., "a.jpg", "b.png").
    """
    path = Path(file_path)
    if not path.exists():
        path = IMAGE_DIR / path.name

    if not path.exists():
        return {
            "status": "error",
            "content": [{"text": f"File not found: {path}"}],
        }

    image_bytes = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".")
    fmt = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "png")

    return {
        "status": "success",
        "content": [
            {"text": f"Image loaded: {path.name} ({len(image_bytes)} bytes)"},
            {"image": {"format": fmt, "source": {"bytes": image_bytes}}},
        ],
    }


def _summarize_for_display(obj):
    """Make trajectory JSON-serializable with truncated binary data."""
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    if isinstance(obj, dict):
        return {k: _summarize_for_display(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_summarize_for_display(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:image/") and ";base64," in obj:
        prefix = obj[: obj.index(";base64,") + len(";base64,")]
        return prefix + obj[len(prefix) : len(prefix) + 20] + "..."
    return obj


async def main():
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    base_url = os.environ.get("SGLANG_BASE_URL", "http://localhost:30000")
    client = SGLangClient(base_url=base_url)

    model_info = await client.get_model_info()
    model_path = os.environ.get("MODEL_PATH", model_info["model_path"])
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = SGLangModel(
        client=client,
        processor=processor,
        tool_parser=HermesToolParser(),
        sampling_params={"max_new_tokens": 8192},
    )

    # -------------------------------------------------------------------------
    # 2. Run VLM Agent
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VLM Agent Example")
    print("=" * 60)

    agent = Agent(
        model=model,
        tools=[read_image],
        hooks=[ToolLimiter(max_tool_iters=5)],
        system_prompt="",
        callback_handler=None,
    )

    # Build prompt with inline image
    image_a_bytes = (IMAGE_DIR / "a.jpg").read_bytes()
    prompt = [
        {"image": {"format": "jpeg", "source": {"bytes": image_a_bytes}}},
        {"text": "This is a.jpg above. Now use read_image to load b.jpg. Then describe both images and compare them."},
    ]

    await agent.invoke_async(prompt)

    print(f"\n[Trajectory]: {json.dumps(_summarize_for_display(agent.messages), indent=2)}")

    # -------------------------------------------------------------------------
    # 3. TITO Data (for RL training)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TITO Token Manager State")
    print("=" * 60)

    tm = model.token_manager
    print(f"  Total tokens:    {len(tm)}")
    print(f"  Segments:        {len(tm.segment_info)}")
    for i, (is_output, length) in enumerate(tm.segment_info):
        seg_type = "RESPONSE" if is_output else "PROMPT"
        print(f"    [{i}] {seg_type:10s} {length} tokens")

    n_output = sum(tm.loss_mask)
    print(f"\n  Initial prompt:  {tm.segment_info[0][1]} tokens")
    print(f"  Rollout tokens:  {len(tm) - tm.segment_info[0][1]}")
    print(f"  Output tokens:   {n_output}")
    print(f"  Loss mask:       {tm.loss_mask[:10]}... (first 10)")

    output_logprobs = [lp for lp, m in zip(tm.logprobs, tm.loss_mask) if m and lp is not None]
    if output_logprobs:
        print(f"  Avg logprob:     {sum(output_logprobs) / len(output_logprobs):.4f}")

    # -------------------------------------------------------------------------
    # 4. VLM State
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VLM State")
    print("=" * 60)
    print(f"  Images accumulated:  {len(model.image_data)}")
    for i, img_url in enumerate(model.image_data):
        print(f"    [{i}] {img_url[:50]}...")

    # -------------------------------------------------------------------------
    # 5. Multimodal Training Tensors (not managed by strands-sglang)
    #
    # For RL training, you typically need pixel_values and image_grid_thw
    # alongside the token trajectory. These come from calling the HF processor
    # on the accumulated images. This is outside strands-sglang's scope but
    # shown here for completeness.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Multimodal Training Tensors (from HF processor)")
    print("=" * 60)

    from io import BytesIO  # noqa: E402

    from PIL import Image  # noqa: E402

    pil_images = []
    for data_url in model.image_data:
        b64_data = data_url.split(";base64,", 1)[1]
        pil_images.append(Image.open(BytesIO(base64.b64decode(b64_data))))

    if pil_images:
        # Re-format the conversation to get the chat template text (with image placeholders).
        # We can't use decoded token text because it contains expanded <|image_pad|> tokens
        # that would be double-counted by the processor.
        chat_msgs = SGLangModel.format_messages(agent.messages, is_multimodal=True)
        chat_text = processor.tokenizer.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=False)
        mm_inputs = processor(text=chat_text, images=pil_images, return_tensors="pt")

        for key, val in mm_inputs.items():
            if key == "input_ids":
                continue  # already tracked by TokenManager
            if hasattr(val, "shape"):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"  {key}: {type(val).__name__}")
    else:
        print("  No images — skipping")

    # -------------------------------------------------------------------------
    # 6. Full Context (decoded)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Full Context (decoded token sequence)")
    print("=" * 60)
    print(model.tokenizer.decode(tm.token_ids, skip_special_tokens=False))

    # Cleanup
    model.reset()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
