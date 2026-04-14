# Copyright 2025-2026 Strands RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VLM agent example with inline image and image-returning tool.

Demonstrates two ways images enter the strands-sglang VLM pipeline:

1. **Inline in user prompt** — a.jpg is embedded directly in the initial message
2. **Via tool result** — the agent calls ``read_image("b.jpg")`` to load the second image

After the agent finishes, we inspect the TITO trajectory.

Usage:
    python examples/vlm_agent/vlm_agent.py
    # Configure via env vars: SGLANG_BASE_URL, MODEL_PATH
"""

import asyncio
import json
import os
from pathlib import Path

from strands import Agent, tool
from transformers import AutoTokenizer

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

    model_info = await client.model_info()
    model_path = os.environ.get("MODEL_PATH", model_info["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = SGLangModel(
        client=client,
        tokenizer=tokenizer,
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

    output_logprobs = [lp for lp, m in zip(tm.logprobs, tm.loss_mask, strict=False) if m and lp is not None]
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
    # 5. Full Context (decoded)
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
