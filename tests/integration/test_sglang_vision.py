# Copyright 2025-2026 Horizon RL Contributors
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

"""Integration tests for VLM (Vision Language Model) support.

Requires an SGLang server running a VLM model (e.g., Qwen3.5-4B).
Tests are automatically skipped if the server is running a text-only model.
"""

import io

import pytest
from PIL import Image

pytestmark = pytest.mark.integration


def _make_test_png(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    """Generate a valid 64x64 PNG image in memory."""
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_RED_PNG = _make_test_png((255, 0, 0))
_BLUE_PNG = _make_test_png((0, 0, 255))


def _image_block(png_bytes: bytes = _RED_PNG) -> dict:
    """Strands ImageContent block."""
    return {"image": {"format": "png", "source": {"bytes": png_bytes}}}


async def test_vlm_multi_turn(vlm_model):
    """Three-turn VLM conversation: image → text-only → another image.

    Covers: generation, token trajectory (segments/logprobs), image accumulation, incremental tokenization.
    """
    # -- Turn 1: image --
    messages = [{"role": "user", "content": [{"text": "What color is this?"}, _image_block(_RED_PNG)]}]

    events = []
    async for event in vlm_model.stream(messages):
        events.append(event)

    # Response produced
    text_deltas = [e["contentBlockDelta"]["delta"]["text"] for e in events if "contentBlockDelta" in e]
    assert len(text_deltas) > 0
    first_response = "".join(text_deltas)

    # Token trajectory: prompt + response segments with correct loss mask
    tm = vlm_model.token_manager
    tokens_after_t1 = len(tm)
    assert tokens_after_t1 > 0
    segments = tm.segment_info
    assert segments[0][0] is False  # prompt
    assert segments[1][0] is True  # response
    prompt_len = segments[0][1]
    assert all(m == 0 for m in tm.loss_mask[:prompt_len])
    assert any(t.logprob is not None for t in tm.tokens if t.loss_mask)

    # image_data: 1 image so far
    assert len(vlm_model.image_data) == 1
    assert vlm_model.image_data[0].startswith("data:image/png;base64,")

    # -- Turn 2: text-only follow-up --
    messages.append({"role": "assistant", "content": [{"text": first_response}]})
    messages.append({"role": "user", "content": [{"text": "Can you be more specific?"}]})

    events2 = []
    async for event in vlm_model.stream(messages):
        events2.append(event)
    second_response = "".join(e["contentBlockDelta"]["delta"]["text"] for e in events2 if "contentBlockDelta" in e)

    tokens_after_t2 = len(tm)
    assert tokens_after_t2 > tokens_after_t1
    assert len(vlm_model.image_data) == 1  # no new images

    # -- Turn 3: another image --
    messages.append({"role": "assistant", "content": [{"text": second_response}]})
    messages.append({"role": "user", "content": [{"text": "Now what color is this one?"}, _image_block(_BLUE_PNG)]})

    async for _ in vlm_model.stream(messages):
        pass

    tokens_after_t3 = len(tm)
    assert tokens_after_t3 > tokens_after_t2
    assert len(vlm_model.image_data) == 2  # red + blue
