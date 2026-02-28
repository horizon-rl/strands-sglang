# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for VLM (Vision Language Model) support.

Requires an SGLang server running a VLM model (e.g., Qwen3-VL-4B-Instruct).
Tests are automatically skipped if the server is running a text-only model.
"""

import io

import pytest
from PIL import Image

pytestmark = pytest.mark.integration


def _make_test_png() -> bytes:
    """Generate a valid 64x64 red PNG image in memory."""
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_TEST_PNG = _make_test_png()


def _image_block() -> dict:
    """Strands ImageContent block."""
    return {"image": {"format": "png", "source": {"bytes": _TEST_PNG}}}


class TestVLMGeneration:
    """Basic VLM generation — model receives an image and produces text."""

    async def test_simple_image_query(self, vlm_model):
        """Model generates a response when given an image."""
        messages = [{"role": "user", "content": [{"text": "What do you see in this image?"}, _image_block()]}]

        events = []
        async for event in vlm_model.stream(messages, system_prompt="Describe images briefly."):
            events.append(event)

        assert events[0] == {"messageStart": {"role": "assistant"}}
        text_deltas = [e["contentBlockDelta"]["delta"]["text"] for e in events if "contentBlockDelta" in e]
        assert len(text_deltas) > 0, "Model should produce text output"

        stop_events = [e for e in events if "messageStop" in e]
        assert stop_events[0]["messageStop"]["stopReason"] == "end_turn"

    async def test_text_only_query_on_vlm(self, vlm_model):
        """Text-only query still works on VLM model."""
        messages = [{"role": "user", "content": [{"text": "What is 2+2?"}]}]

        events = []
        async for event in vlm_model.stream(messages):
            events.append(event)

        text_deltas = [e["contentBlockDelta"]["delta"]["text"] for e in events if "contentBlockDelta" in e]
        assert len(text_deltas) > 0


class TestVLMTITO:
    """Token-in/Token-out tracking with VLM."""

    async def test_prompt_and_response_segments(self, vlm_model):
        """VLM generation produces correct prompt/response segments with proper loss mask."""
        messages = [{"role": "user", "content": [{"text": "What color is this?"}, _image_block()]}]

        async for _ in vlm_model.stream(messages):
            pass

        tm = vlm_model.token_manager
        assert len(tm) > 0
        assert len(tm.token_ids) == len(tm.loss_mask)

        segments = tm.segment_info
        assert len(segments) >= 2, f"Expected >= 2 segments, got {segments}"
        # segment_info: (loss_mask, length) — False=prompt, True=response
        assert segments[0][0] is False  # prompt
        assert segments[1][0] is True  # response

        prompt_len = segments[0][1]
        response_len = segments[1][1]
        assert all(m == 0 for m in tm.loss_mask[:prompt_len])
        assert all(m == 1 for m in tm.loss_mask[prompt_len : prompt_len + response_len])

    async def test_logprobs_for_response_tokens(self, vlm_model):
        """Response tokens should have logprobs when return_logprob=True (default)."""
        messages = [{"role": "user", "content": [{"text": "What is this?"}, _image_block()]}]

        async for _ in vlm_model.stream(messages):
            pass

        tokens = vlm_model.token_manager.tokens
        response_logprobs = [t.logprob for t in tokens if t.loss_mask]
        # First output token may be None (SGLang quirk), rest should be present
        assert any(lp is not None for lp in response_logprobs), "Should have logprobs for response tokens"


class TestVLMImageData:
    """Verify image_data accumulation and forwarding."""

    async def test_image_data_accumulated(self, vlm_model):
        """image_data should contain the image after generation."""
        messages = [{"role": "user", "content": [{"text": "Describe."}, _image_block()]}]

        async for _ in vlm_model.stream(messages):
            pass

        assert len(vlm_model.image_data) == 1
        assert vlm_model.image_data[0].startswith("data:image/png;base64,")

    async def test_multiple_images(self, vlm_model):
        """Multiple images in one message are all accumulated."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": "Compare these images."},
                    _image_block(),
                    _image_block(),
                ],
            }
        ]

        async for _ in vlm_model.stream(messages):
            pass

        assert len(vlm_model.image_data) == 2


class TestVLMMultiTurn:
    """Multi-turn VLM conversations with image references."""

    async def test_two_turn_with_image(self, vlm_model):
        """Second turn can reference image from first turn."""
        messages = [{"role": "user", "content": [{"text": "What is this image?"}, _image_block()]}]

        # First turn
        events1 = []
        async for event in vlm_model.stream(messages):
            events1.append(event)

        first_response = "".join(
            e["contentBlockDelta"]["delta"]["text"] for e in events1 if "contentBlockDelta" in e
        )
        tokens_after_first = len(vlm_model.token_manager)

        # Second turn (no new image)
        messages.append({"role": "assistant", "content": [{"text": first_response}]})
        messages.append({"role": "user", "content": [{"text": "Can you be more specific?"}]})

        async for _ in vlm_model.stream(messages):
            pass

        # Tokens should have grown
        assert len(vlm_model.token_manager) > tokens_after_first
        # Still only 1 image from the first turn
        assert len(vlm_model.image_data) == 1
