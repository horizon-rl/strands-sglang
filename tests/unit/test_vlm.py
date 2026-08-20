"""Unit tests for VLM (Vision Language Model) support."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from strands.types.exceptions import ContextWindowOverflowException

from strands_sglang import SGLangModel
from strands_sglang.client import SGLangClient
from strands_sglang.exceptions import SGLangContextLengthError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 1x1 red PNG (smallest valid PNG)
_RED_PIXEL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
_RED_PIXEL_B64 = base64.b64encode(_RED_PIXEL_PNG).decode()
_RED_PIXEL_DATA_URL = f"data:image/png;base64,{_RED_PIXEL_B64}"


def _image_block() -> dict:
    """Strands ImageContent block."""
    return {"image": {"format": "png", "source": {"bytes": _RED_PIXEL_PNG}}}


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.name_or_path = "/nonexistent"
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    return tokenizer


@pytest.fixture
def client():
    return SGLangClient(base_url="http://localhost:30000")


@pytest.fixture
def vlm_model(client, mock_tokenizer):
    """SGLangModel in VLM mode."""
    client._is_multimodal = True
    model = SGLangModel(client=client, tokenizer=mock_tokenizer)
    model.__dict__["message_separator"] = ""  # override cached_property (mock has no real template)
    return model


@pytest.fixture
def text_model(client, mock_tokenizer):
    """SGLangModel in text-only mode."""
    client._is_multimodal = False
    model = SGLangModel(client=client, tokenizer=mock_tokenizer)
    model.__dict__["message_separator"] = ""  # override cached_property (mock has no real template)
    return model


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


class TestVLMAutoDetect:
    @pytest.mark.asyncio
    async def test_has_image_understanding_true(self, client):
        """Detects multimodal from server's has_image_understanding field."""
        client.model_info = AsyncMock(return_value={"has_image_understanding": True})
        assert await client.is_multimodal() is True

    @pytest.mark.asyncio
    async def test_has_image_understanding_false(self, client):
        """Text-only models have has_image_understanding=False."""
        client.model_info = AsyncMock(return_value={"has_image_understanding": False})
        assert await client.is_multimodal() is False

    @pytest.mark.asyncio
    async def test_missing_field_defaults_false(self, client):
        """Older SGLang servers without has_image_understanding default to False."""
        client.model_info = AsyncMock(return_value={"model_path": "some/model"})
        assert await client.is_multimodal() is False

    @pytest.mark.asyncio
    async def test_server_unreachable_defaults_false(self, client):
        """When server is unreachable, defaults to False."""
        client.model_info = AsyncMock(return_value=None)
        assert await client.is_multimodal() is False

    @pytest.mark.asyncio
    async def test_result_is_cached(self, client):
        """Second call returns cached result without querying server again."""
        client.model_info = AsyncMock(return_value={"has_image_understanding": True})
        await client.is_multimodal()
        await client.is_multimodal()
        client.model_info.assert_awaited_once()


# ---------------------------------------------------------------------------
# format_content_block
# ---------------------------------------------------------------------------


class TestFormatContentBlockMultimodal:
    """format_content_block with is_multimodal=True returns dicts, not strings."""

    def test_text_block_returns_dict(self):
        result = SGLangModel.format_content_block({"text": "hello"}, is_multimodal=True)
        assert result == {"type": "text", "text": "hello"}

    def test_text_block_returns_string_when_not_multimodal(self):
        result = SGLangModel.format_content_block({"text": "hello"}, is_multimodal=False)
        assert result == "hello"

    def test_image_block_returns_data_url(self):
        result = SGLangModel.format_content_block(_image_block(), is_multimodal=True)
        assert result == {"type": "image", "image": _RED_PIXEL_DATA_URL}

    def test_image_block_raises_when_not_multimodal(self):
        """Image blocks require is_multimodal=True (no 'text' key to flatten to)."""
        with pytest.raises(KeyError):
            SGLangModel.format_content_block(_image_block(), is_multimodal=False)

    def test_json_block_returns_dict(self):
        result = SGLangModel.format_content_block({"json": {"key": "val"}}, is_multimodal=True)
        assert result == {"type": "text", "text": '{"key": "val"}'}


# ---------------------------------------------------------------------------
# format_messages with is_multimodal
# ---------------------------------------------------------------------------


class TestFormatMessagesMultimodal:
    def test_mixed_text_and_image(self):
        """Text + image in same message are grouped into list content."""
        messages = [{"role": "user", "content": [{"text": "what is this?"}, _image_block()]}]
        result = SGLangModel.format_messages(messages, is_multimodal=True)
        assert len(result) == 1
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0] == {"type": "text", "text": "what is this?"}
        assert result[0]["content"][1]["type"] == "image"

    def test_tool_result_with_text_and_image(self):
        """Tool result with text + image grouped into list content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_001",
                            "status": "success",
                            "content": [{"text": "Image loaded"}, _image_block()],
                        }
                    }
                ],
            }
        ]
        result = SGLangModel.format_messages(messages, is_multimodal=True)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0] == {"type": "text", "text": "Image loaded"}
        assert result[0]["content"][1]["type"] == "image"


# ---------------------------------------------------------------------------
# Image accumulation across turns
# ---------------------------------------------------------------------------


class TestImageAccumulation:
    def test_images_accumulated_across_calls(self, vlm_model, mock_tokenizer):
        """collect_image_data returns every image in the conversation, in prompt order."""
        # First turn: one image
        messages1 = [{"role": "user", "content": [{"text": "describe"}, _image_block()]}]
        assert vlm_model.collect_image_data(messages1, is_multimodal=True) == [_RED_PIXEL_DATA_URL]

        # Second turn: simulate assistant response + tool result with screenshot
        vlm_model.processed_messages = 2  # len([user_msg]) + 1 after first generation
        messages2 = [
            {"role": "user", "content": [{"text": "describe"}, _image_block()]},
            {
                "role": "assistant",
                "content": [
                    {"text": "I'll use the tool."},
                    {"toolUse": {"toolUseId": "call_001", "name": "screenshot", "input": {}}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_001",
                            "status": "success",
                            "content": [_image_block()],
                        }
                    }
                ],
            },
        ]
        # 1 from the first user message + 1 from the tool result
        assert vlm_model.collect_image_data(messages2, is_multimodal=True) == [_RED_PIXEL_DATA_URL] * 2

    def test_text_only_collects_nothing(self, vlm_model):
        """A text-only server gets no image_data, even when the messages carry images."""
        messages = [{"role": "user", "content": [{"text": "describe"}, _image_block()]}]
        assert vlm_model.collect_image_data(messages, is_multimodal=False) == []

    def test_reset_clears_accumulated_images(self, vlm_model, mock_tokenizer):
        vlm_model.rollout.image_data = [_RED_PIXEL_DATA_URL]

        vlm_model.reset()
        assert vlm_model.rollout.image_data == []


# ---------------------------------------------------------------------------
# stream() — image_data forwarding
# ---------------------------------------------------------------------------


class TestStreamImageData:
    @pytest.mark.asyncio
    async def test_image_data_passed_to_client(self, vlm_model, mock_tokenizer):
        """When message contains an image, image_data is forwarded to client.generate."""
        with patch.object(vlm_model.client, "generate", new_callable=_async_mock_generate) as mock_gen:
            vlm_model.client.is_multimodal = AsyncMock(return_value=True)
            async for _ in vlm_model.stream(
                messages=[{"role": "user", "content": [{"text": "describe"}, _image_block()]}],
            ):
                pass
            assert mock_gen.call_args.kwargs["image_data"] == [_RED_PIXEL_DATA_URL]

    @pytest.mark.asyncio
    async def test_no_image_data_passes_none(self, text_model, mock_tokenizer):
        """When image_data is empty, image_data=None is passed."""
        with patch.object(text_model.client, "generate", new_callable=_async_mock_generate) as mock_gen:
            text_model.client.is_multimodal = AsyncMock(return_value=False)
            async for _ in text_model.stream(
                messages=[{"role": "user", "content": [{"text": "hello"}]}],
            ):
                pass
            assert mock_gen.call_args.kwargs["image_data"] is None

    @pytest.mark.asyncio
    async def test_failed_generate_does_not_duplicate_images(self, vlm_model, mock_tokenizer):
        """Retrying a turn re-sends the same images, not one copy per attempt.

        Strands re-runs the whole event loop cycle after a `ContextWindowOverflowException`, which
        calls `stream()` again for the same turn. SGLang expands image tokens per entry in
        `image_data`, so a duplicated entry desynchronizes the prompt from the tokens we tracked.
        """
        calls = []

        async def _generate(**kwargs):
            calls.append(list(kwargs["image_data"] or []))
            if len(calls) == 1:
                raise SGLangContextLengthError("too long", status=400, body="context length exceeded")
            return {
                "text": "response",
                "output_ids": [100],
                "meta_info": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop"},
                    "e2e_latency": 0.1,
                },
            }

        vlm_model.client.is_multimodal = AsyncMock(return_value=True)
        messages = [{"role": "user", "content": [{"text": "describe"}, _image_block()]}]

        with patch.object(vlm_model.client, "generate", side_effect=_generate):
            with pytest.raises(ContextWindowOverflowException):
                async for _ in vlm_model.stream(messages=messages):
                    pass
            async for _ in vlm_model.stream(messages=messages):
                pass

        assert calls == [[_RED_PIXEL_DATA_URL], [_RED_PIXEL_DATA_URL]]
        assert vlm_model.rollout.image_data == [_RED_PIXEL_DATA_URL]


def _async_mock_generate():
    """Factory for async mock of client.generate."""
    mock = MagicMock()

    async def _generate(**kwargs):
        return {
            "text": "response",
            "output_ids": [100, 101],
            "meta_info": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop"},
                "e2e_latency": 0.1,
            },
        }

    mock.side_effect = _generate
    return mock
