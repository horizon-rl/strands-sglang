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

"""Unit tests for VLM (Vision Language Model) support."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

from strands_sglang import SGLangModel
from strands_sglang.client import SGLangClient

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
def mock_processor():
    """Mock HF processor with tokenizer sub-object."""
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    processor.tokenizer.encode.return_value = [1, 2, 3]
    processor.tokenizer.apply_chat_template.return_value = "formatted prompt"
    # processor(text=..., images=...) returns {"input_ids": [[...]]} (batch dim)
    processor.return_value = {"input_ids": [[10, 20, 30]]}
    return processor


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for text-only comparison."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    return tokenizer


@pytest.fixture
def client():
    return SGLangClient(base_url="http://localhost:30000")


@pytest.fixture
def vlm_model(client, mock_processor):
    """SGLangModel with processor (VLM mode)."""
    return SGLangModel(client=client, processor=mock_processor)


@pytest.fixture
def text_model(client, mock_tokenizer):
    """SGLangModel with tokenizer only (text-only mode)."""
    return SGLangModel(client=client, tokenizer=mock_tokenizer)


# ---------------------------------------------------------------------------
# Constructor and properties
# ---------------------------------------------------------------------------


class TestVLMConstructor:
    def test_processor_stored(self, vlm_model, mock_processor):
        assert vlm_model.processor is mock_processor

    def test_tokenizer_from_processor(self, vlm_model, mock_processor):
        assert vlm_model.tokenizer is mock_processor.tokenizer

    def test_is_multimodal_true(self, vlm_model):
        assert vlm_model.is_multimodal is True

    def test_is_multimodal_false(self, text_model):
        assert text_model.is_multimodal is False

    def test_neither_tokenizer_nor_processor_raises(self, client):
        with pytest.raises(ValueError, match="Either tokenizer"):
            SGLangModel(client=client)

    def test_tokenizer_only(self, client, mock_tokenizer):
        model = SGLangModel(client=client, tokenizer=mock_tokenizer)
        assert model.processor is None
        assert model.tokenizer is mock_tokenizer

    def test_both_tokenizer_and_processor_uses_processor_tokenizer(self, client, mock_processor, mock_tokenizer):
        model = SGLangModel(client=client, tokenizer=mock_tokenizer, processor=mock_processor)
        assert model.tokenizer is mock_processor.tokenizer


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
    def test_text_message_multimodal(self):
        messages = [{"role": "user", "content": [{"text": "describe this"}]}]
        result = SGLangModel.format_messages(messages, is_multimodal=True)
        assert result == [{"role": "user", "content": [{"type": "text", "text": "describe this"}]}]

    def test_image_message_multimodal(self):
        messages = [{"role": "user", "content": [_image_block()]}]
        result = SGLangModel.format_messages(messages, is_multimodal=True)
        assert len(result) == 1
        assert result[0]["content"][0]["type"] == "image"
        assert result[0]["content"][0]["image"] == _RED_PIXEL_DATA_URL

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

    def test_tool_result_single_block(self):
        """Tool result with single block still gets list content in multimodal mode."""
        messages = [
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
            }
        ]
        result = SGLangModel.format_messages(messages, is_multimodal=True)
        assert result[0]["role"] == "tool"
        assert result[0]["content"][0]["type"] == "image"


# ---------------------------------------------------------------------------
# extract_image_urls
# ---------------------------------------------------------------------------


class TestExtractImageUrls:
    def test_extracts_from_image_messages(self):
        messages = [
            {"role": "user", "content": {"type": "image", "image": "data:image/png;base64,abc"}},
            {"role": "user", "content": {"type": "text", "text": "describe this"}},
            {"role": "user", "content": {"type": "image", "image": "data:image/jpeg;base64,xyz"}},
        ]
        result = SGLangModel.extract_image_urls(messages)
        assert result == ["data:image/png;base64,abc", "data:image/jpeg;base64,xyz"]

    def test_no_images(self):
        messages = [{"role": "user", "content": "plain text"}]
        assert SGLangModel.extract_image_urls(messages) == []

    def test_skips_string_content(self):
        """Text-only formatted messages have string content — should be skipped."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": {"type": "image", "image": "data:image/png;base64,abc"}},
        ]
        assert SGLangModel.extract_image_urls(messages) == ["data:image/png;base64,abc"]

    def test_extracts_from_list_content(self):
        """Images in list content (grouped multimodal messages) are extracted."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image", "image": "data:image/png;base64,abc"},
                ],
            }
        ]
        assert SGLangModel.extract_image_urls(messages) == ["data:image/png;base64,abc"]


# ---------------------------------------------------------------------------
# tokenize_prompt_messages — VLM path
# ---------------------------------------------------------------------------


class TestTokenizePromptMessagesVLM:
    def test_first_call_uses_processor(self, vlm_model, mock_processor):
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        result = vlm_model.tokenize_prompt_messages(messages, system_prompt=None)

        assert result == [10, 20, 30]
        mock_processor.assert_called_once()
        call_kwargs = mock_processor.call_args.kwargs
        assert "text" in call_kwargs

    def test_first_call_with_images_passes_image_data(self, vlm_model, mock_processor):
        messages = [{"role": "user", "content": [{"text": "describe"}, _image_block()]}]
        vlm_model.tokenize_prompt_messages(messages, system_prompt=None)

        call_kwargs = mock_processor.call_args.kwargs
        assert call_kwargs["images"] is not None
        assert len(call_kwargs["images"]) == 1

    def test_text_only_message_passes_images_none(self, vlm_model, mock_processor):
        messages = [{"role": "user", "content": [{"text": "no images here"}]}]
        vlm_model.tokenize_prompt_messages(messages, system_prompt=None)

        call_kwargs = mock_processor.call_args.kwargs
        assert call_kwargs["images"] is None

    def test_text_model_uses_tokenizer_encode(self, text_model, mock_tokenizer):
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        result = text_model.tokenize_prompt_messages(messages, system_prompt=None)

        assert result == [1, 2, 3]
        mock_tokenizer.encode.assert_called_once()

    def test_subsequent_call_uses_processor(self, vlm_model, mock_processor):
        """Incremental tokenization also goes through processor when VLM."""
        vlm_model.token_manager.add_prompt([1, 2, 3])
        vlm_model._processed_message_count = 1

        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
            {"role": "user", "content": [{"text": "New message"}]},
        ]
        result = vlm_model.tokenize_prompt_messages(messages, system_prompt=None)

        assert result == [10, 20, 30]
        mock_processor.assert_called_once()


# ---------------------------------------------------------------------------
# Image accumulation across turns
# ---------------------------------------------------------------------------


class TestImageAccumulation:
    def test_images_accumulated_across_calls(self, vlm_model, mock_processor):
        """image_data grows across multiple format_prompt calls."""
        # First turn: one image
        messages1 = [{"role": "user", "content": [{"text": "describe"}, _image_block()]}]
        vlm_model.tokenize_prompt_messages(messages1, system_prompt=None)
        assert len(vlm_model.image_data) == 1

        # Second turn: another image (simulate tool result with screenshot)
        vlm_model.token_manager.add_prompt([10, 20, 30])
        vlm_model._processed_message_count = 1
        messages2 = [
            {"role": "user", "content": [{"text": "describe"}, _image_block()]},
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
        vlm_model.tokenize_prompt_messages(messages2, system_prompt=None)
        assert len(vlm_model.image_data) == 2  # 1 from first + 1 from second (tool result image)

    def test_reset_clears_accumulated_images(self, vlm_model, mock_processor):
        messages = [{"role": "user", "content": [{"text": "describe"}, _image_block()]}]
        vlm_model.tokenize_prompt_messages(messages, system_prompt=None)
        assert len(vlm_model.image_data) > 0

        vlm_model.reset()
        assert vlm_model.image_data == []


# ---------------------------------------------------------------------------
# stream() — image_data forwarding
# ---------------------------------------------------------------------------


class TestStreamImageData:
    @pytest.mark.asyncio
    async def test_image_data_passed_to_client(self, vlm_model, mock_processor):
        """When image_data is non-empty, it's forwarded to client.generate."""
        vlm_model.image_data = [_RED_PIXEL_DATA_URL]

        with patch.object(vlm_model.client, "generate", new_callable=_async_mock_generate) as mock_gen:
            async for _ in vlm_model.stream(
                messages=[{"role": "user", "content": [{"text": "describe"}]}],
            ):
                pass
            assert mock_gen.call_args.kwargs["image_data"] == [_RED_PIXEL_DATA_URL]

    @pytest.mark.asyncio
    async def test_no_image_data_passes_none(self, text_model, mock_tokenizer):
        """When image_data is empty, image_data=None is passed."""
        with patch.object(text_model.client, "generate", new_callable=_async_mock_generate) as mock_gen:
            async for _ in text_model.stream(
                messages=[{"role": "user", "content": [{"text": "hello"}]}],
            ):
                pass
            assert mock_gen.call_args.kwargs["image_data"] is None


def _async_mock_generate():
    """Factory for async mock of client.generate that records call kwargs."""
    mock = MagicMock()

    async def _generate(**kwargs):
        return {"text": "response", "output_ids": [100, 101], "meta_info": {}}

    mock.side_effect = _generate
    return mock
