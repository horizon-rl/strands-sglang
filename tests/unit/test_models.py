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

"""Unit tests for the Bedrock model factories."""

from unittest.mock import MagicMock, patch

from strands_sglang.models import bedrock_mantle_model_factory, bedrock_model_factory


class TestBedrockModelFactory:
    """Tests for bedrock_model_factory (Anthropic / Converse-API models)."""

    def test_remaps_max_new_tokens_and_shares_client(self):
        """max_new_tokens is remapped to max_tokens and one client is shared across instances."""
        instances = []

        def make_model(**kwargs):
            model = MagicMock()
            model.kwargs = kwargs
            model.client = MagicMock(name=f"client-{len(instances)}")
            instances.append(model)
            return model

        with patch("strands.models.bedrock.BedrockModel", side_effect=make_model) as bedrock_cls:
            factory = bedrock_model_factory(
                model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
                boto_session=MagicMock(),
                sampling_params={"max_new_tokens": 4096},
            )
            model_a = factory()
            model_b = factory()

        # Pilot instance + two factory calls == three constructions.
        assert bedrock_cls.call_count == 3
        # max_new_tokens was remapped; max_new_tokens must not leak through.
        _, kwargs = bedrock_cls.call_args_list[0]
        assert kwargs["max_tokens"] == 4096
        assert "max_new_tokens" not in kwargs
        assert kwargs["streaming"] is False
        # The pilot client is shared onto every subsequent model.
        assert model_a.client is instances[0].client
        assert model_b.client is instances[0].client

    def test_defaults_session_and_config(self):
        """A missing boto_session/config falls back to defaults without error."""
        with (
            patch("strands.models.bedrock.BedrockModel", side_effect=lambda **k: MagicMock()),
            patch("boto3.Session") as session_cls,
        ):
            factory = bedrock_model_factory(model_id="model-x")
            factory()
        session_cls.assert_called_once_with()


class TestBedrockMantleModelFactory:
    """Tests for bedrock_mantle_model_factory (GPT via OpenAI Responses API)."""

    def test_builds_responses_model_with_token_auth(self):
        """The factory mints a token per call and points the client at the Mantle base URL."""
        responses_cls = MagicMock(name="OpenAIResponsesModel")
        with (
            patch("strands.models.openai_responses.OpenAIResponsesModel", responses_cls),
            patch("aws_bedrock_token_generator.provide_token", return_value="tok-123") as provide_token,
        ):
            factory = bedrock_mantle_model_factory(
                model_id="openai.gpt-5.4-2026-03-05",
                region="us-east-2",
                sampling_params={"max_new_tokens": 16384},
                reasoning={"effort": "high"},
                stateful=False,
            )
            factory()

        provide_token.assert_called_once_with(region="us-east-2")
        _, kwargs = responses_cls.call_args
        assert kwargs["model_id"] == "openai.gpt-5.4-2026-03-05"
        assert kwargs["stateful"] is False
        assert kwargs["client_args"]["base_url"] == "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
        assert kwargs["client_args"]["api_key"] == "tok-123"
        # max_new_tokens -> max_output_tokens; reasoning forwarded.
        assert kwargs["params"]["max_output_tokens"] == 16384
        assert "max_new_tokens" not in kwargs["params"]
        assert kwargs["params"]["reasoning"] == {"effort": "high"}

    def test_stateful_patches_after_invocation_hook(self):
        """stateful=True neutralizes the message-clearing hook so local messages persist."""
        from strands.models.model import _ModelPlugin

        original = _ModelPlugin._on_after_invocation
        try:
            with (
                patch("strands.models.openai_responses.OpenAIResponsesModel", MagicMock()),
                patch("aws_bedrock_token_generator.provide_token", return_value="tok"),
            ):
                bedrock_mantle_model_factory(model_id="openai.gpt-5.4-2026-03-05", stateful=True)
            # Hook is now a no-op that accepts an event and returns None.
            assert _ModelPlugin._on_after_invocation(object()) is None
        finally:
            _ModelPlugin._on_after_invocation = original
