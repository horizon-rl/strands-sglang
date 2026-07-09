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

"""AWS Bedrock model factories for Strands Agents.

These factories complement the on-policy `SGLangModel` provider with hosted
inference backends, so an agent harness can be pointed at Bedrock during
evaluation, data synthesis, or reward-model rollouts without changing the
`Agent` wiring. Each factory returns a `ModelFactory` — a zero-arg callable
that builds a **fresh** Strands `Model` per rollout so concurrent rollouts stay
isolated.

Two backends are provided:

- `bedrock_model_factory` — Anthropic (and other Converse-API) models via
  `strands.models.bedrock.BedrockModel`.
- `bedrock_mantle_model_factory` — GPT models exposed through the Bedrock Mantle
  OpenAI Responses API, via `strands.models.openai_responses.OpenAIResponsesModel`.

Both depend on optional packages (`boto3`, `strands-agents[openai]`,
`aws-bedrock-token-generator`); imports are deferred so the core provider stays
lightweight. Install with `pip install strands-sglang[bedrock]`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import boto3
    import botocore.config
    from strands.models import Model
    from strands.models.bedrock import BedrockModel
    from strands.models.openai_responses import OpenAIResponsesModel

#: Factory that produces a fresh `Model` per rollout (for concurrent rollout isolation).
ModelFactory = Callable[[], "Model"]

#: Other parameters like temperature and top_p fall back to the model's defaults if unset.
DEFAULT_SAMPLING_PARAMS: dict[str, Any] = {"max_new_tokens": 16384}


def _default_boto_client_config() -> botocore.config.Config:
    """Return a botocore config tuned for high-concurrency, long-running rollouts."""
    import botocore.config

    return botocore.config.Config(
        retries={"max_attempts": 5, "mode": "adaptive"},
        max_pool_connections=100,
        connect_timeout=5.0,
        read_timeout=600.0,
    )


# ---------------------------------------------------------------------------
# Bedrock Model (Anthropic and other Converse-API models)
# ---------------------------------------------------------------------------


def bedrock_model_factory(
    *,
    model_id: str,
    boto_session: boto3.Session | None = None,
    boto_client_config: botocore.config.Config | None = None,
    sampling_params: dict[str, Any] = DEFAULT_SAMPLING_PARAMS,
) -> ModelFactory:
    """Return a factory that creates `BedrockModel` instances.

    Args:
        model_id: Bedrock model ID (e.g. `"us.anthropic.claude-sonnet-4-20250514-v1:0"`).
        boto_session: Boto3 session for AWS credentials. Defaults to a fresh `boto3.Session()`
            (standard credential-resolution chain).
        boto_client_config: Botocore client configuration. Defaults to an adaptive-retry config
            tuned for long-running, high-concurrency rollouts.
        sampling_params: Sampling parameters for the model (e.g. `{"max_new_tokens": 4096}`).

    Notes:
        - A single boto3 client (thread-safe) is created once from the session and shared across
          all model instances. `BedrockModel` doesn't accept a pre-built client, so we extract it
          from a pilot instance and override `model.client` on each subsequent one.
        - The principle of operation is "one boto3 session, one boto3 client".
        - `max_new_tokens` in `sampling_params` is remapped to `max_tokens` for the Bedrock API.
        - Requires the `bedrock` extras: `pip install strands-sglang[bedrock]`.
    """
    import boto3
    from strands.models.bedrock import BedrockModel

    if boto_session is None:
        boto_session = boto3.Session()
    if boto_client_config is None:
        boto_client_config = _default_boto_client_config()

    sampling_params = dict(sampling_params)
    if "max_new_tokens" in sampling_params:
        sampling_params["max_tokens"] = sampling_params.pop("max_new_tokens")

    model_kwargs = dict(
        model_id=model_id,
        boto_session=boto_session,
        boto_client_config=boto_client_config,
        streaming=False,
        **sampling_params,
    )

    # Build one model to extract a properly configured, thread-safe client.
    shared_client = BedrockModel(**model_kwargs).client

    def factory() -> BedrockModel:
        model = BedrockModel(**model_kwargs)
        model.client = shared_client
        return model

    return factory


# ---------------------------------------------------------------------------
# Bedrock Mantle Model (GPT via OpenAI Responses API on AWS)
# ---------------------------------------------------------------------------


def bedrock_mantle_model_factory(
    *,
    model_id: str,
    region: str = "us-east-2",
    sampling_params: dict[str, Any] = DEFAULT_SAMPLING_PARAMS,
    reasoning: dict[str, Any] | None = None,
    stateful: bool = True,
) -> ModelFactory:
    """Return a factory that creates `OpenAIResponsesModel` for GPT models via Bedrock Mantle.

    Args:
        model_id: Bedrock Mantle model ID (e.g. `"openai.gpt-5.4-2026-03-05"`).
        region: AWS region hosting Bedrock Mantle (default `"us-east-2"`).
        sampling_params: Sampling parameters for the model (e.g. `{"max_new_tokens": 16384}`).
        reasoning: Reasoning configuration (e.g. `{"effort": "high"}`).
        stateful: Enable server-side conversation state via previous_response_id.
            When True, reasoning context carries over between turns.

    Notes:
        - Uses `aws_bedrock_token_generator.provide_token()` for SigV4 auth.
        - A fresh token is minted on each factory call to avoid expiry issues.
        - Requires the `bedrock` extras: `pip install strands-sglang[bedrock]`.
    """
    from strands.models.openai_responses import OpenAIResponsesModel

    base_url = f"https://bedrock-mantle.{region}.api.aws/openai/v1"

    sampling_params = dict(sampling_params)
    if "max_new_tokens" in sampling_params:
        sampling_params["max_output_tokens"] = sampling_params.pop("max_new_tokens")

    params: dict[str, Any] = {**sampling_params}
    if reasoning:
        params["reasoning"] = reasoning

    # Patch the class-level hook that clears agent.messages after invocation.
    # Strands clears messages for stateful models (server owns state), but a harness that
    # captures observations from local messages needs them to persist across turns.
    if stateful:
        from strands.models.model import _ModelPlugin

        _ModelPlugin._on_after_invocation = staticmethod(lambda event: None)  # type: ignore[method-assign]

    def factory() -> OpenAIResponsesModel:
        from aws_bedrock_token_generator import provide_token

        token = provide_token(region=region)
        return OpenAIResponsesModel(
            model_id=model_id,
            params=params,
            stateful=stateful,
            client_args={
                "base_url": base_url,
                "api_key": token,
            },
        )

    return factory
