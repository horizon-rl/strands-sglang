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

"""SGLang model provider for Strands Agents for on-policy agentic RL training."""

from .client import SGLangClient
from .exceptions import (
    SGLangClientError,
    SGLangConnectionError,
    SGLangContextLengthError,
    SGLangDecodingError,
    SGLangHTTPError,
    SGLangThrottledError,
)
from .limiter import (
    LoopLimiter,
    LoopLimitReachedError,
    MaxMessagesReachedError,
    MaxToolCallsReachedError,
    MaxToolIterationsReachedError,
)
from .models import (
    ModelFactory,
    bedrock_mantle_model_factory,
    bedrock_model_factory,
)
from .rollout import Rollout
from .sglang import SGLangModel
from .tool_parsers import get_tool_parser
from .utils import get_client, get_client_from_slime_args, get_tokenizer

__all__ = [
    # Utilities
    "get_client",
    "get_client_from_slime_args",
    "get_tokenizer",
    # Client
    "SGLangClient",
    # Exceptions
    "SGLangClientError",
    "SGLangHTTPError",
    "SGLangContextLengthError",
    "SGLangThrottledError",
    "SGLangConnectionError",
    "SGLangDecodingError",
    # Model
    "SGLangModel",
    # Bedrock model factories
    "ModelFactory",
    "bedrock_model_factory",
    "bedrock_mantle_model_factory",
    # Rollout
    "Rollout",
    # Tool parsing
    "get_tool_parser",
    # Hooks
    "LoopLimiter",
    "LoopLimitReachedError",
    "MaxToolIterationsReachedError",
    "MaxToolCallsReachedError",
    "MaxMessagesReachedError",
]
