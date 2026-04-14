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

"""Integration tests for SGLangClient error classification."""

import pytest

from strands_sglang.client import SGLangClient
from strands_sglang.exceptions import SGLangConnectionError, SGLangContextLengthError


async def test_client_error_classification(sglang_base_url):
    """Context-length and connection errors are classified into correct exception types."""
    # Context length: oversized input triggers SGLangContextLengthError
    # Validates CONTEXT_LENGTH_PATTERNS match real server error text
    async with SGLangClient(base_url=sglang_base_url, max_retries=0) as client:
        oversized_input_ids = [1] * 400_000
        with pytest.raises(SGLangContextLengthError) as exc_info:
            await client.generate(input_ids=oversized_input_ids)
        assert exc_info.value.status == 400
        assert len(exc_info.value.body) > 0

    # Connection error: dead port triggers SGLangConnectionError
    async with SGLangClient(base_url="http://localhost:1", max_retries=0, connect_timeout=1.0) as client:
        with pytest.raises(SGLangConnectionError):
            await client.generate(input_ids=[1, 2, 3])
