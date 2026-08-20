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
