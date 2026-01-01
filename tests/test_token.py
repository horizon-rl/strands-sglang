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

"""Unit tests for token module."""

from unittest.mock import MagicMock

import pytest

from strands_sglang import Token, TokenManager


class TestToken:
    """Tests for Token dataclass."""

    def test_token_defaults(self):
        """Token has correct default values."""
        token = Token(token_id=42)
        assert token.token_id == 42
        assert token.logprob is None
        assert token.loss_mask is True

    def test_token_with_all_fields(self):
        """Token accepts all field values."""
        token = Token(token_id=100, logprob=-0.5, loss_mask=False)
        assert token.token_id == 100
        assert token.logprob == -0.5
        assert token.loss_mask is False

    def test_token_is_frozen(self):
        """Token is immutable (frozen=True)."""
        token = Token(token_id=1)
        with pytest.raises(AttributeError):
            token.token_id = 2

    def test_token_is_hashable(self):
        """Frozen tokens are hashable (can be used in sets/dicts)."""
        token1 = Token(token_id=1, logprob=-0.1, loss_mask=True)
        token2 = Token(token_id=1, logprob=-0.1, loss_mask=True)
        token3 = Token(token_id=2)

        assert hash(token1) == hash(token2)
        assert {token1, token2, token3} == {token1, token3}

    def test_token_equality(self):
        """Tokens with same values are equal."""
        token1 = Token(token_id=5, logprob=-0.2, loss_mask=False)
        token2 = Token(token_id=5, logprob=-0.2, loss_mask=False)
        token3 = Token(token_id=5, logprob=-0.3, loss_mask=False)

        assert token1 == token2
        assert token1 != token3


class TestTokenManagerBasic:
    """Basic TokenManager tests."""

    def test_init_without_tokenizer(self):
        """TokenManager can be created without tokenizer."""
        manager = TokenManager()
        assert manager.tokenizer is None
        assert len(manager) == 0

    def test_init_with_tokenizer(self):
        """TokenManager can be created with tokenizer."""
        tokenizer = MagicMock()
        manager = TokenManager(tokenizer=tokenizer)
        assert manager.tokenizer is tokenizer

    def test_empty_manager(self):
        """Empty manager returns empty lists."""
        manager = TokenManager()
        assert manager.tokens == []
        assert manager.token_ids == []
        assert manager.output_mask == []
        assert manager.logprobs == []
        assert manager.segments == []
        assert manager.segment_info == []

    def test_repr_empty(self):
        """Repr shows correct info for empty manager."""
        manager = TokenManager()
        assert repr(manager) == "TokenManager(segments=0, tokens=0, output_tokens=0)"


class TestTokenManagerAddPrompt:
    """Tests for add_prompt method."""

    def test_add_prompt_basic(self):
        """add_prompt adds tokens with loss_mask=False."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])

        assert manager.token_ids == [1, 2, 3]
        assert manager.output_mask == [False, False, False]
        assert manager.logprobs == [None, None, None]

    def test_add_prompt_with_logprobs(self):
        """add_prompt accepts logprobs."""
        manager = TokenManager()
        manager.add_prompt([10, 20], logprobs=[-0.1, -0.2])

        assert manager.token_ids == [10, 20]
        assert manager.logprobs == [-0.1, -0.2]

    def test_add_prompt_empty(self):
        """add_prompt with empty list does nothing."""
        manager = TokenManager()
        manager.add_prompt([])
        assert len(manager) == 0
        assert manager.segments == []

    def test_add_prompt_partial_logprobs(self):
        """add_prompt handles partial logprobs gracefully."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3], logprobs=[-0.1])

        assert manager.logprobs == [-0.1, None, None]


class TestTokenManagerAddResponse:
    """Tests for add_response method."""

    def test_add_response_basic(self):
        """add_response adds tokens with loss_mask=True."""
        manager = TokenManager()
        manager.add_response([4, 5, 6])

        assert manager.token_ids == [4, 5, 6]
        assert manager.output_mask == [True, True, True]
        assert manager.logprobs == [None, None, None]

    def test_add_response_with_logprobs(self):
        """add_response accepts logprobs."""
        manager = TokenManager()
        manager.add_response([100, 200], logprobs=[-0.5, -0.6])

        assert manager.token_ids == [100, 200]
        assert manager.logprobs == [-0.5, -0.6]

    def test_add_response_empty(self):
        """add_response with empty list does nothing."""
        manager = TokenManager()
        manager.add_response([])
        assert len(manager) == 0

    def test_add_response_partial_logprobs(self):
        """add_response handles partial logprobs gracefully."""
        manager = TokenManager()
        manager.add_response([1, 2, 3], logprobs=[-0.1, -0.2])

        assert manager.logprobs == [-0.1, -0.2, None]


class TestTokenManagerMultipleSegments:
    """Tests for multiple segment operations."""

    def test_prompt_response_sequence(self):
        """Typical prompt-response sequence works correctly."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5], logprobs=[-0.1, -0.2])

        assert manager.token_ids == [1, 2, 3, 4, 5]
        assert manager.output_mask == [False, False, False, True, True]
        assert manager.logprobs == [None, None, None, -0.1, -0.2]

    def test_multi_turn_conversation(self):
        """Multi-turn conversation with tool calls."""
        manager = TokenManager()

        # Initial prompt
        manager.add_prompt([1, 2])
        # Model response (tool call)
        manager.add_response([3, 4], logprobs=[-0.1, -0.2])
        # Tool result (treated as prompt)
        manager.add_prompt([5, 6])
        # Final model response
        manager.add_response([7, 8], logprobs=[-0.3, -0.4])

        assert manager.token_ids == [1, 2, 3, 4, 5, 6, 7, 8]
        assert manager.output_mask == [False, False, True, True, False, False, True, True]
        assert manager.logprobs == [None, None, -0.1, -0.2, None, None, -0.3, -0.4]
        assert len(manager) == 8

    def test_segments_property(self):
        """segments property returns copy of segment data."""
        manager = TokenManager()
        manager.add_prompt([1, 2])
        manager.add_response([3, 4])

        segments = manager.segments
        assert len(segments) == 2
        assert len(segments[0]) == 2
        assert len(segments[1]) == 2
        assert all(not t.loss_mask for t in segments[0])
        assert all(t.loss_mask for t in segments[1])

    def test_segment_info(self):
        """segment_info returns correct metadata."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5])
        manager.add_prompt([6])

        info = manager.segment_info
        assert info == [(False, 3), (True, 2), (False, 1)]

    def test_repr_with_data(self):
        """repr shows correct counts."""
        manager = TokenManager()
        manager.add_prompt([1, 2])
        manager.add_response([3, 4, 5])

        assert repr(manager) == "TokenManager(segments=2, tokens=5, output_tokens=3)"


class TestTokenManagerReset:
    """Tests for reset functionality."""

    def test_reset_clears_all(self):
        """reset clears all accumulated tokens."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5])

        manager.reset()

        assert len(manager) == 0
        assert manager.tokens == []
        assert manager.segments == []

    def test_reset_allows_reuse(self):
        """Manager can be reused after reset."""
        manager = TokenManager()
        manager.add_prompt([1, 2])
        manager.add_response([3, 4])

        manager.reset()

        manager.add_prompt([10, 20])
        manager.add_response([30])

        assert manager.token_ids == [10, 20, 30]
        assert len(manager) == 3


class TestTokenManagerDecode:
    """Tests for decode functionality."""

    def test_decode_with_tokenizer(self):
        """decode works when tokenizer is available."""
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "hello"

        manager = TokenManager(tokenizer=tokenizer)
        token = Token(token_id=42)

        result = manager.decode(token)

        assert result == "hello"
        tokenizer.decode.assert_called_once_with([42])

    def test_decode_without_tokenizer(self):
        """decode raises error when no tokenizer."""
        manager = TokenManager()
        token = Token(token_id=42)

        with pytest.raises(ValueError, match="No tokenizer available"):
            manager.decode(token)


class TestTokenManagerTokensProperty:
    """Tests for tokens property and iteration."""

    def test_tokens_returns_flat_list(self):
        """tokens property returns all tokens in order."""
        manager = TokenManager()
        manager.add_prompt([1, 2])
        manager.add_response([3, 4])

        tokens = manager.tokens
        assert len(tokens) == 4
        assert [t.token_id for t in tokens] == [1, 2, 3, 4]

    def test_len_counts_all_tokens(self):
        """__len__ returns total token count across segments."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4])
        manager.add_prompt([5, 6])

        assert len(manager) == 6


class TestEdgeCases:
    """Edge case tests."""

    def test_single_token_segments(self):
        """Single-token segments work correctly."""
        manager = TokenManager()
        manager.add_prompt([1])
        manager.add_response([2])
        manager.add_prompt([3])

        assert manager.token_ids == [1, 2, 3]
        assert manager.output_mask == [False, True, False]

    def test_logprobs_longer_than_tokens(self):
        """Extra logprobs are ignored (only uses indices up to token count)."""
        manager = TokenManager()
        manager.add_prompt([1, 2], logprobs=[-0.1, -0.2, -0.3, -0.4])

        # Only first 2 logprobs used
        assert manager.logprobs == [-0.1, -0.2]

    def test_zero_logprob(self):
        """Zero logprob is valid and preserved."""
        manager = TokenManager()
        manager.add_response([1], logprobs=[0.0])
        assert manager.logprobs == [0.0]

    def test_negative_token_ids(self):
        """Negative token IDs are accepted (some tokenizers use them)."""
        manager = TokenManager()
        manager.add_prompt([-1, -100])
        assert manager.token_ids == [-1, -100]

    def test_large_token_ids(self):
        """Large token IDs are handled."""
        manager = TokenManager()
        manager.add_prompt([100000, 999999])
        assert manager.token_ids == [100000, 999999]
