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

"""Unit tests for token module."""

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


class TestTokenManagerBasic:
    """Basic TokenManager tests."""

    def test_init(self):
        """TokenManager starts empty."""
        manager = TokenManager()
        assert len(manager) == 0

    def test_empty_manager(self):
        """Empty manager returns empty lists."""
        manager = TokenManager()
        assert manager.tokens == []
        assert manager.token_ids == []
        assert manager.loss_mask == []
        assert manager.logprobs == []
        assert manager.segments == []
        assert manager.segment_info == []


class TestTokenManagerAddPrompt:
    """Tests for add_prompt method."""

    def test_add_prompt_basic(self):
        """add_prompt adds tokens with loss_mask=False."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])

        assert manager.token_ids == [1, 2, 3]
        assert manager.loss_mask == [False, False, False]
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

    def test_add_prompt_mismatched_logprobs_raises(self):
        """add_prompt raises ValueError if logprobs length doesn't match token_ids."""
        manager = TokenManager()
        with pytest.raises(ValueError, match="logprobs length"):
            manager.add_prompt([1, 2, 3], logprobs=[-0.1])


class TestTokenManagerAddResponse:
    """Tests for add_response method."""

    def test_add_response_basic(self):
        """add_response adds tokens with loss_mask=True."""
        manager = TokenManager()
        manager.add_prompt([1])
        manager.add_response([4, 5, 6])

        assert manager.token_ids == [1, 4, 5, 6]
        assert manager.loss_mask == [False, True, True, True]

    def test_add_response_with_logprobs(self):
        """add_response accepts logprobs."""
        manager = TokenManager()
        manager.add_prompt([1])
        manager.add_response([100, 200], logprobs=[-0.5, -0.6])

        assert manager.logprobs[1:] == [-0.5, -0.6]

    def test_add_response_empty(self):
        """add_response with empty list does nothing."""
        manager = TokenManager()
        manager.add_response([])
        assert len(manager) == 0

    def test_add_response_without_prompt_raises(self):
        """add_response raises RuntimeError if no prompt segment exists."""
        manager = TokenManager()
        with pytest.raises(RuntimeError, match="First segment must be a prompt"):
            manager.add_response([4, 5, 6])

    def test_add_response_mismatched_logprobs_raises(self):
        """add_response raises ValueError if logprobs length doesn't match token_ids."""
        manager = TokenManager()
        manager.add_prompt([0])
        with pytest.raises(ValueError, match="logprobs length"):
            manager.add_response([1, 2, 3], logprobs=[-0.1, -0.2])


class TestTokenManagerMultipleSegments:
    """Tests for multiple segment operations."""

    def test_prompt_response_sequence(self):
        """Typical prompt-response sequence works correctly."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5], logprobs=[-0.1, -0.2])

        assert manager.token_ids == [1, 2, 3, 4, 5]
        assert manager.loss_mask == [False, False, False, True, True]
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
        assert manager.loss_mask == [False, False, True, True, False, False, True, True]
        assert manager.logprobs == [None, None, -0.1, -0.2, None, None, -0.3, -0.4]
        assert len(manager) == 8

    def test_segments_property(self):
        """segments property returns segment data."""
        manager = TokenManager()
        manager.add_prompt([1, 2])
        manager.add_response([3, 4])

        segments = manager.segments
        assert len(segments) == 2
        assert all(not t.loss_mask for t in segments[0])
        assert all(t.loss_mask for t in segments[1])

    def test_initial_prompt(self):
        """initial_prompt returns the first segment."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5])

        prompt = manager.initial_prompt
        assert [t.token_id for t in prompt] == [1, 2, 3]
        assert all(not t.loss_mask for t in prompt)

    def test_initial_prompt_empty(self):
        """initial_prompt returns empty list when no segments exist."""
        manager = TokenManager()
        assert manager.initial_prompt == []

    def test_segment_info(self):
        """segment_info returns correct metadata."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5])
        manager.add_prompt([6])

        assert manager.segment_info == [(False, 3), (True, 2), (False, 1)]


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
