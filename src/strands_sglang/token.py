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

"""Token management for TITO (Token-In/Token-Out) training.

This module provides:
- Token: A single token with ID, logprob, and output mask
- TokenManager: Manages segment-based token accumulation

For RL training, you typically want:
- token_ids: Flat list of all tokens for the trajectory
- output_mask: Boolean mask for loss computation (True = model output)
- logprobs: Log probabilities for policy gradient
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True, slots=True)
class Token:
    """A single token with its ID, logprob, and loss mask.

    Attributes:
        token_id: The token ID from the tokenizer vocabulary.
        logprob: Log probability (None if not available).
        loss_mask: Whether to include this token in loss computation.
    """

    token_id: int
    logprob: float | None = None
    loss_mask: bool = True


class TokenManager:
    """Manages token accumulation with segment-based prompt/response tracking.

    Tokens are organized into segments, where each segment is either:
    - PROMPT: System messages, user input, tool results (loss_mask=False)
    - RESPONSE: Model outputs (loss_mask=True)

    Example:
        >>> manager = TokenManager(tokenizer)
        >>> manager.add_prompt([1, 2, 3])
        >>> manager.add_response([4, 5], [0.1, 0.2])
        >>> manager.token_ids      # [1, 2, 3, 4, 5]
        >>> manager.output_mask    # [False, False, False, True, True]
        >>> manager.logprobs       # [None, None, None, 0.1, 0.2]
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase | None = None) -> None:
        """Create a TokenManager.

        Args:
            tokenizer: Optional HuggingFace tokenizer for decoding tokens.
        """
        self.tokenizer = tokenizer
        self._segments: list[list[Token]] = []

    def reset(self) -> None:
        """Reset token accumulation for a new episode."""
        self._segments = []

    def add_prompt(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        """Add a prompt segment (system messages, user input, tool results).

        Args:
            token_ids: Token IDs for this segment.
            logprobs: Optional log probabilities (from forward pass).
        """
        if not token_ids:
            return

        tokens = [
            Token(
                token_id=tid,
                logprob=logprobs[i] if logprobs and i < len(logprobs) else None,
                loss_mask=False,
            )
            for i, tid in enumerate(token_ids)
        ]
        self._segments.append(tokens)

    def add_response(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        """Add a response segment (model output).

        Args:
            token_ids: Token IDs for this segment.
            logprobs: Optional log probabilities for each token.
        """
        if not token_ids:
            return

        tokens = [
            Token(
                token_id=tid,
                logprob=logprobs[i] if logprobs and i < len(logprobs) else None,
                loss_mask=True,
            )
            for i, tid in enumerate(token_ids)
        ]
        self._segments.append(tokens)

    @property
    def tokens(self) -> list[Token]:
        """Get all tokens as a flat list."""
        return [token for segment in self._segments for token in segment]

    @property
    def token_ids(self) -> list[int]:
        """Get all token IDs as a flat list."""
        return [token.token_id for token in self.tokens]

    @property
    def loss_mask(self) -> list[bool]:
        """Get loss mask for all tokens.

        Use this for loss computation in RL training - only compute loss
        on tokens where mask is True (model outputs).
        """
        return [token.loss_mask for token in self.tokens]

    @property
    def logprobs(self) -> list[float | None]:
        """Get log probabilities for all tokens."""
        return [token.logprob for token in self.tokens]

    @property
    def segments(self) -> list[list[Token]]:
        """Get tokens organized by segment."""
        return [list(seg) for seg in self._segments]

    @property
    def segment_info(self) -> list[tuple[bool, int]]:
        """Get segment metadata (is_output, length) for each segment.

        Returns:
            List of (is_output, segment_length) tuples.
        """
        return [(seg[0].loss_mask if seg else False, len(seg)) for seg in self._segments]

    def decode(self, token: Token) -> str:
        """Decode a single token to text.

        Args:
            token: Token to decode.

        Returns:
            Decoded text string.

        Raises:
            ValueError: If no tokenizer is available.
        """
        if self.tokenizer is None:
            raise ValueError("No tokenizer available for decoding")
        return self.tokenizer.decode([token.token_id])

    def __len__(self) -> int:
        """Return total number of tokens."""
        return sum(len(seg) for seg in self._segments)

    def __repr__(self) -> str:
        """Return string representation."""
        n_segments = len(self._segments)
        n_tokens = len(self)
        n_output = sum(1 for token in self.tokens if token.loss_mask)
        return f"TokenManager(segments={n_segments}, tokens={n_tokens}, output_tokens={n_output})"
