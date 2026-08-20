from __future__ import annotations

import numpy as np
import pybase64
from numpy.typing import NDArray
from pydantic import BaseModel, Field


class Rollout(BaseModel):
    """The full token-level trajectory of one rollout, for token-in/token-out (TITO) training.

    `token_ids`, `loss_mask` and `logprobs` are flat and index-aligned; `_add_segment` is the one
    place that keeps them the same length.
    """

    token_ids: list[int] = Field(default_factory=list, description="Flat token IDs for the whole rollout.")
    loss_mask: list[int] = Field(
        default_factory=list,
        description="0 for prompt tokens (system, user, tool results), 1 for model output.",
    )
    logprobs: list[float | None] = Field(
        default_factory=list,
        description=(
            "Per-token log probabilities, `None` where the server returned none — the very first "
            "token has no predecessor to score, and a segment appended without logprobs is all None."
        ),
    )
    routed_experts: list[str] = Field(default_factory=list, description="Base64 routed-experts slice, one per turn.")
    image_data: list[str] = Field(default_factory=list, description="Base64 image data URLs seen in the rollout.")
    segment_info: list[tuple[bool, int]] = Field(
        default_factory=list, description="`(is_output, length)` per appended segment, in order."
    )

    def _add_segment(self, token_ids: list[int], logprobs: list[float | None] | None, *, is_output: bool) -> None:
        """Extend the flat fields with one segment and record its boundary.

        Single place that maintains the cross-field length invariant (`token_ids`/`loss_mask`/
        `logprobs` stay equal length): Pydantic does not validate in-place list mutation, so the
        length guard lives here.
        """
        if not token_ids:
            return
        if logprobs is not None and len(logprobs) != len(token_ids):
            raise ValueError(f"logprobs length ({len(logprobs)}) must match token_ids length ({len(token_ids)})")

        n = len(token_ids)
        self.token_ids.extend(token_ids)
        self.loss_mask.extend([int(is_output)] * n)
        self.logprobs.extend(logprobs if logprobs is not None else [None] * n)
        self.segment_info.append((is_output, n))

    def add_prompt(self, token_ids: list[int], logprobs: list[float | None] | None = None) -> None:
        """Append a prompt segment (system messages, user input, tool results); loss_mask=0."""
        self._add_segment(token_ids, logprobs, is_output=False)

    def add_response(self, token_ids: list[int], logprobs: list[float | None] | None = None) -> None:
        """Append a response segment (model output); loss_mask=1."""
        if token_ids and not self.segment_info:
            raise RuntimeError("First segment must be a prompt. Call add_prompt() before add_response().")
        self._add_segment(token_ids, logprobs, is_output=True)

    def add_routed_experts(self, routed_experts: str) -> None:
        """Append this turn's base64 routed-experts slice (one `/generate` call covers the turn)."""
        self.routed_experts.append(routed_experts)

    def decode_routed_experts(self, num_layers: int, top_k: int) -> NDArray[np.int32]:
        """Decode and stitch per-turn routed-experts slices into a shaped numpy array."""
        buffer = b"".join(pybase64.b64decode(s.encode("ascii")) for s in self.routed_experts)
        return np.frombuffer(buffer, dtype=np.int32).reshape(-1, num_layers, top_k)

    @property
    def initial_prompt_length(self) -> int:
        return self.segment_info[0][1] if self.segment_info else 0

    def __len__(self) -> int:
        return len(self.token_ids)

    def __repr__(self) -> str:
        """Concise — dumping the full token lists would be thousands of ints."""
        return f"Rollout(tokens={len(self.token_ids)}, output_tokens={sum(self.loss_mask)}), segments={len(self.segment_info)}"
