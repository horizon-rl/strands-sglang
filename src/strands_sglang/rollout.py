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

"""Rollout trajectory tracking for token-in/token-out training."""

from __future__ import annotations

import numpy as np
import pybase64
from numpy.typing import NDArray
from pydantic import BaseModel, Field


class Rollout(BaseModel):
    """A `Rollout` instance tracks full token-level trajectory and metadata for token-in/token-out (TITO) training.

    Attributes:
        token_ids: Flat token IDs for the whole rollout.
        loss_mask: Flat per-token mask, aligned with `token_ids`. `0` for prompt tokens
            (system, user, tool results); `1` for response tokens (model output).
        logprobs: Flat per-token log probabilities, aligned with `token_ids`.
            `logprobs[0]` is `None` — the first token has no predecessor to score.
        routed_experts: List of base64 encoded routed experts, one per turn.
        image_data: List of base64 encoded image data URLs throughout the rollout.
        segment_info: `(is_output, length)` per appended segment, in order.
    """

    token_ids: list[int] = Field(default_factory=list)
    loss_mask: list[int] = Field(default_factory=list)
    logprobs: list[float | None] = Field(default_factory=list)
    routed_experts: list[str] = Field(default_factory=list)
    image_data: list[str] = Field(default_factory=list)
    segment_info: list[tuple[bool, int]] = Field(default_factory=list)

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

    def decode_routed_experts(
        self, num_layers: int, top_k: int, num_tokens: int | None = None
    ) -> NDArray[np.int32] | None:
        """Decode the routed-experts capture into a ``[rows, num_layers, top_k]`` array.

        Handles BOTH server behaviors without assuming one:

        - **Per-turn-crop servers** honor ``routed_experts_start_len`` and return only this turn's
          rows; the per-turn slices must be JOINED to reconstruct the trajectory.
        - **Full-request servers** ignore ``routed_experts_start_len`` and return routed experts for
          the WHOLE request on every ``/generate`` call; here the LATEST blob already spans the
          entire trajectory and joining all blobs would over-count by ~N for an N-hop rollout.

        Disambiguation: when ``num_tokens`` is given (typically ``len(token_ids) - 1``), we pick the
        interpretation whose row count matches — latest-blob first (full-request server), then
        join-all (per-turn-crop server). When ``num_tokens`` is NOT given we cannot tell the two
        apart, so we keep the legacy per-turn-crop contract and JOIN, falling back to the latest
        blob only if the join does not reshape cleanly. The element width is inferred from the blob
        size (current sglang emits int32, older builds int8) rather than hardcoded.

        Args:
            num_layers: MoE layer count (per-token slab depth).
            top_k: router top-k (per-token slab width).
            num_tokens: if given, the expected row count. When NEITHER interpretation matches it in
                either dtype, returns ``None`` so the caller can drop the sample rather than feed
                misaligned routing into replay. If ``None``, rows are inferred.

        Returns:
            A ``[rows, num_layers, top_k]`` array, or ``None`` if there is no capture or the blob
            size is inconsistent with the requested shape.
        """
        if not self.routed_experts:
            return None
        per_token = num_layers * top_k
        latest = pybase64.b64decode(self.routed_experts[-1].encode("ascii"))
        joined = b"".join(pybase64.b64decode(s.encode("ascii")) for s in self.routed_experts)

        def _reshape(raw: bytes, rows: int | None) -> NDArray[np.int32] | None:
            for dtype in (np.int32, np.int8):
                itemsize = np.dtype(dtype).itemsize
                if len(raw) % itemsize != 0:
                    continue
                count = len(raw) // itemsize
                if rows is not None:
                    if count == rows * per_token:
                        return np.frombuffer(raw, dtype=dtype).reshape(rows, num_layers, top_k)
                elif per_token and count % per_token == 0:
                    return np.frombuffer(raw, dtype=dtype).reshape(-1, num_layers, top_k)
            return None

        if num_tokens is not None:
            # num_tokens disambiguates: full-request (latest spans all) vs per-turn-crop (join).
            candidates = (latest, joined)
        else:
            # Ambiguous: keep the legacy per-turn-crop contract (join), fall back to the latest blob.
            candidates = (joined, latest)
        for raw in candidates:
            decoded = _reshape(raw, num_tokens)
            if decoded is not None:
                return decoded
        return None

    @property
    def initial_prompt_length(self) -> int:
        """Return the length of the initial prompt."""
        return self.segment_info[0][1] if self.segment_info else 0

    def __len__(self) -> int:
        """Return the total number of tokens."""
        return len(self.token_ids)

    def __repr__(self) -> str:
        """Return a concise representation (avoids dumping the full token lists)."""
        return f"Rollout(tokens={len(self.token_ids)}, output_tokens={sum(self.loss_mask)}), segments={len(self.segment_info)}"
