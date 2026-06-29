"""Unit tests for Rollout.decode_routed_experts (multi-turn latest-blob decode)."""

from __future__ import annotations

import numpy as np
import pybase64

from strands_sglang.rollout import Rollout

_NUM_LAYERS = 4
_TOP_K = 8


def _blob(num_tokens: int, dtype) -> str:
    """Base64-encode a [num_tokens, num_layers, top_k] routed-experts array of the given dtype."""
    arr = np.arange(num_tokens * _NUM_LAYERS * _TOP_K, dtype=dtype)
    return pybase64.b64encode(arr.tobytes()).decode("ascii")


def test_decodes_latest_blob_not_concatenation():
    # The server returns the FULL request on every /generate, so each hop's blob spans the whole
    # trajectory-so-far. decode must use only the LATEST blob, not join all of them.
    r = Rollout()
    r.add_routed_experts(_blob(3, np.int32))  # hop 1: 3 tokens
    r.add_routed_experts(_blob(7, np.int32))  # hop 2: full trajectory = 7 tokens
    out = r.decode_routed_experts(_NUM_LAYERS, _TOP_K, num_tokens=7)
    assert out is not None
    assert out.shape == (7, _NUM_LAYERS, _TOP_K)


def test_int8_dtype_is_detected():
    r = Rollout()
    r.add_routed_experts(_blob(5, np.int8))
    out = r.decode_routed_experts(_NUM_LAYERS, _TOP_K, num_tokens=5)
    assert out is not None
    assert out.shape == (5, _NUM_LAYERS, _TOP_K)


def test_shape_mismatch_returns_none():
    # Blob inconsistent with the requested num_tokens -> None (caller drops the sample, no crash).
    r = Rollout()
    r.add_routed_experts(_blob(6, np.int32))
    assert r.decode_routed_experts(_NUM_LAYERS, _TOP_K, num_tokens=9) is None


def test_num_tokens_none_infers_rows():
    # Backward-compatible 2-arg path: rows inferred from blob size.
    r = Rollout()
    r.add_routed_experts(_blob(4, np.int32))
    out = r.decode_routed_experts(_NUM_LAYERS, _TOP_K)
    assert out is not None
    assert out.shape == (4, _NUM_LAYERS, _TOP_K)


def test_empty_capture_returns_none():
    assert Rollout().decode_routed_experts(_NUM_LAYERS, _TOP_K) is None
    assert Rollout().decode_routed_experts(_NUM_LAYERS, _TOP_K, num_tokens=5) is None
