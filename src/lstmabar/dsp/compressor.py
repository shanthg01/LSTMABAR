"""Differentiable feed-forward compressor.

Parameters are ``(B,)`` tensors in physical units; audio is ``(B, T)``.
"""

from torch import Tensor


def compress(
    x: Tensor,
    threshold_db: Tensor,
    ratio: Tensor,
    attack_ms: Tensor,
    release_ms: Tensor,
    sample_rate: int,
    knee_db: float = 6.0,
    makeup_db: Tensor | None = None,
) -> Tensor:
    """Feed-forward compressor: level detection → soft-knee gain computer → attack/release
    smoothing of the gain → apply gain (+ optional makeup).

    Must be differentiable w.r.t. ``x`` and all parameters, and fast enough for training
    (target: batch 8 × 3 s at 44.1 kHz, forward + backward well under 1 s on CPU).
    """
    raise NotImplementedError
