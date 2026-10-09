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

    - Level detection is peak-based (``|x|``) and the gain computer works in dB.
    - ``threshold_db`` is in dBFS (typically -60..0); ``ratio >= 1`` (1 = no compression).
    - ``attack_ms``/``release_ms`` set one-pole smoothing of the gain-reduction curve.
    - A sample-by-sample Python recursion will likely miss the speed target; smooth at a
      reduced control rate or with a vectorized recursion instead.

    Must be differentiable w.r.t. ``x`` and all parameters, and fast enough for training
    (target: batch 8 × 3 s at 44.1 kHz, forward + backward well under 1 s on CPU).
    """
    raise NotImplementedError
