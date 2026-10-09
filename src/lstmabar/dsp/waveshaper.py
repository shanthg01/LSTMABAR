"""Differentiable static nonlinearities spanning soft, hard and asymmetric clipping.

The shaper is memoryless: ``y[n] = f(x[n]; softness, asymmetry, bias)``. Parameters are
``(B,)`` tensors in physical units; audio is ``(B, T)``.
"""

from torch import Tensor


def waveshape(x: Tensor, softness: Tensor, asymmetry: Tensor, bias: Tensor) -> Tensor:
    """Parametric clipper with output bounded to [-1, 1].

    - ``softness`` in [0, 1]: 0 = smooth tanh-like soft clipping, 1 = near-hard clipping
      (must stay differentiable with non-vanishing gradients near the knee).
    - ``asymmetry`` in [0, 1]: 0 = odd-symmetric (odd harmonics only); larger values clip the
      negative half earlier, adding even harmonics (germanium / biased-transistor character).
      At ``asymmetry = 1`` the negative half saturates at -0.5 while the positive half still
      saturates at +1.
    - ``bias`` in [-0.5, 0.5]: DC offset added before shaping; ``f(bias)`` is subtracted
      after shaping so silence maps to silence.

    Small-signal gain (slope at 0) is ~1 at ``bias = 0`` for every ``softness`` and
    ``asymmetry``; input gain belongs to the caller.
    """
    raise NotImplementedError
