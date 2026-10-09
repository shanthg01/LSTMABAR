"""Differentiable static nonlinearities spanning soft, hard and asymmetric clipping.

The shaper is memoryless: ``y[n] = f(x[n]; softness, asymmetry, bias)``. Parameters are
``(B,)`` tensors in physical units; audio is ``(B, T)``.
"""

from torch import Tensor


def waveshape(x: Tensor, softness: Tensor, asymmetry: Tensor, bias: Tensor) -> Tensor:
    """Parametric clipper with output in roughly [-1, 1].

    - ``softness`` in [0, 1]: 0 = smooth tanh-like soft clipping, 1 = near-hard clipping
      (must stay differentiable with non-vanishing gradients near the knee).
    - ``asymmetry`` in [0, 1]: 0 = odd-symmetric (odd harmonics only); larger values clip the
      negative half earlier/harder, adding even harmonics (germanium / biased-transistor
      character).
    - ``bias`` in [-0.5, 0.5]: DC offset added before shaping; ``f(bias)`` is subtracted
      after shaping so silence maps to silence.

    Small-signal gain is ~1 at ``bias = 0`` (input gain belongs to the caller).
    """
    raise NotImplementedError
