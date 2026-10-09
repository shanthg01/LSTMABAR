"""Differentiable static nonlinearities spanning soft, hard and asymmetric clipping.

The shaper is memoryless: ``y[n] = f(x[n]; softness, asymmetry, bias)``. Parameters are
``(B,)`` tensors in physical units; audio is ``(B, T)``.

Core curve is the p-norm saturator ``s_p(x) = x / (1 + |x|^p)^(1/p)``: slope 1 at 0, bounded
by ±1, strictly increasing, and at least C^2 (near 0 it is ``x - sign(x)|x|^(p+1)/p + ...``).
``p = 2`` is a gentle algebraic (tanh-like) knee; ``p -> inf`` approaches a hard clip at ±1.
It is evaluated in the log domain (softplus of ``p log|x|``), so large inputs and large ``p``
never overflow and gradients stay finite at ``x = 0`` and in the saturated region.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

P_MIN = 2.0
"""Knee exponent at ``softness = 0`` (soft, tanh-like)."""
P_MAX = 12.0
"""Knee exponent at ``softness = 1`` (near-hard)."""

_TINY = 1e-30  # floor for |x| inside the log; below it the curve is exactly linear


def _knee_exponent(softness: Tensor) -> Tensor:
    """Geometric interpolation P_MIN -> P_MAX (perceptually more even than linear)."""
    return P_MIN * (P_MAX / P_MIN) ** softness


def _saturate(x: Tensor, p: Tensor, ceiling_neg: Tensor) -> Tensor:
    """``c * s_p(x / c)`` with ceiling ``c = 1`` for ``x >= 0`` and ``ceiling_neg`` for ``x < 0``.

    Scaling by the ceiling keeps slope 1 at 0 and C^p continuity across 0.
    """
    c = torch.where(x < 0, ceiling_neg, torch.ones_like(ceiling_neg))
    u = x / c
    a = u.abs()
    pl = p * torch.log(a.clamp_min(_TINY))
    # Two algebraically identical forms; each keeps autograd accurate on its side of |u| = 1:
    # inside, u * (1 + a^p)^(-1/p) (exact unit slope at 0); outside, sign(u) * (1 + a^-p)^(-1/p)
    # (no 1 - sigmoid cancellation, so the tiny saturated-region slope doesn't round to 0).
    inside = a < 1
    lead = torch.where(inside, u, torch.sign(u))
    q = torch.where(inside, pl, -pl)
    return lead * torch.exp(-F.softplus(q) / p) * c


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

    Implementation: ``g(x) = c * s_p(x / c)`` (``s_p`` from the module docstring) with
    ``p = 2 * 6**softness`` (2 -> 12) and negative-half ceiling ``c = 1 - 0.5 * asymmetry``;
    the output is ``(g(x + bias) - g(bias)) / (1 + |g(bias)|)``. The divisor is exactly 1 at
    ``bias = 0`` and keeps the biased curve inside [-1, 1] (``g`` lies in ``(-c, 1)``).
    Monotonic in ``x`` for all parameters. ``softness``/``asymmetry``/``bias`` are ``(B,)``
    (or broadcastable) and are applied along the last (time) axis.

    Headroom with bias: shifting the operating point spends headroom on one side. At
    ``bias = +0.5`` the positive output ceiling ``(1 - g(b)) / (1 + g(b))`` drops to ~0.34
    (0.38 at softness 0, 0.33 at softness 1). At ``asymmetry = 1, bias = -0.5`` the negative
    ceiling is ~-0.02 at softness 1 (-0.11 at softness 0), i.e. near half-wave rectification,
    with tiny gradients through the clipped half. Callers should keep ``|bias| <= ~0.25`` for
    musically useful settings (the Drive block's range does).
    """
    if x.dim() < 1:
        raise ValueError("x must have a time axis")
    p = _knee_exponent(softness).unsqueeze(-1)
    c_neg = (1.0 - 0.5 * asymmetry).unsqueeze(-1)
    b = bias.unsqueeze(-1)
    g_b = _saturate(b, p, c_neg)
    return (_saturate(x + b, p, c_neg) - g_b) / (1.0 + g_b.abs())


__all__ = ["P_MAX", "P_MIN", "waveshape"]
