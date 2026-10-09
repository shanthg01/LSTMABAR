"""Differentiable feed-forward compressor.

Parameters are ``(B,)`` tensors in physical units; audio is ``(B, T)``.

Design (Giannoulis, Massberg & Reiss 2012, "smooth decoupled" topology, gain-side smoothing):

1. **Control-rate peak detection.** ``|x|`` is max-pooled over non-overlapping frames of
   ``CONTROL_HOP`` samples (zero-padded at the end) and converted to dB with a -120 dBFS floor.
2. **Gain computer** in dB with a quadratic soft knee of width ``knee_db``; the target gain
   reduction ``g_c = y_G - x_G <= 0``.
3. **Attack/release smoothing** of ``g_c`` with a branching one-pole filter run at the control
   rate (``fs_ctrl = sample_rate / CONTROL_HOP``): attack coefficient while the reduction is
   increasing (``g_c < y[n-1]``), release coefficient otherwise, with
   ``alpha = exp(-1 / (tau * fs_ctrl))``. The recursion runs over ~T/CONTROL_HOP frames and is
   vectorized over the batch; forward and backward are hand-written (custom autograd function)
   so the Python loop does not build a per-step autograd graph.
4. The smoothed gain (dB) is linearly interpolated back to the sample rate with each frame's
   value placed at the frame centre, converted to linear gain and multiplied with ``x``
   (plus optional makeup gain).

Approximations relative to a per-sample compressor:

- Detection is the per-frame peak, so attack/release timing is quantized to
  ``CONTROL_HOP / sample_rate`` (~0.73 ms at 44.1 kHz); time constants shorter than a frame act
  as "instant". Frame-centre placement plus linear interpolation means a gain change can start up
  to half a frame before the transient that caused it (a tiny implicit look-ahead).
- The attack/release branch is a hard switch; its *choice* carries no gradient, but each branch
  coefficient depends on its time constant, so attack/release gradients are non-zero.
- ``ratio`` is clamped to ``>= 1``, which zeroes its gradient below 1.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

CONTROL_HOP = 32
"""Samples per control frame (gain computer and smoothing run at ``sample_rate / CONTROL_HOP``)."""

_LEVEL_FLOOR = 1e-6  # -120 dBFS
_MIN_TIME_MS = 1e-3


def gain_computer(level_db: Tensor, threshold_db: Tensor, ratio: Tensor, knee_db: float) -> Tensor:
    """Static soft-knee curve: output level (dB) for input level (dB).

    ``level_db`` is ``(B, N)``; ``threshold_db``/``ratio`` are ``(B,)``. ``knee_db = 0`` gives a
    hard knee.
    """
    t = threshold_db.unsqueeze(-1)
    slope = 1.0 / ratio.clamp(min=1.0).unsqueeze(-1) - 1.0  # (1/R - 1) <= 0
    over = level_db - t
    w = max(float(knee_db), 1e-6)
    above = level_db + slope * over
    knee = level_db + slope * (over + w / 2) ** 2 / (2 * w)
    out = torch.where(2 * over < -w, level_db, knee)
    return torch.where(2 * over > w, above, out)


class _BranchingOnePole(torch.autograd.Function):
    """y[n] = a[n] y[n-1] + (1 - a[n]) g[n], a[n] = attack if g[n] < y[n-1] else release.

    y[-1] = 0. Inputs: g (B, N), a_att (B,), a_rel (B,). Backward treats the branch choice
    as constant (it is piecewise constant in the inputs).
    """

    @staticmethod
    def forward(ctx, g: Tensor, a_att: Tensor, a_rel: Tensor) -> Tensor:
        # The sequential loop runs in float64 NumPy over (B,) rows: per-op overhead is far
        # lower than for small torch tensors. Control-rate signals are small (B x ~T/32).
        g_np = g.detach().to("cpu", torch.float64).numpy().T.copy()  # (N, B)
        aa = a_att.detach().to("cpu", torch.float64).numpy()
        ar = a_rel.detach().to("cpu", torch.float64).numpy()
        y_np = np.empty_like(g_np)
        att_np = np.empty(g_np.shape, dtype=bool)
        prev = np.zeros(g_np.shape[1])
        for i, gi in enumerate(g_np):
            m = gi < prev
            prev = gi + np.where(m, aa, ar) * (prev - gi)
            y_np[i] = prev
            att_np[i] = m
        y = torch.from_numpy(y_np.T.copy()).to(device=g.device, dtype=g.dtype)
        use_att = torch.from_numpy(att_np.T.copy()).to(g.device)
        ctx.save_for_backward(g, y, use_att, a_att, a_rel)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        g, y, use_att, a_att, a_rel = ctx.saved_tensors
        b, n = g.shape
        a = torch.where(use_att, a_att.unsqueeze(-1), a_rel.unsqueeze(-1))
        y_prev = torch.cat([g.new_zeros(b, 1), y[:, :-1]], dim=1)
        # Adjoint recursion: lam[i] = grad_y[i] + a[i+1] * lam[i+1].
        gy_np = grad_y.detach().to("cpu", torch.float64).numpy().T.copy()  # (N, B)
        a_np = a.detach().to("cpu", torch.float64).numpy().T.copy()
        lam_np = np.empty_like(gy_np)
        carry = np.zeros(b)
        for i in range(n - 1, -1, -1):
            carry = gy_np[i] + carry
            lam_np[i] = carry
            carry = a_np[i] * carry
        lam = torch.from_numpy(lam_np.T.copy()).to(device=g.device, dtype=g.dtype)
        grad_g = lam * (1.0 - a)
        grad_a = lam * (y_prev - g)  # dy[i]/da[i]
        grad_att = (grad_a * use_att).sum(-1)
        grad_rel = (grad_a * ~use_att).sum(-1)
        return grad_g, grad_att, grad_rel


def _coeff(time_ms: Tensor, control_rate: float) -> Tensor:
    tau_s = time_ms.clamp(min=_MIN_TIME_MS) / 1000.0
    return torch.exp(-1.0 / (tau_s * control_rate))


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

    Implementation: peak detection, gain computer and smoothing run at the control rate
    ``sample_rate / CONTROL_HOP``; see the module docstring for the design and approximations.
    ``ratio`` is clamped to ``>= 1`` and the time constants to ``>= 1 µs``.
    """
    if x.dim() != 2:
        raise ValueError(f"expected audio of shape (B, T), got {tuple(x.shape)}")
    b, t = x.shape
    hop = CONTROL_HOP
    n_frames = -(-t // hop)
    pad = n_frames * hop - t

    def as_batch(v: Tensor) -> Tensor:
        v = torch.as_tensor(v, device=x.device, dtype=x.dtype)
        return v.reshape(1).expand(b) if v.numel() == 1 else v

    threshold_db, ratio = as_batch(threshold_db), as_batch(ratio)
    attack_ms, release_ms = as_batch(attack_ms), as_batch(release_ms)

    peak = F.max_pool1d(F.pad(x.abs(), (0, pad)).unsqueeze(1), hop, hop).squeeze(1)  # (B, N)
    level_db = 20.0 * torch.log10(peak.clamp(min=_LEVEL_FLOOR))
    gc = gain_computer(level_db, threshold_db, ratio, knee_db) - level_db  # (B, N), <= 0

    control_rate = sample_rate / hop
    smoothed = _BranchingOnePole.apply(
        gc, _coeff(attack_ms, control_rate), _coeff(release_ms, control_rate)
    )

    gain_db = F.interpolate(
        smoothed.unsqueeze(1), size=n_frames * hop, mode="linear", align_corners=False
    ).squeeze(1)[:, :t]
    if makeup_db is not None:
        gain_db = gain_db + as_batch(makeup_db).unsqueeze(-1)
    return x * torch.pow(10.0, gain_db / 20.0)
