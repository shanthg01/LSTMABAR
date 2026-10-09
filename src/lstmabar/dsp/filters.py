"""Differentiable biquad filters (RBJ Audio EQ Cookbook), applied by frequency sampling.

All parameters are physical units with shape ``(B,)``; audio is ``(B, T)``.
Coefficients are normalized so ``a0 == 1``.

Filtering multiplies the input spectrum by the biquads' frequency responses sampled on a
zero-padded rfft grid. This is an FIR approximation of the IIR filter (its impulse response
truncated to the FFT length): exact up to float error for stable filters whose impulse
response decays well within ``T`` samples. Very low-frequency, very high-Q filters applied
to very short clips have long ringing tails and can wrap around circularly.
"""

import math
from typing import Literal

import torch
from torch import Tensor

FilterKind = Literal["lowpass", "highpass", "peak", "lowshelf", "highshelf"]

_KINDS = ("lowpass", "highpass", "peak", "lowshelf", "highshelf")
_MIN_FREQ_HZ = 1.0
_MAX_FREQ_FRAC = 0.499  # of the sample rate, i.e. just below Nyquist
_MIN_Q = 1e-3


def _as_batch(v: Tensor, batch: int, name: str) -> Tensor:
    if v.numel() == 1:
        return v.reshape(1).expand(batch)
    if v.shape != (batch,):
        raise ValueError(f"{name}: expected shape ({batch},), got {tuple(v.shape)}")
    return v


def biquad_coeffs(
    kind: FilterKind,
    freq_hz: Tensor,
    q: Tensor,
    gain_db: Tensor,
    sample_rate: int,
) -> tuple[Tensor, Tensor]:
    """RBJ cookbook biquad. Returns ``(b, a)``, each ``(B, 3)`` with ``a[:, 0] == 1``.

    ``gain_db`` is ignored for lowpass/highpass. Shelves use ``q`` as the cookbook Q
    (shelf slope via Q). ``freq_hz`` is clamped to (0, Nyquist).

    Concretely ``freq_hz`` is clamped to ``[1 Hz, 0.499 * sample_rate]`` and ``q`` to
    ``>= 1e-3``; clamping zeroes the gradient outside those ranges. ``freq_hz`` sets the
    batch size; ``q``/``gain_db`` may be ``(B,)`` or single-element (broadcast). Output
    dtype/device follow ``freq_hz``. Built purely from torch ops, so it is differentiable
    w.r.t. all three parameters.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown filter kind {kind!r}; expected one of {_KINDS}")
    if freq_hz.dim() != 1:
        raise ValueError(f"freq_hz: expected shape (B,), got {tuple(freq_hz.shape)}")
    batch = freq_hz.shape[0]
    q = _as_batch(q.to(freq_hz), batch, "q")
    gain_db = _as_batch(gain_db.to(freq_hz), batch, "gain_db")

    f = freq_hz.clamp(_MIN_FREQ_HZ, _MAX_FREQ_FRAC * sample_rate)
    q = q.clamp_min(_MIN_Q)
    w0 = (2.0 * math.pi / sample_rate) * f
    cos_w = torch.cos(w0)
    alpha = torch.sin(w0) / (2.0 * q)

    if kind == "lowpass":
        k = (1.0 - cos_w) / 2.0
        b = (k, 2.0 * k, k)
        a = (1.0 + alpha, -2.0 * cos_w, 1.0 - alpha)
    elif kind == "highpass":
        k = (1.0 + cos_w) / 2.0
        b = (k, -2.0 * k, k)
        a = (1.0 + alpha, -2.0 * cos_w, 1.0 - alpha)
    else:
        big_a = torch.pow(10.0, gain_db / 40.0)
        if kind == "peak":
            b = (1.0 + alpha * big_a, -2.0 * cos_w, 1.0 - alpha * big_a)
            a = (1.0 + alpha / big_a, -2.0 * cos_w, 1.0 - alpha / big_a)
        else:
            ap1, am1 = big_a + 1.0, big_a - 1.0
            s = 2.0 * torch.sqrt(big_a) * alpha
            if kind == "lowshelf":
                b = (
                    big_a * (ap1 - am1 * cos_w + s),
                    2.0 * big_a * (am1 - ap1 * cos_w),
                    big_a * (ap1 - am1 * cos_w - s),
                )
                a = (ap1 + am1 * cos_w + s, -2.0 * (am1 + ap1 * cos_w), ap1 + am1 * cos_w - s)
            else:  # highshelf
                b = (
                    big_a * (ap1 + am1 * cos_w + s),
                    -2.0 * big_a * (am1 + ap1 * cos_w),
                    big_a * (ap1 + am1 * cos_w - s),
                )
                a = (ap1 - am1 * cos_w + s, 2.0 * (am1 - ap1 * cos_w), ap1 - am1 * cos_w - s)

    b_t = torch.stack(b, dim=-1)
    a_t = torch.stack(a, dim=-1)
    a0 = a_t[:, :1]
    return b_t / a0, a_t / a0


def freqz(b: Tensor, a: Tensor, n_fft: int) -> Tensor:
    """Complex frequency response on the rfft grid: ``(B, n_fft // 2 + 1)``.

    Computed as ``rfft(b) / rfft(a)`` of the zero-padded coefficient vectors. The division
    is guarded by replacing bins where ``|A| < eps`` (machine epsilon of the dtype) with
    ``eps``; this never triggers for a stable filter and keeps the result finite otherwise.
    """
    if b.shape != a.shape or b.dim() != 2:
        raise ValueError(f"b, a: expected matching (B, K), got {tuple(b.shape)}, {tuple(a.shape)}")
    num, den = torch.fft.rfft(torch.stack((b, a)), n=n_fft, dim=-1)
    eps = torch.finfo(b.dtype).eps
    den = torch.where(den.abs() < eps, torch.full_like(den, eps), den)
    return num / den


def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def apply_filters(x: Tensor, coeffs: list[tuple[Tensor, Tensor]]) -> Tensor:
    """Apply a cascade of biquads to ``x`` ``(B, T)`` by multiplying their responses.

    Every ``(b, a)`` is ``(B, 3)`` with the same ``B`` as ``x``. FFT size is the next power
    of two >= ``2 * T`` (zero-padded) so circular wrap of the truncated impulse response is
    negligible; output is trimmed back to ``(B, T)``. Note: ``freq_hz`` clamping in
    :func:`biquad_coeffs` zeroes its gradient at the edges.

    The result approximates causal IIR filtering (``scipy.signal.lfilter``) with zero
    initial state; it is accurate when the cascade's impulse response decays well within
    ``T`` samples. Very low-frequency, high-Q filters on very short clips can wrap. An empty
    ``coeffs`` list returns ``x`` unchanged.
    """
    if x.dim() != 2:
        raise ValueError(f"expected audio of shape (B, T), got {tuple(x.shape)}")
    if not coeffs:
        return x
    batch, n = x.shape
    for b, a in coeffs:
        if b.shape != (batch, 3) or a.shape != (batch, 3):
            raise ValueError(
                f"coeffs: expected (b, a) of shape ({batch}, 3), "
                f"got {tuple(b.shape)}, {tuple(a.shape)}"
            )
    n_fft = _next_pow2(2 * n)
    # Evaluate every biquad's response in one batched rfft call, then take the product.
    # (Multiplying the coefficient polynomials first would be cheaper but is badly
    # conditioned in float32 for low-frequency sections.)
    b_all = torch.cat([b.to(x) for b, _ in coeffs], dim=0)
    a_all = torch.cat([a.to(x) for _, a in coeffs], dim=0)
    h = freqz(b_all, a_all, n_fft).reshape(len(coeffs), batch, -1).prod(dim=0)
    y = torch.fft.irfft(torch.fft.rfft(x, n=n_fft, dim=-1) * h, n=n_fft, dim=-1)
    return y[:, :n]


def biquad(
    x: Tensor,
    kind: FilterKind,
    freq_hz: Tensor,
    sample_rate: int,
    q: Tensor | None = None,
    gain_db: Tensor | None = None,
) -> Tensor:
    """Convenience: single biquad. Defaults: ``q = 1/sqrt(2)``, ``gain_db = 0``."""
    freq_hz = freq_hz.to(x)
    if q is None:
        q = torch.full_like(freq_hz, 1.0 / math.sqrt(2.0))
    if gain_db is None:
        gain_db = torch.zeros_like(freq_hz)
    return apply_filters(x, [biquad_coeffs(kind, freq_hz, q, gain_db, sample_rate)])


def tilt(x: Tensor, tilt_db: Tensor, sample_rate: int, pivot_hz: float = 1000.0) -> Tensor:
    """Tone tilt: low shelf of ``-tilt_db/2`` and high shelf of ``+tilt_db/2`` at ``pivot_hz``.

    Both shelves use ``q = 1/sqrt(2)``. Positive ``tilt_db`` brightens, negative darkens;
    0 dB is (near) identity.
    """
    tilt_db = _as_batch(tilt_db.to(x), x.shape[0], "tilt_db")
    freq = torch.full_like(tilt_db, pivot_hz)
    q = torch.full_like(tilt_db, 1.0 / math.sqrt(2.0))
    low = biquad_coeffs("lowshelf", freq, q, -tilt_db / 2.0, sample_rate)
    high = biquad_coeffs("highshelf", freq, q, tilt_db / 2.0, sample_rate)
    return apply_filters(x, [low, high])
