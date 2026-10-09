"""Band-limited integer-factor resampling for running nonlinearities without aliasing.

Both directions use the same symmetric (linear-phase) Kaiser-windowed-sinc low-pass designed at
the high rate, applied centred so the group delay is exactly compensated (zero phase):

- cutoff ``0.45 * fs`` (``fs`` = the low/original rate), transition band ``0.40 -> 0.50 * fs``,
  so the stopband starts exactly at the original Nyquist;
- ``2 * HALF_WIDTH * factor + 1`` taps (``HALF_WIDTH`` original-rate samples each side);
- Kaiser ``beta`` for ~90 dB stopband attenuation (passband ripple ~1e-4 dB-scale).

Both directions are polyphase at the low rate: upsampling is one ``conv1d`` with ``factor``
output channels (then interleaved); downsampling de-interleaves into ``factor`` input channels
and runs one ``conv1d`` with ``factor`` input channels, so only kept samples are computed. Signals
are zero-extended at the edges, so the first/last ``HALF_WIDTH`` low-rate samples see edge
transients. Filters are built once per ``(factor, device, dtype)`` and cached.
"""

import functools
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

FACTORS = (1, 2, 4, 8)
HALF_WIDTH = 32
"""Filter half-length in original-rate samples."""
CUTOFF = 0.45
"""-6 dB cutoff as a fraction of the original sample rate."""
TRANSITION = 0.10
"""Transition-band width as a fraction of the original sample rate (0.40 -> 0.50)."""
ATTENUATION_DB = 90.0


def _kernel_cache(build: Callable[..., Tensor]) -> Callable[..., Tensor]:
    """Memoize a filter builder, always building outside inference mode and autograd.

    A kernel first built under ``torch.inference_mode()`` would be an inference tensor and
    break later training calls (``save_for_backward`` rejects inference tensors), so every
    cache entry is created as a normal, non-grad tensor regardless of the caller's mode.
    The caches hold one entry per ``factor`` (and per ``(factor, device, dtype)`` for the
    device kernels) and are unbounded by design: at most 4 factors x a handful of devices.
    """

    @functools.cache
    def cached(*args):
        with torch.inference_mode(False), torch.no_grad():
            return build(*args)

    cached.__doc__ = build.__doc__
    return cached


def _check_factor(factor: int) -> None:
    if factor not in FACTORS:
        raise ValueError(f"factor must be one of {FACTORS}, got {factor}")


@_kernel_cache
def _prototype(factor: int) -> Tensor:
    """Float64 CPU low-pass at the high rate, centred, DC gain 1 (sums to 1)."""
    n_taps = 2 * HALF_WIDTH * factor + 1
    d = torch.arange(n_taps, dtype=torch.float64) - HALF_WIDTH * factor
    fc = CUTOFF / factor  # cycles per high-rate sample
    beta = 0.1102 * (ATTENUATION_DB - 8.7)
    window = torch.kaiser_window(n_taps, periodic=False, beta=beta, dtype=torch.float64)
    h = 2 * fc * torch.sinc(2 * fc * d) * window
    return h / h.sum()


@_kernel_cache
def _phase_taps(factor: int) -> Tensor:
    """``(factor, 2 * HALF_WIDTH + 1)`` polyphase split of the prototype (float64, CPU).

    ``g[r, i + HALF_WIDTH] = h[i * factor + r]`` for ``i`` in ``[-HALF_WIDTH, HALF_WIDTH]``
    (``h`` indexed from its centre; zero where out of range).
    """
    h = _prototype(factor)
    i = torch.arange(-HALF_WIDTH, HALF_WIDTH + 1)
    idx = i.unsqueeze(0) * factor + torch.arange(factor).unsqueeze(1) + HALF_WIDTH * factor
    valid = (idx >= 0) & (idx < h.numel())
    return torch.where(valid, h[idx.clamp(0, h.numel() - 1)], torch.zeros((), dtype=h.dtype))


@_kernel_cache
def _up_kernel(factor: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """``(factor, 1, taps)``: ``y[n * factor + r] = factor * sum_i g[r, i] x[n - i]``.

    ``conv1d`` is a cross-correlation, so the taps are flipped; gain ``factor`` is included.
    """
    w = factor * _phase_taps(factor).flip(-1)
    return w.to(device=device, dtype=dtype).unsqueeze(1).contiguous()


@_kernel_cache
def _down_kernel(factor: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """``(1, factor, taps)``: ``y[n] = sum_r sum_i g[r, -i] x[(n - i) * factor + r]``.

    By symmetry of ``h``, ``h[i * factor - r] = g[r, -i]``, which as a cross-correlation over
    the phase-split input is just ``g`` unflipped.
    """
    return _phase_taps(factor).to(device=device, dtype=dtype).unsqueeze(0).contiguous()


class _Fan(torch.autograd.Function):
    """One channel -> ``factor`` channels: ``y[:, r, n] = sum_j w[r, 0, j] x[:, 0, n + j - W]``.

    Zero-padded ``conv1d`` whose backward is the adjoint :class:`_Gather` with flipped taps;
    PyTorch's generic CPU ``conv1d`` input-gradient is several times slower than this.
    """

    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor) -> Tensor:
        ctx.save_for_backward(w)
        return F.conv1d(F.pad(x, (HALF_WIDTH, HALF_WIDTH)), w)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:
        (w,) = ctx.saved_tensors
        return _Gather.apply(grad, w.flip(-1).transpose(0, 1).contiguous()), None


class _Gather(torch.autograd.Function):
    """``factor`` channels -> one: ``y[:, 0, n] = sum_r sum_j w[0, r, j] x[:, r, n + j - W]``."""

    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor) -> Tensor:
        ctx.save_for_backward(w)
        return F.conv1d(F.pad(x, (HALF_WIDTH, HALF_WIDTH)), w)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:
        (w,) = ctx.saved_tensors
        return _Fan.apply(grad, w.flip(-1).transpose(0, 1).contiguous()), None


def _as_batch(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    if x.dim() < 1:
        raise ValueError("x must have a time axis")
    lead = tuple(x.shape[:-1])
    return x.reshape(-1, 1, x.shape[-1]), lead


def upsample(x: Tensor, factor: int) -> Tensor:
    """``(B, T)`` → ``(B, T * factor)`` with an anti-imaging low-pass.

    ``factor`` in {1, 2, 4, 8}. Unity passband gain (zero-stuffing is compensated by
    ``x factor``) and zero phase on its own: sample ``n`` of ``x`` maps to sample
    ``n * factor`` of the output.

    Any leading shape ``(..., T)`` is accepted; ``factor = 1`` returns ``x`` unchanged.
    """
    _check_factor(factor)
    if factor == 1:
        return x
    xb, lead = _as_batch(x)
    t = xb.shape[-1]
    w = _up_kernel(factor, x.device, x.dtype)
    y = _Fan.apply(xb, w)  # (N, factor, T)
    return y.transpose(1, 2).reshape(*lead, t * factor)


def downsample(x: Tensor, factor: int) -> Tensor:
    """``(B, T * factor)`` → ``(B, T)`` with an anti-aliasing low-pass before decimation.

    Requires ``x.shape[-1] % factor == 0``. Unity passband gain and zero phase on its own.

    Any leading shape ``(..., T * factor)`` is accepted; ``factor = 1`` returns ``x``.
    """
    _check_factor(factor)
    if x.shape[-1] % factor != 0:
        raise ValueError(f"length {x.shape[-1]} is not divisible by factor {factor}")
    if factor == 1:
        return x
    xb, lead = _as_batch(x)
    t = xb.shape[-1] // factor
    phases = xb.reshape(-1, t, factor).transpose(1, 2)  # (N, factor, T): x[k * factor + r]
    w = _down_kernel(factor, x.device, x.dtype)
    y = _Gather.apply(phases.contiguous(), w)
    return y.reshape(*lead, t)


def oversampled(fn: Callable[[Tensor], Tensor], x: Tensor, factor: int) -> Tensor:
    """Run ``fn`` at ``factor``× the sample rate: ``downsample(fn(upsample(x)))``.

    Output has the same shape as ``x`` and is time-aligned with it (zero group delay or
    delay compensated).
    """
    _check_factor(factor)
    if factor == 1:
        return fn(x)
    return downsample(fn(upsample(x, factor)), factor)


__all__ = ["FACTORS", "downsample", "oversampled", "upsample"]
