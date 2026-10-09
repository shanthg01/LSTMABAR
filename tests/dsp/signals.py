"""Deterministic test signals and descriptors for the DSP tests.

The Karplus-Strong riff lives in :mod:`lstmabar.dsp.signals` (the recovery CLI needs it too);
this module adds batched wrappers and the spectral descriptors the tests assert on.
"""

import torch
from torch import Tensor

from lstmabar.dsp.signals import pluck_riff, sine

SR = 44100


def riff(batch: int = 1, seconds: float = 1.0, peak: float = 0.5) -> Tensor:
    """``(batch, T)`` riff, a different pluck-noise seed per row."""
    return torch.stack([pluck_riff(seconds, SR, peak=peak, seed=i) for i in range(batch)])


def tone(freq_hz: float, seconds: float = 0.5, amplitude: float = 0.5, batch: int = 1) -> Tensor:
    """``(batch, T)`` sine."""
    return sine(freq_hz, seconds, SR, amplitude).expand(batch, -1).clone()


def burst(seconds: float = 1.0, batch: int = 1) -> Tensor:
    """Loud/quiet alternating 1 kHz sine (100 ms segments at -6 / -30 dBFS)."""
    x = sine(1000.0, seconds, SR, 1.0)
    seg = int(0.1 * SR)
    env = torch.ones_like(x) * 10 ** (-30 / 20)
    for start in range(0, len(x), 2 * seg):
        env[start : start + seg] = 10 ** (-6 / 20)
    return (x * env).expand(batch, -1).clone()


def spectrum(x: Tensor) -> Tensor:
    """Hann-windowed magnitude spectrum, ``(B, T//2+1)``."""
    w = torch.hann_window(x.shape[-1], periodic=False, dtype=x.dtype)
    return torch.fft.rfft(x * w).abs()


def harmonic(x: Tensor, f0: float, k: int, width: int = 3) -> Tensor:
    """Magnitude (peak within ``±width`` bins) of the ``k``-th harmonic of ``f0``, ``(B,)``."""
    mag = spectrum(x)
    centre = round(k * f0 * x.shape[-1] / SR)
    return mag[:, centre - width : centre + width + 1].amax(-1)


def band_energy(x: Tensor, lo: float, hi: float) -> Tensor:
    """Spectral energy between ``lo`` and ``hi`` Hz, ``(B,)``."""
    mag = spectrum(x)
    freqs = torch.fft.rfftfreq(x.shape[-1], 1 / SR)
    return mag[:, (freqs >= lo) & (freqs < hi)].pow(2).sum(-1)


def centroid(x: Tensor) -> Tensor:
    """Spectral centroid (Hz), ``(B,)``."""
    mag = spectrum(x)
    freqs = torch.fft.rfftfreq(x.shape[-1], 1 / SR).to(x.dtype)
    return (mag * freqs).sum(-1) / mag.sum(-1)


def crest_factor(x: Tensor) -> Tensor:
    """Peak / RMS, ``(B,)``."""
    return x.abs().amax(-1) / x.pow(2).mean(-1).sqrt()


def rms_db(x: Tensor) -> Tensor:
    return 10 * torch.log10(x.pow(2).mean(-1))


def harmonic_ratio(x: Tensor, f0: float, n: int = 10) -> Tensor:
    """Energy in harmonics 2..n relative to the fundamental, ``(B,)``."""
    h1 = harmonic(x, f0, 1).pow(2)
    rest = sum(harmonic(x, f0, k).pow(2) for k in range(2, n + 1))
    return rest / h1
