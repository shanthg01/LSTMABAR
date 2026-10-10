"""Linear analog stages for the white-box sims: s-domain transfer functions from component
values, discretized with the bilinear transform and applied with ``scipy.signal.lfilter``.

Coefficients are highest power of ``s`` first (``scipy.signal`` convention). First-order RC
sections are prewarped at their corner frequency, so the -3 dB point is exact at any sample
rate; elsewhere the bilinear frequency warping is ``f_d = (fs/π)·atan(π f / fs)``.
Filters start from rest (zero initial state).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import bilinear, lfilter


@dataclass(frozen=True)
class AnalogFilter:
    """``H(s) = B(s) / A(s)``; ``prewarp_hz`` is matched exactly when discretized."""

    b: tuple[float, ...]
    a: tuple[float, ...]
    prewarp_hz: float | None = None

    def response(self, freqs_hz: np.ndarray | float) -> np.ndarray:
        """Complex analog response at ``freqs_hz``."""
        s = 2j * math.pi * np.asarray(freqs_hz, dtype=np.float64)
        return np.polyval(self.b, s) / np.polyval(self.a, s)

    def digital(self, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
        """Bilinear-transform ``(b, a)`` z-domain coefficients at ``sample_rate``."""
        return to_digital(self.b, self.a, sample_rate, self.prewarp_hz)

    def apply(self, x: np.ndarray, sample_rate: float) -> np.ndarray:
        """Filter ``x`` along its last axis at ``sample_rate``."""
        bz, az = self.digital(sample_rate)
        return lfilter(bz, az, x, axis=-1)


def to_digital(
    b: Sequence[float], a: Sequence[float], sample_rate: float, prewarp_hz: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear transform ``s = 2K (z-1)/(z+1)``; ``K = fs`` without prewarping, otherwise
    ``K = π f0 / tan(π f0 / fs)`` so the analog and digital responses agree at ``f0``."""
    fs = float(sample_rate)
    if prewarp_hz is not None:
        if not 0 < prewarp_hz < fs / 2:
            raise ValueError(f"prewarp frequency {prewarp_hz} outside (0, fs/2)")
        w = math.pi * prewarp_hz / fs
        fs = math.pi * prewarp_hz / math.tan(w)
    bz, az = bilinear(np.asarray(b, np.float64), np.asarray(a, np.float64), fs=fs)
    return np.atleast_1d(bz), np.atleast_1d(az)


def rc_lowpass(r: float, c: float) -> AnalogFilter:
    """Series R, shunt C: ``1 / (1 + sRC)``, prewarped at the corner."""
    tau = r * c
    return AnalogFilter((1.0,), (tau, 1.0), 1.0 / (2 * math.pi * tau))


def rc_highpass(r: float, c: float) -> AnalogFilter:
    """Series C, shunt R: ``sRC / (1 + sRC)``, prewarped at the corner."""
    tau = r * c
    return AnalogFilter((tau, 0.0), (tau, 1.0), 1.0 / (2 * math.pi * tau))


def apply_chain(stages: Sequence[AnalogFilter], x: np.ndarray, sample_rate: float) -> np.ndarray:
    for st in stages:
        x = st.apply(x, sample_rate)
    return x


def chain_response(stages: Sequence[AnalogFilter], freqs_hz: np.ndarray | float) -> np.ndarray:
    h = np.ones_like(np.asarray(freqs_hz, dtype=np.float64), dtype=np.complex128)
    for st in stages:
        h = h * st.response(freqs_hz)
    return h


__all__ = [
    "AnalogFilter",
    "apply_chain",
    "chain_response",
    "rc_highpass",
    "rc_lowpass",
    "to_digital",
]
