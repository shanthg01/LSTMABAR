"""Audio descriptors (NumPy/SciPy). Definitions and units are in each docstring.

Harmonic descriptors take a harmonic-amplitude vector ``amps`` (``(..., K)``, peak amplitudes
of H1..HK as returned by :func:`lstmabar.analysis.harmonics.harmonic_amplitudes` or a
:class:`~lstmabar.analysis.harmonics.HarmonicProfile`); ``NaN`` entries (above Nyquist) are
treated as absent. Signal descriptors take audio ``x`` (``(..., T)``) and a sample rate.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from lstmabar.analysis.harmonics import _as_numpy, harmonic_profile, hnr

_EPS = 1e-12


# --------------------------------------------------------------------------- harmonic


def odd_even_ratio(amps, eps: float = _EPS) -> np.ndarray:
    """Odd/even harmonic energy ratio in dB.

    ``10 log10(sum_{k=3,5,..} A_k^2 / sum_{k=2,4,..} A_k^2)``.

    The fundamental is excluded (it is "odd" but present in every tone), so this measures the
    character of the *distortion*: symmetric clipping (square-like) gives a large positive
    value, asymmetric clipping / saw-like spectra give values near or below 0 dB (an ideal
    saw over H2..H10 gives about -3.0 dB). ``eps`` (energy, relative to H1² = 1 after
    normalization) bounds the result for spectra with no odd or no even content: with the
    default the range is about +-120 dB.
    """
    a = _as_numpy(amps)
    a = np.nan_to_num(a / a[..., :1], nan=0.0)
    e = a**2
    odd = e[..., 2::2].sum(-1)
    even = e[..., 1::2].sum(-1)
    return 10 * np.log10((odd + eps) / (even + eps))


def harmonic_slope(amps, floor_db: float = -80.0, odd_only: bool = False) -> np.ndarray:
    """Spectral slope of the harmonic series in dB/octave.

    Least-squares line through ``(log2 k, 20 log10(A_k / A_1))`` over the harmonics that are
    above ``floor_db`` dBc (absent harmonics, e.g. the even ones of a square wave, are skipped
    rather than counted as -inf). An ideal saw or square wave gives -6.02 dB/oct (``1/k``), a
    triangle -12.04 dB/oct (``1/k²``). ``odd_only`` restricts the fit to odd harmonics.
    Returns ``NaN`` if fewer than two harmonics pass the floor (e.g. a pure sine).
    """
    a = _as_numpy(amps)
    flat = a.reshape(-1, a.shape[-1])
    out = np.full(flat.shape[0], np.nan)
    k = np.arange(1, a.shape[-1] + 1)
    for i, row in enumerate(flat):
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 20 * np.log10(row / row[0])
        keep = np.isfinite(db) & (db > floor_db)
        if odd_only:
            keep &= k % 2 == 1
        if keep.sum() >= 2:
            out[i] = np.polyfit(np.log2(k[keep]), db[keep], 1)[0]
    return out.reshape(a.shape[:-1])


def harmonic_energy_fraction(amps) -> np.ndarray:
    """Fraction of harmonic energy outside the fundamental: ``sum_{k>=2} A_k^2 / sum_k A_k^2``.

    0 for a sine, ~0.16 for a square, ~0.35 for a saw (over 10 harmonics). Unitless.
    """
    a = np.nan_to_num(_as_numpy(amps), nan=0.0)
    e = a**2
    return e[..., 1:].sum(-1) / (e.sum(-1) + _EPS)


# --------------------------------------------------------------------------- signal


def rms(x) -> np.ndarray:
    """Root-mean-square amplitude over the last axis (linear, units of ``x``)."""
    x = _as_numpy(x)
    return np.sqrt(np.mean(x**2, axis=-1))


def rms_db(x, eps: float = 1e-20) -> np.ndarray:
    """RMS level in dBFS: ``10 log10(mean(x²))`` (a full-scale sine is -3.01 dBFS)."""
    x = _as_numpy(x)
    return 10 * np.log10(np.mean(x**2, axis=-1) + eps)


def crest_factor(x, eps: float = _EPS) -> np.ndarray:
    """Peak / RMS (linear, unitless): sqrt(2) ≈ 1.414 for a sine, 1 for a square."""
    x = _as_numpy(x)
    return np.max(np.abs(x), axis=-1) / (rms(x) + eps)


def crest_factor_db(x) -> np.ndarray:
    """Crest factor in dB, ``20 log10(peak / RMS)`` (3.01 dB for a sine)."""
    return 20 * np.log10(crest_factor(x))


def power_spectrum(x, sr: float, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density (Hann, 50% overlap) → ``(freqs_hz, psd)``, ``psd`` ``(..., F)``.

    ``nperseg`` is clipped to the signal length.
    """
    x = _as_numpy(x)
    n = min(nperseg, x.shape[-1])
    return welch(x, fs=sr, window="hann", nperseg=n, noverlap=n // 2, axis=-1)


def spectral_centroid(x, sr: float, nperseg: int = 4096) -> np.ndarray:
    """Spectral centroid in Hz: magnitude-weighted mean frequency of the Welch spectrum.

    ``sum f |X(f)| / sum |X(f)|`` with ``|X| = sqrt(PSD)`` (the librosa convention: magnitude,
    not power, weights). DC is excluded.
    """
    f, psd = power_spectrum(x, sr, nperseg)
    mag = np.sqrt(psd[..., 1:])
    return (mag * f[1:]).sum(-1) / (mag.sum(-1) + _EPS)


def spectral_rolloff(x, sr: float, fraction: float = 0.85, nperseg: int = 4096) -> np.ndarray:
    """Spectral rolloff in Hz: lowest frequency below which ``fraction`` of the *power* lies.

    Computed on the Welch PSD with DC excluded.
    """
    f, psd = power_spectrum(x, sr, nperseg)
    p = psd[..., 1:]
    c = np.cumsum(p, axis=-1)
    idx = np.argmax(c >= fraction * c[..., -1:], axis=-1)
    return f[1:][idx]


def describe(x, sr: int, f0: float | None = None, n_harmonics: int = 10) -> dict[str, float]:
    """All descriptors for a mono clip ``x`` (``(T,)``) as a flat ``dict``.

    Signal descriptors always; harmonic ones (via :func:`harmonic_profile`, which runs pYIN
    unless ``f0`` is given) when a harmonic profile can be measured. Keys: ``rms_db``,
    ``crest_factor_db``, ``centroid_hz``, ``rolloff_hz``, and optionally ``f0_hz``,
    ``hnr_db``, ``odd_even_db``, ``slope_db_per_oct``, ``harmonic_energy_fraction``.
    """
    y = _as_numpy(x)
    out = {
        "rms_db": float(rms_db(y)),
        "crest_factor_db": float(crest_factor_db(y)),
        "centroid_hz": float(spectral_centroid(y, sr)),
        "rolloff_hz": float(spectral_rolloff(y, sr)),
    }
    try:
        prof = harmonic_profile(y, sr, n_harmonics=n_harmonics, f0=f0)
    except ValueError:
        return out
    out.update(
        f0_hz=prof.f0_hz,
        hnr_db=prof.hnr_db,
        odd_even_db=float(odd_even_ratio(prof.amplitudes)),
        slope_db_per_oct=float(harmonic_slope(prof.amplitudes)),
        harmonic_energy_fraction=float(harmonic_energy_fraction(prof.amplitudes)),
    )
    return out


__all__ = [
    "crest_factor",
    "crest_factor_db",
    "describe",
    "harmonic_energy_fraction",
    "harmonic_slope",
    "hnr",
    "odd_even_ratio",
    "power_spectrum",
    "rms",
    "rms_db",
    "spectral_centroid",
    "spectral_rolloff",
]
