"""Archetype readout: project a harmonic profile onto {sine, triangle, square, saw} + noise.

The archetypes are a *measurement* (design §3, §4.6): their harmonic fingerprints are

==========  ===================  ==========
archetype   harmonics present    amplitude
==========  ===================  ==========
sine        H1 only              1
triangle    odd                  1/k²
square      odd                  1/k
saw         all                  1/k
==========  ===================  ==========

Normalization choices (documented because they change the numbers):

- **Amplitude domain.** Profiles and references are vectors of harmonic *peak amplitudes*
  ``A_1..A_K`` (phases ignored). Amplitude rather than power keeps the weaker upper harmonics
  visible to the fit: in power, a square's H3 is only 11% of H1 and the readout would call
  most distortion "sine".
- **Energy (L2) normalization.** Each reference and the measured profile are scaled to unit
  L2 norm over the harmonics that are measurable (below Nyquist; ``NaN`` entries are dropped
  from the profile *and* the references). So the readout depends on spectral *shape* only,
  not on level.
- **NNLS** (``scipy.optimize.nnls``) on the unit-norm vectors gives non-negative
  coefficients; the four harmonic weights are those coefficients divided by their sum, so they
  sum to 1. ``residual`` is the L2 norm of the fit residual (0 = exactly a mix of archetypes,
  1 = orthogonal to all of them).
- **Noise fraction** from the HNR: the noise share of total power,
  ``noise = 1 / (1 + 10^(HNR/10))`` (0 for a clean tone, 0.5 at HNR = 0 dB). The five-way
  :meth:`ArchetypeReadout.vector` scales the harmonic weights by ``1 - noise`` and appends
  ``noise``, so it also sums to 1.

Limitations: NumPy, not differentiable. When only H1 (or H1-H2) is measurable (very high f0
or low sample rate), sine, triangle and square coincide after masking and the readout says
"sine"; not reachable for guitar at 44.1 kHz with 10 harmonics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from lstmabar.analysis.harmonics import (
    _as_numpy,
    harmonic_amplitudes,
    harmonic_profile,
    hnr,
    refine_f0,
)

ARCHETYPES: tuple[str, ...] = ("sine", "triangle", "square", "saw")


def reference_spectra(n_harmonics: int = 10) -> np.ndarray:
    """Archetype amplitude spectra, ``(4, n_harmonics)`` in :data:`ARCHETYPES` order.

    Not normalized (H1 = 1 in each row); :func:`project_archetypes` L2-normalizes them.
    """
    k = np.arange(1, n_harmonics + 1, dtype=np.float64)
    odd = (k % 2 == 1).astype(np.float64)
    sine = (k == 1).astype(np.float64)
    return np.stack([sine, odd / k**2, odd / k, 1.0 / k])


def noise_fraction(hnr_db: float | None) -> float:
    """Noise share of total power from an HNR in dB: ``1 / (1 + 10^(HNR/10))``.

    ``None`` / ``NaN`` (HNR unknown) give 0.
    """
    if hnr_db is None or np.isnan(hnr_db):
        return 0.0
    with np.errstate(over="ignore"):
        return float(1.0 / (1.0 + np.power(10.0, np.float64(hnr_db) / 10)))


@dataclass
class ArchetypeReadout:
    """Archetype weights (sum to 1), noise fraction, and the NNLS fit residual."""

    weights: dict[str, float]
    noise: float
    residual: float

    def vector(self) -> np.ndarray:
        """``[sine, triangle, square, saw, noise]``, summing to 1."""
        w = np.array([self.weights[a] for a in ARCHETYPES]) * (1.0 - self.noise)
        return np.append(w, self.noise)

    @property
    def dominant(self) -> str:
        return max(self.weights, key=self.weights.get)


def project_archetypes(amps, hnr_db: float | None = None) -> ArchetypeReadout:
    """NNLS projection of a harmonic-amplitude profile ``amps`` (``(K,)``) onto the archetypes.

    ``amps`` can be absolute or H1-normalized (it is L2-normalized here). ``NaN`` harmonics are
    dropped together with the matching reference entries. ``hnr_db`` (optional) sets the noise
    fraction; without it the noise fraction is 0.
    """
    a = _as_numpy(amps)
    if a.ndim != 1:
        raise ValueError(f"project_archetypes expects (K,), got {a.shape}")
    keep = np.isfinite(a)
    if not keep[0] or a[0] <= 0:
        raise ValueError("H1 must be finite and positive")
    a = np.clip(a[keep], 0.0, None)
    refs = reference_spectra(len(keep))[:, keep]
    refs = refs / np.linalg.norm(refs, axis=1, keepdims=True)
    target = a / np.linalg.norm(a)
    coef, res = nnls(refs.T, target)
    total = coef.sum()
    weights = coef / total if total > 0 else np.array([1.0, 0.0, 0.0, 0.0])
    return ArchetypeReadout(
        weights={name: float(w) for name, w in zip(ARCHETYPES, weights, strict=True)},
        noise=noise_fraction(hnr_db),
        residual=float(res),
    )


def archetype_readout(
    x,
    sr: int,
    f0: float | None = None,
    n_harmonics: int = 10,
    whole_clip: bool = False,
    refine: bool = True,
    hnr_max_hz: float | None = None,
) -> ArchetypeReadout:
    """Archetype readout of mono audio ``x`` (``(T,)``).

    - ``f0`` unknown: pYIN profile (needs the ``analysis`` extra), median over voiced frames.
    - ``f0`` known: frame-wise profile at that f0 (no librosa), or with ``whole_clip=True`` one
      fit over the whole (stationary) clip, which is what synthetic sine tests want.
    - ``refine`` (default) refines the f0 (given or tracked) with :func:`refine_f0`; a nominal
      note frequency is not exact enough for the fits. ``refine=False`` only for exact f0.
    """
    if f0 is not None and whole_clip:
        if refine:
            f0 = refine_f0(x, sr, f0)
        amps = harmonic_amplitudes(x, sr, f0, n_harmonics)
        return project_archetypes(amps, float(hnr(x, sr, f0, hnr_max_hz)))
    prof = harmonic_profile(
        x, sr, n_harmonics=n_harmonics, f0=f0, refine=refine, hnr_max_hz=hnr_max_hz
    )
    return project_archetypes(prof.amplitudes, prof.hnr_db)


__all__ = [
    "ARCHETYPES",
    "ArchetypeReadout",
    "archetype_readout",
    "noise_fraction",
    "project_archetypes",
    "reference_spectra",
]
