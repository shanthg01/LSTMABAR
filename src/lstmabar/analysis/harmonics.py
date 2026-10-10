"""Harmonic analysis: per-harmonic amplitudes at a known f0, HNR, and pYIN-based profiles.

Two estimators, both pure NumPy:

- :func:`harmonic_amplitudes` (known f0, e.g. the P2 fidelity sweep on sines): a *weighted
  least-squares* fit of ``DC + sum_k a_k cos(2 pi k f0 t) + b_k sin(2 pi k f0 t)`` for
  ``k = 1..n_harmonics``. The weights are a 4-term Blackman-Harris window, whose sidelobes
  (< -92 dB) keep unmodelled content (harmonics above ``n_harmonics``, noise, a fractional
  last period) from leaking into the fitted ones. For a stationary harmonic signal the fit is
  exact; in practice the error is far below 0.01 dB for harmonics well above the noise floor,
  for any clip of at least ~3 periods (0.25 s at 80 Hz is 20 periods).
- :func:`hnr` (harmonic-to-noise ratio): Blackman-Harris-windowed correlation at *every*
  harmonic below Nyquist (O(T * K), no large solve), subtracted from the signal; HNR is the
  windowed power of the harmonic part over the windowed power of the residual. Needs the
  frame to span >= ~4.5 periods so the window main lobes (+-4 DFT bins) don't overlap.

Amplitudes are *peak* amplitudes in the units of ``x``: a full-scale sine ``sin(2 pi f t)``
has ``H1 = 1.0`` (0 dB in :func:`to_db`). Harmonics at or above Nyquist are ``NaN`` (not
measurable); :func:`to_db` / :func:`to_dbc` map them to the floor.

For real audio, :func:`track_f0` runs pYIN (librosa, the optional ``analysis`` extra, imported
lazily) and :func:`harmonic_profile` fits each voiced frame at its tracked f0 and takes the
median over frames of the H1-normalized amplitudes.

Everything here is NumPy (float64) and **not differentiable**: torch inputs are detached and
copied to the CPU. Use these as evaluation metrics, not as training losses.

**f0 must be exact.** :func:`harmonic_amplitudes`, :func:`harmonic_fit` and :func:`hnr` never
refine the f0 they are given; on a 0.5 s clip a 1-cent error already costs ~1 dB at H10 and
5 cents ~40 dB. Pass the exact frequency of a synthetic test tone, or call :func:`refine_f0`
first (:func:`harmonic_profile` and :func:`~lstmabar.analysis.archetypes.archetype_readout`
do so by default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 4-term Blackman-Harris (minimum sidelobe, -92 dB; main lobe half-width 4 bins).
_BH4 = (0.35875, 0.48829, 0.14128, 0.01168)


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch.Tensor without importing torch
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def blackman_harris(n: int) -> np.ndarray:
    """Symmetric 4-term Blackman-Harris window of length ``n``."""
    if n == 1:
        return np.ones(1)
    t = 2 * np.pi * np.arange(n) / (n - 1)
    a0, a1, a2, a3 = _BH4
    return a0 - a1 * np.cos(t) + a2 * np.cos(2 * t) - a3 * np.cos(3 * t)


def _check_f0(f0: float, sr: float) -> None:
    if not (np.isfinite(f0) and 0 < f0 < sr / 2):
        raise ValueError(f"f0 must be in (0, sr/2), got {f0} at sr={sr}")


def _valid_orders(f0: float, sr: float, n_harmonics: int) -> np.ndarray:
    """Harmonic orders ``1..n_harmonics`` strictly below Nyquist (with a small guard band)."""
    k = np.arange(1, n_harmonics + 1)
    return k[k * f0 < 0.499 * sr]


def harmonic_fit(x, sr: float, f0: float, n_harmonics: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Weighted LS harmonic fit; returns ``(amplitudes, phases)``, each ``(..., n_harmonics)``.

    ``x`` is ``(..., T)`` (NumPy or torch). Phases are in radians for ``cos(2 pi k f0 t + phi)``
    with ``t = 0`` at the first sample. Harmonics at/above Nyquist are ``NaN``.
    """
    x = _as_numpy(x)
    _check_f0(f0, sr)
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")
    t_len = x.shape[-1]
    if t_len * f0 / sr < 2.0:
        raise ValueError(
            f"clip too short: {t_len} samples is {t_len * f0 / sr:.2f} periods of {f0} Hz; "
            "need >= 2 (>= 3 recommended)"
        )
    batch_shape = x.shape[:-1]
    xs = x.reshape(-1, t_len)
    orders = _valid_orders(f0, sr, n_harmonics)
    # Centre the time axis for conditioning; phases are shifted back to t = 0 below.
    n = np.arange(t_len)
    tc = (n - (t_len - 1) / 2) / sr
    arg = 2 * np.pi * f0 * np.outer(tc, orders)  # (T, K)
    design = np.concatenate([np.ones((t_len, 1)), np.cos(arg), np.sin(arg)], axis=1)
    sw = np.sqrt(blackman_harris(t_len))[:, None]
    coef, *_ = np.linalg.lstsq(design * sw, xs.T * sw, rcond=None)  # (1 + 2K, B)
    kk = len(orders)
    a, b = coef[1 : 1 + kk], coef[1 + kk :]
    # a cos(w tc) + b sin(w tc) = A cos(w tc + phi), A = |a - ib|, phi = angle(a - ib)
    z = a - 1j * b
    shift = 2 * np.pi * f0 * orders[:, None] * ((t_len - 1) / 2) / sr  # tc = t - t_mid
    amps = np.full((xs.shape[0], n_harmonics), np.nan)
    phases = np.full((xs.shape[0], n_harmonics), np.nan)
    amps[:, :kk] = np.abs(z).T
    phases[:, :kk] = np.angle(z * np.exp(-1j * shift)).T
    return amps.reshape(*batch_shape, n_harmonics), phases.reshape(*batch_shape, n_harmonics)


def harmonic_amplitudes(x, sr: float, f0: float, n_harmonics: int = 10) -> np.ndarray:
    """Peak amplitudes of harmonics ``1..n_harmonics`` of a *known* ``f0``, ``(..., K)``.

    ``x`` is ``(..., T)`` (NumPy or torch, any float dtype; computed in float64). Accurate to
    well under 0.1 dB for harmonics well above the noise floor, for f0 from ~50 Hz to a few kHz
    and clips of >= ~3 periods, **provided ``f0`` is exact** (well under 1 cent; this function
    never refines it, use :func:`refine_f0` for a nominal pitch). Harmonics at/above Nyquist
    are ``NaN``. Crop onsets and transients before calling: the fit assumes a stationary
    signal. Not differentiable (NumPy; torch inputs are detached).
    """
    return harmonic_fit(x, sr, f0, n_harmonics)[0]


def to_db(amps, floor_db: float | None = None) -> np.ndarray:
    """``20 log10(amps)`` (dB re. amplitude 1, i.e. dBFS peak for full-scale audio).

    With ``floor_db``, values below it (including 0 and ``NaN``) are clamped to the floor;
    without it, 0 maps to ``-inf`` and ``NaN`` stays ``NaN``. For an *absolute* level metric
    (e.g. the P2 fidelity H1 term) use it without a floor, so a silent or failed render shows
    up as ``-inf``/``NaN`` instead of reading as the floor.
    """
    a = _as_numpy(amps)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20 * np.log10(a)
    if floor_db is not None:
        db = np.where(np.isnan(db), floor_db, np.maximum(db, floor_db))
    return db


def to_dbc(amps, floor_db: float = -60.0) -> np.ndarray:
    """Harmonic levels in dBc (relative to H1, the first entry along the last axis).

    H1 is 0 dBc; others are ``20 log10(A_k / A_1)`` clamped below at ``floor_db``
    (``NaN`` / above-Nyquist harmonics map to the floor). If H1 itself is 0 or ``NaN`` the
    whole row is ``NaN``.
    """
    a = _as_numpy(amps)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = a / a[..., :1]
        db = 20 * np.log10(rel)
    db = np.where(np.isnan(db), floor_db, np.maximum(db, floor_db))
    bad_h1 = ~(np.isfinite(a[..., :1]) & (a[..., :1] > 0))
    return np.where(bad_h1, np.nan, db)


def _harmonic_basis(t_len: int, sr: float, f0: float, n: int) -> np.ndarray:
    """``exp(-2 pi i k f0 t)`` for ``k = 1..n``, ``(T, n)``, via a cumulative product
    (one ``exp`` per sample instead of per sample and harmonic; error ~``n`` ulp)."""
    step = np.exp(-2j * np.pi * f0 * np.arange(t_len) / sr)[:, None]
    return np.cumprod(np.broadcast_to(step, (t_len, n)), axis=1)


def refine_f0(x, sr: float, f0: float, cents: float = 20.0, max_hz: float = 4000.0) -> float:
    """Refine a coarse f0 estimate (e.g. pYIN's 10-cent grid) for one mono frame.

    Maximizes the Blackman-Harris-windowed harmonic energy ``sum_k |c_k|^2`` over harmonics
    below ``max_hz`` (bounded Brent search within ``+-cents``). The exact f0 matters for the
    fits: at 10 cents off, harmonic 30 of 110 Hz is ~19 Hz away from its bin.
    """
    from scipy.optimize import minimize_scalar

    y = _as_numpy(x)
    y = y - y.mean()
    w = blackman_harris(len(y))
    yw = y * w

    def neg_energy(f: float) -> float:
        n = max(1, int(min(max_hz, 0.499 * sr) // f))
        c = yw @ _harmonic_basis(len(y), sr, f, n)
        return -float(np.sum(np.abs(c) ** 2))

    r = 2 ** (cents / 1200)
    res = minimize_scalar(
        neg_energy,
        bounds=(f0 / r, min(f0 * r, 0.499 * sr)),
        method="bounded",
        options={"xatol": 1e-5 * f0},
    )
    return float(res.x)


def hnr(x, sr: float, f0: float, max_hz: float | None = None, eps: float = 1e-20) -> np.ndarray:
    """Harmonic-to-noise ratio in dB, ``(...)`` for ``x`` of shape ``(..., T)``.

    The harmonic part is the sum of all harmonics ``k f0 < min(max_hz, Nyquist)``, each
    estimated by Blackman-Harris-windowed correlation at the exact harmonic frequency (the
    window's -92 dB sidelobes bound the cross-talk, so the measurable HNR tops out around
    ~90 dB). The noise part is everything else (inharmonic partials, noise, content above
    ``max_hz``), excluding DC. Both powers are Blackman-Harris-weighted means over the frame.
    Requires >= ~4.5 periods in the frame. Returns ``NaN`` for silence (no power above DC).

    Caveats: harmonics in the guard band 0.499-0.5 sr count as noise, and with very short
    frames a harmonic within ~2 bins of Nyquist cross-talks with its mirror image; neither is
    reachable for guitar at 44.1 kHz with the default frame length. String inharmonicity
    (stretched upper partials) also counts as noise; lower ``max_hz`` to discount it.
    """
    x = _as_numpy(x)
    _check_f0(f0, sr)
    t_len = x.shape[-1]
    if t_len * f0 / sr < 4.5:
        raise ValueError(f"hnr needs >= 4.5 periods; got {t_len * f0 / sr:.2f}")
    w = blackman_harris(t_len)
    wsum = w.sum()
    xs = x.reshape(-1, t_len)
    xs = xs - (xs @ w / wsum)[:, None]  # remove the weighted mean (DC)
    top = 0.499 * sr if max_hz is None else min(max_hz, 0.499 * sr)
    orders = np.arange(1, int(math.floor(top / f0)) + 1)
    orders = orders[orders * f0 < top]
    basis = _harmonic_basis(t_len, sr, f0, len(orders))  # (T, K)
    # Real and imaginary parts as separate real matmuls (a real @ complex product misses BLAS).
    xw = xs * w
    b_re, b_im = np.ascontiguousarray(basis.real), np.ascontiguousarray(basis.imag)
    c_re, c_im = 2 * (xw @ b_re) / wsum, 2 * (xw @ b_im) / wsum  # complex amps (B, K)
    harm = c_re @ b_re.T + c_im @ b_im.T  # Re(c @ conj(basis).T), (B, T)
    resid = xs - harm
    p_h = (harm**2) @ w / wsum
    p_n = (resid**2) @ w / wsum
    p_x = (xs**2) @ w / wsum
    out = 10 * np.log10((p_h + eps) / (p_n + eps))
    out = np.where(p_x > 1e-30, out, np.nan)  # silence: HNR undefined
    return out.reshape(x.shape[:-1])


# --------------------------------------------------------------------------- f0 tracking


def _require_librosa():
    try:
        import librosa
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "f0 tracking needs librosa: install the analysis extra "
            "(`uv sync --extra analysis` or `pip install lstmabar[analysis]`)"
        ) from e
    return librosa


def default_frame_length(sr: float) -> int:
    """Power of two covering >= ~90 ms (4096 at 44.1 kHz): >= 6 periods of 70 Hz."""
    return int(2 ** math.ceil(math.log2(0.09 * sr)))


@dataclass
class F0Track:
    """pYIN output. ``f0_hz`` is ``NaN`` where unvoiced; frame ``i`` is centred at ``times[i]``."""

    times: np.ndarray
    f0_hz: np.ndarray
    voiced: np.ndarray
    voiced_prob: np.ndarray
    hop_length: int
    frame_length: int


def track_f0(
    x,
    sr: int,
    fmin: float = 70.0,
    fmax: float = 1100.0,
    frame_length: int | None = None,
    hop_length: int | None = None,
) -> F0Track:
    """Frame-wise f0 of mono audio ``x`` (``(T,)``) with pYIN (``librosa.pyin``).

    Defaults cover the guitar range (low E ~82 Hz, up to ~1.1 kHz at the 24th fret of the high
    E). ``frame_length`` defaults to :func:`default_frame_length`, ``hop_length`` to 1/4 of it.
    """
    librosa = _require_librosa()
    y = _as_numpy(x)
    if y.ndim != 1:
        raise ValueError(f"track_f0 expects mono (T,), got {y.shape}")
    frame_length = frame_length or default_frame_length(sr)
    hop_length = hop_length or frame_length // 4
    f0, voiced, prob = librosa.pyin(
        y.astype(np.float32),
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )
    times = np.arange(len(f0)) * hop_length / sr
    return F0Track(times, f0.astype(np.float64), voiced, prob, hop_length, frame_length)


@dataclass
class HarmonicProfile:
    """Clip-level harmonic profile.

    - ``amplitudes``: median over voiced frames of the H1-normalized amplitudes (H1 = 1),
      ``(n_harmonics,)``; ``NaN`` where a harmonic was above Nyquist in most frames.
    - ``h1``: median absolute H1 peak amplitude; ``f0_hz``: median f0; ``hnr_db``: median HNR.
    - ``frame_amplitudes``/``frame_f0``/``frame_hnr_db``: per analysed frame (absolute amps).
    """

    amplitudes: np.ndarray
    h1: float
    f0_hz: float
    hnr_db: float
    voiced_fraction: float
    frame_amplitudes: np.ndarray
    frame_f0: np.ndarray
    frame_hnr_db: np.ndarray

    def dbc(self, floor_db: float = -60.0) -> np.ndarray:
        return to_dbc(self.amplitudes, floor_db)


def harmonic_profile(
    x,
    sr: int,
    n_harmonics: int = 10,
    f0: float | None = None,
    track: F0Track | None = None,
    max_frames: int = 64,
    min_rms_db: float = -50.0,
    refine: bool = True,
    hnr_max_hz: float | None = None,
    **track_kwargs,
) -> HarmonicProfile:
    """Clip-level harmonic profile of mono audio ``x``.

    With ``f0`` given, the f0 is fixed for every frame and pYIN is skipped (no librosa needed);
    otherwise pYIN (:func:`track_f0`, or a precomputed ``track``) gives a frame-wise f0 and only
    voiced frames are used. Frames are ``frame_length`` long, centred on the pYIN frame times,
    and frames quieter than ``min_rms_db`` (dBFS) are skipped. Voiced runs are eroded by half a
    frame at each end (when that leaves any frame), so frames straddling a note boundary or
    silence don't count. With ``refine`` (default) each frame's f0, tracked *or* given, is
    refined by :func:`refine_f0` within +-20 cents: pYIN's 10-cent grid and nominal note
    frequencies are both too coarse for exact harmonic fits. Pass ``refine=False`` only when
    ``f0`` is exact (synthetic tones). ``hnr_max_hz`` is passed to :func:`hnr` as ``max_hz``.
    At most ``max_frames`` evenly
    spaced frames are analysed (the median is robust; this bounds the cost on long clips).
    Raises ``ValueError`` if no usable frame is found.
    """
    y = _as_numpy(x)
    if y.ndim != 1:
        raise ValueError(f"harmonic_profile expects mono (T,), got {y.shape}")
    if f0 is not None:
        frame_length = track_kwargs.get("frame_length") or default_frame_length(sr)
        hop = track_kwargs.get("hop_length") or frame_length // 4
        n_frames = 1 + len(y) // hop
        centres = np.arange(n_frames) * hop
        f0s = np.full(n_frames, float(f0))
        voiced_fraction = 1.0
    else:
        track = track or track_f0(y, sr, **track_kwargs)
        frame_length, hop = track.frame_length, track.hop_length
        centres = np.arange(len(track.f0_hz)) * hop
        voiced = np.asarray(track.voiced, dtype=bool)
        f0s = np.where(voiced, track.f0_hz, np.nan)
        # Erode voiced runs (and pitch jumps > 1 semitone) by half a frame on each side.
        r = max(1, (frame_length // 2) // hop)
        steady = np.isfinite(f0s)
        with np.errstate(invalid="ignore"):
            jump = np.abs(np.diff(np.log2(f0s))) > 1 / 12
        steady[1:] &= ~jump
        steady[:-1] &= ~jump
        eroded = steady.copy()
        for lag in range(1, r + 1):
            eroded[lag:] &= steady[:-lag]
            eroded[:-lag] &= steady[lag:]
        if eroded.any():
            f0s = np.where(eroded, f0s, np.nan)
        voiced_fraction = float(np.mean(track.voiced)) if len(track.voiced) else 0.0
    half = frame_length // 2
    ok = np.isfinite(f0s) & (centres - half >= 0) & (centres + half <= len(y))
    idx = np.flatnonzero(ok)
    if len(idx) == 0 and f0 is not None and len(y) * f0 / sr >= 4.5:
        # Clip shorter than one frame: analyse it whole.
        frames = y[None, :]
        f0_list = np.array([float(f0)])
    else:
        if len(idx) > max_frames:
            idx = idx[np.linspace(0, len(idx) - 1, max_frames).round().astype(int)]
        frames = np.stack([y[c - half : c + half] for c in centres[idx]]) if len(idx) else None
        f0_list = f0s[idx]
    if frames is None or len(frames) == 0:
        raise ValueError("no voiced frames to analyse")
    loud = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-20) >= min_rms_db
    frames, f0_list = frames[loud], f0_list[loud]
    if len(frames) == 0:
        raise ValueError(f"all voiced frames are below {min_rms_db} dBFS RMS")
    if refine:
        f0_list = np.array([refine_f0(fr, sr, f) for fr, f in zip(frames, f0_list, strict=True)])
    amps = np.stack(
        [harmonic_amplitudes(fr, sr, f, n_harmonics) for fr, f in zip(frames, f0_list, strict=True)]
    )
    hnrs = np.array(
        [float(hnr(fr, sr, f, hnr_max_hz)) for fr, f in zip(frames, f0_list, strict=True)]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = amps / amps[:, :1]
    with np.errstate(all="ignore"):
        # A harmonic is NaN in the profile if it is above Nyquist in most frames.
        profile = np.nanmedian(rel, axis=0) if np.isfinite(rel).any() else rel[0]
        profile = np.where(np.mean(np.isfinite(rel), axis=0) >= 0.5, profile, np.nan)
    return HarmonicProfile(
        amplitudes=profile,
        h1=float(np.median(amps[:, 0])),
        f0_hz=float(np.median(f0_list)),
        hnr_db=float(np.nanmedian(hnrs)) if np.isfinite(hnrs).any() else float("nan"),
        voiced_fraction=voiced_fraction,
        frame_amplitudes=amps,
        frame_f0=f0_list,
        frame_hnr_db=hnrs,
    )


__all__ = [
    "F0Track",
    "HarmonicProfile",
    "blackman_harris",
    "default_frame_length",
    "harmonic_amplitudes",
    "harmonic_fit",
    "harmonic_profile",
    "hnr",
    "refine_f0",
    "to_db",
    "to_dbc",
    "track_f0",
]
