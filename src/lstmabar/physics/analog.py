"""Linear-stage helpers: analog magnitude responses and fitting them with grey-box tone knobs.

Pedal tone stacks are linear, so their exact magnitude response follows from component
values (an s-domain transfer function). :func:`fit_tone` finds the ``Drive.tone_db`` tilt and
``EQ3`` settings (plus a broadband gain, folded into ``drive.level_db`` by the caller) whose
digital response best matches it in dB on a log-spaced grid. Multi-start over ``mid_hz``,
since a swept peak has local minima (see reports/param_recovery.md).
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.signal import freqs as _analog_freqs

from lstmabar.dsp.blocks import EQ3, Drive
from lstmabar.dsp.filters import biquad_coeffs

TILT_PIVOT_HZ = 1000.0
_SHELF_Q = 1.0 / math.sqrt(2.0)


def log_grid(lo: float = 40.0, hi: float = 10000.0, n: int = 64) -> np.ndarray:
    """Log-spaced frequency grid (Hz) used for tone fitting."""
    return np.geomspace(lo, hi, n)


def analog_response_db(b: np.ndarray, a: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    """Magnitude (dB) of the analog filter ``B(s)/A(s)`` (coefficients highest power first)."""
    _, h = _analog_freqs(np.asarray(b, float), np.asarray(a, float), worN=2 * np.pi * freqs_hz)
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))


def _biquad_db(kind: str, f: float, q: float, g: float, freqs_hz: np.ndarray, sr: int):
    t = torch.tensor
    b, a = biquad_coeffs(kind, t([f]), t([q]), t([g]), sr)
    z = torch.exp(-1j * 2 * math.pi * torch.as_tensor(freqs_hz, dtype=torch.float64) / sr)
    zz = torch.stack([torch.ones_like(z), z, z * z])
    num = (b[0].to(torch.complex128)[:, None] * zz).sum(0)
    den = (a[0].to(torch.complex128)[:, None] * zz).sum(0)
    return (20.0 * torch.log10((num / den).abs().clamp_min(1e-12))).numpy()


def greybox_tone_db(
    freqs_hz: np.ndarray,
    sample_rate: int,
    tone_db: float = 0.0,
    low_db: float = 0.0,
    mid_db: float = 0.0,
    mid_hz: float = 800.0,
    high_db: float = 0.0,
) -> np.ndarray:
    """Magnitude (dB) of ``Drive``'s post-clip tilt followed by ``EQ3`` (same filters)."""
    f = np.asarray(freqs_hz, float)
    return (
        _biquad_db("lowshelf", TILT_PIVOT_HZ, _SHELF_Q, -tone_db / 2, f, sample_rate)
        + _biquad_db("highshelf", TILT_PIVOT_HZ, _SHELF_Q, tone_db / 2, f, sample_rate)
        + _biquad_db("lowshelf", EQ3.LOW_HZ, _SHELF_Q, low_db, f, sample_rate)
        + _biquad_db("peak", mid_hz, EQ3.MID_Q, mid_db, f, sample_rate)
        + _biquad_db("highshelf", EQ3.HIGH_HZ, _SHELF_Q, high_db, f, sample_rate)
    )


@dataclass(frozen=True)
class ToneFit:
    """Grey-box tone settings (physical units) matching a target response."""

    tone_db: float
    low_db: float
    mid_db: float
    mid_hz: float
    high_db: float
    gain_db: float  # broadband offset; add to drive.level_db
    rms_error_db: float
    at_bounds: tuple[str, ...] = ()  # knobs pinned at a range limit (fit may be range-limited)

    def drive_knobs(self) -> dict[str, float]:
        return {"tone_db": self.tone_db}

    def eq_knobs(self) -> dict[str, float]:
        return {
            "low_db": self.low_db,
            "mid_db": self.mid_db,
            "mid_hz": self.mid_hz,
            "high_db": self.high_db,
        }


def _spec(block, name):
    return next(s for s in block.param_specs if s.name == name)


def fit_tone(
    freqs_hz: np.ndarray,
    target_db: np.ndarray,
    sample_rate: int = 44100,
    use_eq: bool = True,
    mid_starts_hz: tuple[float, ...] = (300.0, 700.0, 1500.0, 3000.0),
) -> ToneFit:
    """Least-squares fit of tilt (+ ``EQ3`` if ``use_eq``) + broadband gain to ``target_db``.

    Knob bounds are the block ``ParamSpec`` ranges; ``mid_hz`` is optimized in log2 units.
    With ``use_eq=False`` only ``tone_db`` and the gain move (EQ stays flat).
    """
    f = np.asarray(freqs_hz, float)
    target = np.asarray(target_db, float)
    tone = _spec(Drive, "tone_db")
    eq = {n: _spec(EQ3, n) for n in ("low_db", "mid_db", "mid_hz", "high_db")}
    lo_hz, hi_hz = math.log2(eq["mid_hz"].min), math.log2(eq["mid_hz"].max)

    def unpack(v):
        if use_eq:
            t, lo, md, lmh, hi, g = v
            return t, lo, md, 2.0**lmh, hi, g
        t, g = v
        return t, 0.0, 0.0, 800.0, 0.0, g

    def resid(v):
        t, lo, md, mh, hi, g = unpack(v)
        return greybox_tone_db(f, sample_rate, t, lo, md, mh, hi) + g - target

    if use_eq:
        lb = [tone.min, eq["low_db"].min, eq["mid_db"].min, lo_hz, eq["high_db"].min, -60.0]
        ub = [tone.max, eq["low_db"].max, eq["mid_db"].max, hi_hz, eq["high_db"].max, 60.0]
        starts = [[0.0, 0.0, 0.0, math.log2(m), 0.0, 0.0] for m in mid_starts_hz]
    else:
        lb, ub = [tone.min, -60.0], [tone.max, 60.0]
        starts = [[0.0, 0.0]]

    best = None
    for x0 in starts:
        r = least_squares(resid, x0, bounds=(lb, ub))
        if best is None or r.cost < best.cost:
            best = r
    t, lo, md, mh, hi, g = unpack(best.x)
    rms = float(np.sqrt(np.mean(best.fun**2)))
    names = ["tone_db", "low_db", "mid_db", "mid_hz", "high_db", "gain_db"]
    if not use_eq:
        names = ["tone_db", "gain_db"]
    pinned = tuple(
        n
        for n, x, a, b in zip(names, best.x, lb, ub, strict=True)
        if min(abs(x - a), abs(x - b)) < 1e-6 * max(1.0, abs(a), abs(b))
    )
    return ToneFit(float(t), float(lo), float(md), float(mh), float(hi), float(g), rms, pinned)


__all__ = [
    "ToneFit",
    "analog_response_db",
    "fit_tone",
    "greybox_tone_db",
    "log_grid",
]
