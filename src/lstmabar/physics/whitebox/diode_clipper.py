r"""One-capacitor diode-clipper solver in two configurations (P2 decision 5).

Both circuits reduce to one nonlinear state equation for a capacitor voltage ``v``:

.. math::  C \, \dot v = j(t) - G\,v - i_D(v)

where ``i_D`` is an anti-parallel pair of diode strings (Shockley, :mod:`.devices`)::

    i_D(v) = Is·(exp(v / (n_pos·N·Vt)) - 1) - Is·(exp(-v / (n_neg·N·Vt)) - 1)

(a side with ``n = 0`` diodes is absent, i.e. that polarity never clips).

**Shunt** (RAT / DS-1 clipping stage): source ``v_in`` → series ``R`` → node ``v``; ``C`` and
the diodes from the node to ground. KCL at the node: ``(v_in - v)/R = C v' + i_D(v)``, so
``j = v_in / R``, ``G = 1/R`` and the output is ``v``. Small-signal response
``H(s) = (1/R) / (1/R + g_d + sC)``.

**Feedback** (TS808 clipping stage): ideal non-inverting op-amp, ``v+ = v- = v_in``. The gain
leg ``R_gain`` + ``C_gain`` in series from ``v-`` to ground carries ``i_g``, whose only input
is ``v_in`` (the leg sees ``v-`` = ``v_in``), so it is a *linear* source current:
``I_g(s) = V_in(s) · sC_gain / (1 + sR_gain C_gain)``. The feedback network (``R_fb`` ∥
``C_fb`` ∥ diodes) between output and ``v-`` carries the same current; its voltage is
``v = v_out - v_in``: ``i_g = v/R_fb + C_fb v' + i_D(v)``. So ``j = i_g``, ``G = 1/R_fb``,
``C = C_fb`` and the output is ``v_out = v_in + v`` (the unity clean path survives).
Small-signal response ``H(s) = 1 + Y_g(s) / (1/R_fb + g_d + sC_fb)`` with
``Y_g = sC_gain / (1 + sR_gain C_gain)``.

``g_d = Is/(n_pos N Vt) + Is/(n_neg N Vt)`` is the diodes' small-signal conductance at 0 V
(≈9 MΩ for the 1N4148-class pair; it lowers the TS808 small-signal gain by ≈0.1 dB at mid
drive and ≈0.5 dB at full drive, where ``R_fb`` = 551 kΩ).

**Discretization.** At the oversampled rate ``fs`` (``T = 1/fs``) the state equation uses
the trapezoidal rule, with ``a = T / (2C)`` and ``h(v) = G v + i_D(v)``::

    v[n] + a·h(v[n]) = v[n-1] - a·h(v[n-1]) + a·(j[n] + j[n-1])

The left side is strictly increasing in ``v``, so the solution is unique. It is found per
sample with Newton–Raphson started from the linear extrapolation ``2·v[n-1] - v[n-2]``,
clamped to ``[min(v[n-1], -vcrit_neg), max(v[n-1], vcrit_pos)]`` so the start never lies
deep in a conducting string (Newton descends an exponential from above by only about one
thermal slope per iteration), with SPICE ``pnjlim`` step limiting on each diode string's
junction voltage (keeps steps up the exponential from overflowing) until ``|Δv| < tol``
(1e-10 V; at most ``MAX_ITER`` iterations, else ``solve`` raises). The feedback source
``j`` is ``Y_g`` discretized by the same (unwarped) bilinear transform, so the linearized
system equals the bilinear transform of the analog ``H(s)``. Trapezoidal integration is
A-stable but not L-stable; with a stiff conducting diode it can ring at Nyquist of the
oversampled rate, which the decimation low-pass removes.

**Signal chain** of a :class:`DiodeClipper`: ``pre`` linear stages (base rate) → upsample
(the zero-phase Kaiser-sinc prototype of :mod:`lstmabar.dsp.oversample`, ~90 dB stopband,
applied in float64 with ``scipy.signal.upfirdn``, which is much faster than torch's
float64 ``conv1d`` on CPU) → solver → downsample → ``post`` linear stages (base rate) →
``out_gain``. Op-amps are ideal (no rails, slew or bandwidth limit).

**Backends.** The time loop runs in Python with per-step NumPy over the batch (``"numpy"``),
or, when the optional ``numba`` package is installed (``uv sync --extra whitebox``), in a
compiled scalar kernel (``"numba"``, >100× faster: the NumPy loop runs at ~0.1× real time
for small batches). ``"auto"`` picks numba when available.
``python -m lstmabar.physics.whitebox.bench`` measures throughput.
"""

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter, upfirdn

from lstmabar.dsp.oversample import HALF_WIDTH, _prototype
from lstmabar.physics.whitebox.base import WhiteBoxModel
from lstmabar.physics.whitebox.devices import VT, Diode, diode
from lstmabar.physics.whitebox.linear import (
    AnalogFilter,
    apply_chain,
    chain_response,
    to_digital,
)

CONFIGS = ("shunt", "feedback")
OVERSAMPLE_FACTORS = (1, 2, 4, 8)
TOL = 1e-10  # Newton convergence threshold on |Δv| (volts)
MAX_ITER = 100
_EXP_CAP = 80.0  # exponent clamp; pnjlim keeps real iterates far below this


# --- Diode strings ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DiodePair:
    """Anti-parallel strings: ``n_pos`` diodes conduct for ``v > 0``, ``n_neg`` for ``v < 0``."""

    device: Diode
    n_pos: int = 1
    n_neg: int = 1
    vt: float = VT

    def __post_init__(self):
        if self.n_pos < 0 or self.n_neg < 0 or self.n_pos + self.n_neg == 0:
            raise ValueError("n_pos/n_neg must be >= 0 and not both 0")

    def branch(self, n: int) -> tuple[float, float, float]:
        """``(Is, slope, vcrit)`` for one string; an absent string is ``(0, 1, inf)``."""
        if n == 0:
            return 0.0, 1.0, math.inf
        s = self.device.thermal_slope(n, self.vt)
        return self.device.Is, s, s * math.log(s / (math.sqrt(2.0) * self.device.Is))

    def params(self) -> tuple[float, ...]:
        return (*self.branch(self.n_pos), *self.branch(self.n_neg))

    def current(self, v: np.ndarray) -> np.ndarray:
        isp, sp, _, isn, sn, _ = self.params()
        return isp * np.expm1(np.asarray(v) / sp) - isn * np.expm1(-np.asarray(v) / sn)

    def conductance0(self) -> float:
        """Small-signal conductance at 0 V (siemens)."""
        isp, sp, _, isn, sn, _ = self.params()
        return isp / sp + isn / sn


# --- Solver -------------------------------------------------------------------------------------


def _pnjlim_np(vn: np.ndarray, v: np.ndarray, s: float, vcrit: float) -> np.ndarray:
    m = (vn > vcrit) & (np.abs(vn - v) > 2.0 * s)
    if not m.any():
        return vn
    vo, vnn = v[m], vn[m]
    arg = 1.0 + (vnn - vo) / s
    up = np.where(arg > 0, vo + s * np.log(np.where(arg > 0, arg, 1.0)), vcrit)
    fresh = s * np.log(np.maximum(vnn / s, 1e-300))
    vn = vn.copy()
    vn[m] = np.where(vo > 0, up, fresh)
    return vn


def _solve_numpy(j: np.ndarray, a: float, g: float, p: tuple[float, ...], tol, max_iter):
    """``j`` is ``(N, B)``; returns ``(v (N, B), number of unconverged steps)``."""
    isp, sp, vcp, isn, sn, vcn = p
    n_steps, b = j.shape
    out = np.empty_like(j)
    v = np.zeros(b)
    v_old = np.zeros(b)
    h_prev = np.zeros(b)
    j_prev = np.zeros(b)
    bad = 0
    for n in range(n_steps):
        jn = j[n]
        rhs = v - a * h_prev + a * (jn + j_prev)
        # Newton start: linear extrapolation, clamped so it never lands further into a
        # conducting string than v[n-1] or that string's vcrit (descending an exponential
        # from far above the root costs ~one thermal slope per iteration).
        lo, hi = np.minimum(v, -vcn), np.maximum(v, vcp)
        v, v_old = np.minimum(np.maximum(2.0 * v - v_old, lo), hi), v
        for _ in range(max_iter):
            ep = np.exp(np.minimum(v / sp, _EXP_CAP))
            en = np.exp(np.minimum(-v / sn, _EXP_CAP))
            h = g * v + isp * (ep - 1.0) - isn * (en - 1.0)
            dh = g + isp / sp * ep + isn / sn * en
            vn = v - (v + a * h - rhs) / (1.0 + a * dh)
            vn = _pnjlim_np(vn, v, sp, vcp)
            vn = -_pnjlim_np(-vn, -v, sn, vcn)
            done = np.max(np.abs(vn - v)) < tol
            v = vn
            if done:
                break
        else:
            bad += 1
        ep = np.exp(np.minimum(v / sp, _EXP_CAP))
        en = np.exp(np.minimum(-v / sn, _EXP_CAP))
        h_prev = g * v + isp * (ep - 1.0) - isn * (en - 1.0)
        j_prev = jn
        out[n] = v
    return out, bad


def _solve_scalar(j, a, g, isp, sp, vcp, isn, sn, vcn, tol, max_iter, out):
    """Scalar kernel over ``j`` ``(B, N)``; compiled by numba when available.

    Same algorithm as :func:`_solve_numpy`, with ``pnjlim`` inlined for both strings
    (numba compiles it as one function).
    """
    n_batch, n_steps = j.shape
    bad = 0
    for bi in range(n_batch):
        v = 0.0
        v_old = 0.0
        h_prev = 0.0
        j_prev = 0.0
        for n in range(n_steps):
            jn = j[bi, n]
            rhs = v - a * h_prev + a * (jn + j_prev)
            v_last = v
            v = 2.0 * v - v_old  # clamped linear extrapolation (see _solve_numpy)
            v = min(max(v, min(v_last, -vcn)), max(v_last, vcp))
            v_old = v_last
            converged = False
            for _ in range(max_iter):
                ep = math.exp(min(v / sp, _EXP_CAP))
                en = math.exp(min(-v / sn, _EXP_CAP))
                h = g * v + isp * (ep - 1.0) - isn * (en - 1.0)
                dh = g + isp / sp * ep + isn / sn * en
                vn = v - (v + a * h - rhs) / (1.0 + a * dh)
                if vn > vcp and abs(vn - v) > 2.0 * sp:  # pnjlim, positive string
                    if v > 0:
                        arg = 1.0 + (vn - v) / sp
                        vn = v + sp * math.log(arg) if arg > 0 else vcp
                    else:
                        vn = sp * math.log(vn / sp)
                if -vn > vcn and abs(vn - v) > 2.0 * sn:  # pnjlim, negative string
                    if -v > 0:
                        arg = 1.0 + (v - vn) / sn
                        vn = v - sn * math.log(arg) if arg > 0 else -vcn
                    else:
                        vn = -sn * math.log(-vn / sn)
                dv = abs(vn - v)
                v = vn
                if dv < tol:
                    converged = True
                    break
            if not converged:
                bad += 1
            ep = math.exp(min(v / sp, _EXP_CAP))
            en = math.exp(min(-v / sn, _EXP_CAP))
            h_prev = g * v + isp * (ep - 1.0) - isn * (en - 1.0)
            j_prev = jn
            out[bi, n] = v
    return bad


_NUMBA_KERNEL = None


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def _numba_kernel():
    global _NUMBA_KERNEL
    if _NUMBA_KERNEL is None:
        import numba

        _NUMBA_KERNEL = numba.njit(cache=True)(_solve_scalar)
    return _NUMBA_KERNEL


def solve(
    j: np.ndarray,
    capacitance: float,
    conductance: float,
    diodes: DiodePair,
    sample_rate: float,
    backend: str = "auto",
    tol: float = TOL,
    max_iter: int = MAX_ITER,
) -> np.ndarray:
    """Integrate ``C v' = j - G v - i_D(v)`` from rest; ``j`` is ``(B, N)`` amps at
    ``sample_rate``. Returns ``v`` ``(B, N)`` volts. Raises if Newton fails to converge."""
    if backend == "auto":
        backend = "numba" if numba_available() else "numpy"
    j = np.ascontiguousarray(j, dtype=np.float64)
    a = 1.0 / (2.0 * sample_rate * capacitance)
    p = diodes.params()
    if backend == "numba":
        out = np.empty_like(j)
        bad = _numba_kernel()(j, a, conductance, *p, tol, max_iter, out)
    elif backend == "numpy":
        out_t, bad = _solve_numpy(np.ascontiguousarray(j.T), a, conductance, p, tol, max_iter)
        out = out_t.T.copy()
    else:
        raise ValueError(f"unknown backend {backend!r}")
    if bad:
        raise RuntimeError(f"Newton did not converge on {bad} steps")
    return out


# --- Resampling ---------------------------------------------------------------------------------


@functools.cache
def _taps(factor: int) -> np.ndarray:
    return _prototype(factor).numpy()


def resample_up(x: np.ndarray, factor: int) -> np.ndarray:
    """``(B, T)`` → ``(B, T·factor)``, zero-phase, same filter as ``dsp.oversample.upsample``."""
    if factor == 1:
        return x
    t = x.shape[-1]
    y = upfirdn(factor * _taps(factor), x, up=factor, axis=-1)
    d = HALF_WIDTH * factor  # group delay of the centred prototype
    return np.ascontiguousarray(y[..., d : d + t * factor])


def resample_down(x: np.ndarray, factor: int) -> np.ndarray:
    """``(B, T·factor)`` → ``(B, T)``, zero-phase, same filter as ``dsp.oversample.downsample``."""
    if factor == 1:
        return x
    t = x.shape[-1] // factor
    y = upfirdn(_taps(factor), x, down=factor, axis=-1)
    return np.ascontiguousarray(y[..., HALF_WIDTH : HALF_WIDTH + t])


# --- Circuits -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipperCircuit:
    """Component values of the nonlinear stage (SI units); see the module docstring."""

    config: str
    C: float  # shunt: node capacitor; feedback: C_fb
    R: float  # shunt: series resistor; feedback: R_fb
    diodes: DiodePair
    R_gain: float | None = None  # feedback only
    C_gain: float | None = None  # feedback only

    def __post_init__(self):
        if self.config not in CONFIGS:
            raise ValueError(f"config must be one of {CONFIGS}, got {self.config!r}")
        vals = [self.C, self.R]
        if self.config == "feedback":
            if self.R_gain is None or self.C_gain is None:
                raise ValueError("feedback config needs R_gain and C_gain")
            vals += [self.R_gain, self.C_gain]
        if not all(math.isfinite(x) and x > 0 for x in vals):
            raise ValueError(f"component values must be positive and finite: {vals}")

    @property
    def G(self) -> float:
        return 1.0 / self.R

    def source(self) -> AnalogFilter:
        """Admittance from ``v_in`` to the source current ``j``."""
        if self.config == "shunt":
            return AnalogFilter((1.0 / self.R,), (1.0,))
        return AnalogFilter((self.C_gain, 0.0), (self.R_gain * self.C_gain, 1.0))

    def response(self, freqs_hz: np.ndarray | float) -> np.ndarray:
        """Analytic small-signal response ``V_out / V_in`` (diodes linearized at 0 V)."""
        s = 2j * math.pi * np.asarray(freqs_hz, dtype=np.float64)
        node = self.source().response(freqs_hz) / (self.G + self.diodes.conductance0() + s * self.C)
        return node if self.config == "shunt" else 1.0 + node


@dataclass
class DiodeClipper(WhiteBoxModel):
    """``pre`` → oversampled clipper → ``post`` → ``out_gain`` (see the module docstring)."""

    circuit: ClipperCircuit
    pre: Sequence[AnalogFilter] = ()
    post: Sequence[AnalogFilter] = ()
    out_gain: float = 1.0
    oversample: int = 4
    sample_rate: int = 44100
    backend: str = "auto"
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self):
        if self.oversample not in OVERSAMPLE_FACTORS:
            raise ValueError(f"oversample must be one of {OVERSAMPLE_FACTORS}")
        super().__init__(self.sample_rate)

    def response(self, freqs_hz: np.ndarray | float) -> np.ndarray:
        """Analytic small-signal response of the whole chain (volts out / volts in)."""
        return (
            chain_response(self.pre, freqs_hz)
            * self.circuit.response(freqs_hz)
            * chain_response(self.post, freqs_hz)
            * self.out_gain
        )

    def clip_stage(self, v_up: np.ndarray, fs_up: float) -> np.ndarray:
        """Nonlinear stage on oversampled ``(B, N)`` volts."""
        src = self.circuit.source()
        bz, az = to_digital(src.b, src.a, fs_up)  # unwarped: consistent with trapezoid
        j = lfilter(bz, az, v_up, axis=-1)
        v = solve(j, self.circuit.C, self.circuit.G, self.circuit.diodes, fs_up, self.backend)
        return v if self.circuit.config == "shunt" else v_up + v

    def process_volts(self, v: np.ndarray) -> np.ndarray:
        x = np.asarray(v, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected (B, T) volts, got shape {x.shape}")
        x = apply_chain(self.pre, x, self.sample_rate)
        k = self.oversample
        y = resample_down(self.clip_stage(resample_up(x, k), self.sample_rate * k), k)
        y = apply_chain(self.post, y, self.sample_rate)
        return self.out_gain * y


def shunt_clipper(
    R: float,
    C: float,
    part: str = "1N914",
    n_pos: int = 1,
    n_neg: int = 1,
    *,
    device: str = "si",
    pre: Sequence[AnalogFilter] = (),
    post: Sequence[AnalogFilter] = (),
    out_gain: float = 1.0,
    oversample: int = 4,
    sample_rate: int = 44100,
    backend: str = "auto",
) -> DiodeClipper:
    """Series ``R`` from the source, ``C`` and diodes to ground (RAT / DS-1 clipping stage).

    Gain stages before the clipper (ideal op-amps) go in ``pre`` as linear filters (an
    ``AnalogFilter`` with numerator scaled by the gain).
    """
    circuit = ClipperCircuit("shunt", C=C, R=R, diodes=DiodePair(diode(part, device), n_pos, n_neg))
    return DiodeClipper(
        circuit, tuple(pre), tuple(post), out_gain, oversample, sample_rate, backend
    )


def feedback_clipper(
    R_gain: float,
    C_gain: float,
    R_fb: float,
    C_fb: float,
    part: str = "1N914",
    n_pos: int = 1,
    n_neg: int = 1,
    *,
    device: str = "si",
    pre: Sequence[AnalogFilter] = (),
    post: Sequence[AnalogFilter] = (),
    out_gain: float = 1.0,
    oversample: int = 4,
    sample_rate: int = 44100,
    backend: str = "auto",
) -> DiodeClipper:
    """Non-inverting op-amp stage with diodes in the feedback path (TS808 clipping stage)."""
    circuit = ClipperCircuit(
        "feedback",
        C=C_fb,
        R=R_fb,
        diodes=DiodePair(diode(part, device), n_pos, n_neg),
        R_gain=R_gain,
        C_gain=C_gain,
    )
    return DiodeClipper(
        circuit, tuple(pre), tuple(post), out_gain, oversample, sample_rate, backend
    )


__all__ = [
    "CONFIGS",
    "ClipperCircuit",
    "DiodeClipper",
    "DiodePair",
    "feedback_clipper",
    "numba_available",
    "resample_down",
    "resample_up",
    "shunt_clipper",
    "solve",
]
