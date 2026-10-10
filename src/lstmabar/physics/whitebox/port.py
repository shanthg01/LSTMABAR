r"""Diode pair at one port of an arbitrary linear RC network (exact loading).

The shunt clipper of :mod:`.diode_clipper` assumes nothing loads the clipping node. When a
passive stage hangs off it (the DS-1's LP/HP tone stack loads its R14/C10/diode node with a
strongly frequency-dependent 3–18 kΩ), the loading changes the response by several dB. This
module handles any linear network driven by one ideal source ``v_s`` with an anti-parallel
diode pair across one port, by superposition:

.. math::

    v_X = T_\text{open}(s)\,v_s - Z_\text{port}(s)\,i_D, \qquad
    y   = T_\text{port,out}(s)\,v_X

``T_open`` is the open-circuit (diodes removed) port voltage per volt of source and
``Z_port`` the network's Thevenin impedance at the port. The output form assumes the
source reaches the rest of the network only through the port node (true when the source
feeds the port through a series element, as in the DS-1), so the output is the response of
the downstream network to the port voltage, ``T_port,out``. All three are exact rational
functions from nodal analysis (:func:`.linear.nodal_transfer`).

**Discretization.** Everything runs at the oversampled rate with the (unwarped) bilinear
transform, i.e. trapezoidal integration of the whole network. ``Z_port(z) = b(z)/a(z)``
(``a_0 = 1``) splits into its feedthrough ``b_0`` (> 0 for a passive RC impedance: it is
``Z_port(s = 2 f_s)``) and a history term ``h[n] = Σ_{k≥1} b_k i[n-k] - Σ_{k≥1} a_k w[n-k]``
with ``w = Z_port * i``. Each sample solves the scalar equation

.. math:: v + b_0\, i_D(v) = v_\text{oc}[n] - h[n]

which has the same form as the one-capacitor solver's (``a = b_0``, ``G = 0``) and uses the
same clamped-start Newton iteration with ``pnjlim`` step limiting. Then ``i[n] = i_D(v)`` and
``w[n] = v_oc[n] - v``. The output is ``T_port,out`` applied to ``v``.

Small-signal (diodes linearized at 0 V, conductance ``g_d``): ``i_D = g_d v_X`` gives
``v_X = T_open v_s / (1 + g_d Z_port)`` (:meth:`PortClipper.response`).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

from lstmabar.physics.whitebox.base import WhiteBoxModel
from lstmabar.physics.whitebox.diode_clipper import (
    _EXP_CAP,
    MAX_ITER,
    OVERSAMPLE_FACTORS,
    TOL,
    DiodePair,
    numba_available,
    resample_down,
    resample_up,
)
from lstmabar.physics.whitebox.linear import (
    AnalogFilter,
    apply_chain,
    chain_response,
    to_digital,
)


def _solve_port_scalar(voc, b, a, isp, sp, vcp, isn, sn, vcn, tol, max_iter, v_out, i_out):
    """Scalar kernel over ``voc`` ``(B, N)``; ``b``/``a`` digital ``Z_port`` with ``a[0] = 1``.
    Compiled by numba when available. Returns the number of unconverged steps."""
    n_batch, n_steps = voc.shape
    order = len(a) - 1
    b0 = b[0]
    bad = 0
    ih = np.zeros(order + 1)  # ih[k] = i[n-k]
    wh = np.zeros(order + 1)  # wh[k] = w[n-k]
    for bi in range(n_batch):
        for k in range(order + 1):
            ih[k] = 0.0
            wh[k] = 0.0
        v = 0.0
        v_old = 0.0
        for n in range(n_steps):
            # shift histories
            for k in range(order, 0, -1):
                ih[k] = ih[k - 1]
                wh[k] = wh[k - 1]
            hist = 0.0
            for k in range(1, order + 1):
                hist += b[k] * ih[k] - a[k] * wh[k]
            rhs = voc[bi, n] - hist
            v_last = v
            v = 2.0 * v - v_old
            v = min(max(v, min(v_last, -vcn)), max(v_last, vcp))
            v_old = v_last
            converged = False
            for _ in range(max_iter):
                ep = math.exp(min(v / sp, _EXP_CAP))
                en = math.exp(min(-v / sn, _EXP_CAP))
                idd = isp * (ep - 1.0) - isn * (en - 1.0)
                did = isp / sp * ep + isn / sn * en
                vn = v - (v + b0 * idd - rhs) / (1.0 + b0 * did)
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
            i_now = isp * (ep - 1.0) - isn * (en - 1.0)
            ih[0] = i_now
            wh[0] = voc[bi, n] - v
            v_out[bi, n] = v
            i_out[bi, n] = i_now
    return bad


_NUMBA_PORT = None


def _kernel(backend: str):
    global _NUMBA_PORT
    if backend == "auto":
        backend = "numba" if numba_available() else "numpy"
    if backend == "numpy":
        return _solve_port_scalar
    if backend != "numba":
        raise ValueError(f"unknown backend {backend!r}")
    if _NUMBA_PORT is None:
        import numba

        _NUMBA_PORT = numba.njit(cache=True)(_solve_port_scalar)
    return _NUMBA_PORT


def solve_port(
    v_open: np.ndarray,
    z_port: AnalogFilter,
    diodes: DiodePair,
    sample_rate: float,
    backend: str = "auto",
    tol: float = TOL,
    max_iter: int = MAX_ITER,
) -> tuple[np.ndarray, np.ndarray]:
    """Port voltage and diode current ``(B, N)`` for open-circuit voltage ``v_open`` ``(B, N)``.

    The ``"numpy"`` backend runs the same scalar kernel in pure Python (slow; tests only).
    """
    bz, az = to_digital(z_port.b, z_port.a, sample_rate)
    bz, az = bz / az[0], az / az[0]
    n = max(len(bz), len(az))
    bz = np.pad(bz, (0, n - len(bz)))
    az = np.pad(az, (0, n - len(az)))
    if not bz[0] > 0:
        raise ValueError(f"port impedance feedthrough must be positive, got {bz[0]}")
    voc = np.ascontiguousarray(v_open, dtype=np.float64)
    v = np.empty_like(voc)
    i = np.empty_like(voc)
    bad = _kernel(backend)(voc, bz, az, *diodes.params(), tol, max_iter, v, i)
    if bad:
        raise RuntimeError(f"Newton did not converge on {bad} steps")
    return v, i


@dataclass
class PortClipper(WhiteBoxModel):
    """``pre`` → (source ``v_s``) linear network with a diode pair at one port → ``post`` →
    ``out_gain``; everything except ``out_gain`` runs at the oversampled rate."""

    diodes: DiodePair
    z_port: AnalogFilter
    t_open: AnalogFilter
    t_port_out: AnalogFilter
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
        """Analytic small-signal response (diodes linearized at 0 V), volts out / volts in."""
        gd = self.diodes.conductance0()
        vx = self.t_open.response(freqs_hz) / (1.0 + gd * self.z_port.response(freqs_hz))
        y = self.t_port_out.response(freqs_hz) * vx
        return (
            chain_response(self.pre, freqs_hz)
            * y
            * chain_response(self.post, freqs_hz)
            * (self.out_gain)
        )

    def _filt(self, f: AnalogFilter, x: np.ndarray, fs: float) -> np.ndarray:
        bz, az = to_digital(f.b, f.a, fs)  # unwarped: consistent with the port solver
        return lfilter(bz, az, x, axis=-1)

    def process_volts(self, v: np.ndarray) -> np.ndarray:
        x = np.asarray(v, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected (B, T) volts, got shape {x.shape}")
        k = self.oversample
        fs = self.sample_rate * k
        vs = apply_chain(self.pre, resample_up(x, k), fs)
        vx, _ = solve_port(
            self._filt(self.t_open, vs, fs), self.z_port, self.diodes, fs, self.backend
        )
        y = self._filt(self.t_port_out, vx, fs)
        y = apply_chain(self.post, y, fs)
        return self.out_gain * resample_down(y, k)


__all__ = ["PortClipper", "solve_port"]
