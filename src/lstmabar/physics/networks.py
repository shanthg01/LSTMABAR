"""Small-signal circuit algebra: rational s-domain functions and the network blocks pedals share.

:class:`TF` is a rational function ``B(s)/A(s)`` (numpy coefficients, highest power first, as
in ``scipy.signal.freqs``). It is used for impedances, admittances and voltage transfer
functions alike, and supports ``+ - * /`` with other ``TF`` objects and with plain numbers,
so network equations read like the circuit analysis:

    z_leg = res(47) + cap(2.2e-6)                  # R4 + 1/(sC5)
    gain = 1 + par(res(100e3), cap(100e-12)) / par(z_leg, res(560) + cap(4.7e-6))

Coefficients are not reduced (common pole/zero factors are kept, and cascades reach degree
13-16). That does not affect the magnitude response, which is all the derivations use (they
evaluate it directly; checked to ~1e-14 dB against products of separately evaluated
factors). **Do not feed a cascade's ``tf.b`` / ``tf.a`` to ``scipy.signal.bilinear``**:
high-degree polynomials discretize poorly. A white-box model should build and discretize
per stage (low-order TFs, or second-order sections), not the unreduced product.

``rc_lowpass_tf`` / ``rc_highpass_tf`` return ``TF`` objects; they are unrelated to the
discrete-time ``whitebox.linear.rc_lowpass`` stage.

Building blocks (all small-signal, ideal op-amps, no device capacitances):

- :func:`noninverting_gain`: ``1 + Z_f / Z_g``.
- :func:`shunt_feedback_gain`: transistor stage with collector-base feedback.
- :func:`ce_collector_feedback_bias`: DC collector current of such a stage.
- :func:`two_leg_blend`: two Thevenin legs joined by a pot wiper into a load (Big Muff / DS-1
  tone stacks), solved exactly by Millman's theorem, so pot and load loading are included.
- :func:`pot_split`: pot rotation -> (low-end-to-wiper, wiper-to-high-end) resistances.
"""

from __future__ import annotations

import math

import numpy as np

from lstmabar.physics.kb import Pot

THERMAL_VOLTAGE = 0.02585
"""V_T = kT/q at ~300 K (volts), for r_e = V_T / I_C."""

POT_END_OHMS = 1.0
"""Smallest pot section resistance (end / wiper contact resistance); keeps the conductance
of a pot section finite when the wiper sits at an end."""


def _poly(c) -> np.ndarray:
    p = np.trim_zeros(np.atleast_1d(np.asarray(c, dtype=float)), "f")
    return p if p.size else np.zeros(1)


class TF:
    """Rational function of ``s``: ``B(s) / A(s)``."""

    __slots__ = ("a", "b")

    def __init__(self, b, a=(1.0,)) -> None:
        b, a = _poly(b), _poly(a)
        if not np.any(a):
            raise ZeroDivisionError("TF denominator is zero")
        k = np.max(np.abs(a))  # keep coefficient magnitudes bounded as degrees grow
        self.b, self.a = b / k, a / k

    @staticmethod
    def _lift(x) -> TF:
        return x if isinstance(x, TF) else TF([float(x)])

    def __add__(self, other) -> TF:
        o = TF._lift(other)
        if self.a.shape == o.a.shape and np.array_equal(self.a, o.a):
            return TF(np.polyadd(self.b, o.b), self.a)
        return TF(
            np.polyadd(np.polymul(self.b, o.a), np.polymul(o.b, self.a)),
            np.polymul(self.a, o.a),
        )

    __radd__ = __add__

    def __neg__(self) -> TF:
        return TF(-self.b, self.a)

    def __sub__(self, other) -> TF:
        return self + (-TF._lift(other))

    def __rsub__(self, other) -> TF:
        return TF._lift(other) + (-self)

    def __mul__(self, other) -> TF:
        o = TF._lift(other)
        return TF(np.polymul(self.b, o.b), np.polymul(self.a, o.a))

    __rmul__ = __mul__

    def __truediv__(self, other) -> TF:
        o = TF._lift(other)
        return TF(np.polymul(self.b, o.a), np.polymul(self.a, o.b))

    def __rtruediv__(self, other) -> TF:
        return TF._lift(other) / self

    def __call__(self, freqs_hz) -> np.ndarray:
        """Complex response at ``s = j·2π·f``."""
        s = 2j * np.pi * np.asarray(freqs_hz, dtype=float)
        return np.polyval(self.b, s) / np.polyval(self.a, s)

    def mag(self, freqs_hz) -> np.ndarray:
        return np.abs(self(freqs_hz))

    def db(self, freqs_hz) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(self.mag(freqs_hz), 1e-12))

    def __repr__(self) -> str:
        return f"TF(b={self.b.tolist()}, a={self.a.tolist()})"


# --- Elements -----------------------------------------------------------------------------------


def res(r: float) -> TF:
    """Resistor impedance ``R``."""
    return TF([r])


def cap(c: float) -> TF:
    """Capacitor impedance ``1 / (sC)``."""
    return TF([1.0], [c, 0.0])


def par(*zs: TF) -> TF:
    """Parallel impedances ``1 / Σ(1/Z)``."""
    y = sum((1 / z for z in zs[1:]), 1 / zs[0])
    return 1 / y


def divider(z_top: TF, z_bottom: TF) -> TF:
    """Unloaded voltage divider ``Z_bottom / (Z_top + Z_bottom)``."""
    return z_bottom / (z_top + z_bottom)


def rc_lowpass_tf(r: float, c: float) -> TF:
    """``1 / (1 + sRC)``."""
    return TF([1.0], [r * c, 1.0])


def rc_highpass_tf(r: float, c: float) -> TF:
    """``sRC / (1 + sRC)``."""
    return TF([r * c, 0.0], [r * c, 1.0])


def rc_hz(r: float, c: float) -> float:
    """First-order RC corner frequency ``1 / (2πRC)``."""
    return 1.0 / (2.0 * math.pi * r * c)


def pot_split(pot: Pot, pos: float) -> tuple[float, float]:
    """``(R_low, R_high)``: resistance from the pot's low end to the wiper and from the wiper
    to the high end at rotation ``pos`` (taper applied), each at least :data:`POT_END_OHMS`.

    "Low end" is the lug the wiper sits on at rotation 0; each deriver states which circuit
    node that is.
    """
    r_low = pot.resistance(pos)
    return max(r_low, POT_END_OHMS), max(pot.value - r_low, POT_END_OHMS)


# --- Stages -------------------------------------------------------------------------------------


def noninverting_gain(z_f: TF, z_g: TF) -> TF:
    """Ideal non-inverting op-amp stage: ``G(s) = 1 + Z_f(s) / Z_g(s)``."""
    return 1 + z_f / z_g


def ce_collector_feedback_bias(
    v_supply: float, r_c: float, r_e: float, r_f: float, r_b: float, v_be: float = 0.6
) -> float:
    """DC collector current of a common-emitter stage biased by collector-base feedback.

    ``R_f`` runs collector -> base, ``R_b`` base -> ground; base current neglected (β -> ∞):

        V_B = V_C · R_b / (R_f + R_b),   V_B = V_BE + I_C·R_e,   V_C = V_supply - I_C·R_c
        => I_C = (V_supply·k - V_BE) / (R_e + k·R_c),   k = R_b / (R_f + R_b)
    """
    k = r_b / (r_f + r_b)
    return max((v_supply * k - v_be) / (r_e + k * r_c), 1e-9)


def ce_open_loop_gain(r_c: float, r_e: float, i_c: float) -> float:
    """Emitter-degenerated common-emitter gain magnitude ``R_c / (R_e + r_e)``,
    ``r_e = V_T / I_C``."""
    return r_c / (r_e + THERMAL_VOLTAGE / i_c)


def shunt_feedback_gain(z_f: TF, z_s: TF, r_in: float, a_ol: float) -> TF:
    """Inverting stage with collector-base (shunt) feedback, magnitude-only sign convention.

    Source ``v_s`` drives the base through ``Z_s``; ``Z_f`` runs collector -> base; the base
    node also sees ``r_in`` to ground (bias resistor; transistor input current neglected);
    the transistor is an ideal inverting amplifier of gain ``A_ol`` (``v_c = -A_ol·v_b``).
    Kirchhoff at the base gives

        v_c / v_s = -(Z_f/Z_s) / (1 + (1 + Z_f/Z_s + Z_f/r_in) / A_ol)

    which tends to the ideal ``-Z_f/Z_s`` for large ``A_ol``. Returned without the sign.
    """
    ratio = z_f / z_s
    return ratio / (1 + (1 + ratio + z_f / r_in) / a_ol)


def two_leg_blend(v_a: TF, z_a: TF, v_b: TF, z_b: TF, z_load: TF) -> TF:
    """Voltage at a node fed by two Thevenin legs ``(v_a, z_a)``, ``(v_b, z_b)`` and loaded by
    ``z_load`` to ground (Millman):

        v = (v_a/z_a + v_b/z_b) / (1/z_a + 1/z_b + 1/z_load)

    Used for passive LP/HP blend tone stacks, where each leg's series impedance includes its
    pot section, so the result is the exact loaded network response.
    """
    return (v_a / z_a + v_b / z_b) / (1 / z_a + 1 / z_b + 1 / z_load)


def thevenin_rc_lowpass(r: float, c: float) -> tuple[TF, TF]:
    """Series R, shunt C to ground, seen from the C node: ``(1/(1+sRC), R ∥ 1/(sC))``."""
    return rc_lowpass_tf(r, c), par(res(r), cap(c))


def thevenin_cr_highpass(c: float, r: float) -> tuple[TF, TF]:
    """Series C, shunt R to ground, seen from the R node: ``(sRC/(1+sRC), R ∥ 1/(sC))``."""
    return rc_highpass_tf(r, c), par(res(r), cap(c))


# --- Response summaries -------------------------------------------------------------------------


def lower_corner_hz(h: TF, f_ref: float, f_min: float = 1.0) -> float:
    """Highest frequency below ``f_ref`` where ``|h|`` is 3 dB below ``|h(f_ref)|``
    (``f_min`` if it never gets there): the -3 dB high-pass corner of a gain path."""
    f = np.geomspace(f_min, f_ref, 2000)
    m = h.mag(f) / h.mag(f_ref)
    below = np.nonzero(m <= 1.0 / math.sqrt(2.0))[0]
    if below.size == 0:
        return f_min
    i = below[-1]
    if i + 1 >= f.size:
        return float(f[i])
    # log-linear interpolation between the bracketing points
    t = (math.log(1 / math.sqrt(2.0)) - math.log(m[i])) / (math.log(m[i + 1]) - math.log(m[i]))
    return float(f[i] * (f[i + 1] / f[i]) ** t)


def response_info(prefix: str, h: TF, freqs_hz=(100.0, 300.0, 1000.0, 3000.0, 10000.0)):
    """``{f"{prefix}_db_{f}hz": |h(f)| dB}`` at a few frequencies, for ``Preset.info``."""
    return {f"{prefix}_db_{int(f)}hz": float(h.db(f)) for f in freqs_hz}


__all__ = [
    "POT_END_OHMS",
    "THERMAL_VOLTAGE",
    "TF",
    "cap",
    "ce_collector_feedback_bias",
    "ce_open_loop_gain",
    "divider",
    "lower_corner_hz",
    "noninverting_gain",
    "par",
    "pot_split",
    "rc_highpass_tf",
    "rc_hz",
    "rc_lowpass_tf",
    "res",
    "response_info",
    "shunt_feedback_gain",
    "thevenin_cr_highpass",
    "thevenin_rc_lowpass",
    "two_leg_blend",
]
