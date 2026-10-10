r"""Boss DS-1 white-box model (component names from ``pedals/ds1.yaml``).

Chain (all linear stages at the oversampled rate):

1. **Transistor booster (Q2), linear.** First-order coupling high-pass ``C_boost_in2`` /
   ``R_boost_bias`` (C3/R5, 33 Hz, the corner ElectroSmash states) and an inverting gain

   .. math::

      A_0 = \frac{R_c \parallel R_{fb} \parallel R_{10}}{R_e + r_e},\quad
      r_e = \frac{V_T}{I_c},\; I_c = \frac{V_{supply} - V_{collector}}{R_c},\qquad
      A_b = A_0\,\frac{C_3}{C_3 + C_4 (1 + A_0)}

   ``A_0`` is the common-emitter gain with the collector loaded by R8, the feedback
   resistor R7 and the next stage's bias resistor R10, and ``r_e`` from the collector bias
   (5.8 V). The collector-base cap C4 appears at the base multiplied by ``1 + A_0`` (Miller)
   and forms a capacitive divider with the coupling cap C3 when driven from the low-impedance
   input buffer: ``A_b`` ≈ 59 (35.5 dB), consistent with ElectroSmash's ≈56 (35 dB). The
   frequency dependence of that Miller loading (it would also move the C3 corner) is **not**
   modelled: the sources give no node-level netlist for the booster, so the stated 33 Hz
   corner is used. **The booster's soft asymmetric clipping is ignored (ideal, unbounded
   transistor)**; P2 decision 4 accepts this cascaded-stage gap.
2. **Op-amp gain stage**, ideal non-inverting: input coupling ``C_opamp_in`` /
   ``R_opamp_bias`` (C5/R10, 23 Hz), gain ``1 + Z_f / Z_g`` with ``Z_f = R_dist ∥ C_fb``
   (VR1 ∥ C7) and ``Z_g = R_gain + 1/(s C_gain)`` (R13 + C8): 0 dB at Dist min, 26.5 dB above
   the ≈72 Hz C8/R13 corner at Dist max, rolled off above ``1/(2π R_dist C_fb)``.
3. **Clipping node + tone stack + level load, one linear network** solved exactly with
   the diodes at the clipping node (:class:`.port.PortClipper`). The op-amp output drives
   ``R_clip`` (R14) into node X, with ``C_clip`` (C10) and the 1N4148 pair to AC ground. X
   feeds the Big-Muff-style tone stack: ``R_tone_lp`` (R16) to node L, ``C_tone_lp`` (C12)
   L→ground; ``C_tone_hp`` (C11) to node H, ``R_tone_hp`` (R17) H→ground; the tone pot runs
   L→H with the wiper W as the output (clockwise = brighter = wiper toward H). W is loaded by
   the level pot's input resistance (step 4). The tone stack loads X with 3–18 kΩ across the
   band, against R14's 2.2 kΩ, which is why it is not treated as a separate unloaded filter.
4. **Level.** Per Premier Guitar's signal-path description, the level pot (VR2) follows the
   tone stage and its wiper drives ``R_tone_out`` (R18, series) into the output buffer,
   whose bias ``R_out_bias_a ∥ R_out_bias_b`` (500 kΩ) is the load: the pot is a divider
   with its lower leg ∥ (R18 + 500 kΩ), and the pot's input resistance loads W.

Not modelled (documented): coupling high-passes below 10 Hz (C1/R2 7.2 Hz, C2/R4 3.3 Hz,
C13/R20 3.4 Hz, C14/R23 1.6 Hz: < 0.1 dB at 82 Hz), C9 (the series coupling ahead of R14,
whose placement is inferred and which only forms a ≈2% capacitive divider with C10), the
JFET switching, buffer non-idealities, op-amp rails/slew/bandwidth. With ideal stages the
booster + op-amp gain (up to ≈62 dB) is unbounded, so the model reaches the diodes with far
more than the supply could deliver; the diode node is still bounded near ±0.7 V.
"""

from collections.abc import Mapping

from lstmabar.physics.kb import Pedal
from lstmabar.physics.whitebox.base import register_whitebox
from lstmabar.physics.whitebox.devices import VT, diode
from lstmabar.physics.whitebox.diode_clipper import DiodePair
from lstmabar.physics.whitebox.linear import (
    AnalogFilter,
    gain,
    nodal_transfer,
    rc_highpass,
)
from lstmabar.physics.whitebox.port import PortClipper

POT_MIN_OHMS = 1.0
"""Pot segments are floored at 1 Ω so the nodal matrices stay finite at the knob ends."""

NOTES = (
    "transistor booster linear (its soft asymmetric clipping is ignored)",
    "booster Miller-loading frequency dependence not modelled (stated 33 Hz corner used)",
    "op-amps ideal (no rails, slew or bandwidth limit); buffers ideal unity",
    "coupling high-passes below 10 Hz and C9 not modelled",
)


def _par(*rs: float) -> float:
    return 1.0 / sum(1.0 / r for r in rs)


def booster_gain(pedal: Pedal) -> float:
    """Magnitude of the Q2 booster's mid-band voltage gain ``A_b`` (see module doc)."""
    ic = (pedal.c("V_supply") - pedal.c("V_boost_collector")) / pedal.c("R_boost_c")
    re = VT / ic
    rc = _par(pedal.c("R_boost_c"), pedal.c("R_boost_fb"), pedal.c("R_opamp_bias"))
    a0 = rc / (pedal.c("R_boost_e") + re)
    c3, c4 = pedal.c("C_boost_in2"), pedal.c("C_boost_fb")
    return a0 * c3 / (c3 + c4 * (1.0 + a0))


def opamp_stage(pedal: Pedal, dist: float) -> AnalogFilter:
    """Non-inverting gain ``1 + Z_f/Z_g`` (VR1 ∥ C7 over R13 + C8) at Dist rotation ``dist``."""
    rd = pedal.pots["dist"].resistance(dist)
    a = rd * pedal.c("C_fb")
    b = pedal.c("R_gain") * pedal.c("C_gain")
    c8rd = pedal.c("C_gain") * rd
    return AnalogFilter((a * b, a + b + c8rd, 1.0), (a * b, a + b, 1.0))


def level_divider(pedal: Pedal, level: float) -> tuple[float, float]:
    """``(gain, input resistance)`` of the level pot loaded by R18 + the output-buffer bias."""
    pot = pedal.pots["level"]
    f = pot.fraction(level)
    r_load = pedal.c("R_tone_out") + _par(pedal.c("R_out_bias_a"), pedal.c("R_out_bias_b"))
    lower = _par(max(f * pot.value, POT_MIN_OHMS), r_load) if f > 0 else 0.0
    r_in = (1.0 - f) * pot.value + lower
    return lower / r_in, r_in


def _tone_parts(pedal: Pedal, tone: float, r_level_in: float):
    pot = pedal.pots["tone"]
    f = pot.fraction(tone)
    ga = 1.0 / max(pot.value * f, POT_MIN_OHMS)  # L - W
    gb = 1.0 / max(pot.value * (1.0 - f), POT_MIN_OHMS)  # W - H
    g16, c12 = 1.0 / pedal.c("R_tone_lp"), pedal.c("C_tone_lp")
    c11, g17 = pedal.c("C_tone_hp"), 1.0 / pedal.c("R_tone_hp")
    return ga, gb, g16, c12, c11, g17, 1.0 / r_level_in


def tone_network(pedal: Pedal, tone: float, r_level_in: float) -> AnalogFilter:
    """Tone stack driven by an ideal source at the clipping node: ``V_W / V_X``."""
    ga, gb, g16, c12, c11, g17, go = _tone_parts(pedal, tone, r_level_in)
    y = [
        [[c12, g16 + ga], [0.0], [-ga]],
        [[0.0], [c11, g17 + gb], [-gb]],
        [[-ga], [-gb], [ga + gb + go]],
    ]
    return nodal_transfer(y, [[g16], [c11, 0.0], [0.0]], out=2)


def clip_port(pedal: Pedal, tone: float, r_level_in: float) -> tuple[AnalogFilter, AnalogFilter]:
    """``(T_open, Z_port)`` at the clipping node X (nodes X, L, H, W; source via R14)."""
    ga, gb, g16, c12, c11, g17, go = _tone_parts(pedal, tone, r_level_in)
    g14, c10 = 1.0 / pedal.c("R_clip"), pedal.c("C_clip")
    y = [
        [[c10 + c11, g14 + g16], [-g16], [-c11, 0.0], [0.0]],
        [[-g16], [c12, g16 + ga], [0.0], [-ga]],
        [[-c11, 0.0], [0.0], [c11, g17 + gb], [-gb]],
        [[0.0], [-ga], [-gb], [ga + gb + go]],
    ]
    zero = [0.0]
    t_open = nodal_transfer(y, [[g14], zero, zero, zero], out=0)
    z_port = nodal_transfer(y, [[1.0], zero, zero, zero], out=0)
    return t_open, z_port


@register_whitebox("ds1")
def build_ds1(
    pedal: Pedal,
    knobs: Mapping[str, float],
    sample_rate: int = 44100,
    oversample: int = 4,
    backend: str = "auto",
) -> PortClipper:
    clip = pedal.clipping[0]
    if clip.location != "shunt":
        raise ValueError(f"{pedal.id}: expected a shunt clipping stage, got {clip.location!r}")
    level, r_level_in = level_divider(pedal, knobs["level"])
    t_open, z_port = clip_port(pedal, knobs["tone"], r_level_in)
    pre = (
        rc_highpass(pedal.c("R_boost_bias"), pedal.c("C_boost_in2")),
        gain(-booster_gain(pedal)),  # common emitter: inverting
        rc_highpass(pedal.c("R_opamp_bias"), pedal.c("C_opamp_in")),
        opamp_stage(pedal, knobs["dist"]),
    )
    return PortClipper(
        diodes=DiodePair(diode(clip.part, clip.device), clip.n_pos, clip.n_neg),
        z_port=z_port,
        t_open=t_open,
        t_port_out=tone_network(pedal, knobs["tone"], r_level_in),
        pre=pre,
        out_gain=level,
        oversample=oversample,
        sample_rate=sample_rate,
        backend=backend,
        notes=NOTES,
    )


__all__ = [
    "NOTES",
    "booster_gain",
    "build_ds1",
    "clip_port",
    "level_divider",
    "opamp_stage",
    "tone_network",
]
