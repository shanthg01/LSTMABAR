r"""Boss DS-1 white-box model (component names from ``pedals/ds1.yaml``).

Chain (all linear stages at the oversampled rate):

1. **Transistor booster (Q2), linear**: the shunt-feedback small-signal model the grey-box
   derivation uses, :func:`lstmabar.physics.derive.ds1_booster` (C2/R4 leg and C3 into the
   collector-base-feedback stage R7 ∥ C4, open-loop gain from the 5.8 V collector bias), so
   white-box and derived grey-box share one linear model. It rises ~6 dB/oct to a corner
   near 520 Hz (C3 into the Miller-lowered base impedance) and levels off at ~35.8 dB,
   matching ElectroSmash's "35 dB". Inverting. **The booster's soft asymmetric clipping is
   ignored (ideal, unbounded transistor)**; P2 decision 4 accepts this cascaded-stage gap.
2. **Op-amp gain stage**, ideal non-inverting: input coupling ``C_opamp_in`` /
   ``R_opamp_bias`` (C5/R10, 23 Hz), gain ``1 + Z_f / Z_g``
   (:func:`lstmabar.physics.networks.noninverting_gain`) with ``Z_f = R_dist ∥ C_fb``
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
   The output transfer ``V_W / V_X`` is the derivation's
   :func:`lstmabar.physics.derive.lp_hp_blend_tone_stack` with this model's load; the
   port quantities (``T_open``, ``Z_port``) come from a nodal solve of the same network
   (:func:`clip_port`; tests check the two agree).
4. **Level.** Per Premier Guitar's signal-path description, the level pot (VR2) follows the
   tone stage and its wiper drives ``R_tone_out`` (R18, series) into the output buffer,
   whose bias ``R_out_bias_a ∥ R_out_bias_b`` (500 kΩ) is the load: the pot is a divider
   with its lower leg ∥ (R18 + 500 kΩ), and the pot's input resistance (~100 kΩ) loads W.
   **This differs from the derivation**, which loads the tone stack with R18 + C13 +
   500 kΩ and applies the level pot as an unloaded fraction afterwards (its notes list
   "Level-pot loading of the tone stack not modelled"); the white-box follows the cited
   signal path. At level 0.5 the two differ by 0.2-0.8 dB (white-box lower, nearly
   broadband), so the effect on the derived error is small.

Not modelled (documented): coupling high-passes below 10 Hz other than the booster's own
(C1/R2 7.2 Hz, C13/R20 3.4 Hz, C14/R23 1.6 Hz: < 0.1 dB at 82 Hz), C9 (the series
coupling ahead of R14, whose placement is inferred and which only forms a ≈2% capacitive
divider with C10), the JFET switching, buffer non-idealities, op-amp rails/slew/bandwidth. With ideal stages the
booster + op-amp gain (up to ≈62 dB) is unbounded, so the model reaches the diodes with far
more than the supply could deliver; the diode node is still bounded near ±0.7 V.
"""

from collections.abc import Mapping

from lstmabar.physics.derive import ds1_booster, lp_hp_blend_tone_stack
from lstmabar.physics.kb import Pedal
from lstmabar.physics.networks import POT_END_OHMS, cap, noninverting_gain, par, pot_split, res
from lstmabar.physics.whitebox.base import register_whitebox
from lstmabar.physics.whitebox.devices import diode
from lstmabar.physics.whitebox.diode_clipper import DiodePair
from lstmabar.physics.whitebox.linear import (
    AnalogFilter,
    from_tf,
    gain,
    nodal_transfer,
    rc_highpass,
)
from lstmabar.physics.whitebox.port import PortClipper

NOTES = (
    "transistor booster linear (its soft asymmetric clipping is ignored)",
    "op-amps ideal (no rails, slew or bandwidth limit); buffers ideal unity",
    "level pot loads the tone stack (cited signal path; the derivation does not)",
    "coupling high-passes below 10 Hz outside the booster, and C9, not modelled",
)


def _par(*rs: float) -> float:
    return 1.0 / sum(1.0 / r for r in rs)


def booster(pedal: Pedal) -> AnalogFilter:
    """Q2 booster magnitude response (:func:`derive.ds1_booster`); the stage inverts."""
    return from_tf(ds1_booster(pedal))


def opamp_stage(pedal: Pedal, dist: float) -> AnalogFilter:
    """Non-inverting gain ``1 + Z_f/Z_g`` (VR1 ∥ C7 over R13 + C8) at Dist rotation ``dist``."""
    rd = max(pedal.pots["dist"].resistance(dist), POT_END_OHMS)
    z_f = par(res(rd), cap(pedal.c("C_fb")))
    return from_tf(noninverting_gain(z_f, res(pedal.c("R_gain")) + cap(pedal.c("C_gain"))))


def level_divider(pedal: Pedal, level: float) -> tuple[float, float]:
    """``(gain, input resistance)`` of the level pot loaded by R18 + the output-buffer bias."""
    pot = pedal.pots["level"]
    f = pot.fraction(level)
    r_load = pedal.c("R_tone_out") + _par(pedal.c("R_out_bias_a"), pedal.c("R_out_bias_b"))
    lower = _par(max(f * pot.value, POT_END_OHMS), r_load) if f > 0 else 0.0
    r_in = (1.0 - f) * pot.value + lower
    return lower / r_in, r_in


def _tone_parts(pedal: Pedal, tone: float, r_level_in: float):
    r_lw, r_wh = pot_split(pedal.pots["tone"], tone)  # L-W, W-H (as the derivation)
    g16, c12 = 1.0 / pedal.c("R_tone_lp"), pedal.c("C_tone_lp")
    c11, g17 = pedal.c("C_tone_hp"), 1.0 / pedal.c("R_tone_hp")
    return 1.0 / r_lw, 1.0 / r_wh, g16, c12, c11, g17, 1.0 / r_level_in


def tone_network(pedal: Pedal, tone: float, r_level_in: float) -> AnalogFilter:
    """Tone stack driven by an ideal source at the clipping node, ``V_W / V_X``, from the
    derivation's :func:`~lstmabar.physics.derive.lp_hp_blend_tone_stack` loaded by the level
    pot's input resistance ``r_level_in``."""
    tf = lp_hp_blend_tone_stack(
        pedal, tone, "R_tone_lp", "C_tone_lp", "C_tone_hp", "R_tone_hp", res(r_level_in)
    )
    return from_tf(tf)


def tone_network_nodal(pedal: Pedal, tone: float, r_level_in: float) -> AnalogFilter:
    """The same ``V_W / V_X`` from the nodal matrix used by :func:`clip_port` (a check)."""
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
        booster(pedal),
        gain(-1.0),  # common emitter: inverting
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
    "booster",
    "build_ds1",
    "clip_port",
    "level_divider",
    "opamp_stage",
    "tone_network",
    "tone_network_nodal",
]
