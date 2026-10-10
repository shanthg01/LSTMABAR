r"""Ibanez TS808 white-box model.

Chain (component names from ``pedals/ts808.yaml``; op-amps ideal, transistors ideal unity
followers):

1. Input coupling: ``C_clip_in`` into ``R_clip_bias`` (first-order high-pass, ≈16 Hz) at the
   clipping op-amp's (+) input. The Q1 emitter follower is an ideal unity buffer.
2. Clipping stage (feedback configuration of :mod:`.diode_clipper`): gain leg ``R_gain`` +
   ``C_gain``, feedback ``R_fb`` + drive-pot resistance ∥ ``C_fb`` ∥ the anti-parallel
   diodes of the first clipping entry (1S2473, :mod:`.devices`).
3. Active tone stage (IC1b), :func:`tone_stage`, exact for an ideal op-amp.
4. Level: ``R_level_in`` in series with the level pot (a divider to ground); the output
   buffer is an ideal unity follower.

**Tone stage transfer function.** The clipping op-amp's output (an ideal source ``v_in``)
drives ``R_tone_lp`` (R7) into node ``A`` = IC1b's (+) input, loaded by ``C_tone_lp`` (C5)
and ``R_tone_bias`` to AC ground. The tone pot (``P``, 20 k) runs from ``A`` to IC1b's (-)
input; its wiper ``W`` goes to ground through ``Z = R_tone_shunt + 1/(s C_tone_shunt)``.
``R_tone_fb`` (``R_f``) feeds the output back to (-). Let ``R_a`` be the pot resistance from
the (-) end to the wiper and ``R_b`` from the (+) end (``R_a + R_b = P``). Tone clockwise =
brighter = wiper toward (-): ``R_a = P (1 - f)``, ``R_b = P f`` with ``f`` the taper fraction.

With an ideal op-amp both pot ends sit at ``v_A``, so the pot's two halves are in parallel
into ``Z`` and carry currents ``I_a = v_A R_b sC / D`` (through ``R_a``, supplied by the
output through ``R_f``) and ``I_b = v_A R_a sC / D`` (through ``R_b``, drawn from ``A``),
where ``D(s) = sC (R_a R_b + P R_t) + P``. Hence ``v_out = v_A (1 + R_f I_a / v_A)`` and KCL
at ``A`` gives

.. math::

    H(s) = \frac{sC (R_a R_b + P R_t + R_f R_b) + P}
                {R_7 [s^2 C_5 C K + s (C_5 P + G_t C K + R_a C) + G_t P]},
    \quad K = R_a R_b + P R_t,\; G_t = 1/R_7 + 1/R_\text{bias}.

Checks: at ``f = 0`` (wiper at the (+) end) it is the R7/C5 low-pass loaded by the R8/C6
branch (a second treble-cut pole) with unity op-amp gain; at ``f = 1`` it is the R7/C5
low-pass times the high-shelf ``(1 + sC (R_t + R_f)) / (1 + sC R_t)`` (+14.8 dB above
≈3.3 kHz), which levels off the roll-off. DC gain ``R_bias / (R_7 + R_bias)`` (-0.83 dB).

Not modelled (documented approximations): the ≈1.6 Hz output coupling and the 510 k output
buffer bias loading the level wiper (≤0.1 dB), the buffers' own non-idealities, op-amp rails,
slew and bandwidth. Linear stages run at the oversampled rate (``oversample_linear``).
"""

from collections.abc import Mapping

from lstmabar.physics.kb import Pedal
from lstmabar.physics.whitebox.base import register_whitebox
from lstmabar.physics.whitebox.devices import SUBSTITUTES
from lstmabar.physics.whitebox.diode_clipper import DiodeClipper, feedback_clipper
from lstmabar.physics.whitebox.linear import AnalogFilter, rc_highpass

NOTES = (
    "op-amps ideal (no rails, slew or bandwidth limit); transistor buffers ideal unity",
    "output coupling (~1.6 Hz) and output-buffer loading of the level wiper not modelled",
)


def tone_stage(pedal: Pedal, tone: float) -> AnalogFilter:
    """TS808 active tone stage (R7/C5 into IC1b) at tone rotation ``tone``; see module doc."""
    pot = pedal.pots["tone"]
    p = pot.value
    f = pot.fraction(tone)
    ra, rb = p * (1.0 - f), p * f
    r7, c5 = pedal.c("R_tone_lp"), pedal.c("C_tone_lp")
    rt, c = pedal.c("R_tone_shunt"), pedal.c("C_tone_shunt")
    rf, rbias = pedal.c("R_tone_fb"), pedal.c("R_tone_bias")
    gt = 1.0 / r7 + 1.0 / rbias
    k = ra * rb + p * rt
    num = (c * (k + rf * rb), p)
    den = (r7 * c5 * c * k, r7 * (c5 * p + gt * c * k + ra * c), r7 * gt * p)
    return AnalogFilter(num, den)


def level_gain(pedal: Pedal, level: float) -> float:
    """Level pot divider fed through ``R_level_in``: ``f P / (P + R_level_in)``."""
    pot = pedal.pots["level"]
    return pot.fraction(level) * pot.value / (pot.value + pedal.c("R_level_in"))


@register_whitebox("ts808")
def build_ts808(
    pedal: Pedal,
    knobs: Mapping[str, float],
    sample_rate: int = 44100,
    oversample: int = 4,
    backend: str = "auto",
) -> DiodeClipper:
    clip = pedal.clipping[0]
    if clip.location != "feedback":
        raise ValueError(f"{pedal.id}: expected a feedback clipping stage, got {clip.location!r}")
    r_fb = pedal.c("R_fb") + pedal.pots["drive"].resistance(knobs["drive"])
    model = feedback_clipper(
        R_gain=pedal.c("R_gain"),
        C_gain=pedal.c("C_gain"),
        R_fb=r_fb,
        C_fb=pedal.c("C_fb"),
        part=clip.part,
        n_pos=clip.n_pos,
        n_neg=clip.n_neg,
        device=clip.device,
        pre=(rc_highpass(pedal.c("R_clip_bias"), pedal.c("C_clip_in")),),
        post=(tone_stage(pedal, knobs["tone"]),),
        out_gain=level_gain(pedal, knobs["level"]),
        oversample=oversample,
        sample_rate=sample_rate,
        backend=backend,
        oversample_linear=True,
    )
    notes = list(NOTES)
    if clip.part.upper() in SUBSTITUTES:
        notes.append(f"{clip.part} modelled with {SUBSTITUTES[clip.part.upper()]} parameters")
    model.notes = tuple(notes)
    return model


__all__ = ["NOTES", "build_ts808", "level_gain", "tone_stage"]
