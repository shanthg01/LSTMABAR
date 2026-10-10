"""Ibanez TS808 white-box model.

Chain (component names from ``pedals/ts808.yaml``; op-amps ideal, transistors ideal unity
followers):

1. Input coupling: ``C_clip_in`` into ``R_clip_bias`` (first-order high-pass, ≈16 Hz) at the
   clipping op-amp's (+) input. The Q1 emitter follower is an ideal unity buffer.
2. Clipping stage (feedback configuration of :mod:`.diode_clipper`): gain leg ``R_gain`` +
   ``C_gain``, feedback ``R_fb`` + drive-pot resistance ∥ ``C_fb`` ∥ the anti-parallel
   diodes of the first clipping entry (1S2473, :mod:`.devices`).
3. Active tone stage (IC1b, ideal op-amp): the same transfer function the grey-box
   derivation uses, :func:`lstmabar.physics.derive.ts808_tone_stage` (R7/C5 low-pass into
   IC1b, tone pot between its inputs, R8/C6 wiper network, R_tone_fb feedback; derivation
   in its docstring), so white-box and derived grey-box share one linear model. Checked
   against an independent finite-gain nodal solve in the tests.
4. Level: ``R_level_in`` in series with the level pot (a divider to ground), then the
   ``R_out_series`` / ``R_out_shunt`` output divider; the output buffer is an ideal unity
   follower.

Not modelled (documented approximations): the ≈1.6 Hz output coupling, the 510 kΩ output
buffer bias loading the level wiper (≤0.1 dB), buffer non-idealities, op-amp rails, slew and
bandwidth. All linear stages run at the oversampled rate (``oversample_linear``).
"""

from collections.abc import Mapping

from lstmabar.physics.derive import ts808_tone_stage
from lstmabar.physics.kb import Pedal
from lstmabar.physics.whitebox.base import register_whitebox
from lstmabar.physics.whitebox.devices import SUBSTITUTES
from lstmabar.physics.whitebox.diode_clipper import DiodeClipper, feedback_clipper
from lstmabar.physics.whitebox.linear import AnalogFilter, from_tf, rc_highpass

NOTES = (
    "op-amps ideal (no rails, slew or bandwidth limit); transistor buffers ideal unity",
    "output coupling (~1.6 Hz) and output-buffer loading of the level wiper not modelled",
)


def tone_stage(pedal: Pedal, tone: float) -> AnalogFilter:
    """TS808 active tone stage at rotation ``tone`` (:func:`derive.ts808_tone_stage`)."""
    return from_tf(ts808_tone_stage(pedal, tone))


def level_gain(pedal: Pedal, level: float) -> float:
    """Level pot fed through ``R_level_in``, then the ``R_out_series``/``R_out_shunt``
    output divider (as in the derivation)."""
    pot = pedal.pots["level"]
    out = pedal.c("R_out_shunt") / (pedal.c("R_out_shunt") + pedal.c("R_out_series"))
    return pot.fraction(level) * pot.value / (pot.value + pedal.c("R_level_in")) * out


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
