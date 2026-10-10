"""Ibanez TS808 white-box model.

Chain (component names from ``pedals/ts808.yaml``):

1. Clipping stage (feedback configuration of :mod:`.diode_clipper`): gain leg
   ``R_gain`` + ``C_gain``, feedback ``R_fb`` + drive-pot resistance ∥ ``C_fb`` ∥ the
   anti-parallel diodes of the first clipping entry (1S2473, :mod:`.devices`).
2. Post-clip passive low-pass ``R_tone_lp`` / ``C_tone_lp`` (≈723 Hz).
3. Level pot as a divider (taper fraction).

PROVISIONAL, like the TS808 deriver in :mod:`lstmabar.physics.derive`: the tone pot's active
treble network, the input/output buffers and the coupling caps are not modelled yet (the
tone knob has no effect); they are added once the tone-stage components land in the YAML.
The post-clip low-pass is treated as unloaded.
"""

from collections.abc import Mapping

from lstmabar.physics.kb import Pedal
from lstmabar.physics.whitebox.base import register_whitebox
from lstmabar.physics.whitebox.devices import SUBSTITUTES
from lstmabar.physics.whitebox.diode_clipper import DiodeClipper, feedback_clipper
from lstmabar.physics.whitebox.linear import rc_lowpass

NOTES = (
    "tone knob not modelled yet (fixed post-clip RC low-pass only)",
    "input/output buffers and coupling caps not modelled",
    "op-amp ideal (no rails, slew or bandwidth limit)",
)


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
        post=(rc_lowpass(pedal.c("R_tone_lp"), pedal.c("C_tone_lp")),),
        out_gain=pedal.pots["level"].fraction(knobs["level"]),
        oversample=oversample,
        sample_rate=sample_rate,
        backend=backend,
    )
    notes = list(NOTES)
    if clip.part.upper() in SUBSTITUTES:
        notes.append(f"{clip.part} modelled with {SUBSTITUTES[clip.part.upper()]} parameters")
    model.notes = tuple(notes)
    return model


__all__ = ["NOTES", "build_ts808"]
