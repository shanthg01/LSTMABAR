"""Circuit -> grey-box derivations: (pedal, knob rotations) -> physical ``Drive``/``EQ3`` knobs.

Contract (P2 decision 2):

- One deriver per pedal id, registered with :func:`register`. A deriver receives the
  :class:`~lstmabar.physics.kb.Pedal`, the full knob dict (rotations in [0, 1], defaults
  filled in) and a :class:`DeriveContext`, and returns a :class:`Preset` holding **physical**
  values for the board's ``drive`` and ``eq`` blocks (units as in their ``ParamSpec``).
- :func:`derive` validates the knobs, calls the deriver, clamps values to the ``ParamSpec``
  ranges and records every clamp in ``Preset.notes`` (e.g. RAT gain above 60 dB).
- :func:`preset_params` turns a preset into normalized ``BoardParams`` for
  :func:`~lstmabar.dsp.pedalboard.default_pedalboard`.

Registration: derivers live in this module (one ``@register("<pedal id>")`` function per
pedal), so importing :mod:`lstmabar.physics.derive` fills :data:`DERIVERS`.

Level calibration follows :mod:`lstmabar.physics.calibration`. Nonlinear stages use
closed-form formulas; linear tone stacks are fitted with :func:`~lstmabar.physics.analog.fit_tone`.
The waveshaper settings from :func:`shaper_prior` are priors, to be refined by the P2
fidelity fit.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import torch

from lstmabar.dsp.base import ParamSpec
from lstmabar.dsp.blocks import EQ3, Compressor, Drive
from lstmabar.dsp.pedalboard import ENABLED, BoardParams
from lstmabar.physics.analog import analog_response_db, fit_tone, log_grid
from lstmabar.physics.calibration import (
    VOLTS_PER_FULL_SCALE,
    clip_volts,
    db,
    drive_gain_db,
    output_level_db,
)
from lstmabar.physics.kb import ClipStage, Pedal

BLOCK_SPECS: dict[str, tuple[ParamSpec, ...]] = {
    "compressor": Compressor.param_specs,
    "drive": Drive.param_specs,
    "eq": EQ3.param_specs,
}


@dataclass(frozen=True)
class DeriveContext:
    volts_per_fs: float = VOLTS_PER_FULL_SCALE
    sample_rate: int = 44100

    @classmethod
    def from_config(cls, cfg) -> "DeriveContext":
        """From ``physics.volts_per_full_scale`` and ``audio.sample_rate`` of a run config."""
        return cls(
            volts_per_fs=float(cfg.physics.volts_per_full_scale),
            sample_rate=int(cfg.audio.sample_rate),
        )


@dataclass(frozen=True)
class Preset:
    """Grey-box settings for one pedal at one knob setting (physical units)."""

    pedal_id: str
    knobs: Mapping[str, float]
    board: Mapping[str, Mapping[str, float]]
    enabled: Mapping[str, bool] = field(
        default_factory=lambda: {"compressor": False, "drive": True, "eq": True}
    )
    info: Mapping[str, float] = field(default_factory=dict)  # derived circuit quantities
    notes: tuple[str, ...] = ()  # clamps, approximations, provisional parts


Deriver = Callable[[Pedal, dict[str, float], DeriveContext], Preset]
DERIVERS: dict[str, Deriver] = {}


def register(pedal_id: str) -> Callable[[Deriver], Deriver]:
    def deco(fn: Deriver) -> Deriver:
        if pedal_id in DERIVERS:
            raise ValueError(f"deriver for {pedal_id!r} already registered")
        DERIVERS[pedal_id] = fn
        return fn

    return deco


# --- Shared helpers -----------------------------------------------------------------------------


def shaper_prior(clip: ClipStage) -> dict[str, float]:
    """Waveshaper prior from the clipping stage (softness, asymmetry, bias).

    Feedback clipping is softer than shunt clipping, Ge knees are softer than Si, LEDs are
    sharper. Unequal diode counts lower one side's ceiling: ``asymmetry = 2·(1 - n_min/n_max)``
    (Drive's negative ceiling is ``1 - asymmetry/2``; the magnitude spectrum does not depend on
    which side clips first). Drive cannot go below a 2:1 ceiling ratio, so ratios beyond that
    (including single-sided clipping, ``n_min = 0``) saturate at ``asymmetry = 1``; derivers for
    such stages should also use ``bias``. Heuristic values; the fidelity fit replaces them.
    """
    softness = {"feedback": 0.2, "shunt": 0.5, "transistor": 0.3}[clip.location]
    softness += {"ge": -0.1, "ge_transistor": -0.1, "led": 0.1}.get(clip.device, 0.0)
    lo, hi = sorted((clip.n_pos, clip.n_neg))
    asymmetry = 2.0 * (1.0 - lo / hi) if hi else 0.0
    return {
        "softness": min(max(softness, 0.0), 1.0),
        "asymmetry": min(asymmetry, 1.0),
        "bias": 0.0,
    }


def stage_clip_volts(clip: ClipStage) -> float:
    """Clipping voltage of the side that clips last (the grey-box +1 ceiling).

    Uses the stage's ``v_clip`` override when given (required for MOSFET and transistor
    stages, where the clip level comes from the bias point, not a junction drop).
    """
    if clip.v_clip is not None:
        return clip.v_clip
    if clip.location == "transistor" or clip.device == "mosfet":
        raise KeyError(f"clipping stage {clip.stage!r}: set v_clip for {clip.device} stages")
    return clip_volts(clip.device, max(clip.n_pos, clip.n_neg))


def rc_hz(r: float, c: float) -> float:
    """First-order RC corner frequency 1 / (2πRC)."""
    return 1.0 / (2.0 * math.pi * r * c)


def _clamped(block: str, values: Mapping[str, float], notes: list[str]) -> dict[str, float]:
    specs = {s.name: s for s in BLOCK_SPECS[block]}
    out = {}
    for name, v in values.items():
        if name not in specs:
            raise KeyError(f"{block}: unknown knob {name!r}")
        s = specs[name]
        if math.isnan(float(v)):
            raise ValueError(f"{block}.{name}: deriver produced NaN")
        c = min(max(float(v), s.min), s.max)
        if c != v:
            notes.append(f"{block}.{name} {v:.4g} clamped to {c:.4g} {s.unit}".rstrip())
        out[name] = c
    return out


# --- Public API ---------------------------------------------------------------------------------


def derive(
    pedal: Pedal,
    knobs: Mapping[str, float] | None = None,
    ctx: DeriveContext | None = None,
) -> Preset:
    """Derive grey-box settings for ``pedal`` at ``knobs`` (missing knobs use pot defaults)."""
    if pedal.id not in DERIVERS:
        raise KeyError(f"no deriver registered for pedal {pedal.id!r}")
    full = pedal.knobs(knobs)
    preset = DERIVERS[pedal.id](pedal, full, ctx or DeriveContext())
    notes = list(preset.notes)
    board = {b: _clamped(b, vals, notes) for b, vals in preset.board.items()}
    return replace(preset, knobs=full, board=board, notes=tuple(notes))


def preset_params(preset: Preset, batch_size: int = 1) -> BoardParams:
    """Normalized ``BoardParams`` for ``default_pedalboard`` (all blocks, defaults filled)."""
    params: BoardParams = {}
    for block, specs in BLOCK_SPECS.items():
        vals = preset.board.get(block, {})
        p = {}
        for s in specs:
            v = torch.full((batch_size,), float(vals.get(s.name, s.default)), dtype=torch.float64)
            p[s.name] = s.normalize(v).to(torch.float32)
        p[ENABLED] = torch.full((batch_size,), 1.0 if preset.enabled.get(block, True) else 0.0)
        params[block] = p
    return params


# --- Pedals -------------------------------------------------------------------------------------


@register("ts808")
def _ts808(pedal: Pedal, knobs: dict[str, float], ctx: DeriveContext) -> Preset:
    """Ibanez TS808: non-inverting op-amp stage with anti-parallel Si diodes in feedback.

    Gain (above the R_gain/C_gain corner) = 1 + (R_fb + R_drive) / R_gain; the R_gain-C_gain
    leg makes the stage's gain path high-passed at ~720 Hz. C_fb across the feedback
    resistance low-passes the gain path. The unity clean path of feedback clipping is not
    representable by ``Drive`` (topology gap, P2 decision 4).
    """
    clip = pedal.clipping[0]
    r_drive = pedal.pots["drive"].resistance(knobs["drive"])
    r_fb = pedal.c("R_fb") + r_drive
    gain = 1.0 + r_fb / pedal.c("R_gain")
    v_clip = stage_clip_volts(clip)

    # Tone stage (PROVISIONAL): fixed post-clip RC low-pass only; the tone pot's active
    # treble network is added in P2 wave 2 from the cited analysis.
    f = log_grid()
    lp_hz = rc_hz(pedal.c("R_tone_lp"), pedal.c("C_tone_lp"))
    w = 2 * math.pi * lp_hz
    tone = fit_tone(f, analog_response_db([w], [1.0, w], f), ctx.sample_rate)

    level = pedal.pots["level"].fraction(knobs["level"])
    post_gain = level * 10 ** (tone.gain_db / 20.0)
    level_db = output_level_db(v_clip, post_gain, ctx.volts_per_fs) if post_gain > 0 else -math.inf

    return Preset(
        pedal_id=pedal.id,
        knobs=knobs,
        board={
            "drive": {
                "pre_hpf_hz": rc_hz(pedal.c("R_gain"), pedal.c("C_gain")),
                "gain_db": drive_gain_db(gain, v_clip, ctx.volts_per_fs),
                **shaper_prior(clip),
                **tone.drive_knobs(),
                "level_db": level_db,
            },
            "eq": tone.eq_knobs(),
        },
        info={
            "stage_gain_db": db(gain),
            "pre_hpf_hz": rc_hz(pedal.c("R_gain"), pedal.c("C_gain")),
            "feedback_lp_hz": rc_hz(r_fb, pedal.c("C_fb")),
            "tone_lp_hz": lp_hz,
            "tone_fit_rms_db": tone.rms_error_db,
            "v_clip": v_clip,
        },
        notes=(
            "tone knob not modelled yet (fixed post-clip low-pass only)",
            "level pot below ~0.24 reaches Drive's -36 dB floor",
            "feedback clipping's unity clean path is not represented by Drive",
        ),
    )


__all__ = [
    "BLOCK_SPECS",
    "DERIVERS",
    "DeriveContext",
    "Preset",
    "derive",
    "preset_params",
    "rc_hz",
    "register",
    "shaper_prior",
    "stage_clip_volts",
]
