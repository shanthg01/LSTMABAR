"""Circuit -> grey-box derivations: (pedal, knob rotations) -> physical ``Drive``/``EQ3`` knobs.

Contract (P2 decision 2):

- One deriver per pedal id, registered with :func:`register`. A deriver receives the
  :class:`~lstmabar.physics.kb.Pedal`, the full knob dict (rotations in [0, 1], defaults
  filled in) and a :class:`DeriveContext`, and returns a :class:`Preset` holding **physical**
  values for the board's ``drive`` and ``eq`` blocks (units as in their ``ParamSpec``).
- :func:`derive` validates the knobs, calls the deriver, clamps values to the ``ParamSpec``
  ranges and records every clamp in ``Preset.notes`` with what it means (e.g. RAT gain above
  60 dB).
- :func:`preset_params` turns a preset into normalized ``BoardParams`` for
  :func:`~lstmabar.dsp.pedalboard.default_pedalboard`.

Registration: derivers live in this module (one ``@register("<pedal id>")`` function per
pedal), so importing :mod:`lstmabar.physics.derive` fills :data:`DERIVERS`.

Method, shared by every deriver (helpers in :mod:`lstmabar.physics.networks`):

1. **Pre-clip gain path** as an s-domain transfer function from component values, in two
   versions: ``h_full`` (every element) and ``h_hp`` (the pre-clip low-pass elements, e.g.
   feedback caps and Miller caps, removed). ``Drive``'s pre-HPF + flat gain stands in for
   ``h_hp`` (:func:`pre_clip_drive`): ``gain = |h_hp(f_ref)|`` and ``pre_hpf_hz`` is the
   Butterworth corner that best matches ``h_hp``'s roll-off below ``f_ref``. Calibrated with
   :func:`~lstmabar.physics.calibration.drive_gain_db` at the clipping stage's ``V_clip``.
2. **Post-clip linear response** ``h_post`` = (``h_full / h_hp``: the pre-clip low-pass
   ``Drive`` cannot place before the shaper, moved after it; topology gap of decision 4) ×
   tone stack × output filtering, fitted with :func:`~lstmabar.physics.analog.fit_tone` onto
   ``Drive.tone_db`` + ``EQ3``. The fit's broadband gain is folded into ``drive.level_db``.
3. **Level** :func:`~lstmabar.physics.calibration.output_level_db`: the shaper's ±1 is
   ``V_clip`` volts, times the post-clip gain (fit gain, fixed stage gains, volume pot).
4. **Shaper** :func:`shaper_prior` from the clipping device (a prior for the P2 fidelity fit).

Cascaded clipping stages (DS-1 booster + op-amp, Big Muff's two clippers) map onto one
``Drive``: the gains up to the last (dominant) clipping stage are multiplied, so the grey-box
clips once where the circuit clips two or more times. Each deriver's docstring gives its
formulas; ``Preset.info`` records the circuit quantities (corners, stage gains, the gain
response at a few frequencies, the tone-fit error) and ``Preset.notes`` the approximations.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from lstmabar.dsp.base import ParamSpec
from lstmabar.dsp.blocks import EQ3, Compressor, Drive
from lstmabar.dsp.pedalboard import ENABLED, BoardParams
from lstmabar.physics.analog import ToneFit, fit_tone, log_grid
from lstmabar.physics.calibration import (
    VOLTS_PER_FULL_SCALE,
    clip_volts,
    db,
    drive_gain_db,
    output_level_db,
)
from lstmabar.physics.kb import ClipStage, Pedal
from lstmabar.physics.networks import (
    TF,
    THERMAL_VOLTAGE,
    cap,
    ce_collector_feedback_bias,
    ce_open_loop_gain,
    divider,
    noninverting_gain,
    par,
    pot_split,
    rc_highpass,
    rc_hz,
    rc_lowpass,
    res,
    response_info,
    shunt_feedback_gain,
    thevenin_cr_highpass,
    thevenin_rc_lowpass,
    two_leg_blend,
)

BLOCK_SPECS: dict[str, tuple[ParamSpec, ...]] = {
    "compressor": Compressor.param_specs,
    "drive": Drive.param_specs,
    "eq": EQ3.param_specs,
}

GAIN_REF_HZ = 1000.0
"""Default frequency at which a frequency-dependent pre-clip gain is read off (``f_ref``).

Guitar fundamentals and the strongest low harmonics sit below ~1 kHz, and above ~1.5 kHz
the op-amp bandwidth (LM308 in the RAT) and Miller caps (Big Muff) pull the real gain down,
so 1 kHz is the representative mid-band point. It is also ``Drive``'s tilt pivot."""

SOURCE_OHMS = 10e3
"""Default guitar source impedance (ohms) for pedals whose gain depends on it (Fuzz Face).
Order of a passive pickup's DC resistance at full volume; the real source is inductive and
rises with frequency. Override via ``physics.source_ohms``."""


@dataclass(frozen=True)
class DeriveContext:
    volts_per_fs: float = VOLTS_PER_FULL_SCALE
    sample_rate: int = 44100
    source_ohms: float = SOURCE_OHMS

    @classmethod
    def from_config(cls, cfg) -> "DeriveContext":
        """From ``physics.volts_per_full_scale`` (+ optional ``physics.source_ohms``) and
        ``audio.sample_rate`` of a run config."""
        return cls(
            volts_per_fs=float(cfg.physics.volts_per_full_scale),
            sample_rate=int(cfg.audio.sample_rate),
            source_ohms=float(cfg.physics.get("source_ohms", SOURCE_OHMS)),
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
    notes: tuple[str, ...] = ()  # clamps, approximations, topology gaps


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


def _butterworth_hp_db(f: np.ndarray, fc: float) -> np.ndarray:
    """Magnitude (dB) of a 2nd-order Butterworth high-pass (Drive's pre-HPF, analog form)."""
    r = (np.asarray(f, float) / fc) ** 4
    return 10.0 * np.log10(r / (1.0 + r))


@dataclass(frozen=True)
class PreClip:
    """``Drive``'s view of a pre-clip gain path: flat ``gain`` (linear) after an HPF."""

    gain: float
    pre_hpf_hz: float
    fit_rms_db: float


def pre_clip_drive(h_hp: TF, f_ref: float = GAIN_REF_HZ, f_lo: float = 40.0) -> PreClip:
    """Map a pre-clip gain path (low-pass elements removed) onto ``Drive``'s HPF + gain.

    ``gain = |h_hp(f_ref)|``; ``pre_hpf_hz`` minimizes the RMS dB error between Drive's
    2nd-order Butterworth high-pass and ``|h_hp(f)| / gain`` on a log grid over
    ``[f_lo, f_ref]`` (the band below the reference, where the circuit's coupling caps and
    gain legs roll the gain off). A one-parameter fit rather than the -3 dB point, because
    networks such as the RAT's two gain legs roll off at ~6 dB/oct over a decade, which a
    12 dB/oct filter at the -3 dB point would over-cut.
    """
    gain = float(h_hp.mag(f_ref))
    f = np.geomspace(f_lo, f_ref, 48)
    target = h_hp.db(f) - db(gain)
    spec = next(s for s in Drive.param_specs if s.name == "pre_hpf_hz")

    def err(log_fc: float) -> float:
        return float(np.mean((_butterworth_hp_db(f, 10**log_fc) - target) ** 2))

    lo, hi = math.log10(spec.min * 0.5), math.log10(spec.max * 2.0)
    r = minimize_scalar(err, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
    return PreClip(gain, float(10**r.x), math.sqrt(err(r.x)))


def _post_clip(
    pedal: Pedal,
    knobs: dict[str, float],
    ctx: DeriveContext,
    *,
    clip: ClipStage,
    pre: PreClip,
    h_post: TF,
    post_gain: float,
    shaper: Mapping[str, float] | None = None,
    info: Mapping[str, float],
    notes: tuple[str, ...],
) -> Preset:
    """Assemble a preset: fit ``h_post`` (dB on :func:`log_grid`) with tilt + EQ3, fold the
    fit gain and the scalar ``post_gain`` (volume pot, fixed stage gains) into the level."""
    v_clip = stage_clip_volts(clip)
    f = log_grid()
    tone: ToneFit = fit_tone(f, h_post.db(f), ctx.sample_rate)
    g = post_gain * 10 ** (tone.gain_db / 20.0)
    level_db = output_level_db(v_clip, g, ctx.volts_per_fs) if g > 0 else -math.inf
    pinned = [k for k in tone.at_bounds if k != "gain_db"]
    extra = (f"tone fit range-limited: {', '.join(pinned)} at a range limit",) if pinned else ()
    return Preset(
        pedal_id=pedal.id,
        knobs=knobs,
        board={
            "drive": {
                "pre_hpf_hz": pre.pre_hpf_hz,
                "gain_db": drive_gain_db(pre.gain, v_clip, ctx.volts_per_fs),
                **(shaper if shaper is not None else shaper_prior(clip)),
                **tone.drive_knobs(),
                "level_db": level_db,
            },
            "eq": tone.eq_knobs(),
        },
        info={
            **info,
            "drive_stage_gain_db": db(pre.gain),
            "pre_hpf_fit_rms_db": pre.fit_rms_db,
            "v_clip": v_clip,
            "post_gain_db": db(post_gain),
            "tone_fit_gain_db": tone.gain_db,
            "tone_fit_rms_db": tone.rms_error_db,
        },
        notes=notes + extra,
    )


_CLAMP_REASONS = {
    ("drive", "gain_db", "max"): (
        "circuit gain into the clipper exceeds Drive's range (P2 decision 4); the shaper is "
        "already near-square there, so the excess mainly changes sustain and noise"
    ),
    ("drive", "gain_db", "min"): "stage gain below Drive's range (pedal barely clips)",
    ("drive", "level_db", "min"): "output below Drive's floor (volume knob near off)",
    ("drive", "level_db", "max"): "output louder than Drive's ceiling",
    ("drive", "pre_hpf_hz", "min"): "pre-clip high-pass corner below Drive's range (inaudible)",
    ("drive", "pre_hpf_hz", "max"): "pre-clip high-pass corner above Drive's range",
}


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
            side = "max" if c == s.max else "min"
            reason = _CLAMP_REASONS.get((block, name, side), "outside the block's range")
            what = f"{block}.{name} {v:.4g} clamped to {c:.4g} {s.unit}".rstrip()
            notes.append(f"{what}: {reason}")
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


# --- Linear networks (public so white-box models can reuse them) -------------------------------


def ts808_tone_stage(pedal: Pedal, tone: float) -> TF:
    """TS808 tone stage, clipper output -> IC1b output (ideal op-amp; Geofex / Keen topology).

    ``R_tone_lp``/``C_tone_lp`` (R7/C5) feed node P = IC1b's (+) input, which also has
    ``R_tone_bias`` to AC ground. The Tone pot runs from P (rotation 0) to the (-) input N
    (rotation 1); its wiper W goes to ground through ``R_tone_shunt + 1/(s·C_tone_shunt)``
    (admittance ``Y_s``). ``R_tone_fb`` runs from the output to N. With ``R_a`` = P..W,
    ``R_b`` = W..N (``G = 1/R``) and ``v_N = v_P = v`` (ideal op-amp):

        node W:  v_W = v·(G_a + G_b) / Y_w,             Y_w = G_a + G_b + Y_s
        node N:  v_out = v·(1 + R_tone_fb · G_b·Y_s / Y_w)
        node P:  v / v_in = G_7 / (G_7 + s·C_tone_lp + G_bias + G_a·Y_s / Y_w)

    Tone at 0 (wiper at P): ``Y_s`` shunts P, a second low-pass pole (treble cut). Tone at 1
    (wiper at N): the output stage becomes ``1 + R_fb·Y_s``, a treble shelf between
    ``1/(2π·C6·(R_fb + R8))`` (~590 Hz) and ``1/(2π·C6·R8)`` (~3.3 kHz) that levels off the
    723 Hz R7/C5 roll-off.
    """
    r_a, r_b = pot_split(pedal.pots["tone"], tone)
    g_a, g_b = 1.0 / r_a, 1.0 / r_b
    y_s = 1 / (res(pedal.c("R_tone_shunt")) + cap(pedal.c("C_tone_shunt")))
    y_w = g_a + g_b + y_s
    g7 = 1.0 / pedal.c("R_tone_lp")
    v_p = g7 / (g7 + 1 / cap(pedal.c("C_tone_lp")) + 1.0 / pedal.c("R_tone_bias") + g_a * y_s / y_w)
    return v_p * (1 + pedal.c("R_tone_fb") * g_b * y_s / y_w)


def lp_hp_blend_tone_stack(
    pedal: Pedal, tone: float, r_lp: str, c_lp: str, c_hp: str, r_hp: str, z_load: TF
) -> TF:
    """Passive LP/HP blend (Big Muff / DS-1), ideal source -> Tone pot wiper.

    LP leg: series ``R_lp``, shunt ``C_lp``; HP leg: series ``C_hp``, shunt ``R_hp``. As
    Thevenin sources: ``V_L = 1/(1+sR_lpC_lp)``, ``Z_L = R_lp ∥ 1/(sC_lp)``;
    ``V_H = sR_hpC_hp/(1+sR_hpC_hp)``, ``Z_H = R_hp ∥ 1/(sC_hp)``. The Tone pot joins the LP
    node (rotation 0) and the HP node (rotation 1); with sections ``R_LW``/``R_WH`` the wiper
    voltage, loaded by ``z_load``, is (Millman)

        v_W = (V_L/(Z_L+R_LW) + V_H/(Z_H+R_WH)) / (1/(Z_L+R_LW) + 1/(Z_H+R_WH) + 1/z_load)

    exact for the loaded network (pot and next-stage loading included).
    """
    v_l, z_l = thevenin_rc_lowpass(pedal.c(r_lp), pedal.c(c_lp))
    v_h, z_h = thevenin_cr_highpass(pedal.c(c_hp), pedal.c(r_hp))
    r_lw, r_wh = pot_split(pedal.pots["tone"], tone)
    return two_leg_blend(v_l, z_l + r_lw, v_h, z_h + r_wh, z_load)


# --- Pedals -------------------------------------------------------------------------------------


@register("ts808")
def _ts808(pedal: Pedal, knobs: dict[str, float], ctx: DeriveContext) -> Preset:
    """Ibanez TS808: non-inverting op-amp stage with anti-parallel Si diodes in feedback.

    Gain path ``G(s) = 1 + Z_f/Z_g``, ``Z_f = (R_fb + R_drive) ∥ 1/(sC_fb)``,
    ``Z_g = R_gain + 1/(sC_gain)``. Without C_fb it is a first-order shelf whose plateau,
    ``1 + (R_fb + R_drive)/R_gain``, starts at the R_gain/C_gain corner (~720 Hz). So
    ``Drive`` takes the plateau gain and the textbook corner directly (``f_ref`` would sit on
    the shelf's slope at 1 kHz). C_fb's low-pass (``1/(2π(R_fb+R_drive)C_fb)``, ~5.7 kHz at
    full drive) goes into the post-clip fit, followed by :func:`ts808_tone_stage` and the
    level pot (``R_level_in`` in series with the pot, ``R_out_series``/``R_out_shunt`` output
    divider). The unity clean path of feedback clipping is not representable by ``Drive``
    (topology gap, P2 decision 4).
    """
    clip = pedal.clipping[0]
    r_f = pedal.c("R_fb") + pedal.pots["drive"].resistance(knobs["drive"])
    z_g = res(pedal.c("R_gain")) + cap(pedal.c("C_gain"))
    h_in = rc_highpass(pedal.c("R_clip_bias"), pedal.c("C_clip_in"))
    h_hp = h_in * noninverting_gain(res(r_f), z_g)
    h_full = h_in * noninverting_gain(par(res(r_f), cap(pedal.c("C_fb"))), z_g)
    gain = 1.0 + r_f / pedal.c("R_gain")
    hpf = rc_hz(pedal.c("R_gain"), pedal.c("C_gain"))
    pre = PreClip(gain, hpf, float("nan"))

    tone = ts808_tone_stage(pedal, knobs["tone"])
    level_pot = pedal.pots["level"]
    post_gain = (
        level_pot.fraction(knobs["level"])
        * level_pot.value
        / (level_pot.value + pedal.c("R_level_in"))
        * pedal.c("R_out_shunt")
        / (pedal.c("R_out_shunt") + pedal.c("R_out_series"))
    )
    f = log_grid()
    return _post_clip(
        pedal,
        knobs,
        ctx,
        clip=clip,
        pre=pre,
        h_post=(h_full / h_hp) * tone,
        post_gain=post_gain,
        info={
            "stage_gain_db": db(gain),
            "pre_hpf_hz": hpf,
            "feedback_lp_hz": rc_hz(r_f, pedal.c("C_fb")),
            "tone_lp_hz": rc_hz(pedal.c("R_tone_lp"), pedal.c("C_tone_lp")),
            "tone_shunt_hz": rc_hz(pedal.c("R_tone_shunt"), pedal.c("C_tone_shunt")),
            "tone_hf_db": float(np.mean(tone.db(f[f > 2000]))),
        },
        notes=(
            "feedback clipping's unity clean path is not represented by Drive",
            "C_fb's pre-clip low-pass is applied after the shaper (Drive has no pre-clip LP)",
        ),
    )


@register("rat")
def _rat(pedal: Pedal, knobs: dict[str, float], ctx: DeriveContext) -> Preset:
    """ProCo RAT: LM308 non-inverting stage, then 1N914 diodes to ground, passive filter.

    Gain path ``G(s) = 1 + Z_f/Z_g`` with ``Z_f = R_dist ∥ 1/(sC_fb)`` and two gain legs
    ``Z_g = (R_gain_hf + 1/(sC_gain_hf)) ∥ (R_gain_lf + 1/(sC_gain_lf))``: corners
    ``1/(2πR_gain_lf·C_gain_lf)`` = 60 Hz and ``1/(2πR_gain_hf·C_gain_hf)`` = 1539 Hz; the
    gain rises ~6 dB/oct between them from ``1 + R_dist/R_gain_lf`` (45 dB at max) to the HF
    ceiling ``1 + R_dist/(R_gain_hf ∥ R_gain_lf)`` (67 dB). Not applied flat (decision 4):
    ``Drive``'s gain is ``|G(j2π·1 kHz)|`` (~62 dB at max, then clamped to 60 dB) and its HPF is
    fitted to the roll-off below 1 kHz (:func:`pre_clip_drive`). Input coupling
    ``C_in``/``R_in_bias`` (7.2 Hz) included. C_fb's low-pass goes into the post-clip fit.

    Post-clip: the diode node (an ideal source while clipping, so R_clip is ignored) drives
    ``R_filter + R_filter_pot`` into ``C_filter`` (sweep ``1/(2π(R_filter + R_pot)C_filter)``,
    32 kHz .. 475 Hz), loaded by the JFET buffer's ``C_out_in + R_jfet_bias``. JFET follower
    unity, Volume pot a divider after ``C_out``.
    """
    clip = pedal.clipping[0]
    r_dist = max(pedal.pots["distortion"].resistance(knobs["distortion"]), 1.0)
    z_g = par(
        res(pedal.c("R_gain_hf")) + cap(pedal.c("C_gain_hf")),
        res(pedal.c("R_gain_lf")) + cap(pedal.c("C_gain_lf")),
    )
    h_in = rc_highpass(pedal.c("R_in_bias"), pedal.c("C_in"))
    h_hp = h_in * noninverting_gain(res(r_dist), z_g)
    h_full = h_in * noninverting_gain(par(res(r_dist), cap(pedal.c("C_fb"))), z_g)
    pre = pre_clip_drive(h_hp)

    r_filter = pedal.c("R_filter") + pedal.pots["filter"].resistance(knobs["filter"])
    z_load = par(cap(pedal.c("C_filter")), cap(pedal.c("C_out_in")) + res(pedal.c("R_jfet_bias")))
    h_filter = divider(res(r_filter), z_load)
    vol = pedal.pots["volume"]
    h_out = rc_highpass(vol.value, pedal.c("C_out"))
    return _post_clip(
        pedal,
        knobs,
        ctx,
        clip=clip,
        pre=pre,
        h_post=(h_full / h_hp) * h_filter * h_out,
        post_gain=vol.fraction(knobs["volume"]),
        info={
            "gain_hf_ceiling_db": db(
                1 + r_dist * (1 / pedal.c("R_gain_hf") + 1 / pedal.c("R_gain_lf"))
            ),
            "gain_lf_leg_db": db(1 + r_dist / pedal.c("R_gain_lf")),
            "gain_lf_corner_hz": rc_hz(pedal.c("R_gain_lf"), pedal.c("C_gain_lf")),
            "gain_hf_corner_hz": rc_hz(pedal.c("R_gain_hf"), pedal.c("C_gain_hf")),
            "filter_lp_hz": rc_hz(r_filter, pedal.c("C_filter")),
            "feedback_lp_hz": rc_hz(r_dist, pedal.c("C_fb")),
            **response_info("pre_gain", h_full),
        },
        notes=(
            f"gain read at {GAIN_REF_HZ:g} Hz of the frequency-dependent gain legs (HF ceiling "
            "and LF leg in info); HPF fitted to the roll-off below it",
            "LM308 gain-bandwidth and slew limits not modelled (pre-clip low-pass gap)",
        ),
    )


@register("ds1")
def _ds1(pedal: Pedal, knobs: dict[str, float], ctx: DeriveContext) -> Preset:
    """Boss DS-1: transistor booster -> op-amp gain stage -> Si diodes to ground -> LP/HP tone.

    Booster (Q2, collector-base feedback ``Z_f = R_boost_fb ∥ 1/(sC_boost_fb)``), driven by the
    input buffer through ``C_boost_in1`` (shunt ``R_boost_in1``) and ``C_boost_in2``:
    ``Z_s = (R_boost_in1 ∥ 1/(sC_boost_in1)) + 1/(sC_boost_in2)``, gain from
    :func:`~lstmabar.physics.networks.shunt_feedback_gain` with base bias ``R_boost_bias`` and
    open-loop gain ``R_boost_c / (R_boost_e + V_T/I_C)``,
    ``I_C = (V_supply - V_boost_collector)/R_boost_c``. Op-amp: input coupling
    ``C_opamp_in``/``R_opamp_bias`` (23 Hz),
    ``G(s) = 1 + (R_dist ∥ 1/(sC_fb)) / (R_gain + 1/(sC_gain))`` (72 Hz corner; 1 .. 22.3 =
    26.5 dB). The two gains are multiplied up to the diodes (cascade -> one Drive) and read at
    1 kHz (the booster's gain still rises above it).

    The booster's input coupling cap meets the base impedance that the shunt feedback lowers
    to ~``R_boost_fb / (1 + A_ol)`` (~5 kΩ), not ``R_boost_bias`` alone, so the pre-clip
    high-pass sits near 250-300 Hz rather than ElectroSmash's 33 Hz (C3 with R5).

    Post-clip: C_boost_fb/C_fb low-passes, the ``R_clip``/``C_clip`` 7.2 kHz low-pass at the
    diode node (pre-clip in the circuit), then :func:`lp_hp_blend_tone_stack` (234 Hz LP leg,
    1063 Hz HP leg) loaded by ``R_tone_out + 1/(sC_out_in) + R_out_bias_a ∥ R_out_bias_b``
    (the output buffer's input), the voltage across the bias resistors, and the Level pot.
    """
    clip = pedal.clipping[0]
    i_c = (pedal.c("V_supply") - pedal.c("V_boost_collector")) / pedal.c("R_boost_c")
    a_ol = ce_open_loop_gain(pedal.c("R_boost_c"), pedal.c("R_boost_e"), i_c)
    v1, z1 = thevenin_cr_highpass(pedal.c("C_boost_in1"), pedal.c("R_boost_in1"))
    z_s = z1 + cap(pedal.c("C_boost_in2"))

    def booster(z_f: TF) -> TF:
        return v1 * shunt_feedback_gain(z_f, z_s, pedal.c("R_boost_bias"), a_ol)

    r_fb = res(pedal.c("R_boost_fb"))
    r_dist = max(pedal.pots["dist"].resistance(knobs["dist"]), 1.0)
    z_g = res(pedal.c("R_gain")) + cap(pedal.c("C_gain"))
    h_in = rc_highpass(pedal.c("R_in_bias"), pedal.c("C_in")) * rc_highpass(
        pedal.c("R_opamp_bias"), pedal.c("C_opamp_in")
    )
    opamp_hp = noninverting_gain(res(r_dist), z_g)
    opamp_full = noninverting_gain(par(res(r_dist), cap(pedal.c("C_fb"))), z_g)
    h_hp = h_in * booster(r_fb) * opamp_hp
    h_full = h_in * booster(par(r_fb, cap(pedal.c("C_boost_fb")))) * opamp_full
    pre = pre_clip_drive(h_hp)

    r_bias = 1.0 / (1.0 / pedal.c("R_out_bias_a") + 1.0 / pedal.c("R_out_bias_b"))
    z_series = res(pedal.c("R_tone_out")) + cap(pedal.c("C_out_in"))
    stack = lp_hp_blend_tone_stack(
        pedal, knobs["tone"], "R_tone_lp", "C_tone_lp", "C_tone_hp", "R_tone_hp", z_series + r_bias
    )
    h_tone = stack * divider(z_series, res(r_bias))
    h_clip_lp = rc_lowpass(pedal.c("R_clip"), pedal.c("C_clip"))
    h_out = rc_highpass(pedal.c("R_out_pulldown"), pedal.c("C_out"))
    f = log_grid()
    return _post_clip(
        pedal,
        knobs,
        ctx,
        clip=clip,
        pre=pre,
        h_post=(h_full / h_hp) * h_clip_lp * h_tone * h_out,
        post_gain=pedal.pots["level"].fraction(knobs["level"]),
        info={
            "booster_gain_db": float(booster(r_fb).db(GAIN_REF_HZ)),
            "booster_open_loop_db": db(a_ol),
            "opamp_gain_db": db(1 + r_dist / pedal.c("R_gain")),
            "gain_corner_hz": rc_hz(pedal.c("R_gain"), pedal.c("C_gain")),
            "clip_lp_hz": rc_hz(pedal.c("R_clip"), pedal.c("C_clip")),
            "tone_lp_hz": rc_hz(pedal.c("R_tone_lp"), pedal.c("C_tone_lp")),
            "tone_hp_hz": rc_hz(pedal.c("R_tone_hp"), pedal.c("C_tone_hp")),
            "tone_hf_db": float(np.mean(h_tone.db(f[f > 2000]))),
            **response_info("pre_gain", h_full),
        },
        notes=(
            "booster and op-amp cascaded into one Drive (the booster's own soft clipping "
            "is not represented)",
            "R_clip/C_clip 7.2 kHz low-pass applied after the shaper; tone stack driven by "
            "the diode node as an ideal source (clipped regime)",
        ),
    )


@register("big_muff")
def _big_muff(pedal: Pedal, knobs: dict[str, float], ctx: DeriveContext) -> Preset:
    """Big Muff Pi (V3): booster -> Sustain pot -> two diode-feedback clippers -> tone -> Q1.

    Each common-emitter stage is a shunt-feedback amplifier
    (:func:`~lstmabar.physics.networks.shunt_feedback_gain`):
    ``Z_f = R_fb ∥ 1/(sC_miller)``, base bias resistor ``r_in``, open-loop gain
    ``R_c/(R_e + V_T/I_C)`` with ``I_C`` from the collector-feedback bias
    (:func:`~lstmabar.physics.networks.ce_collector_feedback_bias`; gives Q4's ~7 V collector).
    Sources: booster ``Z_s = R_in_series + 1/(sC_in)``; Sustain pot as a divider
    (``R_sustain_floor + R_low`` below the wiper, ``1/(sC_boost_out) + R_high`` above) whose
    Thevenin impedance adds to clip 1's ``Z_s = Z_th + 1/(sC_clip1_in) + R_clip1_in``; clip 2
    ``Z_s = 1/(sC_clip2_in) + R_clip2_in``. Stage outputs are treated as ideal sources. The
    diode branches only conduct when clipping and are left out of the small-signal path.
    All gains up to clip 2 are multiplied into one Drive (``V_clip`` 0.6 V, Si pair), read at
    1 kHz. Only clip 2's Miller low-pass (~1.3 kHz closed loop; its cap sits across the
    clipping diodes, so it filters the output) goes into the post-clip fit: the booster's and
    clip 1's low-passes act before clip 2, which regenerates the harmonics they remove.
    Putting all three after the shaper would cut ~48 dB at 10 kHz.

    Post-clip: :func:`lp_hp_blend_tone_stack` (408 Hz LP leg, 1.81 kHz HP leg) loaded by
    ``1/(sC_out_in) + R_out_bias_hi ∥ R_out_bias_lo``, the voltage at Q1's base, Q1's gain
    ``R_out_c/R_out_e``, the ``C_out`` -> Volume pot high-pass and the Volume pot.
    """
    clip = pedal.clipping[-1]
    vs = pedal.c("V_supply")

    def stage(prefix: str, r_c: str, r_e: str, r_fb: str, r_b: str, c_m: str, z_s: TF):
        i_c = ce_collector_feedback_bias(
            vs, pedal.c(r_c), pedal.c(r_e), pedal.c(r_fb), pedal.c(r_b)
        )
        a_ol = ce_open_loop_gain(pedal.c(r_c), pedal.c(r_e), i_c)
        z_f = res(pedal.c(r_fb))
        hp = shunt_feedback_gain(z_f, z_s, pedal.c(r_b), a_ol)
        full = shunt_feedback_gain(par(z_f, cap(pedal.c(c_m))), z_s, pedal.c(r_b), a_ol)
        info = {f"{prefix}_ic_ma": i_c * 1e3, f"{prefix}_gain_db": float(full.db(GAIN_REF_HZ))}
        return hp, full, info

    z_s4 = res(pedal.c("R_in_series")) + cap(pedal.c("C_in"))
    b_hp, b_full, b_info = stage(
        "booster", "R_boost_c", "R_boost_e", "R_boost_fb", "R_boost_bias", "C_boost_miller", z_s4
    )
    r_low, r_high = pot_split(pedal.pots["sustain"], knobs["sustain"])
    z_top = cap(pedal.c("C_boost_out")) + r_high
    z_bot = res(pedal.c("R_sustain_floor") + r_low)
    v_th, z_th = divider(z_top, z_bot), par(z_top, z_bot)
    z_s3 = z_th + cap(pedal.c("C_clip1_in")) + pedal.c("R_clip1_in")
    c1_hp, c1_full, c1_info = stage(
        "clip1", "R_clip1_c", "R_clip1_e", "R_clip1_fb", "R_clip1_bias", "C_clip1_miller", z_s3
    )
    z_s2 = cap(pedal.c("C_clip2_in")) + pedal.c("R_clip2_in")
    c2_hp, c2_full, c2_info = stage(
        "clip2", "R_clip2_c", "R_clip2_e", "R_clip2_fb", "R_clip2_bias", "C_clip2_miller", z_s2
    )
    h_hp = b_hp * v_th * c1_hp * c2_hp
    h_full = b_full * v_th * c1_full * c2_full
    pre = pre_clip_drive(h_hp)

    r_bias = 1.0 / (1.0 / pedal.c("R_out_bias_hi") + 1.0 / pedal.c("R_out_bias_lo"))
    z_c3 = cap(pedal.c("C_out_in"))
    stack = lp_hp_blend_tone_stack(
        pedal, knobs["tone"], "R_tone_lp", "C_tone_lp", "C_tone_hp", "R_tone_hp", z_c3 + r_bias
    )
    h_tone = stack * divider(z_c3, res(r_bias))
    vol = pedal.pots["volume"]
    h_out = rc_highpass(vol.value, pedal.c("C_out"))
    q1_gain = pedal.c("R_out_c") / pedal.c("R_out_e")
    f = log_grid()
    return _post_clip(
        pedal,
        knobs,
        ctx,
        clip=clip,
        pre=pre,
        h_post=(c2_full / c2_hp) * h_tone * h_out,
        post_gain=q1_gain * vol.fraction(knobs["volume"]),
        info={
            **b_info,
            **c1_info,
            **c2_info,
            "sustain_divider_db": float(v_th.db(GAIN_REF_HZ)),
            "tone_lp_hz": rc_hz(pedal.c("R_tone_lp"), pedal.c("C_tone_lp")),
            "tone_hp_hz": rc_hz(pedal.c("R_tone_hp"), pedal.c("C_tone_hp")),
            "tone_hf_db": float(np.mean(h_tone.db(f[f > 2000]))),
            "output_stage_gain_db": db(q1_gain),
            **response_info("pre_gain", h_full),
            **response_info("tone", h_tone),
        },
        notes=(
            "two clipping stages (and the booster) cascaded into one Drive",
            "clip 2's Miller low-pass applied after the shaper; the booster's and clip 1's "
            "(pre-clip-2) low-passes dropped; stage output impedances, transistor input "
            "currents and the diode + C_clip*_diode branches ignored",
        ),
    )


FUZZ_FACE_HFE = {"ge_transistor": (85.0, 120.0), "si_transistor": (500.0, 500.0)}
"""(Q1, Q2) current gains: ElectroSmash's simulation values for the AC128 germanium build,
Geofex's typical BC108C value for silicon (see the pedal notes). Not stock-fixed: the
fidelity fit / owner may revise them."""

V_BE = {"ge_transistor": 0.2, "si_transistor": 0.6}
"""Base-emitter drop used for the Fuzz Face bias point (volts)."""


@register("fuzz_face_si")
@register("fuzz_face_ge")
def _fuzz_face(pedal: Pedal, knobs: dict[str, float], ctx: DeriveContext) -> Preset:
    """Dallas Arbiter Fuzz Face (Si and Ge): two direct-coupled common-emitter stages with
    shunt-series feedback ``R_fb`` from Q2's emitter to Q1's base.

    Bias: ``I_C2 = (V_supply - V_q2_collector)/(R_out_lo + R_q2_c)``, Q2 emitter at
    ``V_E2 = I_C2·R_fuzz_pot``; ``V_C1 = V_q1_collector`` (if given) else ``V_E2 + V_BE``;
    ``I_C1 = (V_supply - V_C1)/R_q1_c``; ``g_m = I_C/V_T``, ``r_π = h_FE/g_m``.
    The Fuzz pot's wiper-to-emitter part ``R_u`` stays unbypassed; the rest ``R_y`` is
    bypassed by ``C_fuzz_bypass``: ``Z_e = R_u + R_y ∥ 1/(sC)`` (fuzz up -> ``R_u`` -> 0).

        r_in2 = r_π2 + (h_FE2 + 1)·Z_e                      Q2 base input impedance
        A1 = g_m1 · (R_q1_c ∥ r_in2)                        Q1 gain (inverting)
        A2 = h_FE2·(R_out_lo + R_q2_c) / r_in2              Q2 base -> collector
        k  = A1 · (h_FE2 + 1)·Z_e / r_in2                   loop gain to Q2's emitter
        v_b1 / v_s = (1/Z_s) / (1/Z_s + 1/r_π1 + (1 + k)/R_fb),   Z_s = R_source + 1/(sC_in)

    ``Drive`` gain = ``|v_b1/v_s · A1 · A2|`` at 1 kHz (the gain at Q2's collector, where
    ``V_clip`` is referred). The guitar's source impedance sets how much the feedback bites
    (:attr:`DeriveContext.source_ohms`); with a stiff source the feedback does nothing.
    Clipping asymmetry from Q2's swing: toward cutoff ``V_supply - V_q2_collector``, toward
    saturation ``V_q2_collector - V_E2``: ``asymmetry = 2·(1 - swing_sat/swing_cut)``.

    Post-clip: output divider ``R_out_lo/(R_out_lo + R_q2_c)``, ``C_out`` into the Volume pot
    (~31 Hz high-pass), Volume pot fraction.
    """
    clip = pedal.clipping[0]
    hfe1, hfe2 = FUZZ_FACE_HFE[clip.device]
    vs = pedal.c("V_supply")
    r_load = pedal.c("R_out_lo") + pedal.c("R_q2_c")
    fuzz = pedal.pots["fuzz"]
    i_c2 = (vs - pedal.c("V_q2_collector")) / r_load
    v_e2 = i_c2 * fuzz.value
    v_c1 = pedal.components.get("V_q1_collector", v_e2 + V_BE[clip.device])
    i_c1 = (vs - v_c1) / pedal.c("R_q1_c")
    gm1, gm2 = i_c1 / THERMAL_VOLTAGE, i_c2 / THERMAL_VOLTAGE
    r_pi1, r_pi2 = hfe1 / gm1, hfe2 / gm2

    r_y, r_u = pot_split(fuzz, knobs["fuzz"])  # low end = grounded end; high end = emitter
    z_e = res(r_u) + par(res(r_y), cap(pedal.c("C_fuzz_bypass")))
    r_in2 = r_pi2 + (hfe2 + 1) * z_e
    a1 = gm1 * par(res(pedal.c("R_q1_c")), r_in2)
    a2 = hfe2 * r_load / r_in2
    k = a1 * (hfe2 + 1) * z_e / r_in2
    y_s = 1 / (res(ctx.source_ohms) + cap(pedal.c("C_in")))
    v_b = y_s / (y_s + 1.0 / r_pi1 + (1 + k) / pedal.c("R_fb"))
    h = v_b * a1 * a2
    pre = pre_clip_drive(h)

    shaper = shaper_prior(clip)
    swing_cut = vs - pedal.c("V_q2_collector")
    swing_sat = pedal.c("V_q2_collector") - v_e2
    lo, hi = sorted((swing_cut, swing_sat))
    shaper["asymmetry"] = min(max(2.0 * (1.0 - lo / hi), 0.0), 1.0)

    vol = pedal.pots["volume"]
    r_out = pedal.c("R_out_lo") * pedal.c("R_q2_c") / r_load
    h_out = rc_highpass(vol.value + r_out, pedal.c("C_out"))
    return _post_clip(
        pedal,
        knobs,
        ctx,
        clip=clip,
        pre=pre,
        h_post=h_out,
        post_gain=pedal.c("R_out_lo") / r_load * vol.fraction(knobs["volume"]),
        shaper=shaper,
        info={
            "q1_ic_ma": i_c1 * 1e3,
            "q2_ic_ma": i_c2 * 1e3,
            "q1_gain_db": float(a1.db(GAIN_REF_HZ)),
            "q2_gain_db": float(a2.db(GAIN_REF_HZ)),
            "input_attenuation_db": float(v_b.db(GAIN_REF_HZ)),
            "unbypassed_ohms": r_u,
            "output_divider_db": db(pedal.c("R_out_lo") / r_load),
            "output_hp_hz": rc_hz(vol.value + r_out, pedal.c("C_out")),
            # C2 against the bypassed section ∥ (r_e2 + R_u): the emitter-bypass corner
            "bypass_hp_hz": rc_hz(
                1.0 / (1.0 / r_y + 1.0 / (1.0 / gm2 + r_u)), pedal.c("C_fuzz_bypass")
            ),
            "source_ohms": ctx.source_ohms,
            **response_info("pre_gain", h),
        },
        notes=(
            f"gain depends on the guitar source impedance (assumed {ctx.source_ohms:g} ohm) "
            "and on transistor h_FE (assumed); input-impedance interaction not modelled",
            "Q1's one-sided soft saturation not represented (bias 0); asymmetry from Q2's "
            "cutoff/saturation swing",
        ),
    )


__all__ = [
    "BLOCK_SPECS",
    "DERIVERS",
    "GAIN_REF_HZ",
    "DeriveContext",
    "PreClip",
    "Preset",
    "derive",
    "lp_hp_blend_tone_stack",
    "pre_clip_drive",
    "preset_params",
    "rc_hz",
    "register",
    "shaper_prior",
    "stage_clip_volts",
    "ts808_tone_stage",
]
