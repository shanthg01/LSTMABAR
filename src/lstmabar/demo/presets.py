"""Pure helpers behind the M2 pedal picker and the grey-box vs white-box A/B (no gradio import).

Behaviour (P2 task 2.6):

- **Pedal dials** use the pedal's own knobs from the knowledge base (``pots``: label, default)
  on a 0–10 scale, as printed on the enclosure; a dial value ``d`` is the rotation ``d / 10``.
- **Derived board knobs.** Moving a dial runs :func:`~lstmabar.physics.derive.derive` and
  writes the physical result into the board sliders (:func:`preset_controls`), so the grey-box
  setting is visible.
- **EQ lock.** While a pedal is active, the board's ``eq`` block holds the fitted tone stack of
  the pedal (P2 decision 2), not a user EQ, so its sliders and its on/off box are locked
  (:data:`LOCKED_BLOCKS`). Switching back to "Manual board" unlocks them and keeps the values.
- **Tweaks** (:func:`resolve_render`). The app remembers the dial values the board sliders were
  last derived from. At render time, if the sliders still match *that* preset (within one
  slider step), nothing was hand-edited and the grey-box renders the preset derived from the
  *current* dials (so a dial move whose re-derive hasn't reached the sliders yet is not lost).
  Otherwise the sliders were edited: the grey-box renders them as set, labelled "custom
  (edited from <pedal>)". The white-box always uses the current dials, i.e. the same dial
  state as the grey-box whenever it is not custom.
- **White-box** renders only for pedals that have a registered model (looked up at runtime
  via :func:`~lstmabar.physics.whitebox.has_whitebox`) and only with the compiled numba solver
  (:func:`whitebox_ready`; the NumPy fallback would block a render for ~30 s), on a clip trimmed
  to :data:`WHITEBOX_MAX_SECONDS`. Failures give a generic message (details go to the log).
"""

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from lstmabar.audio import loudness_match
from lstmabar.demo.render import (
    LINEAR_SLIDER_STEPS,
    LOG_SLIDER_STEP,
    MAX_PEAK,
    _num,
    _peak_safe,
    iter_specs,
    physical_from_slider,
    slider_config,
    slider_from_physical,
)
from lstmabar.physics.derive import DeriveContext, Preset, derive
from lstmabar.physics.kb import Pedal, Pot

log = logging.getLogger(__name__)

MANUAL = "manual"
MANUAL_LABEL = "Manual board"
LOCKED_BLOCKS: tuple[str, ...] = ("eq",)
DIAL_MAX = 10.0
DIAL_STEP = 0.1
WHITEBOX_MAX_SECONDS = 10.0
WHITEBOX_ERROR = "White-box render failed for this setting; the grey-box output is unaffected."
WHITEBOX_NEEDS_NUMBA = (
    "A white-box model exists for this pedal, but rendering it needs the compiled solver: "
    "`uv sync --extra whitebox`."
)

# Preset.info keys shown in the readout (when a deriver records them), in display order.
# The gain into the clipper is shown separately, with the frequency it was read at.
_INFO_KEYS: tuple[tuple[str, str, str], ...] = (
    ("v_clip", "Clip level", "V"),
    ("post_gain_db", "Post-clip gain", "dB"),
    ("tone_fit_rms_db", "Tone-stack fit error (RMS)", "dB"),
    ("pre_hpf_fit_rms_db", "Pre-HPF fit error (RMS)", "dB"),
)


# --- Pedal dials ----------------------------------------------------------------------------


def pedal_choices(kb: Mapping[str, Pedal]) -> list[tuple[str, str]]:
    """Dropdown choices ``(label, value)``: the manual board, then every pedal by name."""
    return [(MANUAL_LABEL, MANUAL)] + [(p.name, pid) for pid, p in kb.items()]


def dial_config(pot: Pot) -> dict[str, Any]:
    """Keyword arguments for a 0–10 dial slider for ``pot`` (default from the YAML)."""
    taper = {"A": "audio", "B": "linear", "C": "reverse audio"}.get(pot.taper, pot.taper)
    return {
        "minimum": 0.0,
        "maximum": DIAL_MAX,
        "step": DIAL_STEP,
        "value": round(pot.default * DIAL_MAX, 1),
        "label": pot.label,
        "info": f"{_num(pot.value / 1e3)}k {taper}-taper pot, 0–10",
    }


def dials_to_knobs(pedal: Pedal, dial_values: Sequence[float | None]) -> dict[str, float]:
    """0–10 dial values (in ``pedal.pots`` order; ``None`` = YAML default) -> rotations."""
    names = list(pedal.pots)
    if len(dial_values) != len(names):
        raise ValueError(f"{pedal.id}: expected {len(names)} dial values, got {len(dial_values)}")
    out = {}
    for name, d in zip(names, dial_values, strict=True):
        if d is None or not math.isfinite(float(d)):
            out[name] = pedal.pots[name].default
        else:
            out[name] = min(max(float(d) / DIAL_MAX, 0.0), 1.0)
    return out


def derive_preset(pedal: Pedal, dial_values: Sequence[float | None], sample_rate: int) -> Preset:
    return derive(pedal, dials_to_knobs(pedal, dial_values), DeriveContext(sample_rate=sample_rate))


# --- Preset -> board sliders ----------------------------------------------------------------


def preset_board_values(
    board: Any, preset: Preset
) -> tuple[dict[str, bool], dict[str, dict[str, float]]]:
    """Per-block on/off and physical knob values of ``preset`` for the blocks ``board`` has.

    Knobs the preset doesn't set use their spec default; blocks the preset doesn't mention
    stay enabled.
    """
    enabled = {name: bool(preset.enabled.get(name, True)) for name in board.block_names}
    values: dict[str, dict[str, float]] = {name: {} for name in board.block_names}
    for name, spec in iter_specs(board):
        v = preset.board.get(name, {}).get(spec.name, spec.default)
        values[name][spec.name] = min(max(float(v), spec.min), spec.max)
    return enabled, values


def preset_controls(board: Any, preset: Preset) -> tuple[list[bool], list[float]]:
    """Checkbox values (board order) and slider values (``iter_specs`` order) for a preset."""
    enabled, values = preset_board_values(board, preset)
    sliders = [
        slider_from_physical(spec, values[name][spec.name]) for name, spec in iter_specs(board)
    ]
    return [enabled[name] for name in board.block_names], sliders


def is_locked(block_name: str, pedal_active: bool) -> bool:
    """Whether a board block's sliders are read-only (the pedal's tone stack drives them)."""
    return pedal_active and block_name in LOCKED_BLOCKS


def _slider_tolerance(spec: Any) -> float:
    if spec.taper == "log":
        return LOG_SLIDER_STEP
    return float(slider_config(spec)["step"]) or (spec.max - spec.min) / LINEAR_SLIDER_STEPS


def preset_matches(
    board: Any, preset: Preset, enabled_vals: Sequence[Any], slider_vals: Sequence[Any]
) -> bool:
    """True if the UI controls still hold ``preset`` (within one slider step per knob)."""
    want_enabled, want_sliders = preset_controls(board, preset)
    if [bool(v) for v in enabled_vals] != want_enabled:
        return False
    for (_, spec), got, want in zip(iter_specs(board), slider_vals, want_sliders, strict=True):
        if got is None:
            got = slider_from_physical(spec, physical_from_slider(spec, None))
        if abs(float(got) - want) > _slider_tolerance(spec) + 1e-9:
            return False
    return True


@dataclass
class RenderChoice:
    preset: Preset  # derived from the current dials (white-box and readout use these)
    edited: bool  # sliders hand-edited since the last derive: render them as set ("custom")


def resolve_render(
    board: Any,
    pedal: Pedal,
    dial_values: Sequence[float | None],
    derived_dials: Sequence[float | None] | None,
    enabled_vals: Sequence[Any],
    slider_vals: Sequence[Any],
    sample_rate: int,
) -> RenderChoice:
    """Decide what the grey-box renders for an active pedal (see the module docstring).

    ``derived_dials`` are the dial values the board sliders were last derived from (``None``
    if the board was never derived for this pedal, e.g. the pick's own derive is still
    running: then nothing can have been hand-edited yet).
    """
    preset = derive_preset(pedal, dial_values, sample_rate)
    if derived_dials is None:
        return RenderChoice(preset, False)
    if dials_to_knobs(pedal, derived_dials) == preset.knobs:
        base = preset
    else:
        base = derive_preset(pedal, derived_dials, sample_rate)
    return RenderChoice(preset, not preset_matches(board, base, enabled_vals, slider_vals))


# --- Readout --------------------------------------------------------------------------------


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _hz(v: float) -> str:
    """``264 Hz``, ``1 kHz``, ``10 kHz``."""
    return f"{_num(v / 1e3)} kHz" if abs(v) >= 1000 else f"{_num(v)} Hz"


def preset_readout(pedal: Pedal, preset: Preset, whitebox: bool, edited: bool = False) -> str:
    """Markdown summary of a derived preset: dials, key circuit quantities and notes."""
    dials = ", ".join(
        f"{pot.label} {preset.knobs[k] * DIAL_MAX:.1f}" for k, pot in pedal.pots.items()
    )
    status = f"custom (edited from {pedal.name})" if edited else pedal.name
    lines = [f"**Grey-box:** {status} — {dials}"]
    drive = preset.board.get("drive", {})
    rows = []
    if "gain_db" in drive:
        rows.append(("Drive gain (board)", f"{_num(drive['gain_db'])} dB"))
    if "pre_hpf_hz" in drive:
        rows.append(("Pre-clip high-pass corner", _hz(drive["pre_hpf_hz"])))
    gain = _finite(preset.info.get("drive_stage_gain_db"))
    if gain is not None:
        ref = _finite(preset.info.get("gain_ref_hz"))
        where = f" (evaluated at {_hz(ref)})" if ref is not None else ""
        rows.append(("Circuit gain into the clipper", f"{_num(gain)} dB{where}"))
    for key, label, unit in _INFO_KEYS:
        v = _finite(preset.info.get(key))
        if v is not None:
            rows.append((label, f"{_num(v)} {unit}".strip()))
    if rows:
        lines += ["", "| Circuit quantity | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in rows]
    lines.append("")
    lines.append(
        "EQ is locked (sliders and on/off): it holds the fitted tone stack of the pedal, "
        "not a separate user EQ."
    )
    if whitebox:
        lines.append("A white-box circuit simulation is available for this pedal (output C).")
    if preset.notes:
        lines += ["", "**Approximations / clamps:**"] + [f"- {n}" for n in preset.notes]
    return "\n".join(lines)


# --- White-box ------------------------------------------------------------------------------


def whitebox_ready() -> bool:
    """Whether the compiled (numba) white-box solver is available.

    Same check the solver uses to pick its backend; without it the NumPy loop runs at ~0.1x
    real time, too slow for an interactive render, so the demo disables output C.
    """
    try:
        from lstmabar.physics.whitebox.diode_clipper import numba_available
    except Exception:  # pragma: no cover - broken optional install
        log.exception("white-box solver unavailable")
        return False
    return bool(numba_available())


def whitebox_available(pedal_id: str) -> bool:
    """Whether ``pedal_id`` has a registered white-box model (False if the module fails)."""
    try:
        from lstmabar.physics.whitebox import has_whitebox
    except Exception:  # pragma: no cover - broken optional install
        log.exception("white-box registry unavailable")
        return False
    return bool(has_whitebox(pedal_id))


@dataclass
class WhiteBoxResult:
    audio: np.ndarray | None  # mono float32, peak-safe; None on failure
    message: str


def whitebox_render(
    pedal: Pedal,
    knobs: Mapping[str, float],
    clip: np.ndarray,
    sample_rate: int,
    match_loudness: bool = True,
    max_seconds: float | None = None,
    simulate_fn: Any = None,
) -> WhiteBoxResult:
    """Render ``clip`` (trimmed to ``max_seconds``) through ``pedal``'s white-box model.

    The output is peak-safe and, with ``match_loudness``, RMS-matched to the same stretch of
    the dry clip. Any failure returns ``audio=None`` and a generic message; the details are
    logged server-side only. ``simulate_fn`` replaces
    :func:`~lstmabar.physics.whitebox.simulate` (tests).
    """
    max_seconds = WHITEBOX_MAX_SECONDS if max_seconds is None else max_seconds
    n = min(len(clip), max(1, int(round(max_seconds * sample_rate))))
    dry = _peak_safe(np.asarray(clip)[:n])
    try:
        if simulate_fn is None:
            from lstmabar.physics.whitebox import simulate as simulate_fn
        y = np.asarray(simulate_fn(pedal, dry, sample_rate, knobs), dtype=np.float64).reshape(-1)
        if y.shape != dry.shape or not np.all(np.isfinite(y)):
            raise ValueError(
                f"white-box output invalid: shape {y.shape}, finite={np.isfinite(y).all()}"
            )
    except Exception:
        log.exception("white-box render failed for %s", pedal.id)
        return WhiteBoxResult(None, WHITEBOX_ERROR)
    wet = _peak_safe(y)
    if match_loudness:
        wet = loudness_match(wet, dry, max_peak=MAX_PEAK)
    trimmed = len(clip) > n
    note = f" (first {max_seconds:.0f} s of the clip)" if trimmed else ""
    return WhiteBoxResult(wet, f"White-box circuit simulation of {pedal.name}{note}.")


__all__ = [
    "DIAL_MAX",
    "LOCKED_BLOCKS",
    "MANUAL",
    "MANUAL_LABEL",
    "WHITEBOX_ERROR",
    "WhiteBoxResult",
    "derive_preset",
    "dial_config",
    "dials_to_knobs",
    "is_locked",
    "pedal_choices",
    "preset_board_values",
    "preset_controls",
    "preset_matches",
    "preset_readout",
    "whitebox_available",
    "WHITEBOX_NEEDS_NUMBA",
    "RenderChoice",
    "resolve_render",
    "whitebox_ready",
    "whitebox_render",
]
