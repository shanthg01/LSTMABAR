"""Pure helpers behind the M2 pedal picker and the grey-box vs white-box A/B (no gradio import).

Behaviour (P2 task 2.6):

- **Pedal dials** use the pedal's own knobs from the knowledge base (``pots``: label, default)
  on a 0–10 scale, as printed on the enclosure; a dial value ``d`` is the rotation ``d / 10``.
- **Derived board knobs.** Moving a dial runs :func:`~lstmabar.physics.derive.derive` and
  writes the physical result into the board sliders (:func:`preset_controls`), so the grey-box
  setting is visible.
- **EQ lock.** While a pedal is active, the board's ``eq`` block holds the fitted tone stack of
  the pedal (P2 decision 2), not a user EQ, so its sliders are locked (:data:`LOCKED_BLOCKS`).
  Switching back to "Manual board" unlocks them and keeps the current values.
- **Tweaks.** The other board sliders stay editable. At render time the grey-box output uses
  the derived preset exactly when the sliders still match it (:func:`preset_matches`, within
  one slider step); after a tweak it uses the sliders as set and is labelled "custom (edited
  from <pedal>)". The white-box always follows the pedal dials.
- **White-box** renders only for pedals that have a registered model (looked up at runtime
  via :func:`~lstmabar.physics.whitebox.has_whitebox`), on a clip trimmed to
  :func:`whitebox_max_seconds`, and failures give a generic message (details go to the log).
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
WHITEBOX_MAX_SECONDS_NUMPY = 3.0  # the pure-NumPy solver runs at ~0.1x real time
WHITEBOX_ERROR = "White-box render failed for this setting; the grey-box output is unaffected."

# Preset.info keys shown in the readout (when a deriver records them), in display order.
_INFO_KEYS: tuple[tuple[str, str, str], ...] = (
    ("drive_stage_gain_db", "Circuit gain into the clipper", "dB"),
    ("v_clip", "Clip level", "V"),
    ("gain_ref_hz", "Gain read at", "Hz"),
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


# --- Readout --------------------------------------------------------------------------------


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
        rows.append(("Pre-clip high-pass corner", f"{_num(drive['pre_hpf_hz'])} Hz"))
    for key, label, unit in _INFO_KEYS:
        v = preset.info.get(key)
        if v is not None and math.isfinite(float(v)):
            rows.append((label, f"{_num(float(v))} {unit}".strip()))
    if rows:
        lines += ["", "| Circuit quantity | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in rows]
    lines.append("")
    lines.append(
        "EQ is locked: it holds the fitted tone stack of the pedal, not a separate user EQ."
    )
    if whitebox:
        lines.append("A white-box circuit simulation is available for this pedal (output C).")
    if preset.notes:
        lines += ["", "**Approximations / clamps:**"] + [f"- {n}" for n in preset.notes]
    return "\n".join(lines)


# --- White-box ------------------------------------------------------------------------------


def _numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def whitebox_max_seconds() -> float:
    """Clip length used for the white-box render (shorter without the numba kernel)."""
    return WHITEBOX_MAX_SECONDS if _numba_available() else WHITEBOX_MAX_SECONDS_NUMPY


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
    max_seconds = whitebox_max_seconds() if max_seconds is None else max_seconds
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
    "whitebox_max_seconds",
    "whitebox_render",
]
