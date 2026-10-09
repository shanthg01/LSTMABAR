"""Pure rendering helpers behind the demo UI (no gradio import, testable on their own).

Knob values come in three representations, converted via each block's
:class:`~lstmabar.dsp.base.ParamSpec`. Knob names are never hard-coded, so this works with any
board exposing the ``Pedalboard`` API (``blocks``, ``block_names``, ``sample_rate``,
``forward``, ``default_params``, ``describe``):

- **physical** (Hz, dB, ms): used by ``default_knobs``, ``random_knobs``, ``knobs_to_params``
  and the readout;
- **normalized** [0, 1]: what the pedalboard takes;
- **slider value**: what a UI slider holds. Linear knobs use physical units; log-taper knobs
  use their normalized position, because a linear slider across decades would need ~1e6
  steps. Convert with :func:`slider_from_physical` / :func:`physical_from_slider`, and build
  sliders from :func:`slider_config`.
"""

from typing import Any

import numpy as np
import torch
from torch import Tensor

from lstmabar.audio import loudness_match, synth_riff, to_mono_float
from lstmabar.dsp.base import ParamSpec

BoardParams = dict[str, dict[str, Tensor]]

MAX_PEAK = 0.99
LINEAR_SLIDER_STEPS = 200
LOG_SLIDER_STEP = 0.001


def iter_specs(board: Any):
    """Yield ``(block_name, ParamSpec)`` in board order."""
    for name in board.block_names:
        for spec in board.blocks[name].param_specs:
            yield name, spec


# --- Slider <-> physical ------------------------------------------------------------------------


def _clamp(spec: ParamSpec, v: float) -> float:
    return min(max(float(v), spec.min), spec.max)


def slider_from_physical(spec: ParamSpec, v: float) -> float:
    """Physical value -> slider value (normalized position for log-taper knobs)."""
    v = _clamp(spec, v)
    if spec.taper == "log":
        return float(spec.normalize(torch.tensor(v, dtype=torch.float64)))
    return v


def physical_from_slider(spec: ParamSpec, s: float | None) -> float:
    """Slider value -> physical value (``None`` means the spec default)."""
    if s is None:
        return float(spec.default)
    if spec.taper == "log":
        return float(spec.denormalize(torch.tensor(float(s), dtype=torch.float64)))
    return _clamp(spec, s)


def _num(v: float) -> str:
    """4 significant digits, but never scientific notation for large values (20000, not 2e+04)."""
    return f"{v:.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def format_value(spec: ParamSpec, v: float) -> str:
    return f"{_num(v)} {spec.unit}".strip()


def log_slider_info(spec: ParamSpec, physical: float) -> str:
    """Info line under a log-taper slider: current physical value and range."""
    lo, hi = format_value(spec, spec.min), format_value(spec, spec.max)
    return f"= {format_value(spec, physical)}  (range {lo} to {hi})"


def slider_config(spec: ParamSpec) -> dict[str, Any]:
    """Keyword arguments for a UI slider for ``spec``: minimum/maximum/step/value/label/info."""
    unit = f" ({spec.unit})" if spec.unit else ""
    if spec.taper == "log":
        return {
            "minimum": 0.0,
            "maximum": 1.0,
            "step": LOG_SLIDER_STEP,
            "value": slider_from_physical(spec, spec.default),
            "label": f"{spec.name}{unit}, log taper",
            "info": log_slider_info(spec, spec.default),
        }
    return {
        "minimum": spec.min,
        "maximum": spec.max,
        "step": float(f"{(spec.max - spec.min) / LINEAR_SLIDER_STEPS:.3g}"),
        "value": spec.default,
        "label": f"{spec.name}{unit}",
        "info": None,
    }


# --- Physical knobs -> board params -------------------------------------------------------------


def default_knobs(board: Any) -> dict[str, dict[str, float]]:
    """Physical default value of every knob, keyed by block then knob name."""
    out: dict[str, dict[str, float]] = {name: {} for name in board.block_names}
    for name, spec in iter_specs(board):
        out[name][spec.name] = float(spec.default)
    return out


def random_knobs(board: Any, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """Random physical knob values, uniform in normalized space (so log knobs sweep evenly)."""
    out: dict[str, dict[str, float]] = {name: {} for name in board.block_names}
    for name, spec in iter_specs(board):
        u = torch.tensor(float(rng.uniform()), dtype=torch.float64)
        out[name][spec.name] = float(spec.denormalize(u))
    return out


def knobs_to_params(
    board: Any,
    values: dict[str, dict[str, float]],
    enabled: dict[str, bool],
) -> BoardParams:
    """Physical knob values + per-block on/off -> normalized ``BoardParams`` with batch size 1.

    Values are clamped to the spec range first; missing knobs fall back to their spec default
    and missing blocks are enabled.
    """
    params: BoardParams = {}
    for name in board.block_names:
        block_values = values.get(name, {})
        p: dict[str, Tensor] = {}
        for spec in board.blocks[name].param_specs:
            v = _clamp(spec, block_values.get(spec.name, spec.default))
            p[spec.name] = spec.normalize(torch.tensor([v], dtype=torch.float32))
        p["enabled"] = torch.tensor([1.0 if enabled.get(name, True) else 0.0])
        params[name] = p
    return params


# --- Rendering ----------------------------------------------------------------------------------


def _peak_safe(x: np.ndarray, max_peak: float = MAX_PEAK) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m > max_peak:
        x = x * (max_peak / m)
    return x.astype(np.float32)


def render(
    board: Any, audio: np.ndarray, params: BoardParams, match_loudness: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``audio`` through ``board`` -> ``(dry, wet)``, both mono float32 and peak-safe.

    With ``match_loudness`` the wet signal is RMS-matched to the dry one so A/B comparisons
    are not biased by level (it may stay quieter when the peak limit kicks in).
    """
    dry = _peak_safe(audio)
    x = torch.from_numpy(dry.copy()).unsqueeze(0)
    with torch.inference_mode():
        wet_t = board(x, params)
    wet = wet_t.detach().reshape(-1).to(torch.float32).cpu().numpy()
    wet = _peak_safe(wet)
    if match_loudness:
        wet = loudness_match(wet, dry, max_peak=MAX_PEAK)
    return dry, wet


def _fmt(v: Any) -> str:
    if isinstance(v, list | tuple):
        if len(v) == 1:
            return _fmt(v[0])
        return ", ".join(_fmt(x) for x in v)
    if isinstance(v, Tensor):
        return _fmt(v.reshape(-1).tolist())
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return _num(f)


def describe_text(board: Any, params: BoardParams) -> str:
    """Markdown table of the physical settings the board reports for ``params``."""
    desc = board.describe(params)
    units = {(name, spec.name): spec.unit for name, spec in iter_specs(board)}
    lines = ["| Block | Knob | Value |", "|---|---|---|"]
    for block, knobs in desc.items():
        for knob, value in knobs.items():
            if knob == "enabled":
                v = value[0] if isinstance(value, list | tuple) else value
                text = "on" if float(v) >= 0.5 else "off"
            else:
                unit = units.get((block, knob), "")
                text = f"{_fmt(value)} {unit}".strip()
            lines.append(f"| {block} | {knob} | {text} |")
    return "\n".join(lines)


def prepare_input(
    audio_value: tuple[int, np.ndarray] | None,
    riff_kind: str,
    sample_rate: int,
    max_seconds: float,
    riff_seconds: float = 4.0,
) -> np.ndarray:
    """The clip to process: the uploaded/recorded audio if any, else a synthetic riff.

    Uploads are trimmed to ``max_seconds`` before conversion and resampling. Raises
    ``ValueError`` with a user-facing message on a bad sample rate or an empty clip.
    """
    if audio_value is None:
        seconds = min(riff_seconds, max_seconds)
        return synth_riff(riff_kind, seconds=seconds, sample_rate=sample_rate)
    sr, data = audio_value
    if sr is None or int(sr) <= 0:
        raise ValueError(f"The uploaded clip has an invalid sample rate ({sr}).")
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError("The uploaded clip is empty.")
    return to_mono_float(data, int(sr), sample_rate, max_seconds=max_seconds)


def to_int16(x: np.ndarray) -> np.ndarray:
    """Float [-1, 1] -> int16 PCM (what browser audio players expect, avoids gradio warnings)."""
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
