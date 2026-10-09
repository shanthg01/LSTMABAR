"""Pure rendering helpers behind the demo UI (no gradio import, testable on their own).

The UI speaks physical units (Hz, dB, ms); the pedalboard takes normalized [0, 1] knobs. This
module converts between them via each block's :class:`~lstmabar.dsp.base.ParamSpec` and never
hard-codes knob names, so it works with any board exposing the ``Pedalboard`` API:
``blocks`` (ordered name -> EffectBlock), ``block_names``, ``sample_rate``, ``forward``,
``default_params`` and ``describe``.
"""

from typing import Any

import numpy as np
import torch
from torch import Tensor

from lstmabar.audio import loudness_match, synth_riff, to_mono_float

BoardParams = dict[str, dict[str, Tensor]]

MAX_PEAK = 0.99


def iter_specs(board: Any):
    """Yield ``(block_name, ParamSpec)`` in board order."""
    for name in board.block_names:
        for spec in board.blocks[name].param_specs:
            yield name, spec


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

    Missing knobs fall back to their spec default and missing blocks are enabled.
    """
    params: BoardParams = {}
    for name in board.block_names:
        block_values = values.get(name, {})
        p: dict[str, Tensor] = {}
        for spec in board.blocks[name].param_specs:
            v = float(block_values.get(spec.name, spec.default))
            p[spec.name] = spec.normalize(torch.tensor([v], dtype=torch.float32))
        p["enabled"] = torch.tensor([1.0 if enabled.get(name, True) else 0.0])
        params[name] = p
    return params


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
    are not biased by level.
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
    return f"{f:.4g}"


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
    """The clip to process: the uploaded/recorded audio if any, else a synthetic riff."""
    if audio_value is not None:
        sr, data = audio_value
        x = to_mono_float(data, sr, sample_rate)
        if x.size:
            return x[: int(max_seconds * sample_rate)]
    return synth_riff(riff_kind, seconds=min(riff_seconds, max_seconds), sample_rate=sample_rate)


def to_int16(x: np.ndarray) -> np.ndarray:
    """Float [-1, 1] -> int16 PCM (what browser audio players expect, avoids gradio warnings)."""
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
