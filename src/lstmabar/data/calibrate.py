"""Resolution R4 (wave 1b-cal): per-pedal white-box calibration tables for labels.

Contract: ``run(cfg, args)`` fits best-fit board params (``physics.fidelity`` machinery, at
``audio.sample_rate``) on a ``calibration.gain_points`` x ``calibration.tone_points`` knob grid
per pedal in ``render.whitebox_pedals``, jointly over ``calibration.sine_levels_dbfs``.
Each gain chain starts from the derived params at gain = 0 and is warm-started along the
gain axis. Tables are written to ``DataLayout.calibration_table(pedal)`` (JSON, committed).

The renderer only uses :meth:`CalibrationTable.load` and :meth:`CalibrationTable.lookup`.

Procedure per pedal (gain knob ``g``, tone knob ``t``, level knob at its pot default):

1. Grid rotations evenly spaced in [0, 1] (one point = 0.5).
2. Per grid setting, the white-box renders of sines at ``calibration.f0s_hz`` ×
   ``calibration.sine_levels_dbfs`` (+ the fidelity riff) are the target; the 11 Drive + EQ3
   knobs (:func:`~lstmabar.physics.fidelity.fit_knobs`) are fitted jointly to all of them
   with :func:`~lstmabar.physics.fidelity.fit` (``calibration.fit_steps`` Adam steps).
   At gain index 0 the starts are the derived params plus ``calibration.restarts`` random
   starts; each next gain point starts from the previous gain point's best fit (same tone),
   so neighbouring fits stay continuous. The kept fit is the start with the lowest masked
   mean harmonic error (the fidelity gate metric) on the full analysis window.
3. The fitted subset is expanded to a full normalized ``FloatBoard``: every other knob comes
   from the derived preset at the same setting; compressor disabled.
4. The derived (unfitted) label's error at the same setting is stored for comparison.

The level knob is applied analytically at :meth:`CalibrationTable.lookup`: the derivation's
post-clip scalar gain (``Preset.info["post_gain_db"]``, the level pot's divider and fixed
output gains; the only term of ``drive.level_db`` that depends on the level knob) at the
requested level minus that at the fitted level is added to ``drive.level_db``.
"""

from __future__ import annotations

import functools
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lstmabar.data.schema import FloatBoard

ENABLED = "enabled"
GAIN_KNOB = {"ts808": "drive", "ds1": "dist"}
TONE_KNOB = {"ts808": "tone", "ds1": "tone"}
LEVEL_KNOB = {"ts808": "level", "ds1": "level"}
LEVEL_PARAM = ("drive", "level_db")
REPORT_NAME = "calibration"


@dataclass(frozen=True)
class Label:
    """Result of :meth:`CalibrationTable.lookup`."""

    board: FloatBoard  # normalized params, full BoardParams layout incl. "enabled"
    source: str  # "calibrated" (interpolated) | "calibrated_nearest" (fallback)
    clamp_notes: list[str] = field(default_factory=list)  # e.g. level offset hit the range


@dataclass(frozen=True)
class CalibrationTable:
    """Fitted normalized board params on a knob grid for one white-box pedal.

    ``gain_knob`` / ``tone_knob`` name the pedal's pots spanning the grid (e.g. ``drive`` /
    ``tone`` for ts808, ``dist`` / ``tone`` for ds1); ``level_knob`` is applied analytically.
    ``boards[i][j]`` is the fit at ``(gain_values[i], tone_values[j])``.
    """

    pedal_id: str
    gain_knob: str
    tone_knob: str
    level_knob: str | None
    gain_values: list[float]
    tone_values: list[float]
    boards: list[list[FloatBoard]]
    fit_error_db: list[list[float]] = field(default_factory=list)
    max_jump: float = 0.25
    meta: dict = field(default_factory=dict)  # config, git sha, sample rate, levels, f0s
    level_default: float | None = None
    """Level knob rotation the grid was fitted at (the pot default)."""
    derived_error_db: list[list[float]] = field(default_factory=list)
    """Error of the derived (unfitted) label at each grid setting, same metric."""

    @classmethod
    def load(cls, path: str | Path) -> CalibrationTable:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: Mapping) -> CalibrationTable:
        return cls(**dict(data))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    # --- Lookup ---------------------------------------------------------------------------------

    def lookup(self, knobs: Mapping[str, float]) -> Label:
        """Normalized label for full pedal ``knobs`` (rotations in [0, 1]).

        Bilinear interpolation of the grid fits (``source="calibrated"``), or the nearest grid
        point (``"calibrated_nearest"``) where neighbouring fits differ by more than
        ``max_jump``. The level knob's analytic offset is added to ``drive.level_db`` and
        clamped to Drive's range; a clamp is reported in ``Label.clamp_notes``.
        """
        g = _rotation(knobs, self.gain_knob)
        t = _rotation(knobs, self.tone_knob)
        i0, i1, wg = _bracket(self.gain_values, g)
        j0, j1, wt = _bracket(self.tone_values, t)
        corners = [(i0, j0), (i0, j1), (i1, j0), (i1, j1)]
        weights = [(1 - wg) * (1 - wt), (1 - wg) * wt, wg * (1 - wt), wg * wt]
        nearest = (i1 if wg > 0.5 else i0, j1 if wt > 0.5 else j0)
        nb = self.boards[nearest[0]][nearest[1]]
        if self.cell_jump(i0, j0, i1, j1) > self.max_jump:
            board = {b: dict(v) for b, v in nb.items()}
            source = "calibrated_nearest"
        else:
            board = {}
            for b, kv in nb.items():
                board[b] = {}
                for k, v in kv.items():
                    if k == ENABLED:  # gates are copied, never interpolated
                        board[b][k] = v
                    else:
                        board[b][k] = sum(
                            w * self.boards[i][j][b][k]
                            for w, (i, j) in zip(weights, corners, strict=True)
                            if w != 0.0
                        )
            source = "calibrated"
        notes: list[str] = []
        if self.level_knob is not None and self.level_knob in knobs:
            level = _rotation(knobs, self.level_knob)
            ref = self.level_default if self.level_default is not None else level
            if level != ref:
                offset = level_offset_db(self.pedal_id, self.level_knob, level, ref)
                _apply_level_offset(board, offset, notes)
        return Label(board=board, source=source, clamp_notes=notes)

    def cell_jump(self, i0: int, j0: int, i1: int, j1: int) -> float:
        """Largest pairwise jump among the (up to 4) corner boards of a cell."""
        pts = sorted({(i0, j0), (i0, j1), (i1, j0), (i1, j1)})
        jump = 0.0
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                ba = self.boards[pts[a][0]][pts[a][1]]
                bb = self.boards[pts[b][0]][pts[b][1]]
                jump = max(jump, board_jump(ba, bb))
        return jump


def board_jump(a: FloatBoard, b: FloatBoard) -> float:
    """Max absolute normalized difference over all knobs (gates excluded)."""
    return max(
        (abs(a[blk][k] - b[blk][k]) for blk in a for k in a[blk] if k != ENABLED),
        default=0.0,
    )


def _rotation(knobs: Mapping[str, float], name: str) -> float:
    if name not in knobs:
        raise KeyError(f"lookup needs knob {name!r}, got {sorted(knobs)}")
    v = float(knobs[name])
    if not 0.0 <= v <= 1.0:  # also rejects NaN
        raise ValueError(f"knob {name}: rotation {v} outside [0, 1]")
    return v


def _bracket(values: list[float], x: float) -> tuple[int, int, float]:
    """Indices ``(lo, hi)`` around ``x`` on an ascending grid and the weight of ``hi``."""
    n = len(values)
    if n == 1 or x <= values[0]:
        return 0, 0, 0.0
    if x >= values[-1]:
        return n - 1, n - 1, 0.0
    hi = next(k for k in range(1, n) if values[k] >= x)
    lo = hi - 1
    if values[hi] == x:
        return hi, hi, 0.0
    return lo, hi, (x - values[lo]) / (values[hi] - values[lo])


# --- Level knob (analytic) ----------------------------------------------------------------------


@functools.cache
def _kb():
    from lstmabar.physics.kb import load_kb

    return load_kb()


def level_offset_db(pedal_id: str, level_knob: str, level: float, ref: float) -> float:
    """Output-level change (dB) of turning the level knob from ``ref`` to ``level``, other
    knobs equal (``-inf`` at a fully-off pot).

    In the ts808 and ds1 derivations the level knob enters ``drive.level_db`` only through
    the post-clip scalar gain ``Preset.info["post_gain_db"]``, as the level pot's divider
    fraction times fixed factors, so the offset is the ratio of the pot fractions (tested
    against ``derive``; computed directly because ``derive`` costs ~0.5 s per call).
    """
    pot = _kb()[pedal_id].pots[level_knob]
    return _db(pot.fraction(level)) - _db(pot.fraction(ref))


def _db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 0 else -math.inf


def _level_spec():
    from lstmabar.physics.derive import BLOCK_SPECS

    block, knob = LEVEL_PARAM
    return next(s for s in BLOCK_SPECS[block] if s.name == knob)


def _apply_level_offset(board: FloatBoard, offset_db: float, notes: list[str]) -> None:
    import torch

    spec = _level_spec()
    block, knob = LEVEL_PARAM
    v = float(spec.denormalize(torch.tensor(board[block][knob], dtype=torch.float64))) + offset_db
    c = min(max(v, spec.min), spec.max)
    if c != v:
        reason = (
            "output below Drive's floor (level knob near off)"
            if c == spec.min
            else "output louder than Drive's ceiling"
        )
        notes.append(f"{block}.{knob} {v:.4g} clamped to {c:.4g} {spec.unit}: {reason}")
    board[block][knob] = float(spec.normalize(torch.tensor(c, dtype=torch.float64)))


# --- Fitting ------------------------------------------------------------------------------------


def grid_values(n: int) -> list[float]:
    """``n`` rotations evenly spaced in [0, 1] (one point = 0.5)."""
    if n < 1:
        raise ValueError("grid needs at least one point")
    return [0.5] if n == 1 else [k / (n - 1) for k in range(n)]


def fidelity_config(cfg, quick: bool = False):
    """The :class:`~lstmabar.physics.fidelity.FidelityConfig` for calibration fits."""
    from lstmabar.physics.fidelity import FidelityConfig

    c = cfg.calibration
    kw = dict(
        seed=int(cfg.seed),
        sample_rate=int(cfg.audio.sample_rate),
        pitches_hz=tuple(float(f) for f in c.f0s_hz),
        levels_dbfs=tuple(float(v) for v in c.sine_levels_dbfs),
        whitebox_oversample=int(cfg.render.whitebox_oversample),
        restarts=int(c.restarts),
        steps=int(c.fit_steps),
        mrstft_weight=float(c.get("mrstft_weight", 1.0)),
        volts_per_fs=float(cfg.physics.volts_per_full_scale),
    )
    if quick:
        kw.update(
            pitches_hz=(146.83,),
            levels_dbfs=(-15.0,),
            seconds=0.15,
            settle=0.06,
            fit_seconds=0.08,
            fit_settle=0.03,
            riff_seconds=0.3,
            restarts=0,
            steps=3,
            quick=True,
        )
    return FidelityConfig(**kw)


def _full_board(params, fk, u) -> FloatBoard:
    """Derived ``BoardParams`` with the fitted knobs ``fk`` replaced by ``u`` ``(K,)``."""
    from lstmabar.data.schema import board_to_floats

    board = board_to_floats(params)
    for (b, k), v in zip(fk, u.tolist(), strict=True):
        board[b][k] = float(v)
    board["compressor"][ENABLED] = 0.0
    return board


def calibrate_pedal(
    pedal_id: str,
    gain_values: list[float],
    tone_values: list[float],
    fcfg,
    ctx,
    max_jump: float = 0.25,
    log=None,
) -> tuple[CalibrationTable, dict]:
    """Fit one pedal's grid. Returns the table and a per-pedal report dict."""
    import torch

    from lstmabar.physics.derive import derive, preset_params
    from lstmabar.physics.fidelity import (
        _amps,
        _board,
        _white,
        derived_vector,
        evaluate,
        fit,
        fit_knobs,
        make_signals,
    )

    log = log or functools.partial(print, flush=True)
    t_start = time.perf_counter()
    pedal = _kb()[pedal_id]
    g_knob, t_knob, l_knob = GAIN_KNOB[pedal_id], TONE_KNOB[pedal_id], LEVEL_KNOB[pedal_id]
    level_default = float(pedal.default_knobs()[l_knob])
    sig = make_signals(fcfg)
    board = _board(fcfg)
    fk = fit_knobs(board)
    ng, nt = len(gain_values), len(tone_values)
    boards: list[list[FloatBoard | None]] = [[None] * nt for _ in range(ng)]
    fit_err = [[math.nan] * nt for _ in range(ng)]
    der_err = [[math.nan] * nt for _ in range(ng)]
    cells: list[list[dict | None]] = [[None] * nt for _ in range(ng)]
    gen = torch.Generator().manual_seed(int(fcfg.seed) * 7919 + sum(map(ord, pedal_id)))
    for j, tone in enumerate(tone_values):
        prev = None
        for i, gain in enumerate(gain_values):
            t0 = time.perf_counter()
            knobs = pedal.knobs({g_knob: gain, t_knob: tone})
            white, white_riff = _white(pedal, knobs, sig, fcfg)
            white_amps = _amps(white, sig, fcfg)
            u_d, notes = derived_vector(pedal, knobs, fk, ctx)
            d_eval = evaluate(board, fk, u_d[None], sig, white_amps, white_riff, fcfg)[0]
            if prev is None:
                rand = 0.1 + 0.8 * torch.rand(fcfg.restarts, len(fk), generator=gen)
                u0 = torch.cat([u_d[None], rand])
                inits = ["derived"] + [f"random {k + 1}" for k in range(fcfg.restarts)]
            else:
                u0 = prev[None]
                inits = [f"warm ({g_knob} {gain_values[i - 1]:g})"]
            u_fit, _ = fit(board, fk, u0, sig, white, white_riff, fcfg)
            evals = evaluate(board, fk, u_fit, sig, white_amps, white_riff, fcfg)
            best = min(range(len(evals)), key=lambda k: evals[k]["mean_db"])
            prev = u_fit[best].clone()
            boards[i][j] = _full_board(preset_params(derive(pedal, knobs, ctx)), fk, prev)
            fit_err[i][j] = float(evals[best]["mean_db"])
            der_err[i][j] = float(d_eval["mean_db"])
            secs = time.perf_counter() - t0
            cells[i][j] = {
                "knobs": knobs,
                "fit_db": fit_err[i][j],
                "derived_db": der_err[i][j],
                "fit_all_terms_db": float(evals[best]["all_terms_mean_db"]),
                "derived_all_terms_db": float(d_eval["all_terms_mean_db"]),
                "starts": {inits[k]: float(evals[k]["mean_db"]) for k in range(len(evals))},
                "best_start": inits[best],
                "derive_clamps": [n for n in notes if " clamped to " in n],
                "seconds": secs,
            }
            log(
                f"{pedal_id} {g_knob}={gain:g} {t_knob}={tone:g}: fit {fit_err[i][j]:.2f} dB, "
                f"derived {der_err[i][j]:.2f} dB ({inits[best]}, {secs:.0f}s)"
            )
    elapsed = time.perf_counter() - t_start
    from lstmabar.runs import git_info

    table = CalibrationTable(
        pedal_id=pedal_id,
        gain_knob=g_knob,
        tone_knob=t_knob,
        level_knob=l_knob,
        gain_values=list(gain_values),
        tone_values=list(tone_values),
        boards=boards,  # type: ignore[arg-type]
        fit_error_db=fit_err,
        max_jump=float(max_jump),
        meta={
            "metric": "masked mean harmonic error (dB), physics.fidelity.summarize mean_db",
            "sample_rate": fcfg.sample_rate,
            "sine_levels_dbfs": list(fcfg.levels_dbfs),
            "f0s_hz": list(fcfg.pitches_hz),
            "fit_steps": fcfg.steps,
            "restarts": fcfg.restarts,
            "whitebox_oversample": fcfg.whitebox_oversample,
            "drive_oversample": fcfg.drive_oversample,
            "volts_per_fs": fcfg.volts_per_fs,
            "seed": fcfg.seed,
            "quick": fcfg.quick,
            "fit_knobs": [f"{b}.{k}" for b, k in fk],
            "level_offset": "derive info post_gain_db(level) - post_gain_db(level_default)",
            "git": git_info(),
            "seconds": elapsed,
        },
        level_default=level_default,
        derived_error_db=der_err,
    )
    return table, pedal_report(table, cells, elapsed)


def pedal_report(table: CalibrationTable, cells, elapsed: float) -> dict:
    """Grid errors, neighbour jumps and fallback cells of a fitted table."""
    ng, nt = len(table.gain_values), len(table.tone_values)
    b = table.boards
    gain_jumps = [[board_jump(b[i][j], b[i + 1][j]) for j in range(nt)] for i in range(ng - 1)]
    tone_jumps = [[board_jump(b[i][j], b[i][j + 1]) for j in range(nt - 1)] for i in range(ng)]
    cell_rows = []
    for i in range(max(ng - 1, 1)):
        for j in range(max(nt - 1, 1)):
            i1, j1 = min(i + 1, ng - 1), min(j + 1, nt - 1)
            jump = table.cell_jump(i, j, i1, j1)
            cell_rows.append(
                {
                    "gain": [table.gain_values[i], table.gain_values[i1]],
                    "tone": [table.tone_values[j], table.tone_values[j1]],
                    "max_jump": jump,
                    "fallback": jump > table.max_jump,
                }
            )
    fit = [v for row in table.fit_error_db for v in row]
    der = [v for row in table.derived_error_db for v in row]
    return {
        "pedal_id": table.pedal_id,
        "gain_knob": table.gain_knob,
        "tone_knob": table.tone_knob,
        "level_knob": table.level_knob,
        "level_default": table.level_default,
        "gain_values": table.gain_values,
        "tone_values": table.tone_values,
        "fit_error_db": table.fit_error_db,
        "derived_error_db": table.derived_error_db,
        "fit_mean_db": sum(fit) / len(fit),
        "derived_mean_db": sum(der) / len(der),
        "fit_worse_than_derived": sum(f > d for f, d in zip(fit, der, strict=True)),
        "max_gain_jump": max((v for r in gain_jumps for v in r), default=0.0),
        "max_tone_jump": max((v for r in tone_jumps for v in r), default=0.0),
        "gain_jumps": gain_jumps,
        "tone_jumps": tone_jumps,
        "cells": cell_rows,
        "fallback_cells": sum(c["fallback"] for c in cell_rows),
        "grid": cells,
        "seconds": elapsed,
    }


# --- Report -------------------------------------------------------------------------------------


def format_report(rep: dict) -> str:
    c = rep["config"]
    lines = [
        "# White-box calibration tables (P4 R4)",
        "",
        "Per white-box pedal, the grey-box (`Drive` + `EQ3`) label fitted to the circuit "
        "simulation on a gain × tone knob grid (`data/calibration/<pedal>.json`), against "
        "the derived (unfitted) label at the same settings. Metric: the fidelity gate's "
        "masked mean harmonic error in dB (`physics.fidelity`, P2 decision 6), on the "
        "calibration inputs below. Fits are selected in-sample (an expressiveness bound on "
        "these inputs, as in the fidelity check).",
        "",
        f"Generated by `{rep['command']}`; runtime {rep['seconds']:.0f} s on CPU "
        f"({rep['torch_threads']} torch threads); git {rep['git'].get('sha') or 'n/a'}"
        + (" (dirty)" if rep["git"].get("dirty") else "")
        + ".",
        "",
        "## Procedure",
        "",
        f"- Inputs: sines at {', '.join(f'{f:g}' for f in c['f0s_hz'])} Hz × "
        f"{', '.join(f'{v:g}' for v in c['sine_levels_dbfs'])} dBFS peak at "
        f"{c['sample_rate']} Hz (white-box {c['whitebox_oversample']}x oversampled, grey-box "
        f"Drive {c['drive_oversample']}x), {c['settle']:g} s settle + {c['seconds']:g} s "
        f"analysed (fit on the first {c['fit_seconds']:g} s).",
        f"- Grid: {c['gain_points']} gain × {c['tone_points']} tone rotations evenly spaced "
        "in [0, 1]; level knob at its pot default during fits (applied analytically at "
        "lookup from the derivation's post-clip gain).",
        f"- Fit: the 11 Drive + EQ3 knobs, {c['fit_steps']} Adam steps per grid point "
        "(`physics.fidelity.fit`, lr 0.05 cosine), loss = differentiable harmonic metric"
        + (
            f" + {c['mrstft_weight']:g} × MR-STFT on a {c['riff_seconds']:g} s riff"
            if c["mrstft_weight"]
            else " only (the fidelity check's riff MR-STFT term is off: on a 1 s riff at "
            "44.1 kHz it cost ~10x the sines per step)"
        )
        + f". Gain index 0: derived start + {c['restarts']} random "
        "start(s), best kept; every next gain point is warm-started from the previous gain "
        "point's fit at the same tone. Unfitted knobs come from the derived preset; "
        "compressor off.",
        f"- Lookup: bilinear interpolation of normalized params over the gain × tone cell; "
        f"if any two of the cell's corners differ by more than max_jump = {c['max_jump']:g} "
        "(max abs normalized difference over knobs), the nearest grid point is used instead "
        "(`calibrated_nearest`). Gates are copied.",
        "",
        "## Summary",
        "",
        "| pedal | calibrated mean (dB) | derived mean (dB) | grid points where fit > derived | "
        "max gain-axis jump | max tone-axis jump | fallback cells | time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in rep["pedals"].values():
        n = len(p["gain_values"]) * len(p["tone_values"])
        lines.append(
            f"| {p['pedal_id']} | **{p['fit_mean_db']:.2f}** | {p['derived_mean_db']:.2f} | "
            f"{p['fit_worse_than_derived']}/{n} | {p['max_gain_jump']:.3f} | "
            f"{p['max_tone_jump']:.3f} | {p['fallback_cells']}/{len(p['cells'])} | "
            f"{p['seconds']:.0f} |"
        )
    for p in rep["pedals"].values():
        g, t = p["gain_knob"], p["tone_knob"]
        lines += [
            "",
            f"## `{p['pedal_id']}`",
            "",
            f"Fit error, calibrated / derived (dB), rows = {g}, columns = {t}:",
            "",
            f"| {g} \\ {t} | " + " | ".join(f"{v:g}" for v in p["tone_values"]) + " |",
            "|---|" + "---|" * len(p["tone_values"]),
        ]
        for i, gv in enumerate(p["gain_values"]):
            cells = " | ".join(
                f"**{p['fit_error_db'][i][j]:.2f}** / {p['derived_error_db'][i][j]:.2f}"
                for j in range(len(p["tone_values"]))
            )
            lines.append(f"| {gv:g} | {cells} |")
        lines += [
            "",
            "Neighbour jumps (max abs normalized difference) along the gain axis "
            "(rows = lower gain point of the pair):",
            "",
            f"| {g} pair \\ {t} | " + " | ".join(f"{v:g}" for v in p["tone_values"]) + " |",
            "|---|" + "---|" * len(p["tone_values"]),
        ]
        for i, row in enumerate(p["gain_jumps"]):
            pair = f"{p['gain_values'][i]:g}–{p['gain_values'][i + 1]:g}"
            lines.append(f"| {pair} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
        if p["tone_jumps"] and p["tone_jumps"][0]:
            lines += [
                "",
                "Along the tone axis (columns = tone pair):",
                "",
                f"| {g} \\ {t} pair | "
                + " | ".join(
                    f"{p['tone_values'][j]:g}–{p['tone_values'][j + 1]:g}"
                    for j in range(len(p["tone_values"]) - 1)
                )
                + " |",
                "|---|" + "---|" * (len(p["tone_values"]) - 1),
            ]
            for i, row in enumerate(p["tone_jumps"]):
                lines.append(
                    f"| {p['gain_values'][i]:g} | " + " | ".join(f"{v:.3f}" for v in row) + " |"
                )
        fb = [cl for cl in p["cells"] if cl["fallback"]]
        lines += [
            "",
            f"Fallback cells (nearest grid point): {len(fb)} of {len(p['cells'])}"
            + (
                ": "
                + "; ".join(
                    f"{g} {cl['gain'][0]:g}–{cl['gain'][1]:g} × {t} "
                    f"{cl['tone'][0]:g}–{cl['tone'][1]:g} (jump {cl['max_jump']:.3f})"
                    for cl in fb
                )
                if fb
                else ""
            )
            + ".",
            "",
            "Winning start at gain index 0: "
            + "; ".join(
                f"{t} {p['tone_values'][j]:g}: {p['grid'][0][j]['best_start']} ("
                + ", ".join(f"{k} {v:.2f}" for k, v in p["grid"][0][j]["starts"].items())
                + ")"
                for j in range(len(p["tone_values"]))
            )
            + ".",
        ]
    lines.append("")
    return "\n".join(lines)


def _config_subset(cfg, fcfg, quick: bool) -> dict:
    c = cfg.calibration
    return {
        "gain_points": int(c.gain_points) if not quick else 2,
        "tone_points": int(c.tone_points) if not quick else 2,
        "max_jump": float(c.max_jump),
        "f0s_hz": list(fcfg.pitches_hz),
        "sine_levels_dbfs": list(fcfg.levels_dbfs),
        "fit_steps": fcfg.steps,
        "restarts": fcfg.restarts,
        "mrstft_weight": fcfg.mrstft_weight,
        "fit_seconds": fcfg.fit_seconds,
        "sample_rate": fcfg.sample_rate,
        "whitebox_oversample": fcfg.whitebox_oversample,
        "drive_oversample": fcfg.drive_oversample,
        "settle": fcfg.settle,
        "seconds": fcfg.seconds,
        "riff_seconds": fcfg.riff_seconds,
        "seed": fcfg.seed,
        "quick": quick,
    }


def run(cfg, args) -> int:
    import torch

    from lstmabar.data.layout import DataLayout
    from lstmabar.physics.derive import DeriveContext
    from lstmabar.runs import git_info

    quick = bool(getattr(args, "quick", False))
    pedals = getattr(args, "pedals", None) or list(cfg.render.whitebox_pedals)
    unknown = [p for p in pedals if p not in GAIN_KNOB]
    if unknown:
        raise SystemExit(f"no calibration grid for pedals {unknown}; known: {list(GAIN_KNOB)}")
    fcfg = fidelity_config(cfg, quick)
    sub = _config_subset(cfg, fcfg, quick)
    ctx = DeriveContext.from_config(cfg)
    layout = DataLayout.from_config(cfg)
    report_dir = Path(str(cfg.calibration.get("report_dir", "reports")))
    torch.manual_seed(int(cfg.seed))
    t0 = time.perf_counter()
    out = {}
    for pid in pedals:
        table, rep = calibrate_pedal(
            pid,
            grid_values(sub["gain_points"]),
            grid_values(sub["tone_points"]),
            fcfg,
            ctx,
            max_jump=sub["max_jump"],
        )
        table.meta["config"] = sub
        path = layout.calibration_table(pid)
        table.save(path)
        print(f"{pid}: wrote {path.name} ({rep['seconds']:.0f}s)", flush=True)
        out[pid] = rep
    cmd = "lstmabar data calibrate" + (" --quick" if quick else "")
    if getattr(args, "pedals", None):
        cmd += " --pedals " + " ".join(pedals)
    rep = {
        "command": cmd,
        "config": sub,
        "git": git_info(),
        "torch_threads": torch.get_num_threads(),
        "seconds": time.perf_counter() - t0,
        "pedals": out,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{REPORT_NAME}.json").write_text(
        json.dumps(rep, indent=1) + "\n", encoding="utf-8"
    )
    (report_dir / f"{REPORT_NAME}.md").write_text(format_report(rep), encoding="utf-8")
    return 0


__all__ = [
    "CalibrationTable",
    "Label",
    "board_jump",
    "calibrate_pedal",
    "grid_values",
    "level_offset_db",
    "run",
]
