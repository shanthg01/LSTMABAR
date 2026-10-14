"""``test_real`` sim-to-real gap (D11): ``lstmabar eval sim-real`` -> ``reports/sim_real.*``.

What is measured, per group (``dataset:device``):

- **KB items** (``RealItem.kb_pedal`` set: EGFxSet TubeScreamer -> ``ts808``, RAT -> ``rat``, the
  unseen-pedal holdout): the item's dry note is resampled 48 -> 44.1 kHz, scaled to each assumed
  input peak in ``eval.sim_real.input_peak_sweep_dbfs`` (the true input level is unknown: the
  datasets are level-normalized) and rendered through the KB pedal at the item's ``settings``:
  white-box ``ts808`` (``physics.whitebox.simulate``, as ``data.render.RenderContext``), and the
  derived grey-box board (``physics.derive`` -> ``default_pedalboard``) for both pedals. Each
  render is loudness-matched (``audio.loudness_match``) to the real wet (resampled to 44.1 kHz)
  and scored with MR-STFT (``dsp.losses.multi_resolution_stft_loss``) and descriptor errors
  (|Δ centroid| octaves, |Δ odd/even| dB, |Δ HNR| dB; the harmonic terms use a pYIN track of
  the dry and are NaN when its voiced fraction is below ``analysis.min_voiced_fraction``).
  Every sweep level is reported, plus the per-item best level by MR-STFT ("best of sweep",
  which is optimistic: the level is chosen on the very item it is scored on).
  Two references make the numbers interpretable: the dry note itself vs the real wet (no
  pedal at all) and the grey-box at the pedal's **default** knobs (wrong settings, right
  pedal) at the same sweep level.
- **Non-KB items** (EGFxSet BluesDriver, pOD-set pedals): family-level descriptor *shifts*
  (wet - dry) of the real recordings vs the synthetic v1 ``train`` renders of KB pedals of the
  same family (aggregate only). pOD-set uses only the mixed-material ``a`` files (a fixed
  segment, :data:`POD_SEGMENT_S`); the sine-sweep files are skipped.
- **Knob direction**: EGFxSet recorded each unit at a single setting, so knob direction cannot
  be checked within the KB groups. The report instead (1) checks per item that the KB render
  shifts the descriptors (centroid, crest factor) in the same direction as the real pedal
  does relative to the dry, (2) sweeps the KB knobs on a few dry notes (``ts808`` tone/drive,
  ``rat`` filter/distortion) to confirm the documented direction, and (3) compares the
  Spearman correlation of knob and descriptor shift on the pOD-set ``tubedreamer`` gain x tone
  grid (a TS-derived pedal, not a TS808; internal, aggregate only) with the ``ts808``
  grey-box rendered at the same grid (gain -> drive, tone -> tone).

``internal_only`` items (pOD-set) appear in aggregates only, never as per-item rows. Items are
subsampled deterministically per group (``eval.sim_real.max_items_per_group``, by sha256 of the
item id); ``--limit`` caps each group further.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from lstmabar.eval.stats import CI

POD_SEGMENT_S = (60.0, 10.0)
"""(offset, duration) in seconds of the pOD-set ``a`` files analysed (~6 min each; the same
input segment for every pedal and setting)."""
DRY_CACHE = 16  # dry clips (+ pYIN tracks) kept per worker
LONG_FILE_S = 30.0  # files longer than this are cropped to POD_SEGMENT_S
KB_ENGINES: dict[str, tuple[str, ...]] = {"ts808": ("whitebox", "greybox"), "rat": ("greybox",)}
REFERENCE_DEFAULT = "greybox_default"
REFERENCE_DRY = "dry"
ERROR_TERMS: tuple[str, ...] = ("mrstft", "centroid_oct", "odd_even_db", "hnr_db")
SHIFT_TERMS: tuple[str, ...] = ("centroid_oct", "odd_even_db", "hnr_db", "crest_db")
DIRECTION_TERMS: tuple[str, ...] = ("centroid_oct", "crest_db")
TUBEDREAMER = "tubedreamer"
KNOB_SWEEP = {"ts808": ("drive", "tone"), "rat": ("distortion", "filter")}
KNOB_SWEEP_POSITIONS = (0.0, 0.5, 1.0)
KNOB_SWEEP_NOTES = 3
KNOB_SWEEP_LEVEL_DBFS = -12.0


# ============================================================================ small helpers


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> CI:
    """NaN-aware percentile bootstrap CI of the mean over items.

    TODO(P5 wave 1a): switch to ``lstmabar.eval.stats.bootstrap_ci`` once it is implemented on
    ``main`` (same contract: NaNs dropped, ``n < 2`` gives ``lo = hi = mean``)."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    m = float(v.mean())
    if n < 2:
        return CI(m, m, m, n)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    step = max(1, 250_000 // n)  # bound memory (~2 MB): resample in chunks
    for s in range(0, n_resamples, step):
        k = min(step, n_resamples - s)
        means[s : s + k] = v[rng.integers(0, n, size=(k, n))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(m, float(lo), float(hi), n)


def subsample_key(item_id: str) -> str:
    return hashlib.sha256(item_id.encode("utf-8")).hexdigest()


def group_of(item) -> str:
    return f"{item.dataset}:{item.device}"


def pod_kind(item) -> str | None:
    """pOD-set file kind (``a``, ``s0``, ...) from the dry file name, else ``None``."""
    if item.dataset != "podset":
        return None
    return PurePosixPath(item.dry_path).name.split("_", 1)[0]


def select_items(items: Sequence, max_per_group: int | None, limit: int | None) -> list:
    """Deterministic per-group subsample (lowest sha256 of the item id first). pOD-set items
    other than the mixed-material ``a`` files are dropped (sine sweeps)."""
    groups: dict[str, list] = defaultdict(list)
    for r in items:
        k = pod_kind(r.item)
        if k is not None and k != "a":
            continue
        groups[group_of(r.item)].append(r)
    caps = [c for c in (max_per_group, limit) if c is not None]
    cap = min(caps) if caps else None
    out = []
    for g in sorted(groups):
        rows = sorted(groups[g], key=lambda r: subsample_key(r.item.item_id))
        out.extend(rows[:cap] if cap is not None else rows)
    return out


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _log2_ratio(a, b) -> float:
    a, b = _f(a), _f(b)
    return math.log2(a / b) if a > 0 and b > 0 else float("nan")


def descriptor_errors(pred: Mapping[str, Any], real: Mapping[str, Any]) -> dict[str, float]:
    """|Δ centroid| (octaves), |Δ odd/even| dB, |Δ HNR| dB; NaN when a value is missing."""
    return {
        "centroid_oct": abs(_log2_ratio(pred.get("centroid_hz"), real.get("centroid_hz"))),
        "odd_even_db": abs(_f(pred.get("odd_even_db")) - _f(real.get("odd_even_db"))),
        "hnr_db": abs(_f(pred.get("hnr_db")) - _f(real.get("hnr_db"))),
    }


def descriptor_shift(wet: Mapping[str, Any], dry: Mapping[str, Any]) -> dict[str, float]:
    """Signed wet - dry shifts (:data:`SHIFT_TERMS`)."""
    return {
        "centroid_oct": _log2_ratio(wet.get("centroid_hz"), dry.get("centroid_hz")),
        "odd_even_db": _f(wet.get("odd_even_db")) - _f(dry.get("odd_even_db")),
        "hnr_db": _f(wet.get("hnr_db")) - _f(dry.get("hnr_db")),
        "crest_db": _f(wet.get("crest_factor_db")) - _f(dry.get("crest_factor_db")),
    }


def mrstft(pred: np.ndarray, target: np.ndarray, fft_sizes: Sequence[int]) -> float:
    import torch

    from lstmabar.dsp.losses import multi_resolution_stft_loss

    n = min(len(pred), len(target))
    p = torch.from_numpy(np.asarray(pred[:n], dtype=np.float32))[None]
    t = torch.from_numpy(np.asarray(target[:n], dtype=np.float32))[None]
    with torch.no_grad():
        return float(multi_resolution_stft_loss(p, t, tuple(int(f) for f in fft_sizes)))


def _sign_agree(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    return 1.0 if np.sign(a) == np.sign(b) else 0.0


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


# ============================================================================ worker state


class _Worker:
    """Per-process state: config, render context, a one-entry dry cache."""

    def __init__(self, cfg_dict: dict) -> None:
        from omegaconf import OmegaConf

        from lstmabar.data.render import Options, RenderContext, librosa_available

        self.cfg = OmegaConf.create(cfg_dict)
        self.pitch = librosa_available()
        self.ctx = RenderContext(self.cfg, Options(allow_derived_labels=True, pitch=self.pitch))
        self.sr = int(self.cfg.eval.sim_real.resample_to)
        if self.sr != self.ctx.sr:
            raise ValueError(
                f"eval.sim_real.resample_to ({self.sr}) must equal audio.sample_rate "
                f"({self.ctx.sr}): the pedalboard runs at audio.sample_rate"
            )
        self.levels = [float(x) for x in self.cfg.eval.sim_real.input_peak_sweep_dbfs]
        self.fft_sizes = [int(x) for x in self.cfg.eval.metrics.mrstft_fft_sizes]
        self._dry: dict[str, tuple] = {}
        self._boards: dict[tuple, dict] = {}

    # --- audio ---

    def load(self, path: Path) -> np.ndarray:
        import soundfile as sf

        from lstmabar.audio import resample

        info = sf.info(str(path))
        if info.frames / info.samplerate > LONG_FILE_S:
            off, dur = POD_SEGMENT_S
            start = int(off * info.samplerate)
            x, sr = sf.read(str(path), start=start, frames=int(dur * info.samplerate))
        else:
            x, sr = sf.read(str(path))
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return resample(x, int(sr), self.sr).astype(np.float64)

    def dry(self, path: Path):
        """``(dry at 44.1 kHz, f0 track or None, harmonic?, descriptors, voiced fraction)``."""
        key = str(path)
        if key not in self._dry:
            from lstmabar.analysis.harmonics import track_f0
            from lstmabar.data.render import voiced_fraction

            x = self.load(path)
            tr = track_f0(x, self.sr) if self.pitch else None
            vf = voiced_fraction(tr)
            harmonic = vf is not None and vf >= self.ctx.min_vf
            desc = self.describe(x, tr, harmonic)
            if len(self._dry) >= DRY_CACHE:
                self._dry.pop(next(iter(self._dry)))
            self._dry[key] = (x, tr, harmonic, desc, vf)
        return self._dry[key]

    def describe(self, y: np.ndarray, track, harmonic: bool) -> dict[str, float]:
        from lstmabar.data.render import analyse

        desc, _ = analyse(
            y, self.sr, track, harmonic and track is not None, self.ctx.n_harmonics,
            self.ctx.max_frames,
        )  # fmt: skip
        return desc

    # --- rendering ---

    def greybox(self, pedal_id: str, knobs: Mapping[str, float] | None, x: np.ndarray):
        from lstmabar.data.schema import board_to_floats
        from lstmabar.physics.derive import derive, preset_params

        pedal = self.ctx.kb[pedal_id]
        full = pedal.knobs(knobs)
        key = (pedal_id, tuple(sorted(full.items())))
        if key not in self._boards:  # derive (tone fit) is ~1 s; settings repeat per group
            self._boards[key] = board_to_floats(
                preset_params(derive(pedal, full, self.ctx.derive_ctx))
            )
        return self.ctx.render_board(self._boards[key], x)

    def render(self, engine: str, pedal_id: str, knobs, x: np.ndarray) -> np.ndarray:
        if engine == "whitebox":
            return np.asarray(self.ctx.render_whitebox(self.ctx.kb[pedal_id], knobs, x))
        return self.greybox(pedal_id, knobs, x)

    def score(self, y: np.ndarray, real: np.ndarray, real_desc, track, harmonic) -> dict:
        from lstmabar.audio import loudness_match

        y = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(y)):
            return dict.fromkeys(ERROR_TERMS, float("nan")) | {"_shift_desc": {}}
        y = loudness_match(y, real)
        desc = self.describe(y, track, harmonic)
        return (
            {"mrstft": mrstft(y, real, self.fft_sizes)}
            | descriptor_errors(desc, real_desc)
            | {"_shift_desc": desc}
        )


_W: _Worker | None = None


def _init(cfg_dict: dict) -> None:
    import torch

    torch.set_num_threads(1)  # measured: 1 thread per worker is fastest on this machine
    global _W
    _W = _Worker(cfg_dict)


def _worker() -> _Worker:
    if _W is None:
        raise RuntimeError("worker not initialised")
    return _W


# ============================================================================ per-item jobs


def eval_kb_item(job: dict) -> dict:
    """One KB item: every engine x sweep level, the default-knob grey-box, the dry reference."""
    w = _worker()
    pedal = job["kb_pedal"]
    x, track, harmonic, dry_desc, vf = w.dry(Path(job["dry_path"]))
    real = w.load(Path(job["wet_path"]))
    n = min(len(x), len(real))
    x, real = x[:n], real[:n]
    real_desc = w.describe(real, track, harmonic)
    real_shift = descriptor_shift(real_desc, dry_desc)
    knobs = w.ctx.kb[pedal].knobs(job["settings"])

    def finish(s: dict) -> dict:
        d = s.pop("_shift_desc")
        if d:
            sh = descriptor_shift(d, dry_desc)
            for t in DIRECTION_TERMS:
                s[f"agree_{t}"] = _sign_agree(sh[t], real_shift[t])
        return s

    from lstmabar.data import levels

    out: dict[str, Any] = {
        "item_id": job["item_id"],
        "group": job["group"],
        "family": job["family"],
        "kb_pedal": pedal,
        "internal_only": job["internal_only"],
        "voiced_fraction": vf,
        "harmonic": bool(harmonic),
        "knobs": knobs,
        "real_shift": real_shift,
        "dry": finish(w.score(x, real, real_desc, track, harmonic)),
        "levels": {},
    }
    for lv in w.levels:
        xin = levels.apply_input_level(x, lv)
        row = {}
        for eng in KB_ENGINES.get(pedal, ("greybox",)):
            row[eng] = finish(w.score(w.render(eng, pedal, knobs, xin), real, real_desc, track,
                                      harmonic))  # fmt: skip
        y_def = w.greybox(pedal, None, xin)
        row[REFERENCE_DEFAULT] = finish(w.score(y_def, real, real_desc, track, harmonic))
        out["levels"][f"{lv:g}"] = row
    return out


def eval_shift_item(job: dict) -> dict:
    """Any item: real wet - dry descriptor shifts (family comparison), and for the pOD-set
    tubedreamer grid also the ts808 grey-box shifts at each sweep level (direction check)."""
    w = _worker()
    x, track, harmonic, dry_desc, vf = w.dry(Path(job["dry_path"]))
    real = w.load(Path(job["wet_path"]))
    n = min(len(x), len(real))
    real_desc = w.describe(real[:n], track, harmonic)
    out = {
        "item_id": job["item_id"],
        "group": job["group"],
        "family": job["family"],
        "internal_only": job["internal_only"],
        "settings": job["settings"],
        "harmonic": bool(harmonic),
        "real_shift": descriptor_shift(real_desc, dry_desc),
    }
    if job.get("ts_grid"):
        from lstmabar.data import levels

        s = job["settings"]
        knobs = {"drive": float(s.get("gain", 0.5)), "tone": float(s.get("tone", 0.5))}
        out["kb_shift"] = {}
        for lv in w.levels:
            y = w.greybox("ts808", knobs, levels.apply_input_level(x[:n], lv))
            d = w.describe(np.asarray(y, dtype=np.float64), track, harmonic)
            out["kb_shift"][f"{lv:g}"] = descriptor_shift(d, dry_desc)
    return out


def eval_knob_sweep(job: dict) -> dict:
    """KB direction sanity on one dry note: descriptor values at knob positions 0 / 0.5 / 1
    (others at the item's settings)."""
    w = _worker()
    from lstmabar.data import levels

    x, track, harmonic, dry_desc, _ = w.dry(Path(job["dry_path"]))
    xin = levels.apply_input_level(x, KNOB_SWEEP_LEVEL_DBFS)
    pedal = job["kb_pedal"]
    base = w.ctx.kb[pedal].knobs(job["settings"])
    out = {"group": job["group"], "kb_pedal": pedal, "sweeps": {}}
    for knob in KNOB_SWEEP[pedal]:
        rows = []
        for pos in KNOB_SWEEP_POSITIONS:
            y = w.greybox(pedal, base | {knob: pos}, xin)
            d = descriptor_shift(w.describe(np.asarray(y, float), track, False), dry_desc)
            rows.append({"pos": pos, "centroid_oct": d["centroid_oct"], "crest_db": d["crest_db"]})
        out["sweeps"][knob] = rows
    return out


# ============================================================================ aggregation


def _ci(values, cfg, seed_offset: int = 0) -> dict:
    st = cfg.eval.stats
    return bootstrap_ci(
        values, int(st.n_resamples), int(cfg.eval.seed) + seed_offset, float(st.alpha)
    ).to_dict()


def _best(levels_row: Mapping[str, Mapping[str, Mapping[str, float]]], engine: str):
    """``(level, scores)`` of the sweep level with the lowest MR-STFT for ``engine``."""
    cands = [(lv, r[engine]) for lv, r in levels_row.items() if math.isfinite(r[engine]["mrstft"])]
    if not cands:
        return None, None
    return min(cands, key=lambda c: c[1]["mrstft"])


def aggregate_kb(rows: Sequence[dict], cfg) -> dict:
    """Per group: per engine x level (and best-of-sweep) means with CIs, plus references."""
    out: dict[str, Any] = {}
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    for g, rs in sorted(by_group.items()):
        pedal = rs[0]["kb_pedal"]
        engines = [*KB_ENGINES.get(pedal, ("greybox",)), REFERENCE_DEFAULT]
        lvls = list(rs[0]["levels"])
        terms = [*ERROR_TERMS, *(f"agree_{t}" for t in DIRECTION_TERMS)]
        cells: list[dict] = []

        def cell(engine: str, level: str, vals: list[dict], cells=cells, terms=terms) -> None:
            c = {"engine": engine, "level": level}
            for k, t in enumerate(terms):
                c[t] = _ci([v.get(t, float("nan")) for v in vals], cfg, k)
            cells.append(c)

        cell(REFERENCE_DRY, "-", [r["dry"] for r in rs])
        best_levels: dict[str, dict[str, int]] = {}
        for eng in engines:
            for lv in lvls:
                cell(eng, lv, [r["levels"][lv][eng] for r in rs])
            picks = [_best(r["levels"], eng) for r in rs]
            cell(eng, "best", [p[1] for p in picks if p[1] is not None])
            hist: dict[str, int] = defaultdict(int)
            for p in picks:
                if p[0] is not None:
                    hist[p[0]] += 1
            best_levels[eng] = dict(sorted(hist.items(), key=lambda kv: float(kv[0])))
        out[g] = {
            "kb_pedal": pedal,
            "unseen_pedal": pedal == "rat",
            "n_items": len(rs),
            "n_harmonic": int(sum(r["harmonic"] for r in rs)),
            "settings": rs[0]["knobs"],
            "internal_only": any(r["internal_only"] for r in rs),
            "cells": cells,
            "best_level_counts": best_levels,
        }
    return out


def aggregate_shifts(rows: Sequence[dict], synth: Mapping[str, Mapping[str, list]], cfg) -> dict:
    """Real wet - dry shift distributions per group vs the synthetic family/pedal shifts."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    groups = {}
    for g, rs in sorted(by_group.items()):
        fam = rs[0]["family"]
        terms = {}
        for k, t in enumerate(SHIFT_TERMS):
            vals = np.asarray([r["real_shift"][t] for r in rs], float)
            fin = vals[np.isfinite(vals)]
            terms[t] = {
                "real": _ci(vals, cfg, k)
                | {"median": float(np.median(fin)) if fin.size else float("nan")},
                "synthetic_family": _summ(synth.get(f"family:{fam}", {}).get(t, []), cfg, k),
            }
        groups[g] = {
            "family": fam,
            "n_items": len(rs),
            "n_harmonic": int(sum(r["harmonic"] for r in rs)),
            "internal_only": any(r["internal_only"] for r in rs),
            "terms": terms,
        }
    return groups


def _summ(vals: Sequence[float], cfg, k: int) -> dict:
    v = np.asarray(vals, float)
    fin = v[np.isfinite(v)]
    return _ci(fin, cfg, 100 + k) | {"median": float(np.median(fin)) if fin.size else float("nan")}


def aggregate_ts_grid(rows: Sequence[dict], levels_: Sequence[str]) -> dict:
    """Spearman rho of knob vs descriptor shift on the tubedreamer grid: real vs ts808 KB."""
    rows = [r for r in rows if "kb_shift" in r]
    if not rows:
        return {}
    gain = [float(r["settings"].get("gain", float("nan"))) for r in rows]
    tone = [float(r["settings"].get("tone", float("nan"))) for r in rows]
    out: dict[str, Any] = {"n_items": len(rows), "pairs": {}}
    pairs = {
        "tone~centroid_oct": (tone, "centroid_oct"),
        "gain~centroid_oct": (gain, "centroid_oct"),
        "gain~crest_db": (gain, "crest_db"),
    }
    for name, (knob, term) in pairs.items():
        out["pairs"][name] = {
            "real": _spearman(knob, [r["real_shift"][term] for r in rows]),
            "kb": {lv: _spearman(knob, [r["kb_shift"][lv][term] for r in rows]) for lv in levels_},
        }
    return out


def synthetic_shifts(cfg) -> dict[str, dict[str, list[float]]]:
    """wet - dry descriptor shifts of v1 ``train`` KB-pedal renders, by family and by pedal."""
    from lstmabar.data.layout import DataLayout
    from lstmabar.data.render import load_pedals
    from lstmabar.eval.data import iter_items

    kb = load_pedals(cfg)
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    layout = DataLayout.from_config(cfg)
    for it in iter_items(layout, ["train"]):  # streamed: ~24.5k records
        if it.scenario != "pedal" or it.pedal_id not in kb:
            continue
        sh = descriptor_shift(it.descriptors_wet, it.descriptors_dry)
        for key in (f"family:{kb[it.pedal_id].family}", f"pedal:{it.pedal_id}"):
            for t in SHIFT_TERMS:
                out[key][t].append(sh[t])
    return {k: dict(v) for k, v in out.items()}


# ============================================================================ entry points


def _jobs(items: Sequence, kb_ids: set[str]) -> tuple[list[dict], list[dict], list[dict]]:
    kb_jobs, shift_jobs, sweep_jobs = [], [], []
    seen_sweep: dict[str, int] = defaultdict(int)
    for r in items:
        it = r.item
        base = {
            "item_id": it.item_id,
            "group": group_of(it),
            "family": it.family,
            "internal_only": bool(it.internal_only),
            "settings": dict(it.settings),
            "dry_path": str(r.dry_path),
            "wet_path": str(r.wet_path),
        }
        if r.kb_pedal in kb_ids:
            kb_jobs.append(base | {"kb_pedal": r.kb_pedal})
            if r.kb_pedal in KNOB_SWEEP and seen_sweep[r.kb_pedal] < KNOB_SWEEP_NOTES:
                seen_sweep[r.kb_pedal] += 1
                sweep_jobs.append(base | {"kb_pedal": r.kb_pedal})
        else:  # KB items get their real shifts from the KB job
            ts_grid = it.dataset == "podset" and it.device == TUBEDREAMER
            shift_jobs.append(base | {"ts_grid": ts_grid})
    return kb_jobs, shift_jobs, sweep_jobs


TASKS = {"kb": "eval_kb_item", "shift": "eval_shift_item", "sweep": "eval_knob_sweep"}


MEMORY_RETRIES = 4  # the machine is shared: transient MemoryErrors are retried after a pause


def _run_job(job: dict) -> dict:
    import gc

    fn = globals()[TASKS[job["task"]]]
    for attempt in range(MEMORY_RETRIES + 1):
        try:
            return fn(job) | {"task": job["task"]}
        except MemoryError:
            if attempt == MEMORY_RETRIES:
                raise
            gc.collect()
            time.sleep(20.0 * (attempt + 1))
    raise AssertionError("unreachable")


def _map(jobs: list[dict], cfg_dict: dict, workers: int) -> list[dict]:
    """Run every job in one pool (worker start-up, numba and torch imports are paid once)."""
    if not jobs:
        return []
    t0 = time.perf_counter()
    out: list[dict] = []
    if workers <= 1:
        _init(cfg_dict)
        for i, j in enumerate(jobs):
            out.append(_run_job(j))
            _progress("items", i + 1, len(jobs), t0)
        return out
    import multiprocessing

    with ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init,
        initargs=(cfg_dict,),
    ) as ex:  # fmt: skip
        for i, res in enumerate(ex.map(_run_job, jobs, chunksize=2)):
            out.append(res)
            _progress("items", i + 1, len(jobs), t0)
    return out


def _progress(label: str, i: int, n: int, t0: float) -> None:
    if i == n or i % 25 == 0:
        print(f"  {label}: {i}/{n} ({time.perf_counter() - t0:.0f} s)", flush=True)


def evaluate(cfg, limit: int | None = None, workers: int = 1) -> dict[str, Any]:
    """The report's ``test_real`` section (aggregate-only where any item is internal).

    ``workers`` (optional; wave 1 addition) is capped at 2 by :func:`run`."""
    from omegaconf import OmegaConf

    from lstmabar.data.render import librosa_available, load_pedals
    from lstmabar.eval.data import load_test_real

    sc = cfg.eval.sim_real
    all_items = load_test_real(cfg)
    max_per = sc.get("max_items_per_group")
    items = select_items(all_items, int(max_per) if max_per is not None else None, limit)
    kb_ids = set(load_pedals(cfg)) & set(KB_ENGINES)
    kb_jobs, shift_jobs, sweep_jobs = _jobs(items, kb_ids)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    jobs = (
        [j | {"task": "kb"} for j in kb_jobs]
        + [j | {"task": "sweep"} for j in sweep_jobs]
        + [j | {"task": "shift"} for j in shift_jobs]
    )
    print(f"sim-real: {len(kb_jobs)} KB items, {len(shift_jobs)} other items, "
          f"{len(sweep_jobs)} knob sweeps; {workers} worker(s)", flush=True)  # fmt: skip
    rows = _map(jobs, cfg_dict, workers)
    kb_rows = [r for r in rows if r["task"] == "kb"]
    shift_rows = [r for r in rows if r["task"] == "shift"]
    sweep_rows = [r for r in rows if r["task"] == "sweep"]
    levels_ = [f"{float(x):g}" for x in sc.input_peak_sweep_dbfs]

    counts: dict[str, int] = defaultdict(int)
    for r in all_items:
        counts[group_of(r.item)] += 1
    selected: dict[str, int] = defaultdict(int)
    for r in items:
        selected[group_of(r.item)] += 1
    internal_groups = sorted({group_of(r.item) for r in all_items if r.internal_only})

    per_item = [
        {
            "item_id": r["item_id"],
            "group": r["group"],
            "harmonic": r["harmonic"],
            "dry_mrstft": r["dry"]["mrstft"],
            "mrstft": {
                lv: {e: v["mrstft"] for e, v in row.items()} for lv, row in r["levels"].items()
            },
        }
        for r in kb_rows
        if not r["internal_only"]
    ]
    return {
        "protocol": {
            "sample_rate": int(sc.resample_to),
            "input_peak_sweep_dbfs": [float(x) for x in sc.input_peak_sweep_dbfs],
            "max_items_per_group": max_per,
            "limit": limit,
            "mrstft_fft_sizes": [int(x) for x in cfg.eval.metrics.mrstft_fft_sizes],
            "min_voiced_fraction": float(cfg.analysis.min_voiced_fraction),
            "pyin": librosa_available(),
            "pod_segment_s": list(POD_SEGMENT_S),
            "bootstrap": {
                "n_resamples": int(cfg.eval.stats.n_resamples),
                "alpha": float(cfg.eval.stats.alpha),
                "seed": int(cfg.eval.seed),
            },  # fmt: skip
        },
        "groups": {
            g: {
                "n_total": counts[g],
                "n_evaluated": selected.get(g, 0),
                "internal_only": g in internal_groups,
            }  # fmt: skip
            for g in sorted(counts)
        },
        "kb": aggregate_kb(kb_rows, cfg),
        "family_shifts": aggregate_shifts([*kb_rows, *shift_rows], synthetic_shifts(cfg), cfg),
        "ts_grid_direction": aggregate_ts_grid(shift_rows, levels_),
        "kb_knob_sweeps": sweep_rows,
        "per_item": per_item,
    }


def run(cfg, args) -> int:
    """``lstmabar eval sim-real``: writes ``reports/sim_real.{md,json}``."""
    from lstmabar.runs import create_run_dir

    workers = min(2, int(getattr(args, "workers", None) or cfg.eval.workers or 1))
    limit = getattr(args, "limit", None)
    t0 = time.perf_counter()
    res = evaluate(cfg, limit=limit, workers=workers)
    res["protocol"]["runtime_s"] = round(time.perf_counter() - t0, 1)
    run_dir = create_run_dir(cfg, cfg.paths.runs, "eval_sim_real")
    (run_dir / "sim_real.json").write_text(_dumps(res), encoding="utf-8")
    out = Path(str(cfg.eval.report_dir))
    out.mkdir(parents=True, exist_ok=True)
    (out / "sim_real.json").write_text(_dumps(res), encoding="utf-8")
    (out / "sim_real.md").write_text(render_markdown(res), encoding="utf-8")
    print(f"wrote {out.as_posix()}/sim_real.md and .json (run dir {run_dir.name})")
    return 0


def _clean(v):
    if isinstance(v, float):
        return round(v, 6) if math.isfinite(v) else None
    if isinstance(v, np.generic):
        return _clean(v.item())
    if isinstance(v, Mapping):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_clean(x) for x in v]
    return v


def _dumps(res: Mapping) -> str:
    return json.dumps(_clean(res), indent=1, sort_keys=False) + "\n"


# ============================================================================ markdown


def _fmt(c: Mapping | None, digits: int = 3) -> str:
    if not c or c.get("n", 0) == 0 or c.get("mean") is None or not math.isfinite(c["mean"]):
        return "n/a"
    return f"{c['mean']:.{digits}f} [{c['lo']:.{digits}f}, {c['hi']:.{digits}f}] (n={c['n']})"


def _num(x, digits: int = 2) -> str:
    return "n/a" if x is None or not math.isfinite(float(x)) else f"{float(x):+.{digits}f}"


def render_markdown(res: Mapping) -> str:
    p = res["protocol"]
    L: list[str] = [
        "# test_real sim-to-real gap (P5 D11)",
        "",
        "Generated by `lstmabar eval sim-real` (`src/lstmabar/eval/sim_real.py`). Every number "
        "is a mean over items with a 95 % percentile bootstrap CI "
        f"({p['bootstrap']['n_resamples']} resamples) and n. Audio at {p['sample_rate']} Hz; "
        "renders are loudness-matched (RMS) to the real wet before MR-STFT "
        f"(FFT sizes {p['mrstft_fft_sizes']}, lower is closer) and descriptor errors.",
        f"Assumed input peak sweep: {p['input_peak_sweep_dbfs']} dBFS (true level unknown: the "
        "datasets are level-normalized). **best** = per item, the sweep level with the lowest "
        "MR-STFT for that engine; it is chosen on the item it is scored on, so it is an "
        "optimistic bound, not a calibrated result.",
        f"Harmonic terms (odd/even, HNR) need the dry's pYIN voiced fraction >= "
        f"{p['min_voiced_fraction']} (pYIN available: {p['pyin']}). pOD-set items (internal "
        "only) appear in aggregates only.",
        "",
        "## Items",
        "",
        "| group | total | evaluated | internal only |",
        "|---|---:|---:|---|",
    ]
    for g, c in res["groups"].items():
        L.append(f"| {g} | {c['n_total']} | {c['n_evaluated']} | {c['internal_only']} |")
    L += [
        "",
        f"Subsample: at most {p['max_items_per_group']} items per group (lowest sha256 of the item "
        "id), pOD-set sine-sweep files skipped (mixed-material `a` files only, segment "
        f"{p['pod_segment_s'][0]:g} s + {p['pod_segment_s'][1]:g} s)"
        + (f"; `--limit {p['limit']}`" if p.get("limit") else "")
        + ".",
        "",
    ]
    for g, k in res["kb"].items():
        tag = " (unseen pedal: holdout `rat`)" if k["unseen_pedal"] else ""
        L += [
            f"## {g} -> KB `{k['kb_pedal']}`{tag}",
            "",
            f"Settings (KB knob rotations): {k['settings']}. Items: {k['n_items']} "
            f"({k['n_harmonic']} with harmonic terms).",
            "",
            "| engine | input peak | MR-STFT | abs d centroid (oct) | abs d odd/even (dB) | "
            "abs d HNR (dB) | centroid shift sign agrees | crest shift sign agrees |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in k["cells"]:
            L.append(
                f"| {c['engine']} | {c['level']} | {_fmt(c['mrstft'])} | "
                f"{_fmt(c['centroid_oct'])} | {_fmt(c['odd_even_db'], 2)} | "
                f"{_fmt(c['hnr_db'], 2)} | {_fmt(c['agree_centroid_oct'], 2)} | "
                f"{_fmt(c['agree_crest_db'], 2)} |"
            )
        L += ["", "Best-of-sweep level counts (MR-STFT): " + "; ".join(
            f"{e}: {h}" for e, h in k["best_level_counts"].items()) + ".", ""]  # fmt: skip
    L += [
        "`dry` = the clean note itself vs the real wet (no pedal); `greybox_default` = the "
        "grey-box at the KB pedal's default knobs (all 0.5) at the same input level. The sign "
        "columns are the fraction of items whose render moves the spectral centroid / crest "
        "factor relative to the dry in the same direction as the real pedal does.",
        "",
        "## Family-level descriptor shifts (wet - dry)",
        "",
        "Real recordings vs the synthetic v1 `train` renders of KB pedals of the same family "
        "(overdrive: ts808; distortion: ds1, rat). Aggregate only; mean [CI] (n), median.",
        "",
        "| group | family | term | real | synthetic family |",
        "|---|---|---|---|---|",
    ]
    for g, s in res["family_shifts"].items():
        for t, v in s["terms"].items():
            L.append(
                f"| {g} | {s['family']} | {t} | {_fmt(v['real'], 2)}, "
                f"med {_num(v['real']['median'])}"
                f" | {_fmt(v['synthetic_family'], 2)}, med "
                f"{_num(v['synthetic_family']['median'])} |"
            )
    L += ["", "## Knob direction", ""]
    sweeps = res.get("kb_knob_sweeps") or []
    if sweeps:
        L += [
            f"KB grey-box knob sweeps on {KNOB_SWEEP_NOTES} dry notes per pedal at "
            f"{KNOB_SWEEP_LEVEL_DBFS:g} dBFS (other knobs at the item settings): mean shift vs "
            "dry at positions 0 / 0.5 / 1.",
            "",
            "| pedal | knob | centroid shift (oct) | crest shift (dB) |",
            "|---|---|---|---|",
        ]
        by: dict[tuple[str, str], list] = defaultdict(list)
        for s in sweeps:
            for knob, rows in s["sweeps"].items():
                by[(s["kb_pedal"], knob)].append(rows)
        for (ped, knob), lst in sorted(by.items()):
            cen = [np.nanmean([r[i]["centroid_oct"] for r in lst]) for i in range(len(lst[0]))]
            cr = [np.nanmean([r[i]["crest_db"] for r in lst]) for i in range(len(lst[0]))]
            L.append(
                f"| {ped} | {knob} | {' / '.join(_num(v) for v in cen)} | "
                f"{' / '.join(_num(v) for v in cr)} |"
            )
        L.append("")
    ts = res.get("ts_grid_direction") or {}
    if ts:
        L += [
            f"pOD-set `tubedreamer` gain x tone grid ({ts['n_items']} items, internal: aggregate "
            "only) vs the `ts808` grey-box at drive = gain, tone = tone: Spearman rho of knob "
            "vs descriptor shift.",
            "",
            "| pair | real | ts808 KB per input peak |",
            "|---|---|---|",
        ]
        for name, v in ts["pairs"].items():
            kb = ", ".join(f"{lv}: {_num(r)}" for lv, r in v["kb"].items())
            L.append(f"| {name} | {_num(v['real'])} | {kb} |")
        L.append("")
    return "\n".join(L).rstrip() + "\n"


__all__ = [
    "bootstrap_ci",
    "descriptor_errors",
    "descriptor_shift",
    "eval_kb_item",
    "eval_knob_sweep",
    "eval_shift_item",
    "evaluate",
    "render_markdown",
    "run",
    "select_items",
]
