"""Scoring a method on a split (P5 task 5.1 / D1-D3, D8).

Per query: predict a board -> render it with ``default_pedalboard`` on the query's dry input
(at the query's level, D1) -> metrics against the target. Per-query scores are averaged into
one :class:`UnitScores` per statistical unit (synthetic item / gold item, D3) before any
statistics. Renders of predictions are never written to the dataset tree.

Rendering and analysis run in-process, or in ``eval.workers`` spawn processes over chunks of
units (``workers <= 1``: in-process). Predicted renders are cached per worker, keyed by
``(audio_key, input level, rounded board vector)``, so a method that predicts the same board
for several queries on one input (no-op, mean params, both captions of an item) renders and
analyses it once. The cache is a small LRU, so memory stays bounded; target wets are read per
unit and dropped.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import time
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from lstmabar.data import levels
from lstmabar.data.gold import load_targets, region_from_dict
from lstmabar.data.schema import FloatBoard, floats_to_board
from lstmabar.dsp.pedalboard import default_pedalboard
from lstmabar.eval import metrics as M
from lstmabar.eval.baselines.base import Method, board_vector, complete_board
from lstmabar.eval.data import GoldItem, Query, SynthItem, load_dry_track, load_render_audio

SYNTH_SCORES: tuple[str, ...] = (
    "param_l1",
    "gate_acc",
    "mrstft",
    "centroid_oct",
    "odd_even_db",
    "hnr_db",
    "archetype_l1",
)
"""Per-unit keys of :func:`score_synthetic` (``gate_acc`` = :func:`metrics.gate_correct`;
gate F1 is computed over the whole split from ``preds``/``labels``)."""

GOLD_SCORES: tuple[str, ...] = (
    "afc_mrstft",
    "afc_descriptor",
    "region_l1",
    "mrstft_target",
)
"""Per-unit keys of :func:`score_gold` (2AFC with the primary and secondary distance, region
distance, MR-STFT to the target render). Per-axis 2AFC keys are added as
``afc_mrstft/<axis>`` when the item has a distractor on that axis."""

RENDER_CACHE_SIZE = 32  # predicted renders kept per worker (each ~1-2 MB)
UNITS_PER_JOB = 16  # units per pool task
BOARD_ROUND = 6  # decimals of the board-vector cache key
NAN = float("nan")


@dataclass
class UnitScores:
    unit_id: str
    split: str  # synthetic split name | "gold"
    method: str
    seed: int
    scores: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)  # gold_source, subsets, internal_only, ...

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SplitResult:
    split: str
    method: str
    seed: int
    units: list[UnitScores]
    preds: dict[str, FloatBoard]  # query_id -> board (gate F1, region checks, the report)
    extra: dict[str, Any] = field(default_factory=dict)  # e.g. gate F1 per block


def predict_all(method: Method, queries: Sequence[Query], batch_size: int = 64):
    """``{query_id: board}`` for every query, in batches (boards are completed and validated
    with ``complete_board``)."""
    out: dict[str, FloatBoard] = {}
    qs = list(queries)
    for i in range(0, len(qs), max(1, int(batch_size))):
        batch = qs[i : i + max(1, int(batch_size))]
        boards = method.predict(batch)
        if len(boards) != len(batch):
            raise ValueError(f"{method.name}: {len(boards)} boards for {len(batch)} queries")
        for q, b in zip(batch, boards, strict=True):
            out[q.query_id] = complete_board(b)
    return out


# ============================================================================ per-worker context


def _nanmean(values) -> float:
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else NAN


def _mean_scores(rows: Sequence[dict[str, float]], keys: Sequence[str] = ()) -> dict[str, float]:
    """Key-wise NaN-aware mean of per-query rows (keys missing from a row count as absent)."""
    names = list(keys)
    for r in rows:
        names += [k for k in r if k not in names]
    return {k: _nanmean(r[k] for r in rows if k in r) for k in names}


class _Ctx:
    """Board, analysis settings and small caches of one scoring process."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.sr = int(cfg.audio.sample_rate)
        self.board = default_pedalboard(self.sr)
        self.wet_peak = float(cfg.render.wet_peak_dbfs)
        self.min_vf = float(cfg.analysis.min_voiced_fraction)
        m = cfg.eval.metrics
        self.fft_sizes = tuple(int(s) for s in m.mrstft_fft_sizes)
        self.threshold = float(m.gate_threshold)
        # Harmonic analysis of predicted renders dominates the cost (~90 % per query); off
        # skips the harmonic terms (NaN) but keeps the centroid. Default on (D2).
        self.harmonic_on = bool(m.get("harmonic_descriptors", True))
        self._renders: OrderedDict[tuple, tuple] = OrderedDict()
        self._dry: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._tracks: dict[str, Any] = {}
        self._targets = None
        self.n_renders = 0

    @property
    def targets(self):
        if self._targets is None:
            self._targets = load_targets(str(self.cfg.gold.targets_file))
        return self._targets

    def track(self, clip_id: str):
        if clip_id not in self._tracks:
            if len(self._tracks) > 256:
                self._tracks.clear()
            try:
                self._tracks[clip_id] = load_dry_track(self.cfg, clip_id)
            except (OSError, KeyError, ValueError):
                self._tracks[clip_id] = None
        return self._tracks[clip_id]

    def harmonic(self, voiced_fraction, clip_id: str):
        """``(harmonic, track)``: harmonic terms iff the dry voiced fraction reaches the
        threshold and the dry pYIN track is cached (and ``eval.metrics.harmonic_descriptors``
        is not false)."""
        if (
            not self.harmonic_on
            or voiced_fraction is None
            or not math.isfinite(float(voiced_fraction))
        ):
            return False, None
        if float(voiced_fraction) < self.min_vf:
            return False, None
        tr = self.track(clip_id)
        return tr is not None, tr

    def dry(self, q: Query) -> np.ndarray:
        key = (q.audio_key, round(q.input_peak_dbfs, 3))
        if key not in self._dry:
            self._dry[key] = q.load_dry()
            while len(self._dry) > 4:
                self._dry.popitem(last=False)
        self._dry.move_to_end(key)
        return self._dry[key]

    def render(self, q: Query, board: FloatBoard, harmonic: bool, track):
        """``(stored-style wet or None, descriptors, archetype)`` of ``board`` on ``q``'s
        input; ``None`` audio when the render is non-finite or silent."""
        key = (
            q.audio_key,
            q.clip_id,
            round(q.input_peak_dbfs, 3),
            tuple(round(v, BOARD_ROUND) for v in board_vector(board)),
            bool(harmonic),
        )
        if key in self._renders:
            self._renders.move_to_end(key)
            return self._renders[key]
        x = self.dry(q)
        with torch.no_grad():
            xt = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
            y = self.board(xt, floats_to_board(board))[0].double().numpy()
        self.n_renders += 1
        if not np.all(np.isfinite(y)) or levels.peak_dbfs(y) <= levels.SILENCE_DBFS:
            out: tuple = (None, {}, {})
        else:
            y, _ = levels.normalize_peak(y, self.wet_peak)
            desc, arch = M.analyse_render(y, self.cfg, track, harmonic)
            out = (y, desc, arch)
        self._renders[key] = out
        while len(self._renders) > RENDER_CACHE_SIZE:
            self._renders.popitem(last=False)
        return out

    def mrstft(self, y, target) -> float:
        if y is None:
            return NAN
        return M.mrstft(y, target, self.fft_sizes)


_CTX: _Ctx | None = None
_CTX_KEY: str | None = None


def _context(cfg) -> _Ctx:
    """One context per process (rebuilt only if the config changes)."""
    global _CTX, _CTX_KEY
    if not isinstance(cfg, dict):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    key = json.dumps(cfg, sort_keys=True, default=str)
    if _CTX is None or key != _CTX_KEY:
        _CTX = _Ctx(OmegaConf.create(cfg))
        _CTX_KEY = key
    return _CTX


def _init_worker() -> None:
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore", message=".*numba.*")


# ============================================================================ unit scoring


def _synth_unit(ctx: _Ctx, item: SynthItem, boards: list[tuple[Query, FloatBoard]], meta):
    label = item.label
    it = item.item
    harmonic, track = ctx.harmonic(it.voiced_fraction, it.clip_id)
    wet = item.load_wet()
    rows = []
    for q, board in boards:
        y, desc, arch = ctx.render(q, board, harmonic, track)
        row = {
            "param_l1": M.param_l1(board, label)["mean"],
            "gate_acc": M.gate_correct(board, label, ctx.threshold),
            "mrstft": ctx.mrstft(y, wet),
        }
        if y is None:
            row |= dict.fromkeys(M.DESCRIPTOR_TERMS, NAN)
        else:
            row |= M.descriptor_errors(
                desc, it.descriptors_wet, arch, it.archetype_wet, harmonic=harmonic
            )
        rows.append(row)
    return _mean_scores(rows, SYNTH_SCORES), {
        **meta,
        "n_queries": len(rows),
        "harmonic": bool(harmonic),
    }


def _gold_unit(ctx: _Ctx, item: GoldItem, preds: dict[str, FloatBoard]):
    region = region_from_dict(item.region, item.gold_id)
    refs = [dict(c.target.render["board_params"]) for c in item.clips]
    rows = []
    axes: list[str] = []
    for q, clip in zip(item.queries(), item.clips, strict=True):
        board = preds[q.query_id]
        t = clip.target
        harmonic, track = ctx.harmonic(t.render.get("voiced_fraction"), clip.clip_id)
        y, desc, _ = ctx.render(q, board, harmonic, track)
        d_t = ctx.mrstft(y, load_render_audio(ctx.cfg, t))
        dd_t = M.descriptor_distance(desc, t.render["descriptors_wet"], harmonic) if desc else NAN
        d_ds, dd_ds = [], []
        row: dict[str, float] = {}
        per_axis: dict[str, list[float]] = defaultdict(list)
        for d in clip.distractors:
            dm = ctx.mrstft(y, load_render_audio(ctx.cfg, d))
            d_ds.append(dm)
            dd_ds.append(
                M.descriptor_distance(desc, d.render["descriptors_wet"], harmonic) if desc else NAN
            )
            if d.axis:
                per_axis[d.axis].append(M.two_afc(d_t, [dm]))
                if d.axis not in axes:
                    axes.append(d.axis)
        row["afc_mrstft"] = M.two_afc(d_t, d_ds)
        row["afc_descriptor"] = M.two_afc(dd_t, dd_ds)
        row["region_l1"] = M.region_distance(board, region, ctx.targets, refs)
        row["mrstft_target"] = d_t
        for axis, v in per_axis.items():
            row[f"afc_mrstft/{axis}"] = _nanmean(v)
        rows.append(row)
    meta = {
        "gold_source": item.source,
        "subsets": item.subsets(),
        "in_scope": item.in_scope,
        "ambiguous": item.ambiguous,
        "unseen_pedal": item.unseen_pedal,
        "confidence": item.confidence,
        "axes": sorted(axes),
        "n_clips": len(rows),
    }
    return _mean_scores(rows, GOLD_SCORES), meta


def _run_job(job: dict) -> list[tuple[str, dict, dict]]:
    """Score one chunk of units (top-level and picklable: the pool's worker function)."""
    ctx = _context(job["cfg"])
    out = []
    if job["kind"] == "synth":
        for item, boards in job["units"]:
            scores, meta = _synth_unit(ctx, item, boards, {})
            out.append((item.item_id, scores, meta))
    else:
        for item, preds in job["units"]:
            scores, meta = _gold_unit(ctx, item, preds)
            out.append((item.gold_id, scores, meta))
    return out


def _workers(cfg) -> int:
    try:
        return max(1, int(cfg.eval.get("workers") or 1))
    except (AttributeError, TypeError, ValueError):
        return 1


def _dispatch(cfg, kind: str, units: list) -> list[tuple[str, dict, dict]]:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    jobs = [
        {"kind": kind, "cfg": cfg_dict, "units": units[i : i + UNITS_PER_JOB]}
        for i in range(0, len(units), UNITS_PER_JOB)
    ]
    workers = min(_workers(cfg), len(jobs))
    if workers <= 1:
        threads = torch.get_num_threads()
        torch.set_num_threads(1)  # same numerics/threading as the pool workers
        try:
            return [r for job in jobs for r in _run_job(job)]
        finally:
            torch.set_num_threads(threads)
    mp = multiprocessing.get_context("spawn")  # Windows-safe; no forked torch state
    with ProcessPoolExecutor(workers, mp_context=mp, initializer=_init_worker) as pool:
        return [r for res in pool.map(_run_job, jobs) for r in res]


def score_synthetic(
    method: Method, items: Sequence[SynthItem], cfg, split: str, seed: int = 0
) -> SplitResult:
    """Score ``method`` on synthetic items (D1: each prediction is rendered on the item's dry
    clip at the item's own level). Per query: MR-STFT to the item's wet, descriptor and
    archetype errors to its stored wet descriptors (harmonic terms iff the item's dry voiced
    fraction reaches ``analysis.min_voiced_fraction`` and the dry pYIN track is cached), param
    L1 masked by the label's gates and gate accuracy; queries are averaged per item.
    ``extra["gate_f1"]`` is the per-block + macro gate F1 over every query."""
    t0 = time.perf_counter()
    queries = [q for it in items for q in it.queries()]
    preds = predict_all(method, queries)
    t_pred = time.perf_counter() - t0
    units_in = [(it, [(q, preds[q.query_id]) for q in it.queries()]) for it in items]
    results = _dispatch(cfg, "synth", units_in)
    by_id = {it.item_id: it for it in items}
    units = [
        UnitScores(
            uid, split, method.name, seed, scores, {**meta, "scenario": by_id[uid].item.scenario}
        )
        for uid, scores, meta in results
    ]
    labels = [by_id[q.unit_id].label for q in queries]
    gate = M.gate_f1(
        [preds[q.query_id] for q in queries], labels, float(cfg.eval.metrics.gate_threshold)
    )
    extra = {
        "gate_f1": gate,
        "n_queries": len(queries),
        "seconds": {"predict": t_pred, "total": time.perf_counter() - t0},
    }
    return SplitResult(split, method.name, seed, units, preds, extra)


def score_gold(method: Method, items: Sequence[GoldItem], cfg, seed: int = 0) -> SplitResult:
    """Gold 2AFC etc. Also scores val gold-proxy items (``proxy.load_proxy_items``: GoldItems
    with ``split == "val"``); the result's ``split`` is the items' split.

    Per clip: the prediction rendered on the clip at the item's level; MR-STFT (primary) and
    descriptor distance (secondary, against each render's stored ``descriptors_wet``) to the
    target and to each distractor render; 2AFC with both distances (and per distractor axis,
    ``afc_mrstft/<axis>``); region distance with the item's target renders' ``board_params``
    as references; MR-STFT to the target. Clips are averaged per gold item."""
    t0 = time.perf_counter()
    splits = {g.split for g in items}
    if len(splits) > 1:
        raise ValueError(f"score_gold: items from several splits {sorted(splits)}")
    split = next(iter(splits)) if splits else "gold"
    queries = [q for g in items for q in g.queries()]
    preds = predict_all(method, queries)
    t_pred = time.perf_counter() - t0
    units_in = [(g, {q.query_id: preds[q.query_id] for q in g.queries()}) for g in items]
    results = _dispatch(cfg, "gold", units_in)
    units = [UnitScores(uid, split, method.name, seed, s, meta) for uid, s, meta in results]
    extra = {
        "n_queries": len(queries),
        "seconds": {"predict": t_pred, "total": time.perf_counter() - t0},
    }
    return SplitResult(split, method.name, seed, units, preds, extra)


# ============================================================================ val selection (D8)


def _proxy_items(cfg) -> list[GoldItem] | None:
    """Val proxy items, or ``None`` (with a warning) when unavailable."""
    manifest = Path(str(cfg.data.manifests_dir)) / f"{cfg.eval.proxy.version}.jsonl"
    if not manifest.exists():
        warnings.warn(
            f"val proxy manifest {manifest.name} not found: selection on synthetic val only "
            "(run `lstmabar eval proxy`)",
            stacklevel=3,
        )
        return None
    try:
        from lstmabar.eval.proxy import load_proxy_items

        limit = cfg.eval.get("max_items")
        return list(load_proxy_items(cfg, limit=int(limit) if limit else None))
    except NotImplementedError:
        warnings.warn(
            "eval.proxy.load_proxy_items is not implemented yet: selection on synthetic val only",
            stacklevel=3,
        )
        return None


def score_val(method: Method, val: Sequence[SynthItem], cfg, seed: int = 0) -> SplitResult:
    """The D8 selection run: :func:`score_synthetic` on val plus :func:`score_gold` on the val
    proxy items (merged per unit: ``afc_*`` keys join the synthetic keys). Without proxy items
    (manifest missing or proxy not implemented) the synthetic result is returned with a
    warning and ``extra["proxy"] = False``."""
    res = score_synthetic(method, val, cfg, "val", seed)
    proxy = _proxy_items(cfg)
    res.extra["proxy"] = False
    if not proxy:
        return res
    want = {it.item_id for it in val}
    proxy = [g for g in proxy if g.gold_id in want]
    if not proxy:
        warnings.warn("no val proxy item matches the scored val items", stacklevel=2)
        return res
    gold = score_gold(method, proxy, cfg, seed)
    by_unit = {u.unit_id: u for u in gold.units}
    for u in res.units:
        g = by_unit.get(u.unit_id)
        if g is not None:
            u.scores |= {k: v for k, v in g.scores.items() if k.startswith("afc_")}
            u.meta["proxy_axes"] = g.meta.get("axes", [])
    res.extra |= {"proxy": True, "n_proxy": len(gold.units), "proxy_preds": gold.preds}
    return res


def _fallback_selection_score(result: SplitResult, cfg) -> float:
    w = cfg.eval.proxy.selection_weights
    mean = {
        k: _nanmean(u.scores.get(k, NAN) for u in result.units)
        for k in ("mrstft", "param_l1", "afc_mrstft")
    }
    s = float(w.mrstft) * mean["mrstft"] + float(w.param_l1) * mean["param_l1"]
    if math.isfinite(mean["afc_mrstft"]):
        s += float(w.afc) * (1.0 - mean["afc_mrstft"])
    return s


def _selection_score(result: SplitResult, cfg) -> float:
    try:
        from lstmabar.eval.proxy import selection_score

        return float(selection_score(result, cfg))
    except NotImplementedError:
        warnings.warn(
            "eval.proxy.selection_score is not implemented yet: selecting on "
            "weighted MR-STFT + param L1 (+ proxy 2AFC if present)",
            stacklevel=2,
        )
        return _fallback_selection_score(result, cfg)


def select_hparams(method: Method, val: Sequence[SynthItem], cfg, seed: int = 0) -> dict:
    """Try every ``method.hparam_grid()`` entry on val, keep the one minimizing
    ``proxy.selection_score`` (ties: first in grid order), apply it, and return
    ``{"chosen": {...}, "scores": [...]}``; falls back to ``method.select(val)`` when the grid
    is empty. NaN scores never win."""
    grid = list(method.hparam_grid())
    if not grid:
        return {"chosen": method.select(val), "scores": []}
    scores = []
    best, best_s = None, math.inf
    for hp in grid:
        method.set_hparams(dict(hp))
        s = _selection_score(score_val(method, val, cfg, seed), cfg)
        scores.append({"hparams": dict(hp), "score": s})
        if math.isfinite(s) and s < best_s:
            best, best_s = dict(hp), s
    if best is None:
        best = dict(grid[0])
    method.set_hparams(best)
    return {"chosen": best, "scores": scores}


__all__ = [
    "GOLD_SCORES",
    "SYNTH_SCORES",
    "SplitResult",
    "UnitScores",
    "predict_all",
    "score_gold",
    "score_synthetic",
    "score_val",
    "select_hparams",
]
