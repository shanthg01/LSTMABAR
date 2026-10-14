"""Val "gold-proxy" for selection (D8).

Gold is test-only, so baselines, ``k`` and P6 early stopping select on val: MR-STFT to the
target wet + masked param L1, plus a **synthetic val 2AFC** whose distractors follow the gold
axis rules. :func:`label_to_region` maps a point label to a categorical ``gold.Region`` (levels
from the targets-file tables, nearest level when a value falls between table ranges) so
``gold.draft_distractors`` can be reused unchanged; distractors are sampled with ``gold.sample``
and rendered on the item's own dry clip at its own input level into
``data/rendered/<eval.proxy.version>/`` with a committed manifest
``data/manifests/<eval.proxy.version>.jsonl`` (``gold.GoldRender`` records, ``gold_source``
``"val_proxy"``) and ``<version>.index.json`` (counts, hashes).

Only the targets file's ``tables`` and ``pedals`` are read (:func:`load_tables`); its gold rows
and the gold instruction files are never touched.

**Label -> region.** Free boards (and pedal items whose pedal has no targets-file table): the
drive gate and ``gain_db`` give the gain level (``boost`` family when ``gain_db`` is nearer
``tables.boost_gain_db`` than ``tables.gain_db.low``), drive family + clipping are the nearest
``tables.family`` (updated by ``tables.clipping``) entry over the drive voicing knobs, the EQ
high shelf gives brightness (the axis ``gold.separation`` measures; EQ off counts as flat),
``mid_db`` the mids, and the compressor gate + ratio the compression level.
"Nearest" is the interval gap in normalized-knob units, ties broken by the distance to the
range centre (and, for the drive voicing, by preferring no clipping override).
Pedal items with a table -> ``kind: pedal`` with gain/brightness levels from the knob rotations.

**Distractors.** ``gold.draft_distractors`` with ``main_ask_eq`` = the region is a free board
without drive (as ``gold.draft_row``). Each drafted distractor is checked with
``gold.separation`` against the item's *actual* label (a point, :func:`label_concrete`) and
dropped when below ``gold.min_gap``; the first ``eval.proxy.distractors_per_item`` survivors
are rendered (counts of dropped ones are in the index). Engine as in ``gold.plan``: white-box
for same-pedal distractors of ``render.whitebox_pedals`` (so a pedal item's tone distractor
uses the same engine as its target wet), grey-box otherwise.

Never select on any gold split.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from lstmabar.data import gold, render
from lstmabar.data.gold import (
    BLOCKS,
    BRIGHTNESS,
    COMPRESSION,
    FAMILIES,
    GAINS,
    MIDS,
    Concrete,
    Distractor,
    GoldRender,
    Region,
    Targets,
)
from lstmabar.data.layout import DataLayout
from lstmabar.data.schema import DryClip, RenderItem, read_jsonl, to_dict
from lstmabar.dsp.pedalboard import ENABLED
from lstmabar.physics.derive import BLOCK_SPECS, DeriveContext

PROXY_SOURCE = "val_proxy"
PROXY_SPLIT = "val"
SCORE_KEYS = {"mrstft": "mrstft", "param_l1": "param_l1", "afc": "afc_mrstft"}
"""``eval.proxy.selection_weights`` key -> per-unit score key of ``runner.score_val``."""

Range = tuple[float, float]


# ============================================================================ tables


def load_tables(path: str | Path) -> Targets:
    """The targets file's ``tables`` and ``pedals`` only (its gold ``items`` are dropped
    unparsed, so nothing about gold instructions reaches the proxy)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        text = gold.items_prefix(text)  # parse only what precedes the gold rows
    except gold.GoldError:
        pass
    doc = yaml.safe_load(text) or {}
    keep = {k: doc[k] for k in ("version", "status", "tables", "pedals") if k in doc}
    return gold.targets_from_dict({**keep, "items": []}, Path(path))


def _spec(block: str, knob: str):
    return next(s for s in BLOCK_SPECS[block] if s.name == knob)


def _norm(block: str, knob: str, v: float) -> float:
    return float(_spec(block, knob).normalize(torch.tensor(float(v), dtype=torch.float64)))


def _phys(block: str, knob: str, u: float) -> float:
    return float(_spec(block, knob).denormalize(torch.tensor(float(u), dtype=torch.float64)))


def _gap_centre(u: float, r: Range, block: str | None = None, knob: str | None = None):
    """``(gap, |u - centre|)`` of a normalized value to a range (physical when ``block`` is
    given, else already normalized)."""
    lo, hi = (_norm(block, knob, r[0]), _norm(block, knob, r[1])) if block else r
    return max(lo - u, u - hi, 0.0), abs(u - (lo + hi) / 2)


def _nearest(options: Sequence[tuple[str, Mapping[str, Range]]], values, block: str | None):
    """The option whose ranges are nearest ``values`` (``{knob: normalized}``): summed gap,
    then summed centre distance; first in order on a full tie."""

    def cost(ranges):
        g = c = 0.0
        for k, r in ranges.items():
            dg, dc = _gap_centre(values[k], r, block, k)
            g, c = g + dg, c + dc
        return g, c

    return min(options, key=lambda o: cost(o[1]))[0]


# ============================================================================ label -> region


def _on(board: Mapping[str, Mapping[str, float]], block: str) -> bool:
    return float(board[block].get(ENABLED, 1.0)) >= 0.5


def _eq_values(board) -> dict[str, float]:
    """Normalized EQ knobs, flat (0 dB gains) when the EQ is off."""
    eq = dict(board["eq"])
    if not _on(board, "eq"):
        for k in ("low_db", "mid_db", "high_db"):
            eq[k] = _norm("eq", k, 0.0)
    return eq


def _drive_voicing(drive: Mapping[str, float], t, families: Sequence[str]):
    options = []
    for fam in families:  # no clipping override first: preferred on a tie
        for clip in (None, *gold.CLIPPING):
            ranges = {k: gold._r(v) for k, v in t["family"][fam].items()}
            if clip:
                ranges |= {k: gold._r(v) for k, v in t["clipping"][clip].items()}
            options.append(((fam, clip), ranges))

    def cost(o):
        g = c = 0.0
        for k, r in o[1].items():
            dg, dc = _gap_centre(drive[k], r, "drive", k)
            g, c = g + dg, c + dc
        return g, o[0][1] is not None, c

    return min(options, key=cost)[0]


def free_region(board: Mapping[str, Mapping[str, float]], targets: Targets) -> Region:
    """Categorical free-board region of a normalized board label (module docstring)."""
    t = targets.tables
    eq = _eq_values(board)
    brightness = _nearest(
        [(lvl, {"high_db": gold._r(t["brightness"][lvl]["high_db"])}) for lvl in BRIGHTNESS],
        eq,
        "eq",
    )
    mids = _nearest(
        [(lvl, {"mid_db": gold._r(t["mids"][lvl]["mid_db"])}) for lvl in MIDS], eq, "eq"
    )
    compression = "none"
    if _on(board, "compressor"):
        comp = board["compressor"]
        compression = _nearest(
            [(lvl, {"ratio": gold._r(t["compression"][lvl]["ratio"])}) for lvl in COMPRESSION[1:]],
            comp,
            "compressor",
        )
    if not _on(board, "drive"):
        return Region("free", "none", brightness, mids, compression)
    drive = board["drive"]
    levels = [("boost", {"gain_db": gold._r(t["boost_gain_db"])})]
    levels += [(lvl, {"gain_db": gold._r(t["gain_db"][lvl])}) for lvl in GAINS[1:]]
    gain = _nearest(levels, drive, "drive")
    if gain == "boost":
        family, clipping = _drive_voicing(drive, t, ("boost",))
        gain = "low"
    else:
        family, clipping = _drive_voicing(drive, t, [f for f in FAMILIES if f != "boost"])
    return Region("free", gain, brightness, mids, compression, family, clipping)


def _pedal_level(v: float, table: Mapping[str, Range], levels: Sequence[str]) -> str:
    return min((lvl for lvl in levels if lvl in table), key=lambda lvl: _gap_centre(v, table[lvl]))


def label_to_region(item: RenderItem, targets: Targets) -> Region:
    """Categorical region of a val item's label.

    Pedal items (``scenario == "pedal"``) with a targets-file pedal table -> ``kind: pedal``
    with gain/brightness levels from the pedal's knob rotations (``item.pedal_knobs``); other
    pedal items and free-board items -> ``kind: free`` from ``board_params`` (gates decide
    ``gain: none`` / ``compression: none``; drive family and clipping from the nearest table
    entries).
    """
    if item.scenario == "pedal" and item.pedal_id in targets.pedals and item.pedal_knobs:
        p = targets.pedals[item.pedal_id]
        knobs = item.pedal_knobs
        gain = _pedal_level(float(knobs[p.gain_knob]), p.gain, GAINS[1:])
        bright = "neutral"
        if p.tone_knob:
            bright = _pedal_level(float(knobs[p.tone_knob]), p.tone, BRIGHTNESS)
        return Region("pedal", gain, bright, pedal=item.pedal_id)
    return free_region(item.board_params, targets)


def region_levels(region: Region) -> Mapping[str, str]:
    """``{axis: level}`` of a region (for the report's per-axis breakdown). Pedal regions have
    gain and brightness only (mids and compression are the pedal's own voicing)."""
    if region.kind == "pedal":
        return {"gain": region.gain, "brightness": region.brightness}
    return {
        "gain": region.gain,
        "brightness": region.brightness,
        "mids": region.mids,
        "compression": region.compression,
    }


def label_concrete(item: RenderItem, region: Region) -> Concrete:
    """The item's actual label as a point :class:`gold.Concrete` of ``region``'s kind (for
    ``gold.separation``): pedal knob rotations, or the free board in physical units with its
    gates (EQ gains flat when the EQ is off)."""
    if region.kind == "pedal":
        knobs = {k: (float(v), float(v)) for k, v in (item.pedal_knobs or {}).items()}
        return Concrete("pedal", pedal=region.pedal, knobs=knobs)
    board = item.board_params
    eq = _eq_values(board)
    ranges: dict[str, dict[str, Range]] = {}
    for block in BLOCKS:
        vals = eq if block == "eq" else board[block]
        ranges[block] = {}
        for s in BLOCK_SPECS[block]:
            u = float(vals.get(s.name, s.default_normalized))
            v = _phys(block, s.name, u)
            ranges[block][s.name] = (v, v)
    gates = {b: 1.0 if _on(board, b) else 0.0 for b in BLOCKS}
    return Concrete("free", gates=gates, ranges=ranges)


def _label_gain_db(item: RenderItem) -> float | None:
    """A pedal item's drive gain into the clipper (dB) from its stored ``derive`` label (the
    actual knobs; saves a ~0.5-1 s ``derive`` call per item), ``None`` if not stored."""
    d = item.board_params_derived
    if not d or "drive" not in d or "gain_db" not in d["drive"]:
        return None
    return _phys("drive", "gain_db", float(d["drive"]["gain_db"]))


def main_ask_eq(region: Region) -> bool:
    """As ``gold.draft_row``: a free board without drive is an EQ/compression request."""
    return region.kind == "free" and region.gain == "none"


def item_distractors(
    item: RenderItem, region: Region, targets: Targets, kb, ctx: DeriveContext, cfg
) -> tuple[list[Distractor], list[tuple[str, float]]]:
    """``(kept, dropped)``: drafted distractors that separate from the item's actual label
    (first ``eval.proxy.distractors_per_item``), and ``(axis, gap)`` of the dropped ones."""
    drafted = gold.draft_distractors(
        region, targets, kb, ctx, float(cfg.gold.gain_split_db), main_ask_eq(region)
    )
    lc = label_concrete(item, region)
    kept, dropped = [], []
    for d in drafted:
        dc = gold.expand(d.region, targets)
        g = _label_gain_db(item) if lc.kind == "pedal" and d.axis == "gain" else None
        if g is not None:
            gap = gold._gap((g, g), gold.gain_range_db(dc, kb, ctx, gold._gain_knob(dc, targets)))
        else:
            gap = gold.separation(lc, dc, d.axis, kb, ctx, targets)
        if gap < gold.min_gap(d.axis, cfg, pedal=lc.kind == "pedal"):
            dropped.append((d.axis, float(gap)))
        else:
            kept.append(d)
    return kept[: int(cfg.eval.proxy.distractors_per_item)], dropped


# ============================================================================ planning


def proxy_cfg(cfg):
    """``cfg`` with ``data.version = eval.proxy.version`` (audio under
    ``data/rendered/<version>/``). The version must start with ``valproxy`` and differ from the
    dataset and gold versions, so the frozen v1 and gold trees are never written."""
    version = str(cfg.eval.proxy.version)
    if (
        not version.startswith("valproxy")
        or version == str(cfg.data.version)
        or version == str(cfg.gold.version)
    ):
        raise ValueError(f"eval.proxy.version {version!r} must start with 'valproxy'")
    return OmegaConf.merge(cfg, {"data": {"version": version}})


def manifest_paths(cfg, partial: bool = False) -> tuple[Path, Path]:
    """``(manifest, index)`` in ``data.manifests_dir``; ``--limit`` runs write
    ``<version>.partial.*`` so they never replace a complete manifest."""
    layout = DataLayout.from_config(cfg)
    stem = str(cfg.eval.proxy.version) + (".partial" if partial else "")
    return layout.manifests / f"{stem}.jsonl", layout.manifests / f"{stem}.index.json"


def _seed(cfg) -> int:
    return int(cfg.eval.proxy.get("seed", cfg.eval.seed))


def plan(items: Sequence[RenderItem], targets: Targets, kb, ctx, cfg):
    """``(specs, meta, stats)``: one ``RenderSpec`` per kept distractor (in item order), its
    ``(item, label region, k, distractor)`` and drafting counts."""
    seed = _seed(cfg)
    whitebox = tuple(str(p) for p in cfg.render.whitebox_pedals)
    specs, meta = [], []
    stats: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        region = label_to_region(it, targets)
        kept, dropped = item_distractors(it, region, targets, kb, ctx, cfg)
        for axis, _ in dropped:
            stats["dropped_separation"][axis] += 1
        stats["items_by_distractors"][str(len(kept))] += 1
        for k, d in enumerate(kept):
            conc = gold.expand(d.region, targets)
            s = render.item_seed(seed, f"valproxy:{it.item_id}:d{k}", 0)
            knobs, board = gold.sample(conc, np.random.default_rng(s))
            engine = "whitebox" if conc.kind == "pedal" and conc.pedal in whitebox else "greybox"
            specs.append(
                render.RenderSpec(
                    item_id=f"{it.item_id}-d{k}",
                    clip_id=it.clip_id,
                    split=PROXY_SOURCE,
                    seed=s,
                    scenario=conc.scenario,
                    engine=engine,
                    pedal_id=conc.pedal,
                    pedal_knobs=knobs,
                    free_board=board,
                    input_peak_dbfs=float(it.input_peak_dbfs),
                )
            )
            meta.append((it, region, k, d))
            stats["distractors"][d.axis] += 1
            stats["engine"][engine] += 1
    return specs, meta, stats


def _record(it: RenderItem, region: Region, k: int | None, d, rendered: RenderItem | None):
    role = "target" if d is None else "distractor"
    reg = region if d is None else d.region
    return GoldRender(
        item_id=it.item_id if d is None else f"{it.item_id}-d{k}",
        gold_id=it.item_id,
        gold_source=PROXY_SOURCE,
        role=role,
        distractor_index=k,
        axis=None if d is None else d.axis,
        clip_index=0,
        clip_id=it.clip_id,
        in_scope=True,
        ambiguous=False,
        confidence="high",
        unseen_pedal=False,
        region=reg.to_yaml(),
        render=to_dict(it if d is None else rendered),
    )


# ============================================================================ loading


def load_proxy_items(cfg, limit: int | None = None):
    """Val proxy units as ``eval.data.GoldItem`` s (``split="val"``, ``source=PROXY_SOURCE``,
    instruction = the item's first caption by ``caption_id``, ``input_peak_dbfs`` = the item's
    level, one clip whose target is the item's own wet as a ``GoldRender`` and whose
    distractors are the proxy renders), so ``runner.score_gold`` scores them unchanged.

    Keys: ``gold_id`` = the val ``item_id`` (so ``score_val`` merges per unit with the
    synthetic scores), ``text_key`` = that first ``caption_id`` and ``audio_key`` = the
    ``item_id`` (the val embedding-cache keys of ``SynthItem.queries``).

    Reads the complete manifest (``data/manifests/<eval.proxy.version>.jsonl``). Val items
    without a kept distractor (all drafted ones failed separation) are not 2AFC units and
    are left out; ``limit`` caps the val items considered, in ``item_id`` order.

    The proxy 2AFC uses only the **first caption** of each item (one query per unit), while
    the synthetic val scores average over all of an item's captions.

    The manifest is committed but the distractor audio (``data/rendered/<version>/``) is
    gitignored: a missing first distractor wet raises ``FileNotFoundError`` asking to run
    ``lstmabar eval proxy`` (which re-renders it, resumably)."""
    from lstmabar.eval.data import GoldClip, GoldItem, load_split

    path, _ = manifest_paths(cfg)
    if not path.exists():
        raise FileNotFoundError(f"val proxy manifest not found ({path.name}): lstmabar eval proxy")
    renders = gold.read_manifest(path)
    if renders:
        first = DataLayout.from_config(cfg).resolve_rendered(renders[0].render["wet_path"])
        if not first.exists():
            raise FileNotFoundError(
                f"val proxy audio missing ({renders[0].item_id}): the manifest is committed but "
                "its renders are gitignored; run `lstmabar eval proxy` to render them"
            )
    by_item: dict[str, list[GoldRender]] = defaultdict(list)
    for r in renders:
        if r.gold_source != PROXY_SOURCE or r.role != "distractor":
            raise ValueError(f"{path.name}: unexpected record {r.item_id}")
        by_item[r.gold_id].append(r)
    targets = load_tables(str(cfg.gold.targets_file))
    out = []
    for si in load_split(cfg, PROXY_SPLIT, limit):
        ds = sorted(by_item.get(si.item_id, []), key=lambda r: r.distractor_index)
        if not ds or not si.captions:
            continue
        region = label_to_region(si.item, targets)
        target = _record(si.item, region, None, None, None)
        clip = GoldClip(
            clip_index=0,
            clip_id=si.item.clip_id,
            target=target,
            distractors=tuple(ds),
            dry_path=si.dry_path,
        )
        out.append(
            GoldItem(
                gold_id=si.item_id,
                source=PROXY_SOURCE,
                instruction=si.captions[0].text,
                in_scope=True,
                ambiguous=False,
                unseen_pedal=False,
                confidence="high",
                region=region.to_yaml(),
                clips=(clip,),
                input_peak_dbfs=float(si.item.input_peak_dbfs),
                sample_rate=si.sample_rate,
                split=PROXY_SPLIT,
                text_key=si.captions[0].caption_id,
                audio_key=si.item_id,
            )
        )
    return out


# ============================================================================ selection


def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _mean_score(result, key: str, penalize_failed: bool = False) -> float:
    """Mean of ``key`` over units with a finite value (NaN if none). With ``penalize_failed``
    a unit whose renders all failed (``meta["render_failed"] > 0`` and a non-finite value)
    counts as the worst finite value observed, so failing to render can never help a
    candidate. (Units where only some queries failed keep the mean of the others, as
    ``runner`` averages them.)"""
    vals = [float(u.scores[key]) for u in result.units if _finite(u.scores.get(key))]
    if not vals:
        return float("nan")
    if penalize_failed:
        worst = max(vals)
        vals += [
            worst
            for u in result.units
            if not _finite(u.scores.get(key)) and int(u.meta.get("render_failed", 0) or 0) > 0
        ]
    return float(np.mean(vals))


def selection_terms(result) -> dict[str, float]:
    """The unweighted terms of :func:`selection_score` (``afc`` as ``1 - mean 2AFC``; NaN
    when the result has no proxy 2AFC scores)."""
    terms = {}
    for name, key in SCORE_KEYS.items():
        m = _mean_score(result, key, penalize_failed=name == "mrstft")
        terms[name] = 1.0 - m if name == "afc" else m
    return terms


def selection_score(result, cfg) -> float:
    """Scalar D8 selection objective of a ``runner.score_val`` result (lower is better):
    ``w_mrstft * mean MR-STFT to the target wet + w_param * mean masked param L1
    + w_afc * (1 - mean proxy 2AFC)`` with weights from ``eval.proxy.selection_weights``.

    Means are over units with a finite score. Failed renders: a unit whose renders all
    failed counts as the worst observed MR-STFT; param L1 needs no render, so it is always
    finite; ``runner.score_gold`` already scores a failed render as a 2AFC loss. Without
    proxy 2AFC scores (no proxy manifest, ``--limit`` or fixture runs: ``extra["proxy"]``
    not True) the 2AFC term is dropped with a warning, as ``runner``'s fallback does; it
    raises only when the result claims proxy scores but none is finite. A term with weight
    0 is skipped."""
    w = {k: float(v) for k, v in cfg.eval.proxy.selection_weights.items()}
    unknown = set(w) - set(SCORE_KEYS)
    if unknown:
        raise KeyError(f"eval.proxy.selection_weights: unknown terms {sorted(unknown)}")
    total = 0.0
    for name, key in SCORE_KEYS.items():
        wt = w.get(name, 0.0)
        if wt == 0.0:
            continue
        m = _mean_score(result, key, penalize_failed=name == "mrstft")
        if not math.isfinite(m):
            if name == "afc" and not (getattr(result, "extra", None) or {}).get("proxy"):
                warnings.warn(
                    "selection_score: no proxy 2AFC scores; selecting on MR-STFT + param L1 "
                    "only (run `lstmabar eval proxy`)",
                    stacklevel=2,
                )
                continue
            raise ValueError(f"selection_score: no finite {key!r} scores in the val result")
        total += wt * (1.0 - m if name == "afc" else m)
    return float(total)


# ============================================================================ CLI step


def _fixture_cfg(cfg, args):
    from lstmabar.eval.fixtures import (
        FIXTURE_GOLD_VERSION,
        eval_fixture_config,
        make_eval_fixture,
    )

    if str(cfg.gold.version) == FIXTURE_GOLD_VERSION:  # already a fixture config (tests)
        return cfg
    root = Path(str(cfg.data.cache_dir)) / "eval_fixtures"
    fcfg = eval_fixture_config(root, list(getattr(args, "overrides", None) or []))
    if not DataLayout.from_config(fcfg).dry_manifest.exists():
        make_eval_fixture(fcfg, seconds=0.5)
    return fcfg


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(cfg, args) -> int:
    """``lstmabar eval proxy``: draft + render the val proxy distractors (resumable shards,
    ``eval.proxy.workers`` <= 2), then write the manifest and index. ``--limit N`` drafts the
    first N val items only (``<version>.partial.*``)."""
    from lstmabar.eval.data import load_split

    if getattr(args, "fixtures", False):
        cfg = _fixture_cfg(cfg, args)
    t0 = time.perf_counter()
    p = cfg.eval.proxy
    targets = load_tables(str(cfg.gold.targets_file))
    kb = render.load_pedals(cfg)
    dctx = DeriveContext.from_config(cfg)
    limit = getattr(args, "limit", None)
    all_val = load_split(cfg, PROXY_SPLIT)
    val = all_val[:limit] if limit else all_val
    partial = len(val) < len(all_val)
    items = [s.item for s in val]
    specs, meta, stats = plan(items, targets, kb, dctx, cfg)
    t_plan = time.perf_counter() - t0
    print(f"val proxy: {len(items)} items -> {len(specs)} distractors ({t_plan:.0f} s drafting)")

    pcfg = proxy_cfg(cfg)
    opts = render._options(pcfg, args)
    render._preflight(pcfg, opts, kb)
    layout = DataLayout.from_config(pcfg)
    dry_manifest = DataLayout.from_config(cfg).dry_manifest  # the dataset's clips (v1)
    clips = {c.clip_id: c for c in read_jsonl(dry_manifest, DryClip)}
    shards = render.shard_specs(specs, int(p.get("shard_size", 100)))
    workers = int(getattr(args, "workers", None) or p.get("workers", 2))
    workers = max(1, min(workers, 2))  # 16 GB machine
    gold._render_jobs(pcfg, opts, shards, clips, workers)
    t_render = time.perf_counter() - t0 - t_plan

    rendered: dict[str, RenderItem] = {}
    for i in range(len(shards)):
        for it in read_jsonl(layout.shard_dir(i) / render.ITEMS_FILE, RenderItem):
            rendered[it.item_id] = it
    records = [
        _record(it, region, k, d, rendered[f"{it.item_id}-d{k}"]) for it, region, k, d in meta
    ]
    write_outputs(cfg, records, items, targets, stats, partial)
    total = time.perf_counter() - t0
    print(f"timing: drafting {t_plan:.0f} s, rendering {t_render:.0f} s, total {total:.0f} s")
    return 0


def level_counts(items: Sequence[RenderItem], targets: Targets) -> dict[str, dict[str, int]]:
    """Label-region levels of the val items per axis (plus region kind)."""
    out: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        r = label_to_region(it, targets)
        out["kind"][r.kind if r.kind == "free" else f"pedal:{r.pedal}"] += 1
        for axis, lvl in region_levels(r).items():
            out[axis][lvl] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def write_outputs(cfg, records, items, targets: Targets, stats, partial: bool) -> None:
    """Write the manifest (one distractor :class:`gold.GoldRender` per line; targets are the
    val items themselves, built by :func:`load_proxy_items`) and its index."""
    manifest, index_path = manifest_paths(cfg, partial)
    text = "".join(
        json.dumps(gold._clean(r.to_dict()), sort_keys=True, ensure_ascii=False) + "\n"
        for r in records
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(text, encoding="utf-8", newline="\n")
    files = "".join(f"{r.render['sha256']}  {r.render['wet_path']}\n" for r in records)
    layout = DataLayout.from_config(cfg)
    dataset_sha = None
    if layout.index.exists():
        dataset_sha = json.loads(layout.index.read_text(encoding="utf-8")).get("dataset_sha256")
    tables = {"tables": gold._plain(targets.tables), "pedals": _pedals_plain(targets)}
    p = cfg.eval.proxy
    index: dict[str, Any] = {
        "version": str(p.version),
        "partial": partial,
        "dataset_version": str(cfg.data.version),
        "dataset_sha256": dataset_sha,
        "val_items": len(items),
        "val_items_sha256": _sha256_text("".join(f"{it.item_id} {it.sha256}\n" for it in items)),
        "targets_tables_sha256": _sha256_text(json.dumps(tables, sort_keys=True)),
        "settings": {
            "distractors_per_item": int(p.distractors_per_item),
            "seed": _seed(cfg),
            "gain_split_db": float(cfg.gold.gain_split_db),
            "min_gap": OmegaConf.to_container(cfg.gold.min_gap, resolve=True),
            "whitebox_pedals": [str(x) for x in cfg.render.whitebox_pedals],
        },
        "label_levels": level_counts(items, targets),
        "counts": {k: dict(sorted(v.items())) for k, v in sorted(stats.items())},
        "renders": len(records),
        "manifest_sha256": _sha256_text(text),
        "files_sha256": _sha256_text(files),
    }
    index_path.write_text(
        json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {manifest.name} ({len(records)} renders) and {index_path.name}")
    print(json.dumps(index["counts"], indent=1, sort_keys=True))


def _pedals_plain(targets: Targets) -> dict[str, Any]:
    return {
        pid: {
            "gain_knob": t.gain_knob,
            "tone_knob": t.tone_knob,
            "fixed": dict(t.fixed),
            "gain": {k: list(v) for k, v in t.gain.items()},
            "tone": {k: list(v) for k, v in t.tone.items()},
        }
        for pid, t in sorted(targets.pedals.items())
    }


__all__ = [
    "PROXY_SOURCE",
    "free_region",
    "item_distractors",
    "label_concrete",
    "label_to_region",
    "load_proxy_items",
    "load_tables",
    "manifest_paths",
    "plan",
    "proxy_cfg",
    "region_levels",
    "run",
    "selection_score",
    "selection_terms",
]
