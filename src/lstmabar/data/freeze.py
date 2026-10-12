"""Task 4.6 (wave 2): freeze dataset v1 — integrity checks, content hashes, index, data card.

``run(cfg, args)`` (``lstmabar data freeze``) loads the dry-clip manifest, every finished
shard's ``items.jsonl`` / ``captions.jsonl`` and, if present, the ``test_real`` manifest, and
**fails loudly** (nothing is written) unless all of these hold:

- **Splits** (decision 11 as revised by R11): clip ids are unique; every clip split is
  ``train`` / ``val`` / ``test``; no ``recording_id`` appears in two splits; no ``test``
  performer appears in ``train`` (test is the performer-disjoint set; val may share
  performers with train under R11, which the index records as ``performer_overlap``).
- **Items**: ids unique; each item's clip exists; ``item.split == clip.split``, except
  ``test_unseen_pedal`` items, which must sit on ``test`` clips and render the holdout pedal
  (``render.holdout_pedal``); the holdout pedal never appears in any other split.
- **Files**: every dry and wet file exists and its SHA-256 matches its record; no stray audio
  in shard dirs; ``test_real`` files exist and match ``sha256_dry`` / ``sha256_wet``.
- **Captions**: ids unique, each refers to an existing item, every item has a caption, and
  every caption passes :func:`lstmabar.data.vocab.item_caption_violations` (no numbers, units,
  brands, forbidden or holdout pedal names; the rule shared with the caption post-filter).
  The caption's own ``names_pedal`` flag is used, so a caption flagged as not naming its
  pedal may not contain any pedal name.

Shards without ``DONE`` (or without ``captions.jsonl``) are an error unless
``--allow-partial``, which skips them and records them in the index (``partial: true``).

Outputs (all deterministic for a given dataset and commit; sorted keys, no timestamps):

- ``DataLayout.file_hashes``: ``"<sha256>  <relpath>"`` per dry and wet file, sorted by path,
  paths relative to ``data.rendered_dir``. **``test_real`` audio is not listed** there: its
  per-file hashes already live in ``test_real.jsonl`` (verified here), and that manifest's
  hash enters the dataset hash. Listing hashes implies no redistribution (internal-only audio
  is never published; see the data card).
- ``DataLayout.index``: schema version, dataset hash (SHA-256 over the sorted file-hash lines
  plus the hashes of every manifest), per-shard hashes, counts per split / scenario / engine /
  pedal / label source, clamp and calibration-fallback counts, caption counts, level and
  voicing summaries, a config subset and the git SHA.
- ``DATA_CARD.md`` (``freeze.data_card``), regenerated from the index by
  :func:`lstmabar.data.card.render_data_card`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from lstmabar.data.layout import DONE_MARKER, DataLayout
from lstmabar.data.levels import SILENCE_DBFS, sha256_file
from lstmabar.data.schema import (
    FREE_SCENARIOS,
    SCHEMA_VERSION,
    Caption,
    DryClip,
    RenderItem,
    TestRealItem,
    read_jsonl,
)
from lstmabar.data.vocab import item_caption_violations

ITEMS_FILE = "items.jsonl"
CAPTIONS_FILE = "captions.jsonl"
UNSEEN_SPLIT = "test_unseen_pedal"
CLIP_SPLITS = ("train", "val", "test")
ITEM_SPLITS = (*CLIP_SPLITS, UNSEEN_SPLIT)
SCENARIOS = ("pedal", *FREE_SCENARIOS)
ENGINES = ("whitebox", "greybox")
LABEL_SOURCES = ("derived", "calibrated", "calibrated_nearest", "free")
AUDIO_SUFFIXES = (".flac", ".wav")
MAX_LISTED = 25  # issues printed per check


class FreezeError(RuntimeError):
    """Raised when a freeze check fails; nothing is written."""


@dataclass
class Shard:
    name: str
    items: list[RenderItem]
    captions: list[Caption]
    hashes: dict[str, str]  # manifest file name -> sha256


@dataclass
class Dataset:
    clips: list[DryClip]
    shards: list[Shard]
    test_real: list[TestRealItem] | None
    skipped_shards: list[str] = field(default_factory=list)

    @property
    def items(self) -> list[RenderItem]:
        return [it for s in self.shards for it in s.items]

    @property
    def captions(self) -> list[Caption]:
        return [c for s in self.shards for c in s.captions]


class Issues:
    """Collects check failures by kind so one run reports all of them."""

    def __init__(self) -> None:
        self.by_kind: dict[str, list[str]] = defaultdict(list)

    def add(self, kind: str, msg: str) -> None:
        self.by_kind[kind].append(msg)

    def __bool__(self) -> bool:
        return bool(self.by_kind)

    def report(self) -> str:
        lines = []
        for kind in sorted(self.by_kind):
            msgs = self.by_kind[kind]
            lines.append(f"{kind}: {len(msgs)}")
            lines += [f"  - {m}" for m in msgs[:MAX_LISTED]]
            if len(msgs) > MAX_LISTED:
                lines.append(f"  ... and {len(msgs) - MAX_LISTED} more")
        return "\n".join(lines)


# ============================================================================ loading


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_dataset(layout: DataLayout, allow_partial: bool = False) -> Dataset:
    """Dry manifest + finished shards + optional ``test_real`` manifest."""
    if not layout.dry_manifest.exists():
        raise FreezeError("dry-clip manifest not found: run `lstmabar data ingest` first")
    clips = list(read_jsonl(layout.dry_manifest, DryClip))
    shards: list[Shard] = []
    skipped: list[str] = []
    problems: list[str] = []
    dirs = sorted(layout.wet_dir.glob("shard-*")) if layout.wet_dir.exists() else []
    for d in dirs:
        if not d.is_dir():
            continue
        missing = [f for f in (DONE_MARKER, ITEMS_FILE, CAPTIONS_FILE) if not (d / f).exists()]
        if missing:
            (skipped if allow_partial else problems).append(f"{d.name} (missing {missing})")
            continue
        items = list(read_jsonl(d / ITEMS_FILE, RenderItem))
        caps = list(read_jsonl(d / CAPTIONS_FILE, Caption))
        _check_done(d, len(items), problems)
        hashes = {f: sha256_file(d / f) for f in (ITEMS_FILE, CAPTIONS_FILE)}
        shards.append(Shard(d.name, items, caps, hashes))
    if problems:
        raise FreezeError(
            "unfinished shards (render/captions not complete; --allow-partial skips them):\n"
            + "\n".join(f"  - {p}" for p in problems)
        )
    if not shards:
        raise FreezeError("no finished shards: run `lstmabar data render` and `captions` first")
    test_real = None
    if layout.test_real_manifest.exists():
        test_real = list(read_jsonl(layout.test_real_manifest, TestRealItem))
    return Dataset(clips, shards, test_real, skipped)


def _check_done(shard_dir: Path, n_items: int, problems: list[str]) -> None:
    """The renderer's ``DONE`` holds ``{"plan_sha256", "items"}``; check the count if given."""
    try:
        done = json.loads((shard_dir / DONE_MARKER).read_text(encoding="utf-8") or "{}")
    except ValueError:
        return  # a bare marker carries no count
    if isinstance(done, Mapping) and "items" in done and int(done["items"]) != n_items:
        problems.append(f"{shard_dir.name}: DONE says {done['items']} items, found {n_items}")


# ============================================================================ checks


def check_splits(clips: Sequence[DryClip], issues: Issues) -> None:
    ids = Counter(c.clip_id for c in clips)
    for cid, n in sorted(ids.items()):
        if n > 1:
            issues.add("duplicate clip_id", f"{cid} x{n}")
    rec_splits: dict[str, set[str]] = defaultdict(set)
    perf_splits: dict[str, set[str]] = defaultdict(set)
    for c in clips:
        if c.split not in CLIP_SPLITS:
            issues.add("bad clip split", f"{c.clip_id}: {c.split!r}")
        rec_splits[c.recording_id].add(c.split)
        perf_splits[c.performer].add(c.split)
    for rec, splits in sorted(rec_splits.items()):
        if len(splits) > 1:
            issues.add("recording in several splits", f"{rec}: {sorted(splits)}")
    for perf, splits in sorted(perf_splits.items()):
        if {"train", "test"} <= splits:
            issues.add("test performer in train", f"{perf}")


def check_items(
    items: Sequence[RenderItem], clips: Mapping[str, DryClip], holdout: str | None, issues: Issues
) -> None:
    ids = Counter(it.item_id for it in items)
    for iid, n in sorted(ids.items()):
        if n > 1:
            issues.add("duplicate item_id", f"{iid} x{n}")
    for it in items:
        clip = clips.get(it.clip_id)
        if clip is None:
            issues.add("item without clip", f"{it.item_id}: clip {it.clip_id!r}")
            continue
        if it.split not in ITEM_SPLITS:
            issues.add("bad item split", f"{it.item_id}: {it.split!r}")
        elif it.split == UNSEEN_SPLIT:
            if clip.split != "test":
                issues.add("unseen-pedal item off test", f"{it.item_id} on {clip.split} clip")
            if holdout is None or it.pedal_id != holdout:
                issues.add("unseen-pedal item without holdout", f"{it.item_id}: {it.pedal_id!r}")
        elif it.split != clip.split:
            issues.add("item/clip split mismatch", f"{it.item_id}: {it.split} vs {clip.split}")
        if holdout and it.pedal_id == holdout and it.split != UNSEEN_SPLIT:
            issues.add("holdout pedal outside test_unseen_pedal", f"{it.item_id} ({it.split})")
        if it.scenario not in SCENARIOS:
            issues.add("bad scenario", f"{it.item_id}: {it.scenario!r}")
        elif (it.scenario == "pedal") != (it.pedal_id is not None):
            issues.add("pedal_id/scenario mismatch", f"{it.item_id}")
        if it.engine not in ENGINES:
            issues.add("bad engine", f"{it.item_id}: {it.engine!r}")
        if it.label_source not in LABEL_SOURCES:
            issues.add("bad label_source", f"{it.item_id}: {it.label_source!r}")
        if not it.item_id.startswith(f"{it.clip_id}-"):
            issues.add("item id not prefixed by clip id", it.item_id)


def check_captions(
    captions: Sequence[Caption],
    items: Mapping[str, RenderItem],
    holdout: str | None,
    issues: Issues,
) -> None:
    ids = Counter(c.caption_id for c in captions)
    for cid, n in sorted(ids.items()):
        if n > 1:
            issues.add("duplicate caption_id", f"{cid} x{n}")
    covered = set()
    for c in captions:
        it = items.get(c.item_id)
        if it is None:
            issues.add("caption without item", f"{c.caption_id}: item {c.item_id!r}")
            continue
        covered.add(c.item_id)
        bad = item_caption_violations(c.text, it.pedal_id, c.names_pedal, holdout, it.split)
        if bad:
            issues.add("caption rule violation", f"{c.caption_id}: {'; '.join(bad)}")
        if c.kind == "paraphrase" and not (c.model and c.prompt_version):
            issues.add("paraphrase without model/prompt version", c.caption_id)
        if c.kind not in ("template", "paraphrase"):
            issues.add("bad caption kind", f"{c.caption_id}: {c.kind!r}")
    for iid in sorted(set(items) - covered):
        issues.add("item without captions", iid)


def check_files(
    layout: DataLayout, ds: Dataset, issues: Issues, allow_partial: bool = False
) -> dict[str, str]:
    """Hash every dry and wet file against its record; returns ``{relpath: sha256}``."""
    hashes: dict[str, str] = {}
    records = [(c.path, c.sha256, c.clip_id) for c in ds.clips]
    records += [(it.wet_path, it.sha256, it.item_id) for it in ds.items]
    for rel, expected, rid in records:
        path = layout.resolve_rendered(rel)
        if not path.is_file():
            issues.add("missing file", f"{rid}: {rel}")
            continue
        if rel in hashes:
            issues.add("file shared by records", f"{rid}: {rel}")
            continue
        actual = sha256_file(path)
        hashes[rel] = actual
        if actual != expected:
            issues.add("hash mismatch", f"{rid}: {rel}")
    for s in ds.shards:
        sdir = layout.wet_dir / s.name
        for f in sorted(sdir.iterdir()):
            if f.suffix.lower() in AUDIO_SUFFIXES:
                rel = f.relative_to(layout.rendered).as_posix()
                if rel not in hashes:
                    issues.add("stray audio file", rel)
    for t in ds.test_real or []:
        for rel, expected in ((t.dry_path, t.sha256_dry), (t.wet_path, t.sha256_wet)):
            path = layout.test_real / rel
            if not path.is_file():
                if not allow_partial:
                    issues.add("missing test_real file", f"{t.item_id}: {rel}")
            elif expected and sha256_file(path) != expected:
                issues.add("test_real hash mismatch", f"{t.item_id}: {rel}")
    return hashes


# ============================================================================ index


def _round(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


def summarize(values: Iterable[float | None]) -> dict[str, Any]:
    """``{n, n_missing, min, p05, median, p95, max, mean}`` over the finite values."""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0}
    a = np.asarray(vals)
    q = np.percentile(a, [5, 50, 95])
    return {
        "n": len(vals),
        "min": _round(a.min()),
        "p05": _round(q[0]),
        "median": _round(q[1]),
        "p95": _round(q[2]),
        "max": _round(a.max()),
        "mean": _round(a.mean()),
    }


def _counts(keys: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(keys).items()))


def _nested(pairs: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter] = defaultdict(Counter)
    for a, b in pairs:
        out[a][b] += 1
    return {a: dict(sorted(c.items())) for a, c in sorted(out.items())}


def _pedal_key(it: RenderItem) -> str:
    return it.pedal_id if it.scenario == "pedal" and it.pedal_id else f"free:{it.scenario}"


def clip_rms_dbfs(path: Path) -> float | None:
    """RMS of a stored dry clip in dBFS (``None`` if unreadable)."""
    try:
        x, _ = sf.read(str(path), dtype="float64", always_2d=False)
    except (OSError, RuntimeError):
        return None
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    return 20.0 * math.log10(rms) if rms > 0 else SILENCE_DBFS


def dataset_hash(file_lines: Sequence[str], manifest_hashes: Mapping[str, str]) -> str:
    """SHA-256 over the sorted file-hash lines, then the sorted manifest-hash lines."""
    blob = "\n".join(sorted(file_lines)) + "\n--\n"
    blob += "\n".join(f"{h}  {name}" for name, h in sorted(manifest_hashes.items())) + "\n"
    return _sha256_bytes(blob.encode("utf-8"))


CONFIG_KEYS = ("seed", "audio", "data", "ingest", "render", "calibration", "analysis", "splits")
CAPTION_CONFIG_KEYS = (
    "model",
    "prompt_version",
    "templates_per_item",
    "paraphrases_per_item",
    "pedal_name_fraction",
)


def config_subset(cfg) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in CONFIG_KEYS:
        if k in cfg:
            v = OmegaConf.to_container(cfg[k], resolve=True) if k != "seed" else cfg[k]
            out[k] = v
    if "data" in out:  # absolute local paths never go into a committed file
        out["data"] = {"version": out["data"].get("version")}
    if "captions" in cfg:
        cc = cfg.captions
        out["captions"] = {k: cc[k] for k in CAPTION_CONFIG_KEYS if k in cc}
    return json.loads(json.dumps(out, sort_keys=True, default=str))


def build_index(
    cfg,
    layout: DataLayout,
    ds: Dataset,
    file_hashes: Mapping[str, str],
    git_sha: str | None = None,
) -> dict[str, Any]:
    items, caps, clips = ds.items, ds.captions, ds.clips
    holdout = _holdout(cfg)
    file_lines = [f"{h}  {p}" for p, h in sorted(file_hashes.items())]
    manifests: dict[str, str] = {
        f"manifests/{layout.version}/dry_clips.jsonl": sha256_file(layout.dry_manifest)
    }
    for s in ds.shards:
        for f, h in s.hashes.items():
            manifests[f"{layout.version}/wet/{s.name}/{f}"] = h
    if ds.test_real is not None:
        manifests["manifests/test_real.jsonl"] = sha256_file(layout.test_real_manifest)

    shard_hashes = {}
    for s in ds.shards:
        lines = [
            f"{file_hashes[it.wet_path]}  {it.wet_path}"
            for it in s.items
            if it.wet_path in file_hashes
        ]
        shard_hashes[s.name] = {
            "items": len(s.items),
            "captions": len(s.captions),
            "sha256": dataset_hash(lines, {f: h for f, h in s.hashes.items()}),
        }

    min_vf = float(cfg.analysis.min_voiced_fraction) if "analysis" in cfg else 0.3
    perf_rms: dict[str, list[float]] = defaultdict(list)
    for c in clips:
        r = clip_rms_dbfs(layout.resolve_rendered(c.path))
        if r is not None and r > SILENCE_DBFS:
            perf_rms[c.performer].append(r - c.source_gain_db)  # at the source level

    perf_splits: dict[str, set[str]] = defaultdict(set)
    for c in clips:
        perf_splits[c.performer].add(c.split)

    vf_by_clip = {}
    for it in items:
        vf_by_clip.setdefault(it.clip_id, it.voiced_fraction)
    vfs = list(vf_by_clip.values())
    n_vf_missing = sum(v is None for v in vfs)

    caps_by_item: dict[str, Counter] = defaultdict(Counter)
    for c in caps:
        caps_by_item[c.item_id][c.kind] += 1
    n_para = int(cfg.captions.paraphrases_per_item) if "captions" in cfg else 0
    item_split = {it.item_id: it.split for it in items}

    pedal_items = [it for it in items if it.scenario == "pedal"]
    free_items = [it for it in items if it.scenario != "pedal"]
    disabled = _nested(
        (it.scenario, b)
        for it in free_items
        for b, knobs in sorted(it.board_params.items())
        if float(knobs.get("enabled", 1.0)) < 0.5
    )

    index: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "version": layout.version,
        "dataset_sha256": dataset_hash(file_lines, manifests),
        "git_sha": git_sha,
        "partial": bool(ds.skipped_shards),
        "skipped_shards": sorted(ds.skipped_shards),
        "holdout_pedal": holdout,
        "files": {"count": len(file_lines), "manifests": dict(sorted(manifests.items()))},
        "shards": shard_hashes,
        "clips": {
            "total": len(clips),
            "by_split": _counts(c.split for c in clips),
            "by_source": _counts(c.source for c in clips),
            "by_performer_split": _nested((c.performer, c.split) for c in clips),
            "recordings_by_split": {
                s: len({c.recording_id for c in clips if c.split == s})
                for s in sorted({c.split for c in clips})
            },
            "by_technique": _counts(c.technique or "(none)" for c in clips),
            "performer_overlap": {
                p: sorted(s) for p, s in sorted(perf_splits.items()) if len(s) > 1
            },
            "source_gain_db_by_performer": {
                p: summarize(c.source_gain_db for c in clips if c.performer == p)
                for p in sorted(perf_splits)
            },
            "source_rms_dbfs_by_performer": {p: summarize(v) for p, v in sorted(perf_rms.items())},
            "duration_s": summarize(c.duration_s for c in clips),
        },
        "items": {
            "total": len(items),
            "by_split": _counts(it.split for it in items),
            "by_scenario": _counts(it.scenario for it in items),
            "by_engine": _counts(it.engine for it in items),
            "by_pedal": _counts(_pedal_key(it) for it in items),
            "by_label_source": _counts(it.label_source for it in items),
            "by_split_pedal": _nested((it.split, _pedal_key(it)) for it in items),
            "by_pedal_engine": _nested((_pedal_key(it), it.engine) for it in pedal_items),
            "clamped": {
                "total": sum(it.clamped for it in items),
                "by_pedal": _counts(it.pedal_id for it in pedal_items if it.clamped),
            },
            "label_clamped": {
                "total": sum(bool(it.label_clamp_notes) for it in items),
                "by_pedal": _counts(_pedal_key(it) for it in items if it.label_clamp_notes),
            },
            "calibrated_nearest": {
                "total": sum(it.label_source == "calibrated_nearest" for it in items),
                "by_pedal": _counts(
                    it.pedal_id for it in pedal_items if it.label_source == "calibrated_nearest"
                ),
            },
            "free_board_disabled_blocks": disabled,
            "input_peak_dbfs": summarize(it.input_peak_dbfs for it in items),
            "wet_gain_db": summarize(it.wet_gain_db for it in items),
            "renders_per_clip": summarize(Counter(it.clip_id for it in items).values()),
        },
        "voicing": {
            "clips_with_items": len(vf_by_clip),
            "voiced_fraction": summarize(vfs),
            "not_measured": n_vf_missing,
            "below_min": sum(v is not None and v < min_vf for v in vfs),
            "min_voiced_fraction": min_vf,
        },
        "captions": {
            "total": len(caps),
            "by_kind": _counts(c.kind for c in caps),
            "by_split": _counts(item_split.get(c.item_id, "?") for c in caps),
            "names_pedal": sum(c.names_pedal for c in caps),
            "models": _counts(
                f"{c.model}/{c.prompt_version}" for c in caps if c.kind == "paraphrase"
            ),
            "paraphrases_per_item": n_para,
            "items_template_only": sum(
                1 for it in items if caps_by_item[it.item_id]["paraphrase"] == 0
            ),
            "items_missing_paraphrases": sum(
                1 for it in items if caps_by_item[it.item_id]["paraphrase"] < n_para
            ),
        },
        "test_real": _test_real_summary(ds.test_real),
        "config": config_subset(cfg),
    }
    return json.loads(json.dumps(index, sort_keys=True))


def _test_real_summary(items: Sequence[TestRealItem] | None) -> dict[str, Any] | None:
    if items is None:
        return None
    return {
        "total": len(items),
        "by_dataset": _counts(t.dataset for t in items),
        "by_device": _counts(f"{t.dataset}/{t.device}" for t in items),
        "by_kb_pedal": _counts(t.kb_pedal_id or "(none)" for t in items),
        "internal_only": sum(t.internal_only for t in items),
        "sample_rates": _counts(str(t.sample_rate) for t in items),
        "licenses": _counts(f"{t.dataset}: {t.license}" for t in items),
    }


def _holdout(cfg) -> str | None:
    if "render" in cfg and cfg.render.get("holdout_pedal"):
        return str(cfg.render.holdout_pedal)
    return None


# ============================================================================ step


def verify(cfg, layout: DataLayout, allow_partial: bool = False) -> tuple[Dataset, dict[str, str]]:
    """Load and check everything; raises :class:`FreezeError` listing every failure."""
    ds = load_dataset(layout, allow_partial)
    holdout = _holdout(cfg)
    issues = Issues()
    check_splits(ds.clips, issues)
    clips = {c.clip_id: c for c in ds.clips}
    check_items(ds.items, clips, holdout, issues)
    check_captions(ds.captions, {it.item_id: it for it in ds.items}, holdout, issues)
    hashes = check_files(layout, ds, issues, allow_partial)
    if issues:
        raise FreezeError("freeze checks failed:\n" + issues.report())
    return ds, hashes


def write_outputs(layout: DataLayout, index: Mapping[str, Any], hashes: Mapping[str, str]) -> None:
    layout.file_hashes.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{h}  {p}\n" for p, h in sorted(hashes.items())]
    with layout.file_hashes.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    with layout.index.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(index, sort_keys=True, indent=2) + "\n")


def data_card_path(cfg) -> Path:
    if "freeze" in cfg and cfg.freeze.get("data_card"):
        return Path(str(cfg.freeze.data_card))
    return Path("DATA_CARD.md")


def run(cfg, args) -> int:
    from lstmabar.data.card import render_data_card
    from lstmabar.runs import git_info

    layout = DataLayout.from_config(cfg)
    allow_partial = bool(getattr(args, "allow_partial", False))
    try:
        ds, hashes = verify(cfg, layout, allow_partial)
    except FreezeError as e:
        print(str(e))
        return 1
    index = build_index(cfg, layout, ds, hashes, git_info().get("sha"))
    write_outputs(layout, index, hashes)
    card = data_card_path(cfg)
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(render_data_card(index), encoding="utf-8", newline="\n")
    print(
        f"frozen {layout.version}: {index['items']['total']} items, "
        f"{index['captions']['total']} captions, {index['files']['count']} files; "
        f"dataset sha256 {index['dataset_sha256']}"
    )
    if ds.skipped_shards:
        print(f"PARTIAL: skipped {len(ds.skipped_shards)} unfinished shards")
    return 0


__all__ = [
    "Dataset",
    "FreezeError",
    "build_index",
    "check_captions",
    "check_files",
    "check_items",
    "check_splits",
    "dataset_hash",
    "load_dataset",
    "run",
    "summarize",
    "verify",
    "write_outputs",
]
