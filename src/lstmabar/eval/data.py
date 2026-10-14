"""P5 evaluation data (wave 0 contract): synthetic splits with captions, gold items grouped by
item and clip, and ``test_real``.

Every evaluated input is a :class:`Query` (one instruction text on one dry input at one input
level), which is what methods (:mod:`lstmabar.eval.baselines.base`) predict from.

Level rule (D1): synthetic ``val`` / ``test`` / ``test_unseen_pedal`` queries use the item's own
``input_peak_dbfs`` (the target was rendered there); gold queries use
``audio.inference_peak_dbfs`` (−15 dBFS, R3), the level the gold targets were rendered at.

Embedding cache keys (D5): text = ``caption_id`` (synthetic) or the gold id; audio =
``item_id`` (synthetic: the dry clip at that item's level) or :func:`gold_audio_key` (a test
clip at the inference level). ``test_real`` has no instruction text and no queries (D11).

Reading ``data/gold/`` here is evaluation, not caption work (design §5): this module must never
be imported by caption code.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lstmabar.data import levels
from lstmabar.data.gold import GoldRender, load_gold, read_manifest
from lstmabar.data.layout import DONE_MARKER, DataLayout
from lstmabar.data.schema import (
    SPLITS,
    Caption,
    DryClip,
    FloatBoard,
    RenderItem,
    TestRealItem,
    read_jsonl,
)

ITEMS_FILE = "items.jsonl"
CAPTIONS_FILE = "captions.jsonl"

GOLD_SUBSETS: tuple[str, ...] = ("all", "headline", "ambiguous", "out_of_scope", "unseen_pedal")
"""``headline`` = in scope, not ambiguous and not an unseen-pedal item (D2); the other three
are reported separately. An item can be in several of the non-headline subsets."""


def gold_audio_key(clip_id: str, peak_dbfs: float) -> str:
    """Embedding-cache id of a test clip at a given input level (gold and demo inputs)."""
    return f"{clip_id}@{peak_dbfs:+.1f}"


@dataclass(frozen=True)
class Query:
    """One instruction on one dry input. ``dry_path`` is a local path: never report it."""

    query_id: str  # caption_id (synthetic) | "<gold_id>-c<clip_index>" (gold)
    unit_id: str  # the statistical unit (D3): item_id | gold_id
    text: str
    text_key: str
    audio_key: str
    clip_id: str
    input_peak_dbfs: float
    sample_rate: int
    dry_path: Path = field(repr=False)

    def load_dry(self) -> np.ndarray:
        """The stored dry clip scaled to ``input_peak_dbfs`` (mono float64)."""
        x, sr = levels.read_audio(self.dry_path)
        if sr != self.sample_rate:
            raise ValueError(f"{self.clip_id}: sample rate {sr}, expected {self.sample_rate}")
        return levels.apply_input_level(x, self.input_peak_dbfs)


# ============================================================================ synthetic splits


@dataclass(frozen=True)
class SynthItem:
    item: RenderItem
    captions: tuple[Caption, ...]
    dry_path: Path = field(repr=False)
    wet_path: Path = field(repr=False)
    sample_rate: int = 44100

    @property
    def item_id(self) -> str:
        return self.item.item_id

    @property
    def label(self) -> FloatBoard:
        """Normalized point label with gates (mask knobs by ``"enabled"``)."""
        return self.item.board_params

    def queries(self) -> list[Query]:
        return [
            Query(
                query_id=c.caption_id,
                unit_id=self.item.item_id,
                text=c.text,
                text_key=c.caption_id,
                audio_key=self.item.item_id,
                clip_id=self.item.clip_id,
                input_peak_dbfs=float(self.item.input_peak_dbfs),
                sample_rate=self.sample_rate,
                dry_path=self.dry_path,
            )
            for c in self.captions
        ]

    def load_dry(self) -> np.ndarray:
        """Dry clip at the item's own input level (what the target was rendered from)."""
        x, _ = levels.read_audio(self.dry_path)
        return levels.apply_input_level(x, float(self.item.input_peak_dbfs))

    def load_wet(self) -> np.ndarray:
        """The stored wet render (peak-normalized to ``render.wet_peak_dbfs``)."""
        return levels.read_audio(self.wet_path)[0]


def shard_dirs(layout: DataLayout) -> list[Path]:
    """Finished wet shards (``DONE`` present) in order."""
    if not layout.wet_dir.exists():
        return []
    return sorted(
        d
        for d in layout.wet_dir.iterdir()
        if d.is_dir() and (d / DONE_MARKER).exists() and (d / ITEMS_FILE).exists()
    )


def dry_clips(layout: DataLayout) -> dict[str, DryClip]:
    return {c.clip_id: c for c in read_jsonl(layout.dry_manifest, DryClip)}


def iter_items(layout: DataLayout, splits: Sequence[str] | None = None) -> Iterator[RenderItem]:
    want = set(splits) if splits is not None else None
    for d in shard_dirs(layout):
        for it in read_jsonl(d / ITEMS_FILE, RenderItem):
            if want is None or it.split in want:
                yield it


def load_split(cfg, split: str, limit: int | None = None) -> list[SynthItem]:
    """Items of one synthetic split with their captions, sorted by ``item_id``.

    ``limit`` keeps the first N items in that order (smoke runs). Raises when the split is
    unknown or has no items (an unrendered dataset should fail loudly, not report n = 0).
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    layout = DataLayout.from_config(cfg)
    sr = int(cfg.audio.sample_rate)
    clips = dry_clips(layout)
    items: dict[str, RenderItem] = {}
    caps: dict[str, list[Caption]] = defaultdict(list)
    for d in shard_dirs(layout):
        for it in read_jsonl(d / ITEMS_FILE, RenderItem):
            if it.split == split:
                items[it.item_id] = it
        cpath = d / CAPTIONS_FILE
        if cpath.exists():
            for c in read_jsonl(cpath, Caption):
                if c.item_id in items:
                    caps[c.item_id].append(c)
    if not items:
        raise FileNotFoundError(
            f"no rendered items for split {split!r} under {layout.wet_dir.name}"
        )
    out: list[SynthItem] = []
    for iid in sorted(items):
        it = items[iid]
        out.append(
            SynthItem(
                item=it,
                captions=tuple(sorted(caps.get(iid, []), key=lambda c: c.caption_id)),
                dry_path=layout.resolve_rendered(clips[it.clip_id].path),
                wet_path=layout.resolve_rendered(it.wet_path),
                sample_rate=sr,
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


# ============================================================================ gold


@dataclass(frozen=True)
class GoldClip:
    clip_index: int
    clip_id: str
    target: GoldRender
    distractors: tuple[GoldRender, ...]  # ordered by distractor_index
    dry_path: Path = field(repr=False)


@dataclass(frozen=True)
class GoldItem:
    gold_id: str
    source: str  # gold_source, never relabeled
    instruction: str
    in_scope: bool
    ambiguous: bool
    unseen_pedal: bool
    confidence: str
    region: dict  # the target region (targets-file schema)
    clips: tuple[GoldClip, ...]
    input_peak_dbfs: float = -15.0
    sample_rate: int = 44100

    def subsets(self) -> list[str]:
        """The :data:`GOLD_SUBSETS` this item belongs to (always includes ``"all"``)."""
        out = ["all"]
        if self.in_scope and not self.ambiguous and not self.unseen_pedal:
            out.append("headline")
        if self.ambiguous:
            out.append("ambiguous")
        if not self.in_scope:
            out.append("out_of_scope")
        if self.unseen_pedal:
            out.append("unseen_pedal")
        return out

    def queries(self) -> list[Query]:
        """One query per clip (same text, different dry input)."""
        return [
            Query(
                query_id=f"{self.gold_id}-c{c.clip_index}",
                unit_id=self.gold_id,
                text=self.instruction,
                text_key=self.gold_id,
                audio_key=gold_audio_key(c.clip_id, self.input_peak_dbfs),
                clip_id=c.clip_id,
                input_peak_dbfs=self.input_peak_dbfs,
                sample_rate=self.sample_rate,
                dry_path=c.dry_path,
            )
            for c in self.clips
        ]


def load_gold_items(cfg, limit: int | None = None) -> list[GoldItem]:
    """Gold items from ``eval.gold_manifest`` joined with their instruction text, sorted by id.

    Only items with a rendered target are returned (unlabelled friend items have none yet).
    """
    layout = DataLayout.from_config(cfg)
    renders = read_manifest(str(cfg.eval.gold_manifest))
    texts, _ = load_gold(str(cfg.gold.gold_dir))
    clips = dry_clips(layout)
    peak = float(cfg.audio.inference_peak_dbfs)
    sr = int(cfg.audio.sample_rate)
    by_item: dict[str, dict[int, list[GoldRender]]] = defaultdict(lambda: defaultdict(list))
    for r in renders:
        by_item[r.gold_id][r.clip_index].append(r)
    out: list[GoldItem] = []
    for gid in sorted(by_item):
        gclips: list[GoldClip] = []
        first: GoldRender | None = None
        for ci in sorted(by_item[gid]):
            rs = by_item[gid][ci]
            targets = [r for r in rs if r.role == "target"]
            if len(targets) != 1:
                raise ValueError(f"gold {gid} clip {ci}: expected one target, got {len(targets)}")
            t = targets[0]
            first = first or t
            ds = sorted((r for r in rs if r.role == "distractor"), key=lambda r: r.distractor_index)
            gclips.append(
                GoldClip(
                    clip_index=ci,
                    clip_id=t.clip_id,
                    target=t,
                    distractors=tuple(ds),
                    dry_path=layout.resolve_rendered(clips[t.clip_id].path),
                )
            )
        assert first is not None
        if gid not in texts:
            raise KeyError(f"gold {gid} has renders but no instruction")
        out.append(
            GoldItem(
                gold_id=gid,
                source=first.gold_source,
                instruction=str(texts[gid]["instruction"]),
                in_scope=bool(first.in_scope),
                ambiguous=bool(first.ambiguous),
                unseen_pedal=bool(first.unseen_pedal),
                confidence=first.confidence,
                region=dict(first.region),
                clips=tuple(gclips),
                input_peak_dbfs=peak,
                sample_rate=sr,
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def load_render_audio(cfg, render: GoldRender | RenderItem) -> np.ndarray:
    """A gold render's (or any ``RenderItem``'s) stored wet audio."""
    layout = DataLayout.from_config(cfg)
    rel = render.render["wet_path"] if isinstance(render, GoldRender) else render.wet_path
    return levels.read_audio(layout.resolve_rendered(rel))[0]


# ============================================================================ test_real


@dataclass(frozen=True)
class RealItem:
    item: TestRealItem
    dry_path: Path = field(repr=False)
    wet_path: Path = field(repr=False)

    @property
    def internal_only(self) -> bool:
        """Never publish per-item audio or numbers for these (aggregate only, D6)."""
        return bool(self.item.internal_only)

    def load_dry(self) -> tuple[np.ndarray, int]:
        return levels.read_audio(self.dry_path)

    def load_wet(self) -> tuple[np.ndarray, int]:
        return levels.read_audio(self.wet_path)


def load_test_real(cfg, limit: int | None = None) -> list[RealItem]:
    layout = DataLayout.from_config(cfg)
    out = [
        RealItem(
            item=it,
            dry_path=layout.test_real / it.dry_path,
            wet_path=layout.test_real / it.wet_path,
        )
        for it in sorted(
            read_jsonl(layout.test_real_manifest, TestRealItem), key=lambda r: r.item_id
        )
    ]
    return out[:limit] if limit is not None else out


__all__ = [
    "GOLD_SUBSETS",
    "GoldClip",
    "GoldItem",
    "Query",
    "RealItem",
    "SynthItem",
    "dry_clips",
    "gold_audio_key",
    "iter_items",
    "load_gold_items",
    "load_render_audio",
    "load_split",
    "load_test_real",
    "shard_dirs",
]
