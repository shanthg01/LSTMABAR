"""Task 4.5: gold pairing. Each gold instruction gets a target tone *region* and one or more
contradicting distractor regions; concrete settings sampled from them are rendered on the
held-out ``test`` DI clips for the P5 2AFC metric (design §7, P4 decision 7).

This module reads ``data/gold/`` (allowed only in the gold-pairing session); it never touches
caption code, ``data/manifests/v1/`` or the data card.

**Targets file** (``gold.targets_file``, owner-reviewed YAML): ``tables`` map the gold set's
coarse levels (gain, brightness, mids, compression, drive family/clipping) to physical ranges
of the free board (``default_pedalboard``: compressor -> drive -> eq); ``pedals`` map gain and
brightness levels to KB pedal knob rotations; ``items`` hold one row per gold id::

    - id: h006
      source: simulated_human          # copied from the gold file, never relabeled
      in_scope: true                   # copied from the gold target
      target: {kind: pedal, pedal: ts808, gain: medium, brightness: neutral}
      distractors:
      - {axis: gain, kind: free, gain: none, brightness: neutral, mids: neutral, compression: none}
      confidence: high                 # high | medium | low
      ambiguous: false
      ambiguity: []                    # relative | vague | conflicting
      unseen_pedal: false              # the holdout pedal (render.holdout_pedal) appears
      rationale: ...

A region is ``kind: free`` (``gain``, ``brightness``, ``mids``, ``compression`` levels plus
``family``/``clipping`` when it drives) or ``kind: pedal`` (a KB ``pedal`` with ``gain`` and
``brightness`` levels; its mids/compression are the pedal's own). ``overrides`` set explicit
ranges (``"eq.low_db": [lo, hi]`` in physical units for free regions, ``"<knob>": [lo, hi]``
rotations for pedals). :func:`expand` turns a region into concrete ranges; :func:`sample`
draws one setting uniformly in normalized-knob space.

**Distractor rules** (:func:`draft_distractors`; each distractor contradicts the target on one
``axis`` and keeps the rest): ``gain`` -> clean if the target's drive gain reaches
``gold.gain_split_db``, else high-gain hard-clipped drive; ``brightness`` -> the opposite
extreme (pedals: tone knob, only if the pedal has one); ``mids`` -> scooped <-> pushed (free
only); ``compression`` -> off (free only; heavy, or light when compression is the main ask).
EQ/compression-only requests list their own axes first, every other item starts with
``gain``. :func:`separation` checks that each distractor's range on its axis is disjoint from
the target's (gain via ``derive`` on a knob grid for pedals).

**Rendering** (``lstmabar data gold-pair``): per gold item ``gold.clips_per_item`` test clips
(ranked by a hash of the gold id, so items use different clips), at the inference input level
(``audio.inference_peak_dbfs``, R3); per clip one target sample and one sample per distractor
(first ``gold.distractors_per_item``), all on the same clip and level. Rendering reuses
:func:`lstmabar.data.render.render_shard` (white-box for ``render.whitebox_pedals`` with
calibrated labels, grey-box otherwise) with ``data.version = gold.version``, so audio lands in
``data/rendered/<gold.version>/`` (gitignored). The committed artefacts are
``data/manifests/<gold.version>.jsonl`` (:class:`GoldRender` per line) and
``data/manifests/<gold.version>.index.json`` (counts, file hashes, input hashes).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from lstmabar.data import render
from lstmabar.data.layout import DataLayout
from lstmabar.data.schema import DryClip, FloatBoard, RenderItem, read_jsonl, to_dict
from lstmabar.dsp.pedalboard import ENABLED
from lstmabar.physics.derive import BLOCK_SPECS, DeriveContext, derive
from lstmabar.physics.kb import Pedal

SOURCES: tuple[str, ...] = ("human", "simulated_human", "llm_persona", "llm_draft")
GAINS: tuple[str, ...] = ("none", "low", "medium", "high", "extreme")
BRIGHTNESS: tuple[str, ...] = ("dark", "slightly_dark", "neutral", "slightly_bright", "bright")
MIDS: tuple[str, ...] = ("scooped", "neutral", "pushed")
COMPRESSION: tuple[str, ...] = ("none", "light", "heavy")
FAMILIES: tuple[str, ...] = ("boost", "overdrive", "distortion", "fuzz")
CLIPPING: tuple[str, ...] = ("soft", "hard", "asymmetric")
AXES: tuple[str, ...] = ("gain", "brightness", "mids", "compression")
CONFIDENCE: tuple[str, ...] = ("high", "medium", "low")
AMBIGUITY: tuple[str, ...] = ("relative", "vague", "conflicting")
KINDS: tuple[str, ...] = ("free", "pedal")
ROLES: tuple[str, ...] = ("target", "distractor")
BLOCKS: tuple[str, ...] = ("compressor", "drive", "eq")
SPLIT = "gold"

Range = tuple[float, float]


class GoldError(ValueError):
    """The targets file is malformed; the message names the row and key."""


# ============================================================================ regions


@dataclass(frozen=True)
class Region:
    kind: str
    gain: str
    brightness: str = "neutral"
    mids: str = "neutral"
    compression: str = "none"
    family: str | None = None  # free drive only
    clipping: str | None = None  # free drive only (None: the family's default shaper)
    pedal: str | None = None  # kind == "pedal"
    overrides: Mapping[str, Range] = field(default_factory=dict)

    def to_yaml(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.kind == "pedal":
            out |= {"pedal": self.pedal, "gain": self.gain, "brightness": self.brightness}
        else:
            if self.gain != "none":
                out |= {"family": self.family}
                if self.clipping:
                    out["clipping"] = self.clipping
            out |= {
                "gain": self.gain,
                "brightness": self.brightness,
                "mids": self.mids,
                "compression": self.compression,
            }
        if self.overrides:
            out["overrides"] = {k: [float(a), float(b)] for k, (a, b) in self.overrides.items()}
        return out


@dataclass(frozen=True)
class Distractor:
    axis: str
    region: Region

    def to_yaml(self) -> dict[str, Any]:
        return {"axis": self.axis, **self.region.to_yaml()}


@dataclass(frozen=True)
class TargetRow:
    id: str
    source: str
    in_scope: bool
    target: Region
    distractors: tuple[Distractor, ...]
    confidence: str
    ambiguous: bool
    ambiguity: tuple[str, ...]
    unseen_pedal: bool
    rationale: str

    def to_yaml(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "in_scope": self.in_scope,
            "target": self.target.to_yaml(),
            "distractors": [d.to_yaml() for d in self.distractors],
            "confidence": self.confidence,
            "ambiguous": self.ambiguous,
            "ambiguity": list(self.ambiguity),
            "unseen_pedal": self.unseen_pedal,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class PedalTable:
    gain_knob: str
    tone_knob: str | None
    fixed: Mapping[str, float]
    gain: Mapping[str, Range]
    tone: Mapping[str, Range]  # brightness level -> tone knob range (empty: no tone knob)


@dataclass(frozen=True)
class Targets:
    version: int
    status: str
    tables: Mapping[str, Any]
    pedals: Mapping[str, PedalTable]
    items: tuple[TargetRow, ...]
    path: Path | None = None

    def by_id(self) -> dict[str, TargetRow]:
        return {r.id: r for r in self.items}


def _choice(v: Any, options: Sequence[str], where: str) -> str:
    if v not in options:
        raise GoldError(f"{where}: {v!r} is not one of {tuple(options)}")
    return str(v)


def _range(v: Any, where: str) -> Range:
    if (
        not isinstance(v, list | tuple)
        or len(v) != 2
        or not all(isinstance(x, int | float) and not isinstance(x, bool) for x in v)
    ):
        raise GoldError(f"{where}: expected [lo, hi], got {v!r}")
    lo, hi = float(v[0]), float(v[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        raise GoldError(f"{where}: invalid range [{lo}, {hi}]")
    return lo, hi


_REGION_KEYS = {
    "kind",
    "gain",
    "brightness",
    "mids",
    "compression",
    "family",
    "clipping",
    "pedal",
    "overrides",
}


def region_from_dict(d: Mapping[str, Any], where: str, extra: Iterable[str] = ()) -> Region:
    if not isinstance(d, Mapping):
        raise GoldError(f"{where}: expected a mapping")
    unknown = set(d) - _REGION_KEYS - set(extra)
    if unknown:
        raise GoldError(f"{where}: unknown keys {sorted(unknown)}")
    kind = _choice(d.get("kind"), KINDS, f"{where}.kind")
    gain = _choice(d.get("gain"), GAINS, f"{where}.gain")
    brightness = _choice(d.get("brightness", "neutral"), BRIGHTNESS, f"{where}.brightness")
    overrides = {
        str(k): _range(v, f"{where}.overrides.{k}") for k, v in (d.get("overrides") or {}).items()
    }
    if kind == "pedal":
        for k in ("mids", "compression", "family", "clipping"):
            if k in d:
                raise GoldError(f"{where}: {k!r} is set by the pedal itself (kind: pedal)")
        if gain == "none":
            raise GoldError(f"{where}: a pedal region needs a gain level, not 'none'")
        if not d.get("pedal"):
            raise GoldError(f"{where}.pedal: required for kind: pedal")
        return Region(
            kind,
            gain,
            brightness,
            pedal=str(d["pedal"]),
            overrides=overrides,
        )
    if "pedal" in d:
        raise GoldError(f"{where}: 'pedal' only with kind: pedal")
    mids = _choice(d.get("mids", "neutral"), MIDS, f"{where}.mids")
    compression = _choice(d.get("compression", "none"), COMPRESSION, f"{where}.compression")
    family = clipping = None
    if gain != "none":
        family = _choice(d.get("family"), FAMILIES, f"{where}.family")
        if d.get("clipping") is not None:
            clipping = _choice(d["clipping"], CLIPPING, f"{where}.clipping")
    elif "family" in d or "clipping" in d:
        raise GoldError(f"{where}: family/clipping only when the region drives (gain != none)")
    if family == "boost" and gain != "low":
        raise GoldError(f"{where}: family boost is only defined at gain 'low'")
    return Region(kind, gain, brightness, mids, compression, family, clipping, None, overrides)


def _pedal_table(pid: str, d: Mapping[str, Any], where: str) -> PedalTable:
    w = f"{where}.{pid}"
    gain = {
        _choice(k, GAINS[1:], f"{w}.gain"): _range(v, f"{w}.gain.{k}")
        for k, v in (d.get("gain") or {}).items()
    }
    if set(gain) != set(GAINS[1:]):
        raise GoldError(f"{w}.gain: needs every level of {GAINS[1:]}")
    tone_knob = d.get("tone_knob")
    tone = {
        _choice(k, BRIGHTNESS, f"{w}.tone"): _range(v, f"{w}.tone.{k}")
        for k, v in (d.get("tone") or {}).items()
    }
    if tone_knob and set(tone) != set(BRIGHTNESS):
        raise GoldError(f"{w}.tone: needs every level of {BRIGHTNESS}")
    if not tone_knob and tone:
        raise GoldError(f"{w}.tone: given without a tone_knob")
    for name, (lo, hi) in [*gain.items(), *tone.items()]:
        if lo < 0 or hi > 1:
            raise GoldError(f"{w}: knob range {name} [{lo}, {hi}] outside [0, 1]")
    return PedalTable(
        gain_knob=str(d["gain_knob"]),
        tone_knob=str(tone_knob) if tone_knob else None,
        fixed={str(k): float(v) for k, v in (d.get("fixed") or {}).items()},
        gain=gain,
        tone=tone,
    )


_ROW_KEYS = {
    "id",
    "source",
    "in_scope",
    "target",
    "distractors",
    "confidence",
    "ambiguous",
    "ambiguity",
    "unseen_pedal",
    "rationale",
}


def row_from_dict(d: Mapping[str, Any], where: str = "items") -> TargetRow:
    if not isinstance(d, Mapping):
        raise GoldError(f"{where}: expected a mapping")
    rid = str(d.get("id") or "")
    w = f"{where}[{rid or '?'}]"
    unknown = set(d) - _ROW_KEYS
    missing = _ROW_KEYS - set(d)
    if unknown or missing:
        raise GoldError(f"{w}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    dist = []
    for i, x in enumerate(d["distractors"] or []):
        if not isinstance(x, Mapping):
            raise GoldError(f"{w}.distractors[{i}]: expected a mapping")
        axis = _choice(x.get("axis"), AXES, f"{w}.distractors[{i}].axis")
        dist.append(Distractor(axis, region_from_dict(x, f"{w}.distractors[{i}]", ("axis",))))
    if not dist:
        raise GoldError(f"{w}.distractors: at least one distractor is required")
    ambiguity = tuple(_choice(a, AMBIGUITY, f"{w}.ambiguity") for a in d["ambiguity"] or [])
    ambiguous = d["ambiguous"]
    if not isinstance(ambiguous, bool) or ambiguous != bool(ambiguity):
        raise GoldError(f"{w}: ambiguous must be true exactly when ambiguity is non-empty")
    if not isinstance(d["in_scope"], bool) or not isinstance(d["unseen_pedal"], bool):
        raise GoldError(f"{w}: in_scope and unseen_pedal must be booleans")
    if not str(d["rationale"] or "").strip():
        raise GoldError(f"{w}.rationale: required")
    return TargetRow(
        id=rid,
        source=_choice(d["source"], SOURCES, f"{w}.source"),
        in_scope=d["in_scope"],
        target=region_from_dict(d["target"], f"{w}.target"),
        distractors=tuple(dist),
        confidence=_choice(d["confidence"], CONFIDENCE, f"{w}.confidence"),
        ambiguous=ambiguous,
        ambiguity=ambiguity,
        unseen_pedal=d["unseen_pedal"],
        rationale=str(d["rationale"]).strip(),
    )


def _check_tables(t: Mapping[str, Any]) -> None:
    def spec(block: str, knob: str):
        for s in BLOCK_SPECS[block]:
            if s.name == knob:
                return s
        raise GoldError(f"tables: unknown {block} knob {knob!r}")

    def within(block: str, knob: str, r: Any, where: str) -> Range:
        lo, hi = _range(r, where)
        s = spec(block, knob)
        if lo < s.min or hi > s.max:
            raise GoldError(f"{where}: [{lo}, {hi}] outside {block}.{knob} [{s.min}, {s.max}]")
        return lo, hi

    for k in ("gain_db", "boost_gain_db", "family", "clipping", "brightness", "mids"):
        if k not in t:
            raise GoldError(f"tables.{k}: missing")
    if set(t["gain_db"]) != set(GAINS[2:]) | {"low"}:
        raise GoldError(f"tables.gain_db: needs levels {GAINS[1:]}")
    for lvl, r in t["gain_db"].items():
        within("drive", "gain_db", r, f"tables.gain_db.{lvl}")
    within("drive", "gain_db", t["boost_gain_db"], "tables.boost_gain_db")
    if set(t["family"]) != set(FAMILIES):
        raise GoldError(f"tables.family: needs {FAMILIES}")
    for fam, knobs in t["family"].items():
        for knob, r in knobs.items():
            within("drive", knob, r, f"tables.family.{fam}.{knob}")
    if set(t["clipping"]) != set(CLIPPING):
        raise GoldError(f"tables.clipping: needs {CLIPPING}")
    for c, knobs in t["clipping"].items():
        for knob, r in knobs.items():
            within("drive", knob, r, f"tables.clipping.{c}.{knob}")
    for key, levels in (("brightness", BRIGHTNESS), ("mids", MIDS)):
        if set(t[key]) != set(levels):
            raise GoldError(f"tables.{key}: needs {levels}")
        for lvl, knobs in t[key].items():
            for knob, r in knobs.items():
                within("eq", knob, r, f"tables.{key}.{lvl}.{knob}")
    if set(t.get("compression") or {}) != set(COMPRESSION[1:]):
        raise GoldError(f"tables.compression: needs {COMPRESSION[1:]}")
    for lvl, knobs in t["compression"].items():
        for knob, r in knobs.items():
            within("compressor", knob, r, f"tables.compression.{lvl}.{knob}")
    for block, knobs in (t.get("fixed") or {}).items():
        for knob, r in knobs.items():
            within(block, knob, r, f"tables.fixed.{block}.{knob}")


def targets_from_dict(doc: Mapping[str, Any], path: Path | None = None) -> Targets:
    if not isinstance(doc, Mapping):
        raise GoldError("targets file: top level must be a mapping")
    unknown = set(doc) - {"version", "status", "tables", "pedals", "items"}
    if unknown:
        raise GoldError(f"targets file: unknown keys {sorted(unknown)}")
    tables = doc.get("tables") or {}
    _check_tables(tables)
    pedals = {
        str(pid): _pedal_table(str(pid), d, "pedals")
        for pid, d in (doc.get("pedals") or {}).items()
    }
    items = tuple(row_from_dict(r) for r in doc.get("items") or [])
    ids = [r.id for r in items]
    dup = sorted(k for k, n in Counter(ids).items() if n > 1)
    if dup:
        raise GoldError(f"items: duplicate ids {dup}")
    status = _choice(doc.get("status"), ("draft", "reviewed"), "status")
    out = Targets(int(doc.get("version", 1)), status, tables, pedals, items, path)
    for row in items:
        for i, reg in enumerate([row.target, *(d.region for d in row.distractors)]):
            expand(reg, out, where=f"items[{row.id}].{'target' if i == 0 else f'd{i - 1}'}")
    return out


def load_targets(path: str | Path) -> Targets:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        try:
            doc = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise GoldError(f"{path.name}: invalid YAML: {e}") from None
    return targets_from_dict(doc, path)


# ============================================================================ expansion


@dataclass(frozen=True)
class Concrete:
    """Concrete ranges of a region: pedal knob rotations, or free-board physical ranges with
    0/1 gates (every knob of every block, so a sample is a complete ``FloatBoard``)."""

    kind: str
    pedal: str | None = None
    knobs: Mapping[str, Range] = field(default_factory=dict)  # pedal rotations
    gates: Mapping[str, float] = field(default_factory=dict)  # free board
    ranges: Mapping[str, Mapping[str, Range]] = field(default_factory=dict)  # block -> knob

    @property
    def scenario(self) -> str:
        if self.kind == "pedal":
            return "pedal"
        on = {b for b, g in self.gates.items() if g}
        if "drive" in on:
            return "drive"
        return {
            frozenset(): "bypass",
            frozenset({"eq"}): "eq",
            frozenset({"compressor"}): "comp",
            frozenset({"compressor", "eq"}): "comp_eq",
        }[frozenset(on)]


def _r(v: Any) -> Range:
    return float(v[0]), float(v[1])


def expand(region: Region, targets: Targets, where: str = "region") -> Concrete:
    t = targets.tables
    if region.kind == "pedal":
        if region.pedal not in targets.pedals:
            raise GoldError(f"{where}: pedal {region.pedal!r} has no entry under pedals")
        p = targets.pedals[region.pedal]
        knobs: dict[str, Range] = {k: (v, v) for k, v in p.fixed.items()}
        knobs[p.gain_knob] = p.gain[region.gain]
        if p.tone_knob:
            knobs[p.tone_knob] = p.tone[region.brightness]
        elif region.brightness != "neutral":
            raise GoldError(f"{where}: {region.pedal} has no tone knob; brightness must be neutral")
        for k, r in region.overrides.items():
            if k not in knobs and k not in (p.gain_knob, p.tone_knob):
                raise GoldError(f"{where}.overrides: {k!r} is not a {region.pedal} knob")
            if r[0] < 0 or r[1] > 1:
                raise GoldError(f"{where}.overrides.{k}: rotation outside [0, 1]")
            knobs[k] = r
        return Concrete("pedal", pedal=region.pedal, knobs=knobs)

    ranges: dict[str, dict[str, Range]] = {
        b: {s.name: (s.default, s.default) for s in BLOCK_SPECS[b]} for b in BLOCKS
    }
    for block, knobs_ in (t.get("fixed") or {}).items():
        for knob, r in knobs_.items():
            ranges[block][knob] = _r(r)
    gates = {"compressor": 0.0, "drive": 0.0, "eq": 1.0}
    ranges["eq"] |= {k: _r(v) for k, v in t["brightness"][region.brightness].items()}
    ranges["eq"] |= {k: _r(v) for k, v in t["mids"][region.mids].items()}
    if region.compression != "none":
        gates["compressor"] = 1.0
        ranges["compressor"] |= {k: _r(v) for k, v in t["compression"][region.compression].items()}
    if region.gain != "none":
        gates["drive"] = 1.0
        ranges["drive"] |= {k: _r(v) for k, v in t["family"][region.family].items()}
        if region.clipping:
            ranges["drive"] |= {k: _r(v) for k, v in t["clipping"][region.clipping].items()}
        g = t["boost_gain_db"] if region.family == "boost" else t["gain_db"][region.gain]
        ranges["drive"]["gain_db"] = _r(g)
    for key, r in region.overrides.items():
        block, _, knob = key.partition(".")
        if block not in ranges or knob not in ranges[block]:
            raise GoldError(f"{where}.overrides: unknown board knob {key!r}")
        s = next(s for s in BLOCK_SPECS[block] if s.name == knob)
        if r[0] < s.min or r[1] > s.max:
            raise GoldError(f"{where}.overrides.{key}: outside [{s.min}, {s.max}]")
        ranges[block][knob] = r
    return Concrete("free", gates=gates, ranges=ranges)


def _spec(block: str, knob: str):
    return next(s for s in BLOCK_SPECS[block] if s.name == knob)


def _norm(block: str, knob: str, v: float) -> float:
    return float(_spec(block, knob).normalize(torch.tensor(float(v), dtype=torch.float64)))


def sample(
    c: Concrete, rng: np.random.Generator
) -> tuple[dict[str, float] | None, FloatBoard | None]:
    """One setting, uniform in normalized-knob space: ``(pedal_knobs, None)`` for a pedal
    region, ``(None, board)`` (normalized, gates 0./1.) for a free one. Draw order is fixed
    (sorted names), so a sample is a pure function of the generator state."""
    if c.kind == "pedal":
        return {k: float(rng.uniform(*c.knobs[k])) for k in sorted(c.knobs)}, None
    board: FloatBoard = {}
    for block in BLOCKS:
        vals: dict[str, float] = {}
        for knob in sorted(c.ranges[block]):
            lo, hi = c.ranges[block][knob]
            u_lo, u_hi = _norm(block, knob, lo), _norm(block, knob, hi)
            vals[knob] = float(rng.uniform(u_lo, u_hi)) if u_hi > u_lo else u_lo
        vals[ENABLED] = float(c.gates[block])
        board[block] = vals
    return None, board


# ============================================================================ separation

_PEDAL_GRID = 5


def _grid(r: Range, n: int = _PEDAL_GRID) -> list[float]:
    return [float(x) for x in np.linspace(r[0], r[1], n)] if r[1] > r[0] else [r[0]]


_DERIVED_GAIN: dict[tuple, float] = {}


def _derived_gain_db(pedal: Pedal, knobs: Mapping[str, float], ctx: DeriveContext) -> float:
    key = (pedal.id, ctx, tuple(sorted(knobs.items())))
    if key not in _DERIVED_GAIN:  # derive costs ~0.15 s; rows share grid points
        _DERIVED_GAIN[key] = float(derive(pedal, dict(knobs), ctx).board["drive"]["gain_db"])
    return _DERIVED_GAIN[key]


def gain_range_db(
    c: Concrete, kb: Mapping[str, Pedal], ctx: DeriveContext, gain_knob: str | None = None
) -> Range:
    """Drive gain into the clipper (dB): free board from its range (0 when drive is off),
    pedals from ``derive`` on a grid over ``gain_knob``'s range (the pre-clip gain does not
    depend on the tone/level knobs, left at their pot defaults)."""
    if c.kind == "free":
        return c.ranges["drive"]["gain_db"] if c.gates["drive"] else (0.0, 0.0)
    pedal = kb[c.pedal]
    if gain_knob:  # other knobs at their pot defaults: shared cache entries across rows
        base: dict[str, float] = {}
        knobs = [gain_knob]
    else:
        base = {k: round((lo + hi) / 2, 6) for k, (lo, hi) in c.knobs.items()}
        knobs = sorted(c.knobs)
    vals = []
    for k in knobs:
        for x in _grid(c.knobs[k]):
            vals.append(_derived_gain_db(pedal, base | {k: round(x, 6)}, ctx))
    return float(min(vals)), float(max(vals))


def _gain_knob(c: Concrete, targets: Targets) -> str | None:
    return targets.pedals[c.pedal].gain_knob if c.kind == "pedal" else None


def _gap(a: Range, b: Range) -> float:
    """Distance between two intervals (negative = overlap depth)."""
    return max(a[0] - b[1], b[0] - a[1])


def separation(
    target: Concrete,
    distractor: Concrete,
    axis: str,
    kb: Mapping[str, Pedal],
    ctx: DeriveContext,
    targets: Targets,
) -> float:
    """Gap between target and distractor on ``axis`` (> 0: disjoint). Units: dB for gain,
    dB of the EQ shelf/peak for free brightness/mids, knob rotation for pedal brightness,
    compressor ratio for compression (an off compressor counts as ratio 1)."""
    if axis == "gain":
        return _gap(
            gain_range_db(target, kb, ctx, _gain_knob(target, targets)),
            gain_range_db(distractor, kb, ctx, _gain_knob(distractor, targets)),
        )
    if target.kind != distractor.kind or target.pedal != distractor.pedal:
        raise GoldError(f"{axis} distractor must keep the target's kind and pedal")
    if axis == "brightness":
        if target.kind == "pedal":
            knob = targets.pedals[target.pedal].tone_knob
            if not knob:
                raise GoldError(f"{target.pedal} has no tone knob")
            return _gap(target.knobs[knob], distractor.knobs[knob])
        return _gap(target.ranges["eq"]["high_db"], distractor.ranges["eq"]["high_db"])
    if target.kind == "pedal":
        raise GoldError(f"{axis} distractors are free-board only")
    if axis == "mids":
        return _gap(target.ranges["eq"]["mid_db"], distractor.ranges["eq"]["mid_db"])

    def ratio(c: Concrete) -> Range:
        return c.ranges["compressor"]["ratio"] if c.gates["compressor"] else (1.0, 1.0)

    return _gap(ratio(target), ratio(distractor))


def min_gap(axis: str, cfg, pedal: bool = False) -> float:
    """``gold.min_gap[axis]`` (``brightness_knob`` for pedal tone knobs)."""
    key = "brightness_knob" if axis == "brightness" and pedal else axis
    return float(cfg.gold.min_gap[key])


def check_separation(targets: Targets, kb: Mapping[str, Pedal], ctx: DeriveContext, cfg):
    """``[(row id, distractor index, axis, gap)]`` for every distractor below its minimum
    gap (``gold.min_gap``). Empty = every distractor clearly contradicts its target."""
    bad = []
    for row in targets.items:
        tc = expand(row.target, targets)
        for i, d in enumerate(row.distractors):
            gap = separation(tc, expand(d.region, targets), d.axis, kb, ctx, targets)
            if gap < min_gap(d.axis, cfg, pedal=tc.kind == "pedal"):
                bad.append((row.id, i, d.axis, gap))
    return bad


def unseen(row: TargetRow, holdout: str) -> bool:
    regions = [row.target, *(d.region for d in row.distractors)]
    return any(r.kind == "pedal" and r.pedal == holdout for r in regions)


# ============================================================================ drafting

OPPOSITE_BRIGHTNESS = {
    "dark": "bright",
    "slightly_dark": "bright",
    "slightly_bright": "dark",
    "bright": "dark",
}
OPPOSITE_MIDS = {"scooped": "pushed", "pushed": "scooped"}
HIGH_GAIN = {"family": "distortion", "clipping": "hard", "gain": "high"}


def draft_distractors(
    target: Region,
    targets: Targets,
    kb: Mapping[str, Pedal],
    ctx: DeriveContext,
    gain_split_db: float,
    main_ask_eq: bool,
) -> list[Distractor]:
    """The distractor rules (module docstring), in priority order."""
    by_axis: dict[str, Distractor] = {}
    t = target
    if t.kind == "pedal":
        base = Region("free", "none", t.brightness)
    else:
        base = Region("free", "none", t.brightness, t.mids, t.compression)
    tc = expand(t, targets)
    if gain_range_db(tc, kb, ctx, _gain_knob(tc, targets))[1] >= gain_split_db:
        by_axis["gain"] = Distractor("gain", base)
    else:
        by_axis["gain"] = Distractor(
            "gain",
            replace(base, **HIGH_GAIN),
        )
    if t.brightness in OPPOSITE_BRIGHTNESS:
        flip = OPPOSITE_BRIGHTNESS[t.brightness]
        if t.kind == "free":
            by_axis["brightness"] = Distractor("brightness", replace(t, brightness=flip))
        elif targets.pedals[t.pedal].tone_knob:
            ov = {k: v for k, v in t.overrides.items() if k != targets.pedals[t.pedal].tone_knob}
            by_axis["brightness"] = Distractor(
                "brightness", replace(t, brightness=flip, overrides=ov)
            )
    if t.kind == "free" and t.mids in OPPOSITE_MIDS:
        by_axis["mids"] = Distractor("mids", replace(t, mids=OPPOSITE_MIDS[t.mids]))
    if t.kind == "free" and (
        t.compression == "heavy" or (t.compression == "light" and main_ask_eq)
    ):
        by_axis["compression"] = Distractor("compression", replace(t, compression="none"))
    order = ("brightness", "mids", "compression", "gain") if main_ask_eq else AXES
    return [by_axis[a] for a in order if a in by_axis]


# ============================================================================ planning


@dataclass(frozen=True)
class GoldSpec:
    """One render: the generic :class:`~lstmabar.data.render.RenderSpec` plus gold metadata."""

    spec: render.RenderSpec
    gold_id: str
    role: str
    distractor_index: int | None
    axis: str | None
    clip_index: int


def test_clips(cfg) -> list[DryClip]:
    """The ``test`` split clips of the frozen dataset (``gold.dry_manifest``)."""
    path = Path(str(cfg.gold.dry_manifest))
    if not path.exists():
        raise FileNotFoundError(f"dry-clip manifest not found: {path.name}")
    clips = [c for c in read_jsonl(path, DryClip) if c.split == "test"]
    if not clips:
        raise ValueError("no test clips in the dry-clip manifest")
    return clips


def pick_clips(gold_id: str, clips: Sequence[DryClip], n: int, seed: int) -> list[DryClip]:
    """``n`` clips ranked by ``sha256(seed:gold_id:clip_id)``: deterministic per item and
    independent of the other items."""

    def key(c: DryClip) -> str:
        return hashlib.sha256(f"{seed}:{gold_id}:{c.clip_id}".encode()).hexdigest()

    return sorted(clips, key=key)[: min(n, len(clips))]


def _render_spec(
    item_id: str,
    clip: DryClip,
    seed: int,
    conc: Concrete,
    peak: float,
    whitebox: Sequence[str],
) -> render.RenderSpec:
    knobs, board = sample(conc, np.random.default_rng(seed))
    engine = "whitebox" if conc.kind == "pedal" and conc.pedal in whitebox else "greybox"
    return render.RenderSpec(
        item_id=item_id,
        clip_id=clip.clip_id,
        split=SPLIT,
        seed=seed,
        scenario=conc.scenario,
        engine=engine,
        pedal_id=conc.pedal,
        pedal_knobs=knobs,
        free_board=board,
        input_peak_dbfs=peak,
    )


def plan(targets: Targets, clips: Sequence[DryClip], cfg) -> list[GoldSpec]:
    """Every render of every row, in targets-file order (a row's renders are consecutive)."""
    g = cfg.gold
    seed = int(g.seed)
    n_clips = int(g.clips_per_item)
    n_dist = int(g.distractors_per_item)
    peak = float(cfg.audio.inference_peak_dbfs)
    whitebox = tuple(str(p) for p in cfg.render.whitebox_pedals)
    out: list[GoldSpec] = []
    for row in targets.items:
        tc = expand(row.target, targets)
        dcs = [(d, expand(d.region, targets)) for d in row.distractors[:n_dist]]
        for j, clip in enumerate(pick_clips(row.id, clips, n_clips, seed)):
            iid = f"{row.id}-c{j}-t"
            s = render.item_seed(seed, f"gold:{row.id}:t", j)
            out.append(
                GoldSpec(
                    _render_spec(iid, clip, s, tc, peak, whitebox), row.id, "target", None, None, j
                )
            )
            for k, (d, dc) in enumerate(dcs):
                iid = f"{row.id}-c{j}-d{k}"
                s = render.item_seed(seed, f"gold:{row.id}:d{k}", j)
                out.append(
                    GoldSpec(
                        _render_spec(iid, clip, s, dc, peak, whitebox),
                        row.id,
                        "distractor",
                        k,
                        d.axis,
                        j,
                    )
                )
    return out


# ============================================================================ gold items

GOLD_TARGET_KEYS = (
    "in_scope",
    "family",
    "gain",
    "brightness",
    "mids",
    "compression",
    "clipping",
    "reference_pedal",
)
_LABEL_KEYS = GOLD_TARGET_KEYS[:-1]  # reference_pedal may be null on a labelled item


def load_gold(gold_dir: str | Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Gold items by id from every ``instructions_*.yaml`` and the files' sha256."""
    items: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for f in sorted(Path(gold_dir).glob("instructions_*.yaml")):
        data = f.read_bytes()
        hashes[f.name] = hashlib.sha256(data).hexdigest()
        for it in yaml.safe_load(data.decode("utf-8")) or []:
            if it["id"] in items:
                raise GoldError(f"gold id {it['id']!r} appears twice")
            items[it["id"]] = it
    if not items:
        raise FileNotFoundError("no gold instruction files (instructions_*.yaml)")
    return items, hashes


def labelled(item: Mapping[str, Any]) -> bool:
    """True when the owner has labelled the item's ``target`` (friend items start null)."""
    t = item.get("target") or {}
    return all(t.get(k) is not None for k in _LABEL_KEYS)


KB_ALIASES: tuple[tuple[str, str], ...] = (
    (r"tube ?screamer|\bts-?9\b|\bts-?808\b", "ts808"),
    (r"\bds-?1\b", "ds1"),
    (r"\brat2?\b|pro ?co", "rat"),
    (r"big ?muff|\bmuff\b", "big_muff"),
    (r"fuzz ?face", "fuzz_face"),
)
"""``reference_pedal`` label -> KB pedal (``fuzz_face`` resolves to ``_si`` only when the
instruction says silicon). Other named pedals (Klon, Blues Driver, Tone Bender, ...) have no
KB entry and map to a free-board region of their family."""


def kb_pedal(reference: str | None, instruction: str = "") -> str | None:
    if not reference:
        return None
    for pattern, pid in KB_ALIASES:
        if re.search(pattern, reference, flags=re.IGNORECASE):
            if pid == "fuzz_face":
                return "fuzz_face_si" if "silicon" in instruction.lower() else "fuzz_face_ge"
            return pid
    return None


PEDAL_MIDS = {"ts808": "pushed", "rat": "pushed", "ds1": "scooped", "big_muff": "scooped"}
"""Each KB pedal's own mid voicing (KB descriptors); fuzz faces are neutral."""

PEDAL_CONFLICTS = {("big_muff", "pushed"), ("ts808", "scooped")}
"""(pedal, mids) requests the KB pedal cannot produce: drafted as a free-board region."""


def draft_target(item: Mapping[str, Any], targets: Targets) -> tuple[Region, list[str]]:
    """Rule-based region for a labelled gold item, plus notes for the rationale."""
    t = item["target"]
    notes: list[str] = []
    family, gain = t["family"], t["gain"]
    drives = family in ("overdrive", "distortion", "fuzz") and gain != "none"
    pid = kb_pedal(t["reference_pedal"], item.get("instruction", ""))
    if t["reference_pedal"] and not pid:
        notes.append(f"{t['reference_pedal']} has no KB entry")
    if pid and pid in targets.pedals and drives:
        if (pid, t["mids"]) in PEDAL_CONFLICTS:
            notes.append(f"mids {t['mids']} conflicts with {pid}, so free board")
        else:
            table = targets.pedals[pid]
            bright = t["brightness"] if table.tone_knob else "neutral"
            if bright != t["brightness"]:
                notes.append(f"{pid} has no tone knob: brightness {t['brightness']} not realized")
            own = PEDAL_MIDS.get(pid, "neutral")
            if t["mids"] not in ("neutral", own):
                notes.append(f"mids {t['mids']} not realized ({pid} voicing: {own})")
            if t["compression"] != "none":
                notes.append(f"compression {t['compression']} not realized (pedal preset)")
            return Region("pedal", gain, bright, pedal=pid), notes
    if family == "clean" and gain == "low":
        return Region("free", "low", t["brightness"], t["mids"], t["compression"], "boost"), notes
    if not drives:
        return Region("free", "none", t["brightness"], t["mids"], t["compression"]), notes
    clip = t["clipping"] if t["clipping"] in CLIPPING else None
    region = Region("free", gain, t["brightness"], t["mids"], t["compression"], family, clip)
    return region, notes


def _confidence(item: Mapping[str, Any], region: Region, notes: Sequence[str]) -> str:
    if not item["target"]["in_scope"]:
        return "low"
    if notes or (region.kind == "free" and region.gain != "none" and region.clipping is None):
        return "medium"
    return "high"


def _levels(t: Mapping[str, Any]) -> str:
    parts = [f"{t['family']}/{t['gain']}", t["brightness"], f"mids {t['mids']}"]
    parts += [f"comp {t['compression']}", f"clip {t['clipping']}"]
    if t["reference_pedal"]:
        parts.append(f"ref {t['reference_pedal']}")
    return ", ".join(parts)


def _describe(region: Region) -> str:
    if region.kind == "pedal":
        return f"KB {region.pedal} at {region.gain} gain, {region.brightness} tone"
    if region.gain == "none":
        on = ["EQ"] + (["compressor"] if region.compression != "none" else [])
        return "free board, no drive (" + " + ".join(on) + ")"
    clip = f", {region.clipping} clipping" if region.clipping else ""
    return f"free-board {region.family}{clip} at {region.gain} gain"


def draft_row(
    item: Mapping[str, Any],
    targets: Targets,
    kb: Mapping[str, Pedal],
    ctx: DeriveContext,
    cfg,
    *,
    target: Region | None = None,
    ambiguity: Sequence[str] = (),
    confidence: str | None = None,
    note: str = "",
) -> TargetRow | None:
    """Draft a row for a labelled item (``None`` if unlabelled). Keyword overrides record a
    curator decision; distractors are always re-derived from the final target."""
    if not labelled(item):
        return None
    region, notes = draft_target(item, targets)
    if target is not None:
        region, notes = target, []
    expand(region, targets, where=f"draft[{item['id']}]")
    main_ask_eq = region.kind == "free" and region.gain == "none"
    dist = draft_distractors(region, targets, kb, ctx, float(cfg.gold.gain_split_db), main_ask_eq)
    conf = confidence or ("low" if ambiguity else _confidence(item, region, notes))
    rationale = f"Gold label: {_levels(item['target'])}. Region: {_describe(region)}."
    if notes:
        rationale += " " + "; ".join(notes) + "."
    if note:
        rationale += " " + note
    row = TargetRow(
        id=str(item["id"]),
        source=str(item["source"]),
        in_scope=bool(item["target"]["in_scope"]),
        target=region,
        distractors=tuple(dist),
        confidence=conf,
        ambiguous=bool(ambiguity),
        ambiguity=tuple(ambiguity),
        unseen_pedal=False,
        rationale=rationale,
    )
    return replace(row, unseen_pedal=unseen(row, str(cfg.render.holdout_pedal)))


def check_rows(targets: Targets, gold: Mapping[str, Mapping], kb, ctx, cfg) -> list[str]:
    """Problems with the rows: unknown ids, ``source``/``in_scope`` that differ from the gold
    file, a wrong ``unseen_pedal`` flag, unknown pedals, or distractors that don't separate."""
    problems = []
    holdout = str(cfg.render.holdout_pedal)
    for row in targets.items:
        it = gold.get(row.id)
        if it is None:
            problems.append(f"{row.id}: not in the gold files")
            continue
        if row.source != it["source"]:
            problems.append(f"{row.id}: source {row.source!r} != gold {it['source']!r}")
        if labelled(it) and row.in_scope != bool(it["target"]["in_scope"]):
            problems.append(f"{row.id}: in_scope differs from the gold label")
        if row.unseen_pedal != unseen(row, holdout):
            problems.append(f"{row.id}: unseen_pedal should be {unseen(row, holdout)}")
        for r in [row.target, *(d.region for d in row.distractors)]:
            if r.kind == "pedal" and r.pedal not in kb:
                problems.append(f"{row.id}: pedal {r.pedal!r} is not in the KB")
    if problems:
        return problems
    for rid, i, axis, gap in check_separation(targets, kb, ctx, cfg):
        problems.append(f"{rid}: distractor {i} ({axis}) overlaps the target (gap {gap:.3g})")
    return problems


# ============================================================================ targets file IO


def _flow(d: Any) -> str:
    s = yaml.safe_dump(d, default_flow_style=True, sort_keys=False, width=10_000).strip()
    return s.removesuffix("...").strip()  # a bare scalar gets a document-end marker


def _block(d: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(d), sort_keys=False, default_flow_style=None, width=100)


def dump_targets(
    header: str, targets: Targets, rows: Sequence[TargetRow], prefix: str | None = None
) -> str:
    """The targets file text, ending with one row per item (flow-style regions, compact for
    review). ``prefix`` (an existing file's text before ``items:``, see :func:`items_prefix`)
    is kept verbatim with its comments, except for the ``status:`` line; without it the text
    starts with ``header`` and generated ``version``/``status``/``tables``/``pedals``."""
    if prefix is not None:
        body = re.sub(r"(?m)^status: \w+$", f"status: {targets.status}", prefix.rstrip())
        lines = [body, "items:"]
    else:
        pedals = {
            pid: {
                "gain_knob": p.gain_knob,
                **({"tone_knob": p.tone_knob} if p.tone_knob else {}),
                "fixed": dict(p.fixed),
                "gain": {k: list(v) for k, v in p.gain.items()},
                **({"tone": {k: list(v) for k, v in p.tone.items()}} if p.tone else {}),
            }
            for pid, p in targets.pedals.items()
        }
        body = f"version: {targets.version}\nstatus: {targets.status}\n"
        body += _block({"tables": _plain(targets.tables)}) + _block({"pedals": pedals})
        lines = [header.rstrip(), body.rstrip(), "items:"]
    for row in rows:
        d = row.to_yaml()
        lines.append(f"- id: {d['id']}")
        lines.append(f"  source: {d['source']}")
        lines.append(f"  in_scope: {_flow(d['in_scope'])}")
        lines.append(f"  target: {_flow(d['target'])}")
        lines.append("  distractors:")
        lines += [f"  - {_flow(x)}" for x in d["distractors"]]
        lines.append(f"  confidence: {d['confidence']}")
        lines.append(f"  ambiguous: {_flow(d['ambiguous'])}")
        lines.append(f"  ambiguity: {_flow(d['ambiguity'])}")
        lines.append(f"  unseen_pedal: {_flow(d['unseen_pedal'])}")
        lines.append(f"  rationale: {_flow(d['rationale'])}")
    return "\n".join(lines) + "\n"


def items_prefix(text: str) -> str:
    """A targets file's text before its top-level ``items:`` key."""
    m = re.search(r"(?m)^items:", text)
    if m is None:
        raise GoldError("targets file: no top-level 'items:' key")
    return text[: m.start()]


def _plain(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_plain(x) for x in v]
    return v


def summary(targets: Targets, gold: Mapping[str, Mapping]) -> dict[str, dict[str, int]]:
    """Per gold ``source``: items, paired rows, unlabelled, ambiguous, unseen-pedal,
    out-of-scope, confidence and target-kind counts (sources reported separately)."""
    out: dict[str, Counter] = defaultdict(Counter)
    rows = targets.by_id()
    for gid, it in gold.items():
        c = out[str(it["source"])]
        c["items"] += 1
        if not labelled(it):
            c["unlabelled"] += 1
        row = rows.get(gid)
        if row is None:
            continue
        c["paired"] += 1
        c["ambiguous"] += row.ambiguous
        c["unseen_pedal"] += row.unseen_pedal
        c["out_of_scope"] += not row.in_scope
        c[f"confidence_{row.confidence}"] += 1
        c[f"target_{row.target.kind}"] += 1
        c["distractors"] += len(row.distractors)
    return {s: dict(sorted(out[s].items())) for s in SOURCES if s in out}


def _setup(cfg):
    kb = render.load_pedals(cfg)
    ctx = DeriveContext.from_config(cfg)
    gold, hashes = load_gold(Path(str(cfg.gold.gold_dir)))
    tpath = Path(str(cfg.gold.targets_file))
    return kb, ctx, gold, hashes, load_targets(tpath), tpath


def targets_step(cfg, args) -> int:
    """``lstmabar data gold-targets``: validate the targets file against the gold files and
    the separation rule and print per-source counts. ``--draft-missing`` appends rule-based
    draft rows for labelled gold items without one (and sets ``status: draft``)."""
    kb, ctx, gold, _, targets, tpath = _setup(cfg)
    if getattr(args, "draft_missing", False):
        have = set(targets.by_id())
        new = [
            r
            for gid, it in gold.items()
            if gid not in have and (r := draft_row(it, targets, kb, ctx, cfg)) is not None
        ]
        if new:
            text = tpath.read_text(encoding="utf-8")
            targets = replace(targets, status="draft")
            out = dump_targets("", targets, [*targets.items, *new], prefix=items_prefix(text))
            tpath.write_text(out, encoding="utf-8", newline="\n")
            print(f"appended {len(new)} draft rows (status: draft; review them)")
            targets = load_targets(tpath)
    problems = check_rows(targets, gold, kb, ctx, cfg)
    for p in problems:
        print("PROBLEM:", p)
    missing = [g for g, it in gold.items() if labelled(it) and g not in targets.by_id()]
    if missing:
        print(f"{len(missing)} labelled gold items have no row (--draft-missing): {missing[:5]}")
    print(f"status: {targets.status}; rows: {len(targets.items)}")
    print(json.dumps(summary(targets, gold), indent=1))
    return 1 if problems else 0


# ============================================================================ friend examples


def example_regions(cfg, targets: Targets) -> list[tuple[str, Region]]:
    """``gold.examples`` as ``[(example_id, region)]``. The holdout pedal is refused: its
    name and sound stay out of anything shown to people writing test instructions."""
    holdout = str(cfg.render.holdout_pedal)
    out = []
    for ex in cfg.gold.examples:
        d = OmegaConf.to_container(ex, resolve=True)
        eid = str(d.pop("id"))
        if not re.fullmatch(r"ex\d\d", eid):
            raise GoldError(f"gold.examples: id {eid!r} must match exNN")
        region = region_from_dict(d, f"gold.examples[{eid}]")
        if region.kind == "pedal" and region.pedal == holdout:
            raise GoldError(f"gold.examples[{eid}]: the holdout pedal {holdout!r} is not allowed")
        expand(region, targets, where=f"gold.examples[{eid}]")
        out.append((eid, region))
    ids = [e for e, _ in out]
    if len(set(ids)) != len(ids):
        raise GoldError("gold.examples: duplicate ids")
    return out


def examples_step(cfg, args) -> int:
    """``lstmabar data gold-examples``: render the example tones friends describe (protocol
    mode A, docs/gold-collection.md) on test clips at the inference level, as 16-bit WAV
    pairs ``<id>_dry.wav`` / ``<id>_wet.wav`` (both peak-normalized) plus ``examples.json``
    (regions and sampled settings: owner only, never shared)."""
    import soundfile as sf

    from lstmabar.data import levels

    targets = load_targets(Path(str(cfg.gold.targets_file)))
    regions = example_regions(cfg, targets)
    clips = test_clips(cfg)
    ctx = render.RenderContext(cfg, render.Options(allow_derived_labels=True, pitch=False))
    layout = DataLayout.from_config(cfg)
    out_dir = layout.rendered / "gold_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    peak = float(cfg.audio.inference_peak_dbfs)
    seed = int(cfg.gold.seed)
    meta = []
    for eid, region in regions:
        clip = pick_clips(f"example:{eid}", clips, 1, seed)[0]
        conc = expand(region, targets)
        knobs, board = sample(conc, np.random.default_rng(render.item_seed(seed, eid, 0)))
        dry, _ = levels.read_audio(layout.resolve_rendered(clip.path))
        x = levels.apply_input_level(dry, peak)
        if conc.kind == "pedal":
            pedal = ctx.kb[conc.pedal]
            if conc.pedal in tuple(str(p) for p in cfg.render.whitebox_pedals):
                wet = ctx.render_whitebox(pedal, knobs, x)
            else:
                wet = ctx.render_board(to_floats(derive(pedal, knobs, ctx.derive_ctx)), x)
        else:
            wet = ctx.render_board(board, x)
        for name, y in (("dry", x), ("wet", np.asarray(wet, dtype=np.float64))):
            y, _ = levels.normalize_peak(y, float(cfg.render.wet_peak_dbfs))
            sf.write(str(out_dir / f"{eid}_{name}.wav"), y, ctx.sr, subtype="PCM_16")
        meta.append(
            {
                "id": eid,
                "clip_id": clip.clip_id,
                "region": region.to_yaml(),
                "pedal_knobs": knobs,
                "board": board,
                "input_peak_dbfs": peak,
            }
        )
        print(f"{eid}: {_describe(region)} on {clip.clip_id}")
    (out_dir / "examples.json").write_text(
        json.dumps(_clean(meta), indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {len(meta)} example pairs; share only the WAV files")
    return 0


def to_floats(preset) -> FloatBoard:
    from lstmabar.data.schema import board_to_floats
    from lstmabar.physics.derive import preset_params

    return board_to_floats(preset_params(preset))


# ============================================================================ rendering


@dataclass(frozen=True)
class GoldRender:
    """One rendered gold sample (one line of ``data/manifests/<gold.version>.jsonl``)."""

    item_id: str  # "<gold id>-c<clip index>-t" | "<gold id>-c<clip index>-d<k>"
    gold_id: str
    gold_source: str  # human | simulated_human | llm_persona | llm_draft (never relabeled)
    role: str  # target | distractor
    distractor_index: int | None
    axis: str | None  # the distractor's contradicted axis
    clip_index: int
    clip_id: str
    in_scope: bool
    ambiguous: bool
    confidence: str
    unseen_pedal: bool
    region: dict[str, Any]  # the targets-file region this sample was drawn from
    render: dict[str, Any]  # RenderItem (split "gold", wet_path under the gold version)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GoldRender:
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise KeyError(f"GoldRender: unknown fields {sorted(unknown)}")
        return cls(**dict(d))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_manifest(path: str | Path) -> list[GoldRender]:
    with Path(path).open(encoding="utf-8") as f:
        return [GoldRender.from_dict(json.loads(line)) for line in f if line.strip()]


def gold_cfg(cfg):
    """``cfg`` with ``data.version = gold.version``, so the renderer writes audio under
    ``data/rendered/<gold.version>/``. The version must start with ``gold`` (never a dataset
    version: the frozen v1 tree is never written)."""
    version = str(cfg.gold.version)
    if version == str(cfg.data.version) or not version.startswith("gold"):
        raise ValueError(f"gold.version {version!r} must start with 'gold' and differ from v1")
    return OmegaConf.merge(cfg, {"data": {"version": version}})


def manifest_paths(cfg) -> tuple[Path, Path]:
    """``(manifest, index)`` files in ``data.manifests_dir`` (not in a version dir)."""
    layout = DataLayout.from_config(cfg)
    version = str(cfg.gold.version)
    return layout.manifests / f"{version}.jsonl", layout.manifests / f"{version}.index.json"


def _render_jobs(cfg, opts, shards, by_clip, workers: int) -> None:
    """Render the shards not yet DONE for this plan, as ``render.run`` does."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    layout = DataLayout.from_config(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    fp = render.render_fingerprint(cfg, opts)
    jobs = []
    for i, shard in enumerate(shards):
        h = render.shard_hash(shard, fp)
        done = render._done_hash(layout, i)
        if done == h:
            continue
        if done is not None:
            raise RuntimeError(
                f"{layout.shard_name(i)} is DONE but was rendered from a different plan "
                "(targets, config or clips changed): delete the gold wet dir or bump gold.version"
            )
        clip_ids = dict.fromkeys(s.clip_id for s in shard)
        jobs.append(
            {
                "cfg": cfg_dict,
                "opts": asdict(opts),
                "shard": i,
                "plan_sha256": h,
                "specs": [asdict(s) for s in shard],
                "clips": [to_dict(by_clip[c]) for c in clip_ids],
            }
        )
    n = sum(len(s) for s in shards)
    print(f"gold plan: {n} renders in {len(shards)} shards ({len(jobs)} to render)", flush=True)
    t0 = time.perf_counter()
    workers = max(1, min(workers, len(jobs) or 1))
    if workers == 1:
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for job in jobs:
                render._report(render.render_shard(job), t0)
        finally:
            torch.set_num_threads(threads)
    else:
        mp = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=mp, initializer=render._init_worker) as pool:
            for fut in as_completed([pool.submit(render.render_shard, j) for j in jobs]):
                render._report(fut.result(), t0)


def run(cfg, args) -> int:
    """``lstmabar data gold-pair``: validate, plan, render (resumable shards), then write the
    manifest and index. ``--limit N`` renders the first N rows only (index marked partial)."""
    kb, ctx, gold, gold_hashes, targets, tpath = _setup(cfg)
    if targets.status != "reviewed" and not getattr(args, "allow_draft", False):
        raise RuntimeError(
            "the targets file is status: draft; the owner reviews it first "
            "(--allow-draft for a smoke run)"
        )
    problems = check_rows(targets, gold, kb, ctx, cfg)
    if problems:
        raise GoldError("targets file failed validation:\n" + "\n".join(problems[:20]))
    limit = getattr(args, "limit", None)
    used = replace(targets, items=targets.items[:limit] if limit else targets.items)
    partial = len(used.items) < len(targets.items) or targets.status != "reviewed"

    clips = test_clips(cfg)
    expect = cfg.gold.get("expect_test_clips")
    if expect is not None and len(clips) != int(expect):
        raise ValueError(f"expected {expect} test clips, found {len(clips)}")
    specs = plan(used, clips, cfg)
    gcfg = gold_cfg(cfg)
    opts = render._options(gcfg, args)
    render._preflight(gcfg, opts, kb)
    layout = DataLayout.from_config(gcfg)
    shards = render.shard_specs([g.spec for g in specs], int(cfg.gold.shard_size))
    workers = int(getattr(args, "workers", None) or cfg.gold.workers)
    _render_jobs(gcfg, opts, shards, {c.clip_id: c for c in clips}, workers)

    items: dict[str, RenderItem] = {}
    for i in range(len(shards)):
        for it in read_jsonl(layout.shard_dir(i) / render.ITEMS_FILE, RenderItem):
            items[it.item_id] = it
    rows = used.by_id()
    records = []
    for g in specs:
        row = rows[g.gold_id]
        reg = row.target if g.role == "target" else row.distractors[g.distractor_index].region
        records.append(
            GoldRender(
                item_id=g.spec.item_id,
                gold_id=g.gold_id,
                gold_source=row.source,
                role=g.role,
                distractor_index=g.distractor_index,
                axis=g.axis,
                clip_index=g.clip_index,
                clip_id=g.spec.clip_id,
                in_scope=row.in_scope,
                ambiguous=row.ambiguous,
                confidence=row.confidence,
                unseen_pedal=row.unseen_pedal,
                region=reg.to_yaml(),
                render=to_dict(items[g.spec.item_id]),
            )
        )
    write_outputs(cfg, records, targets, tpath, gold_hashes, partial)
    return 0


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_counts(records: Sequence[GoldRender]) -> dict[str, dict[str, int]]:
    """Per gold ``source``: items, renders by role/engine, ambiguous, unseen-pedal and
    out-of-scope items."""
    counts: dict[str, Counter] = defaultdict(Counter)
    first: dict[str, GoldRender] = {}
    for r in records:
        c = counts[r.gold_source]
        c["renders"] += 1
        c[f"{r.role}_renders"] += 1
        c[f"engine_{r.render['engine']}"] += 1
        first.setdefault(r.gold_id, r)
    for r in first.values():
        c = counts[r.gold_source]
        c["items"] += 1
        c["ambiguous"] += r.ambiguous
        c["unseen_pedal"] += r.unseen_pedal
        c["out_of_scope"] += not r.in_scope
    return {s: dict(sorted(counts[s].items())) for s in SOURCES if s in counts}


def write_outputs(cfg, records, targets: Targets, tpath: Path, gold_hashes, partial: bool):
    """Write the committed manifest (one :class:`GoldRender` per line) and its index."""
    manifest, index_path = manifest_paths(cfg)
    text = "".join(
        json.dumps(_clean(r.to_dict()), sort_keys=True, ensure_ascii=False) + "\n" for r in records
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(text, encoding="utf-8", newline="\n")
    files = "".join(f"{r.render['sha256']}  {r.render['wet_path']}\n" for r in records)
    dry = Path(str(cfg.gold.dry_manifest))
    v1_index = dry.parent / "index.json"
    dataset_sha = None
    if v1_index.exists():
        dataset_sha = json.loads(v1_index.read_text(encoding="utf-8")).get("dataset_sha256")
    index = {
        "version": str(cfg.gold.version),
        "partial": partial,
        "targets_file": tpath.name,
        "targets_sha256": hashlib.sha256(tpath.read_bytes()).hexdigest(),
        "targets_status": targets.status,
        "gold_files_sha256": dict(sorted(gold_hashes.items())),
        "dry_manifest_sha256": hashlib.sha256(dry.read_bytes()).hexdigest(),
        "dataset_sha256": dataset_sha,
        "settings": {
            "clips_per_item": int(cfg.gold.clips_per_item),
            "distractors_per_item": int(cfg.gold.distractors_per_item),
            "input_peak_dbfs": float(cfg.audio.inference_peak_dbfs),
            "seed": int(cfg.gold.seed),
        },
        "counts": render_counts(records),
        "renders": len(records),
        "manifest_sha256": _sha256_text(text),
        "files_sha256": _sha256_text(files),
    }
    index_path.write_text(
        json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {manifest.name} ({len(records)} renders) and {index_path.name}")


def _clean(v: Any) -> Any:
    """JSON-safe (NaN -> None), as :func:`lstmabar.data.schema.to_dict` does."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Mapping):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_clean(x) for x in v]
    return v


__all__ = [
    "AXES",
    "SOURCES",
    "Concrete",
    "Distractor",
    "GoldError",
    "GoldRender",
    "GoldSpec",
    "Region",
    "TargetRow",
    "Targets",
    "check_rows",
    "check_separation",
    "draft_distractors",
    "example_regions",
    "examples_step",
    "draft_row",
    "draft_target",
    "dump_targets",
    "expand",
    "gain_range_db",
    "gold_cfg",
    "kb_pedal",
    "labelled",
    "load_gold",
    "load_targets",
    "manifest_paths",
    "pick_clips",
    "plan",
    "read_manifest",
    "render_counts",
    "run",
    "sample",
    "separation",
    "summary",
    "targets_from_dict",
    "targets_step",
    "test_clips",
    "unseen",
]
