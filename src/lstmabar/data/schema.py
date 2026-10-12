"""Dataset v1 record types (P4 contract) and JSONL helpers.

Every record is a frozen dataclass that round-trips through ``to_dict`` / ``from_dict`` and
one JSON object per line (:func:`write_jsonl` / :func:`read_jsonl`). Paths inside records are
POSIX-style and relative to the dataset roots in ``configs/data/v1.yaml`` (never absolute, so
manifests are portable and never leak local paths).

Board parameters are stored **normalized** (``{block: {knob: float, "enabled": 0.|1.}}``,
the ``BoardParams`` layout of :mod:`lstmabar.dsp.pedalboard` with floats instead of
tensors); see :func:`board_to_floats` / :func:`floats_to_board`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import torch

from lstmabar.dsp.pedalboard import BoardParams

SCHEMA_VERSION = 1

Split = Literal["train", "val", "test", "test_unseen_pedal"]
SPLITS: tuple[str, ...] = ("train", "val", "test", "test_unseen_pedal")

Scenario = Literal["pedal", "bypass", "eq", "comp", "comp_eq", "drive"]
"""``pedal`` = a KB pedal preset; the rest are free-board scenarios (R5)."""
FREE_SCENARIOS: tuple[str, ...] = ("bypass", "eq", "comp", "comp_eq", "drive")

Engine = Literal["whitebox", "greybox"]

FloatBoard = dict[str, dict[str, float]]

LabelSource = Literal["derived", "calibrated", "calibrated_nearest", "free"]
"""How ``RenderItem.board_params`` was produced: ``derive()`` (grey-box pedals), the
calibration table (white-box, interpolated or nearest grid point, R4), or sampled directly
(free board)."""

_DRIVE = re.compile(r"^[A-Za-z]:")


def check_relpath(value: str, what: str) -> None:
    """Paths in records are relative POSIX: no drive letter, root, backslash or ``..``."""
    if not value:
        return
    if _DRIVE.match(value) or value.startswith("/") or "\\" in value:
        raise ValueError(f"{what} must be a relative POSIX path, got {value!r}")
    if ".." in value.split("/"):
        raise ValueError(f"{what} must not contain '..', got {value!r}")


class _RelPaths:
    """Mixin: validate the dataclass fields named in ``_PATH_FIELDS``."""

    _PATH_FIELDS: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in self._PATH_FIELDS:
            check_relpath(getattr(self, name), f"{type(self).__name__}.{name}")


@dataclass(frozen=True)
class DryClip(_RelPaths):
    """One onset-aligned DI clip stored at ``ingest.dry_peak_dbfs`` (R3). Written by ingest
    only; nothing rewrites ``dry_clips.jsonl`` afterwards."""

    _PATH_FIELDS = ("source_file", "path")

    clip_id: str  # stable, e.g. "gt-p1-<recording>-<offset ms>"
    source: str  # "guitar_techs" | "egfxset" | "own" | "fixture"
    performer: str
    recording_id: str  # source recording; splits never share one (decision 11)
    technique: str  # dataset's technique/content label ("chords", "scales", ...), or ""
    source_file: str  # path relative to data.sources_dir
    offset_s: float
    duration_s: float
    sample_rate: int
    source_gain_db: float  # gain applied to the source to reach the stored peak
    path: str  # FLAC path relative to data.rendered_dir
    sha256: str = ""
    split: str = ""  # "train" | "val" | "test" (lstmabar.data.splits)
    license: str = ""


@dataclass(frozen=True)
class RenderItem(_RelPaths):
    """One wet render of a dry clip, with its labels and measured analysis.

    Analysis values may be ``None`` after a JSONL round-trip (NaN is stored as ``null``):
    readers such as the caption code must treat ``None`` as "not measured".
    """

    _PATH_FIELDS = ("wet_path",)

    item_id: str  # "<clip_id>-r<k>"
    clip_id: str
    split: str
    seed: int  # per-item seed; the render is a pure function of (clip, seed, config)
    scenario: str  # Scenario
    engine: str  # Engine
    pedal_id: str | None  # KB pedal for scenario == "pedal"
    pedal_knobs: dict[str, float] | None  # pedal-space label (knob rotations in [0, 1])
    board_params: FloatBoard  # training label (calibrated for white-box, R4)
    board_params_derived: FloatBoard | None  # derive() label (pedal items only)
    label_source: str  # LabelSource
    clamped: bool  # a derive() " clamped to " note fired (decision 10, R6): derived label
    clamp_notes: list[str]
    input_peak_dbfs: float  # level the dry clip was scaled to before the pedal (R3)
    wet_gain_db: float  # true wet = stored file * 10**(wet_gain_db / 20)
    wet_path: str  # relative to data.rendered_dir
    sha256: str = ""
    label_clamp_notes: list[str] = field(default_factory=list)  # clamps in the calibrated label
    voiced_fraction: float | None = None  # of the dry clip's pYIN track (decision 5)
    descriptors_dry: dict[str, float] = field(default_factory=dict)
    descriptors_wet: dict[str, float] = field(default_factory=dict)
    archetype_dry: dict[str, float] = field(default_factory=dict)  # ARCHETYPES + "noise"
    archetype_wet: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Caption:
    caption_id: str  # "<item_id>-c<k>"
    item_id: str
    kind: str  # "template" | "paraphrase"
    text: str
    template_id: str = ""
    names_pedal: bool = False
    model: str | None = None  # LLM model id for paraphrases
    prompt_version: str | None = None


@dataclass(frozen=True)
class TestRealItem(_RelPaths):
    """A real-pedal recording with its clean input (task 4.4)."""

    __test__ = False  # not a pytest test class
    _PATH_FIELDS = ("dry_path", "wet_path")

    item_id: str
    dataset: str  # "egfxset" | "podset" | "tonetwist"
    device: str  # dataset's device label, e.g. "TubeScreamer"
    family: str  # overdrive | distortion | fuzz | ...
    kb_pedal_id: str | None  # matching KB pedal, if any (ts808, rat, ...)
    settings: dict[str, float]  # knob rotations in [0, 1] where known
    dry_path: str  # relative to data.test_real_dir
    wet_path: str
    sample_rate: int
    internal_only: bool  # never publish audio or per-item audio (ToneTwist, pOD-set)
    license: str
    notes: str = ""
    sha256_dry: str = ""
    sha256_wet: str = ""


Record = DryClip | RenderItem | Caption | TestRealItem
R = TypeVar("R", DryClip, RenderItem, Caption, TestRealItem)


def _clean(v: Any) -> Any:
    """JSON-safe value: NaN/inf floats -> None, tuples -> lists, recursively."""
    if isinstance(v, np.generic | torch.Tensor):
        v = v.item()
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Mapping):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_clean(x) for x in v]
    return v


def to_dict(rec: Record) -> dict[str, Any]:
    return _clean(dataclasses.asdict(rec))


def from_dict(cls: type[R], d: Mapping[str, Any]) -> R:
    """Strict: unknown fields raise, so a schema change bumps ``SCHEMA_VERSION`` (recorded in
    the dataset index) instead of being silently dropped by old readers."""
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown fields {sorted(unknown)}")
    return cls(**dict(d))


def write_jsonl(path: str | Path, records: Iterable[Record]) -> int:
    """Write records as JSON lines (sorted keys, UTF-8, ``\\n``). Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(to_dict(rec), sort_keys=True, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, cls: type[R]) -> Iterator[R]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield from_dict(cls, json.loads(line))


def board_to_floats(params: BoardParams, index: int = 0) -> FloatBoard:
    """Item ``index`` of a batched ``BoardParams`` as plain floats."""
    return {b: {k: float(v[index]) for k, v in knobs.items()} for b, knobs in params.items()}


def floats_to_board(board: Mapping[str, Mapping[str, float]], batch_size: int = 1) -> BoardParams:
    """Inverse of :func:`board_to_floats` (float32 tensors of shape ``(batch_size,)``)."""
    return {
        b: {k: torch.full((batch_size,), float(v)) for k, v in knobs.items()}
        for b, knobs in board.items()
    }


__all__ = [
    "FREE_SCENARIOS",
    "SCHEMA_VERSION",
    "SPLITS",
    "Caption",
    "DryClip",
    "Engine",
    "FloatBoard",
    "LabelSource",
    "RenderItem",
    "Scenario",
    "Split",
    "TestRealItem",
    "board_to_floats",
    "check_relpath",
    "floats_to_board",
    "from_dict",
    "read_jsonl",
    "to_dict",
    "write_jsonl",
]
