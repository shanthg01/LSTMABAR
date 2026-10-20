"""Nearest-pedal explanation for a text-predicted board (M4 demo, P6 task 6.8). No gradio import.

After a text method sets the board, the demo names the knowledge-base pedal whose grey-box
board is closest to the prediction, at which dial setting, and why. Nothing here is a model
output: it is a lookup of the predicted board against the boards that
:func:`~lstmabar.physics.derive.derive` gives for each KB pedal on a coarse dial grid
(:func:`pedal_atlas`).

Distance (documented in :func:`board_distance`), in **normalized** board space ([0, 1] per
knob, the space the methods predict in):

- three groups of knobs, each scored by the mean absolute difference of its knobs:
  *gain* (``drive.gain_db``), *clipping* (``drive.softness``, ``asymmetry``, ``bias``) and
  *tone* (``drive.pre_hpf_hz``, ``drive.tone_db`` and the four ``eq`` knobs);
- the distance is the mean of the three group scores (equal weights), so it lies in [0, 1];
- ``drive.level_db`` (output level: the A/B is loudness-matched) and the compressor (every
  pedal preset has it off) are left out;
- a block whose gate is off is compared at its neutral (default) knobs, so a board with the
  EQ off reads as a flat EQ. Every pedal preset has the drive on; when the prediction has it
  off, the explanation says so (no pedal is then really close).

The nearest pedal describes the predicted settings, not an independent judgement of
the method: kNN-bge, for one, predicts the per-knob median label (majority vote for gates)
of renders that are often of a single pedal, so its nearest pedal is close to circular.

The held-out pedal (``render.holdout_pedal``, the Rat in data v1: never in the training
split) is not excluded, since this is a demo lookup, not an evaluation, but it is labelled.
"""

from __future__ import annotations

import itertools
import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lstmabar.data.schema import FloatBoard
from lstmabar.dsp.pedalboard import ENABLED
from lstmabar.eval.baselines.base import complete_board, default_board

log = logging.getLogger(__name__)

GROUPS: dict[str, tuple[tuple[str, str], ...]] = {
    "gain": (("drive", "gain_db"),),
    "clipping": (("drive", "softness"), ("drive", "asymmetry"), ("drive", "bias")),
    "tone": (
        ("drive", "pre_hpf_hz"),
        ("drive", "tone_db"),
        ("eq", "low_db"),
        ("eq", "mid_db"),
        ("eq", "mid_hz"),
        ("eq", "high_db"),
    ),
}
"""Knob groups scored by :func:`board_distance` (normalized knobs)."""

GROUP_LABELS = {"gain": "drive gain", "clipping": "clipping shape", "tone": "tone / EQ"}
DIAL_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.875, 1.0)
"""Pot rotations tried per pedal knob in :func:`pedal_atlas` (every combination): steps of
2.5 on the 0-10 dials, plus 8.75, where high-gain settings sit (156 derives for the v1 KB)."""
LEVEL_POTS = frozenset({"level", "volume"})
"""Output-level pots: they only move ``drive.level_db``, which the distance ignores, so the
atlas holds them at their default (a derive takes ~0.1-0.5 s; this keeps the atlas small)."""
DEFAULT_HOLDOUT = "rat"
"""``render.holdout_pedal`` of data v1, used when the config does not say."""
CLOSE = 0.05
"""A group difference at or below this reads as "close" in the reason."""


@dataclass(frozen=True)
class AtlasEntry:
    """One KB pedal at one dial setting and its grey-box board (normalized)."""

    pedal_id: str
    pedal_name: str
    dials: Mapping[str, float]  # pot name -> rotation in [0, 1]
    dial_labels: Mapping[str, str]  # pot name -> label printed on the pedal
    board: FloatBoard


@dataclass(frozen=True)
class NearestPedal:
    entry: AtlasEntry
    distance: float
    groups: Mapping[str, float]  # group -> mean |difference| of its knobs
    drive_off: bool  # the predicted board has the drive gate off
    holdout: bool  # the entry's pedal is the held-out (unseen in training) pedal

    def dial_text(self) -> str:
        """``Gain 7.5, Tone 5`` on the pedal's own 0-10 dials."""
        return ", ".join(
            f"{self.entry.dial_labels.get(k, k)} {_dial(v)}" for k, v in self.entry.dials.items()
        )

    def reason(self) -> str:
        order = sorted(self.groups, key=lambda g: self.groups[g])
        close = [g for g in order if self.groups[g] <= CLOSE]
        parts = []
        if close:
            parts.append("closest on " + ", ".join(_group(g, self.groups[g]) for g in close))
        far = [g for g in order if g not in close]
        if far:
            parts.append(
                ("differs on " if close else "differences: ")
                + ", ".join(_group(g, self.groups[g]) for g in far)
            )
        return "; ".join(parts)

    def markdown(self) -> str:
        name = self.entry.pedal_name
        if self.holdout:
            name += " (held-out pedal: never in the training split)"
        md = (
            f"**Nearest pedal:** {name} at {self.dial_text()} "
            f"(board distance {self.distance:.3f}; {self.reason()})."
        )
        if self.drive_off:
            md += " The predicted board has the drive off, so no drive pedal is really close."
        return md


def _dial(rotation: float) -> str:
    v = round(10.0 * float(rotation), 1)
    return f"{v:g}"


def _group(g: str, d: float) -> str:
    return f"{GROUP_LABELS.get(g, g)} (Δ {d:.2f})"


def _neutral(board: Mapping[str, Mapping[str, float]]) -> FloatBoard:
    """Complete board with gated-off blocks at their default knobs."""
    full = complete_board(board, enabled=1.0)
    neutral = default_board(1.0)
    for b, knobs in full.items():
        if knobs.get(ENABLED, 1.0) < 0.5:
            full[b] = {**neutral[b], ENABLED: knobs[ENABLED]}
    return full


def group_distances(
    predicted: Mapping[str, Mapping[str, float]], reference: Mapping[str, Mapping[str, float]]
) -> dict[str, float]:
    """Mean absolute difference per knob group (:data:`GROUPS`) of two normalized boards."""
    a, b = _neutral(predicted), _neutral(reference)
    return {
        g: sum(abs(a[blk][k] - b[blk][k]) for blk, k in knobs) / len(knobs)
        for g, knobs in GROUPS.items()
    }


def board_distance(
    predicted: Mapping[str, Mapping[str, float]], reference: Mapping[str, Mapping[str, float]]
) -> float:
    """Mean of the :func:`group_distances` (equal group weights; in [0, 1])."""
    d = group_distances(predicted, reference)
    return sum(d.values()) / len(d)


def pedal_atlas(
    kb: Mapping[str, Any], sample_rate: int = 44100, grid: Sequence[float] = DIAL_GRID
) -> list[AtlasEntry]:
    """Grey-box boards of every KB pedal on ``grid`` (every combination of its pots; level
    pots, :data:`LEVEL_POTS`, stay at their default). Pedals whose derivation fails are
    skipped (logged)."""
    from lstmabar.data.gold import to_floats
    from lstmabar.physics.derive import DeriveContext, derive

    ctx = DeriveContext(sample_rate=int(sample_rate))
    out: list[AtlasEntry] = []
    for pid, pedal in kb.items():
        pots = list(pedal.pots)
        labels = {k: str(pedal.pots[k].label) for k in pots}
        try:
            axes = [
                (float(pedal.pots[k].default),) if k in LEVEL_POTS else tuple(grid) for k in pots
            ]
            for combo in itertools.product(*axes):
                dials = {k: float(v) for k, v in zip(pots, combo, strict=True)}
                board = to_floats(derive(pedal, dials, ctx))
                out.append(AtlasEntry(pid, str(pedal.name), dials, labels, board))
        except Exception:
            log.exception("could not derive the atlas for pedal %s", pid)
            out = [e for e in out if e.pedal_id != pid]
    return out


def nearest_pedals(
    predicted: Mapping[str, Mapping[str, float]],
    atlas: Sequence[AtlasEntry],
    holdout: str | None = DEFAULT_HOLDOUT,
    per_pedal: bool = True,
) -> list[NearestPedal]:
    """Atlas entries ranked by :func:`board_distance` to ``predicted`` (nearest first); with
    ``per_pedal`` only each pedal's best dial setting is kept."""
    full = complete_board(predicted, enabled=1.0)
    drive_off = full["drive"][ENABLED] < 0.5
    scored = []
    for e in atlas:
        g = group_distances(full, e.board)
        scored.append(
            NearestPedal(
                entry=e,
                distance=sum(g.values()) / len(g),
                groups=g,
                drive_off=drive_off,
                holdout=holdout is not None and e.pedal_id == holdout,
            )
        )
    scored.sort(key=lambda n: (n.distance, n.entry.pedal_id))
    if not per_pedal:
        return scored
    best: dict[str, NearestPedal] = {}
    for n in scored:
        best.setdefault(n.entry.pedal_id, n)
    return list(best.values())


class PedalExplainer:
    """Builds the atlas once (lazily, thread-safe; or takes a prebuilt ``atlas``) and
    explains predicted boards."""

    def __init__(
        self,
        kb: Mapping[str, Any],
        sample_rate: int = 44100,
        holdout: str | None = DEFAULT_HOLDOUT,
        grid: Sequence[float] = DIAL_GRID,
        atlas: Sequence[AtlasEntry] | None = None,
    ) -> None:
        self.kb = dict(kb)
        self.sample_rate = int(sample_rate)
        self.holdout = holdout
        self.grid = tuple(grid)
        self._atlas: list[AtlasEntry] | None = None if atlas is None else list(atlas)
        self._lock = threading.Lock()

    def atlas(self) -> list[AtlasEntry]:
        with self._lock:
            if self._atlas is None:
                self._atlas = pedal_atlas(self.kb, self.sample_rate, self.grid)
            return self._atlas

    def warm_up(self) -> threading.Thread:
        """Build the atlas in a background thread (a few tens of seconds of derives)."""

        def run() -> None:
            try:
                self.atlas()
            except Exception:
                log.exception("could not build the pedal atlas")

        t = threading.Thread(target=run, name="demo-pedal-atlas", daemon=True)
        t.start()
        return t

    def nearest(self, predicted: Mapping[str, Mapping[str, float]]) -> NearestPedal | None:
        ranked = nearest_pedals(predicted, self.atlas(), self.holdout)
        return ranked[0] if ranked else None

    def markdown(self, predicted: Mapping[str, Mapping[str, float]]) -> str:
        """The explanation line, or ``""`` when there is no KB (never raises)."""
        try:
            n = self.nearest(predicted)
        except Exception:
            log.exception("nearest-pedal explanation failed")
            return ""
        return "" if n is None else n.markdown()


__all__ = [
    "DIAL_GRID",
    "GROUPS",
    "LEVEL_POTS",
    "AtlasEntry",
    "NearestPedal",
    "PedalExplainer",
    "board_distance",
    "group_distances",
    "nearest_pedals",
    "pedal_atlas",
]
