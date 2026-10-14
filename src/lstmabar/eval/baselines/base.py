"""Method interface for P5 baselines and the P6 model (wave 0 contract, D4).

A method maps queries (:class:`lstmabar.eval.data.Query`: instruction text + dry input) to
**normalized free-board parameters with gates** (``FloatBoard`` over ``default_pedalboard``:
compressor -> drive -> eq). Methods never output pedal names; the board is the only output
space, so every method is scored with the same renderer.

Lifecycle used by the runner: ``fit(train)`` once; then hyperparameter selection on val only
(D8, never on gold): for each ``d`` in ``hparam_grid()`` the runner calls ``set_hparams(d)``,
scores val with ``runner.score_val`` and keeps the ``d`` minimizing
``proxy.selection_score`` (``runner.select_hparams``); methods with cheaper internal
criteria may instead override ``select(val)``. Then ``predict`` per split.
"""

from __future__ import annotations

import importlib
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from lstmabar.data.schema import FloatBoard
from lstmabar.dsp.pedalboard import ENABLED, default_pedalboard
from lstmabar.eval.data import Query, SynthItem


class Method(ABC):
    """A text (+ optional audio) -> board method."""

    name: str = "method"
    uses_audio: bool = False  # True: reads Query.load_dry() / audio embeddings
    deterministic: bool = True  # False: run with >= eval.stats.min_seeds seeds (D3)

    def __init__(self, cfg: Any = None, seed: int = 0) -> None:
        self.cfg = cfg
        self.seed = seed

    def fit(self, train: Sequence[SynthItem]) -> None:  # noqa: B027 - optional hook
        """Learn from the train split (labels, captions). Default: nothing to learn."""

    def hparam_grid(self) -> list[dict[str, Any]]:
        """Candidate hyperparameters for runner-driven val selection (``[]``: none)."""
        return []

    def set_hparams(self, hparams: dict[str, Any]) -> None:  # noqa: B027 - optional hook
        """Apply one entry of :meth:`hparam_grid`."""

    def select(self, val: Sequence[SynthItem]) -> dict[str, Any]:
        """Method-internal val selection (D8). Returns what was chosen (for the report).
        Default: nothing (the runner handles :meth:`hparam_grid`)."""
        return {}

    @abstractmethod
    def predict(self, queries: Sequence[Query]) -> list[FloatBoard]:
        """One complete normalized board (see :func:`complete_board`) per query, in order."""

    def info(self) -> dict[str, Any]:
        """Report metadata (model ids, chosen k, cost, ...)."""
        return {"name": self.name}


REGISTRY: dict[str, str] = {
    # name: "module:Class" under lstmabar.eval.baselines (wave 1/2 fill these in)
    "noop": "trivial:NoOp",
    "mean": "trivial:MeanParams",
    "knn_clap": "knn:KnnClap",
    "knn_bge": "knn:KnnBge",
    "llm_zero_shot": "llm:LlmZeroShot",
    "text2fx": "text2fx:Text2Fx",
    "p6_model": "lstmabar.models.method:ModelMethod",  # P6 (6.1); fully qualified
}


def build_method(name: str, cfg: Any = None, seed: int = 0) -> Method:
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; known: {sorted(REGISTRY)}")
    module, cls = REGISTRY[name].split(":")
    if "." not in module:  # short form: a module under lstmabar.eval.baselines
        module = f"lstmabar.eval.baselines.{module}"
    m = importlib.import_module(module)
    return getattr(m, cls)(cfg, seed=seed)


# ============================================================================ board helpers

_BOARD = default_pedalboard()
BOARD_LAYOUT: tuple[tuple[str, str], ...] = tuple(_BOARD.layout())
"""``(block, knob)`` pairs in ``Pedalboard.layout`` order (knobs, then ``"enabled"``)."""


def default_board(enabled: float = 0.0) -> FloatBoard:
    """Every knob at its normalized default; every gate set to ``enabled``."""
    out: FloatBoard = {}
    for name in _BOARD.block_names:
        block = _BOARD.blocks[name]
        out[name] = {s.name: float(s.default_normalized) for s in block.param_specs}
        out[name][ENABLED] = float(enabled)
    return out


def complete_board(board: Mapping[str, Mapping[str, float]], enabled: float = 0.0) -> FloatBoard:
    """Fill missing blocks/knobs with defaults (missing gates with ``enabled``), clamp every
    value to [0, 1] and reject unknown names or non-finite values."""
    out = default_board(enabled)
    for b, knobs in board.items():
        if b not in out:
            raise KeyError(f"unknown block {b!r}")
        for k, v in knobs.items():
            if k not in out[b]:
                raise KeyError(f"unknown knob {b}.{k}")
            v = float(v)
            if not math.isfinite(v):
                raise ValueError(f"{b}.{k} is not finite")
            out[b][k] = min(max(v, 0.0), 1.0)
    return out


def board_vector(board: Mapping[str, Mapping[str, float]]) -> list[float]:
    """A complete board as a flat list in :data:`BOARD_LAYOUT` order."""
    return [float(board[b][k]) for b, k in BOARD_LAYOUT]


def vector_board(v: Sequence[float]) -> FloatBoard:
    """Inverse of :func:`board_vector`."""
    if len(v) != len(BOARD_LAYOUT):
        raise ValueError(f"expected {len(BOARD_LAYOUT)} values, got {len(v)}")
    out: FloatBoard = {}
    for (b, k), x in zip(BOARD_LAYOUT, v, strict=True):
        out.setdefault(b, {})[k] = float(x)
    return out


__all__ = [
    "BOARD_LAYOUT",
    "REGISTRY",
    "Method",
    "board_vector",
    "build_method",
    "complete_board",
    "default_board",
    "vector_board",
]
