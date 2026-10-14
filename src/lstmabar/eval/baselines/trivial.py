"""Trivial D4 baselines: no-op (every gate off) and mean params (masked train mean).

Both are text-blind, deterministic and have no hyperparameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from lstmabar.data.schema import FloatBoard
from lstmabar.dsp.pedalboard import ENABLED
from lstmabar.eval.baselines.base import Method, complete_board, default_board
from lstmabar.eval.data import Query, SynthItem


class NoOp(Method):
    """Every gate off (knobs at their defaults, which a disabled block ignores)."""

    name = "noop"

    def predict(self, queries: Sequence[Query]) -> list[FloatBoard]:
        return [default_board(0.0) for _ in queries]


class MeanParams(Method):
    """Per knob: the mean of the train labels over the items whose block is enabled (the
    label's gate >= 0.5; the knob default when no train item enables the block). Gate: on
    iff the train mean of that gate is >= 0.5. One item = one label (captions ignored)."""

    name = "mean"

    def __init__(self, cfg: Any = None, seed: int = 0) -> None:
        super().__init__(cfg, seed)
        self._board: FloatBoard | None = None
        self._n_train = 0

    def fit(self, train: Sequence[SynthItem]) -> None:
        if not train:
            raise ValueError("MeanParams.fit: empty train split")
        labels = [complete_board(it.label) for it in train]
        out = default_board(0.0)
        for block, knobs in out.items():
            gates = np.array([lab[block][ENABLED] for lab in labels], dtype=np.float64)
            on = gates >= 0.5
            for knob in knobs:
                if knob == ENABLED:
                    continue
                if on.any():
                    vals = np.array([lab[block][knob] for lab in labels], dtype=np.float64)
                    out[block][knob] = float(vals[on].mean())
            out[block][ENABLED] = 1.0 if float(gates.mean()) >= 0.5 else 0.0
        self._board = complete_board(out)
        self._n_train = len(train)

    def predict(self, queries: Sequence[Query]) -> list[FloatBoard]:
        if self._board is None:
            raise RuntimeError("MeanParams.predict called before fit")
        return [{b: dict(k) for b, k in self._board.items()} for _ in queries]

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "train_items": self._n_train}


__all__ = ["MeanParams", "NoOp"]
