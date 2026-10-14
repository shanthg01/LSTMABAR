"""The P6 model as a P5 :class:`~lstmabar.eval.baselines.base.Method` (registry name
``p6_model``), so it is scored by the same runner and renderer as every baseline.

Checkpoint: the constructor's ``checkpoint`` or ``cfg.eval.model.checkpoint`` (a stage-1
``checkpoint.pt``; ``"{seed}"`` in the path is formatted with the method's seed, so the
runner's seed loop picks one trained checkpoint per seed, D3). The model ids, inputs and
mapping come from the checkpoint, never from the eval config, so they cannot drift.

Embeddings: looked up in the embedding cache (``cfg.eval.embeddings.dir``) by the query's
``text_key`` / ``audio_key`` and split. Queries missing from the cache (the demo), or whose
cached text sha256 differs from the query text, are encoded on the fly with
:mod:`lstmabar.models.encoders` when ``cfg.eval.model.encode_missing`` (default true).
Audio-cache staleness is **not** checked here (it would mean loading and hashing every dry
input), matching ``embed_cache.lookup_queries``; ``lstmabar eval embed`` owns audio freshness.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from lstmabar.data.schema import FloatBoard
from lstmabar.eval.baselines.base import Method, complete_board, vector_board
from lstmabar.eval.data import Query
from lstmabar.models.embed_cache import EmbeddingTable, load_table, text_sha256
from lstmabar.models.predictor import load_checkpoint


def _get(cfg, dotted: str, default=None):
    """``cfg.<dotted>`` for a DictConfig or a plain dict, ``default`` if absent or null."""
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        v = OmegaConf.select(cfg, dotted, default=None)
        return default if v is None else v
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or node.get(part) is None:
            return default
        node = node[part]
    return node


class ModelMethod(Method):
    name = "p6_model"
    uses_audio = True  # per instance: False for text-only checkpoints
    deterministic = True  # per instance: False when the checkpoint path has "{seed}"

    def __init__(
        self,
        cfg: Any = None,
        seed: int = 0,
        checkpoint: str | Path | None = None,
        cache_dir: str | Path | None = None,
        encode_missing: bool | None = None,
    ) -> None:
        super().__init__(cfg, seed)
        ckpt = checkpoint if checkpoint is not None else _get(cfg, "eval.model.checkpoint")
        self._ckpt_template = None if ckpt is None else str(ckpt)
        self.deterministic = self._ckpt_template is None or "{seed}" not in self._ckpt_template
        self._cache_dir = cache_dir
        enc = (
            encode_missing if encode_missing is not None else _get(cfg, "eval.model.encode_missing")
        )
        self.encode_missing = True if enc is None else bool(enc)
        self.batch_size = int(_get(cfg, "eval.model.batch_size", 1024))
        self.model = None
        self.payload: dict[str, Any] = {}
        self._tables: dict[tuple[str, str, str], EmbeddingTable | None] = {}
        self._encoders: dict[tuple[str, str], Any] = {}
        if self._ckpt_template is not None:
            self._load()

    # -- loading ------------------------------------------------------------------------

    @property
    def checkpoint_path(self) -> Path:
        if self._ckpt_template is None:
            raise ValueError(
                "p6_model needs a checkpoint: set eval.model.checkpoint (configs/train/"
                "stage1.yaml) or pass checkpoint="
            )
        return Path(self._ckpt_template.replace("{seed}", str(self.seed)))

    def _load(self) -> None:
        path = self.checkpoint_path
        if not path.exists():
            raise FileNotFoundError(f"p6_model checkpoint not found: {path.name}")
        model, payload = load_checkpoint(path)
        meta = payload.get("meta") or {}
        need = [
            k
            for k, used in (
                ("text_model", model.config.inputs != "audio"),
                ("audio_model", model.config.inputs != "text"),
            )
            if used and not meta.get(k)
        ]
        if need:
            raise ValueError(f"{path.name}: checkpoint meta lacks {need} (not a stage-1 run?)")
        self.model, self.payload = model, payload
        self.uses_audio = self.model.config.inputs != "text"
        if self.deterministic:
            warnings.warn(
                f"p6_model: fixed checkpoint {path.name} (no '{{seed}}' in the path) gives one "
                "trained seed; D3 requires >= eval.stats.min_seeds seeds for trained methods",
                stacklevel=2,
            )

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self.payload.get("meta", {}))

    @property
    def cache_dir(self) -> str:
        if self._cache_dir is not None:
            return str(self._cache_dir)
        d = _get(self.cfg, "eval.embeddings.dir")
        if d is None:
            d = _get(self.meta.get("config"), "eval.embeddings.dir")
        if d is None:
            raise ValueError("no embedding cache dir (cfg.eval.embeddings.dir)")
        return str(d)

    # -- embeddings ---------------------------------------------------------------------

    def _table(self, model: str, split: str, kind: str) -> EmbeddingTable | None:
        key = (model, split, kind)
        if key not in self._tables:
            try:
                self._tables[key] = load_table(self.cache_dir, model, split, kind)
            except FileNotFoundError:
                self._tables[key] = None
        return self._tables[key]

    def _encoder(self, model: str, kind: str):
        key = (model, kind)
        if key not in self._encoders:
            try:
                from lstmabar.models import encoders

                load = encoders.load_text_encoder if kind == "text" else encoders.load_audio_encoder
                self._encoders[key] = load(model)
            except NotImplementedError as e:
                raise RuntimeError(
                    f"p6_model: query not in the {model} {kind} embedding cache and the "
                    "on-the-fly encoders are not implemented yet; build the cache with "
                    "`lstmabar eval embed`"
                ) from e
            except ImportError as e:
                raise RuntimeError(
                    f"p6_model: encoding {kind} on the fly needs the `clap` extra (transformers)"
                ) from e
        return self._encoders[key]

    def _embed(self, queries: Sequence[Query], model: str, kind: str) -> np.ndarray:
        attr = "text_key" if kind == "text" else "audio_key"
        rows: list[np.ndarray | None] = []
        missing: list[int] = []
        for i, q in enumerate(queries):
            key = getattr(q, attr)
            t = self._table(model, q.split, kind)
            hit = t is not None and key in t
            if hit and kind == "text" and q.text:
                hit = not t.stale({key: text_sha256(q.text)})
            if hit:
                rows.append(t.lookup([key])[0])
            else:
                rows.append(None)
                missing.append(i)
        if missing:
            if not self.encode_missing:
                q = queries[missing[0]]
                raise KeyError(
                    f"p6_model: {len(missing)} queries missing or stale in the {model} {kind} "
                    f"cache (first: {q.split}/{getattr(q, attr)!r}) and encode_missing is off"
                )
            enc = self._encoder(model, kind)
            if kind == "text":
                vecs = enc.encode_text([queries[i].text for i in missing])
            else:
                rates = {queries[i].sample_rate for i in missing}
                if len(rates) != 1:
                    raise ValueError(f"mixed sample rates in one batch: {sorted(rates)}")
                vecs = enc.encode_audio([queries[i].load_dry() for i in missing], rates.pop())
            for i, v in zip(missing, np.asarray(vecs, dtype=np.float32), strict=True):
                rows[i] = v
        return np.stack(rows).astype(np.float32)

    # -- Method interface -------------------------------------------------------------------

    def predict(self, queries: Sequence[Query]) -> list[FloatBoard]:
        if self.model is None:
            self._load()
        assert self.model is not None
        cfgm = self.model.config
        meta = self.meta
        out: list[FloatBoard] = []
        for s in range(0, len(queries), self.batch_size):
            qs = list(queries[s : s + self.batch_size])
            text = audio = None
            if cfgm.inputs != "audio":
                text = torch.from_numpy(self._embed(qs, str(meta["text_model"]), "text"))
            if cfgm.inputs != "text":
                audio = torch.from_numpy(self._embed(qs, str(meta["audio_model"]), "audio"))
            with torch.no_grad():
                board = self.model.board(self.model(text, audio)).double().numpy()
            out.extend(complete_board(vector_board(row.tolist())) for row in board)
        return out

    def info(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.model is not None:
            meta = self.meta
            d.update(
                checkpoint=self.checkpoint_path.name,  # never a local path (reports)
                git_sha=(meta.get("git") or {}).get("sha"),
                text_model=meta.get("text_model"),
                audio_model=meta.get("audio_model"),
                inputs=self.model.config.inputs,
                mapping=self.model.config.mapping,
                train_seed=meta.get("seed"),
                best_epoch=meta.get("epoch"),
            )
        return d


__all__ = ["ModelMethod"]
