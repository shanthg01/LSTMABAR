"""P6 stage-1 data (P6 prep notes, "Data loading"): cached embeddings + labels, no audio.

Per item: one caption sampled per step (template or paraphrase, uniformly among the item's
captions), the item's dry-input audio embedding (``audio_key = item_id``, the dry clip at the
item's own level, P5 D5), the normalized label in ``BOARD_LAYOUT`` order with a knob mask from
the label's gates, and a sample weight: ``train.clamped_weight`` for clamped grey-box items
(``engine == "greybox" and clamped``; P4 decision 10 / R6), 1 otherwise (white-box items train
on their calibrated ``board_params``, R4).

Everything is held as tensors (24.5k items x 512-d is ~50 MB per modality), and batches are
sliced directly, so an epoch on CPU takes seconds.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from lstmabar.eval.baselines.base import board_vector, complete_board
from lstmabar.eval.data import SynthItem
from lstmabar.models.embed_cache import EmbeddingTable, load_table, text_sha256
from lstmabar.models.predictor import BoardHeadLayout


def item_weight(item: SynthItem, clamped_weight: float) -> float:
    """``clamped_weight`` for clamped grey-box items (their derived label collapsed at the
    60 dB gain clamp), else 1."""
    it = item.item
    return float(clamped_weight) if (it.engine == "greybox" and it.clamped) else 1.0


def label_arrays(
    items: Sequence[SynthItem], layout: BoardHeadLayout | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(knobs (N, K), gates (N, G), knob_mask (N, K))`` float32 from the items' labels; a
    knob is unmasked iff its block's label gate is on (>= 0.5)."""
    layout = layout or BoardHeadLayout.default()
    board = np.array([board_vector(complete_board(i.label)) for i in items], dtype=np.float32)
    board = board.reshape(len(items), len(layout.layout))
    knobs = board[:, list(layout.knob_index)]
    gates = (board[:, list(layout.gate_index)] >= 0.5).astype(np.float32)
    mask = gates[:, list(layout.knob_block)]
    return knobs, gates, mask


@dataclass
class Batch:
    text: Tensor | None
    audio: Tensor | None
    knobs: Tensor
    gates: Tensor
    knob_mask: Tensor
    weights: Tensor
    item_index: Tensor

    def __len__(self) -> int:
        return int(self.knobs.shape[0])


class Stage1Dataset(Dataset):
    """Embeddings + labels for one split.

    ``text_table`` / ``audio_table`` may be ``None`` for single-modality models. Captions of
    each item are looked up by ``caption_id`` in the text table (``KeyError`` if missing);
    with ``check_text_sha`` a caption whose cached sha256 differs from its text raises.
    Items without captions are dropped only if a text table is used (``n_dropped``).
    """

    def __init__(
        self,
        items: Sequence[SynthItem],
        text_table: EmbeddingTable | None,
        audio_table: EmbeddingTable | None,
        clamped_weight: float = 1.0,
        check_text_sha: bool = True,
    ) -> None:
        if text_table is None and audio_table is None:
            raise ValueError("need at least one of text_table / audio_table")
        self.layout = BoardHeadLayout.default()
        if text_table is not None:
            kept = [i for i in items if i.captions]
            self.n_dropped = len(items) - len(kept)
            items = kept
        else:
            self.n_dropped = 0
        if not items:
            raise ValueError("no items with captions")
        self.items = list(items)
        self.item_ids = [i.item_id for i in self.items]
        knobs, gates, mask = label_arrays(self.items, self.layout)
        self.knobs = torch.from_numpy(knobs)
        self.gates = torch.from_numpy(gates)
        self.knob_mask = torch.from_numpy(mask)
        self.weights = torch.tensor(
            [item_weight(i, clamped_weight) for i in self.items], dtype=torch.float32
        )

        self.text: Tensor | None = None
        if text_table is not None:
            cap_ids = [c.caption_id for i in self.items for c in i.captions]
            if check_text_sha:
                stale = text_table.stale(
                    {c.caption_id: text_sha256(c.text) for i in self.items for c in i.captions}
                )
                if stale:
                    raise ValueError(
                        f"{len(stale)} captions missing or stale in the {text_table.model} text "
                        f"cache (first: {stale[0]!r}); run `lstmabar eval embed`"
                    )
            self.text = torch.from_numpy(text_table.lookup(cap_ids))
            counts = [len(i.captions) for i in self.items]
            self.cap_count = torch.tensor(counts, dtype=torch.long)
            self.cap_offset = torch.cumsum(self.cap_count, 0) - self.cap_count
            self.cap_item = torch.repeat_interleave(torch.arange(len(self.items)), self.cap_count)
        self.audio: Tensor | None = None
        if audio_table is not None:
            self.audio = torch.from_numpy(audio_table.lookup(self.item_ids))
        self._caption_choice: Tensor | None = None
        self._audio_perm: Tensor | None = None

    # -- torch Dataset view (one sampled caption per item per epoch) -----------------------

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, generator: torch.Generator, permute_audio: bool = False) -> None:
        """Resample the caption of every item (and, for the shuffled-audio ablation, a
        permutation of audio embeddings across items) for the next epoch."""
        self._caption_choice = self.sample_captions(generator) if self.text is not None else None
        self._audio_perm = (
            torch.randperm(len(self), generator=generator)
            if (permute_audio and self.audio is not None)
            else None
        )

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        b = self._gather(torch.tensor([i]), self._caption_choice, self._audio_perm)
        return {k: v[0] for k, v in vars(b).items() if v is not None}

    # -- fast batching ------------------------------------------------------------------

    def sample_captions(self, generator: torch.Generator) -> Tensor:
        """``(N,)`` caption row index per item, uniform among the item's captions."""
        assert self.text is not None
        u = torch.rand(len(self), generator=generator)
        pick = torch.minimum((u * self.cap_count).long(), self.cap_count - 1)
        return self.cap_offset + pick

    def _gather(self, idx: Tensor, cap_choice: Tensor | None, audio_perm: Tensor | None) -> Batch:
        text = None
        if self.text is not None:
            if cap_choice is None:
                cap_choice = self.cap_offset  # first caption (deterministic default)
            text = self.text[cap_choice[idx]]
        audio = None
        if self.audio is not None:
            a_idx = audio_perm[idx] if audio_perm is not None else idx
            audio = self.audio[a_idx]
        return Batch(
            text=text,
            audio=audio,
            knobs=self.knobs[idx],
            gates=self.gates[idx],
            knob_mask=self.knob_mask[idx],
            weights=self.weights[idx],
            item_index=idx,
        )

    def train_batches(
        self, batch_size: int, generator: torch.Generator, permute_audio: bool = False
    ) -> Iterator[Batch]:
        """One epoch: items shuffled, one caption sampled per item."""
        self.set_epoch(generator, permute_audio)
        order = torch.randperm(len(self), generator=generator)
        for s in range(0, len(self), batch_size):
            yield self._gather(order[s : s + batch_size], self._caption_choice, self._audio_perm)

    def eval_batches(self, batch_size: int) -> Iterator[Batch]:
        """Every (item, caption) pair once, in order (each caption is one eval query); per
        item only when there is no text."""
        if self.text is None:
            for s in range(0, len(self), batch_size):
                yield self._gather(torch.arange(s, min(s + batch_size, len(self))), None, None)
            return
        n_caps = int(self.text.shape[0])
        for s in range(0, n_caps, batch_size):
            rows = torch.arange(s, min(s + batch_size, n_caps))
            idx = self.cap_item[rows]
            b = self._gather(idx, None, None)
            b.text = self.text[rows]
            yield b

    def steps_per_epoch(self, batch_size: int) -> int:
        return -(-len(self) // batch_size)


def load_tables(cfg, split: str) -> tuple[EmbeddingTable | None, EmbeddingTable | None]:
    """The (text, audio) tables a ``cfg.model`` needs for one split."""
    cache_dir = str(cfg.eval.embeddings.dir)
    inputs = str(cfg.model.inputs)
    text = (
        load_table(cache_dir, str(cfg.model.text_model), split, "text")
        if inputs != "audio"
        else None
    )
    audio = (
        load_table(cache_dir, str(cfg.model.audio_model), split, "audio")
        if inputs != "text"
        else None
    )
    return text, audio


__all__ = ["Batch", "Stage1Dataset", "item_weight", "label_arrays", "load_tables"]
