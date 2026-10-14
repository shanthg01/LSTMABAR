"""Frozen text/audio encoders (CLAP checkpoints, bge-base). Wave 0: interface; wave 1 (b)
implements the loaders (``clap`` extra: transformers; loads offline from the HF cache).

Gotchas (P5 brief D5, verified with transformers 5.19): ``get_text_features`` /
``get_audio_features`` return an output object whose 512-d embedding is ``.pooler_output``;
CLAP's feature extractor expects 48 kHz audio (resample 44.1 kHz renders). bge-base:
``transformers.AutoModel``, CLS pooling, L2-normalized (no sentence-transformers).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class TextEncoder(Protocol):
    name: str  # embed_cache.FAMILIES key
    dim: int

    def encode_text(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """``(N, dim)`` float32, L2-normalized."""
        ...


class AudioEncoder(Protocol):
    name: str
    dim: int

    def encode_audio(
        self, audio: Sequence[np.ndarray], sample_rate: int, batch_size: int = 16
    ) -> np.ndarray:
        """``(N, dim)`` float32, L2-normalized; resamples to the encoder's rate itself."""
        ...


def load_text_encoder(name: str, device: str = "cpu") -> TextEncoder:
    raise NotImplementedError


def load_audio_encoder(name: str, device: str = "cpu") -> AudioEncoder:
    """CLAP checkpoints only (bge has no audio tower)."""
    raise NotImplementedError


__all__ = ["AudioEncoder", "TextEncoder", "load_audio_encoder", "load_text_encoder"]
