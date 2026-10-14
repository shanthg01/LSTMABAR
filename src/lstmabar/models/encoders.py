"""Frozen text/audio encoders (CLAP checkpoints, bge-base). Loaders need the ``clap`` extra
(transformers) and load offline from the local Hugging Face cache (``local_files_only``).

Gotchas (P5 brief D5, verified with transformers 5.19): ``get_text_features`` /
``get_audio_features`` return an output object whose 512-d embedding is ``.pooler_output``;
CLAP's feature extractor expects 48 kHz audio (resample 44.1 kHz renders). bge-base:
``transformers.AutoModel``, CLS pooling, L2-normalized (no sentence-transformers).

Clips shorter than the CLAP window (10 s) are repeat-padded by the feature extractor; longer
ones would be cropped by ``rand_trunc`` (random), so callers embed clips of at most 10 s (all
dataset clips are 3 s).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from lstmabar.audio import resample
from lstmabar.models.embed_cache import FAMILIES, HF_IDS, l2_normalize

CLAP_SAMPLE_RATE = 48000
BGE_MAX_LENGTH = 512


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


def _features(out):
    """``get_*_features`` returns a tensor in older transformers, an output object in 5.x."""
    import torch

    return out if isinstance(out, torch.Tensor) else out.pooler_output


def is_cached(name: str) -> bool:
    """Whether the model's config is in the local HF cache (no network)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover - transformers depends on huggingface_hub
        return False
    hit = try_to_load_from_cache(HF_IDS[name], "config.json")
    return isinstance(hit, str)


class ClapEncoder:
    """Text + audio towers of a CLAP checkpoint (``transformers.ClapModel``)."""

    def __init__(self, name: str, device: str = "cpu"):
        import torch
        from transformers import ClapModel, ClapProcessor

        if FAMILIES.get(name) != "clap":
            raise KeyError(f"{name!r} is not a CLAP checkpoint; known: {sorted(FAMILIES)}")
        self.name = name
        self.device = torch.device(device)
        hf = HF_IDS[name]
        self.processor = ClapProcessor.from_pretrained(hf, local_files_only=True)
        self.model = ClapModel.from_pretrained(hf, local_files_only=True).eval().to(self.device)
        self.dim = int(self.model.config.projection_dim)

    def encode_text(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        import torch

        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                t = self.processor.tokenizer(
                    list(texts[i : i + batch_size]),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                out.append(_features(self.model.get_text_features(**t)).float().cpu().numpy())
        return _stack(out, self.dim)

    def encode_audio(
        self, audio: Sequence[np.ndarray], sample_rate: int, batch_size: int = 16
    ) -> np.ndarray:
        import torch

        out = []
        with torch.inference_mode():
            for i in range(0, len(audio), batch_size):
                chunk = [
                    resample(np.asarray(a, dtype=np.float32), sample_rate, CLAP_SAMPLE_RATE)
                    for a in audio[i : i + batch_size]
                ]
                a_in = self.processor.feature_extractor(
                    chunk, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
                ).to(self.device)
                out.append(_features(self.model.get_audio_features(**a_in)).float().cpu().numpy())
        return _stack(out, self.dim)


class BgeEncoder:
    """``BAAI/bge-base-en-v1.5`` sentence embeddings: CLS pooling, L2-normalized."""

    def __init__(self, name: str = "bge-base-en-v1.5", device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if FAMILIES.get(name) != "bge":
            raise KeyError(f"{name!r} is not a bge model; known: {sorted(FAMILIES)}")
        self.name = name
        self.device = torch.device(device)
        hf = HF_IDS[name]
        self.tokenizer = AutoTokenizer.from_pretrained(hf, local_files_only=True)
        self.model = AutoModel.from_pretrained(hf, local_files_only=True).eval().to(self.device)
        self.dim = int(self.model.config.hidden_size)

    def encode_text(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        import torch

        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                t = self.tokenizer(
                    list(texts[i : i + batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=BGE_MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                h = self.model(**t).last_hidden_state[:, 0]
                out.append(h.float().cpu().numpy())
        return _stack(out, self.dim)


def _stack(chunks: list[np.ndarray], dim: int) -> np.ndarray:
    if not chunks:
        return np.zeros((0, dim), dtype=np.float32)
    return l2_normalize(np.concatenate(chunks, axis=0))


def load_text_encoder(name: str, device: str = "cpu") -> TextEncoder:
    if name not in FAMILIES:
        raise KeyError(f"unknown embedding model {name!r}; known: {sorted(FAMILIES)}")
    return ClapEncoder(name, device) if FAMILIES[name] == "clap" else BgeEncoder(name, device)


def load_audio_encoder(name: str, device: str = "cpu") -> AudioEncoder:
    """CLAP checkpoints only (bge has no audio tower)."""
    if FAMILIES.get(name) != "clap":
        raise KeyError(f"{name!r} has no audio tower; CLAP checkpoints only")
    return ClapEncoder(name, device)


__all__ = [
    "CLAP_SAMPLE_RATE",
    "AudioEncoder",
    "BgeEncoder",
    "ClapEncoder",
    "TextEncoder",
    "is_cached",
    "load_audio_encoder",
    "load_text_encoder",
]
