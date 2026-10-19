"""Frozen text/audio encoders (CLAP checkpoints, bge-base). Loaders need the ``clap`` extra
(transformers). A checkpoint already in the local Hugging Face cache loads offline
(``local_files_only``); one that is not (a fresh Colab VM) is downloaded, unless
``HF_HUB_OFFLINE`` is set.

Gotchas (P5 brief D5, verified with transformers 5.19): ``get_text_features`` /
``get_audio_features`` return an output object whose 512-d embedding is ``.pooler_output``;
CLAP's feature extractor expects 48 kHz audio (resample 44.1 kHz renders). bge-base:
``transformers.AutoModel``, CLS pooling, L2-normalized (no sentence-transformers).

Clips shorter than the CLAP window (10 s) are repeat-padded by the feature extractor; longer
ones would be cropped by ``rand_trunc`` (random), so callers embed clips of at most 10 s (all
dataset clips are 3 s).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from lstmabar.audio import resample
from lstmabar.models.embed_cache import FAMILIES, HF_IDS, l2_normalize

CLAP_SAMPLE_RATE = 48000
CLAP_MAX_SECONDS = 10  # feature-extractor window; longer clips are randomly cropped
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


def local_only(name: str) -> bool:
    """Load from the local HF cache only: the model is cached, or ``HF_HUB_OFFLINE`` is
    set. Otherwise ``from_pretrained`` may download it (needs network)."""
    import os

    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return is_cached(name)


def is_cached(name: str) -> bool:
    """Whether the model's config is in the local HF cache (no network)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover - transformers depends on huggingface_hub
        return False
    hit = try_to_load_from_cache(HF_IDS[name], "config.json")
    return isinstance(hit, str)


class ClapEncoder:
    """Text + audio towers of a CLAP checkpoint (``transformers.ClapModel``).

    ``towers`` keeps only some towers in memory (the other one is dropped after loading, e.g.
    for bulk text embedding on a memory-starved machine); calling a dropped tower raises.
    """

    def __init__(self, name: str, device: str = "cpu", towers: tuple[str, ...] = ("text", "audio")):
        import torch
        from transformers import ClapModel, ClapProcessor

        if FAMILIES.get(name) != "clap":
            raise KeyError(f"{name!r} is not a CLAP checkpoint; known: {sorted(FAMILIES)}")
        self.name = name
        self.device = torch.device(device)
        hf = HF_IDS[name]
        local = local_only(name)
        self.processor = ClapProcessor.from_pretrained(hf, local_files_only=local)
        model = ClapModel.from_pretrained(hf, local_files_only=local).eval()
        self.towers = tuple(towers)
        for tower in ("text", "audio"):
            if tower not in self.towers:
                setattr(model, f"{tower}_model", None)
                setattr(model, f"{tower}_projection", None)
        self.model = model.to(self.device)
        self.dim = int(self.model.config.projection_dim)

    def _need(self, tower: str) -> None:
        if tower not in self.towers:
            raise RuntimeError(f"{self.name}: loaded without its {tower} tower")

    def encode_text(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        import torch

        self._need("text")
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

        self._need("audio")
        out = []
        with torch.inference_mode():
            for i in range(0, len(audio), batch_size):
                chunk = [
                    resample(np.asarray(a, dtype=np.float32), sample_rate, CLAP_SAMPLE_RATE)
                    for a in audio[i : i + batch_size]
                ]
                if any(len(c) > CLAP_MAX_SECONDS * CLAP_SAMPLE_RATE for c in chunk):
                    warnings.warn(
                        f"{self.name}: clip longer than {CLAP_MAX_SECONDS} s; CLAP's feature "
                        "extractor crops it at a random offset (rand_trunc)",
                        stacklevel=2,
                    )
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
        local = local_only(name)
        self.tokenizer = AutoTokenizer.from_pretrained(hf, local_files_only=local)
        self.model = AutoModel.from_pretrained(hf, local_files_only=local).eval().to(self.device)
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


def load_text_encoder(name: str, device: str = "cpu", lean: bool = False) -> TextEncoder:
    """``lean``: a CLAP checkpoint keeps only its text tower in memory."""
    if name not in FAMILIES:
        raise KeyError(f"unknown embedding model {name!r}; known: {sorted(FAMILIES)}")
    if FAMILIES[name] == "clap":
        return ClapEncoder(name, device, towers=("text",) if lean else ("text", "audio"))
    return BgeEncoder(name, device)


def load_audio_encoder(name: str, device: str = "cpu", lean: bool = False) -> AudioEncoder:
    """CLAP checkpoints only (bge has no audio tower). ``lean``: keep only the audio tower."""
    if FAMILIES.get(name) != "clap":
        raise KeyError(f"{name!r} has no audio tower; CLAP checkpoints only")
    return ClapEncoder(name, device, towers=("audio",) if lean else ("text", "audio"))


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
