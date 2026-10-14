"""Differentiable CLAP audio embedding (Text2FX baseline, P5 D4).

``ClapEncoder.encode_audio`` runs numpy/scipy preprocessing (polyphase resampling to 48 kHz,
``ClapFeatureExtractor``'s log-mel front end) under ``inference_mode``, so no gradient reaches
the audio. :class:`ClapAudioDiff` reimplements that preprocessing in torch and runs the audio
tower + projection with gradients w.r.t. the **input audio** (the CLAP weights are frozen):

- :func:`resample_poly` matches ``scipy.signal.resample_poly`` (Kaiser-windowed FIR,
  ``lstmabar.audio.resample``) as a polyphase gather;
- :class:`ClapAudioDiff` repeat-pads to the 10 s window (``padding="repeatpad"``), takes the
  centred, reflect-padded Hann STFT, power, the extractor's own Slaney mel filters and
  ``10 log10(max(mel, 1e-10))``, i.e. the ``truncation="rand_trunc"`` path that both cached
  checkpoints use for clips of at most 10 s (longer clips are rejected: the extractor crops
  them at a random offset).

Verified against ``encode_audio`` (cosine > 0.99 per clip) in ``tests/models/test_clap_diff.py``
(slow; needs the ``clap`` extra and a cached checkpoint).
"""

from __future__ import annotations

from functools import lru_cache
from math import gcd

import numpy as np
import torch
from torch import Tensor, nn

CLAP_SAMPLE_RATE = 48000
CLAP_MAX_SECONDS = 10


# ============================================================================ resampling


def _output_len(len_h: int, in_len: int, up: int, down: int) -> int:
    # scipy.signal._upfirdn._output_len
    return (((in_len - 1) * up + len_h) - 1) // down + 1


@lru_cache(maxsize=8)
def _poly_plan(n_in: int, up: int, down: int) -> tuple[Tensor, Tensor, Tensor]:
    """Gather indices ``(n_out, L)`` into the zero-padded input, the matching phase indices
    and the ``(up, L)`` polyphase filter bank of ``scipy.signal.resample_poly``."""
    from scipy.signal import firwin

    max_rate = max(up, down)
    half_len = 10 * max_rate
    h = firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0)) * up
    n_out = n_in * up // down + bool(n_in * up % down)
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    while _output_len(len(h) + n_pre_pad + n_post_pad, n_in, up, down) < n_out + n_pre_remove:
        n_post_pad += 1
    h = np.concatenate((np.zeros(n_pre_pad), h, np.zeros(n_post_pad)))
    taps = -(-len(h) // up)
    h = np.concatenate((h, np.zeros(taps * up - len(h))))
    bank = h.reshape(taps, up).T  # bank[p, i] = h[p + i * up]
    # upfirdn: y[m] = sum_j x[j] h[m * down - j * up]; with n = m * down = j0 * up + p,
    # y[m] = sum_i x[j0 - i] bank[p, i].
    m = np.arange(n_pre_remove, n_pre_remove + n_out)
    n = m * down
    j0, p = n // up, n % up
    j = j0[:, None] - np.arange(taps)[None, :]  # (n_out, L)
    j = np.where((j >= 0) & (j < n_in), j + 1, 0)  # index 0 of the padded input is a zero
    return (
        torch.from_numpy(j.astype(np.int64)),
        torch.from_numpy(p.astype(np.int64)),
        torch.from_numpy(bank.astype(np.float64)),
    )


def resample_poly(x: Tensor, orig_sr: int, target_sr: int) -> Tensor:
    """``(B, T)`` -> ``(B, T')``: differentiable ``scipy.signal.resample_poly`` (default
    Kaiser window, zero padding), as in :func:`lstmabar.audio.resample`."""
    if orig_sr == target_sr:
        return x
    g = gcd(int(orig_sr), int(target_sr))
    up, down = int(target_sr) // g, int(orig_sr) // g
    idx, phase, bank = _poly_plan(int(x.shape[-1]), up, down)
    xp = torch.nn.functional.pad(x, (1, 0))  # zero at index 0 for out-of-range taps
    taps = bank.to(x.dtype)[phase]  # (n_out, L)
    return (xp[:, idx] * taps).sum(-1)


# ============================================================================ CLAP audio tower


class ClapAudioDiff(nn.Module):
    """``(B, T)`` audio at ``sample_rate`` -> ``(B, D)`` L2-normalized CLAP audio embedding,
    differentiable w.r.t. the audio. Wraps a loaded :class:`~lstmabar.models.encoders.ClapEncoder`
    (its audio tower must be loaded); the tower's weights are frozen in place."""

    def __init__(self, encoder) -> None:
        super().__init__()
        encoder._need("audio")
        fe = encoder.processor.feature_extractor
        if fe.truncation != "rand_trunc" or fe.padding != "repeatpad":
            raise ValueError(
                f"{encoder.name}: only truncation=rand_trunc + padding=repeatpad are "
                f"implemented (got {fe.truncation}/{fe.padding})"
            )
        if int(fe.sampling_rate) != CLAP_SAMPLE_RATE:
            raise ValueError(f"{encoder.name}: feature extractor rate {fe.sampling_rate}")
        self.name = encoder.name
        self.dim = int(encoder.dim)
        self.model = encoder.model
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_fft = int(fe.fft_window_size)
        self.hop = int(fe.hop_length)
        self.max_samples = int(fe.nb_max_samples)
        self.register_buffer(
            "mel", torch.from_numpy(np.asarray(fe.mel_filters_slaney, dtype=np.float32))
        )  # (n_freq, n_mels)
        self.register_buffer("window", torch.hann_window(self.n_fft, periodic=True))

    def _repeatpad(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        if n > self.max_samples:
            raise ValueError(
                f"clip of {n / CLAP_SAMPLE_RATE:.2f} s exceeds the {CLAP_MAX_SECONDS} s CLAP "
                "window (the extractor would crop it at a random offset)"
            )
        if n < self.max_samples:
            x = x.repeat(1, self.max_samples // n)
            x = torch.nn.functional.pad(x, (0, self.max_samples - x.shape[-1]))
        return x

    def log_mel(self, x48: Tensor) -> Tensor:
        """``(B, T48)`` at 48 kHz -> ``(B, 1, frames, n_mels)`` model input features."""
        x = self._repeatpad(x48)
        spec = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop,
            window=self.window.to(x.dtype),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.real.square() + spec.imag.square()  # (B, F, frames)
        mel = torch.einsum("bft,fm->btm", power, self.mel.to(x.dtype))
        return (10.0 * torch.log10(mel.clamp_min(1e-10)))[:, None]

    def forward(self, audio: Tensor, sample_rate: int) -> Tensor:
        if audio.dim() != 2:
            raise ValueError(f"expected (B, T) audio, got {tuple(audio.shape)}")
        x48 = resample_poly(audio.float(), int(sample_rate), CLAP_SAMPLE_RATE)
        feats = self.log_mel(x48)
        is_longer = torch.zeros(feats.shape[0], 1, dtype=torch.bool, device=feats.device)
        out = self.model.get_audio_features(input_features=feats, is_longer=is_longer)
        emb = out if isinstance(out, Tensor) else out.pooler_output
        return torch.nn.functional.normalize(emb.float(), dim=-1)


__all__ = ["CLAP_SAMPLE_RATE", "ClapAudioDiff", "resample_poly"]
