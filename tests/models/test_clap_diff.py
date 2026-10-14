"""Differentiable CLAP audio path (models/clap_diff.py)."""

import numpy as np
import pytest
import torch

from lstmabar.audio import resample, synth_riff
from lstmabar.models.clap_diff import resample_poly


@pytest.mark.parametrize("n", [1, 7, 1000, 13230])
def test_resample_poly_matches_scipy(n):
    x = np.random.default_rng(n).standard_normal(n).astype(np.float32)
    ref = resample(x, 44100, 48000)
    got = resample_poly(torch.from_numpy(x)[None].double(), 44100, 48000)[0].numpy()
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-5)


def test_resample_poly_differentiable_and_batched():
    x = torch.randn(2, 441, dtype=torch.float64, requires_grad=True)
    y = resample_poly(x, 44100, 48000)
    assert y.shape == (2, 480)
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert resample_poly(x, 48000, 48000) is x


@pytest.mark.slow
@pytest.mark.parametrize("name", ["clap-htsat-unfused", "larger_clap_general"])
def test_matches_encode_audio(name):
    pytest.importorskip("transformers")
    from lstmabar.models.clap_diff import ClapAudioDiff
    from lstmabar.models.encoders import ClapEncoder, is_cached

    if not is_cached(name):
        pytest.skip(f"{name} not in the local HF cache")
    torch.set_num_threads(2)
    enc = ClapEncoder(name, towers=("audio",))
    diff = ClapAudioDiff(enc)
    clips = [
        synth_riff(k, seconds=3.0, seed=i)
        for i, k in enumerate(["single_notes", "power_chords", "clean_arpeggio"])
    ]
    clips.append(np.tanh(8.0 * clips[1]).astype(np.float32))  # a distorted one
    ref = enc.encode_audio(clips, 44100)
    with torch.no_grad():
        got = torch.cat([diff(torch.from_numpy(c)[None], 44100) for c in clips]).numpy()
    cos = (ref * got).sum(-1)
    assert np.all(cos > 0.99), cos
    x = torch.from_numpy(clips[0])[None].requires_grad_(True)
    diff(x, 44100)[0, 0].backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="10 s"):
        diff(torch.zeros(1, 11 * 44100), 44100)
