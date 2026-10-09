import math

import numpy as np
import pytest
import torch

from lstmabar.dsp.losses import multi_resolution_stft_loss

SR = 44100


def _signal(batch: int = 2, n: int = SR // 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / SR
    tone = 0.5 * torch.sin(2 * math.pi * 220 * t) + 0.2 * torch.sin(2 * math.pi * 1320 * t)
    return tone + 0.05 * torch.randn(batch, n, generator=g)


def _np_reference(pred, target, fft_sizes, hop_ratio, eps):
    def mag(x, n_fft, hop):
        win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)  # periodic Hann
        xp = np.pad(x, ((0, 0), (n_fft // 2, n_fft // 2)), mode="reflect")
        n_frames = 1 + (xp.shape[1] - n_fft) // hop
        frames = np.stack([xp[:, i * hop : i * hop + n_fft] for i in range(n_frames)], axis=1)
        spec = np.fft.rfft(frames * win, axis=-1)
        return np.sqrt(np.maximum(np.abs(spec) ** 2, eps))

    total = 0.0
    for n_fft in fft_sizes:
        hop = int(n_fft * hop_ratio)
        mp, mt = mag(pred, n_fft, hop), mag(target, n_fft, hop)
        sc = np.sqrt(((mt - mp) ** 2).sum(axis=(1, 2))) / (np.sqrt((mt**2).sum(axis=(1, 2))) + eps)
        total += sc.mean() + np.abs(np.log(mt) - np.log(mp)).mean()
    return total / len(fft_sizes)


def test_identical_inputs_zero():
    x = _signal()
    loss = multi_resolution_stft_loss(x, x.clone())
    assert loss.dim() == 0
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_increases_with_noise_level():
    target = _signal()
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(target.shape, generator=g)
    losses = [float(multi_resolution_stft_loss(target + s * noise, target))
              for s in (1e-3, 1e-2, 1e-1, 1.0)]
    assert losses[0] > 0
    assert all(a < b for a, b in zip(losses, losses[1:], strict=False))


def test_roughly_symmetric():
    a, b = _signal(seed=0), 0.6 * _signal(seed=1)
    lab, lba = float(multi_resolution_stft_loss(a, b)), float(multi_resolution_stft_loss(b, a))
    assert lab > 0 and lba > 0
    assert 1 / 3 < lab / lba < 3


def test_gradient_flows_to_pred():
    target = _signal()
    pred = (target + 0.1 * torch.randn(target.shape)).requires_grad_()
    multi_resolution_stft_loss(pred, target).backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_silent_target_and_silent_pred_are_finite():
    pred = (0.1 * torch.randn(2, 8000)).requires_grad_()
    loss = multi_resolution_stft_loss(pred, torch.zeros(2, 8000))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()

    silent = torch.zeros(2, 8000, requires_grad=True)
    loss = multi_resolution_stft_loss(silent, torch.zeros(2, 8000))
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(silent.grad).all()


def test_matches_numpy_reference():
    pred, target = _signal(n=4000, seed=2).double(), _signal(n=4000, seed=3).double()
    sizes, hop_ratio, eps = (256, 512), 0.25, 1e-7
    ref = _np_reference(pred.numpy(), target.numpy(), sizes, hop_ratio, eps)
    out = multi_resolution_stft_loss(pred, target, sizes, hop_ratio, eps)
    assert out.dtype == torch.float64
    assert float(out) == pytest.approx(ref, rel=1e-9)


def test_dtype_and_shape_checks():
    x = _signal(n=4000)
    assert multi_resolution_stft_loss(x.double(), x.double()).dtype == torch.float64
    with pytest.raises(ValueError):
        multi_resolution_stft_loss(x, x[:, :-1])
