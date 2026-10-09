import time

import pytest
import torch

from lstmabar.dsp.oversample import HALF_WIDTH, downsample, oversampled, upsample
from lstmabar.dsp.waveshaper import waveshape

FS = 44100
FACTORS = [2, 4, 8]


def _tone(freq, n, fs=FS, dtype=torch.float64, phase=0.3):
    t = torch.arange(n, dtype=dtype)
    return torch.sin(2 * torch.pi * freq * t / fs + phase)


def _bandlimited_noise(n, seed=0):
    """White noise low-passed well inside the passband (< 0.35 fs) via FFT masking."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, n, generator=g, dtype=torch.float64)
    spec = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(n)
    spec[:, freqs > 0.35] = 0
    return torch.fft.irfft(spec, n)


@pytest.mark.parametrize("factor", FACTORS)
def test_roundtrip_reconstructs_bandlimited_signal(factor):
    x = _bandlimited_noise(4096)
    y = downsample(upsample(x, factor), factor)
    edge = 2 * HALF_WIDTH
    err = (y - x)[:, edge:-edge].abs().max() / x.abs().max()
    assert err < 1e-3


@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("freq", [100.0, 1000.0, 10000.0, 17000.0])
def test_upsample_zero_phase_and_unity_gain(factor, freq):
    n = 2048
    x = _tone(freq, n).unsqueeze(0)
    y = upsample(x, factor)
    assert y.shape == (1, n * factor)
    expected = _tone(freq, n * factor, fs=FS * factor).unsqueeze(0)  # exact high-rate sine
    edge = 2 * HALF_WIDTH * factor
    assert (y - expected)[:, edge:-edge].abs().max() < 1e-3
    assert (y[:, ::factor] - x)[:, 2 * HALF_WIDTH : -2 * HALF_WIDTH].abs().max() < 1e-3


@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("freq", [100.0, 1000.0, 10000.0, 17000.0])
def test_downsample_passband_unity_gain_zero_phase(factor, freq):
    n = 2048
    x = _tone(freq, n * factor, fs=FS * factor).unsqueeze(0)
    y = downsample(x, factor)
    edge = 2 * HALF_WIDTH
    assert (y - _tone(freq, n))[:, edge:-edge].abs().max() < 1e-3


@pytest.mark.parametrize("factor", FACTORS)
def test_downsample_rejects_above_new_nyquist(factor):
    n = 4096
    fs_hi = FS * factor
    edge = 2 * HALF_WIDTH
    for freq in (0.5 * FS + 200.0, 0.6 * FS, 0.5 * fs_hi - 500.0):
        y = downsample(_tone(freq, n * factor, fs=fs_hi).unsqueeze(0), factor)
        peak = y[:, edge:-edge].abs().max()
        assert 20 * torch.log10(peak) < -60, (freq, float(peak))


def _alias_db(factor):
    n = torch.arange(3 * FS, dtype=torch.float64)
    x = (20 * torch.sin(2 * torch.pi * 5000 * n / FS)).unsqueeze(0)
    z = torch.zeros(1, dtype=torch.float64)
    s = torch.ones(1, dtype=torch.float64)
    y = oversampled(lambda u: waveshape(u, s, z, z), x, factor)[0, FS : 2 * FS]
    power = torch.fft.rfft(y).abs() ** 2  # 1 s window: bin k = k Hz
    k = torch.arange(power.numel())
    harmonic = (k % 5000 == 0) & (k > 0)
    return float(10 * torch.log10(power[~harmonic].sum() / power[5000]))


def test_oversampling_reduces_aliasing():
    a1, a8 = _alias_db(1), _alias_db(8)
    assert a8 < a1 - 20, (a1, a8)


@pytest.mark.parametrize("factor", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shapes_dtype_device_preserved(factor, dtype):
    x = torch.randn(3, 512, dtype=dtype)
    up = upsample(x, factor)
    assert up.shape == (3, 512 * factor) and up.dtype == dtype and up.device == x.device
    down = downsample(up, factor)
    assert down.shape == x.shape and down.dtype == dtype
    out = oversampled(torch.tanh, x, factor)
    assert out.shape == x.shape and out.dtype == dtype


def test_factor_one_is_identity():
    x = torch.randn(2, 100)
    assert torch.equal(upsample(x, 1), x)
    assert torch.equal(downsample(x, 1), x)
    assert torch.equal(oversampled(torch.tanh, x, 1), torch.tanh(x))


def test_invalid_inputs_raise():
    x = torch.randn(2, 30)
    with pytest.raises(ValueError):
        downsample(x, 4)
    with pytest.raises(ValueError):
        upsample(x, 3)
    with pytest.raises(ValueError):
        oversampled(torch.tanh, x, 16)


def test_gradients_flow_through_oversampled():
    x = (0.5 * torch.randn(2, 1024)).requires_grad_(True)
    s = torch.tensor([0.3, 0.9], requires_grad=True)
    a = torch.tensor([0.2, 0.6], requires_grad=True)
    b = torch.tensor([0.1, -0.1], requires_grad=True)
    y = oversampled(lambda u: waveshape(5 * u, s, a, b), x, 4)
    y.square().sum().backward()
    for g in (x.grad, s.grad, a.grad, b.grad):
        assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_gradcheck_resamplers():
    x = torch.randn(1, 40, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: upsample(t, 2), (x,))
    assert torch.autograd.gradcheck(lambda t: downsample(t, 2), (x,))
    assert torch.autograd.gradgradcheck(lambda t: oversampled(torch.tanh, t, 4), (x,))


def test_timing_report(capsys):
    """Report CPU fwd+bwd time of oversampled(waveshape) at 4x, batch 8 x 3 s (not asserted)."""
    torch.manual_seed(0)
    x = (0.3 * torch.randn(8, 3 * FS)).requires_grad_(True)
    s, a, b = (torch.full((8,), v, requires_grad=True) for v in (0.7, 0.3, 0.1))
    best = float("inf")
    for _ in range(2):
        t0 = time.perf_counter()
        y = oversampled(lambda u: waveshape(10 * u, s, a, b), x, 4)
        y.square().mean().backward()
        best = min(best, time.perf_counter() - t0)
    with capsys.disabled():
        print(f"\n[timing] oversampled(waveshape) x4, 8 x 3 s @ 44.1 kHz, fwd+bwd: {best:.3f} s")
    assert best < 30.0
