import pytest
import torch

from lstmabar.dsp.waveshaper import waveshape

F64 = torch.float64


def _p(*vals, n=1):
    return [torch.full((n,), float(v), dtype=F64) for v in vals]


GRID = [(s, a) for s in (0.0, 0.5, 1.0) for a in (0.0, 0.5, 1.0)]


def _sine(n=4096, k0=8, amp=1.0):
    t = torch.arange(n, dtype=F64)
    return (amp * torch.sin(2 * torch.pi * k0 * t / n)).unsqueeze(0)


def _harmonics(y, k0=8, n_harm=40):
    spec = torch.fft.rfft(y[0]).abs() ** 2
    return spec[[k0 * h for h in range(1, n_harm + 1)]]


@pytest.mark.parametrize("softness,asymmetry", GRID)
@pytest.mark.parametrize("bias", [-0.5, 0.0, 0.5])
def test_bounded_zero_at_zero_and_monotonic(softness, asymmetry, bias):
    x = torch.linspace(-1e3, 1e3, 20001, dtype=F64).unsqueeze(0)
    x = torch.cat([x, torch.linspace(-5, 5, 20001, dtype=F64).unsqueeze(0)])
    s, a, b = _p(softness, asymmetry, bias, n=2)
    y = waveshape(x, s, a, b)
    assert torch.isfinite(y).all()
    assert y.abs().max() <= 1.0 + 1e-12  # bounded up to float rounding
    assert (y.diff(dim=-1) >= -1e-12).all()  # monotonic up to rounding on the plateau
    zero = waveshape(torch.zeros(2, 3, dtype=F64), s, a, b)
    assert torch.allclose(zero, torch.zeros_like(zero), atol=1e-12)


def test_extremes_hit_ceilings():
    x = torch.tensor([[-1e4, 1e4]], dtype=F64)
    s, a, b = _p(1.0, 1.0, 0.0)
    y = waveshape(x, s, a, b)
    assert y[0, 0] == pytest.approx(-0.5, abs=1e-3)
    assert y[0, 1] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("softness,asymmetry", GRID)
def test_unit_slope_at_zero(softness, asymmetry):
    h = 1e-4
    x = torch.tensor([[-h, h]], dtype=F64)
    s, a, b = _p(softness, asymmetry, 0.0)
    y = waveshape(x, s, a, b)
    assert float((y[0, 1] - y[0, 0]) / (2 * h)) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("softness", [0.0, 0.5, 1.0])
def test_symmetric_gives_odd_harmonics_only(softness):
    y = waveshape(_sine(amp=4.0), *_p(softness, 0.0, 0.0))
    harm = _harmonics(y)
    even = harm[1::2].sum()
    assert 10 * torch.log10(even / harm[0]) < -80


@pytest.mark.parametrize("asymmetry,bias", [(0.5, 0.0), (1.0, 0.0), (0.0, 0.3), (0.0, -0.3)])
def test_asymmetry_or_bias_adds_even_harmonics(asymmetry, bias):
    y = waveshape(_sine(amp=4.0), *_p(0.5, asymmetry, bias))
    harm = _harmonics(y)
    assert 10 * torch.log10(harm[1] / harm[0]) > -40


def test_harder_knee_has_more_high_harmonics():
    x = _sine(amp=3.0)

    def high_ratio(softness):
        harm = _harmonics(waveshape(x, *_p(softness, 0.0, 0.0)))
        return float(harm[4:].sum() / harm[0])  # 5th harmonic and above

    ratios = [high_ratio(s) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(r1 > r0 for r0, r1 in zip(ratios, ratios[1:], strict=False))


def test_gradients_finite_and_nonzero():
    x = torch.tensor(
        [[0.0, 1e-8, -1e-8, 0.3, -0.7, 1.0, 5.0, -50.0, 1e4]] * 3, dtype=torch.float32
    ).requires_grad_(True)
    s = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    a = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    b = torch.tensor([0.0, 0.2, -0.4], requires_grad=True)
    y = waveshape(x, s, a, b)
    (y * torch.linspace(0.5, 2, y.shape[-1])).sum().backward()
    for g in (x.grad, s.grad, a.grad, b.grad):
        assert torch.isfinite(g).all()
    assert (x.grad[:, 0] > 0.1).all()  # healthy slope at x = 0
    assert (x.grad[:, 6] > 0).all()  # saturated region still has gradient
    assert (s.grad.abs() > 0).all()
    assert (a.grad.abs() > 0).all()
    assert (b.grad.abs() > 0).all()


def test_gradcheck():
    x = torch.tensor([[-2.0, -0.3, 0.05, 0.8, 3.0], [-1.0, 0.0, 0.2, 1.5, 6.0]], dtype=F64)
    s = torch.tensor([0.2, 0.9], dtype=F64)
    a = torch.tensor([0.3, 0.7], dtype=F64)
    b = torch.tensor([0.1, -0.25], dtype=F64)
    inputs = tuple(t.requires_grad_(True) for t in (x, s, a, b))
    assert torch.autograd.gradcheck(waveshape, inputs)


def test_shape_dtype_device_preserved():
    x = torch.randn(4, 100)
    s, a, b = (torch.rand(4) for _ in range(3))
    y = waveshape(x, s, a, b - 0.5)
    assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
