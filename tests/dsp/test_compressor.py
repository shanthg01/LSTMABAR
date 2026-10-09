import math
import time

import pytest
import torch

from lstmabar.dsp.compressor import _BranchingOnePole, compress

SR = 44100


def _sine(amp_db: float, seconds: float = 1.0, freq: float = 1000.0, batch: int = 1):
    t = torch.arange(int(seconds * SR)) / SR
    x = 10 ** (amp_db / 20) * torch.sin(2 * math.pi * freq * t)
    return x.expand(batch, -1).clone()


def _p(v, b: int = 1, dtype=torch.float32):
    return torch.full((b,), float(v), dtype=dtype)


def _run(x, threshold=-20.0, ratio=4.0, attack=5.0, release=100.0, **kw):
    b = x.shape[0]
    return compress(
        x,
        _p(threshold, b, x.dtype),
        _p(ratio, b, x.dtype),
        _p(attack, b, x.dtype),
        _p(release, b, x.dtype),
        SR,
        **kw,
    )


def _peak_db(y):
    return 20 * math.log10(float(y.abs().max()))


def test_ratio_one_is_identity():
    x = torch.randn(2, SR // 2) * 0.5
    assert torch.allclose(_run(x, threshold=-40.0, ratio=1.0), x, atol=1e-6)


def test_below_threshold_unchanged():
    x = _sine(-30.0)
    assert torch.allclose(_run(x, threshold=-20.0), x, atol=1e-7)


@pytest.mark.parametrize("ratio", [2.0, 4.0, 10.0])
def test_steady_state_static_curve(ratio):
    in_db, thr = -6.0, -20.0
    y = _run(_sine(in_db), threshold=thr, ratio=ratio)
    expected = thr + (in_db - thr) / ratio
    assert _peak_db(y[:, SR // 2 :]) == pytest.approx(expected, abs=1.0)


def test_larger_ratio_more_reduction():
    x = _sine(-6.0)
    peaks = [_peak_db(_run(x, ratio=r)[:, SR // 2 :]) for r in (1.5, 3.0, 6.0, 20.0)]
    assert all(a > b for a, b in zip(peaks, peaks[1:], strict=False))


def _step_gain_db(attack, release, loud_first: bool):
    # DC steps make the instantaneous gain y/x exact.
    quiet, loud = 0.01, 1.0  # -40 dBFS (below threshold) and 0 dBFS
    n = SR // 2
    a, b = (loud, quiet) if loud_first else (quiet, loud)
    x = torch.cat([torch.full((n,), a), torch.full((n,), b)]).unsqueeze(0)
    y = _run(x, threshold=-20.0, ratio=10.0, attack=attack, release=release, knee_db=0.0)
    return (20 * torch.log10(y / x))[0], n


def test_attack_time():
    attack = 20.0
    g, n = _step_gain_db(attack, 200.0, loud_first=False)
    full = -18.0  # 0 dB in, -20 threshold, ratio 10 -> -18 dB reduction
    assert float(g[-1]) == pytest.approx(full, abs=0.1)
    t63 = int(torch.nonzero(g[n:] <= 0.632 * full)[0]) / SR * 1000
    assert attack / 2 <= t63 <= attack * 2


def test_release_time():
    release = 100.0
    g, n = _step_gain_db(5.0, release, loud_first=True)
    assert float(g[n - 1]) == pytest.approx(-18.0, abs=0.1)
    t63 = int(torch.nonzero(g[n:] >= 0.368 * -18.0)[0]) / SR * 1000
    assert release / 2 <= t63 <= release * 2


def test_makeup_adds_exact_db():
    x = torch.randn(2, SR // 4) * 0.3
    y = _run(x, ratio=1.0, makeup_db=_p(6.0, 2))
    assert torch.allclose(y, x * 10 ** (6.0 / 20), atol=1e-6)
    y0 = _run(x, ratio=4.0)
    y1 = _run(x, ratio=4.0, makeup_db=_p(-3.0, 2))
    assert torch.allclose(y1, y0 * 10 ** (-3.0 / 20), atol=1e-6)


def test_gradients_finite_and_nonzero():
    torch.manual_seed(0)
    t = torch.arange(SR // 2) / SR
    env = 0.05 + 0.95 * (torch.sin(2 * math.pi * 3 * t) > 0).float()  # bursts
    x = (env * torch.randn(2, SR // 2) * 0.5).requires_grad_()
    params = [_p(v, 2).requires_grad_() for v in (-25.0, 4.0, 10.0, 80.0, 2.0)]
    thr, ratio, att, rel, mk = params
    y = compress(x, thr, ratio, att, rel, SR, makeup_db=mk)
    y.square().mean().backward()
    for name, v in zip(
        ("x", "threshold", "ratio", "attack", "release", "makeup"), (x, *params), strict=True
    ):
        assert v.grad is not None, name
        assert torch.isfinite(v.grad).all(), name
        assert v.grad.abs().sum() > 0, name


def test_recursion_gradcheck():
    torch.manual_seed(0)
    g = (-torch.rand(2, 25, dtype=torch.float64) * 10).requires_grad_()
    a_att = torch.tensor([0.3, 0.5], dtype=torch.float64, requires_grad=True)
    a_rel = torch.tensor([0.9, 0.95], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_BranchingOnePole.apply, (g, a_att, a_rel))


def test_batch_independence():
    torch.manual_seed(1)
    x = torch.randn(2, SR // 4) * 0.5
    thr, ratio = torch.tensor([-30.0, -10.0]), torch.tensor([8.0, 2.0])
    att, rel = torch.tensor([2.0, 30.0]), torch.tensor([50.0, 300.0])
    y = compress(x, thr, ratio, att, rel, SR)
    for i in range(2):
        yi = compress(
            x[i : i + 1], thr[i : i + 1], ratio[i : i + 1], att[i : i + 1], rel[i : i + 1], SR
        )
        assert torch.allclose(y[i : i + 1], yi, atol=1e-6)


def test_length_not_multiple_of_hop_and_silence():
    x = torch.zeros(1, 1001)
    y = _run(x)
    assert y.shape == x.shape and torch.isfinite(y).all()


devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_device(device, dtype):
    x = (torch.randn(2, 4000) * 0.5).to(device=device, dtype=dtype)
    y = _run(x)
    assert y.dtype == dtype and y.device == x.device and torch.isfinite(y).all()


def test_speed_batch8_3s():
    x = (torch.randn(8, 3 * SR) * 0.5).requires_grad_()
    params = [_p(v, 8).requires_grad_() for v in (-20.0, 4.0, 10.0, 100.0)]
    compress(x, *params, SR).sum().backward()  # warm-up
    t0 = time.perf_counter()
    y = compress(x, *params, SR)
    t1 = time.perf_counter()
    y.square().mean().backward()
    t2 = time.perf_counter()
    print(f"\ncompress batch 8 x 3 s: forward {t1 - t0:.3f} s, backward {t2 - t1:.3f} s")
    # Target is "well under 1 s" (~0.15-0.3 s locally); 5 s is only a loose guard because
    # shared CI runners are noisy.
    assert t2 - t0 < 5.0


@pytest.mark.parametrize(
    "bad", [torch.full((2, 1), -20.0), torch.full((3,), -20.0), torch.full((1, 2), -20.0)]
)
def test_param_shape_validation(bad):
    x = torch.randn(2, 1000) * 0.5
    ok = _p(4.0, 2)
    with pytest.raises(ValueError, match="threshold_db"):
        compress(x, bad, ok, ok, ok, SR)
    with pytest.raises(ValueError, match="makeup_db"):
        compress(x, _p(-20.0, 2), ok, ok, ok, SR, makeup_db=bad)


def test_scalar_params_broadcast():
    x = torch.randn(2, 1000) * 0.5
    y = compress(
        x, torch.tensor(-20.0), torch.tensor([4.0]), torch.tensor(5.0), torch.tensor(50.0), SR
    )
    assert torch.allclose(y, _run(x, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0))


def test_zero_length_raises():
    with pytest.raises(ValueError, match="zero length"):
        _run(torch.zeros(1, 0))


def test_backward_only_needed_grads():
    x = (torch.randn(1, 4000) * 0.5).requires_grad_()
    compress(x, _p(-30.0), _p(4.0), _p(5.0), _p(50.0), SR).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
