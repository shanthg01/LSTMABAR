import math

import numpy as np
import pytest
import scipy.signal
import torch

from lstmabar.dsp.filters import apply_filters, biquad, biquad_coeffs, freqz, tilt

SR = 44100
KINDS = ["lowpass", "highpass", "peak", "lowshelf", "highshelf"]
# (freq_hz, q, gain_db)
CASES = [(80.0, 0.707, 6.0), (1000.0, 2.0, -9.0), (5000.0, 0.5, 12.0), (15000.0, 4.0, -3.0)]


def ref_rbj(kind, f, q, g, sr):
    """Numpy reference of the RBJ Audio EQ Cookbook, normalized so a0 = 1."""
    w0 = 2 * np.pi * f / sr
    c, s = np.cos(w0), np.sin(w0)
    alpha = s / (2 * q)
    big_a = 10 ** (g / 40)
    sq = 2 * np.sqrt(big_a) * alpha
    if kind == "lowpass":
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]
        a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * big_a, -2 * c, 1 - alpha * big_a]
        a = [1 + alpha / big_a, -2 * c, 1 - alpha / big_a]
    elif kind == "lowshelf":
        b = [
            big_a * ((big_a + 1) - (big_a - 1) * c + sq),
            2 * big_a * ((big_a - 1) - (big_a + 1) * c),
            big_a * ((big_a + 1) - (big_a - 1) * c - sq),
        ]
        a = [
            (big_a + 1) + (big_a - 1) * c + sq,
            -2 * ((big_a - 1) + (big_a + 1) * c),
            (big_a + 1) + (big_a - 1) * c - sq,
        ]
    else:
        b = [
            big_a * ((big_a + 1) + (big_a - 1) * c + sq),
            -2 * big_a * ((big_a - 1) + (big_a + 1) * c),
            big_a * ((big_a + 1) + (big_a - 1) * c - sq),
        ]
        a = [
            (big_a + 1) - (big_a - 1) * c + sq,
            2 * ((big_a - 1) - (big_a + 1) * c),
            (big_a + 1) - (big_a - 1) * c - sq,
        ]
    b, a = np.array(b), np.array(a)
    return b / a[0], a / a[0]


def _params(dtype=torch.float64):
    f = torch.tensor([c[0] for c in CASES], dtype=dtype)
    q = torch.tensor([c[1] for c in CASES], dtype=dtype)
    g = torch.tensor([c[2] for c in CASES], dtype=dtype)
    return f, q, g


def _rms_db(y):
    return 20 * math.log10(float(y.square().mean().sqrt()))


@pytest.mark.parametrize("kind", KINDS)
def test_coeffs_match_reference(kind):
    f, q, g = _params()
    b, a = biquad_coeffs(kind, f, q, g, SR)
    assert b.shape == a.shape == (len(CASES), 3)
    assert torch.allclose(a[:, 0], torch.ones_like(a[:, 0]))
    for i, (fi, qi, gi) in enumerate(CASES):
        rb, ra = ref_rbj(kind, fi, qi, gi, SR)
        np.testing.assert_allclose(b[i].numpy(), rb, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(a[i].numpy(), ra, rtol=1e-10, atol=1e-12)


def test_coeffs_clamp_freq_and_q():
    b, a = biquad_coeffs(
        "lowpass", torch.tensor([0.0, 1e6]), torch.tensor([0.0, -1.0]), torch.zeros(2), SR
    )
    assert torch.isfinite(b).all() and torch.isfinite(a).all()
    rb, ra = ref_rbj("lowpass", 0.499 * SR, 1e-3, 0.0, SR)
    np.testing.assert_allclose(a[1].numpy(), ra, rtol=1e-4)
    with pytest.raises(ValueError):
        biquad_coeffs("bandpass", torch.ones(1), torch.ones(1), torch.ones(1), SR)


@pytest.mark.parametrize("kind", KINDS)
def test_freqz_matches_scipy(kind):
    f, q, g = _params()
    b, a = biquad_coeffs(kind, f, q, g, SR)
    n_fft = 4096
    h = freqz(b, a, n_fft)
    assert h.shape == (len(CASES), n_fft // 2 + 1)
    for i in range(len(CASES)):
        _, h_ref = scipy.signal.freqz(
            b[i].numpy(), a[i].numpy(), worN=n_fft // 2 + 1, include_nyquist=True
        )
        np.testing.assert_allclose(h[i].numpy(), h_ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_apply_filters_matches_lfilter(kind):
    torch.manual_seed(0)
    n = SR // 2
    f, q, g = _params()
    x = torch.randn(len(CASES), n, dtype=torch.float64)
    b, a = biquad_coeffs(kind, f, q, g, SR)
    y = apply_filters(x, [(b, a)])
    # From sample 0: checks causality / zero initial state, not just steady state.
    for i in range(len(CASES)):
        y_ref = scipy.signal.lfilter(b[i].numpy(), a[i].numpy(), x[i].numpy())
        np.testing.assert_allclose(y[i].numpy(), y_ref, rtol=1e-8, atol=1e-8 * np.abs(y_ref).max())


def test_single_element_params_broadcast():
    x = torch.randn(3, 2048, dtype=torch.float64)
    f = torch.tensor([700.0], dtype=torch.float64)
    y = biquad(x, "peak", f, SR, q=torch.tensor(1.2), gain_db=torch.tensor([4.0]))
    y_full = biquad(
        x, "peak", f.expand(3), SR, q=torch.full((3,), 1.2), gain_db=torch.full((3,), 4.0)
    )
    assert torch.allclose(y, y_full)
    b, a = biquad_coeffs("lowpass", torch.tensor(500.0), torch.ones(4), torch.zeros(1), SR)
    assert b.shape == a.shape == (4, 3)
    assert torch.allclose(b, b[:1].expand(4, 3))
    with pytest.raises(ValueError):
        biquad(x, "peak", torch.ones(2) * 500.0, SR)
    with pytest.raises(ValueError):
        biquad_coeffs("peak", torch.ones(3), torch.ones(2), torch.ones(3), SR)
    with pytest.raises(ValueError):
        biquad_coeffs("peak", torch.ones(2, 2), torch.ones(1), torch.ones(1), SR)


def test_float32_params_computed_in_float64():
    f32 = torch.tensor([37.3, 1234.5], dtype=torch.float32)
    q = torch.tensor([0.7, 3.0], dtype=torch.float32)
    g = torch.tensor([5.0, -5.0], dtype=torch.float32)
    b, a = biquad_coeffs("lowshelf", f32, q, g, SR)
    assert b.dtype == a.dtype == torch.float64
    b64, a64 = biquad_coeffs("lowshelf", f32.double(), q.double(), g.double(), SR)
    assert torch.equal(b, b64) and torch.equal(a, a64)
    x = torch.randn(2, 1024, dtype=torch.float64)
    assert torch.equal(biquad(x, "lowpass", f32, SR), biquad(x, "lowpass", f32.double(), SR))
    assert biquad(x.float(), "lowpass", f32, SR).dtype == torch.float32


def test_cascade_matches_sequential_lfilter_float32():
    torch.manual_seed(1)
    x = torch.randn(2, SR // 4)
    f = torch.tensor([200.0, 3000.0])
    coeffs = [
        biquad_coeffs("highpass", f / 2, torch.full((2,), 0.7), torch.zeros(2), SR),
        biquad_coeffs("peak", f, torch.full((2,), 1.5), torch.tensor([6.0, -6.0]), SR),
        biquad_coeffs("lowpass", f * 4, torch.full((2,), 0.7), torch.zeros(2), SR),
    ]
    y = apply_filters(x, coeffs)
    assert y.dtype == torch.float32
    skip = int(0.01 * SR)
    for i in range(2):
        y_ref = x[i].double().numpy()
        for b, a in coeffs:
            y_ref = scipy.signal.lfilter(b[i].double().numpy(), a[i].double().numpy(), y_ref)
        err = np.abs(y[i, skip:].double().numpy() - y_ref[skip:]).max()
        assert err <= 1e-3 * np.abs(y_ref[skip:]).max()


def _tone(freq, n=SR, dtype=torch.float64):
    t = torch.arange(n, dtype=dtype) / SR
    return torch.sin(2 * math.pi * freq * t).unsqueeze(0)


def test_lowpass_attenuates_two_octaves_above():
    cutoff = 1000.0
    x = _tone(4 * cutoff)
    y = biquad(x, "lowpass", torch.tensor([cutoff]), SR)
    sl = slice(SR // 10, SR - SR // 10)
    assert _rms_db(y[:, sl]) - _rms_db(x[:, sl]) < -20.0


def test_peak_gain_at_center():
    sr = 48000
    center = 1000.0  # 48 samples per period
    n = sr
    t = torch.arange(n, dtype=torch.float64) / sr
    x = torch.sin(2 * math.pi * center * t).unsqueeze(0)
    b, a = biquad_coeffs(
        "peak", torch.tensor([center]), torch.tensor([1.0]), torch.tensor([6.0]), sr
    )
    y = apply_filters(x, [(b, a)])
    sl = slice(4800, 4800 + 48 * 800)
    assert _rms_db(y[:, sl]) - _rms_db(x[:, sl]) == pytest.approx(6.0, abs=0.1)


def test_tilt_zero_is_identity():
    torch.manual_seed(2)
    x = torch.randn(3, 8192)
    y = tilt(x, torch.zeros(3), SR)
    assert torch.allclose(y, x, atol=1e-5)


def test_tilt_positive_brightens():
    lo, hi = _tone(100.0), _tone(10000.0)
    sl = slice(SR // 10, SR - SR // 10)
    up = torch.tensor([6.0])
    d_lo = _rms_db(tilt(lo, up, SR)[:, sl]) - _rms_db(lo[:, sl])
    d_hi = _rms_db(tilt(hi, up, SR)[:, sl]) - _rms_db(hi[:, sl])
    assert d_hi > 2.5 and d_lo < -2.5
    d_lo_neg = _rms_db(tilt(lo, -up, SR)[:, sl]) - _rms_db(lo[:, sl])
    assert d_lo_neg > 2.5


@pytest.mark.parametrize("kind", KINDS)
def test_gradients_finite_nonzero(kind):
    torch.manual_seed(3)
    x = torch.randn(2, 4096, requires_grad=True)
    f = torch.tensor([500.0, 3000.0], requires_grad=True)
    q = torch.tensor([0.8, 2.0], requires_grad=True)
    g = torch.tensor([4.0, -4.0], requires_grad=True)
    y = apply_filters(x, [biquad_coeffs(kind, f, q, g, SR)])
    y.square().mean().backward()
    params = [x, f, q] if kind in ("lowpass", "highpass") else [x, f, q, g]
    for p in params:
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", KINDS)
def test_gradcheck(kind):
    torch.manual_seed(4)
    sr = 8000
    x = torch.randn(2, 64, dtype=torch.float64, requires_grad=True)
    f = torch.tensor([600.0, 1500.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.9, 1.3], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([3.0, -5.0], dtype=torch.float64, requires_grad=True)

    def fn(x, f, q, g):
        return apply_filters(x, [biquad_coeffs(kind, f, q, g, sr)])

    assert torch.autograd.gradcheck(fn, (x, f, q, g))


def test_gradcheck_cascade_tilt():
    torch.manual_seed(6)
    x = torch.randn(2, 64, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([4.0, -3.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, t: tilt(x, t, 8000, pivot_hz=800.0), (x, t))


def test_batch_independence():
    torch.manual_seed(5)
    x = torch.randn(3, 4096, dtype=torch.float64)
    f = torch.tensor([300.0, 2000.0, 8000.0], dtype=torch.float64)
    q = torch.tensor([0.5, 1.0, 3.0], dtype=torch.float64)
    g = torch.tensor([-6.0, 3.0, 9.0], dtype=torch.float64)

    def run(sl):
        return apply_filters(
            x[sl],
            [
                biquad_coeffs("peak", f[sl], q[sl], g[sl], SR),
                biquad_coeffs("highshelf", f[sl] * 2, q[sl], -g[sl], SR),
            ],
        )

    y = run(slice(None))
    for i in range(3):
        assert torch.allclose(y[i : i + 1], run(slice(i, i + 1)), atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shape_dtype_preserved(dtype):
    x = torch.randn(2, 1000, dtype=dtype)
    y = biquad(x, "highpass", torch.tensor([100.0, 200.0]), SR)
    assert y.shape == x.shape and y.dtype == dtype
    assert tilt(x, torch.tensor(3.0), SR).shape == x.shape
    assert apply_filters(x, []) is x


def test_shape_mismatch_raises():
    x = torch.randn(2, 100)
    b, a = biquad_coeffs("lowpass", torch.tensor([1000.0] * 3), torch.ones(3), torch.zeros(3), SR)
    with pytest.raises(ValueError):
        apply_filters(x, [(b, a)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device_preserved():
    x = torch.randn(2, 4096, device="cuda")
    f = torch.tensor([500.0, 2000.0], device="cuda")
    y = biquad(x, "peak", f, SR, gain_db=torch.tensor([3.0, -3.0], device="cuda"))
    assert y.device == x.device and y.dtype == x.dtype
    y_cpu = biquad(x.cpu(), "peak", f.cpu(), SR, gain_db=torch.tensor([3.0, -3.0]))
    assert torch.allclose(y.cpu(), y_cpu, atol=1e-4)
