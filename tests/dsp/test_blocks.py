import pytest
import torch

from lstmabar.dsp.blocks import EQ3, Compressor, Drive, Gain
from lstmabar.dsp.oversample import FACTORS
from tests.dsp.signals import (
    SR,
    band_energy,
    burst,
    centroid,
    crest_factor,
    harmonic,
    harmonic_ratio,
    riff,
    rms_db,
    tone,
)

ALL_BLOCKS = [Gain, Drive, EQ3, Compressor]


def knob(block, name: str, value: float, batch: int = 1) -> torch.Tensor:
    """Normalized ``(batch,)`` tensor for a physical ``value`` of ``block``'s knob."""
    spec = next(s for s in block.param_specs if s.name == name)
    return spec.normalize(torch.full((batch,), float(value), dtype=torch.float64)).float()


def sweep(block, x, name, values, **fixed):
    """Render ``x`` (1, T) once per physical value of ``name`` (batched), other knobs fixed."""
    n = len(values)
    xb = x.expand(n, -1)
    params = {k: knob(block, k, v, n) for k, v in fixed.items()}
    spec = next(s for s in block.param_specs if s.name == name)
    params[name] = spec.normalize(torch.tensor(values, dtype=torch.float64)).float()
    return block(xb, params)


def gain_at(block, freq, name, values, **fixed):
    x = tone(freq, 0.5, 0.01)
    y = sweep(block, x, name, values, **fixed)
    c = 4096  # skip filter onset transients
    return rms_db(y[:, c:-c]) - rms_db(x[:, c:-c])


# -- contracts ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_BLOCKS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_defaults_render_finite_audio_with_shape_and_dtype(cls, dtype):
    block = cls(SR)
    x = riff(2, 0.5).to(dtype)
    y = block(x)
    assert y.shape == x.shape and y.dtype == dtype
    assert torch.isfinite(y).all()
    assert y.abs().max() > 1e-3


@pytest.mark.parametrize("cls", ALL_BLOCKS)
def test_specs_have_units_and_sane_defaults(cls):
    for s in cls.param_specs:
        assert s.min <= s.default <= s.max
        if s.name.endswith("_db"):
            assert s.unit == "dB"
        if s.name.endswith("_hz"):
            assert s.unit == "Hz" and s.taper == "log"
            assert 1.0 < s.min and s.max < 0.499 * SR  # inside the biquad clamps
        if s.name.endswith("_ms"):
            assert s.unit == "ms"


def test_compressor_ratio_and_drive_bias_inside_low_level_ranges():
    ratio = next(s for s in Compressor.param_specs if s.name == "ratio")
    assert ratio.min >= 1.0  # compress() clamps ratio to >= 1
    bias = next(s for s in Drive.param_specs if s.name == "bias")
    assert -0.25 <= bias.min and bias.max <= 0.25  # waveshape's musically useful range


@pytest.mark.parametrize("cls", ALL_BLOCKS)
def test_batch_independence(cls):
    block = cls(SR)
    x = riff(3, 0.5)
    g = torch.Generator().manual_seed(0)
    params = {n: torch.rand(3, generator=g) for n in block.param_names}
    y = block(x, params)
    for i in range(3):
        yi = block(x[i : i + 1], {n: v[i : i + 1] for n, v in params.items()})
        torch.testing.assert_close(y[i : i + 1], yi, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("cls", ALL_BLOCKS)
def test_every_knob_gets_finite_nonzero_gradient(cls):
    block = cls(SR)
    x = riff(2, 0.5)
    g = torch.Generator().manual_seed(1)
    # typical interior settings (not range edges, where the [0, 1] clamp zeroes the gradient)
    params = {
        n: (0.2 + 0.6 * torch.rand(2, generator=g)).requires_grad_() for n in block.param_names
    }
    target = riff(2, 0.5) * 0.3
    loss = (block(x, params) - target).pow(2).mean()
    loss.backward()
    for n, p in params.items():
        assert torch.isfinite(p.grad).all(), n
        assert (p.grad.abs() > 0).all(), n


def test_drive_rejects_bad_oversample_factor():
    assert Drive(SR, oversample=8).oversample == 8
    with pytest.raises(ValueError):
        Drive(SR, oversample=3)
    assert 3 not in FACTORS


# -- Gain ---------------------------------------------------------------------------------


def test_gain_scales_by_db():
    x = riff(1, 0.3)
    y = Gain(SR)(x, {"gain_db": knob(Gain, "gain_db", 6.0)})
    torch.testing.assert_close(y, x * 10 ** (6 / 20))


# -- Drive --------------------------------------------------------------------------------

_CLEAN_DRIVE = dict(pre_hpf_hz=20.0, softness=0.3, asymmetry=0.0, bias=0.0, tone_db=0.0)


def test_drive_gain_adds_harmonics_and_lowers_crest_factor():
    d = Drive(SR)
    x = tone(220.0, 0.5, 0.3)
    y = sweep(d, x, "gain_db", [0.0, 15.0, 30.0, 45.0], **_CLEAN_DRIVE)
    hr = harmonic_ratio(y, 220.0)
    assert (hr.diff() > 0).all(), hr
    yr = sweep(d, riff(1, 0.5), "gain_db", [0.0, 20.0, 40.0], **_CLEAN_DRIVE)
    cf = crest_factor(yr)
    assert (cf.diff() < 0).all(), cf


def test_drive_symmetric_is_odd_only_and_asymmetry_adds_second_harmonic():
    d = Drive(SR)
    x = tone(220.0, 0.5, 0.3)
    y = sweep(d, x, "asymmetry", [0.0, 0.5, 1.0], **{**_CLEAN_DRIVE, "gain_db": 20.0})
    h2 = harmonic(y, 220.0, 2) / harmonic(y, 220.0, 1)
    h3 = harmonic(y, 220.0, 3) / harmonic(y, 220.0, 1)
    assert h2[0] < 1e-3 < h3[0]  # odd-symmetric: only odd harmonics
    assert (h2.diff() > 0).all(), h2
    assert h2[2] > 0.05


def test_drive_bias_adds_even_harmonics():
    d = Drive(SR)
    x = tone(220.0, 0.5, 0.3)
    fixed = {k: v for k, v in _CLEAN_DRIVE.items() if k != "bias"}
    y = sweep(d, x, "bias", [0.0, 0.2], **fixed, gain_db=20.0)
    h2 = harmonic(y, 220.0, 2) / harmonic(y, 220.0, 1)
    assert h2[0] < 1e-3 and h2[1] > 0.02, h2


def test_drive_softness_makes_clipping_harder():
    d = Drive(SR)
    x = tone(220.0, 0.5, 0.3)
    fixed = {k: v for k, v in _CLEAN_DRIVE.items() if k != "softness"}
    y = sweep(d, x, "softness", [0.0, 0.5, 1.0], **fixed, gain_db=24.0)
    upper = sum(harmonic(y, 220.0, k) ** 2 for k in (9, 11, 13, 15)) / harmonic(y, 220.0, 1) ** 2
    assert (upper.diff() > 0).all(), upper


def test_drive_tone_raises_spectral_centroid():
    d = Drive(SR)
    fixed = {k: v for k, v in _CLEAN_DRIVE.items() if k != "tone_db"}
    y = sweep(d, riff(1, 0.5), "tone_db", [-12.0, 0.0, 12.0], **fixed, gain_db=20.0)
    c = centroid(y)
    assert (c.diff() > 0).all(), c


def test_drive_pre_hpf_removes_low_end_before_clipping():
    d = Drive(SR)
    fixed = {k: v for k, v in _CLEAN_DRIVE.items() if k != "pre_hpf_hz"}
    y = sweep(d, tone(110.0, 0.5, 0.05), "pre_hpf_hz", [20.0, 300.0, 1500.0], **fixed, gain_db=0.0)
    e = rms_db(y[:, 4096:-4096])
    assert (e.diff() < 0).all(), e
    assert e[0] - e[2] > 20.0


def test_drive_level_is_output_gain():
    d = Drive(SR)
    fixed = {**_CLEAN_DRIVE, "gain_db": 20.0}
    y = sweep(d, riff(1, 0.5), "level_db", [-24.0, -12.0], **fixed)
    torch.testing.assert_close(y[1], y[0] * 10 ** (12 / 20), atol=1e-6, rtol=1e-4)


def test_drive_output_bounded_by_level():
    d = Drive(SR)
    y = sweep(d, riff(1, 0.5) * 2, "gain_db", [60.0], **_CLEAN_DRIVE, level_db=0.0)
    # The shaper is bounded to ±1; band-limiting a near-square wave (decimation filter, Gibbs)
    # overshoots, so the bound after downsampling is looser but still finite.
    assert y.abs().max() < 2.0


# -- EQ3 ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "freq"),
    [("low_db", 50.0), ("mid_db", 800.0), ("high_db", 12000.0)],
)
def test_eq_band_gain_lands_in_its_band(name, freq):
    eq = EQ3(SR)
    g = gain_at(eq, freq, name, [-12.0, 0.0, 12.0])
    torch.testing.assert_close(g, torch.tensor([-12.0, 0.0, 12.0]), atol=1.0, rtol=0)


def test_eq_bands_are_local():
    eq = EQ3(SR)
    assert gain_at(eq, 8000.0, "low_db", [12.0]).abs().item() < 0.5
    assert gain_at(eq, 40.0, "high_db", [12.0]).abs().item() < 0.5
    assert gain_at(eq, 6000.0, "mid_db", [12.0], mid_hz=400.0).abs().item() < 1.0


def test_eq_mid_hz_moves_the_peak():
    eq = EQ3(SR)
    g = gain_at(eq, 2000.0, "mid_hz", [400.0, 2000.0], mid_db=12.0)
    assert g[1] == pytest.approx(12.0, abs=0.5)
    assert g[0] < 3.0


def test_eq_defaults_are_identity():
    x = riff(1, 0.5)
    torch.testing.assert_close(EQ3(SR)(x), x, atol=1e-5, rtol=0)


# -- Compressor -----------------------------------------------------------------------------


def _loud_quiet_db(y):
    seg = int(0.1 * SR)
    loud = torch.cat(
        [y[:, s + seg // 2 : s + seg] for s in range(0, y.shape[-1] - seg, 2 * seg)], -1
    )
    quiet = torch.cat(
        [y[:, s + seg + seg // 2 : s + 2 * seg] for s in range(0, y.shape[-1] - 2 * seg, 2 * seg)],
        -1,
    )
    return rms_db(loud) - rms_db(quiet)


def test_compressor_reduces_dynamics_more_with_lower_threshold_and_higher_ratio():
    c = Compressor(SR)
    x = burst(1.0)
    base = _loud_quiet_db(x)
    by_thr = _loud_quiet_db(sweep(c, x, "threshold_db", [0.0, -20.0, -40.0], ratio=8.0))
    assert by_thr[0] == pytest.approx(base.item(), abs=0.5)
    assert (by_thr.diff() < 0).all(), by_thr
    by_ratio = _loud_quiet_db(sweep(c, x, "ratio", [1.0, 4.0, 20.0], threshold_db=-40.0))
    assert by_ratio[0] == pytest.approx(base.item(), abs=0.5)
    assert (by_ratio.diff() < 0).all(), by_ratio


def test_compressor_makeup_and_time_constants():
    c = Compressor(SR)
    x = burst(1.0)
    fixed = dict(threshold_db=-30.0, ratio=8.0)
    y = sweep(c, x, "makeup_db", [0.0, 12.0], **fixed)
    torch.testing.assert_close(y[1], y[0] * 10 ** (12 / 20), atol=1e-6, rtol=1e-4)
    # slower attack lets more of each loud onset through
    seg = int(0.1 * SR)
    ya = sweep(c, x, "attack_ms", [0.5, 50.0], **fixed, release_ms=50.0)
    onset = ya[:, 2 * seg : 2 * seg + int(0.02 * SR)]
    assert onset.abs().amax(-1).diff().item() > 0
    # longer release keeps the quiet segment after a loud one attenuated for longer
    yr = sweep(c, x, "release_ms", [10.0, 1000.0], **fixed, attack_ms=1.0)
    after = band_energy(yr[:, seg : seg + int(0.05 * SR)], 900.0, 1100.0)
    assert after.diff().item() < 0
