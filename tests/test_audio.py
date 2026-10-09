import numpy as np
import pytest
import soundfile as sf

from lstmabar.audio import (
    RIFF_KINDS,
    _pluck,
    load_audio,
    loudness_match,
    peak_normalize,
    rms,
    synth_riff,
    to_mono_float,
)


def test_to_mono_float_int16_stereo_and_resample():
    sr = 22050
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([tone, tone], axis=1) * 16384
    x = to_mono_float(stereo.astype(np.int16), sr, 44100)
    assert x.dtype == np.float32
    assert x.shape == (44100,)
    assert np.max(np.abs(x)) == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize(
    "data,expected",
    [
        (np.array([-2147483648, 1073741824], np.int32), [-1.0, 0.5]),
        (np.array([0, 128, 255], np.uint8), [-1.0, 0.0, 127 / 128]),
        (np.array([0.25, -0.5], np.float64), [0.25, -0.5]),
    ],
)
def test_to_mono_float_dtypes(data, expected):
    x = to_mono_float(data, 44100, 44100)
    assert x.dtype == np.float32
    np.testing.assert_allclose(x, expected, atol=1e-6)


def test_to_mono_float_channels_first():
    data = np.stack([np.ones(1000), -np.ones(1000) * 0.5]).astype(np.float32)  # (C, T)
    x = to_mono_float(data, 44100, 44100)
    assert x.shape == (1000,)
    np.testing.assert_allclose(x, 0.25)


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
def test_load_audio_roundtrip(tmp_path, subtype):
    sr = 48000
    t = np.arange(sr * 2) / sr
    stereo = 0.5 * np.stack([np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
    path = tmp_path / "x.wav"
    sf.write(path, stereo, sr, subtype=subtype)
    x = load_audio(path, sample_rate=44100)
    assert x.dtype == np.float32 and x.shape == (88200,)
    assert np.max(np.abs(x)) == pytest.approx(0.5, abs=0.01)
    short = load_audio(path, sample_rate=44100, max_seconds=0.5)
    assert short.shape == (22050,)


def test_peak_normalize_and_rms():
    x = np.array([0.1, -0.2, 0.05], np.float32)
    y = peak_normalize(x, 0.9)
    assert np.max(np.abs(y)) == pytest.approx(0.9)
    assert np.array_equal(peak_normalize(np.zeros(4)), np.zeros(4))
    assert rms(np.ones(10) * 0.5) == pytest.approx(0.5)
    assert rms(np.zeros(0)) == 0.0


def test_loudness_match_rms_and_peak_safety():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 0.1, 1000).astype(np.float32)
    y = rng.normal(0, 0.01, 1000).astype(np.float32)
    out = loudness_match(y, ref)
    assert rms(out) == pytest.approx(rms(ref), rel=1e-4)
    # A spiky signal can't reach the target RMS without clipping: the peak limit wins.
    spiky = np.zeros(1000, np.float32)
    spiky[0] = 1.0
    out = loudness_match(spiky, np.full(1000, 0.5, np.float32), max_peak=0.99)
    assert np.max(np.abs(out)) == pytest.approx(0.99)
    assert np.array_equal(loudness_match(np.zeros(5), ref), np.zeros(5))


@pytest.mark.parametrize("kind", RIFF_KINDS)
def test_synth_riff(kind):
    x = synth_riff(kind, seconds=2.0, sample_rate=22050, seed=1)
    assert x.dtype == np.float32 and x.shape == (44100,)
    assert np.all(np.isfinite(x))
    assert np.max(np.abs(x)) == pytest.approx(0.8)
    assert rms(x) > 0.02
    np.testing.assert_array_equal(x, synth_riff(kind, seconds=2.0, sample_rate=22050, seed=1))
    assert not np.array_equal(x, synth_riff(kind, seconds=2.0, sample_rate=22050, seed=2))


def test_synth_riff_rejects_unknown_kind():
    with pytest.raises(ValueError):
        synth_riff("polka")  # type: ignore[arg-type]


def test_pluck_pitch_and_decay():
    sr = 44100
    f0 = 110.0
    y = _pluck(f0, sr, sr, np.random.default_rng(0), t60=1.0)
    # Pitch from the autocorrelation peak (the strongest partial need not be the fundamental).
    seg = y[: sr // 2]
    ac = np.correlate(seg, seg, mode="full")[len(seg) - 1 :]
    lo, hi = sr // 500, sr // 60
    lag = lo + int(np.argmax(ac[lo:hi]))
    assert sr / lag == pytest.approx(f0, rel=0.02)
    assert rms(y[-sr // 10 :]) < 0.1 * rms(y[: sr // 10])


@pytest.mark.parametrize("seconds", [0.001, 0.01, 0.05])
@pytest.mark.parametrize("kind", RIFF_KINDS)
def test_synth_riff_very_short(kind, seconds):
    x = synth_riff(kind, seconds=seconds, sample_rate=44100)
    assert x.shape == (max(round(seconds * 44100), 1),)
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize("seconds", [0.0, -1.0])
def test_synth_riff_rejects_non_positive_seconds(seconds):
    with pytest.raises(ValueError):
        synth_riff("single_notes", seconds=seconds)


def test_to_mono_float_trims_and_rejects_bad_rate():
    data = np.ones((48000 * 10, 2), np.int16)
    x = to_mono_float(data, 48000, 44100, max_seconds=1.5)
    assert x.shape == (66150,)
    for sr in (0, -1):
        with pytest.raises(ValueError, match="sample rate"):
            to_mono_float(data, sr, 44100)
