"""Decoding uploads without a system ffmpeg: libsndfile first, PyAV for m4a/AAC and friends."""

import builtins

import numpy as np
import pytest
import soundfile as sf

from lstmabar.audio import AudioDecodeError, decode_audio, load_audio

SR = 44100


def _tone(seconds: float = 1.0, freq: float = 440.0, sr: int = SR) -> np.ndarray:
    """0.5-amplitude sine with 20 ms fades (an abrupt stop makes AAC overshoot at the end)."""
    t = np.arange(int(seconds * sr)) / sr
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    fade = int(0.02 * sr)
    ramp = np.linspace(0.0, 1.0, fade)
    x[:fade] *= ramp
    x[-fade:] *= ramp[::-1]
    return x.astype(np.float32)


def _write_m4a(path, x: np.ndarray, sr: int = SR) -> None:
    av = pytest.importorskip("av")
    with av.open(str(path), "w", format="mp4") as out:
        stream = out.add_stream("aac", rate=sr, layout="mono")
        frame_size = stream.codec_context.frame_size or 1024
        for i in range(0, len(x), frame_size):
            chunk = x[i : i + frame_size]
            frame = av.AudioFrame.from_ndarray(chunk[None, :], format="flt", layout="mono")
            frame.sample_rate = sr
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def _dominant_hz(x: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spec) * sr / len(x))


def test_wav_decodes_with_soundfile(tmp_path):
    p = tmp_path / "clip.wav"
    sf.write(p, _tone(), SR)
    data, sr = decode_audio(p)
    assert sr == SR and data.shape == (SR, 1)


def test_m4a_decodes_without_system_ffmpeg(tmp_path):
    p = tmp_path / "clip.m4a"
    _write_m4a(p, _tone(1.0, 440.0))
    y = load_audio(p, sample_rate=SR)
    assert y.dtype == np.float32
    assert abs(len(y) - SR) < 0.1 * SR  # AAC adds priming/padding samples
    assert abs(_dominant_hz(y, SR) - 440.0) < 5.0
    steady = y[int(0.2 * SR) : int(0.8 * SR)]
    assert abs(float(np.sqrt(np.mean(steady**2))) - 0.5 / np.sqrt(2)) < 0.02  # level preserved
    assert np.abs(y).max() <= 1.0


def test_m4a_respects_max_seconds_and_resamples(tmp_path):
    p = tmp_path / "long.m4a"
    _write_m4a(p, _tone(3.0, 440.0))
    y = load_audio(p, sample_rate=22050, max_seconds=1.0)
    assert len(y) == 22050


def test_garbage_file_raises_user_facing_error(tmp_path):
    pytest.importorskip("av")
    p = tmp_path / "notaudio.m4a"
    p.write_bytes(b"definitely not audio" * 100)
    with pytest.raises(AudioDecodeError, match="notaudio.m4a") as info:
        load_audio(p)
    # user-facing message must not leak server-side paths (shown publicly with --share)
    assert str(tmp_path) not in str(info.value)
    assert str(tmp_path).replace("\\", "\\\\") not in str(info.value)


def test_absurd_sample_rate_rejected(tmp_path):
    p = tmp_path / "fast.wav"
    sf.write(p, np.zeros(1000, dtype=np.float32), 400_000)
    with pytest.raises(AudioDecodeError, match="sample rate") as info:
        load_audio(p)
    assert str(tmp_path) not in str(info.value)


def test_missing_pyav_gives_install_hint(tmp_path, monkeypatch):
    p = tmp_path / "clip.xyz"
    p.write_bytes(b"\x00" * 1000)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "av":
            raise ImportError("No module named 'av'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(AudioDecodeError, match="uv sync --extra demo"):
        decode_audio(p)


def test_prepare_input_accepts_file_paths(tmp_path):
    from lstmabar.demo.render import prepare_input

    p = tmp_path / "clip.wav"
    sf.write(p, _tone(2.0), SR)
    y = prepare_input(str(p), "power_chords", SR, max_seconds=1.5)
    assert len(y) == int(1.5 * SR)
