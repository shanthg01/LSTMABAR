"""Archetype panel helpers (P3.5): formatting, no-pitch / missing-librosa paths, excerpting.

pYIN is replaced by fakes so these stay fast and don't need librosa.
"""

import numpy as np
import pytest

from lstmabar.analysis.harmonics import HarmonicProfile
from lstmabar.demo.archetype_panel import (
    COMPONENTS,
    NO_LIBROSA,
    NO_PITCH,
    ArchetypePanel,
    analyse_signals,
    archetype_markdown,
    archetype_rows,
    harmonic_rows,
    loudest_window,
)

SR = 8000
F0 = 200.0


def _tone(kind: str, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    k = np.arange(1, 11)
    amps = {"sine": (k == 1) * 1.0, "square": (k % 2 == 1) / k}[kind]
    return 0.5 * np.sum(amps[:, None] * np.sin(2 * np.pi * F0 * k[:, None] * t), axis=0)


def _fake_track(x, sr):
    return "track"


def _profile_at_f0(seg, sr, n_harmonics, track, max_frames):
    """Stand-in for pYIN + harmonic_profile: fit at the known f0 (no librosa)."""
    from lstmabar.analysis.harmonics import harmonic_profile

    assert track == "track"
    return harmonic_profile(seg, sr, n_harmonics=n_harmonics, f0=F0, max_frames=max_frames)


def test_panel_dry_vs_wet_shows_the_archetype_shift():
    panel = analyse_signals(
        {"dry": _tone("sine"), "grey-box": _tone("square")},
        SR,
        track_fn=_fake_track,
        profile_fn=_profile_at_f0,
    )
    assert panel.message == ""
    dry, wet = panel.signals
    assert dry.readout.dominant == "sine" and wet.readout.dominant == "square"
    assert dry.dbc.shape == (10,) and dry.dbc[0] == 0.0
    assert wet.dbc[2] == pytest.approx(20 * np.log10(1 / 3), abs=0.5)  # H3 of a square

    md = archetype_markdown(panel)
    assert "| | dry | grey-box |" in md
    for comp in COMPONENTS:
        assert f"| {comp} |" in md
    assert "| **dominant** | sine | square |" in md
    assert "median f0 200 Hz" in md

    rows = archetype_rows(panel)
    assert {r["signal"] for r in rows} == {"dry", "grey-box"}
    for label in ("dry", "grey-box"):
        shares = [r["share"] for r in rows if r["signal"] == label]
        assert len(shares) == len(COMPONENTS) and sum(shares) == pytest.approx(1.0)
    harms = harmonic_rows(panel)
    assert [r["harmonic"] for r in harms if r["signal"] == "dry"] == list(range(1, 11))


def test_panel_no_pitch_is_a_friendly_message():
    def no_voiced(seg, sr, **kw):
        raise ValueError("no voiced frames to analyse")

    panel = analyse_signals(
        {"dry": np.zeros(SR), "wet": np.zeros(SR)},
        SR,
        track_fn=_fake_track,
        profile_fn=no_voiced,
    )
    assert panel.signals == [] and panel.message == NO_PITCH
    assert archetype_markdown(panel) == NO_PITCH
    assert archetype_rows(panel) == [] and harmonic_rows(panel) == []


def test_panel_marks_a_single_unpitched_signal():
    def only_dry(seg, sr, **kw):
        if np.max(np.abs(seg)) < 1e-6:
            raise ValueError("all voiced frames are below -50 dBFS RMS")
        return _profile_at_f0(seg, sr, **kw)

    panel = analyse_signals(
        {"dry": _tone("sine"), "white-box": np.zeros(SR)},
        SR,
        track_fn=_fake_track,
        profile_fn=only_dry,
    )
    assert panel.signals[1].readout is None
    md = archetype_markdown(panel)
    assert "no stable pitch" in md and "| **dominant** | sine | – |" in md
    assert {r["signal"] for r in archetype_rows(panel)} == {"dry"}


def test_panel_without_librosa_explains_the_extra():
    def missing(x, sr):
        raise ImportError("f0 tracking needs librosa")

    panel = analyse_signals({"dry": _tone("sine")}, SR, track_fn=missing)
    assert panel.message == NO_LIBROSA and "analysis" in archetype_markdown(panel)


def test_panel_unexpected_error_is_generic():
    def boom(x, sr):
        raise RuntimeError(r"C:\secret\path")

    panel = analyse_signals({"dry": _tone("sine")}, SR, track_fn=boom)
    assert "secret" not in panel.message and panel.signals == []


def test_panel_uses_a_bounded_common_excerpt():
    seen = []

    def spy(seg, sr, **kw):
        seen.append(len(seg))
        return _profile_at_f0(seg, sr, **kw)

    long = _tone("sine", seconds=6.0)
    panel = analyse_signals(
        {"dry": long, "wet": long, "white-box": long[: 4 * SR]},
        SR,
        seconds=3.0,
        track_fn=_fake_track,
        profile_fn=spy,
    )
    assert seen == [3 * SR] * 3 and panel.seconds == pytest.approx(3.0)


def test_loudest_window_finds_the_loud_part():
    x = np.zeros(10 * SR)
    x[6 * SR : 8 * SR] = 1.0
    w = loudest_window(x, SR, 3.0)
    assert w.stop - w.start == 3 * SR and w.start <= 6 * SR and w.stop >= 8 * SR
    assert loudest_window(x[:SR], SR, 3.0) == slice(0, SR)


def test_empty_panel_markdown():
    assert archetype_markdown(ArchetypePanel()) == NO_PITCH


def test_real_profile_type_is_accepted():
    """The fake profile function returns the real dataclass the panel expects."""
    prof = _profile_at_f0(_tone("sine"), SR, 10, "track", 16)
    assert isinstance(prof, HarmonicProfile)
