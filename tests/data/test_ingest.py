"""P4 task 4.1: Guitar-TECHS DI ingest (onsets, clip cutting, levels, splits, remote zips)."""

import argparse
import http.client
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lstmabar.data import ingest as ig
from lstmabar.data.fixtures import FIXTURE_PERFORMERS, fixture_config
from lstmabar.data.layout import DataLayout
from lstmabar.data.levels import peak_dbfs, read_audio, sha256_file
from lstmabar.data.schema import DryClip, check_relpath, read_jsonl

SR = 44100


def _plucks(onsets_s, seconds, sr=SR, freq=220.0, decay=0.15, amp=0.5, seed=0):
    """Decaying sine bursts with a little noise floor at the given onset times."""
    rng = np.random.default_rng(seed)
    x = 1e-5 * rng.standard_normal(int(seconds * sr))
    t = np.arange(int(1.0 * sr)) / sr
    note = amp * np.sin(2 * np.pi * freq * t) * np.exp(-t / decay)
    for o in onsets_s:
        s = int(o * sr)
        seg = note[: len(x) - s]
        x[s : s + len(seg)] += seg
    return x.astype(np.float32)


def _member(**kw):
    base = dict(
        archive="P1_chords.zip",
        name="P1_chords/audio/directinput/directinput_Set1_m7.wav",
        performer="P1",
        content="chords",
        take="Set1_m7",
        file_size=0,
        compress_size=0,
    )
    base.update(kw)
    return ig.DIMember(**base)


# --- onsets ---------------------------------------------------------------------------------


def test_detect_onsets_finds_synthetic_plucks():
    truth = [0.5, 1.3, 2.0, 2.9, 4.1, 5.0]
    x = _plucks(truth, 6.0)
    found = ig.detect_onsets(x, SR)
    assert len(found) == len(truth)
    # Within two hops (~23 ms) of the true onset.
    assert np.max(np.abs(found - np.asarray(truth))) < 2 * 512 / SR


def test_detect_onsets_level_independent_and_silent():
    truth = [0.4, 1.4, 2.2]
    loud = ig.detect_onsets(_plucks(truth, 3.0), SR)
    quiet = ig.detect_onsets(_plucks(truth, 3.0) * 1e-2, SR)
    np.testing.assert_allclose(loud, quiet)
    assert ig.detect_onsets(np.zeros(SR), SR).size == 0


# --- clip cutting ---------------------------------------------------------------------------


def test_cut_clips_non_overlapping_pre_onset_and_skips_silence():
    # Plucks at 0.5 / 1.0 (inside the first clip) / 5.0, then a near-silent stretch with an
    # onset at 9.0 whose clip is too quiet, then one at 13.0 that does not fit.
    x = _plucks([0.5, 1.0, 5.0], 14.0)
    x[int(9.0 * SR) : int(9.0 * SR) + 100] += 1e-4  # tiny click
    onsets = [0.5, 1.0, 5.0, 9.0, 13.0]
    starts = ig.cut_clips(x, SR, onsets, 3.0, 0.02, -45.0)
    pre = round(0.02 * SR)
    assert starts == [round(0.5 * SR) - pre, round(5.0 * SR) - pre]
    n = 3 * SR
    assert all(b >= a + n for a, b in zip(starts, starts[1:], strict=False))


def test_cut_clips_skipped_onset_does_not_block_next():
    # A faint pluck at 0.3 s (its clip [0.1, 3.1] is too quiet) and a loud one at 3.2 s whose
    # clip [3.0, 6.0] overlaps it: the skipped clip must not reserve its time span.
    x = _plucks([0.3], 8.0, amp=1e-4) + _plucks([3.2], 8.0, seed=1)
    starts = ig.cut_clips(x, SR, [0.3, 3.2], 3.0, 0.2, -45.0)
    assert starts == [round(3.0 * SR)]


def test_di_to_mono_drops_silent_channel_and_averages_equal_ones():
    a = np.sin(np.linspace(0, 100, 1000)).astype(np.float32)
    np.testing.assert_allclose(ig.di_to_mono(np.stack([a, a], 1)), a, atol=1e-7)
    np.testing.assert_allclose(ig.di_to_mono(np.stack([a, 1e-6 * a], 1)), a, atol=1e-7)
    np.testing.assert_allclose(ig.di_to_mono(a), a)


def test_clips_from_file_levels_relpaths_metadata(tmp_path):
    cfg = fixture_config(tmp_path)
    layout = DataLayout.from_config(cfg)
    src = tmp_path / "src.wav"
    x48 = 0.1 * _plucks([0.5, 4.0, 8.0], 12.0, sr=48000)
    sf.write(src, np.stack([x48, x48], 1), 48000, subtype="PCM_24")
    m = _member(performer="fx1", take="Set1_m7")
    clips = ig.clips_from_file(src, m, layout, cfg)
    assert len(clips) == 3
    for c in clips:
        assert isinstance(c, DryClip)
        y, sr = read_audio(layout.resolve_rendered(c.path))
        assert sr == SR and len(y) == 3 * SR
        assert peak_dbfs(y) == pytest.approx(cfg.ingest.dry_peak_dbfs, abs=0.01)
        assert c.source_gain_db > 0  # the source was quiet
        assert c.path == f"v1/dry/{c.clip_id}.flac"
        assert c.source_file == "guitar_techs/P1_chords/audio/directinput/directinput_Set1_m7.wav"
        check_relpath(c.path, "path")
        assert c.sha256 == sha256_file(layout.resolve_rendered(c.path))
        assert c.license == "CC BY 4.0" and c.technique == "chords"
        assert c.recording_id == "fx1_chords_Set1_m7"
    assert clips[0].offset_s == pytest.approx(0.48, abs=0.03)
    assert clips[0].clip_id == f"gt-fx1-chords-set1_m7-{round(clips[0].offset_s * 1000):07d}"
    assert len({c.clip_id for c in clips}) == 3


# --- naming ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "P2_techniques/audio/directinput/directinput_PalmMute.wav",
            {"performer": "P2", "content": "techniques", "take": "PalmMute"},
        ),
        (
            "P3_music/audio/directinput/directinput_07.wav",
            {"performer": "P3", "content": "music", "take": "07"},
        ),
        ("P1_chords/audio/micamp/micamp_Set1_m7.wav", None),
        ("__MACOSX/P1_chords/audio/directinput/._directinput_Set1_m7.wav", None),
        ("P1_chords/audio/.DS_Store", None),
        ("P1_chords/video/ego/ego_Set1_m7.mp3", None),
    ],
)
def test_parse_di_member(name, expected):
    assert ig.parse_di_member(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "P1_chords/audio/directinput/directinput_../../../evil.wav",
        "P1_chords/audio/directinput/directinput_..\\..\\evil.wav",
        "P1_chords\\audio\\directinput\\directinput_x.wav",
        "/P1_chords/audio/directinput/directinput_x.wav",
        "C:/P1_chords/audio/directinput/directinput_x.wav",
        "P1_chords/audio/directinput/directinput_.hidden.wav",
        "../P1_chords/audio/directinput/directinput_x.wav",
    ],
)
def test_parse_di_member_rejects_unsafe_names(name):
    assert ig.parse_di_member(name) is None


def test_extract_member_refuses_escape(tmp_path):
    zpath = tmp_path / "z.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("P1_x/audio/directinput/directinput_a.wav", b"abc")
    zf = zipfile.ZipFile(zpath)
    m = _member(name="P1_x/audio/directinput/directinput_a.wav", file_size=3)
    with pytest.raises(ValueError, match="outside"):
        ig.extract_member(zf, m, tmp_path / "elsewhere" / "a.wav", root=tmp_path / "root")
    assert ig.extract_member(zf, m, tmp_path / "root" / "a.wav", root=tmp_path / "root")
    zf.close()


def test_http_fetcher_retries_incomplete_read(monkeypatch):
    calls = []

    class Resp:
        status = 206

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            if len(calls) == 1:
                raise http.client.IncompleteRead(b"ab", 2)
            return b"abcd"

    def fake_urlopen(req, timeout=None):
        calls.append(req.get_header("Range"))
        return Resp()

    monkeypatch.setattr(ig.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ig.time, "sleep", lambda s: None)
    assert ig.http_fetcher("https://zenodo.org/x")(0, 3) == b"abcd"
    assert calls == ["bytes=0-3", "bytes=0-3"]


@pytest.mark.parametrize(("value", "expected"), [("7", 7.0), (None, 2.0), ("Wed, 21 Oct", 2.0)])
def test_retry_after_parsing(value, expected):
    assert ig._retry_after(value, 2.0) == expected


def test_member_technique():
    assert _member(content="techniques", take="PalmMute").technique == "palmmute"
    assert _member(content="scales", take="Bb").technique == "scales"


# --- splits ---------------------------------------------------------------------------------


def _clip(performer, rec, i=0):
    return DryClip(
        clip_id=f"gt-{performer}-{rec}-{i}",
        source="guitar_techs",
        performer=performer,
        recording_id=rec,
        technique="x",
        source_file="guitar_techs/a.wav",
        offset_s=0.0,
        duration_s=3.0,
        sample_rate=SR,
        source_gain_db=0.0,
        path=f"v1/dry/{performer}-{rec}-{i}.flac",
    )


def test_assign_splits_over_list(tmp_path):
    cfg = fixture_config(tmp_path)
    clips = [_clip("fx1", r, i) for r in ("a", "b", "c") for i in range(3)] + [_clip("fx2", "b")]
    clips += [_clip("fx3", r, i) for r in ("r1", "r2") for i in range(2)]
    out = ig.assign_and_check_splits(clips, cfg)
    assert {c.split for c in out if c.performer == "fx3"} == {"test"}
    fx1 = [c for c in out if c.performer == "fx1"]
    assert {c.split for c in fx1} == {"train", "val"}
    # A recording never spans splits.
    for r in ("a", "b", "c"):
        assert len({c.split for c in fx1 if c.recording_id == r}) == 1
    assert [c.split for c in out if c.performer == "fx2"] == ["train"]  # single recording


def test_assign_splits_warns_empty_val(tmp_path, capsys):
    cfg = fixture_config(tmp_path)
    ig.assign_and_check_splits([_clip("fx1", "a"), _clip("fx3", "only")], cfg)
    assert "warning" in capsys.readouterr().err


def test_config_performers_are_guitar_techs():
    from lstmabar.config import load_config
    from lstmabar.data.fixtures import REPO_ROOT

    cfg = load_config(REPO_ROOT / "configs" / "data" / "v1.yaml")
    assert list(cfg.ingest.train_performers) == ["P1", "P2"]
    assert cfg.ingest.heldout_performer == "P3"


# --- remote zip over range requests ---------------------------------------------------------


def _counting(fetch):
    calls = []

    def f(a, b):
        calls.append((a, b))
        return fetch(a, b)

    return f, calls


def test_range_zip_lists_and_extracts_without_reading_everything(tmp_path):
    zpath = tmp_path / "remote.zip"
    big = np.random.default_rng(0).bytes(3 << 20)  # incompressible, read only if extracted
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("P9_chords/video/ego/ego_x.mp3", big)
        buf = io.BytesIO()
        sf.write(buf, np.zeros((4800, 2), np.float32), 48000, format="WAV", subtype="PCM_24")
        zf.writestr("P9_chords/audio/directinput/directinput_x.wav", buf.getvalue())
        zf.writestr("P9_chords/audio/micamp/micamp_x.wav", buf.getvalue())
    size = zpath.stat().st_size
    fetch, calls = _counting(ig.file_fetcher(zpath))

    zf = ig.open_remote_zip(fetch, size)
    members = ig.di_members("remote.zip", zf)
    assert [m.name for m in members] == ["P9_chords/audio/directinput/directinput_x.wav"]
    assert sum(b - a + 1 for a, b in calls) < 256 << 10  # central directory only

    dest = tmp_path / "out" / members[0].name
    assert ig.extract_member(zf, members[0], dest) is True
    data, sr = sf.read(dest)
    assert sr == 48000 and data.shape == (4800, 2)
    assert sum(b - a + 1 for a, b in calls) < 1 << 20  # never streamed the 3 MB video
    assert ig.extract_member(zf, members[0], dest) is False  # resume: already extracted

    with zf.open(members[0].name) as f:
        assert ig.wav_header_info(f.read(4096)) == (2, 48000, 24, 4800 * 2 * 3)


def test_range_file_seek_and_read(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(bytes(range(256)) * 4)
    rf = ig.RangeFile(ig.file_fetcher(p), 1024)
    rf.seek(-10, io.SEEK_END)
    assert rf.read(100) == bytes(range(246, 256))
    rf.seek(5)
    assert rf.read(3) == bytes([5, 6, 7]) and rf.tell() == 8


# --- end to end (offline fixture source) ----------------------------------------------------


def test_fixture_ingest_end_to_end(tmp_path, capsys):
    cfg = fixture_config(tmp_path / "unused")
    cfg.data.cache_dir = str(tmp_path / "cache")
    args = argparse.Namespace(fixtures=True, list=False, limit=None, overrides=[])
    assert ig.run(cfg, args) == 0
    root = tmp_path / "cache" / "fixtures"
    layout = DataLayout.from_config(fixture_config(root))
    clips = list(read_jsonl(layout.dry_manifest, DryClip))
    assert clips
    assert {c.performer for c in clips} == set(FIXTURE_PERFORMERS)
    assert {c.split for c in clips if c.performer != FIXTURE_PERFORMERS[2]} == {"train", "val"}
    assert {c.split for c in clips if c.performer == FIXTURE_PERFORMERS[2]} == {"test"}
    for c in clips:
        assert Path(layout.resolve_rendered(c.path)).exists()
        assert (layout.sources / c.source_file).exists()
    # Only DI was extracted.
    extracted = [p for p in (layout.sources / "guitar_techs").rglob("*") if p.is_file()]
    assert extracted and all("/directinput/" in p.as_posix() for p in extracted)

    # Re-run is resumable and deterministic; --limit caps the clip count.
    assert ig.run(cfg, args) == 0
    again = list(read_jsonl(layout.dry_manifest, DryClip))
    assert again == clips
    args.limit = 2
    assert ig.run(cfg, args) == 0
    assert len(list(read_jsonl(ig.manifest_path(layout, 2), DryClip))) == 2
    assert list(read_jsonl(layout.dry_manifest, DryClip)) == clips  # full manifest untouched

    args.list, args.limit = True, None
    capsys.readouterr()
    assert ig.run(cfg, args) == 0
    assert "DI takes" in capsys.readouterr().out


def test_cli_parses_list_flag():
    from lstmabar.cli import build_parser

    args = build_parser().parse_args(["data", "ingest", "--list"])
    assert args.list is True


@pytest.mark.network
def test_zenodo_listing_live():
    archives = ig.zenodo_archives()
    assert {a.key for a in archives} >= {"P1_chords.zip", "P3_music.zip"}
    small = archives[0]
    members = ig.di_members(small.key, ig.open_remote_zip(small.fetch, small.size))
    assert members
