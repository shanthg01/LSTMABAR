"""Task 4.4: real-pedal test set ingest (no network except tests marked ``network``)."""

import hashlib
import io
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from lstmabar.data import remote_zip, test_real
from lstmabar.data.fixtures import fixture_config
from lstmabar.data.layout import DataLayout
from lstmabar.data.remote_zip import file_range_reader, list_entries, read_member
from lstmabar.data.schema import TestRealItem, read_jsonl
from lstmabar.data.test_real import (
    EgfxNote,
    RemoteFile,
    RemoteRecord,
    knob_rotation,
    map_settings,
    pair_egfxset,
    parse_egfx_name,
    parse_egfx_settings,
    parse_pod_metadata,
    parse_pod_name,
    parse_pod_norm_summary,
    podset_pedals,
    select_podset_pedals,
    wav_to_flac,
)

# Three rows in the format of the real egfxset_metadata.csv (stray header spaces, literals).
EGFX_CSV = (
    "Effect ,Model,Effect Type,Knob Names,Knob Type,Setting \n"
    "blues driver,Boss  BD-2 Blues Driver,distortion,\"['level', 'tone', 'gain']\","
    "\"['volume','eq','effect amount']\",\"[0.5,0.5,1.0]\"\n"
    "tube screamer,Ibanez Mini Tube Screamer,distortion,\"['tone', 'overdrive', 'level']\","
    "\"['eq','effect amount','volume']\",\"[0.5,1.0,0.5]\"\n"
    "distortion,Pro Co Sound RAT2 Distortion,distortion,\"['distortion', 'filter', 'volume']\","
    "\"['effect amount','eq','volume']\",\"[1.0, 0.5,1.0]\"\n"
    "chorus,Boss CE-3 Chorus,modulation,\"['rate', 'depth', 'stereo mode']\","
    "\"['rate','effect amount','selector']\",\"['120 bpm', 1.0, False]\"\n"
)
POD_META = (
    "index,shortname,brand,model,datasetname\n"
    "13,TUB,Jam Pedals,Tube Dreamer,tubedreamer\n"
    "10,OCD,Fulltone,OCD v1.3,ocd\n"
    "14,KTR,Klon,KTR,ktr\n"
)
POD_NORM = (
    ",pedal,file_db_mean,file_db_std,file_db_max,norm_db_mean,norm_db_std,norm_db_max,"
    "appied_gain_db_mean,appied_gain_db_std,appied_gain_db_max\n"
    "0,ocd,-6.99,3.82,-2.90,-7.10,3.82,-3.00,-0.10,0.00,-0.10\n"
    "1,tubedreamer,-9.29,3.88,-3.70,-8.58,3.88,-3.00,0.70,0.00,0.70\n"
)


# --- remote_zip ------------------------------------------------------------------------------


def _make_zip(path: Path, members: dict[str, bytes], comment: bytes = b"") -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for i, (name, data) in enumerate(members.items()):
            method = zipfile.ZIP_STORED if i % 2 else zipfile.ZIP_DEFLATED
            zf.writestr(name, data, compress_type=method)
        zf.comment = comment


@pytest.mark.parametrize("comment", [b"", b"an archive comment"])
def test_central_directory_matches_zipfile(tmp_path, comment):
    rng = np.random.default_rng(0)
    members = {
        "dir/": b"",
        "dir/a.wav": rng.integers(0, 255, 5000, dtype=np.uint8).tobytes(),
        "dir/b.txt": b"hello " * 300,
        "dir/ünïcode.wav": b"\x00" * 1000,
    }
    p = tmp_path / "x.zip"
    _make_zip(p, members, comment)
    read = file_range_reader(p)
    entries = list_entries(read, p.stat().st_size)
    with zipfile.ZipFile(p) as zf:
        infos = zf.infolist()
    assert [e.name for e in entries] == [i.filename for i in infos]
    for e, i in zip(entries, infos, strict=True):
        assert (e.file_size, e.compressed_size, e.header_offset, e.crc) == (
            i.file_size,
            i.compress_size,
            i.header_offset,
            i.CRC,
        )
        assert read_member(read, e) == members[e.name]
    assert entries[0].is_dir and not entries[1].is_dir


def test_remote_zip_rejects_non_zip(tmp_path):
    p = tmp_path / "not.zip"
    p.write_bytes(b"x" * 100)
    with pytest.raises(ValueError, match="not a zip"):
        list_entries(file_range_reader(p), 100)


def test_read_member_checks_crc(tmp_path):
    p = tmp_path / "x.zip"
    _make_zip(p, {"a": b"abc" * 100, "b": b"stored data"})
    read = file_range_reader(p)
    e = list_entries(read, p.stat().st_size)[1]
    bad = remote_zip.ZipEntry(e.name, e.compressed_size, e.file_size, e.header_offset, 0, 0)
    with pytest.raises(ValueError, match="CRC"):
        read_member(read, bad)


# --- settings --------------------------------------------------------------------------------


def test_knob_rotation_formats():
    assert knob_rotation(0.5) == 0.5
    assert knob_rotation(1) == 1.0
    assert knob_rotation("75%") == 0.75
    assert knob_rotation("7 o'clock") == 0.0
    assert knob_rotation("12 o'clock") == 0.5
    assert knob_rotation("5 o'clock") == 1.0
    assert knob_rotation("2:30 oclock") == pytest.approx(0.75)
    assert knob_rotation("120 bpm") is None
    assert knob_rotation("tape echo") is None
    assert knob_rotation(False) is None
    with pytest.raises(ValueError):
        knob_rotation(1.5)
    with pytest.raises(ValueError):
        knob_rotation("6 o'clock")  # dead zone below the pot's travel


def test_egfx_csv_and_mapping_to_kb_pots():
    meta = parse_egfx_settings(EGFX_CSV)
    assert meta["tube screamer"]["model"] == "Ibanez Mini Tube Screamer"
    assert meta["blues driver"]["model"] == "Boss BD-2 Blues Driver"  # whitespace collapsed
    ts = map_settings(
        meta["tube screamer"]["knobs"], {"overdrive": "drive"}, ["drive", "tone", "level"]
    )
    assert ts == {"drive": 1.0, "tone": 0.5, "level": 0.5}
    rat = map_settings(meta["distortion"]["knobs"], None, ["distortion", "filter", "volume"])
    assert rat == {"distortion": 1.0, "filter": 0.5, "volume": 1.0}
    assert map_settings(meta["chorus"]["knobs"]) == {"depth": 1.0}  # tempo/selector skipped
    with pytest.raises(KeyError, match="not KB pots"):
        map_settings(meta["tube screamer"]["knobs"], None, ["drive", "tone", "level"])


def test_egfx_settings_match_kb_pot_names():
    from lstmabar.physics.kb import load_kb

    kb = load_kb()
    cfg = fixture_config(Path("unused"))
    meta = parse_egfx_settings(EGFX_CSV)
    ec = cfg.test_real.egfxset
    for unit, kb_id in ec.kb_map.items():
        if kb_id is None:
            continue
        row = meta[ec.csv_effect[unit]]
        s = map_settings(row["knobs"], dict(ec.knob_map).get(unit), list(kb[kb_id].pots))
        assert set(s) == set(kb[kb_id].pots)
        kb[kb_id].knobs(s)  # rotations valid for the KB pedal


# --- pairing ---------------------------------------------------------------------------------


def test_parse_egfx_name():
    assert parse_egfx_name("RAT/Bridge-Middle/6-1.wav") == ("RAT", EgfxNote("Bridge-Middle", 6, 1))
    assert parse_egfx_name("Clean/Middle/.DS_Store") is None
    assert parse_egfx_name("Clean/Neck/") is None
    assert parse_egfx_name("TubeScreamer/Bridge-Middle/1-23.wav") is None  # no fret 23
    assert parse_egfx_name("RAT/Bridge/7-1.wav") is None


def test_pair_egfxset_synthetic_lists():
    clean = ["Clean/", "Clean/Neck/", "Clean/Middle/.DS_Store"] + [
        f"Clean/{p}/{s}-{f}.wav" for p in ("Neck", "Bridge") for s in (1, 6) for f in (0, 17)
    ]
    units = {
        "RAT": [
            f"RAT/{p}/{s}-{f}.wav" for p in ("Neck", "Bridge") for s in (1, 6) for f in (0, 17)
        ],
        "TubeScreamer": [
            "TubeScreamer/Neck/1-0.wav",
            "TubeScreamer/Bridge/1-23.wav",  # mislabelled in the real dataset; not guessed
            "TubeScreamer/Middle/6-0.wav",  # no clean counterpart
            "RAT/Neck/1-0.wav",  # wrong folder
        ],
    }
    pr = pair_egfxset(clean, units)
    assert len(pr.pairs["RAT"]) == 8 and pr.unpaired["RAT"] == []
    assert pr.pairs["TubeScreamer"] == [EgfxNote("Neck", 1, 0)]
    assert pr.unpaired["TubeScreamer"] == sorted(
        ["TubeScreamer/Bridge/1-23.wav", "TubeScreamer/Middle/6-0.wav", "RAT/Neck/1-0.wav"]
    )
    assert pr.clean_needed == set(pr.pairs["RAT"])


def test_pod_parsing_and_selection():
    pf = parse_pod_name("ocd/a_ocd_g5_t0.wav")
    assert (pf.pedal, pf.kind, pf.gain, pf.tone) == ("ocd", "a", 5, 0)
    assert pf.settings() == {"gain": 1.0, "tone": 0.0}
    assert parse_pod_name("ocd/s1_ocd_g2_t3.wav").settings() == {"gain": 0.4, "tone": 0.6}
    assert parse_pod_name("ocd/0-info_ocd.txt") is None
    assert parse_pod_name("ocd/a_ktr_g1_t1.wav") is None  # pedal name must match folder
    assert parse_pod_name("ocd/a_ocd_g6_t1.wav") is None
    keys = ["0_input.zip", "0_dataset_metadata.csv", "ocd.zip", "ktr.zip", "385.zip"]
    avail = podset_pedals(keys)
    assert avail == ["385", "ktr", "ocd"]
    assert select_podset_pedals(avail, ["ocd", "ktr"], 5) == ["ocd", "ktr"]
    assert select_podset_pedals(avail, ["ocd", "ktr"], 1) == ["ocd"]
    assert select_podset_pedals(avail, [], 2) == ["385", "ktr"]
    with pytest.raises(ValueError, match="no pedals"):
        select_podset_pedals(avail, ["klon"], 3)
    assert parse_pod_metadata(POD_META)["ktr"] == {
        "brand": "Klon",
        "model": "KTR",
        "shortname": "KTR",
    }
    assert parse_pod_norm_summary(POD_NORM) == {"ocd": -0.10, "tubedreamer": 0.70}


def test_wav_to_flac_is_lossless(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.integers(-(2**23), 2**23, 4800).astype(np.int32) << 8  # 24-bit in int32
    buf = io.BytesIO()
    sf.write(buf, x, 48000, format="WAV", subtype="PCM_24")
    wav_to_flac(buf.getvalue(), tmp_path / "a" / "x.flac")
    y, sr = sf.read(str(tmp_path / "a" / "x.flac"), dtype="int32")
    assert sr == 48000 and np.array_equal(x, y)
    assert sf.info(str(tmp_path / "a" / "x.flac")).subtype == "PCM_24"
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32) / 2**31, 48000, format="WAV", subtype="FLOAT")
    with pytest.raises(ValueError, match="losslessly"):
        wav_to_flac(buf.getvalue(), tmp_path / "f.flac")


# --- end to end against a local fake Zenodo ---------------------------------------------------


def _wav(seed: int, n: int = 480) -> bytes:
    x = np.random.default_rng(seed).uniform(-0.5, 0.5, n)
    buf = io.BytesIO()
    sf.write(buf, x, 48000, format="WAV", subtype="PCM_24")
    return buf.getvalue()


class FakeRemote:
    """Zenodo stand-in over local files; counts downloads."""

    def __init__(self, root: Path, records: dict[str, tuple[str, dict[str, bytes]]]):
        self.records: dict[str, RemoteRecord] = {}
        self.downloads: list[str] = []
        for rec, (lic, files) in records.items():
            d = root / rec
            d.mkdir(parents=True)
            out = {}
            for key, data in files.items():
                (d / key).write_bytes(data)
                md5 = hashlib.md5(data).hexdigest()
                out[key] = RemoteFile(key, len(data), f"md5:{md5}", str(d / key))
            self.records[rec] = RemoteRecord(rec, lic, out)

    def record(self, record):
        return self.records[record]

    def entries(self, f):
        return list_entries(file_range_reader(f.url), f.size)

    def fetch(self, f):
        return Path(f.url).read_bytes()

    def download(self, f, dest):
        self.downloads.append(f.key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f.url, dest)
        test_real.verify_checksum(dest, f.checksum)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _fake_remote(root: Path) -> FakeRemote:
    notes = [("Neck", 1, 0), ("Neck", 6, 22), ("Bridge-Middle", 1, 17)]
    clean = {f"Clean/{p}/{s}-{f}.wav": _wav(i) for i, (p, s, f) in enumerate(notes)}
    clean["Clean/Middle/.DS_Store"] = b"junk"

    def unit(name, seed, swap=False):
        m = {}
        for i, (p, s, f) in enumerate(notes):
            if swap and (s, f) == (1, 17):
                f = 23  # the real TubeScreamer archive's mislabelled note
            m[f"{name}/{p}/{s}-{f}.wav"] = _wav(seed + i)
        return m

    egfx = {
        "Clean.zip": _zip_bytes(clean),
        "TubeScreamer.zip": _zip_bytes(unit("TubeScreamer", 10, swap=True)),
        "RAT.zip": _zip_bytes(unit("RAT", 20)),
        "BluesDriver.zip": _zip_bytes(unit("BluesDriver", 30)),
        "Chorus.zip": b"never fetched",
        "egfxset_metadata.csv": EGFX_CSV.encode(),
    }
    inputs = {f"input/{k}_0-input.wav": _wav(40 + i) for i, k in enumerate(("a", "n", "s0"))}
    inputs["input/0_info.txt"] = b"info"

    def pod(p, seed):
        m = {f"{p}/0-info_{p}.txt": b"info"}
        for g in (0, 5):
            for t in (0, 3):
                for k in ("a", "n", "s0"):
                    m[f"{p}/{k}_{p}_g{g}_t{t}.wav"] = _wav(seed + 10 * g + t)
        return m

    podset = {
        "0_input.zip": _zip_bytes(inputs),
        "0_dataset_metadata.csv": POD_META.encode(),
        "0_normalization_summary.csv": POD_NORM.encode(),
        "ocd.zip": _zip_bytes(pod("ocd", 100)),
        "tubedreamer.zip": _zip_bytes(pod("tubedreamer", 200)),
        "ktr.zip": _zip_bytes(pod("ktr", 300)),
    }
    return FakeRemote(
        root / "remote", {"7044411": ("cc-by-4.0", egfx), "15389653": ("cc-by-nc-4.0", podset)}
    )


def _cfg(tmp_path, *extra):
    return fixture_config(
        tmp_path,
        [
            "test_real.podset.pedals=[ocd,tubedreamer]",
            "test_real.podset.kinds=[a,s0]",
            *extra,
        ],
    )


def test_list_dry_run_downloads_nothing(tmp_path, capsys):
    remote = _fake_remote(tmp_path)
    cfg = _cfg(tmp_path)
    assert test_real.run(cfg, SimpleNamespace(list=True), remote=remote) == 0
    out = capsys.readouterr().out
    assert remote.downloads == []
    assert not DataLayout.from_config(cfg).test_real.exists()
    assert "TubeScreamer: 2 pairs, 1 unpaired" in out
    assert "Disk: download" in out and "egfxset: 8 items" in out and "podset: 16 items" in out


def test_end_to_end_manifest(tmp_path):
    remote = _fake_remote(tmp_path)
    cfg = _cfg(tmp_path)
    layout = DataLayout.from_config(cfg)
    assert test_real.run(cfg, SimpleNamespace(list=False), remote=remote) == 0
    # one archive at a time, only the needed ones, each deleted after extraction
    assert remote.downloads == [
        "Clean.zip",
        "TubeScreamer.zip",
        "RAT.zip",
        "BluesDriver.zip",
        "0_input.zip",
        "ocd.zip",
        "tubedreamer.zip",
    ]
    assert not list(layout.test_real.rglob("*.zip")) and not list(layout.test_real.rglob("*.part"))
    assert not list(layout.test_real.rglob("*.wav"))  # stored as FLAC only
    items = list(read_jsonl(layout.test_real_manifest, TestRealItem))
    assert [i.item_id for i in items] == sorted(i.item_id for i in items)
    by_id = {i.item_id: i for i in items}
    assert len(items) == 3 * 3 - 1 + 2 * 4 * 2  # TS loses its mislabelled note
    ts = by_id["egfxset-tubescreamer-neck-1-0"]
    assert ts.kb_pedal_id == "ts808" and ts.family == "overdrive"
    assert ts.settings == {"drive": 1.0, "tone": 0.5, "level": 0.5}
    assert ts.dry_path == "egfxset/Clean/Neck/1-0.flac"
    assert ts.wet_path == "egfxset/TubeScreamer/Neck/1-0.flac"
    assert ts.license == "cc-by-4.0" and ts.internal_only is False and ts.sample_rate == 48000
    assert by_id["egfxset-rat-bridge-middle-1-17"].settings == {
        "distortion": 1.0,
        "filter": 0.5,
        "volume": 1.0,
    }
    bd = by_id["egfxset-bluesdriver-neck-6-22"]
    assert bd.kb_pedal_id is None and bd.settings == {"level": 0.5, "tone": 0.5, "gain": 1.0}
    pod = by_id["podset-ocd-a-g5t3"]
    assert pod.internal_only is True and pod.license == "cc-by-nc-4.0"
    assert pod.settings == {"gain": 1.0, "tone": 0.6}
    assert pod.dry_path == "podset/input/a_0-input.flac"
    assert pod.wet_path == "podset/ocd/a_ocd_g5_t3.flac"
    assert "Fulltone OCD v1.3" in pod.notes and "-0.10 dB" in pod.notes
    assert by_id["podset-tubedreamer-s0-g0t0"].dry_path == "podset/input/s0_0-input.flac"
    assert not any(i.item_id.startswith("podset-ktr") or "-n-" in i.item_id for i in items)
    assert all(i.internal_only == (i.dataset == "podset") for i in items)
    for i in items:
        assert (layout.test_real / i.dry_path).exists()
        assert (layout.test_real / i.wet_path).exists()
        assert len(i.sha256_dry) == len(i.sha256_wet) == 64
    # extracted audio is bit-identical to the archive member
    x, _ = sf.read(io.BytesIO(_wav(20)), dtype="int32")
    y, _ = sf.read(str(layout.test_real / "egfxset/RAT/Neck/1-0.flac"), dtype="int32")
    assert np.array_equal(x, y)

    # resumable / idempotent: a rerun downloads nothing and writes the same manifest
    before = layout.test_real_manifest.read_bytes()
    remote.downloads.clear()
    test_real.run(cfg, SimpleNamespace(list=False), remote=remote)
    assert remote.downloads == []
    assert layout.test_real_manifest.read_bytes() == before

    # a missing extracted file re-triggers only its archive
    (layout.test_real / "egfxset/RAT/Neck/1-0.flac").unlink()
    test_real.run(cfg, SimpleNamespace(list=False), remote=remote)
    assert remote.downloads == ["RAT.zip"]


def test_rerun_one_dataset_keeps_the_other(tmp_path):
    remote = _fake_remote(tmp_path)
    test_real.run(_cfg(tmp_path), SimpleNamespace(list=False), remote=remote)
    cfg = _cfg(tmp_path, "test_real.egfxset.enabled=false", "test_real.podset.pedals=[ocd]")
    test_real.run(cfg, SimpleNamespace(list=False), remote=remote)
    items = list(read_jsonl(DataLayout.from_config(cfg).test_real_manifest, TestRealItem))
    assert sum(i.dataset == "egfxset" for i in items) == 8
    assert {i.device for i in items if i.dataset == "podset"} == {"ocd"}


def test_tonetwist_is_a_stub(tmp_path):
    cfg = _cfg(tmp_path, "test_real.tonetwist.enabled=true")
    with pytest.raises(NotImplementedError, match="ToneTwist"):
        test_real.run(cfg, SimpleNamespace(list=True), remote=_fake_remote(tmp_path))


def test_download_resumes_and_checks_md5(tmp_path, monkeypatch):
    data = bytes(range(256)) * 40
    f = RemoteFile("x.zip", len(data), "md5:" + hashlib.md5(data).hexdigest(), "http://x")
    seen = []

    class Resp(io.BytesIO):
        def __init__(self, payload, status):
            super().__init__(payload)
            self.status = status

        def __enter__(self):
            return self

    def fake_get(url, headers=None, timeout=60.0):
        rng = (headers or {}).get("Range")
        seen.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        return Resp(data[start:], 206 if rng else 200)

    monkeypatch.setattr(test_real, "_get", fake_get)
    dest = tmp_path / "d" / "x.zip"
    dest.parent.mkdir()
    (tmp_path / "d" / "x.zip.part").write_bytes(data[:1000])
    test_real.Zenodo().download(f, dest)
    assert seen == ["bytes=1000-"] and dest.read_bytes() == data
    bad = RemoteFile("y.zip", len(data), "md5:" + "0" * 32, "http://x")
    with pytest.raises(OSError, match="mismatch"):
        test_real.Zenodo().download(bad, tmp_path / "d" / "y.zip")
    assert not (tmp_path / "d" / "y.zip.part").exists()


# --- network (skipped by default; run with `pytest -m network`) -------------------------------


@pytest.mark.network
def test_egfxset_record_listing():
    z = test_real.Zenodo()
    rec = z.record("7044411")
    assert rec.license == "cc-by-4.0"
    names = [e.name for e in z.entries(rec.files["RAT.zip"])]
    assert sum(n.endswith(".wav") for n in names) == 690
    meta = parse_egfx_settings(z.fetch(rec.files["egfxset_metadata.csv"]).decode("utf-8-sig"))
    assert meta["tube screamer"]["knobs"] == {"tone": 0.5, "overdrive": 1.0, "level": 0.5}


@pytest.mark.network
def test_podset_record_listing():
    rec = test_real.Zenodo().record("15389653")
    assert rec.license == "cc-by-nc-4.0"
    assert len(podset_pedals(rec.files)) == 27


def test_member_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError, match=r"\.\."):
        test_real._relpath("egfxset", "RAT/../1-0.wav")
    assert test_real._relpath("podset", "ocd/a_ocd_g0_t0.wav") == "podset/ocd/a_ocd_g0_t0.flac"
