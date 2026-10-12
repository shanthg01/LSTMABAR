"""Task 4.4 (wave 1d): real-pedal test set (EGFxSet units, a few pOD-set pedals).

``run(cfg, args)`` lists each Zenodo record through its API and each needed archive's central
directory through HTTP range requests (:mod:`lstmabar.data.remote_zip`), plans which members
are needed, then downloads **one archive at a time** into ``data.test_real_dir/<dataset>/``
(resumable ``.part`` file, md5 checked), extracts only the planned members as lossless
FLAC (same bit depth as the source WAV), deletes the archive and leaves a ``.done`` marker
(R9). It pairs each real-pedal recording with its clean input and writes ``TestRealItem``
records to ``DataLayout.test_real_manifest``. ``--list`` prints the plan and a disk estimate
without downloading anything.

Datasets (decision 8, docs/data-licenses.md):

- **EGFxSet** (Zenodo 7044411, CC BY 4.0): 5 s single notes, 48 kHz / 24-bit mono, files
  ``<Unit>/<Pickup>/<string>-<fret>.wav``; each effected note pairs with the ``Clean`` note of
  the same pickup/string/fret. Each unit was recorded at one setting (``egfxset_metadata.csv``,
  knob rotations already in [0, 1]). Test-only (R2).
- **pOD-set** (Zenodo 15389653, CC BY-NC 4.0): per-pedal zips of
  ``<pedal>/<kind>_<pedal>_g<G>_t<T>.wav`` (G, T in 0..5 = knob marks 0, 2, ..., 10), all
  rendered from the same input files in ``0_input.zip`` (``input/<kind>_0-input.wav``):
  ``a`` = ~6 min mixed material (Yeh et al., not pure DI), ``s0``/``s1``/``s2`` = sine sweeps
  at -6/-12/-24 dBFS. ``internal_only``.
- **ToneTwist**: not implemented (internal only, never republished); opt-in stub.

Both EGFxSet and pOD-set outputs are level-normalized, so the true pedal input voltage is
unknown (decision 8): the notes field says so per item.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import soundfile as sf

from lstmabar.data.layout import DataLayout
from lstmabar.data.levels import sha256_file
from lstmabar.data.remote_zip import ZipEntry, list_remote_zip
from lstmabar.data.schema import TestRealItem, check_relpath, read_jsonl, write_jsonl

ZENODO_RECORD_API = "https://zenodo.org/api/records/{record}"
SMALL_FILE_MAX = 50_000_000  # bytes; metadata files fetched into memory
# Measured FLAC / WAV size ratios (24-bit, lossless) on one file each (2026-10-12): EGFxSet
# clean 0.49, RAT 0.73; pOD-set tubedreamer "a" 0.50. Used only for the --list estimate.
FLAC_RATIO = {"egfxset": 0.65, "podset": 0.55}
DISK_MARGIN = 1_000_000_000  # bytes kept free on top of an archive + its extraction
_USER_AGENT = "lstmabar-data/1 (+https://github.com/shanthg01)"
_FLAC_SUBTYPES = ("PCM_16", "PCM_24")

EGFX_NOTE = re.compile(
    r"^(?P<unit>[^/]+)/(?P<pickup>[^/]+)/(?P<string>[1-6])-(?P<fret>\d{1,2})\.wav$"
)
EGFX_MAX_FRET = 22
POD_FILE = re.compile(
    r"^(?P<pedal>[^/]+)/(?P<kind>a|n|s0|s1|s2)_(?P=pedal)_g(?P<g>[0-5])_t(?P<t>[0-5])\.wav$"
)
POD_INPUT = re.compile(r"^input/(?P<kind>a|n|s0|s1|s2)_0-input\.wav$")
POD_GRID_STEPS = 5  # g/t index 0..5 -> rotation index / 5
POD_KIND_NOTES = {
    "a": "~6 min mixed instrumental material (Yeh et al. 2024), not pure DI guitar",
    "s0": "sine sweep 15 Hz-24 kHz at -6 dBFS (input file)",
    "s1": "sine sweep 15 Hz-24 kHz at -12 dBFS (input file)",
    "s2": "sine sweep 15 Hz-24 kHz at -24 dBFS (input file)",
    "n": "noise floor (no input)",
}
INTERNAL_ONLY = {"egfxset": False, "podset": True, "tonetwist": True}


# --- Remote access --------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteFile:
    key: str
    size: int
    checksum: str  # "md5:<hex>" (Zenodo) or ""
    url: str


@dataclass(frozen=True)
class RemoteRecord:
    record: str
    license: str
    files: dict[str, RemoteFile]


class Remote(Protocol):
    """What the ingest needs from a file host (Zenodo; a local fake in tests)."""

    def record(self, record: str) -> RemoteRecord: ...

    def entries(self, f: RemoteFile) -> list[ZipEntry]: ...

    def fetch(self, f: RemoteFile) -> bytes: ...

    def download(self, f: RemoteFile, dest: Path) -> None: ...


def _get(url: str, headers: Mapping[str, str] | None = None, timeout: float = 60.0):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def verify_checksum(path: Path, checksum: str) -> None:
    if not checksum:
        return
    algo, _, want = checksum.partition(":")
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    if h.hexdigest() != want:
        raise OSError(f"{path.name}: {algo} mismatch (download corrupted; it was removed)")


class Zenodo:
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout

    def record(self, record: str) -> RemoteRecord:
        with _get(ZENODO_RECORD_API.format(record=record), timeout=self.timeout) as r:
            d = json.load(r)
        lic = d.get("metadata", {}).get("license", {})
        lic = lic.get("id", "") if isinstance(lic, Mapping) else str(lic)
        files = {
            f["key"]: RemoteFile(f["key"], int(f["size"]), f.get("checksum", ""), _file_url(f))
            for f in d.get("files", [])
        }
        return RemoteRecord(str(record), lic, files)

    def entries(self, f: RemoteFile) -> list[ZipEntry]:
        return list_remote_zip(f.url, f.size)

    def fetch(self, f: RemoteFile) -> bytes:
        if f.size > SMALL_FILE_MAX:
            raise ValueError(f"{f.key} is {f.size} bytes; fetch() is for small files only")
        with _get(f.url, timeout=self.timeout) as r:
            return r.read()

    def download(self, f: RemoteFile, dest: Path) -> None:
        """Resumable: bytes already in ``<dest>.part`` are kept and the rest is requested."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        if have > f.size:
            part.unlink()
            have = 0
        if have < f.size:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with _get(f.url, headers, timeout=self.timeout) as r:
                if have and r.status != 206:
                    have = 0
                with part.open("ab" if have else "wb") as out:
                    shutil.copyfileobj(r, out, 1 << 20)
        if part.stat().st_size != f.size:
            raise OSError(f"{f.key}: incomplete download; rerun to resume")
        try:
            verify_checksum(part, f.checksum)
        except OSError:
            part.unlink()
            raise
        part.replace(dest)


def _file_url(f: Mapping[str, Any]) -> str:
    links = f.get("links", {})
    return links.get("self") or links.get("content") or links.get("download", "")


# --- EGFxSet ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class EgfxNote:
    pickup: str
    string: int
    fret: int

    @property
    def stem(self) -> str:
        return f"{self.string}-{self.fret}"


def parse_egfx_name(name: str) -> tuple[str, EgfxNote] | None:
    """``"RAT/Bridge/6-1.wav"`` -> ``("RAT", EgfxNote("Bridge", 6, 1))``; ``None`` for
    directories, ``.DS_Store`` and impossible frets."""
    m = EGFX_NOTE.match(name)
    if not m or int(m["fret"]) > EGFX_MAX_FRET:
        return None
    return m["unit"], EgfxNote(m["pickup"], int(m["string"]), int(m["fret"]))


@dataclass
class EgfxPairing:
    pairs: dict[str, list[EgfxNote]]  # unit -> notes present in both unit and clean
    unpaired: dict[str, list[str]]  # unit -> member names without a clean match (or invalid)
    clean_needed: set[EgfxNote]


def pair_egfxset(clean_names: Iterable[str], unit_names: Mapping[str, Iterable[str]]):
    """Pair every effected note with the clean note of the same pickup/string/fret.

    Only exact name matches pair: e.g. TubeScreamer has ``Bridge-Middle/1-23.wav`` (no fret 23
    on the 22-fret Strat) where Clean has ``1-17``; such files are reported, not guessed.
    """
    clean: set[EgfxNote] = set()
    for n in clean_names:
        p = parse_egfx_name(n)
        if p is not None:
            clean.add(p[1])
    pairs: dict[str, list[EgfxNote]] = {}
    unpaired: dict[str, list[str]] = {}
    for unit, names in unit_names.items():
        ok: list[EgfxNote] = []
        bad: list[str] = []
        for n in names:
            if n.endswith("/"):
                continue
            p = parse_egfx_name(n)
            if p is None or p[0] != unit:
                if n.endswith(".wav"):
                    bad.append(n)
            elif p[1] in clean:
                ok.append(p[1])
            else:
                bad.append(n)
        pairs[unit] = sorted(ok, key=lambda x: (x.pickup, x.string, x.fret))
        unpaired[unit] = sorted(bad)
    needed = {n for notes in pairs.values() for n in notes}
    return EgfxPairing(pairs, unpaired, needed)


def parse_egfx_settings(text: str) -> dict[str, dict[str, Any]]:
    """``egfxset_metadata.csv`` -> ``{effect: {"model", "type", "knobs": {name: value}}}``.

    Header cells carry stray spaces (``"Effect "``); list cells are Python literals.
    """
    rows = csv.reader(io.StringIO(text))
    header = [h.strip().lower() for h in next(rows)]
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not any(c.strip() for c in row):
            continue
        r = dict(zip(header, row, strict=False))
        names = ast.literal_eval(r["knob names"].strip())
        values = ast.literal_eval(r["setting"].strip())
        if len(names) != len(values):
            raise ValueError(f"egfxset csv: {r['effect']!r}: {len(names)} knobs, {len(values)}")
        out[r["effect"].strip().lower()] = {
            "model": " ".join(r["model"].split()),
            "type": r["effect type"].strip(),
            "knobs": {str(k).strip(): v for k, v in zip(names, values, strict=True)},
        }
    return out


def map_settings(
    knobs: Mapping[str, Any],
    rename: Mapping[str, str] | None = None,
    allowed: Iterable[str] | None = None,
) -> dict[str, float]:
    """Dataset knob values -> ``{pot: rotation in [0, 1]}``.

    Values may be fractions (``0.5``), percentages (``"50%"``) or clock positions
    (``"12 o'clock"``, 7 o'clock = 0, 5 o'clock = 1, a 300 degree sweep). Non-numeric settings
    (selectors, tempos) are skipped. ``rename`` maps dataset names to KB pot names; with
    ``allowed`` (the KB pot names) any other name raises.
    """
    rename = dict(rename or {})
    out: dict[str, float] = {}
    for name, raw in knobs.items():
        v = knob_rotation(raw)
        if v is None:
            continue
        out[rename.get(name, name)] = v
    if allowed is not None:
        extra = set(out) - set(allowed)
        if extra:
            raise KeyError(f"settings {sorted(extra)} are not KB pots {sorted(allowed)}")
    return out


_CLOCK = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*o'?clock$", re.IGNORECASE)


def knob_rotation(raw: Any) -> float | None:
    """One knob value as a rotation in [0, 1], or ``None`` if it is not a rotation."""
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int | float):
        v = float(raw)
    else:
        s = str(raw).strip()
        if s.endswith("%"):
            v = float(s[:-1]) / 100.0
        elif m := _CLOCK.match(s):
            hour = int(m[1]) + int(m[2] or 0) / 60.0
            v = ((hour - 7.0) % 12.0) / 10.0  # 7:00 -> 0, 12:00 -> 0.5, 5:00 -> 1
        else:
            try:
                v = float(s)
            except ValueError:
                return None  # selectors, tempos ("120 bpm"), modes
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"knob setting {raw!r} is outside [0, 1]")
    return v


# --- pOD-set ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PodFile:
    pedal: str
    kind: str
    gain: int  # 0..5
    tone: int  # 0..5

    def settings(self) -> dict[str, float]:
        return {"gain": self.gain / POD_GRID_STEPS, "tone": self.tone / POD_GRID_STEPS}


def parse_pod_name(name: str) -> PodFile | None:
    m = POD_FILE.match(name)
    if not m:
        return None
    return PodFile(m["pedal"], m["kind"], int(m["g"]), int(m["t"]))


def podset_pedals(record_keys: Iterable[str]) -> list[str]:
    return sorted(k[:-4] for k in record_keys if k.endswith(".zip") and not k.startswith("0_"))


def select_podset_pedals(
    available: Iterable[str], preferred: Iterable[str], max_pedals: int
) -> list[str]:
    """``preferred`` in order (all must exist), capped at ``max_pedals``; empty -> the first
    ``max_pedals`` available pedals alphabetically."""
    avail = sorted(set(available))
    pref = list(preferred)
    missing = [p for p in pref if p not in avail]
    if missing:
        raise ValueError(f"pOD-set has no pedals {missing}; available: {avail}")
    chosen = pref or avail
    return chosen[: max(0, int(max_pedals))]


def pod_input_names(names: Iterable[str]) -> dict[str, str]:
    """kind -> member name in ``0_input.zip``."""
    out: dict[str, str] = {}
    for n in names:
        m = POD_INPUT.match(n)
        if m:
            out[m["kind"]] = n
    return out


def parse_pod_metadata(text: str) -> dict[str, dict[str, str]]:
    """``0_dataset_metadata.csv`` -> ``{datasetname: {"brand", "model", "shortname"}}``."""
    out: dict[str, dict[str, str]] = {}
    for r in csv.DictReader(io.StringIO(text)):
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if r.get("datasetname"):
            out[r["datasetname"]] = {k: r.get(k, "") for k in ("brand", "model", "shortname")}
    return out


def parse_pod_norm_summary(text: str) -> dict[str, float]:
    """``0_normalization_summary.csv`` -> per-pedal applied gain in dB (constant per pedal)."""
    out: dict[str, float] = {}
    for r in csv.DictReader(io.StringIO(text)):
        try:
            out[r["pedal"].strip()] = float(r["appied_gain_db_mean"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


# --- Planning ---------------------------------------------------------------------------------


@dataclass
class ArchivePlan:
    dataset: str
    file: RemoteFile
    members: dict[str, str]  # member name -> relpath under data.test_real_dir (.flac)
    extracted_bytes: int  # uncompressed size of the planned members

    @property
    def marker_name(self) -> str:
        return f".{self.file.key}.done"


@dataclass
class DatasetPlan:
    dataset: str
    license: str
    archives: list[ArchivePlan]
    items: list[dict[str, Any]]  # TestRealItem kwargs without hashes / sample_rate
    report: list[str] = field(default_factory=list)


def _relpath(dataset: str, member: str) -> str:
    """Archive member -> relpath of its FLAC; member names are untrusted (no ``..`` etc.)."""
    if not member.endswith(".wav"):
        raise ValueError(f"not a WAV member: {member!r}")
    rel = DataLayout.test_real_relpath(dataset, *member[: -len(".wav")].split("/")) + ".flac"
    check_relpath(rel, "archive member")
    return rel


def plan_egfxset(cfg, remote: Remote, kb_pots: Mapping[str, Iterable[str]]) -> DatasetPlan:
    ec = cfg.test_real.egfxset
    rec = remote.record(str(ec.record))
    clean_key = f"{ec.clean}.zip"
    units = list(ec.units)
    want = [clean_key, *(f"{u}.zip" for u in units)]
    missing = [k for k in [*want, ec.csv] if k not in rec.files]
    if missing:
        raise KeyError(f"EGFxSet record {rec.record} has no {missing}")
    entries = {k: remote.entries(rec.files[k]) for k in want}
    names = {k: [e.name for e in v] for k, v in entries.items()}
    pairing = pair_egfxset(names[clean_key], {u: names[f"{u}.zip"] for u in units})
    meta = parse_egfx_settings(remote.fetch(rec.files[ec.csv]).decode("utf-8-sig"))
    kb_map = dict(ec.kb_map)
    knob_map = {k: dict(v) for k, v in dict(ec.knob_map).items()}
    family = dict(ec.family)
    effect_of = dict(ec.csv_effect)

    def archive(key: str, folder: str, notes: Iterable[EgfxNote]) -> ArchivePlan:
        sizes = {e.name: e.file_size for e in entries[key]}
        members = {}
        for n in notes:
            m = f"{folder}/{n.pickup}/{n.stem}.wav"
            members[m] = _relpath("egfxset", m)
        return ArchivePlan("egfxset", rec.files[key], members, sum(sizes[m] for m in members))

    archives = [archive(clean_key, ec.clean, sorted(pairing.clean_needed, key=_note_key))]
    items: list[dict[str, Any]] = []
    report = [f"EGFxSet record {rec.record} ({rec.license})"]
    for u in units:
        archives.append(archive(f"{u}.zip", u, pairing.pairs[u]))
        row = meta[effect_of[u].lower()]
        kb = kb_map.get(u)
        allowed = kb_pots[kb] if kb else None
        settings = map_settings(row["knobs"], knob_map.get(u), allowed)
        report.append(
            f"  {u}: {len(pairing.pairs[u])} pairs, {len(pairing.unpaired[u])} unpaired "
            f"{pairing.unpaired[u][:3]}; {row['model']}; settings {settings}; kb={kb}"
        )
        for n in pairing.pairs[u]:
            items.append(
                dict(
                    item_id=f"egfxset-{_slug(u)}-{_slug(n.pickup)}-{n.stem}",
                    dataset="egfxset",
                    device=u,
                    family=str(family[u]),
                    kb_pedal_id=kb,
                    settings=dict(settings),
                    dry_path=_relpath("egfxset", f"{ec.clean}/{n.pickup}/{n.stem}.wav"),
                    wet_path=_relpath("egfxset", f"{u}/{n.pickup}/{n.stem}.wav"),
                    internal_only=INTERNAL_ONLY["egfxset"],
                    license=rec.license,
                    notes=(
                        f"{row['model']}; single 5 s note, string {n.string} fret {n.fret}, "
                        f"{n.pickup} pickup; recordings are level-normalized, so the true "
                        f"input level is unknown"
                    ),
                )
            )
    return DatasetPlan("egfxset", rec.license, archives, items, report)


def plan_podset(cfg, remote: Remote) -> DatasetPlan:
    pc = cfg.test_real.podset
    rec = remote.record(str(pc.record))
    pedals = select_podset_pedals(
        podset_pedals(rec.files), list(pc.pedals or []), int(pc.max_pedals)
    )
    kinds = list(pc.kinds)
    meta = parse_pod_metadata(remote.fetch(rec.files[pc.metadata_csv]).decode("utf-8-sig"))
    norm: dict[str, float] = {}
    if pc.norm_csv in rec.files:
        norm = parse_pod_norm_summary(remote.fetch(rec.files[pc.norm_csv]).decode("utf-8-sig"))
    in_file = rec.files[pc.input_archive]
    in_entries = remote.entries(in_file)
    inputs = pod_input_names(e.name for e in in_entries)
    missing = [k for k in kinds if k not in inputs]
    if missing:
        raise KeyError(f"pOD-set input archive has no input for kinds {missing}")
    in_sizes = {e.name: e.file_size for e in in_entries}
    in_members = {inputs[k]: _relpath("podset", inputs[k]) for k in kinds}
    archives = [ArchivePlan("podset", in_file, in_members, sum(in_sizes[m] for m in in_members))]
    kb_map = dict(pc.kb_map or {})
    items: list[dict[str, Any]] = []
    report = [f"pOD-set record {rec.record} ({rec.license}); pedals {pedals}"]
    for p in pedals:
        f = rec.files[f"{p}.zip"]
        ents = remote.entries(f)
        members: dict[str, str] = {}
        size = 0
        for e in ents:
            pf = parse_pod_name(e.name)
            if pf is None or pf.pedal != p or pf.kind not in kinds:
                continue
            members[e.name] = _relpath("podset", e.name)
            size += e.file_size
            m = meta.get(p, {})
            label = " ".join(x for x in (m.get("brand", ""), m.get("model", "")) if x) or p
            gain_note = f" (applied {norm[p]:+.2f} dB)" if p in norm else ""
            items.append(
                dict(
                    item_id=f"podset-{_slug(p)}-{pf.kind}-g{pf.gain}t{pf.tone}",
                    dataset="podset",
                    device=p,
                    family=str(pc.family),
                    kb_pedal_id=kb_map.get(p),
                    settings=pf.settings(),
                    dry_path=in_members[inputs[pf.kind]],
                    wet_path=members[e.name],
                    internal_only=INTERNAL_ONLY["podset"],
                    license=rec.license,
                    notes=(
                        f"{label}; gain {2 * pf.gain}/10, tone {2 * pf.tone}/10; input: "
                        f"{POD_KIND_NOTES[pf.kind]}; outputs normalized per pedal so the "
                        f"loudest file peaks at -3 dBFS{gain_note}; true input voltage unknown"
                    ),
                )
            )
        grid = {(x.gain, x.tone) for x in map(parse_pod_name, members) if x}
        report.append(f"  {p}: {len(members)} files, {len(grid)} grid points")
        archives.append(ArchivePlan("podset", f, members, size))
    return DatasetPlan("podset", rec.license, archives, items, report)


def _note_key(n: EgfxNote) -> tuple[str, int, int]:
    return (n.pickup, n.string, n.fret)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# --- Download + extraction (R9) ---------------------------------------------------------------


def archive_done(root: Path, plan: ArchivePlan) -> bool:
    return (root / plan.dataset / plan.marker_name).exists() and all(
        (root / rel).exists() for rel in plan.members.values()
    )


def wav_to_flac(data: bytes, dest: Path) -> None:
    """Lossless: integer samples are copied as-is into FLAC of the same bit depth."""
    info = sf.info(io.BytesIO(data))
    if info.subtype not in _FLAC_SUBTYPES:
        raise ValueError(f"{dest.name}: cannot store {info.subtype} losslessly as FLAC")
    x, sr = sf.read(io.BytesIO(data), dtype="int32", always_2d=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    sf.write(str(tmp), x, sr, format="FLAC", subtype=info.subtype)
    tmp.replace(dest)


def fetch_archive(remote: Remote, plan: ArchivePlan, root: Path, log=print) -> None:
    """Download (resumable), extract the planned members, delete the archive, mark done."""
    if archive_done(root, plan):
        log(f"  {plan.dataset}/{plan.file.key}: done, skipping")
        return
    ddir = root / plan.dataset / "_download"
    archive = ddir / plan.file.key
    if not archive.exists():
        free = shutil.disk_usage(root).free
        need = plan.file.size + plan.extracted_bytes + DISK_MARGIN
        if free < need:
            raise OSError(f"{plan.file.key}: {need / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free")
        log(f"  {plan.dataset}/{plan.file.key}: downloading {plan.file.size / 1e6:.0f} MB")
        remote.download(plan.file, archive)
    log(f"  {plan.dataset}/{plan.file.key}: extracting {len(plan.members)} members")
    with zipfile.ZipFile(archive) as zf:
        for member, rel in plan.members.items():
            dest = root / rel
            if not dest.exists():
                wav_to_flac(zf.read(member), dest)
    archive.unlink()
    (root / plan.dataset / plan.marker_name).write_text(
        json.dumps({"archive": plan.file.key, "members": len(plan.members)}) + "\n",
        encoding="utf-8",
    )
    try:
        ddir.rmdir()
    except OSError:
        pass


def build_items(plan: DatasetPlan, root: Path) -> list[TestRealItem]:
    hashes: dict[str, str] = {}
    rates: dict[str, int] = {}

    def h(rel: str) -> str:
        if rel not in hashes:
            p = root / rel
            hashes[rel] = sha256_file(p)
            rates[rel] = int(sf.info(str(p)).samplerate)
        return hashes[rel]

    out = []
    for kw in plan.items:
        sd, sw = h(kw["dry_path"]), h(kw["wet_path"])
        sr_dry, sr_wet = rates[kw["dry_path"]], rates[kw["wet_path"]]
        if sr_dry != sr_wet:
            raise ValueError(f"{kw['item_id']}: dry {sr_dry} Hz vs wet {sr_wet} Hz")
        out.append(TestRealItem(**kw, sample_rate=sr_wet, sha256_dry=sd, sha256_wet=sw))
    return out


# --- Entry point ------------------------------------------------------------------------------


def _kb_pots(cfg) -> dict[str, list[str]]:
    from lstmabar.physics.kb import load_kb

    return {k: list(p.pots) for k, p in load_kb(cfg.paths.pedals).items()}


def make_plans(cfg, remote: Remote) -> list[DatasetPlan]:
    tr = cfg.test_real
    if tr.tonetwist.enabled:
        raise NotImplementedError(
            "ToneTwist AFx ingest is not implemented: internal-only numbers, never republish "
            "audio (decision 8); set test_real.tonetwist.enabled=false"
        )
    plans = []
    if tr.egfxset.enabled:
        plans.append(plan_egfxset(cfg, remote, _kb_pots(cfg)))
    if tr.podset.enabled:
        plans.append(plan_podset(cfg, remote))
    return plans


def disk_estimate(plans: Iterable[DatasetPlan]) -> dict[str, float]:
    plans = list(plans)
    archives = [a for p in plans for a in p.archives]
    wav = sum(a.extracted_bytes for a in archives)
    flac = sum(a.extracted_bytes * FLAC_RATIO.get(a.dataset, 1.0) for a in archives)
    peak = max((a.file.size + a.extracted_bytes for a in archives), default=0)
    return {
        "download_gb": sum(a.file.size for a in archives) / 1e9,
        "extracted_wav_gb": wav / 1e9,
        "final_flac_gb_est": flac / 1e9,
        "peak_transient_gb": peak / 1e9,
    }


def print_plan(plans: list[DatasetPlan], root: Path, out=None) -> None:
    out = out if out is not None else sys.stdout
    for p in plans:
        for line in p.report:
            print(line, file=out)
        for a in p.archives:
            state = "done" if archive_done(root, a) else "todo"
            print(
                f"    [{state}] {a.file.key}: archive {a.file.size / 1e6:.0f} MB, "
                f"{len(a.members)} members, {a.extracted_bytes / 1e6:.0f} MB as WAV",
                file=out,
            )
        print(f"  {p.dataset}: {len(p.items)} items", file=out)
    est = disk_estimate(plans)
    print(
        "Disk: download {download_gb:.2f} GB (one archive at a time); extracted "
        "{extracted_wav_gb:.2f} GB as WAV, ~{final_flac_gb_est:.2f} GB as FLAC (kept); "
        "peak transient (archive + its extraction) {peak_transient_gb:.2f} GB".format(**est),
        file=out,
    )
    existing = next(q for q in (root, *root.parents) if q.exists())
    print(f"Free disk: {shutil.disk_usage(existing).free / 1e9:.1f} GB", file=out)


def write_manifest(path: Path, items: list[TestRealItem], datasets: Iterable[str]) -> int:
    """Replace the given datasets' items in the manifest; keep the others."""
    datasets = set(datasets)
    keep = []
    if path.exists():
        keep = [r for r in read_jsonl(path, TestRealItem) if r.dataset not in datasets]
    allr = sorted([*keep, *items], key=lambda r: r.item_id)
    ids = [r.item_id for r in allr]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item ids in the test_real manifest")
    return write_jsonl(path, allr)


def run(cfg, args, remote: Remote | None = None) -> int:
    remote = remote if remote is not None else Zenodo()
    layout = DataLayout.from_config(cfg)
    root = layout.test_real
    plans = make_plans(cfg, remote)
    if getattr(args, "list", False):
        print_plan(plans, root)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    print_plan(plans, root)
    items: list[TestRealItem] = []
    for p in plans:
        for a in p.archives:  # one at a time (R9)
            fetch_archive(remote, a, root)
        items.extend(build_items(p, root))
    n = write_manifest(layout.test_real_manifest, items, [p.dataset for p in plans])
    print(f"Wrote {len(items)} items ({n} in manifest) to {layout.test_real_manifest.name}")
    return 0


__all__ = [
    "ArchivePlan",
    "DatasetPlan",
    "EgfxNote",
    "PodFile",
    "RemoteFile",
    "RemoteRecord",
    "Zenodo",
    "build_items",
    "disk_estimate",
    "fetch_archive",
    "knob_rotation",
    "make_plans",
    "map_settings",
    "pair_egfxset",
    "parse_egfx_name",
    "parse_egfx_settings",
    "parse_pod_name",
    "plan_egfxset",
    "plan_podset",
    "run",
    "select_podset_pedals",
    "wav_to_flac",
    "write_manifest",
]
