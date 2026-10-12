"""Task 4.1 (wave 1a): Guitar-TECHS DI ingest -> onset-aligned dry clips + manifest.

Contract: ``run(cfg, args)`` lists the Zenodo record's files (R9), extracts the DI tracks
only into ``data.sources_dir``, resamples to ``audio.sample_rate``, cuts ``audio.clip_seconds``
clips starting ``ingest.pre_onset_ms`` before detected onsets (skipping clips quieter than
``ingest.min_rms_dbfs``), stores each at ``ingest.dry_peak_dbfs`` (:mod:`lstmabar.data.levels`)
under :class:`DataLayout` and writes ``DryClip`` records (splits assigned once over the full
clip list by :func:`lstmabar.data.splits.assign_splits`) to ``DataLayout.dry_manifest``.

**Dataset layout** (Zenodo record 14963133, CC BY 4.0; inspected through the zip central
directories): one zip per player and content type, ``P1_{chords,scales,singlenotes,techniques}``,
``P2_{...}`` and ``P3_music`` (player 3 recorded only the 12 musical excerpts). Each zip holds
``<P>_<content>/audio/{directinput,micamp}/*.wav``, ``video/{ego,exo}/*.mp3``, ``midi/`` and
macOS junk (``__MACOSX/``, ``.DS_Store``). DI members are
``<P>_<content>/audio/directinput/directinput_<take>.wav``: 24-bit PCM, 48 kHz except P1
Bendings/Vibrato (2-channel, 44.1 kHz); otherwise mono for P1 and dual-mono stereo (identical
channels, checked on sample takes) for P2/P3. :func:`di_to_mono` drops a channel more than 40 dB
below the other and otherwise averages the channels (non-identical channels are averaged too).

**Download (R9), without archives on disk:** Zenodo serves HTTP range requests, so
:class:`RangeFile` exposes a remote zip as a seekable file and the stdlib :mod:`zipfile` reads
its central directory and streams the DI members only (2.0 of 3.8 GB compressed). The transient
disk peak is one ``.part`` member instead of one archive; members already extracted at the right
size are skipped, so the step is resumable. Member names are validated (relative POSIX, a strict
take pattern) and every destination must resolve inside ``sources_dir/guitar_techs``.
``--list`` prints the plan (files, sizes, DI hours per performer, disk estimate) and
downloads nothing. A ``--limit`` /
``ingest.max_clips`` run writes ``dry_clips.limit<N>.jsonl``, never the full manifest.

``--fixtures`` runs the same code path offline against a synthetic Guitar-TECHS-shaped zip
under ``<data.cache_dir>/fixtures`` (fixture performers, nothing touches the real dataset dirs).
"""

from __future__ import annotations

import http.client
import io
import json
import math
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from lstmabar.audio import resample
from lstmabar.data.layout import DataLayout
from lstmabar.data.levels import normalize_peak, sha256_file, write_flac
from lstmabar.data.schema import DryClip, check_relpath, write_jsonl
from lstmabar.data.splits import assign_splits

SOURCE = "guitar_techs"
LICENSE = "CC BY 4.0"
ZENODO_RECORD = "14963133"
ZENODO_API = "https://zenodo.org/api/records/{record}"
ZENODO_PREFIX = "https://zenodo.org/"
USER_AGENT = "lstmabar-ingest (+https://github.com/shanthg01/LSTMABAR)"

Fetch = Callable[[int, int], bytes]
"""``fetch(start, end)`` returns bytes ``start..end`` inclusive (an HTTP ``Range``)."""

# ``<P>_<content>/audio/directinput/directinput_<take>.wav``
DI_MEMBER = re.compile(
    r"^(?P<performer>[A-Za-z]+\d+)_(?P<content>[A-Za-z]+)/audio/directinput/"
    r"directinput_(?P<take>[A-Za-z0-9][A-Za-z0-9_.-]*)\.wav$"
)


# --------------------------------------------------------------------------------------------
# Remote zips through HTTP range requests
# --------------------------------------------------------------------------------------------


def http_fetcher(url: str, retries: int = 6, timeout: float = 60.0) -> Fetch:
    """Range fetcher for ``url``; retries rate limits (429, honouring ``Retry-After``), 5xx and
    network errors with backoff. Refuses servers that ignore ``Range`` (status 200)."""

    def fetch(start: int, end: int) -> bytes:
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT}
        )
        delay = 2.0
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    if r.status != 206:
                        raise RuntimeError(f"server ignored the Range header (HTTP {r.status})")
                    data = r.read()
                if len(data) != end - start + 1:
                    raise OSError(f"short read: {len(data)} of {end - start + 1} bytes")
                return data
            except urllib.error.HTTPError as e:
                if (e.code != 429 and e.code < 500) or attempt == retries:
                    raise
                wait = _retry_after(e.headers.get("Retry-After") if e.headers else None, delay)
            except (OSError, http.client.HTTPException):
                # URLError, timeouts, resets, short reads, IncompleteRead mid-body
                if attempt == retries:
                    raise
                wait = delay
            time.sleep(min(wait, 120.0))
            delay *= 2
        raise AssertionError("unreachable")

    return fetch


def _retry_after(value, default: float) -> float:
    """Seconds from a ``Retry-After`` header (delta-seconds form); ``default`` otherwise."""
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


def file_fetcher(path: str | Path) -> Fetch:
    """Range fetcher over a local file (tests and ``--fixtures``)."""
    path = Path(path)

    def fetch(start: int, end: int) -> bytes:
        with path.open("rb") as f:
            f.seek(start)
            return f.read(end - start + 1)

    return fetch


class RangeFile(io.RawIOBase):
    """A read-only, seekable file of known ``size`` whose bytes come from ``fetch``.

    Wrap it in :class:`io.BufferedReader` (see :func:`open_remote_zip`) so small reads are
    batched into one request per buffer.
    """

    def __init__(self, fetch: Fetch, size: int):
        super().__init__()
        self._fetch, self._size, self._pos = fetch, int(size), 0
        self.requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        pos = base + offset
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return pos

    def readinto(self, b) -> int:
        n = min(len(b), self._size - self._pos)
        if n <= 0:
            return 0
        data = self._fetch(self._pos, self._pos + n - 1)
        self.requests += 1
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)


def open_remote_zip(fetch: Fetch, size: int, buffer_size: int = 1 << 16) -> zipfile.ZipFile:
    """A :class:`zipfile.ZipFile` over a range-fetched file. Opening it reads only the end of
    central directory and the central directory; members are read on demand."""
    return zipfile.ZipFile(io.BufferedReader(RangeFile(fetch, size), buffer_size))


# --------------------------------------------------------------------------------------------
# Guitar-TECHS naming
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DIMember:
    archive: str  # zip file name on Zenodo
    name: str  # member path inside the zip (also its path under sources_dir/guitar_techs)
    performer: str  # "P1" | "P2" | "P3"
    content: str  # chords | scales | singlenotes | techniques | music
    take: str  # e.g. "Set1_m7", "PalmMute", "07"
    file_size: int
    compress_size: int

    @property
    def recording_id(self) -> str:
        return f"{self.performer}_{self.content}_{self.take}"

    @property
    def technique(self) -> str:
        """The content type, or the technique name for the ``techniques`` zips."""
        return self.take.lower() if self.content.lower() == "techniques" else self.content.lower()

    @property
    def source_file(self) -> str:
        """Path relative to ``data.sources_dir`` (POSIX)."""
        return f"{SOURCE}/{self.name}"


def parse_di_member(name: str) -> dict[str, str] | None:
    """``{performer, content, take}`` for a DI wav member, else ``None`` (micamp, video, MIDI,
    ``__MACOSX`` resource forks, ``.DS_Store``) and for unsafe names (absolute, drive letter,
    backslashes, ``..``)."""
    try:
        check_relpath(name, "zip member")
    except ValueError:
        return None
    if name.startswith("__MACOSX/") or "/." in name:
        return None
    m = DI_MEMBER.match(name)
    return m.groupdict() if m else None


def di_members(archive: str, zf: zipfile.ZipFile) -> list[DIMember]:
    out = []
    for info in zf.infolist():
        p = parse_di_member(info.filename)
        if p is not None:
            out.append(
                DIMember(
                    archive,
                    info.filename,
                    **p,
                    file_size=info.file_size,
                    compress_size=info.compress_size,
                )
            )
    return sorted(out, key=lambda m: m.name)


def clip_id_for(member: DIMember, offset_s: float) -> str:
    """Stable clip id ``gt-<performer>-<content>-<take>-<offset ms>`` (lowercase, [a-z0-9_-])."""
    take = re.sub(r"[^a-z0-9_]+", "_", member.take.lower())
    ms = int(round(offset_s * 1000))
    return f"gt-{member.performer.lower()}-{member.content.lower()}-{take}-{ms:07d}"


# --------------------------------------------------------------------------------------------
# Archive listing and DI extraction
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Archive:
    key: str
    size: int
    fetch: Fetch


def zenodo_archives(record: str = ZENODO_RECORD) -> list[Archive]:
    """The record's zip files (smallest first, so ``--limit`` runs reach every player soon)."""
    req = urllib.request.Request(
        ZENODO_API.format(record=record), headers={"User-Agent": USER_AGENT}
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                meta = json.load(r)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
            time.sleep(2.0 * 2**attempt)
    files = [f for f in meta["files"] if f["key"].endswith(".zip")]
    for f in files:
        if not str(f["links"]["self"]).startswith(ZENODO_PREFIX):
            raise ValueError(f"unexpected download host for {f['key']!r}")
    return [
        Archive(f["key"], int(f["size"]), http_fetcher(f["links"]["self"]))
        for f in sorted(files, key=lambda f: (int(f["size"]), f["key"]))
    ]


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def wav_header_info(head: bytes) -> tuple[int, int, int, int | None]:
    """``(channels, sample_rate, bits, data_bytes)`` from the first bytes of a RIFF/WAVE file
    (``data_bytes`` is ``None`` if the data chunk header is not within ``head``)."""
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE header")
    i, fmt = 12, None
    while i + 8 <= len(head):
        cid, size = head[i : i + 4], struct.unpack("<I", head[i + 4 : i + 8])[0]
        if cid == b"fmt ":
            _, ch, sr, _, _, bits = struct.unpack("<HHIIHH", head[i + 8 : i + 24])
            fmt = (ch, sr, bits)
        elif cid == b"data":
            if fmt is None:
                break
            return (*fmt, size)
        i += 8 + size + (size & 1)
    if fmt is None:
        raise ValueError("no fmt chunk in the WAV header")
    return (*fmt, None)


def extract_member(
    zf: zipfile.ZipFile,
    member: DIMember,
    dest: Path,
    chunk: int = 1 << 20,
    root: Path | None = None,
) -> bool:
    """Stream one member to ``dest`` (via ``dest.part``; zipfile checks the CRC). Returns
    ``False`` if ``dest`` already exists at the right size (resume). With ``root``, ``dest``
    must resolve inside it (defence in depth against path escapes from member names)."""
    if root is not None and not dest.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"refusing to extract {member.name!r} outside the sources dir")
    if dest.exists() and dest.stat().st_size == member.file_size:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    done, last = 0, -1
    with zf.open(member.name) as src, part.open("wb") as out:
        while block := src.read(chunk):
            out.write(block)
            done += len(block)
            pct = int(100 * done / max(member.file_size, 1)) // 10 * 10
            if pct != last:
                _progress(f"    {member.name}: {pct}% of {_fmt_bytes(member.file_size)}")
                last = pct
    _progress("", end=True)
    os.replace(part, dest)
    return True


def _progress(msg: str, end: bool = False) -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\r" + msg.ljust(100)[:100] + ("\n" if end else ""))
        sys.stdout.flush()


# --------------------------------------------------------------------------------------------
# Onsets and clip cutting
# --------------------------------------------------------------------------------------------


def onset_strength(
    x: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512, block: int = 2048
) -> np.ndarray:
    """Log-magnitude spectral flux per frame (frame ``i`` is centred at ``i * hop``).

    The signal is peak-normalized first, so the envelope does not depend on the recording
    level. Frames are processed in blocks to bound memory on long takes.
    """
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x), initial=0.0))
    if peak <= 0 or x.size == 0:
        return np.zeros(max(1, 1 + x.size // hop), dtype=np.float32)
    x = np.pad(x / peak, (n_fft // 2, n_fft // 2))
    n_frames = 1 + (x.size - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    flux = np.zeros(n_frames, dtype=np.float32)
    prev = None
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    for s in range(0, n_frames, block):
        spec = np.log1p(100.0 * np.abs(np.fft.rfft(frames[s : s + block] * win, axis=1)))
        if prev is not None:
            spec_prev = np.vstack([prev[None], spec[:-1]])
        else:
            spec_prev = np.vstack([spec[:1], spec[:-1]])
        flux[s : s + len(spec)] = np.maximum(spec - spec_prev, 0.0).sum(axis=1)
        prev = spec[-1]
    return flux


def detect_onsets(
    x: np.ndarray,
    sr: int,
    hop: int = 512,
    delta: float = 0.07,
    wait_s: float = 0.05,
    floor_db: float = -60.0,
) -> np.ndarray:
    """Onset times (seconds) by peak-picking the spectral flux (librosa-style ``peak_pick``).

    A frame is an onset when it is the local maximum within ±30 ms, exceeds the local mean over
    -100/+70 ms by ``delta`` (envelope normalized to max 1), is at least ``wait_s`` after the
    previous onset, and the signal around it is no quieter than ``floor_db`` re the file peak
    (no onsets in the noise floor).
    """
    x = np.asarray(x, dtype=np.float32)
    env = onset_strength(x, sr, hop=hop)
    if env.max(initial=0.0) <= 0:
        return np.zeros(0)
    env = env / env.max()

    def frames(sec: float) -> int:
        return max(1, int(round(sec * sr / hop)))

    pre_max, post_max = frames(0.03), frames(0.03)
    pre_avg, post_avg = frames(0.10), frames(0.07)
    local_max = maximum_filter1d(env, size=pre_max + post_max + 1, origin=(pre_max - post_max) // 2)
    local_avg = uniform_filter1d(env, size=pre_avg + post_avg + 1, origin=(pre_avg - post_avg) // 2)
    cand = np.flatnonzero((env >= local_max) & (env >= local_avg + delta))
    # Level gate: frame RMS re the file peak.
    peak = float(np.max(np.abs(x), initial=0.0))
    out: list[int] = []
    wait = frames(wait_s)
    for i in cand:
        if out and i - out[-1] < wait:
            continue
        c = i * hop
        seg = x[max(0, c - hop) : c + 4 * hop]
        if seg.size == 0:
            continue
        lvl = 20 * math.log10(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) / peak + 1e-12)
        if lvl < floor_db:
            continue
        out.append(int(i))
    return np.asarray(out, dtype=np.float64) * hop / sr


def rms_dbfs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    r = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    return 20 * math.log10(r) if r > 0 else -200.0


def cut_clips(
    x: np.ndarray,
    sr: int,
    onsets_s: Iterable[float],
    clip_s: float,
    pre_onset_s: float,
    min_rms_dbfs: float,
) -> list[int]:
    """Start samples of non-overlapping clips, each ``pre_onset_s`` before an onset.

    Greedy in time order: an onset is used if its clip fits in the signal, does not overlap
    the previous clip and is not quieter than ``min_rms_dbfs`` (RMS at the source level); a
    skipped onset does not block the next one.
    """
    n = int(round(clip_s * sr))
    pre = int(round(pre_onset_s * sr))
    starts: list[int] = []
    free_from = 0
    for t in sorted(onsets_s):
        s = max(0, int(round(t * sr)) - pre)
        if s < free_from or s + n > len(x):
            continue
        if rms_dbfs(x[s : s + n]) < min_rms_dbfs:
            continue
        starts.append(s)
        free_from = s + n
    return starts


def di_to_mono(data: np.ndarray) -> np.ndarray:
    """Mono DI from a ``(T,)`` or ``(T, C)`` array.

    Guitar-TECHS P2/P3 DI files are stereo: a channel that is silent (> 40 dB below the other)
    is dropped, otherwise the channels are averaged (identical channels average to themselves).
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        return data
    if data.shape[1] == 1:
        return data[:, 0]
    r = np.sqrt(np.mean(data.astype(np.float64) ** 2, axis=0))
    loud = float(r.max())
    keep = r >= loud * 10 ** (-40 / 20) if loud > 0 else np.ones_like(r, dtype=bool)
    return data[:, keep].mean(axis=1).astype(np.float32)


def clips_from_file(
    path: Path,
    member: DIMember,
    layout: DataLayout,
    cfg,
    limit: int | None = None,
) -> list[DryClip]:
    """Cut, normalize and write the dry clips of one extracted DI take."""
    data, src_sr = sf.read(str(path), dtype="float32", always_2d=True)
    sr = int(cfg.audio.sample_rate)
    x = resample(di_to_mono(data), int(src_sr), sr)
    del data
    clip_s = float(cfg.audio.clip_seconds)
    starts = cut_clips(
        x,
        sr,
        detect_onsets(x, sr),
        clip_s,
        float(cfg.ingest.pre_onset_ms) / 1000.0,
        float(cfg.ingest.min_rms_dbfs),
    )
    if limit is not None:
        starts = starts[: max(0, limit)]
    n = int(round(clip_s * sr))
    clips = []
    for s in starts:
        offset_s = s / sr
        clip_id = clip_id_for(member, offset_s)
        y, gain_db = normalize_peak(x[s : s + n], float(cfg.ingest.dry_peak_dbfs))
        rel = layout.dry_relpath(clip_id)
        out = layout.resolve_rendered(rel)
        write_flac(out, y, sr, str(cfg.ingest.dry_subtype))
        clips.append(
            DryClip(
                clip_id=clip_id,
                source=SOURCE,
                performer=member.performer,
                recording_id=member.recording_id,
                technique=member.technique,
                source_file=member.source_file,
                offset_s=round(offset_s, 6),
                duration_s=clip_s,
                sample_rate=sr,
                source_gain_db=round(float(gain_db), 4),
                path=rel,
                sha256=sha256_file(out),
                split="",
                license=LICENSE,
            )
        )
    return clips


# --------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------


def assign_and_check_splits(clips: list[DryClip], cfg) -> list[DryClip]:
    """Assign splits once over the whole list; warn when the held-out performer has a single
    recording (val or test would then be empty)."""
    heldout = cfg.ingest.heldout_performer
    recs = {c.recording_id for c in clips if c.performer == heldout}
    if heldout is not None and len(recs) < 2:
        print(
            f"warning: held-out performer {heldout!r} has {len(recs)} recording(s) in this run; "
            "val/test cannot both be non-empty",
            file=sys.stderr,
        )
    splits = assign_splits(((c.source, c.performer, c.recording_id) for c in clips), cfg)
    return [replace(c, split=s) for c, s in zip(clips, splits, strict=True)]


def _check_performers(cfg) -> None:
    if not list(cfg.ingest.train_performers) or cfg.ingest.heldout_performer is None:
        raise ValueError("ingest.train_performers / ingest.heldout_performer are not set")


def list_plan(archives: list[Archive], cfg, read_headers: bool = True) -> dict:
    """Print what ingest would fetch (DI members per archive, sizes, hours, disk estimate)."""
    sr = int(cfg.audio.sample_rate)
    clip_s = float(cfg.audio.clip_seconds)
    hours: dict[str, float] = defaultdict(float)
    di_bytes = di_compressed = total = 0
    n_members = 0
    for a in archives:
        zf = open_remote_zip(a.fetch, a.size)
        members = di_members(a.key, zf)
        sz = sum(m.file_size for m in members)
        di_bytes += sz
        di_compressed += sum(m.compress_size for m in members)
        total += a.size
        n_members += len(members)
        print(f"{a.key}: archive {_fmt_bytes(a.size)}, {len(members)} DI takes, {_fmt_bytes(sz)}")
        for m in members:
            line = f"  {m.name}  {_fmt_bytes(m.file_size)}"
            if read_headers:
                with zf.open(m.name) as f:
                    ch, wsr, bits, nbytes = wav_header_info(f.read(4096))
                secs = (nbytes or m.file_size) / (ch * bits // 8) / wsr
                hours[m.performer] += secs / 3600
                line += f"  {ch} ch {wsr} Hz {bits}-bit {secs:.0f} s"
            print(line)
    n_clips_max = {p: int(h * 3600 // clip_s) for p, h in hours.items()}
    # 16-bit FLAC of guitar DI compresses to roughly 60 % of PCM.
    dry_bytes = sum(n_clips_max.values()) * clip_s * sr * 2 * 0.6
    print(
        f"\n{n_members} DI takes in {len(archives)} archives ({_fmt_bytes(total)} in total); "
        f"DI download {_fmt_bytes(di_compressed)}, extracted {_fmt_bytes(di_bytes)}"
    )
    for p in sorted(hours):
        print(f"  {p}: {hours[p]:.2f} h DI, <= {n_clips_max[p]} clips of {clip_s:g} s")
    if hours:
        print(f"  dry clips (upper bound): ~{_fmt_bytes(dry_bytes)}")
    return {
        "archives": len(archives),
        "di_takes": n_members,
        "di_bytes": di_bytes,
        "hours": dict(hours),
        "max_clips": n_clips_max,
    }


def manifest_path(layout: DataLayout, limit: int | None) -> Path:
    """The full ``dry_clips.jsonl``, or ``dry_clips.limit<N>.jsonl`` for a limited run, so a
    smoke run never overwrites the committed full manifest."""
    if limit is None:
        return layout.dry_manifest
    return layout.dry_manifest.with_name(f"dry_clips.limit{int(limit)}.jsonl")


def ingest(archives: list[Archive], cfg, layout: DataLayout, limit: int | None = None) -> list:
    """Extract DI takes, cut clips, assign splits, write the manifest. Returns the clips."""
    _check_performers(cfg)
    out_root = layout.sources / SOURCE
    clips: list[DryClip] = []
    for ai, a in enumerate(archives, 1):
        if limit is not None and len(clips) >= limit:
            break
        # Large buffer for member streaming: one request per 4 MB.
        zf = open_remote_zip(a.fetch, a.size, buffer_size=4 << 20)
        members = di_members(a.key, zf)
        print(f"[{ai}/{len(archives)}] {a.key}: {len(members)} DI takes", flush=True)
        for m in members:
            if limit is not None and len(clips) >= limit:
                break
            dest = out_root / m.name
            fetched = extract_member(zf, m, dest, root=out_root)
            left = None if limit is None else limit - len(clips)
            new = clips_from_file(dest, m, layout, cfg, limit=left)
            clips.extend(new)
            how = "downloaded" if fetched else "cached"
            print(f"  {m.recording_id}: {len(new)} clips ({how}, total {len(clips)})", flush=True)
    clips = assign_and_check_splits(clips, cfg)
    ids = [c.clip_id for c in clips]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate clip ids: {dup[:5]}")
    manifest = manifest_path(layout, limit)
    if limit is not None:
        print(
            f"warning: limited run (--limit / ingest.max_clips = {limit}); writing "
            f"{manifest.name}, not the full manifest {layout.dry_manifest.name}",
            file=sys.stderr,
        )
    write_jsonl(manifest, clips)
    counts: dict[str, int] = defaultdict(int)
    for c in clips:
        counts[c.split] += 1
    print(f"wrote {len(clips)} clips to {manifest.name}: {dict(sorted(counts.items()))}")
    return clips


# --------------------------------------------------------------------------------------------
# Synthetic fixture source (offline smoke runs)
# --------------------------------------------------------------------------------------------


def make_fixture_zip(path: str | Path, performers: Iterable[str], seconds: float = 8.0) -> Path:
    """A small zip in the Guitar-TECHS layout: per performer two DI takes (stereo 24-bit WAV at
    48 kHz, plucks separated by silence), plus a mic'd-amp take and macOS junk that ingest must
    ignore."""
    from lstmabar.audio import synth_riff

    sr = 48000
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p_idx, perf in enumerate(performers):
            for k, kind in enumerate(("single_notes", "power_chords")):
                riff = synth_riff(kind, seconds / 2, sr, seed=10 * p_idx + k)
                gap = np.zeros(int(sr * 0.5), dtype=np.float32)
                x = 0.3 * np.concatenate([riff, gap, riff[: len(riff) // 2], gap])
                buf = io.BytesIO()
                sf.write(buf, np.stack([x, x], axis=1), sr, format="WAV", subtype="PCM_24")
                base = f"{perf}_fixture/audio"
                zf.writestr(f"{base}/directinput/directinput_take{k}.wav", buf.getvalue())
                if k == 0:
                    zf.writestr(f"{base}/micamp/micamp_take{k}.wav", buf.getvalue())
                    zf.writestr(f"__MACOSX/{base}/directinput/._directinput_take{k}.wav", b"x")
                    zf.writestr(f"{perf}_fixture/.DS_Store", b"x")
    return path


def _fixture_setup(cfg, args) -> tuple:
    from lstmabar.data.fixtures import FIXTURE_PERFORMERS, fixture_config

    root = Path(str(cfg.data.cache_dir)) / "fixtures"
    fcfg = fixture_config(root, list(getattr(args, "overrides", None) or []))
    zpath = make_fixture_zip(root / "remote" / "fixture_di.zip", FIXTURE_PERFORMERS)
    archives = [Archive(zpath.name, zpath.stat().st_size, file_fetcher(zpath))]
    return fcfg, archives


def run(cfg, args) -> int:
    if getattr(args, "fixtures", False):
        cfg, archives = _fixture_setup(cfg, args)
    else:
        archives = zenodo_archives()
    if getattr(args, "list", False):
        list_plan(archives, cfg)
        return 0
    limit = getattr(args, "limit", None)
    if limit is None and cfg.ingest.get("max_clips") is not None:
        limit = int(cfg.ingest.max_clips)
    layout = DataLayout.from_config(cfg)
    ingest(archives, cfg, layout, limit=limit)
    return 0


__all__ = [
    "LICENSE",
    "SOURCE",
    "ZENODO_RECORD",
    "Archive",
    "DIMember",
    "RangeFile",
    "assign_and_check_splits",
    "clip_id_for",
    "clips_from_file",
    "cut_clips",
    "detect_onsets",
    "di_members",
    "di_to_mono",
    "extract_member",
    "file_fetcher",
    "http_fetcher",
    "ingest",
    "list_plan",
    "manifest_path",
    "make_fixture_zip",
    "onset_strength",
    "open_remote_zip",
    "parse_di_member",
    "rms_dbfs",
    "run",
    "wav_header_info",
    "zenodo_archives",
]
