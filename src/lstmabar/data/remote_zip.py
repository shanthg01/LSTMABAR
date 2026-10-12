"""List and extract members of a remote zip via HTTP range requests (stdlib only).

Only the end-of-central-directory record and the central directory are fetched, so the
contents of a multi-GB archive can be listed without downloading it. Members can also be
fetched one by one (:func:`read_member`), which is how the test-real ingest avoids holding
whole archives when it only needs a few files.

Everything is built on a ``read_range(start, end)`` callable (``end`` exclusive), so the
parser is tested on a local zip without any network (:func:`file_range_reader`).
"""

from __future__ import annotations

import struct
import urllib.request
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ReadRange = Callable[[int, int], bytes]

_EOCD_SIG = b"PK\x05\x06"
_EOCD64_LOC_SIG = b"PK\x06\x07"
_EOCD64_SIG = b"PK\x06\x06"
_CDH_SIG = b"PK\x01\x02"
_LFH_SIG = b"PK\x03\x04"
_EOCD_MIN = 22
_MAX_COMMENT = 0xFFFF
_USER_AGENT = "lstmabar-data/1 (+https://github.com/shanthg01)"


@dataclass(frozen=True)
class ZipEntry:
    name: str
    compressed_size: int
    file_size: int
    header_offset: int
    compress_type: int  # 0 = stored, 8 = deflate
    crc: int

    @property
    def is_dir(self) -> bool:
        return self.name.endswith("/")


def http_range_reader(url: str, timeout: float = 60.0) -> ReadRange:
    def read(start: int, end: int) -> bytes:
        if end <= start:
            return b""
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{end - 1}", "User-Agent": _USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 206:
                raise OSError(f"server ignored the Range header (HTTP {r.status})")
            data = r.read()
        if len(data) != end - start:
            raise OSError(f"short range read: wanted {end - start} bytes, got {len(data)}")
        return data

    return read


def http_size(url: str, timeout: float = 60.0) -> int:
    """Total size of a remote file, from a one-byte range request's Content-Range."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        cr = r.headers.get("Content-Range", "")
        if r.status != 206 or "/" not in cr:
            raise OSError("server does not support range requests")
        return int(cr.rsplit("/", 1)[1])


def file_range_reader(path: str | Path) -> ReadRange:
    p = Path(path)

    def read(start: int, end: int) -> bytes:
        with p.open("rb") as f:
            f.seek(start)
            return f.read(max(0, end - start))

    return read


def _find_eocd(read: ReadRange, size: int) -> tuple[int, bytes]:
    tail_len = min(size, _EOCD_MIN + _MAX_COMMENT)
    tail = read(size - tail_len, size)
    i = tail.rfind(_EOCD_SIG)
    while i >= 0:
        if i + _EOCD_MIN <= len(tail):
            (comment_len,) = struct.unpack_from("<H", tail, i + 20)
            if i + _EOCD_MIN + comment_len == len(tail):
                return size - tail_len + i, tail
        i = tail.rfind(_EOCD_SIG, 0, i)
    raise ValueError("not a zip file (end of central directory not found)")


def _central_directory(read: ReadRange, size: int) -> tuple[int, int, int]:
    """``(cd_offset, cd_size, n_entries)``, following the zip64 locator when present."""
    eocd_pos, tail = _find_eocd(read, size)
    base = size - len(tail)
    rec = tail[eocd_pos - base : eocd_pos - base + _EOCD_MIN]
    _, _, _, _, n_entries, cd_size, cd_offset, _ = struct.unpack("<4sHHHHIIH", rec)
    loc_pos = eocd_pos - 20
    if loc_pos >= 0:
        loc = read(loc_pos, loc_pos + 20)
        if loc[:4] == _EOCD64_LOC_SIG:
            (eocd64_off,) = struct.unpack_from("<Q", loc, 8)
            e64 = read(eocd64_off, eocd64_off + 56)
            if e64[:4] != _EOCD64_SIG:
                raise ValueError("bad zip64 end of central directory")
            n_entries, cd_size, cd_offset = struct.unpack_from("<QQQ", e64, 32)
    return cd_offset, cd_size, n_entries


def _zip64_extra(extra: bytes, sizes: list[int]) -> list[int]:
    """Replace 0xFFFFFFFF fields (file size, compressed size, offset) from the 0x0001 extra."""
    i = 0
    while i + 4 <= len(extra):
        tag, ln = struct.unpack_from("<HH", extra, i)
        if tag == 0x0001:
            j = i + 4
            out = list(sizes)
            for k, v in enumerate(sizes):
                if v == 0xFFFFFFFF and j + 8 <= i + 4 + ln:
                    (out[k],) = struct.unpack_from("<Q", extra, j)
                    j += 8
            return out
        i += 4 + ln
    return sizes


def parse_central_directory(cd: bytes) -> list[ZipEntry]:
    entries: list[ZipEntry] = []
    i = 0
    while i + 46 <= len(cd) and cd[i : i + 4] == _CDH_SIG:
        (
            flags,
            method,
            crc,
            csize,
            usize,
            name_len,
            extra_len,
            comment_len,
            offset,
        ) = struct.unpack_from("<8xH H4xI II HHH8xI", cd, i)
        name_raw = cd[i + 46 : i + 46 + name_len]
        extra = cd[i + 46 + name_len : i + 46 + name_len + extra_len]
        name = name_raw.decode("utf-8" if flags & 0x800 else "cp437")
        usize, csize, offset = _zip64_extra(extra, [usize, csize, offset])
        entries.append(ZipEntry(name, csize, usize, offset, method, crc))
        i += 46 + name_len + extra_len + comment_len
    return entries


def list_entries(read: ReadRange, size: int) -> list[ZipEntry]:
    cd_offset, cd_size, n_entries = _central_directory(read, size)
    entries = parse_central_directory(read(cd_offset, cd_offset + cd_size))
    if len(entries) != n_entries:
        raise ValueError(f"central directory lists {len(entries)} of {n_entries} entries")
    return entries


def list_remote_zip(url: str, size: int | None = None) -> list[ZipEntry]:
    """Entries of a remote zip; two to four small range requests, no full download."""
    return list_entries(http_range_reader(url), size if size is not None else http_size(url))


def read_member(read: ReadRange, entry: ZipEntry) -> bytes:
    """Fetch and decompress one member (stored or deflate), checking its CRC."""
    lfh = read(entry.header_offset, entry.header_offset + 30)
    if lfh[:4] != _LFH_SIG:
        raise ValueError(f"bad local header for {entry.name}")
    name_len, extra_len = struct.unpack_from("<HH", lfh, 26)
    start = entry.header_offset + 30 + name_len + extra_len
    raw = read(start, start + entry.compressed_size)
    if entry.compress_type == 0:
        data = raw
    elif entry.compress_type == 8:
        data = zlib.decompress(raw, -15)
    else:
        raise ValueError(f"{entry.name}: unsupported compression {entry.compress_type}")
    if zlib.crc32(data) & 0xFFFFFFFF != entry.crc:
        raise ValueError(f"{entry.name}: CRC mismatch")
    return data


__all__ = [
    "ZipEntry",
    "file_range_reader",
    "http_range_reader",
    "http_size",
    "list_entries",
    "list_remote_zip",
    "parse_central_directory",
    "read_member",
]
