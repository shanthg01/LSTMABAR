"""On-disk layout of dataset v1 (P4 contract).

::

    data/sources/<dataset>/...                      raw / extracted downloads (gitignored)
    data/rendered/<version>/dry/<clip_id>.flac      dry clips (gitignored)
    data/rendered/<version>/wet/shard-00000/<item_id>.flac
    data/rendered/<version>/wet/shard-00000/items.jsonl      RenderItem per line
    data/rendered/<version>/wet/shard-00000/captions.jsonl   Caption per line
    data/rendered/<version>/wet/shard-00000/DONE             written last (resumability)
    data/cache/f0/<clip_id>.npz                     pYIN tracks of dry clips
    data/cache/llm/                                 caption cache + spend ledger.jsonl
    data/manifests/<version>/dry_clips.jsonl        DryClip per line (committed)
    data/manifests/<version>/index.json             counts, shard hashes, dataset hash (committed)
    data/manifests/<version>/files.sha256           "<sha256>  <relpath>" per file (committed)
    data/manifests/test_real.jsonl                  TestRealItem per line (committed)
    data/calibration/<pedal>.json                   white-box label tables (committed, R4)

Records store paths relative to these roots (``DryClip.path`` / ``RenderItem.wet_path``
relative to ``data.rendered_dir``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DONE_MARKER = "DONE"


@dataclass(frozen=True)
class DataLayout:
    sources: Path
    rendered: Path
    cache: Path
    test_real: Path
    manifests: Path
    calibration: Path
    version: str = "v1"

    @classmethod
    def from_config(cls, cfg, root: str | Path | None = None) -> DataLayout:
        """From ``cfg.data``; relative dirs resolve against ``root`` (default: the CWD)."""
        base = Path(root) if root is not None else Path.cwd()
        d = cfg.data

        def p(key: str) -> Path:
            v = Path(str(d[key]))
            return v if v.is_absolute() else base / v

        return cls(
            sources=p("sources_dir"),
            rendered=p("rendered_dir"),
            cache=p("cache_dir"),
            test_real=p("test_real_dir"),
            manifests=p("manifests_dir"),
            calibration=p("calibration_dir"),
            version=str(d.version),
        )

    # --- dry clips ---
    @property
    def dry_dir(self) -> Path:
        return self.rendered / self.version / "dry"

    def dry_relpath(self, clip_id: str) -> str:
        return f"{self.version}/dry/{clip_id}.flac"

    @property
    def dry_manifest(self) -> Path:
        return self.manifests / self.version / "dry_clips.jsonl"

    # --- wet shards ---
    @property
    def wet_dir(self) -> Path:
        return self.rendered / self.version / "wet"

    @staticmethod
    def shard_name(index: int) -> str:
        return f"shard-{index:05d}"

    def shard_dir(self, index: int) -> Path:
        return self.wet_dir / self.shard_name(index)

    def wet_relpath(self, shard: int, item_id: str) -> str:
        return f"{self.version}/wet/{self.shard_name(shard)}/{item_id}.flac"

    def shard_done(self, index: int) -> bool:
        return (self.shard_dir(index) / DONE_MARKER).exists()

    # --- cache ---
    def f0_cache(self, clip_id: str) -> Path:
        return self.cache / "f0" / f"{clip_id}.npz"

    @property
    def llm_cache(self) -> Path:
        return self.cache / "llm"

    @property
    def ledger(self) -> Path:
        return self.llm_cache / "ledger.jsonl"

    # --- committed artefacts ---
    @property
    def index(self) -> Path:
        return self.manifests / self.version / "index.json"

    @property
    def file_hashes(self) -> Path:
        return self.manifests / self.version / "files.sha256"

    @property
    def test_real_manifest(self) -> Path:
        return self.manifests / "test_real.jsonl"

    def calibration_table(self, pedal_id: str) -> Path:
        return self.calibration / f"{pedal_id}.json"

    def resolve_rendered(self, relpath: str) -> Path:
        return self.rendered / relpath


__all__ = ["DONE_MARKER", "DataLayout"]
