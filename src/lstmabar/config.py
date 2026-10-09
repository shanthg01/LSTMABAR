"""YAML config loading with ``defaults`` composition and dotlist overrides.

A config file may list other config files (relative to itself) under ``defaults``; they are
merged in order, then the file's own keys, then command-line overrides such as
``train.lr=1e-4``.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    cfg = _load_with_defaults(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def _load_with_defaults(path: Path) -> DictConfig:
    raw = OmegaConf.load(path)
    defaults = raw.pop("defaults", []) or []
    merged = OmegaConf.create()
    for d in defaults:
        merged = OmegaConf.merge(merged, _load_with_defaults(path.parent / d))
    return OmegaConf.merge(merged, raw)
