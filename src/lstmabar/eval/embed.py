"""``lstmabar eval embed`` (task 5.5, D5). Wave 0: signature; wave 1 (b) implements it.

Builds/refreshes ``data/cache/{clap,bge}/<model>/<split>_<kind>.npz`` for every model in
``eval.embeddings`` (``clap_models`` get text + audio + wet, ``text_models`` text only) and
every split (synthetic splits + ``gold``). Incremental: only missing or stale ids (source
sha256 changed) are recomputed.
"""

from __future__ import annotations


def run(cfg, args) -> int:
    raise NotImplementedError


__all__ = ["run"]
