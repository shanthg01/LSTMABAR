"""``lstmabar eval embed`` (task 5.5, D5): build/refresh the embedding caches.

Writes ``<eval.embeddings.dir>/{clap,bge}/<model>/<split>_<kind>.npz`` for every model in
``eval.embeddings`` (``clap_models`` get text + audio + wet, ``text_models`` text only):

- ``text``: every caption of the synthetic splits (id ``caption_id``) and the gold
  instructions (split ``gold``, id ``gold_id``);
- ``audio``: per synthetic item the dry clip at the item's ``input_peak_dbfs`` (id
  ``item_id``); per distinct gold clip the dry clip at ``audio.inference_peak_dbfs`` (id
  :func:`lstmabar.eval.data.gold_audio_key`);
- ``wet``: target renders of ``val`` / ``test`` / ``test_unseen_pedal`` items (id ``item_id``)
  and every gold render, target and distractors (id ``GoldRender.item_id``).

Incremental: each row stores the sha256 of its source (caption text, or the exact float32
array embedded); rows whose sha matches are kept, missing or stale ones are recomputed. The
table is saved every ``eval.embeddings.save_every`` new rows so an interrupted build keeps
its progress. Ids no longer in the source are pruned, except under ``--limit``.

Order: all text first (kNN needs only text), then gold audio + wets, then the val / test /
unseen audio + wets, then train audio. A summary (counts, dims, file and content digests;
aggregate only) is written to ``<eval.report_dir>/embedding_cache.md``.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lstmabar.data.schema import SPLITS
from lstmabar.models.embed_cache import (
    FAMILIES,
    HF_IDS,
    KINDS,
    EmbeddingTable,
    audio_sha256,
    cache_path,
    text_sha256,
)
from lstmabar.models.encoders import load_audio_encoder, load_text_encoder

log = logging.getLogger(__name__)

GOLD = "gold"
ALL_SPLITS: tuple[str, ...] = (*SPLITS, GOLD)
WET_SPLITS: tuple[str, ...] = ("val", "test", "test_unseen_pedal", GOLD)
REPORT_NAME = "embedding_cache.md"
DEFAULT_SAVE_EVERY = 2000
DEFAULT_NUM_THREADS = 6

Row = tuple[str, str, object]  # (id, source sha256, payload: text or float32 array)


@dataclass
class BuildStats:
    model: str
    split: str
    kind: str
    n_rows: int
    n_computed: int
    n_kept: int
    n_pruned: int
    seconds: float

    @property
    def rate(self) -> float:
        return self.n_computed / self.seconds if self.seconds > 0 else 0.0


# ============================================================================ sources


def _synth_split(cfg, split: str, limit: int | None):
    from lstmabar.eval.data import load_split

    return load_split(cfg, split, limit=limit)


def text_rows(cfg, split: str, limit: int | None) -> Iterator[Row]:
    if split == GOLD:
        from lstmabar.eval.data import load_gold_items

        for g in load_gold_items(cfg, limit=limit):
            yield g.gold_id, text_sha256(g.instruction), g.instruction
        return
    for it in _synth_split(cfg, split, limit):
        for c in it.captions:
            yield c.caption_id, text_sha256(c.text), c.text


def audio_rows(cfg, split: str, limit: int | None) -> Iterator[Row]:
    if split == GOLD:
        from lstmabar.eval.data import load_gold_items

        seen: set[str] = set()
        for g in load_gold_items(cfg, limit=limit):
            for q in g.queries():
                if q.audio_key in seen:
                    continue
                seen.add(q.audio_key)
                x = np.asarray(q.load_dry(), dtype=np.float32)
                yield q.audio_key, audio_sha256(x), x
        return
    for it in _synth_split(cfg, split, limit):
        x = np.asarray(it.queries_for_audio().load_dry(), dtype=np.float32)
        yield it.item_id, audio_sha256(x), x


def wet_rows(cfg, split: str, limit: int | None) -> Iterator[Row]:
    if split == GOLD:
        from lstmabar.eval.data import load_gold_items, load_render_audio

        for g in load_gold_items(cfg, limit=limit):
            for c in g.clips:
                for r in (c.target, *c.distractors):
                    x = np.asarray(load_render_audio(cfg, r), dtype=np.float32)
                    yield r.item_id, audio_sha256(x), x
        return
    for it in _synth_split(cfg, split, limit):
        x = np.asarray(it.load_wet(), dtype=np.float32)
        yield it.item_id, audio_sha256(x), x


SOURCES: dict[str, Callable[..., Iterator[Row]]] = {
    "text": text_rows,
    "audio": audio_rows,
    "wet": wet_rows,
}


# ============================================================================ table builder


def build_table(
    path: Path,
    model: str,
    kind: str,
    rows: Iterable[Row],
    encode: Callable[[list], np.ndarray],
    batch_size: int,
    save_every: int = DEFAULT_SAVE_EVERY,
    prune: bool = True,
    split: str = "",
    dim: int | None = None,
) -> BuildStats:
    """Fill ``path`` from ``rows``, embedding only missing / stale ids with ``encode``.

    ``encode`` maps a list of payloads to ``(n, dim)`` L2-normalized float32 vectors; an
    existing table of another model, kind or ``dim`` is discarded.
    """
    t0 = time.perf_counter()
    old: EmbeddingTable | None = None
    if path.exists():
        try:
            old = EmbeddingTable.load(path)
        except Exception as e:  # corrupt / partial file: rebuild
            log.warning("%s/%s_%s: unreadable cache (%s); rebuilding", model, split, kind, e)
        if old is not None and (
            old.model != HF_IDS[model] or old.kind != kind or (dim is not None and old.dim != dim)
        ):
            log.warning("%s/%s_%s: cache of another model/kind/dim; rebuilding", model, split, kind)
            old = None
    vecs: dict[str, np.ndarray] = {}
    shas: dict[str, str] = {}
    if old is not None:
        for i, k in enumerate(old.ids):
            vecs[k] = old.vectors[i]
            shas[k] = old.sha256[i]
    seen: set[str] = set()
    pending: list[Row] = []
    n_computed = 0
    since_save = 0
    if dim is None and old is not None:
        dim = old.dim

    def flush() -> None:
        nonlocal n_computed, since_save, dim
        if not pending:
            return
        v = np.asarray(encode([p for _, _, p in pending]), dtype=np.float32)
        if v.shape[0] != len(pending):
            raise ValueError(f"encoder returned {v.shape[0]} vectors for {len(pending)} inputs")
        if dim is not None and v.shape[1] != dim:
            raise ValueError(f"{model}: encoder returned dim {v.shape[1]}, expected {dim}")
        dim = int(v.shape[1])
        for (k, h, _), row in zip(pending, v, strict=True):
            vecs[k] = row
            shas[k] = h
        n_computed += len(pending)
        since_save += len(pending)
        pending.clear()

    def table(final: bool) -> EmbeddingTable:
        keys = sorted(k for k in vecs if (k in seen or not (final and prune)))
        d = dim if dim is not None else 0
        arr = np.stack([vecs[k] for k in keys]) if keys else np.zeros((0, d), np.float32)
        return EmbeddingTable(
            ids=keys, vectors=arr, sha256=[shas[k] for k in keys], model=HF_IDS[model], kind=kind
        )

    for k, h, payload in rows:
        if k in seen:
            raise ValueError(f"{model}/{split}_{kind}: duplicate source id {k!r}")
        seen.add(k)
        if shas.get(k) == h:
            continue
        pending.append((k, h, payload))
        if len(pending) >= batch_size:
            flush()
            if since_save >= save_every:
                table(final=False).save(path)
                since_save = 0
                dt = time.perf_counter() - t0
                log.info(
                    "%s/%s_%s: %d embedded (%.1f/s), saved",
                    model,
                    split,
                    kind,
                    n_computed,
                    n_computed / dt,
                )
    flush()
    n_kept = len(seen) - n_computed
    final = table(final=True)
    n_pruned = len(vecs) - len(final) if prune else 0
    if n_computed or n_pruned or not path.exists():
        final.save(path)
    stats = BuildStats(
        model=model,
        split=split,
        kind=kind,
        n_rows=len(final),
        n_computed=n_computed,
        n_kept=n_kept,
        n_pruned=n_pruned,
        seconds=time.perf_counter() - t0,
    )
    log.info(
        "%s/%s_%s: %d rows (%d embedded at %.1f/s, %d kept, %d pruned) in %.0f s",
        model,
        split,
        kind,
        stats.n_rows,
        stats.n_computed,
        stats.rate,
        stats.n_kept,
        stats.n_pruned,
        stats.seconds,
    )
    return stats


# ============================================================================ plan + run


def plan(clap_models: Sequence[str], text_models: Sequence[str], splits: Sequence[str]):
    """``[(model, split, kind)]`` in build order: text, gold audio/wet, val/test/unseen
    audio/wet, train audio."""
    models = list(dict.fromkeys([*clap_models, *text_models]))
    out: list[tuple[str, str, str]] = []
    for s in ALL_SPLITS:
        if s in splits:
            out += [(m, s, "text") for m in models]
    audio_order = [GOLD, "val", "test", "test_unseen_pedal", "train"]
    for s in audio_order:
        if s not in splits:
            continue
        for m in clap_models:
            out.append((m, s, "audio"))
            if s in WET_SPLITS:
                out.append((m, s, "wet"))
    return out


class _EncoderCache:
    """Keeps one loaded encoder at a time (memory: the machine is shared)."""

    def __init__(self, device: str):
        self.device = device
        self.key: tuple[str, str] | None = None
        self.enc = None

    def get(self, model: str, kind: str):
        tower = "text" if kind == "text" else "audio"
        if FAMILIES[model] == "clap":
            key = (model, "clap")  # one ClapModel serves both towers
        else:
            key = (model, tower)
        if key != self.key:
            self.enc = None
            t0 = time.perf_counter()
            loader = load_text_encoder if tower == "text" else load_audio_encoder
            self.enc = loader(model, self.device)
            self.key = key
            log.info("loaded %s in %.1f s", model, time.perf_counter() - t0)
        return self.enc


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def content_digest(t: EmbeddingTable) -> str:
    """sha256 over ids, source shas and float32 vectors (independent of zip metadata)."""
    h = hashlib.sha256()
    for k, s in zip(t.ids, t.sha256, strict=True):
        h.update(k.encode("utf-8") + b"\0" + s.encode("ascii") + b"\n")
    h.update(np.ascontiguousarray(t.vectors, dtype=np.float32).tobytes())
    return h.hexdigest()


def summarize(cache_dir: Path, models: Sequence[str]) -> list[dict]:
    rows = []
    for m in models:
        for s in ALL_SPLITS:
            for k in KINDS:
                p = cache_path(cache_dir, m, s, k)
                if not p.exists():
                    continue
                t = EmbeddingTable.load(p)
                rows.append(
                    {
                        "model": m,
                        "hf_id": t.model,
                        "split": s,
                        "kind": k,
                        "n": len(t),
                        "dim": t.dim,
                        "file": p.relative_to(cache_dir).as_posix(),
                        "file_sha256": _file_sha256(p),
                        "content_sha256": content_digest(t),
                    }
                )
    return rows


def write_report(path: Path, rows: list[dict], stats: list[BuildStats], fixtures: bool) -> Path:
    lines = [
        "# Embedding cache (P5 task 5.5, D5)",
        "",
        "Generated by `lstmabar eval embed`"
        + (" on the synthetic eval fixture" if fixtures else "")
        + ". Paths are relative to the cache dir (`data/cache/`). `content_sha256` hashes ids, "
        "source shas and vectors (stable across re-saves); `file_sha256` is the npz file as "
        "written. Vectors are float32, L2-normalized.",
        "",
        "| model | split | kind | n | dim | file | file sha256 | content sha256 |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['split']} | {r['kind']} | {r['n']} | {r['dim']} | "
            f"`{r['file']}` | `{r['file_sha256'][:16]}` | `{r['content_sha256'][:16]}` |"
        )
    if stats:
        lines += [
            "",
            "## Last build",
            "",
            "| model | split | kind | embedded | kept | pruned | rows/s |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for s in stats:
            lines.append(
                f"| {s.model} | {s.split} | {s.kind} | {s.n_computed} | {s.n_kept} | "
                f"{s.n_pruned} | {s.rate:.1f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def build(
    cfg,
    models: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    limit: int | None = None,
    device: str = "cpu",
) -> list[BuildStats]:
    e = cfg.eval.embeddings
    clap = [str(m) for m in e.clap_models]
    text = [str(m) for m in e.text_models]
    if models is not None:
        unknown = sorted(set(models) - set(clap) - set(text))
        if unknown:
            raise ValueError(f"--models {unknown} not in eval.embeddings; known: {clap + text}")
        clap = [m for m in clap if m in models]
        text = [m for m in text if m in models]
    splits = list(splits) if splits is not None else list(ALL_SPLITS)
    bad = sorted(set(splits) - set(ALL_SPLITS))
    if bad:
        raise ValueError(f"unknown splits {bad}; expected a subset of {ALL_SPLITS}")
    cache_dir = Path(str(e.dir))
    bs = int(e.batch_size)
    save_every = int(e.get("save_every", DEFAULT_SAVE_EVERY))
    sr = int(cfg.audio.sample_rate)
    encoders = _EncoderCache(device)
    out: list[BuildStats] = []
    for model, split, kind in plan(clap, text, splits):
        enc = encoders.get(model, kind)
        if kind == "text":
            fn, batch = (lambda xs, enc=enc, b=bs: enc.encode_text(xs, batch_size=b)), bs
        else:
            ab = max(1, bs // 2)
            fn, batch = (lambda xs, enc=enc, b=ab: enc.encode_audio(xs, sr, b)), ab
        out.append(
            build_table(
                cache_path(cache_dir, model, split, kind),
                model,
                kind,
                SOURCES[kind](cfg, split, limit),
                fn,
                batch_size=batch,
                save_every=save_every,
                prune=limit is None,
                split=split,
                dim=int(enc.dim),
            )
        )
    return out


def run(cfg, args) -> int:
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fixtures = bool(getattr(args, "fixtures", False))
    if fixtures:
        from lstmabar.eval.fixtures import eval_fixture_config, make_eval_fixture

        root = Path(tempfile.mkdtemp(prefix="lstmabar-eval-fx-"))
        cfg = eval_fixture_config(root, list(getattr(args, "overrides", None) or []))
        make_eval_fixture(cfg, seconds=0.5)
        log.info("eval fixture written to a temp dir")
    torch.set_num_threads(int(cfg.eval.embeddings.get("num_threads", DEFAULT_NUM_THREADS)))
    limit = getattr(args, "limit", None)
    if limit is None and cfg.eval.get("max_items") is not None:
        limit = int(cfg.eval.max_items)
    stats = build(cfg, getattr(args, "models", None), getattr(args, "splits", None), limit)
    e = cfg.eval.embeddings
    rows = summarize(Path(str(e.dir)), [*e.clap_models, *e.text_models])
    report = write_report(Path(str(cfg.eval.report_dir)) / REPORT_NAME, rows, stats, fixtures)
    print(f"embedding cache: {len(rows)} tables; summary -> {report.name}")
    return 0


__all__ = [
    "ALL_SPLITS",
    "BuildStats",
    "build",
    "build_table",
    "content_digest",
    "plan",
    "run",
    "summarize",
    "write_report",
]
