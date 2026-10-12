"""Task 4.7 (decision 6): one-shot vocabulary overlap audit, held-out instructions vs captions.

``run(cfg, args)`` (``lstmabar data vocab-audit``) reads the held-out instruction files
(``freeze.gold_dir``, ``*.yaml``: lists of items with ``instruction`` and ``source``) and the
**frozen** captions (every shard's ``captions.jsonl``, verified against the hashes in the
freeze index), and writes ``reports/vocab_overlap.md`` (``freeze.vocab_report``) with
**aggregate numbers only**: vocabulary sizes and the fraction of instruction tokens / types
and bigrams covered by the caption vocabulary, per instruction ``source`` (``human``,
``simulated_human``, ``llm_persona``, ``llm_draft``, each reported separately and never
relabeled). It never prints or writes an instruction, a caption, or a word list.

**It runs once, at freeze.** The caption templates and paraphrase prompt must not be changed
after seeing this report; if they are, the dataset becomes v1.1 and is reported as "tuned
after gold audit". The step refuses to overwrite an existing report for that reason (delete
it deliberately to re-run, e.g. after a new freeze that did not touch the caption code).

This is the only data module that reads the held-out set; the caption code never imports it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from lstmabar.data.freeze import CAPTIONS_FILE
from lstmabar.data.layout import DataLayout
from lstmabar.data.schema import Caption, read_jsonl
from lstmabar.data.vocab import normalize_text

SOURCE_ORDER = ("human", "simulated_human", "llm_persona", "llm_draft")
_TOKEN = re.compile(r"[a-z]+(?:['-][a-z]+)*")

STOPWORDS = frozenset(
    """a an the and or but if so of to in on at by for with from into onto as is are was were be
    been being it its it's this that these those i me my we our you your he she they them their
    his her do does did done have has had can could would should will just very really quite
    some any more most less much many few not no too also than then there here what which who
    how when where like make makes made give gives get gets want please let up down out over
    one bit little lot""".split()
)
"""Generic English function words (not derived from either text set) for the content-word
rows."""


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalize_text(text).lower())


def bigrams(toks: Sequence[str]) -> list[tuple[str, str]]:
    return list(zip(toks, toks[1:], strict=False))


def load_instructions(gold_dir: Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """``[(source, instruction)]`` from every ``*.yaml`` in ``gold_dir`` and the files'
    sha256 (by file name). Items keep their own ``source`` (never relabeled)."""
    files = sorted(gold_dir.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no instruction files (*.yaml) in {gold_dir.name}/")
    out: list[tuple[str, str]] = []
    hashes: dict[str, str] = {}
    for f in files:
        data = f.read_bytes()
        hashes[f.name] = hashlib.sha256(data).hexdigest()
        doc = yaml.safe_load(data.decode("utf-8"))
        items = doc if isinstance(doc, list) else _first_list(doc)
        for it in items or []:
            if isinstance(it, Mapping) and isinstance(it.get("instruction"), str):
                out.append((str(it.get("source") or "unknown"), it["instruction"]))
    return out, hashes


def _first_list(doc: Any) -> list | None:
    if isinstance(doc, Mapping):
        for v in doc.values():
            if isinstance(v, list):
                return v
    return None


def load_frozen_captions(layout: DataLayout) -> list[Caption]:
    """Captions of every frozen shard; each file must match the freeze index's hash."""
    if not layout.index.exists():
        raise FileNotFoundError("no dataset index: run `lstmabar data freeze` first")
    index = json.loads(layout.index.read_text(encoding="utf-8"))
    manifests: Mapping[str, str] = index["files"]["manifests"]
    caps: list[Caption] = []
    for shard in sorted(index["shards"]):
        rel = f"{layout.version}/wet/{shard}/{CAPTIONS_FILE}"
        path = layout.wet_dir / shard / CAPTIONS_FILE
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if digest != manifests.get(rel):
            raise RuntimeError(f"{shard}/{CAPTIONS_FILE} differs from the frozen dataset")
        caps += list(read_jsonl(path, Caption))
    return caps


class Vocab:
    def __init__(self, texts: Iterable[str]) -> None:
        self.uni: set[str] = set()
        self.bi: set[tuple[str, str]] = set()
        self.n_texts = 0
        self.n_tokens = 0
        for t in texts:
            toks = tokens(t)
            self.uni.update(toks)
            self.bi.update(bigrams(toks))
            self.n_texts += 1
            self.n_tokens += len(toks)


def _frac(num: int, den: int) -> float | None:
    return num / den if den else None


def coverage(texts: Sequence[str], ref: Vocab) -> dict[str, Any]:
    """Aggregate coverage of ``texts`` by the reference vocabulary ``ref``."""
    uni = Counter()
    bi = Counter()
    for t in texts:
        toks = tokens(t)
        uni.update(toks)
        bi.update(bigrams(toks))
    content = Counter({w: n for w, n in uni.items() if w not in STOPWORDS})

    def cov(counter: Counter, vocab: set) -> tuple[float | None, float | None]:
        tok = _frac(sum(n for w, n in counter.items() if w in vocab), sum(counter.values()))
        typ = _frac(sum(1 for w in counter if w in vocab), len(counter))
        return tok, typ

    out: dict[str, Any] = {
        "items": len(texts),
        "tokens": sum(uni.values()),
        "types": len(uni),
        "content_types": len(content),
        "bigram_types": len(bi),
    }
    out["token_cov"], out["type_cov"] = cov(uni, ref.uni)
    out["content_token_cov"], out["content_type_cov"] = cov(content, ref.uni)
    out["bigram_token_cov"], out["bigram_type_cov"] = cov(bi, ref.bi)
    out["jaccard"] = _frac(len(set(uni) & ref.uni), len(set(uni) | ref.uni))
    return out


def audit(instructions: Sequence[tuple[str, str]], captions: Sequence[Caption]) -> dict[str, Any]:
    """Aggregate overlap numbers per instruction source, against all captions and against
    template captions only."""
    refs = {
        "all": Vocab(c.text for c in captions),
        "template": Vocab(c.text for c in captions if c.kind == "template"),
        "paraphrase": Vocab(c.text for c in captions if c.kind == "paraphrase"),
    }
    by_source: dict[str, list[str]] = defaultdict(list)
    for src, text in instructions:
        by_source[src].append(text)
    order = [s for s in SOURCE_ORDER if s in by_source]
    order += sorted(s for s in by_source if s not in SOURCE_ORDER)
    rows = {s: {name: coverage(by_source[s], ref) for name, ref in refs.items()} for s in order}
    return {
        "captions": {
            name: {"texts": r.n_texts, "tokens": r.n_tokens, "types": len(r.uni)}
            for name, r in refs.items()
        },
        "sources": rows,
        "missing_sources": [s for s in SOURCE_ORDER if s not in by_source],
    }


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f} %"


def render_report(
    result: Mapping[str, Any], gold_hashes: Mapping[str, str], dataset_sha: str | None
) -> str:
    lines = [
        "# Vocabulary overlap: held-out instructions vs dataset captions",
        "",
        "Generated once at freeze by `lstmabar data vocab-audit` "
        "(src/lstmabar/data/vocab_audit.py). Aggregate numbers only: no instruction, caption "
        "or word list is shown. **Caption templates and the paraphrase prompt must not change "
        'after this report;** if they do, the dataset becomes v1.1 ("tuned after gold '
        'audit").',
        "",
        f"Dataset SHA-256: `{dataset_sha or 'unknown'}`.",
        "",
        "Instruction files (sha256):",
        "",
        *[f"- `{name}`: `{h}`" for name, h in sorted(gold_hashes.items())],
        "",
        "## Caption vocabularies",
        "",
        "| captions | texts | tokens | word types |",
        "|---|---|---|---|",
        *[
            f"| {k} | {v['texts']} | {v['tokens']} | {v['types']} |"
            for k, v in result["captions"].items()
        ],
        "",
        "## Coverage by instruction source",
        "",
        "Each source is reported separately (`human` is the headline set; `simulated_human`, "
        "`llm_persona` and `llm_draft` are LLM-written). *Token coverage* = share of "
        "instruction word occurrences whose word appears anywhere in the captions; *type "
        "coverage* = share of distinct instruction words; *content* rows drop a generic "
        "English stopword list; *bigram* = adjacent word pairs; *Jaccard* = shared / union "
        "word types.",
        "",
    ]
    if result["missing_sources"]:
        lines += [
            "No items yet for: " + ", ".join(f"`{s}`" for s in result["missing_sources"]) + ".",
            "",
        ]
    for ref in ("all", "template", "paraphrase"):
        lines += [
            f"### Against {ref} captions",
            "",
            "| source | items | tokens | types | token cov | type cov | content token cov | "
            "content type cov | bigram token cov | bigram type cov | Jaccard |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for src, by_ref in result["sources"].items():
            r = by_ref[ref]
            lines.append(
                f"| {src} | {r['items']} | {r['tokens']} | {r['types']} | "
                f"{_pct(r['token_cov'])} | {_pct(r['type_cov'])} | "
                f"{_pct(r['content_token_cov'])} | {_pct(r['content_type_cov'])} | "
                f"{_pct(r['bigram_token_cov'])} | {_pct(r['bigram_type_cov'])} | "
                f"{_pct(r['jaccard'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def _cfg_path(cfg, key: str, default: str) -> Path:
    if "freeze" in cfg and cfg.freeze.get(key):
        return Path(str(cfg.freeze[key]))
    return Path(default)


def run(cfg, args) -> int:
    layout = DataLayout.from_config(cfg)
    report = _cfg_path(cfg, "vocab_report", "reports/vocab_overlap.md")
    if report.exists():
        print(
            f"{report.name} already exists: the audit runs once per freeze. Delete it "
            "deliberately to re-run (templates must not change after seeing it)."
        )
        return 1
    captions = load_frozen_captions(layout)
    instructions, hashes = load_instructions(_cfg_path(cfg, "gold_dir", "data/gold"))
    result = audit(instructions, captions)
    index = json.loads(layout.index.read_text(encoding="utf-8"))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        render_report(result, hashes, index.get("dataset_sha256")), encoding="utf-8", newline="\n"
    )
    print(f"wrote {report} ({len(instructions)} instructions, {len(captions)} captions)")
    return 0


__all__ = [
    "STOPWORDS",
    "audit",
    "bigrams",
    "coverage",
    "load_frozen_captions",
    "load_instructions",
    "render_report",
    "run",
    "tokens",
]
