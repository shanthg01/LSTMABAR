"""``reports/p5_baselines.{md,json}`` (task 5.4, D6).

JSON layout (``schema: 1``)::

    {
      "schema": 1,
      "created": "...", "git": {...}, "config": {...resolved eval config...},
      "hashes": {"dataset_sha256": ..., "gold_index_sha256": ..., "embeddings": {relpath: sha}},
      "methods": [{"name": ..., "info": {...}, "seeds": [0, ...]}],
      "rows": [Row, ...],               # long format, one cell per row
      "comparisons": [ComparisonRow, ...],
      "test_real": {...},               # D11 section (sim-to-real gap), aggregate only for
                                        # internal_only items
      "notes": ["...", ...]             # caveats the markdown prints verbatim
    }

A cell is ``method x split x metric x subset x gold_source``; ``gold_source`` is ``"-"`` for
synthetic splits and one of ``data.gold.SOURCES`` (``human`` carried from the start, "n/a,
0 items" until friend items are paired) for gold. No path in the report is absolute.

Also in the JSON (additions to schema 1, all optional for readers): ``run`` (run dir name,
``--limit``, fixtures flag), ``harmonic`` (``{split: bool}``, where harmonic descriptors were
on), ``render_failed`` (``{method: {split: n}}``) and per method ``status`` / ``splits`` /
``selection``.

Multi-seed methods: per-unit scores are averaged over seeds before the item bootstrap; the
row's ``seed_sd`` is the SD of the per-seed means (D3). Gate F1 rows (``gate_f1/<block>``) are
split-level statistics, so their ``lo = hi = mean`` and ``n`` is the number of queries.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from lstmabar.eval.runner import GOLD_SCORES, SYNTH_SCORES, SplitResult

REPORT_SCHEMA = 1
GOLD = "gold"
NO_SOURCE = "-"
HEADLINE_METRIC = "afc_mrstft"
SECONDARY_METRIC = "afc_descriptor"
BASELINE_REFS = ("noop", "mean")
MODEL = "p6_model"
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Row:
    method: str
    split: str  # test | test_unseen_pedal | val | gold
    metric: str  # runner.SYNTH_SCORES / GOLD_SCORES key, or "gate_f1/<block>"
    subset: str  # "all" | data.GOLD_SUBSETS
    gold_source: str  # "-" | data.gold.SOURCES
    mean: float
    lo: float
    hi: float
    n: int
    seed_sd: float | None = None  # trained methods: SD of per-seed means (D3)
    n_seeds: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonRow:
    a: str
    b: str
    split: str
    metric: str
    subset: str
    gold_source: str
    comparison: dict[str, Any]  # stats.Comparison.to_dict()


# ============================================================================ helpers


def _gold_sources() -> tuple[str, ...]:
    from lstmabar.data.gold import SOURCES

    return tuple(SOURCES)


def _gold_subsets() -> tuple[str, ...]:
    from lstmabar.eval.data import GOLD_SUBSETS

    return tuple(GOLD_SUBSETS)


def _finite(x) -> bool:
    return x is not None and isinstance(x, int | float) and math.isfinite(float(x))


def _clean_json(obj: Any) -> Any:
    """NaN/inf -> None, tuples -> lists, absolute paths scrubbed (strict JSON)."""
    from lstmabar.eval.run import scrub

    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float | np.floating):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return scrub(str(obj))


def _sha256_text_file(path: Path) -> str | None:
    """sha256 of a committed text file with line endings normalized to LF (stable across
    checkouts with or without ``core.autocrlf``)."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(p: str | Path) -> Path:
    """A config path: absolute as is; relative against the CWD, else the repo root."""
    p = Path(str(p))
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def compute_hashes(cfg) -> dict[str, Any]:
    """Dataset sha256 (from the committed dataset index), the gold index file sha256 and the
    sha256 of every embedding-cache file (keyed by path relative to the cache dir).
    ``missing`` says why a hash is absent (never a path)."""
    out: dict[str, Any] = {
        "dataset_sha256": None,
        "gold_index_sha256": None,
        "embeddings": {},
        "missing": {},
    }
    try:
        index = _resolve(cfg.data.manifests_dir) / str(cfg.data.version) / "index.json"
        if not index.exists():
            out["missing"]["dataset_sha256"] = (
                f"dataset index {cfg.data.version}/index.json not found"
            )
        else:
            sha = json.loads(index.read_text("utf-8")).get("dataset_sha256")
            out["dataset_sha256"] = sha
            if not sha:
                out["missing"]["dataset_sha256"] = "dataset index has no dataset_sha256"
    except (OSError, ValueError, AttributeError) as e:
        out["missing"]["dataset_sha256"] = f"dataset index unreadable ({type(e).__name__})"
    try:
        gidx = _resolve(cfg.eval.gold_index)
        out["gold_index_sha256"] = _sha256_text_file(gidx)
        if gidx.exists():
            out["gold_index_dataset_sha256"] = json.loads(gidx.read_text("utf-8")).get(
                "dataset_sha256"
            )
        else:
            out["missing"]["gold_index_sha256"] = f"gold index {gidx.name} not found"
    except (OSError, ValueError, AttributeError) as e:
        out["missing"]["gold_index_sha256"] = f"gold index unreadable ({type(e).__name__})"
    try:
        edir = _resolve(cfg.eval.embeddings.dir)
        for fam in ("clap", "bge"):
            for p in sorted((edir / fam).glob("*/*.npz")):
                out["embeddings"][p.relative_to(edir).as_posix()] = _sha256_file(p)
        if not out["embeddings"]:
            out["missing"]["embeddings"] = "no embedding-cache files (lstmabar eval embed)"
    except (OSError, AttributeError) as e:
        out["missing"]["embeddings"] = f"embedding cache unreadable ({type(e).__name__})"
    return out


def _config_section(cfg) -> dict[str, Any]:
    c = OmegaConf.to_container(cfg, resolve=True)
    keep = {"eval": c.get("eval", {})}
    for k in ("data", "gold"):
        if isinstance(c.get(k), dict) and "version" in c[k]:
            keep[k] = {"version": c[k]["version"]}
    keep["audio"] = {k: c.get("audio", {}).get(k) for k in ("sample_rate", "inference_peak_dbfs")}
    return _clean_json(_relativize(keep, cfg))


def _relativize(obj: Any, cfg) -> Any:
    """Absolute config paths -> relative to the repo / data root (never absolute)."""
    roots: list[tuple[Path, str]] = []
    try:
        roots.append((Path(str(cfg.paths.data)).resolve(), "${paths.data}"))
    except (AttributeError, OSError):
        pass
    roots.append((REPO_ROOT, "."))

    def fix(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, str) and Path(v).is_absolute():
            p = Path(v).resolve()
            for root, label in roots:
                try:
                    return f"{label}/{p.relative_to(root).as_posix()}"
                except ValueError:
                    continue
            return f"<abs>/{p.name}"
        return v

    return fix(obj)


def _cell_values(results: Sequence[SplitResult]):
    """``{(method, split): {"seeds": [...], "units": {unit_id: {seed: UnitScores}}}}``."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in results:
        d = out.setdefault(
            (r.method, r.split), {"seeds": [], "units": defaultdict(dict), "extra": {}}
        )
        d["seeds"].append(r.seed)
        d["extra"][r.seed] = r.extra
        for u in r.units:
            d["units"][u.unit_id][r.seed] = u
    return out


def _metric_keys(units: dict, base: Sequence[str]) -> list[str]:
    keys = list(base)
    extra: set[str] = set()
    for per_seed in units.values():
        for u in per_seed.values():
            extra.update(k for k in u.scores if k not in keys)
    return keys + sorted(extra)


def _per_unit(units: dict, metric: str, keep=None) -> tuple[list[str], np.ndarray, list[float]]:
    """``(unit ids, seed-averaged per-unit values, per-seed means)``."""
    ids = sorted(u for u, ps in units.items() if keep is None or keep(next(iter(ps.values()))))
    seeds = sorted({s for ps in units.values() for s in ps})
    mat = np.full((len(ids), len(seeds)), np.nan)
    for i, uid in enumerate(ids):
        for j, s in enumerate(seeds):
            u = units[uid].get(s)
            if u is not None:
                v = u.scores.get(metric)
                mat[i, j] = float(v) if _finite(v) else np.nan
    with np.errstate(all="ignore"), _quiet():
        vals = np.nanmean(mat, axis=1) if mat.size else np.zeros(0)
        seed_means = [float(np.nanmean(mat[:, j])) for j in range(len(seeds))] if ids else []
    return ids, vals, seed_means


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        return self._w.__exit__(*a)


def _row(method, split, metric, subset, source, vals, seed_means, st) -> Row:
    from lstmabar.eval.stats import bootstrap_ci, seed_summary

    ci = bootstrap_ci(vals, int(st.n_resamples), int(st.seed), float(st.alpha))
    sd = None
    n_seeds = len(seed_means) or 1
    if n_seeds > 1:
        sd = seed_summary(seed_means)["sd"]
    return Row(method, split, metric, subset, source, ci.mean, ci.lo, ci.hi, ci.n, sd, n_seeds)


def _stats_cfg(cfg):
    st = cfg.eval.stats
    return OmegaConf.create(
        {
            "n_resamples": int(st.get("n_resamples", 10000)),
            "alpha": float(st.get("alpha", 0.05)),
            "seed": int(cfg.eval.get("seed", 0)),
        }
    )


def _gold_keep(subset: str, source: str):
    def keep(u) -> bool:
        return u.meta.get("gold_source") == source and subset in (u.meta.get("subsets") or [])

    return keep


# ============================================================================ build


def build_rows(results: Sequence[SplitResult], cfg) -> list[Row]:
    st = _stats_cfg(cfg)
    rows: list[Row] = []
    sources, subsets = _gold_sources(), _gold_subsets()
    for (method, split), d in _cell_values(results).items():
        units = d["units"]
        if split == GOLD:
            for metric in _metric_keys(units, GOLD_SCORES):
                for subset in subsets:
                    for source in sources:
                        _, vals, sm = _per_unit(units, metric, _gold_keep(subset, source))
                        rows.append(_row(method, split, metric, subset, source, vals, sm, st))
            continue
        for metric in _metric_keys(units, SYNTH_SCORES):
            _, vals, sm = _per_unit(units, metric)
            rows.append(_row(method, split, metric, "all", NO_SOURCE, vals, sm, st))
        gates: dict[str, list[float]] = defaultdict(list)
        n_q = 0
        for extra in d["extra"].values():
            n_q = int(extra.get("n_queries", n_q) or 0)
            for block, v in (extra.get("gate_f1") or {}).items():
                gates[block].append(float(v) if _finite(v) else np.nan)
        for block, vs in gates.items():
            v = np.asarray(vs, dtype=np.float64)
            fin = v[np.isfinite(v)]
            mean = float(fin.mean()) if fin.size else math.nan
            sd = float(fin.std(ddof=1)) if fin.size >= 2 else None
            n = n_q if fin.size else 0
            rows.append(
                Row(
                    method,
                    split,
                    f"gate_f1/{block}",
                    "all",
                    NO_SOURCE,
                    mean,
                    None,  # split-level statistic: no item CI
                    None,
                    n,
                    sd,
                    len(vs),
                )
            )
    return rows


def _pairs_for(cells, a: str, b: str, split: str, metric: str, keep=None):
    da, db = cells.get((a, split)), cells.get((b, split))
    if da is None or db is None:
        return None
    ia, va, _ = _per_unit(da["units"], metric, keep)
    ib, vb, _ = _per_unit(db["units"], metric, keep)
    common = sorted(set(ia) & set(ib))
    if not common:
        return None
    ma, mb = dict(zip(ia, va, strict=True)), dict(zip(ib, vb, strict=True))
    return np.array([ma[u] for u in common]), np.array([mb[u] for u in common])


def build_comparisons(results: Sequence[SplitResult], cfg) -> list[ComparisonRow]:
    """Every method vs ``noop`` and vs ``mean`` (and ``p6_model`` vs every baseline) on the
    headline gold 2AFC (per gold source) and MR-STFT on each synthetic test split."""
    from lstmabar.eval.stats import compare

    st = _stats_cfg(cfg)
    cells = _cell_values(results)
    methods = list(dict.fromkeys(r.method for r in results))
    pairs: list[tuple[str, str]] = []
    cand = [(m, ref) for ref in BASELINE_REFS if ref in methods for m in methods if m != ref]
    if MODEL in methods:
        cand += [(MODEL, m) for m in methods if m != MODEL]
    for a, b in cand:  # one direction per pair (diff = A - B)
        if (a, b) not in pairs and (b, a) not in pairs:
            pairs.append((a, b))
    out: list[ComparisonRow] = []
    targets = [(GOLD, HEADLINE_METRIC, "headline", s) for s in _gold_sources()]
    targets += [(str(s), "mrstft", "all", NO_SOURCE) for s in cfg.eval.splits]
    for a, b in pairs:
        for split, metric, subset, source in targets:
            keep = _gold_keep(subset, source) if split == GOLD else None
            p = _pairs_for(cells, a, b, split, metric, keep)
            if p is None:
                continue
            c = compare(p[0], p[1], int(st.n_resamples), int(st.seed), float(st.alpha))
            if c.n == 0:
                continue
            out.append(ComparisonRow(a, b, split, metric, subset, source, c.to_dict()))
    return out


TEST_REAL_SECTIONS = ("protocol", "groups", "kb", "family_shifts", "ts_grid_direction")
"""Aggregate sections of ``reports/sim_real.json`` embedded in the report. Per-item sections
(``per_item``, ``kb_knob_sweeps``) are never embedded."""


def _sanitize_test_real(obj: Any) -> Any:
    """Inside an aggregate section, list entries that are records (dicts) are kept only when
    they are explicitly ``internal_only: false``; group-level aggregates (dict values) are
    kept, including pooled internal_only groups (aggregate only, D6)."""
    if isinstance(obj, dict):
        return {k: _sanitize_test_real(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [
            _sanitize_test_real(v)
            for v in obj
            if not isinstance(v, dict) or v.get("internal_only") is False or _is_cell(v)
        ]
    return obj


def _is_cell(v: dict) -> bool:
    """An aggregate cell (e.g. a ``kb`` engine x level cell of CIs), not an item record."""
    return "item_id" not in v and any(
        isinstance(x, dict) and {"mean", "n"} <= set(x) for x in v.values()
    )


def load_test_real(cfg) -> dict[str, Any] | None:
    path = _resolve(cfg.eval.report_dir) / "sim_real.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "error", "error": "reports/sim_real.json could not be read"}
    if not isinstance(data, dict):
        return {"status": "error", "error": "reports/sim_real.json is not an object"}
    out = {k: _sanitize_test_real(data[k]) for k in TEST_REAL_SECTIONS if k in data}
    out["omitted_sections"] = sorted(k for k in data if k not in TEST_REAL_SECTIONS)
    out["status"] = "embedded"
    return out


def _harmonic_by_split(results: Sequence[SplitResult], cfg) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for r in results:
        h = r.extra.get("harmonic_descriptors")
        if h is None:
            continue
        out[r.split] = bool(out.get(r.split, True) and h)
    return out


def _render_failed(results: Sequence[SplitResult]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for r in results:
        n = int(r.extra.get("n_render_failed", 0) or 0)
        out[r.method][r.split] = out[r.method].get(r.split, 0) + n
    return dict(out)


def _n_human_headline(results: Sequence[SplitResult]) -> int:
    return len(
        {
            u.unit_id
            for r in results
            if r.split == GOLD
            for u in r.units
            if u.meta.get("gold_source") == "human" and "headline" in (u.meta.get("subsets") or [])
        }
    )


def _gold_note(results: Sequence[SplitResult]) -> str:
    """D9 wording when there are no real human items; human-headline wording otherwise."""
    n_h = _n_human_headline(results)
    caveat = (
        "The training paraphrases are also Claude-written (claude-haiku-5-5), so LLM-written "
        "test instructions are closer in style to the training captions than real people's; "
        "this weakens the template-circularity guard of design §5."
    )
    if n_h == 0:
        return (
            "No real human gold items yet (human column n/a). The gold headline is reported on "
            "each LLM-written set (simulated_human, llm_persona, llm_draft) separately, never "
            "pooled or relabeled, and claims are worded 'on LLM-written test instructions' "
            f"(D9). {caveat}"
        )
    small = (
        f" With only {n_h} headline items (< ~30) the human headline is weak; the LLM-written "
        "sets are reported next to it."
        if n_h < 30
        else ""
    )
    return (
        f"The gold headline is the real human set (n = {n_h} headline items); the LLM-written "
        "sets (simulated_human, llm_persona, llm_draft) are supplementary, each reported "
        f"separately and never pooled or relabeled (D9).{small} {caveat}"
    )


def _notes(cfg, results, harmonic, failed, methods, extras) -> list[str]:
    notes = [
        "v1 model (W266 submission, tag v1-w266): not comparable and not evaluated: it uses a "
        "different label scheme and the legacy DDSP engine, not the v2 free board (D4).",
        _gold_note(results),
        "Paired comparisons: Wilcoxon p-values are uncorrected for multiple comparisons (many "
        "method pairs x gold sources x splits); read them with the effect sizes and CIs, not "
        "as a family of significance tests.",
        "Headline gold subset = in scope, not ambiguous and not unseen-pedal; ambiguous, "
        "out-of-scope and unseen-pedal items are reported separately. Gain distractors also "
        "change the drive voicing (gain + voicing), and light-compression vs off contrasts are "
        "weak (targets file header).",
        "The secondary (descriptor) 2AFC distance averages the scaled centroid, odd/even and "
        "HNR errors with the unscaled archetype L1. The archetypes are a measured harmonic "
        "readout from the same analysis, so that term is not independent of the harmonic "
        "terms; MR-STFT is the primary 2AFC distance.",
        "CLAP score is a secondary metric only (spike A: mean word AUC ~0.6; only the "
        "drive-amount contrast is robust); it is not part of this table.",
        "test_unseen_pedal and test_real are supporting evidence, not text-adherence results.",
        "test_real: items marked internal_only are reported in aggregate only; no per-item "
        "numbers of those items appear in this report.",
        "Gate F1 is a split-level statistic over all queries (no item bootstrap: lo = hi = "
        "mean, n = queries). Every other cell is mean [95 % bootstrap CI over items] (n items).",
        "Failed renders (non-finite or silent output) count as a 2AFC loss on gold; on "
        "synthetic splits their audio metrics are NaN (dropped from n).",
    ]
    off = sorted(s for s, h in harmonic.items() if not h)
    on = sorted(s for s, h in harmonic.items() if h)
    if off:
        notes.append(
            "Harmonic descriptors (odd/even, HNR, archetype terms of predicted renders) were OFF "
            f"for: {', '.join(off)} (eval.metrics.harmonic_descriptors=false"
            + (", eval.select_harmonic=false for val selection" if "val" in off else "")
            + "); those terms are n/a there and the descriptor 2AFC uses the centroid (and "
            "archetype where available) only."
            + (f" They were on for: {', '.join(on)}." if on else "")
        )
    elif on:
        notes.append(f"Harmonic descriptors were on for every scored split ({', '.join(on)}).")
    nf = {m: s for m, s in failed.items() if any(s.values())}
    if nf:
        parts = [f"{m}: " + ", ".join(f"{k} {v}" for k, v in s.items() if v) for m, s in nf.items()]
        notes.append("Failed renders (queries): " + "; ".join(parts) + ".")
    else:
        notes.append("Failed renders: none.")
    for m in methods:
        if why := _reasons(m):
            notes.append(f"{m['name']} not run: {why}")
    if extras.get("limit"):
        notes.append(
            f"Smoke run: at most {extras['limit']} items per evaluated split (--limit); these "
            "numbers are not the reported baseline results."
        )
    if extras.get("fixtures"):
        notes.append("Synthetic eval fixture (--fixtures): numbers are meaningless placeholders.")
    return notes


def build_report(
    results: Sequence[SplitResult], cfg, extras: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The report dict (``schema: 1``). ``extras``: ``methods`` (run status records from
    ``eval.run``: name, status, splits, seeds, info, selection), ``run_dir`` (a name, never a
    path), ``limit``, ``fixtures``; ``test_real`` overrides ``reports/sim_real.json``."""
    from lstmabar.runs import git_info

    extras = dict(extras or {})
    results = list(results)
    by_method = defaultdict(set)
    for r in results:
        by_method[r.method].add(r.seed)
    records = list(extras.get("methods") or [])
    known = {m["name"] for m in records}
    records += [{"name": m, "status": "ok", "splits": {}} for m in by_method if m not in known]
    methods = []
    for m in records:
        entry = {
            "name": m["name"],
            "info": m.get("info") or {},
            "seeds": sorted(by_method.get(m["name"], set())),  # seeds with scores
            "status": m.get("status", "ok"),
            "splits": m.get("splits") or {},
            "deterministic": m.get("deterministic"),
            "selection": m.get("selection") or {},
        }
        methods.append(entry)
    harmonic = _harmonic_by_split(results, cfg)
    failed = _render_failed(results)
    test_real = extras.get("test_real")
    if test_real is None:
        test_real = load_test_real(cfg)
    report = {
        "schema": REPORT_SCHEMA,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(REPO_ROOT),
        "config": _config_section(cfg),
        "hashes": compute_hashes(cfg),
        "run": {
            "run_dir": extras.get("run_dir"),
            "limit": extras.get("limit"),
            "fixtures": bool(extras.get("fixtures", False)),
        },
        "methods": methods,
        "rows": [r.to_dict() for r in build_rows(results, cfg)],
        "comparisons": [asdict(c) for c in build_comparisons(results, cfg)],
        "test_real": test_real if test_real is not None else {"status": "not run"},
        "harmonic": harmonic,
        "render_failed": failed,
        "notes": _notes(cfg, results, harmonic, failed, methods, extras),
    }
    return _clean_json(report)


# ============================================================================ markdown


def _num(x) -> str:
    if not _finite(x):
        return "n/a"
    x = float(x)
    return f"{x:.3f}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"


def fmt_cell(row: dict[str, Any] | None) -> str:
    """``mean [lo, hi] (n)``; ``n/a, 0 items`` when empty; ``; seed SD s`` for multi-seed."""
    if row is None:
        return "-"
    n = int(row.get("n") or 0)
    if n == 0 or not _finite(row.get("mean")):
        return "n/a, 0 items"
    s = f"{_num(row['mean'])} [{_num(row['lo'])}, {_num(row['hi'])}] ({n})"
    if int(row.get("n_seeds") or 1) > 1:
        s += f"; seed SD {_num(row.get('seed_sd'))} ({row['n_seeds']} seeds)"
    return s


def _index(rows: Sequence[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    return {(r["method"], r["split"], r["metric"], r["subset"], r["gold_source"]): r for r in rows}


def _table(header: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in body]
    return out + [""]


def _status(m: dict[str, Any], split: str) -> str | None:
    """``"not run: ..."`` if the method has no scores for ``split``."""
    v = (m.get("splits") or {}).get(split)
    if v is not None and str(v).startswith("not run"):
        return "not run (see Methods)"
    return None


def _reasons(m: dict[str, Any]) -> str:
    """The method's distinct "not run" reasons, each once with the splits it applies to."""
    by: dict[str, list[str]] = {}
    for s, v in (m.get("splits") or {}).items():
        if str(v).startswith("not run"):
            by.setdefault(str(v).removeprefix("not run: "), []).append(s)
    return "; ".join(f"{r} [{', '.join(ss)}]" for r, ss in by.items())


def _method_cells(report, split, metrics, subset, source, idx) -> list[list[str]]:
    body = []
    for m in report["methods"]:
        st = _status(m, split)
        if st:
            body.append([m["name"], st] + [""] * (len(metrics) - 1))
            continue
        body.append(
            [m["name"]]
            + [fmt_cell(idx.get((m["name"], split, k, subset, source))) for k in metrics]
        )
    return body


def _sel_summary(m: dict[str, Any]) -> tuple[str, str]:
    sels = m.get("selection") or {}
    if not sels:
        return "-", "-"
    first = sels[sorted(sels)[0]]
    chosen = first.get("chosen") or {}
    ch = ", ".join(f"{k}={v}" for k, v in chosen.items()) or "-"
    if len(sels) > 1:
        chs = {json.dumps(s.get("chosen") or {}, sort_keys=True) for s in sels.values()}
        if len(chs) > 1:
            ch += f" (varies over {len(sels)} seeds)"
    crit = str(first.get("criterion") or "-")
    if first.get("harmonic_descriptors") is False:
        crit += "; harmonic descriptors off"
    return ch, crit


def render_markdown(report: dict[str, Any]) -> str:
    idx = _index(report["rows"])
    sources = list(_gold_sources())
    subsets = list(_gold_subsets())
    cfg = report.get("config", {}).get("eval", {})
    synth_splits = [str(s) for s in cfg.get("splits", ["test", "test_unseen_pedal"])]
    val_split = str(cfg.get("select_split", "val"))
    present = {(r["split"], r["metric"]) for r in report["rows"]}
    L: list[str] = ["# P5 baselines", ""]
    git = report.get("git") or {}
    h = report.get("hashes") or {}
    run = report.get("run") or {}
    L += [
        f"Generated {report.get('created')} at commit `{git.get('sha') or 'unknown'}`"
        + (" (dirty tree)" if git.get("dirty") else "")
        + f"; run `{run.get('run_dir') or '-'}`; config `configs/eval/p5.yaml`.",
        "",
        f"- dataset sha256: `{h.get('dataset_sha256') or 'n/a'}`",
        f"- gold index sha256 (LF-normalized): `{h.get('gold_index_sha256') or 'n/a'}`",
        f"- embedding cache files: {len(h.get('embeddings') or {})} (sha256 in the JSON)",
        "",
        "Every cell: mean [95 % bootstrap CI] (n items). Higher is better for 2AFC (chance "
        "0.5) and gate accuracy/F1; lower is better for MR-STFT, L1 and descriptor errors.",
        "",
    ]
    if run.get("limit") or run.get("fixtures"):
        L += [
            "> **Smoke run** ("
            + ("fixture data" if run.get("fixtures") else f"--limit {run.get('limit')}")
            + "): not the reported baseline results.",
            "",
        ]

    # ---- methods
    L += ["## Methods", ""]
    body = []
    for m in report["methods"]:
        ch, crit = _sel_summary(m)
        seeds = m.get("seeds") or []
        status = m.get("status", "ok")
        if why := _reasons(m):
            status += f": {why}"
        body.append([m["name"], status, str(len(seeds)), ch, crit])
    L += _table(["method", "status", "seeds", "chosen (val)", "selection criterion"], body)

    # ---- headline gold 2AFC
    L += ["## Gold 2AFC, headline subset (per gold source)", ""]
    L += [
        "Primary distance MR-STFT; LLM-written sources are reported separately ('on LLM-written "
        "test instructions'); `human` stays n/a until real friend items are paired.",
        "",
    ]
    for metric, label in ((HEADLINE_METRIC, "MR-STFT 2AFC"), (SECONDARY_METRIC, "descriptor 2AFC")):
        L += [f"### {label} (headline)", ""]
        L += _table(["method", *sources], _method_cells_by_source(report, metric, "headline", idx))
    L += ["### MR-STFT 2AFC by subset", ""]
    for subset in [s for s in subsets if s != "headline"]:
        L += [f"**{subset}**", ""]
        L += _table(
            ["method", *sources], _method_cells_by_source(report, HEADLINE_METRIC, subset, idx)
        )
    axes = sorted(
        {
            m.split("/", 1)[1]
            for s, m in present
            if s == GOLD and m.startswith(HEADLINE_METRIC + "/")
        }
    )
    if axes:
        L += ["### MR-STFT 2AFC per distractor axis (headline)", ""]
        body = []
        for m in report["methods"]:
            if _status(m, GOLD):
                continue
            for src in sources:
                cells = [
                    fmt_cell(idx.get((m["name"], GOLD, f"{HEADLINE_METRIC}/{a}", "headline", src)))
                    for a in axes
                ]
                if all(c in ("-", "n/a, 0 items") for c in cells):
                    continue
                body.append([m["name"], src, *cells])
        L += _table(["method", "gold source", *axes], body)
    L += ["### Other gold metrics (headline)", ""]
    body = []
    for m in report["methods"]:
        if st := _status(m, GOLD):
            body.append([m["name"], "-", st, ""])
            continue
        for src in sources:
            body.append(
                [m["name"], src]
                + [
                    fmt_cell(idx.get((m["name"], GOLD, k, "headline", src)))
                    for k in ("region_l1", "mrstft_target")
                ]
            )
    L += _table(["method", "gold source", "region L1", "MR-STFT to target"], body)

    # ---- synthetic test splits
    synth_metrics = list(SYNTH_SCORES)
    for split in synth_splits:
        L += [f"## Synthetic `{split}`", ""]
        L += _table(
            ["method", *synth_metrics],
            _method_cells(report, split, synth_metrics, "all", NO_SOURCE, idx),
        )
        L += _gate_table(report, split, idx, present)

    # ---- val selection
    L += [f"## Val selection table (`{val_split}`; selection only, D8)", ""]
    vmetrics = synth_metrics + sorted(
        {m for s, m in present if s == val_split and m.startswith("afc_")}
    )
    L += _table(
        ["method", *vmetrics], _method_cells(report, val_split, vmetrics, "all", NO_SOURCE, idx)
    )

    # ---- comparisons
    L += ["## Paired comparisons", ""]
    L += [
        "diff = A - B on paired items: mean [95 % paired bootstrap CI] (n); Wilcoxon "
        "signed-rank p; Cliff's delta; paired d (mean diff / SD).",
        "",
    ]
    body = []
    for c in report["comparisons"]:
        cmp_ = c["comparison"]
        d = cmp_["diff"]
        where = f"{c['split']} {c['subset']}" + (
            f" {c['gold_source']}" if c["gold_source"] != NO_SOURCE else ""
        )
        body.append(
            [
                c["a"],
                c["b"],
                where,
                c["metric"],
                f"{_num(d['mean'])} [{_num(d['lo'])}, {_num(d['hi'])}] ({cmp_['n']})",
                _num(cmp_.get("p_wilcoxon")),
                _num(cmp_.get("cliffs_delta")),
                _num(cmp_.get("d_paired")),
            ]
        )
    L += (
        _table(
            ["A", "B", "split / subset / source", "metric", "diff", "p", "Cliff's delta", "d"], body
        )
        if body
        else ["(no comparable pairs)", ""]
    )

    # ---- test_real
    L += ["## test_real: sim-to-real gap (D11)", ""]
    tr = report.get("test_real") or {}
    L += _test_real_md(tr)

    # ---- notes
    L += ["## Notes", ""]
    L += [f"- {n}" for n in report.get("notes", [])]
    return "\n".join(L).rstrip() + "\n"


def _ci(c: Any) -> str:
    if not isinstance(c, dict):
        return _num(c)
    return fmt_cell({"mean": c.get("mean"), "lo": c.get("lo"), "hi": c.get("hi"), "n": c.get("n")})


def _test_real_md(tr: dict[str, Any]) -> list[str]:
    if tr.get("status") != "embedded":
        why = tr.get("error") or "`lstmabar eval sim-real` writes `reports/sim_real.json`"
        return [f"Not run ({why}).", ""]
    L = [
        "From `reports/sim_real.json`, aggregate sections only (per-item sections omitted: "
        + (", ".join(tr.get("omitted_sections") or []) or "none")
        + "). internal_only groups appear only as group aggregates. Text-driven methods are "
        "not evaluated on test_real in P5 (no instruction text, D11).",
        "",
    ]
    groups = tr.get("groups") or {}
    if groups:
        L += ["### Groups", ""]
        L += _table(
            ["group", "items", "evaluated", "internal_only"],
            [
                [g, str(v.get("n_total")), str(v.get("n_evaluated")), str(v.get("internal_only"))]
                for g, v in groups.items()
            ],
        )
    kb = tr.get("kb") or {}
    if kb:
        L += ["### KB render vs real wet (loudness-matched)", ""]
        body = []
        for g, v in kb.items():
            for c in v.get("cells") or []:
                body.append(
                    [
                        g,
                        str(v.get("kb_pedal")),
                        str(c.get("engine")),
                        str(c.get("level")),
                        _ci(c.get("mrstft")),
                        _ci(c.get("centroid_oct")),
                        _ci(c.get("odd_even_db")),
                        _ci(c.get("hnr_db")),
                    ]
                )
        L += _table(
            [
                "group",
                "KB pedal",
                "engine",
                "input dBFS",
                "MR-STFT",
                "|d centroid| oct",
                "|d odd/even| dB",
                "|d HNR| dB",
            ],
            body,
        )
    fam = tr.get("family_shifts") or {}
    if fam:
        L += ["### Family-level descriptor shifts (wet - dry): real vs synthetic family", ""]
        body = []
        for g, v in fam.items():
            for term, t in (v.get("terms") or {}).items():
                body.append(
                    [
                        g,
                        str(v.get("family")),
                        term,
                        _ci(t.get("real")),
                        _ci(t.get("synthetic_family")),
                    ]
                )
        L += _table(["group", "family", "term", "real", "synthetic family"], body)
    return L


def _method_cells_by_source(report, metric, subset, idx) -> list[list[str]]:
    sources = list(_gold_sources())
    body = []
    for m in report["methods"]:
        st = _status(m, GOLD)
        if st:
            body.append([m["name"], st] + [""] * (len(sources) - 1))
            continue
        body.append(
            [m["name"]] + [fmt_cell(idx.get((m["name"], GOLD, metric, subset, s))) for s in sources]
        )
    return body


def _gate_table(report, split, idx, present) -> list[str]:
    blocks = sorted(
        m.split("/", 1)[1] for s, m in present if s == split and m.startswith("gate_f1/")
    )
    if not blocks:
        return []
    blocks = [b for b in blocks if b != "macro"] + (["macro"] if "macro" in blocks else [])
    out = [f"Gate F1 (`{split}`, over queries)", ""]
    body = []
    for m in report["methods"]:
        if _status(m, split):
            continue
        cells = []
        for b in blocks:
            r = idx.get((m["name"], split, f"gate_f1/{b}", "all", NO_SOURCE))
            if r is None or not int(r.get("n") or 0) or not _finite(r.get("mean")):
                cells.append("n/a")
            else:
                c = f"{_num(r['mean'])} ({r['n']})"
                if int(r.get("n_seeds") or 1) > 1:
                    c += f"; seed SD {_num(r.get('seed_sd'))}"
                cells.append(c)
        body.append([m["name"], *cells])
    return out + _table(["method", *blocks], body)


def write_report(report: dict[str, Any], out_dir: str | Path, name: str) -> tuple[Path, Path]:
    """Write ``<out_dir>/<name>.md`` and ``<name>.json``; returns ``(md, json)`` paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = _clean_json(report)
    json_path = out / f"{name}.json"
    md_path = out / f"{name}.md"
    json_path.write_text(
        json.dumps(rep, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    md_path.write_text(render_markdown(rep), encoding="utf-8", newline="\n")
    return md_path, json_path


__all__ = [
    "REPORT_SCHEMA",
    "ComparisonRow",
    "Row",
    "build_comparisons",
    "build_report",
    "build_rows",
    "compute_hashes",
    "fmt_cell",
    "render_markdown",
    "write_report",
]
