"""``lstmabar eval run`` / ``eval report`` (task 5.4, D6).

``run``: for each method in ``eval.methods`` (``--methods``): build, ``fit(train)``, select
hyperparameters on val (``runner.select_hparams``, D8; never gold), score val (the selection
table), the synthetic test splits (``eval.splits``) and gold, write per-unit scores to a run
dir (``create_run_dir``), then the report (``<eval.report_dir>/<eval.report_name>.{md,json}``).
``report``: rebuild the report from a run dir's scores (no scoring).

Run dir layout (``runs/<name>/<stamp>/``)::

    config.yaml, meta.json                 create_run_dir
    scores/<method>/seed<k>/<split>.jsonl  one UnitScores per line (written when the split ends)
    scores/<method>/seed<k>/<split>.meta.json   SplitResult.extra (gate F1, failed renders, ...)
    scores/<method>/seed<k>/selection.json      chosen hparams, criterion, warnings
    methods/<method>.json                  status per split ("ok" / "not run: <reason>"), info

A split's ``.jsonl`` appears only once the split is scored (written to a temp name and
renamed), so a crash keeps the finished method x split files and ``run --run-dir <dir>`` resumes
by skipping them (a finished selection is reused, not re-run). Methods with
``deterministic=False`` run for ``eval.stats.min_seeds`` seeds.

Methods that cannot run (module not implemented, missing embedding cache, LLM cache miss
without spend, no checkpoint) are recorded per split as ``"not run: <reason>"``; other errors
propagate (resume later).

Harmonic descriptors are ~90 % of scoring time; ``eval.select_harmonic: false`` runs the val
selection (and the val table) with ``eval.metrics.harmonic_descriptors=false`` while the
reported splits keep the configured value. The report states where they were off.

``--fixtures`` runs on the synthetic eval fixture in a temp dir (data and run dir); the report
goes to ``eval.report_dir`` only when that is given as an override, else it is printed.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from lstmabar.runs import create_run_dir
from lstmabar.seed import seed_everything

SCORES_DIR = "scores"
METHODS_DIR = "methods"
SELECTION_FILE = "selection.json"
GOLD = "gold"

_ABS_PATH = re.compile(
    r"(?:(?<![\w])[A-Za-z]:[\\/]|\\\\|(?<![^\s'\"(=\[])/(?=[\w.~-]+/))[^\s'\"`,;)\]]*"
)


def scrub(text: str) -> str:
    """Replace absolute paths in ``text`` by ``<path>/<basename>`` (user-facing strings)."""

    def repl(m: re.Match) -> str:
        p = m.group(0).rstrip(".:")
        tail = m.group(0)[len(p) :]
        name = re.split(r"[\\/]", p.rstrip("\\/"))[-1]
        return f"<path>/{name}{tail}" if name else f"<path>{tail}"

    return _ABS_PATH.sub(repl, str(text))


# ============================================================================ not-run reasons


def _cannot_run_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [ImportError, NotImplementedError, FileNotFoundError]
    try:
        from lstmabar.eval.baselines.llm import LlmCacheMiss

        types.append(LlmCacheMiss)
    except ImportError:  # pragma: no cover
        pass
    return tuple(types)


def _reason(e: BaseException) -> str:
    if isinstance(e, ImportError | NotImplementedError):
        msg = str(e) or type(e).__name__
        return scrub(f"not implemented ({type(e).__name__}: {msg})")
    return scrub(f"{type(e).__name__}: {e}")


# ============================================================================ run-dir files


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _split_path(run_dir: Path, method: str, seed: int, split: str) -> Path:
    return run_dir / SCORES_DIR / method / f"seed{seed}" / f"{split}.jsonl"


def write_split(run_dir: Path, result, harmonic: bool) -> Path:
    """Write one finished SplitResult (meta first, then the units file atomically)."""
    path = _split_path(run_dir, result.method, result.seed, result.split)
    extra = {k: v for k, v in result.extra.items() if k != "proxy_preds"}
    _json_dump(
        {**extra, "harmonic_descriptors": bool(harmonic), "n_units": len(result.units)},
        path.with_suffix(".meta.json"),
    )
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for u in result.units:
            f.write(json.dumps(u.to_dict(), default=str) + "\n")
    os.replace(tmp, path)
    return path


def load_results(run_dir: str | Path) -> list:
    """Every finished SplitResult in a run dir (``preds`` are not stored: empty)."""
    from lstmabar.eval.runner import SplitResult, UnitScores

    out = []
    root = Path(run_dir) / SCORES_DIR
    if not root.exists():
        return out
    for path in sorted(root.glob("*/seed*/*.jsonl")):
        method = path.parent.parent.name
        seed = int(path.parent.name.removeprefix("seed"))
        units = [
            UnitScores(**json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        meta_path = path.with_suffix(".meta.json")
        extra = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        out.append(SplitResult(path.stem, method, seed, units, {}, extra))
    return out


def load_method_records(run_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(run_dir) / METHODS_DIR
    if not root.exists():
        return []
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*.json"))]
    order = {}
    try:
        order = {
            m: i for i, m in enumerate(OmegaConf.load(Path(run_dir) / "config.yaml").eval.methods)
        }
    except Exception:  # noqa: BLE001 - ordering only
        pass
    return sorted(recs, key=lambda r: (order.get(r["name"], 1 << 20), r["name"]))


# ============================================================================ helpers


def _limit(cfg, args) -> int | None:
    lim = getattr(args, "limit", None)
    if lim is None:
        lim = cfg.eval.get("max_items")
    return int(lim) if lim else None


def _report_splits(cfg, args) -> list[str]:
    """Splits to score: the val table, the synthetic test splits and gold (``--splits``)."""
    default = [str(cfg.eval.select_split), *[str(s) for s in cfg.eval.splits], GOLD]
    want = getattr(args, "splits", None)
    if not want:
        return default
    unknown = [s for s in want if s not in default]
    if unknown:
        raise SystemExit(f"eval run: unknown splits {unknown}; expected a subset of {default}")
    return [s for s in default if s in want]


def _selection_cfg(cfg):
    """Config for val selection: harmonic descriptors off if ``eval.select_harmonic`` is false."""
    if cfg.eval.get("select_harmonic", True) is False:
        return OmegaConf.merge(cfg, {"eval": {"metrics": {"harmonic_descriptors": False}}})
    return cfg


def _harmonic(cfg) -> bool:
    return bool(cfg.eval.metrics.get("harmonic_descriptors", True))


def _seeds(method, cfg) -> list[int]:
    base = int(cfg.eval.seed)
    if getattr(method, "deterministic", True):
        return [base]
    return [base + i for i in range(int(cfg.eval.stats.min_seeds))]


class _Data:
    """Lazy split loading (each split read once per run)."""

    def __init__(self, cfg, limit: int | None) -> None:
        self.cfg = cfg
        self.limit = limit
        self._cache: dict[str, Any] = {}

    def get(self, split: str):
        if split not in self._cache:
            from lstmabar.eval import data as D

            if split == GOLD:
                self._cache[split] = D.load_gold_items(self.cfg, self.limit)
            elif split == "train":  # fit on the full train split (only eval splits are capped)
                self._cache[split] = D.load_split(self.cfg, split)
            else:
                self._cache[split] = D.load_split(self.cfg, split, self.limit)
        return self._cache[split]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, str):
        return scrub(obj)
    if obj is None or isinstance(obj, bool | int | float):
        return obj
    return scrub(str(obj))


# ============================================================================ one method


def _select(method, data: _Data, cfg, sel_cfg, seed: int, sel_path: Path) -> dict[str, Any]:
    """Val selection (D8), reused from ``selection.json`` on resume."""
    from lstmabar.eval import runner

    if sel_path.exists():
        rec = json.loads(sel_path.read_text(encoding="utf-8"))
        if method.hparam_grid() and rec.get("chosen"):
            method.set_hparams(dict(rec["chosen"]))
        elif not method.hparam_grid():
            method.select(data.get(str(cfg.eval.select_split)))
        return rec
    val = data.get(str(cfg.eval.select_split))
    grid = method.hparam_grid()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sel = runner.select_hparams(method, val, sel_cfg, seed)
    msgs = list(dict.fromkeys(scrub(str(w.message)) for w in caught))
    info = method.info() if hasattr(method, "info") else {}
    if grid:
        proxy = not any("proxy" in m for m in msgs)
        criterion = "runner.select_hparams: proxy.selection_score on val" + (
            " (with the val gold-proxy 2AFC)" if proxy else " (synthetic val only, no proxy)"
        )
        if any("selection_score is not implemented" in m for m in msgs):
            criterion = (
                "runner.select_hparams: fallback weighted MR-STFT + param L1 on val "
                "(proxy.selection_score not implemented)"
            )
    else:
        criterion = (info.get("selection") or {}).get("criterion") or (
            "none (no hyperparameters)" if not sel.get("chosen") else "method.select(val)"
        )
    rec = {
        "chosen": sel.get("chosen") or {},
        "criterion": criterion,
        "grid": [dict(g) for g in grid],
        "scores": sel.get("scores", []),
        "fallback": bool(sel.get("fallback", False)),
        "harmonic_descriptors": _harmonic(sel_cfg),
        "n_val_items": len(val),
        "warnings": msgs,
    }
    _json_dump(_jsonable(rec), sel_path)
    return rec


def _score(method, split: str, data: _Data, cfg, sel_cfg, seed: int):
    from lstmabar.eval import runner

    if split == GOLD:
        return runner.score_gold(method, data.get(GOLD), cfg, seed), _harmonic(cfg)
    if split == str(cfg.eval.select_split):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # already recorded by the selection
            return runner.score_val(method, data.get(split), sel_cfg, seed), _harmonic(sel_cfg)
    return runner.score_synthetic(method, data.get(split), cfg, split, seed), _harmonic(cfg)


def run_method(name: str, cfg, run_dir: Path, splits: list[str], data: _Data) -> dict[str, Any]:
    """Score one method on ``splits`` for every seed it needs; returns its status record."""
    from lstmabar.eval.baselines.base import build_method

    cannot = _cannot_run_types()
    sel_cfg = _selection_cfg(cfg)
    rec: dict[str, Any] = {"name": name, "seeds": [], "splits": {}, "info": {}, "selection": {}}
    rec_path = run_dir / METHODS_DIR / f"{name}.json"
    seed0 = int(cfg.eval.seed)
    prev = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}

    def not_run(reason: str, which=None) -> dict[str, Any]:
        for s in which or splits:
            rec["splits"][s] = f"not run: {reason}"
        rec["status"] = (
            "not run" if all(v.startswith("not run") for v in rec["splits"].values()) else "partial"
        )
        _json_dump(_jsonable(rec), rec_path)
        print(f"[eval run] {name}: not run ({reason})", flush=True)
        return rec

    try:
        seed_everything(seed0)
        method = build_method(name, cfg, seed=seed0)
    except cannot as e:
        return not_run(_reason(e))
    except (ValueError, KeyError) as e:  # e.g. p6_model without a checkpoint
        return not_run(_reason(e))
    seeds = _seeds(method, cfg)
    rec["seeds"] = seeds
    rec["deterministic"] = bool(getattr(method, "deterministic", True))
    for seed in seeds:
        todo = [s for s in splits if not _split_path(run_dir, name, seed, s).exists()]
        if not todo:
            print(f"[eval run] {name} seed {seed}: all splits done (resume)", flush=True)
            continue
        try:
            if seed != seed0:
                seed_everything(seed)
                method = build_method(name, cfg, seed=seed)
            print(f"[eval run] {name} seed {seed}: fit", flush=True)
            method.fit(data.get("train"))
            sel = _select(
                method,
                data,
                cfg,
                sel_cfg,
                seed,
                run_dir / SCORES_DIR / name / f"seed{seed}" / SELECTION_FILE,
            )
            rec["selection"][str(seed)] = sel
        except cannot as e:
            return not_run(_reason(e), todo)
        for split in todo:
            try:
                print(f"[eval run] {name} seed {seed}: scoring {split}", flush=True)
                res, harmonic = _score(method, split, data, cfg, sel_cfg, seed)
            except cannot as e:
                not_run(_reason(e), [split])
                continue
            write_split(run_dir, res, harmonic)
        rec["info"] = method.info()
    for split in splits:
        done = all(_split_path(run_dir, name, s, split).exists() for s in seeds)
        if done:
            rec["splits"][split] = "ok"
        elif not str(rec["splits"].get(split, "")).startswith("not run"):
            rec["splits"][split] = "not run: incomplete"
    # selections of seeds finished in an earlier run
    for seed in seeds:
        p = run_dir / SCORES_DIR / name / f"seed{seed}" / SELECTION_FILE
        if str(seed) not in rec["selection"] and p.exists():
            rec["selection"][str(seed)] = json.loads(p.read_text(encoding="utf-8"))
    vals = list(rec["splits"].values())
    rec["status"] = (
        "ok"
        if all(v == "ok" for v in vals)
        else "not run"
        if not any(v == "ok" for v in vals)
        else "partial"
    )
    if not rec["info"]:  # every seed finished in an earlier run
        rec["info"] = prev.get("info", {})
    _json_dump(_jsonable(rec), rec_path)
    return rec


# ============================================================================ estimate


def _estimate(cfg, args) -> int:
    """Print the LLM baseline's cost for this run's texts; no spending."""
    from lstmabar.eval.baselines.llm import LlmZeroShot, estimate_lines

    data = _Data(cfg, _limit(cfg, args))
    texts: list[str] = []
    for split in _report_splits(cfg, args):
        texts += [q.text for it in data.get(split) for q in it.queries()]
    method = LlmZeroShot(cfg, seed=int(cfg.eval.seed))
    method.estimate_only = True
    for line in estimate_lines(method.estimate(texts)):
        print(line, flush=True)
    return 0


# ============================================================================ entry points


def _explicit(args, key: str) -> bool:
    return any(str(o).split("=", 1)[0].strip() == key for o in (args.overrides or []))


def _fixture_cfg(args, root: Path):
    from lstmabar.eval.fixtures import eval_fixture_config, make_eval_fixture

    overrides = list(getattr(args, "overrides", None) or [])
    cfg = eval_fixture_config(root, overrides)
    if not _explicit(args, "eval.workers"):
        cfg.eval.workers = 1
    make_eval_fixture(cfg, seconds=0.5)
    return cfg


def _run(cfg, args, fixtures: bool = False) -> int:
    if getattr(args, "workers", None):
        cfg.eval.workers = int(args.workers)
    limit = _limit(cfg, args)
    cfg.eval.max_items = limit  # runner._proxy_items reads it
    if getattr(args, "estimate", False):
        return _estimate(cfg, args)
    methods = list(getattr(args, "methods", None) or cfg.eval.methods)
    from lstmabar.eval.baselines.base import REGISTRY

    unknown = [m for m in methods if m not in REGISTRY]
    if unknown:  # a typo (or an override swallowed by --methods) must fail loudly
        raise SystemExit(f"eval run: unknown methods {unknown}; known: {sorted(REGISTRY)}")
    splits = _report_splits(cfg, args)
    if getattr(args, "run_dir", None):
        run_dir = Path(args.run_dir)
        if not (run_dir / "config.yaml").exists():
            raise SystemExit("eval run --run-dir: not a run dir (no config.yaml)")
        print(f"[eval run] resuming {run_dir.name}", flush=True)
    else:
        run_dir = create_run_dir(cfg, root=str(cfg.paths.runs), name=str(cfg.name))
    seed_everything(int(cfg.eval.seed))
    data = _Data(cfg, limit)
    for name in methods:
        run_method(name, cfg, run_dir, splits, data)
    extras = {"limit": limit, "fixtures": fixtures, "requested_methods": methods}
    _json_dump(extras, run_dir / "run.json")
    return _write(
        cfg, run_dir, extras, print_md=fixtures and not _explicit(args, "eval.report_dir")
    )


def _write(cfg, run_dir: Path, extras: dict[str, Any], print_md: bool = False) -> int:
    from lstmabar.eval.report import build_report, render_markdown, write_report

    results = load_results(run_dir)
    records = load_method_records(run_dir)
    names = extras.get("requested_methods") or [r["name"] for r in records]
    order = {n: i for i, n in enumerate(names)}
    records = sorted(
        [r for r in records if r["name"] in order] or records,
        key=lambda r: order.get(r["name"], len(order)),
    )
    keep = {r["name"] for r in records}
    results = [r for r in results if r.method in keep]
    rep = build_report(
        results,
        cfg,
        {**extras, "methods": records, "run_dir": f"{run_dir.parent.name}/{run_dir.name}"},
    )
    md_path, json_path = write_report(rep, cfg.eval.report_dir, str(cfg.eval.report_name))
    print(f"[eval run] report: {md_path.name}, {json_path.name}", flush=True)
    if print_md:
        md = render_markdown(rep)
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(md.encode(enc, errors="replace").decode(enc, errors="replace"))
    return 0


def run(cfg, args) -> int:
    if getattr(args, "fixtures", False):
        with tempfile.TemporaryDirectory(prefix="lstmabar-evalfx-") as tmp:
            fcfg = _fixture_cfg(args, Path(tmp))
            if _explicit(args, "paths.runs"):
                fcfg.paths.runs = cfg.paths.runs
            return _run(fcfg, args, fixtures=True)
    return _run(cfg, args)


def report(cfg, args) -> int:
    """Re-render the report from ``--run-dir`` (its saved config, plus dotlist overrides)."""
    if not getattr(args, "run_dir", None):
        raise SystemExit("eval report needs --run-dir")
    run_dir = Path(args.run_dir)
    if not (run_dir / "config.yaml").exists():
        raise SystemExit("eval report --run-dir: not a run dir (no config.yaml)")
    rcfg = OmegaConf.load(run_dir / "config.yaml")
    overrides = list(getattr(args, "overrides", None) or [])
    if overrides:
        rcfg = OmegaConf.merge(rcfg, OmegaConf.from_dotlist(overrides))
    extras_path = run_dir / "run.json"
    extras = json.loads(extras_path.read_text(encoding="utf-8")) if extras_path.exists() else {}
    if getattr(args, "methods", None):
        extras["requested_methods"] = list(args.methods)
    return _write(rcfg, run_dir, extras)


__all__ = ["load_method_records", "load_results", "report", "run", "run_method", "scrub"]
