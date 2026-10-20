"""``lstmabar eval results`` (P6 task 6.7): ``reports/v2_results.{md,json}``.

Collects, from finished run dirs (no scoring):

- the main eval run dir (baselines + the main ``p6_model``, 5 seeds),
- extra model run dirs (e.g. the bge-text variant, scored as ``p6_model`` in its own dir),
- the Text2FX subset run dir (``--limit``; every method on the same items),
- stage-1 training summaries of the main config and each ablation (val metrics, 5 seeds),

and writes the headline tables: gold 2AFC (MR-STFT, headline subset) per LLM-written source
(D9: never pooled), synthetic test / unseen-pedal metrics, paired comparisons of each model
against the best baseline, the ablation table and the Text2FX subset. Every cell is a 95 %
bootstrap CI over items (seed-averaged per item for multi-seed methods) with ``n``; seed SDs
of the split means are reported next to them. The hand-written discussion
(``reports/v2_results_discussion.md``) is appended verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from lstmabar.eval.stats import bootstrap_ci, compare, seed_summary

SOURCES = ("simulated_human", "llm_persona", "llm_draft")


def _units(path: Path) -> dict[str, dict]:
    return {
        u["unit_id"]: u
        for u in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x)
    }


def _seeds(run_dir: Path, method: str) -> list[int]:
    root = run_dir / "scores" / method
    return sorted(int(p.name.removeprefix("seed")) for p in root.glob("seed*"))


def unit_scores(run_dir: Path, method: str, split: str, key: str, keep=None):
    """``(ids, seed-averaged per-unit values, per-seed split means)``."""
    per = [
        _units(run_dir / "scores" / method / f"seed{s}" / f"{split}.jsonl")
        for s in _seeds(run_dir, method)
    ]
    ids = sorted(i for i in per[0] if keep is None or keep(per[0][i]))
    vals = np.array([np.nanmean([p[i]["scores"].get(key, np.nan) for p in per]) for i in ids])
    seed_means = [float(np.nanmean([p[i]["scores"].get(key, np.nan) for i in ids])) for p in per]
    return ids, vals, seed_means


def _cell(vals, seed_means, seed: int) -> dict[str, Any]:
    ci = bootstrap_ci(vals, seed=seed)
    out = ci.to_dict()
    if len(seed_means) > 1:
        out["seed_sd"] = seed_summary(seed_means)["sd"]
        out["n_seeds"] = len(seed_means)
    return out


def _gold_keep(source: str):
    def keep(u: dict) -> bool:
        m = u["meta"]
        return m.get("gold_source") == source and "headline" in m.get("subsets", [])

    return keep


def collect(cfg) -> dict[str, Any]:
    r = cfg.results
    seed = int(cfg.eval.seed)
    main = Path(str(r.main_run))
    methods: dict[str, tuple[Path, str]] = {m: (main, m) for m in r.main_methods}
    for name, spec in (r.get("extra_models") or {}).items():
        methods[name] = (Path(str(spec.run_dir)), str(spec.method))
    out: dict[str, Any] = {"methods": list(methods), "gold": {}, "synthetic": {}, "paired": []}

    for s in SOURCES:
        out["gold"][s] = {}
        for name, (rd, m) in methods.items():
            _, v, sm = unit_scores(rd, m, "gold", "afc_mrstft", _gold_keep(s))
            out["gold"][s][name] = _cell(v, sm, seed)
    for split in ("test", "test_unseen_pedal"):
        out["synthetic"][split] = {}
        for key in r.synthetic_metrics:
            out["synthetic"][split][key] = {}
            for name, (rd, m) in methods.items():
                _, v, sm = unit_scores(rd, m, split, key)
                out["synthetic"][split][key][name] = _cell(v, sm, seed)

    ref = str(r.reference)
    for name in r.compare:
        rows = []
        for s in SOURCES:
            a_ids, a, _ = unit_scores(*methods[name], "gold", "afc_mrstft", _gold_keep(s))
            b_ids, b, _ = unit_scores(*methods[ref], "gold", "afc_mrstft", _gold_keep(s))
            assert a_ids == b_ids, "paired comparison needs the same gold items"
            rows.append(
                {
                    "split": "gold",
                    "source": s,
                    "metric": "afc_mrstft",
                    **compare(a, b, seed=seed).to_dict(),
                }
            )
        for split in ("test", "test_unseen_pedal"):
            a_ids, a, _ = unit_scores(*methods[name], split, "mrstft")
            b_ids, b, _ = unit_scores(*methods[ref], split, "mrstft")
            assert a_ids == b_ids
            rows.append(
                {
                    "split": split,
                    "source": "-",
                    "metric": "mrstft",
                    **compare(a, b, seed=seed).to_dict(),
                }
            )
        out["paired"].append({"a": name, "b": ref, "rows": rows})

    out["ablations"] = {}
    for name, pattern in r.ablations.items():
        summaries = [json.loads(p.read_text("utf-8")) for p in sorted(Path(".").glob(str(pattern)))]
        if not summaries:
            continue
        row: dict[str, Any] = {"n_seeds": len(summaries)}
        for key in ("param_l1", "gate_acc", "gate_bce"):
            row[key] = seed_summary([s["val"][key] for s in summaries])
        row["gate_f1_compressor"] = seed_summary(
            [s["val"]["gate_f1"]["compressor"] for s in summaries]
        )
        out["ablations"][name] = row

    sub = Path(str(r.subset_run))
    out["subset"] = {"run": f"{sub.parent.name}/{sub.name}", "gold": {}, "test": {}}
    sub_methods = sorted(p.name for p in (sub / "scores").iterdir() if p.is_dir())
    for name in sub_methods:
        for s in SOURCES:
            ids, v, sm = unit_scores(sub, name, "gold", "afc_mrstft", _gold_keep(s))
            if ids:
                out["subset"]["gold"].setdefault(s, {})[name] = _cell(v, sm, seed)
        for key in ("mrstft", "param_l1"):
            _, v, sm = unit_scores(sub, name, "test", key)
            out["subset"]["test"].setdefault(key, {})[name] = _cell(v, sm, seed)
    out["subset"]["methods"] = sub_methods
    t2 = json.loads((sub / "methods" / "text2fx.json").read_text("utf-8"))["info"]
    out["subset"]["text2fx_info"] = {
        k: t2.get(k)
        for k in (
            "device",
            "steps",
            "n_optimized",
            "seconds_per_query",
            "mean_sim_init",
            "mean_sim_returned",
        )
    }
    return out


def _fmt(c: dict[str, Any] | None, nd: int = 3) -> str:
    if not c or c.get("n", 0) == 0:
        return "n/a"
    s = f"{c['mean']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}] ({c['n']})"
    if c.get("seed_sd") is not None and c.get("n_seeds", 1) > 1:
        s += f"; seed SD {c['seed_sd']:.3f}"
    return s


def render(res: dict[str, Any], discussion: str = "") -> str:
    L = ["# v2 results (P6 stage 1)", ""]
    L.append(
        "Generated by `lstmabar eval results` from the run dirs in `configs/eval/v2_results.yaml`. "
        "Every cell: mean [95 % bootstrap CI over items] (n items); multi-seed methods are "
        "averaged per item over their seeds, with the SD of the per-seed means. Gold numbers are "
        "the headline subset (in scope, not ambiguous, not unseen-pedal) of each **LLM-written** "
        "gold set, reported separately (D9; no real human items yet). Higher is better for 2AFC "
        "(chance 0.5), lower for MR-STFT and L1."
    )
    L += ["", "## Gold 2AFC (MR-STFT), headline", ""]
    L.append("| method | " + " | ".join(SOURCES) + " |")
    L.append("|---|" + "---|" * len(SOURCES))
    for m in res["methods"]:
        L.append(f"| {m} | " + " | ".join(_fmt(res["gold"][s].get(m)) for s in SOURCES) + " |")
    for split in ("test", "test_unseen_pedal"):
        L += ["", f"## Synthetic `{split}`", ""]
        keys = list(res["synthetic"][split])
        L.append("| method | " + " | ".join(keys) + " |")
        L.append("|---|" + "---|" * len(keys))
        for m in res["methods"]:
            L.append(
                f"| {m} | "
                + " | ".join(_fmt(res["synthetic"][split][k].get(m)) for k in keys)
                + " |"
            )
    L += ["", "## Paired comparisons (A - B, items paired)", ""]
    L.append(
        "| A | B | split | source | metric | diff [95 % CI] (n) | Wilcoxon p | Cliff's delta |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for p in res["paired"]:
        for row in p["rows"]:
            d = row["diff"]
            L.append(
                f"| {p['a']} | {p['b']} | {row['split']} | {row['source']} | {row['metric']} | "
                f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] ({d['n']}) | "
                f"{row['p_wilcoxon']:.3g} | {row['cliffs_delta']:+.3f} |"
            )
    L += ["", "## Stage-1 ablations (val, selection split; mean ± SD over seeds)", ""]
    L.append("| variant | seeds | masked param L1 | gate acc | gate BCE | compressor gate F1 |")
    L.append("|---|---|---|---|---|---|")
    for name, row in res["ablations"].items():
        cells = [
            f"{row[k]['mean']:.4f} ± {row[k]['sd']:.4f}"
            for k in ("param_l1", "gate_acc", "gate_bce", "gate_f1_compressor")
        ]
        L.append(f"| {name} | {row['n_seeds']} | " + " | ".join(cells) + " |")
    sub = res["subset"]
    t2 = sub["text2fx_info"]
    L += ["", "## Text2FX subset (first 100 items per split, every method on the same items)", ""]
    L.append(
        f"Run `{sub['run']}`. Text2FX ran on Colab ({t2['device']}, {t2['steps']} steps, "
        f"{t2['n_optimized']} optimizations, {t2['seconds_per_query']:.1f} s each; mean CLAP "
        f"similarity {t2['mean_sim_init']:.3f} at init -> {t2['mean_sim_returned']:.3f} for the "
        "returned boards). The first 100 gold ids are all `llm_draft` items, so the subset's gold "
        "column covers that source only."
    )
    L += [
        "",
        "| method | gold headline 2AFC ("
        + ", ".join(sub["gold"])
        + ") | test MR-STFT | test param L1 |",
        "|---|---|---|---|",
    ]
    for m in sub["methods"]:
        g = "; ".join(_fmt(sub["gold"][s].get(m)) for s in sub["gold"])
        t_mr = _fmt(sub["test"]["mrstft"].get(m))
        t_l1 = _fmt(sub["test"]["param_l1"].get(m))
        L.append(f"| {m} | {g} | {t_mr} | {t_l1} |")
    if discussion:
        L += ["", discussion.strip()]
    return "\n".join(L) + "\n"


def run(cfg, args) -> int:
    res = collect(cfg)
    out = Path(str(cfg.results.report_dir))
    out.mkdir(parents=True, exist_ok=True)
    disc = Path(str(cfg.results.discussion))
    text = disc.read_text(encoding="utf-8") if disc.exists() else ""
    (out / "v2_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    (out / "v2_results.md").write_text(render(res, text), encoding="utf-8", newline="\n")
    print("[eval results] v2_results.md, v2_results.json", flush=True)
    return 0


__all__ = ["collect", "render", "run", "unit_scores"]
