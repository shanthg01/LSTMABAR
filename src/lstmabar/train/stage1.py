"""P6 stage 1 (task 6.2): ``L_param`` on cached embeddings (CPU, minutes on the 24.5k train
items). ``lstmabar train stage1 [--limit N] [--fixtures] [dotlist overrides]``.

- Data: :class:`lstmabar.train.dataset.Stage1Dataset` on the train split and on
  ``eval.select_split`` (val); gold is never loaded here.
- Loss: :func:`lstmabar.train.losses.param_loss` (masked L1 + gate BCE, weights in
  ``train.loss``), clamped grey-box items weighted by ``train.clamped_weight``.
- AdamW + linear warmup + cosine decay sized to ``epochs x steps_per_epoch`` of the real data.
- Early stopping on the val **parameter part** of the D8 gold-proxy: ``param_weight x`` masked
  param L1 (normalized knobs) ``+ gate_weight x`` gate BCE, unweighted, over every (val item,
  caption) pair. The full proxy score (MR-STFT + 2AFC) of the chosen checkpoint is computed by
  the eval runner (``runner.score_val``) later, not here.
- Outputs (``runs/<name>/<stamp>/``, via ``create_run_dir``): ``config.yaml`` + ``meta.json``,
  ``metrics.json`` (rewritten every epoch), ``checkpoint.pt`` (the best epoch: state dict,
  resolved config and git SHA embedded) and ``summary.json``.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from lstmabar.eval.data import load_split
from lstmabar.models.embed_cache import FAMILIES, EmbeddingTable, cache_path, l2_normalize
from lstmabar.models.embed_cache import text_sha256 as _text_sha
from lstmabar.models.predictor import ParamPredictor, PredictorConfig, save_checkpoint
from lstmabar.runs import create_run_dir, git_info
from lstmabar.seed import seed_everything
from lstmabar.train.dataset import Batch, Stage1Dataset, load_tables
from lstmabar.train.losses import gate_bce, masked_l1, param_loss

FORBIDDEN_SELECT_SPLITS = ("gold", "test", "test_unseen_pedal")


# ============================================================================ model / targets


def check_model_cfg(cfg) -> None:
    m = cfg.model
    if str(m.inputs) != "text" and FAMILIES.get(str(m.audio_model)) != "clap":
        raise ValueError(f"model.audio_model must be a CLAP checkpoint, got {m.audio_model!r}")
    if str(m.inputs) != "audio" and str(m.text_model) not in FAMILIES:
        raise ValueError(f"unknown model.text_model {m.text_model!r}")


def build_model(cfg, text_dim: int, audio_dim: int) -> ParamPredictor:
    m = cfg.model
    return ParamPredictor(
        PredictorConfig(
            text_dim=int(text_dim),
            audio_dim=int(audio_dim),
            inputs=str(m.inputs),
            mapping=str(m.mapping),
            adapter_dim=int(m.adapter_dim),
            hidden_dim=int(m.hidden_dim),
            dropout=float(m.dropout),
        )
    )


def _forward(model: ParamPredictor, b: Batch) -> dict[str, torch.Tensor]:
    return model(b.text, b.audio)


def _loss(model: ParamPredictor, out, b: Batch, cfg, weighted: bool = True):
    """``L_param`` in the mapping's own space (normalized knobs for ``physics``, linear
    physical values for ``unconstrained``, see :mod:`lstmabar.models.predictor`)."""
    if model.config.mapping == "physics":
        pred, target = out["knobs"], b.knobs
    else:
        pred, target = out["raw"], model.heads_layout.normalized_to_linear(b.knobs)
    lw = cfg.train.loss
    return param_loss(
        pred,
        out["gate_logits"],
        target,
        b.gates,
        b.knob_mask,
        b.weights if weighted else None,
        param_weight=float(lw.param_weight),
        gate_weight=float(lw.gate_weight),
    )


# ============================================================================ evaluation


@torch.no_grad()
def evaluate(model: ParamPredictor, ds: Stage1Dataset, cfg, batch_size: int = 1024) -> dict:
    """Val metrics over every (item, caption) query, unweighted, in normalized knob space for
    both mappings: ``param_l1`` (masked by label gates), ``gate_bce``, ``gate_acc``, per-block
    ``gate_f1`` (threshold ``eval.metrics.gate_threshold``) and the selection ``score``."""
    model.eval()
    knobs, gates, mask, logits = [], [], [], []
    for b in ds.eval_batches(batch_size):
        out = _forward(model, b)
        knobs.append((out["knobs"], b.knobs))
        gates.append(b.gates)
        mask.append(b.knob_mask)
        logits.append(out["gate_logits"])
    pred_k = torch.cat([k[0] for k in knobs])
    true_k = torch.cat([k[1] for k in knobs])
    g_true = torch.cat(gates)
    g_logit = torch.cat(logits)
    m = torch.cat(mask)
    l1 = float(masked_l1(pred_k, true_k, m))
    bce = float(gate_bce(g_logit, g_true))
    thr = float(cfg.eval.metrics.gate_threshold)
    g_pred = (torch.sigmoid(g_logit) >= thr).float()
    f1 = {}
    for j, name in enumerate(model.heads_layout.blocks):
        tp = float((g_pred[:, j] * g_true[:, j]).sum())
        fp = float((g_pred[:, j] * (1 - g_true[:, j])).sum())
        fn = float(((1 - g_pred[:, j]) * g_true[:, j]).sum())
        f1[name] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None
    lw = cfg.train.loss
    return {
        "n_queries": int(pred_k.shape[0]),
        "param_l1": l1,
        "gate_bce": bce,
        "gate_acc": float((g_pred == g_true).float().mean()),
        "gate_f1": f1,
        "score": float(lw.param_weight) * l1 + float(lw.gate_weight) * bce,
    }


# ============================================================================ training


def lr_lambda(step: int, total: int, warmup: int, min_frac: float) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def train_model(
    cfg, train_ds: Stage1Dataset, val_ds: Stage1Dataset, run_dir: Path | None = None
) -> dict[str, Any]:
    """Train with early stopping on the val selection score. Returns a summary (best epoch,
    val metrics, history) and, with ``run_dir``, writes ``metrics.json`` per epoch and the best
    ``checkpoint.pt``. Seeded by ``cfg.seed`` (dropout, batching, caption sampling)."""
    seed = int(cfg.seed)
    seed_everything(seed)
    tc = cfg.train
    if tc.get("num_threads"):
        # A tiny MLP: many threads only add contention on a busy CPU (10 threads were
        # ~15x slower per step than 1 on the dev machine).
        torch.set_num_threads(int(tc.num_threads))
    text_dim = train_ds.text.shape[1] if train_ds.text is not None else 0
    audio_dim = train_ds.audio.shape[1] if train_ds.audio is not None else 0
    model = build_model(cfg, text_dim, audio_dim)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tc.lr), weight_decay=float(tc.weight_decay)
    )
    bs = int(tc.batch_size)
    epochs = int(tc.epochs)
    steps = train_ds.steps_per_epoch(bs)
    total = epochs * steps
    warmup = int(round(float(tc.warmup_frac) * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total, warmup, float(tc.min_lr_frac))
    )
    gen = torch.Generator().manual_seed(seed)
    shuffle_audio = bool(cfg.model.shuffle_audio)
    patience = int(tc.patience)
    grad_clip = float(tc.grad_clip) if tc.get("grad_clip") else 0.0

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best: dict[str, Any] = {"score": math.inf, "epoch": -1}
    best_state = None
    bad = 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        sums = {"loss": 0.0, "param_l1": 0.0, "gate_bce": 0.0}
        n = 0
        for b in train_ds.train_batches(bs, gen, permute_audio=shuffle_audio):
            out = _forward(model, b)
            loss, parts = _loss(model, out, b, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            for k in sums:
                sums[k] += parts[k] * len(b)
            n += len(b)
        val = evaluate(model, val_ds, cfg)
        row = {
            "epoch": epoch,
            "step": (epoch + 1) * steps,
            "lr": float(sched.get_last_lr()[0]),
            "train": {k: v / max(n, 1) for k, v in sums.items()},
            "val": val,
            "seconds": round(time.perf_counter() - t0, 2),
        }
        history.append(row)
        if val["score"] < best["score"] - float(tc.min_delta):
            best = {"score": val["score"], "epoch": epoch, "val": val}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if run_dir is not None:
            (run_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if bad >= patience:
            break
    if best_state is None:
        raise RuntimeError(
            f"stage1: no finite val score in {len(history)} epochs (NaN/inf loss?); "
            "check the embeddings and train.lr"
        )
    model.load_state_dict(best_state)
    model.eval()
    summary = {
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "steps_per_epoch": steps,
        "total_steps_planned": total,
        "val": best["val"],
        "n_train_items": len(train_ds),
        "n_val_items": len(val_ds),
        "n_train_dropped_no_caption": train_ds.n_dropped,
        "n_train_clamped_weighted": int((train_ds.weights != 1.0).sum()),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    if run_dir is not None:
        meta = {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "git": git_info(),
            "text_model": str(cfg.model.text_model) if model.config.inputs != "audio" else None,
            "audio_model": str(cfg.model.audio_model) if model.config.inputs != "text" else None,
            "epoch": best["epoch"],
            "val": best["val"],
            "seed": seed,
        }
        save_checkpoint(run_dir / "checkpoint.pt", model, meta)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["model"] = model
    return summary


# ============================================================================ fixtures


def write_fixture_embeddings(
    cfg, splits=("train", "val", "test", "test_unseen_pedal"), dim: int = 32
) -> None:
    """Fake, deterministic embedding tables for the synthetic eval fixture: a text vector is
    a function of the caption **text** (equal texts share a vector, so captions carry signal),
    an audio vector a function of the item id. For tests and ``--fixtures`` smoke runs only."""

    def vec(key: str) -> np.ndarray:
        rng = np.random.default_rng(int(_text_sha(key)[:15], 16))
        return l2_normalize(rng.standard_normal(dim).astype(np.float32))

    cache_dir = str(cfg.eval.embeddings.dir)
    pairs = [(str(cfg.model.text_model), "text")]
    if FAMILIES[str(cfg.model.audio_model)] == "clap":
        pairs.append((str(cfg.model.audio_model), "audio"))
    for split in splits:
        items = load_split(cfg, split)
        caps = [c for i in items for c in i.captions]
        for model, kind in pairs:
            if kind == "text":
                ids = [c.caption_id for c in caps]
                vecs = [vec(c.text) for c in caps]
                shas = [_text_sha(c.text) for c in caps]
            else:
                ids = [i.item_id for i in items]
                vecs = [vec("audio:" + i.item_id) for i in items]
                shas = ["fixture"] * len(ids)
            path = cache_path(cache_dir, model, split, kind)
            EmbeddingTable(ids, np.stack(vecs), shas, model, kind).save(path)


def fixture_config(cfg, root: str | Path) -> DictConfig:
    """``cfg``'s model/train/eval.model sections on top of the eval fixture paths."""
    from lstmabar.eval.fixtures import eval_fixture_config

    base = eval_fixture_config(root)
    keep = {
        "name": cfg.get("name", "train_stage1"),
        "seed": cfg.get("seed", 0),
        "model": cfg.model,
        "train": cfg.train,
        "eval": {"model": cfg.eval.get("model", {})},
    }
    return OmegaConf.merge(base, OmegaConf.create(keep))


# ============================================================================ entry point


def load_data(cfg, limit: int | None = None) -> tuple[Stage1Dataset, Stage1Dataset]:
    select = str(cfg.eval.select_split)
    if select in FORBIDDEN_SELECT_SPLITS:
        raise ValueError(f"eval.select_split must be a val split, not {select!r} (D8)")
    check_model_cfg(cfg)
    weight = float(cfg.train.clamped_weight)
    sets = []
    for split in ("train", select):
        items = load_split(cfg, split, limit)
        text, audio = load_tables(cfg, split)
        sets.append(Stage1Dataset(items, text, audio, clamped_weight=weight))
    return sets[0], sets[1]


def run(cfg, args) -> int:
    if not getattr(args, "fixtures", False):
        return _run(cfg, args)
    from lstmabar.eval.fixtures import make_eval_fixture

    # The fixture data (audio + fake embeddings) lives in a temp dir removed afterwards; the
    # run dir (config, metrics, checkpoint) stays under paths.runs.
    with tempfile.TemporaryDirectory(prefix="lstmabar_stage1_fx_") as root:
        runs_root = cfg.paths.runs
        cfg = fixture_config(cfg, root)
        cfg.paths.runs = runs_root
        make_eval_fixture(cfg, seconds=0.5)
        write_fixture_embeddings(cfg)
        return _run(cfg, args)


def _run(cfg, args) -> int:
    limit = getattr(args, "limit", None)
    if limit is None and cfg.eval.get("max_items") is not None:
        limit = int(cfg.eval.max_items)
    seed_everything(int(cfg.seed))
    train_ds, val_ds = load_data(cfg, limit)
    run_dir = create_run_dir(cfg, root=str(cfg.paths.runs))
    summary = train_model(cfg, train_ds, val_ds, run_dir)
    v = summary["val"]
    f1 = ", ".join(f"{k} {x:.3f}" if x is not None else f"{k} n/a" for k, x in v["gate_f1"].items())
    print(
        f"stage1: best epoch {summary['best_epoch']} of {summary['epochs_run']} "
        f"({summary['seconds']:.0f} s); val param L1 {v['param_l1']:.4f}, gate BCE "
        f"{v['gate_bce']:.4f}, gate acc {v['gate_acc']:.3f}; gate F1 {f1}"
    )
    print(f"run dir: {run_dir.name} (under paths.runs/{cfg.name})")
    return 0


__all__ = [
    "build_model",
    "evaluate",
    "fixture_config",
    "load_data",
    "lr_lambda",
    "run",
    "train_model",
    "write_fixture_embeddings",
]
