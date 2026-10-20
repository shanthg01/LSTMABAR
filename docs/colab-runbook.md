# Colab runbook (P5 Text2FX, later P6 stage 2)

Where work runs (decided 2026-10-14):

| Job | Where | Why |
|---|---|---|
| Embedding cache (`lstmabar eval embed`) | local | text tables are done; audio runs at ~2 clips/s on the quieter machine (val/test/unseen ≈ 1–2 h, train ≈ 3–4 h per CLAP checkpoint). Move to Colab only if it stalls |
| No-op / mean / kNN / P6 stage 1 scoring and training | local | CPU-bound, cheap once the machine is not overloaded |
| **Text2FX** (`--methods text2fx`) | **Colab GPU** | ~26 min per optimization at 200 steps on the local CPU |
| P6 stage 2 + seeds | Colab GPU | `L_audio` through the oversampled pedalboard (`lstmabar train stage2`, section 4) |
| pOD-set (`internal_only`) audio | local only | licence (CC BY-NC, internal); nothing on Colab needs it |

Colab uses the same CLI as local runs (`colab/bootstrap.ipynb`); no project logic in the notebook.

## 1. Upload bundle for Text2FX (~2 GB)

Text2FX scores `test`, `test_unseen_pedal` and gold, so it needs the dry clips, the shard
metadata, the test/unseen wets, the gold renders, the manifests, the pYIN track cache and the
CLAP text tables. From the repo root in Git Bash on the local machine (the archives go to a
folder **outside** the repo, so they can't be committed by accident; the `tr -d '\r'` strips
the CRLF line endings Windows Python prints, which `tar -T` would otherwise treat as part of
each file name):

```bash
set -o pipefail
O=../lstmabar_colab_upload && mkdir -p "$O"
uv run python - <<'EOF' | tr -d '\r' > "$O/colab_wets.txt"
from lstmabar.config import load_config
from lstmabar.eval.data import load_split
cfg = load_config("configs/eval/p5.yaml")
for split in ("test", "test_unseen_pedal"):
    for it in load_split(cfg, split):
        print(it.item.wet_path)  # relative to data/rendered
EOF
O=$(cd "$O" && pwd)
wc -l < "$O/colab_wets.txt"   # expect 840 (560 test + 280 unseen wets)
(cd data/rendered && ls v1/wet/*/items.jsonl v1/wet/*/captions.jsonl v1/wet/*/DONE v1/wet/plan.json > "$O/colab_meta.txt" \
  && tar -czf "$O/lstmabar_rendered.tgz" v1/dry gold_v1 -T "$O/colab_meta.txt" -T "$O/colab_wets.txt")
tar -czf "$O/lstmabar_rest.tgz" data/manifests data/calibration data/cache/f0 data/cache/clap data/cache/bge
```

Upload `lstmabar_rendered.tgz` and `lstmabar_rest.tgz` (from `../lstmabar_colab_upload/`) to
`MyDrive/lstmabar/` (the Drive web UI is fine).

Also upload the shared run dir's three top-level files (the scores stay local):
create `MyDrive/lstmabar/runs/eval_p5/20261014-204858/` and upload `config.yaml`,
`fingerprint.json` and `meta.json` from the local `runs/eval_p5/20261014-204858/`. Without
them Cell C stops with `not a run dir (no config.yaml)`. Do **not** upload `data/test_real/`, `data/sources/` or `data/cache/llm/`.

## 2. Notebook cells

Open `colab/bootstrap.ipynb` in Colab (File → Open notebook → GitHub → `shanthg01/LSTMABAR`),
set Runtime → Change runtime type → **T4 GPU**, and run its cells (clone, install, mount
Drive, info/smoke). Then add these cells below them (+ Code) and run them in order.
File → Save a copy in Drive keeps the added cells.

**Cell A: setup** (extras, data to the VM's local disk, gold instructions):

```python
!pip install -q -e ".[dev,clap,dsp,analysis,whitebox]"
!mkdir -p /content/data/rendered
!tar -xzf {DRIVE_ROOT}/lstmabar_rendered.tgz -C /content/data/rendered
!tar -xzf {DRIVE_ROOT}/lstmabar_rest.tgz -C /content
# (rest.tgz extracts to /content/data/manifests, calibration and cache. No braces in comments on
# `!` lines: IPython then skips expanding {DRIVE_ROOT} on that line.)
!mkdir -p /content/data/gold && cp -r /content/LSTMABAR/data/gold/. /content/data/gold/
DATA = "/content/data"
RUN = f"{DRIVE_ROOT}/runs/eval_p5/20261014-204858"   # the shared local run dir, copied to Drive
!ls -d {RUN} {DATA}/manifests/v1/dry_clips.jsonl {DATA}/rendered/v1/dry {DATA}/rendered/gold_v1   # all must exist
# A failed ! command does not stop the cell: read the output for "No such file" / "Error".
!python -c "import torch; print('cuda', torch.cuda.is_available())"
```

`cuda True` is required (Text2FX uses the GPU via `eval.baselines.text2fx.device: auto`). The
CLAP checkpoint downloads from the Hugging Face Hub on first use (~0.7 GB).

## 3. Text2FX run

The run dir is the **same one** as the local no-op/mean/kNN run, so every method ends up in one
report. Resuming on another machine works because the run's `fingerprint.json` is
path-independent (dataset and gold-index hashes, not `paths.data`); a different dataset or
eval setting is refused with a diff. Run dirs created before this (no `fingerprint.json`)
cannot be resumed; score files written before it are re-scored.

Dotlist overrides go **before** `--methods` / `--splits` (those flags take a list and would
swallow anything after them).

**Cell B: timing smoke** (2 items per split, a throwaway run dir on the VM; note the time):

```python
%%time
!lstmabar eval run paths.data={DATA} paths.runs=/content/runs eval.report_dir=/content/reports eval.baselines.text2fx.max_items=null --workers 1 --limit 2 --methods text2fx --splits test gold
```

That is 10 optimizations (2 test items x 2 captions + 2 gold items x 3 clips); divide the wall
time by 10 for seconds per optimization.

**Cell C: the real run** into the shared run dir on Drive (resumable: re-run the same cell
after a disconnect; finished splits are kept):

```python
!lstmabar eval run paths.data={DATA} eval.report_dir=/content/reports eval.baselines.text2fx.max_items=null --run-dir {RUN} --workers 1 --methods text2fx --splits test test_unseen_pedal gold
```

- Text2FX has no hyperparameter grid, so its selection step does no optimization; `--splits`
  leaves out the val table (2,752 optimizations) on purpose.
- Size: test 1,074 + unseen 531 + gold 1,350 = 2,955 optimizations. If the smoke timing
  projects beyond a session (~10 h on the free tier), run a deterministic subset with
  `--limit N` in a **separate** run dir (the limit is part of the fingerprint) and report it
  as a subset.
- `eval.baselines.text2fx.*` only enters Text2FX's own score fingerprint, so these overrides
  do not invalidate the other methods' scores in the shared run dir.
- Afterwards download `RUN/scores/text2fx/` and `RUN/methods/text2fx.json` into the local
  `runs/eval_p5/20261014-204858/` (do **not** copy Colab's `run.json` over the local one)
  and rebuild the report locally:
  `uv run lstmabar eval report --run-dir runs/eval_p5/20261014-204858`. The report includes
  every method that has scores in the run dir.
- GPU scores are not bitwise reproducible (CUDA atomic adds in the backward pass);
  `methods/text2fx.json` records the device that produced them.

## 4. P6 stage 2 (`lstmabar train stage2`)

Stage 2 fine-tunes the stage-1 **bge-text** checkpoints (`runs/stage1_ab_bge_text/seed{seed}/`,
the best stage-1 variant) with `L_param + L_audio` through the oversampled pedalboard
(`configs/train/stage2.yaml`, `src/lstmabar/train/stage2.py`). It reads the dry clips and
**every** train wet, so it needs the full `data/rendered/v1/`, not the section 1 bundle.

**Cost.** Measured on the local CPU (4 threads, busy machine), batch 32 with 1–1.5 s crops:
**7.8 s/step** end to end (render + MR-STFT + backward ≈ 7–12 s/step in isolation; decoding a
crop ≈ 14 ms/item). One seed is 8 epochs × 767 steps = 6,136 steps (fewer if early stopping
fires): ~13 h per seed on this CPU, so local training is out. GPU speed is **not measured
yet**. The filters and the oversampler are FFT/conv ops, and the compressor's smoothing loop
runs in NumPy on the host. Run Cell E first and extrapolate from its s/step:
`hours per seed ≈ s/step × 6,136 / 3,600`, plus the val evaluations (step 0 and one per epoch).

### Upload bundle (~9 GB)

From the **main checkout's** repo root in Git Bash (archives go **outside** the repo). No file
lists are built with Python here, so the CRLF issue from section 1 does not arise. FLAC is
already compressed, so the rendered audio goes into a plain `.tar`:

```bash
set -o pipefail
O=../lstmabar_colab_upload && mkdir -p "$O" && O=$(cd "$O" && pwd)
(cd data/rendered && tar -cf "$O/lstmabar_rendered_full.tar" v1 gold_v1 valproxy_v1)
tar -czf "$O/lstmabar_rest.tgz" data/manifests data/calibration data/cache/f0 data/cache/clap data/cache/bge
tar -czf "$O/lstmabar_stage1_bge.tgz" runs/stage1_ab_bge_text/seed*/checkpoint.pt
tar -tf "$O/lstmabar_rendered_full.tar" | grep -c '\.flac$'   # dry clips + all wets + gold + val-proxy renders
tar -tzf "$O/lstmabar_stage1_bge.tgz"                        # expect 5 checkpoint.pt files
```

- `lstmabar_rendered_full.tar` (~8.6 GB: 7.1 + 0.98 + 0.54): `v1/` (dry + all 108 wet shards with their
  `items.jsonl` / `captions.jsonl` / `DONE` and `plan.json`), `gold_v1/` (gold scoring) and
  `valproxy_v1/` (the val 2AFC table in `eval run`). It replaces section 1's
  `lstmabar_rendered.tgz`; delete that one from Drive if space is tight (free tier: 15 GB).
- `lstmabar_rest.tgz` is section 1's command, re-run so the caches are current (it holds the
  CLAP and bge tables of every split, train included).
- `lstmabar_stage1_bge.tgz`: the five stage-1 bge checkpoints (~4 MB each).

Upload the three files to `MyDrive/lstmabar/`. Nothing `internal_only` is included.

### Cells

Run the bootstrap cells (T4 GPU runtime) as in section 2, then add these.

**Cell D: setup** (data to the VM's disk, checkpoints into the clone):

```python
!pip install -q -e ".[dev,clap,dsp,analysis,whitebox]"
!mkdir -p /content/data/rendered
!tar -xf {DRIVE_ROOT}/lstmabar_rendered_full.tar -C /content/data/rendered
!tar -xzf {DRIVE_ROOT}/lstmabar_rest.tgz -C /content
!tar -xzf {DRIVE_ROOT}/lstmabar_stage1_bge.tgz -C /content/LSTMABAR
!mkdir -p /content/data/gold && cp -r /content/LSTMABAR/data/gold/. /content/data/gold/
DATA = "/content/data"
OUT = DRIVE_ROOT + "/runs"
S2 = DRIVE_ROOT + "/runs/stage2_bge/seed{seed}/checkpoint.pt"  # literal "{seed}": a template
!ls -d {DATA}/rendered/v1/dry {DATA}/rendered/v1/wet/plan.json {DATA}/rendered/gold_v1 {DATA}/rendered/valproxy_v1 {DATA}/manifests/v1/dry_clips.jsonl {DATA}/cache/bge runs/stage1_ab_bge_text/seed4/checkpoint.pt
# A failed ! command does not stop the cell: read the output for "No such file" / "Error".
!python -c "import torch; print('cuda', torch.cuda.is_available())"
```

`cuda True` is required. `S2` is a plain string (not an f-string), so `{seed}` stays literal;
IPython expands `{S2}` on a `!` line once and does not re-expand the inserted text.

**Cell E: timing smoke** (50 steps, a throwaway run dir on the VM; read `s/step` from the
last printed line):

```python
!lstmabar train stage2 --config configs/train/stage2.yaml paths.data={DATA} paths.runs=/content/runs stage2.max_steps=50 stage2.eval_every=25
```

**Cell F: train 5 seeds** (each seed's run dir goes to Drive under `runs/stage2_bge_s<N>/`, and
the best checkpoint is copied to the `S2` template; after a disconnect re-run the cell: seeds
whose checkpoint exists are skipped, and an interrupted seed restarts from its stage-1
checkpoint):

```python
import os
for s in range(5):
    if os.path.exists(S2.replace("{seed}", str(s))):
        print("seed", s, "done")
        continue
    !lstmabar train stage2 --config configs/train/stage2.yaml paths.data={DATA} paths.runs={OUT} seed={s} name=stage2_bge_s{s} stage2.export_checkpoint={S2}
```

- The default `stage2.init_checkpoint` is `runs/stage1_ab_bge_text/seed{seed}/checkpoint.pt`
  relative to the clone (`/content/LSTMABAR`), which Cell D fills.
- Selection is on val only (stage-1 parameter criterion + MR-STFT on 128 fixed val items);
  step 0, i.e. the stage-1 checkpoint itself, is a candidate, so `best_step: 0` in
  `summary.json` means stage 2 did not improve on stage 1.
- GPU runs are not bitwise reproducible (CUDA atomic adds); `summary.json` records the
  device and `seconds_per_step`.

**Cell G: score** as `p6_model` in its own run dir (dotlist overrides **before**
`--methods`):

```python
!lstmabar eval run paths.data={DATA} paths.runs={OUT} name=eval_p6_stage2 eval.report_dir=/content/reports eval.report_name=p6_stage2 eval.model.checkpoint={S2} --workers 2 --methods p6_model
```

- This scores val, `test`, `test_unseen_pedal` and gold for the 5 seeds, like the local
  `eval_p6_bge` run. Scoring is CPU-bound (renders + harmonic analysis), and Colab has 2 vCPUs.
  It can equally run locally: download `MyDrive/lstmabar/runs/stage2_bge/` into the local
  `runs/stage2_bge/` and run the same command there without `paths.data` / `paths.runs` and
  with `eval.model.checkpoint=runs/stage2_bge/seed{seed}/checkpoint.pt`.
- Afterwards download `runs/eval_p6_stage2/` and `runs/stage2_bge_s*/` (configs, metrics,
  summaries) into the local `runs/`, and add the run to `results.extra_models` in
  `configs/eval/v2_results.yaml` for `lstmabar eval results`.
