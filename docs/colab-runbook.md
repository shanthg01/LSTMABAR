# Colab runbook (P5 Text2FX, later P6 stage 2)

Where work runs (decided 2026-10-14):

| Job | Where | Why |
|---|---|---|
| Embedding cache (`lstmabar eval embed`) | local | text tables are done; audio runs at ~2 clips/s on the quieter machine (val/test/unseen ≈ 1–2 h, train ≈ 3–4 h per CLAP checkpoint). Move to Colab only if it stalls |
| No-op / mean / kNN / P6 stage 1 scoring and training | local | CPU-bound, cheap once the machine is not overloaded |
| **Text2FX** (`--methods text2fx`) | **Colab GPU** | ~26 min per optimization at 200 steps on the local CPU |
| P6 stage 2 + seeds | Colab GPU | `L_audio` through the oversampled pedalboard (`lstmabar train stage2`, not implemented yet) |
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
`MyDrive/lstmabar/` (the Drive web UI is fine). Do **not** upload `data/test_real/`, `data/sources/` or `data/cache/llm/`.

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
!tar -xzf {DRIVE_ROOT}/lstmabar_rest.tgz -C /content   # -> /content/data/{manifests,calibration,cache}
!mkdir -p /content/data/gold && cp -r /content/LSTMABAR/data/gold/. /content/data/gold/
DATA = "/content/data"
RUN = f"{DRIVE_ROOT}/runs/eval_p5/20261014-204858"   # the shared local run dir, copied to Drive
!ls {DATA} {RUN}
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
  `runs/eval_p5/20261014-204858/` and rebuild the report locally:
  `uv run lstmabar eval report --run-dir runs/eval_p5/20261014-204858`.

## 4. P6 stage 2 (later)

Needs the full `data/rendered/v1/` (7.1 GB: add the train wets) plus `data/cache/clap/` and a
stage-1 checkpoint; commands land with `lstmabar train stage2` (P6 prep notes, "Colab data").
