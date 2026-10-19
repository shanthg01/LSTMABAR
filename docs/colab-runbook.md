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
O=../lstmabar_colab_upload && mkdir -p $O
uv run python - <<'EOF' | tr -d '\r' > $O/colab_wets.txt
from lstmabar.config import load_config
from lstmabar.eval.data import load_split
cfg = load_config("configs/eval/p5.yaml")
for split in ("test", "test_unseen_pedal"):
    for it in load_split(cfg, split):
        print(it.item.wet_path)  # relative to data/rendered
EOF
O=$(cd $O && pwd)
(cd data/rendered && ls v1/wet/*/items.jsonl v1/wet/*/captions.jsonl v1/wet/*/DONE v1/wet/plan.json > $O/colab_meta.txt \
  && tar -czf $O/lstmabar_rendered.tgz v1/dry gold_v1 -T $O/colab_meta.txt -T $O/colab_wets.txt)
tar -czf $O/lstmabar_rest.tgz data/manifests data/calibration data/cache/f0 data/cache/clap data/cache/bge
```

Upload `lstmabar_rendered.tgz` and `lstmabar_rest.tgz` (from `../lstmabar_colab_upload/`) to `MyDrive/lstmabar/` (the Drive web
UI is fine). Do **not** upload `data/test_real/`, `data/sources/` or `data/cache/llm/`.

## 2. Notebook cells (after the bootstrap cells)

Runtime → GPU. Run the bootstrap notebook's clone + mount cells, then:

```python
!pip install -q -e ".[dev,clap,dsp,analysis,whitebox]"
# Extract to the VM's local disk (much faster than reading from Drive), keep outputs on Drive.
!mkdir -p /content/data/rendered
!tar -xzf {DRIVE_ROOT}/lstmabar_rendered.tgz -C /content/data/rendered
!tar -xzf {DRIVE_ROOT}/lstmabar_rest.tgz -C /content   # creates /content/data/{manifests,...}
# Gold instructions are committed in the repo; paths.data must contain them too.
!mkdir -p /content/data/gold && cp -r /content/LSTMABAR/data/gold/. /content/data/gold/
DATA = "/content/data"
```

The HF checkpoints download on first use (Colab has network; no `HF_HUB_OFFLINE`).

## 3. Text2FX run

The eval run dir must be the **same one** as the local no-op/mean/kNN run, so all methods end
up in one report. Copy it to Drive first (from the local machine, e.g. zip
`runs/eval_p5/<stamp>/` and unzip to `MyDrive/lstmabar/runs/eval_p5/<stamp>/`). Resuming on
another machine works because the run's `fingerprint.json` is path-independent (it records the
dataset and gold-index hashes, not `paths.data`); a different dataset or eval setting is
refused with a diff. Run dirs created before this (no `fingerprint.json`) cannot be resumed;
score files written before it are re-scored.

```python
RUN = f"{DRIVE_ROOT}/runs/eval_p5/<stamp>"
# 1) timing smoke (2 items per split, a throwaway run dir; writes p5_baselines_smoke.*)
!lstmabar eval run --methods text2fx --splits test gold --limit 2 paths.data={DATA} paths.runs=/content/runs eval.report_dir=/content/reports eval.baselines.text2fx.max_items=null
# 2) the real run into the shared run dir (resumable: re-run the same cell after a disconnect)
!lstmabar eval run --run-dir {RUN} --methods text2fx --splits test test_unseen_pedal gold paths.data={DATA} eval.report_dir=/content/reports eval.baselines.text2fx.max_items=null
```

- Text2FX has no hyperparameter grid, so its selection step does no optimization; `--splits` leaves out the
  val table (2,752 optimizations) on purpose.
- Size: test 1,074 + unseen 531 + gold 1,350 = 2,955 optimizations. Decide from the smoke
  timing: if it projects beyond a session (~10 h), run a deterministic subset with `--limit N`
  in a **separate** run dir (the limit is part of the fingerprint) and report it as a subset.
- `eval.baselines.text2fx.*` only enters Text2FX's own score fingerprint, so this override
  does not invalidate the other methods' scores in the shared run dir.
- Afterwards copy `RUN` back to the local `runs/eval_p5/<stamp>/` and rebuild the report
  locally: `uv run lstmabar eval report --run-dir runs/eval_p5/<stamp>`.

## 4. P6 stage 2 (later)

Needs the full `data/rendered/v1/` (7.1 GB: add the train wets) plus `data/cache/clap/` and a
stage-1 checkpoint; commands land with `lstmabar train stage2` (P6 prep notes, "Colab data").
