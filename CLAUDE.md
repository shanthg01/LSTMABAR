# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LSTMABAR (Language-to-Sound Transformation Model using Archetype-Based Audio Representation) — a W266 (UC Berkeley NLP) final project by Shanth Gopalswamy and Zain Nasim. Research code, not a packaged library: no requirements file, build system, linter, or test suite. The final paper and slides are the PDFs at the repo root; `README.md` and `architecture_diagram.mermaid` describe the intended design.

## Running code

All Python modules live flat in `scripts/` and import each other by bare module name (`from text_tower import TextEncoder`), so run everything with `scripts/` as the working directory:

```bash
cd scripts
python audio_tower.py            # each component module has a __main__ smoke demo
python lstmabar_model.py         # end-to-end demo of the full model
python musiccaps_loader.py       # needs musiccaps-public.csv (Kaggle) in cwd; downloads clips via yt_dlp
```

Dependencies (inferred from imports): torch, torchaudio, transformers, sentence-transformers, pennylane, librosa, soundfile, scipy, numpy, pandas, scikit-learn, matplotlib, seaborn, yt_dlp, tqdm.

Training, tuning, RLHF and evaluation happen in notebooks in `scripts/` (`training_pipeline.ipynb`, `rlhf_pipeline.ipynb`, `rlhf_hybrid_quantum.ipynb`, `best_model_evaluation.ipynb`, `evaluation_metrics.ipynb`). Notebooks in `notebooks/` are earlier per-component explorations.

## Architecture

`lstmabar_model.py` wires the pipeline; each stage is its own module:

1. **Text tower** (`text_tower.py` → `TextEncoder`): sentence-transformer (default `all-MiniLM-L6-v2`) projected to `embedding_dim` (768). Optional `QuantumAttentionBlock` — a PennyLane `default.qubit` VQC controlled by `use_quantum_attention`, `n_qubits`, `circuit_depth`, `noise_strength`.
2. **Audio tower** (`audio_tower.py` → `AudioEncoder`): mel spectrogram → ResNet (`audio_architecture='resnet'`) or AST (`'ast'`), plus auxiliary archetype prediction.
3. **Contrastive alignment** (`contrastive_alignment.py`): CLAP-style InfoNCE with temperature + auxiliary match classifier.
4. **Archetype predictor** (`archetype_predictor.py`): MLP over `[d_t; d_a]` → softmax 5-vector. Also holds `ArchetypeLoss` and `RLHFTrainer` (policy-gradient fine-tuning from feedback).
5. **DDSP engine** (`ddsp_transformation.py`): decomposes audio into harmonic/noise/residual and re-mixes per archetype weights with learnable filters.

`LSTMABARTrainer` (same file as the model) handles AdamW + cosine LR, train/val steps, and history. `LSTMABAR.save_checkpoint` / `load_checkpoint` write `.pth` files.

**Archetype order is fixed everywhere: `['sine', 'square', 'sawtooth', 'triangle', 'noise']`** (loader, model, DDSP). Keep it consistent when touching any of them.

### Data flow

`musiccaps_loader.MusicCapsLoader` downloads MusicCaps clips to `scripts/musiccaps_audio/`, derives archetype weight targets from caption keywords (TF-IDF weighted), and saves `musiccaps_training_data_{train,val,test}.npz`. These, `scripts/checkpoints/`, and `.pth` weights are gitignored — they must be regenerated locally.

### Experiments and results

- `training_pipeline.ipynb` runs a hyperparameter sweep of named configs (`classical_baseline_small/large`, `classical_ast`, `quantum_{4,6,8}qubit_depth*`). Per-config checkpoints/plots go to `scripts/tuning_checkpoints/<name>/`; JSON results to `scripts/tuning_results/<name>_<timestamp>.json`; summary in `tuning_results/final_report.txt` and `summary.csv`.
- `best_model_evaluation.ipynb` writes test metrics, plots, and a LaTeX table to `scripts/evaluation_results/`.
- `improved_text_encoders.py` contains alternative text encoders (deeper projection heads, HF backbones, descriptor-aware) used by `evaluation_metrics.ipynb` for comparison; it is not wired into `LSTMABAR`.

### Legacy files

`*_old_ZN.py` / `*_old_ZN.ipynb` are superseded versions by a co-author kept for reference. Edit the non-suffixed files.
