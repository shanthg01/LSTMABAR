# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LSTMABAR v2: text-driven guitar tone transformation grounded in pedal circuit physics. Guitar audio + instruction → parameters for a differentiable, physically grounded pedalboard → rendered audio + explanation. Solo side project; the main deliverable is a Gradio demo.

**Status:** P0 and P1 are done (M1 demo works). **Next: P2** (pedal knowledge base + white-box circuit sims), with P3 (harmonic analysis) in parallel. Start from the "P2 kickoff brief" in [docs/execution-plan.md](docs/execution-plan.md).

Read [docs/design.md](docs/design.md) (architecture, data, evaluation; §4.3 describes the implemented DSP; §10 is the decisions log) and [docs/execution-plan.md](docs/execution-plan.md) (phases, exit gates, status) before substantial work.

`legacy/` holds the v1 W266 submission (tag `v1-w266`) and is read-only reference. Its design flaws are documented in design §1; do not reuse its labelling scheme, DDSP engine, or RLHF code.

## Commands

The environment is managed by `uv` (Python 3.11 pinned in `.python-version`; CPU torch from the PyTorch CPU index). On this Windows machine bare `python` is the Store stub, so use `uv run` or `py`.

```bash
uv sync --extra dev                      # core + tests; add --extra demo (gradio, PyAV), whitebox (numba), analysis (librosa), ...
uv run pytest                            # all tests except those marked slow
uv run pytest tests/test_foundation.py::test_cli_smoke   # single test
uv run pytest -m slow                    # slow checks only (P1.7 parameter recovery, ~7 min CPU)
uv run ruff check . && uv run ruff format --check .      # lint + format (CI enforces both)
uv run lstmabar demo                     # M2 Gradio demo: pedal presets, white-box A/B, archetype panel (needs --extra demo; whitebox/analysis for A/B and panel)
uv run lstmabar recover                  # parameter-recovery run → reports/param_recovery.{md,json}
uv run lstmabar fidelity                 # P2 grey-box vs white-box gate → reports/greybox_fidelity.{md,json} (~20 min; --quick)
uv run lstmabar info                     # versions, device, git state (CLI: src/lstmabar/cli.py)
```

CI (`.github/workflows/ci.yml`) runs ruff check + format check + pytest on Ubuntu, Python 3.11 and 3.13 (Colab's version), with the `dev` and `whitebox` extras: tests needing gradio/PyAV or librosa skip there. Slow tests are local only.

## Architecture conventions

- Code lives in `src/lstmabar/`: `dsp/` (implemented differentiable pedalboard), `demo/` (Gradio app; `render.py` holds the gradio-free logic), `audio.py` (decode/resample/loudness/synth riffs), `physics/` (P2: pedal KB, circuit → grey-box derivations, `whitebox/` offline sims), `analysis/` (P3), `data/`, `models/`, `train/`, `eval/` (later phases).
- **DSP contracts (`dsp/base.py`):**
  - Audio is `(B, T)`; knobs are `(B,)`.
  - `EffectBlock`s take **normalized** knobs in [0, 1] and map them to physical units via `ParamSpec` (linear/log taper). Low-level functions (`filters`, `waveshaper`, `oversample`, `compressor`) take physical units.
  - Knobs are constant per clip, processing is zero-latency, and gating/ordering belongs to `Pedalboard`.
  - Optimize knobs as logits through a sigmoid; the clamp zeroes gradients outside [0, 1].
- **Pedalboard (`dsp/pedalboard.py`):** `BoardParams = {block: {knob: (B,), "enabled": (B,)}}`; `default_pedalboard()` = compressor → drive → eq.
- **Changing knobs or the chain:** adjust a block's ranges or chain only together with design §4.3, and re-pass `lstmabar recover` (the P1 exit gate is an audio-match criterion; don't claim parameter recovery beyond what the report shows).
- **Demo:** UI controls are generated from `param_specs` (never hard-code knob names). Uploads are decoded by `audio.decode_audio` (libsndfile → PyAV, no system ffmpeg). User-facing errors must never include server paths.
- **Configs and runs:** configs are YAML in `configs/`, loaded by `lstmabar.config.load_config` (`defaults:` composition + dotlist overrides). Experiments call `seed_everything` and write to `lstmabar.runs.create_run_dir` (`runs/<name>/<timestamp>/` with resolved config + git SHA). `runs/` and `data/{raw,rendered,cache}/` are gitignored.
- **Colab:** heavy training runs via `colab/bootstrap.ipynb`, which only clones, installs and calls the CLI. No project logic in notebooks.
- **Archetypes:** the archetypes (sine/triangle/square/saw + noise) are a *measured* harmonic readout of audio, not a prediction target.
- **Results:** every result is reported against the baselines in design §7 with multiple seeds.

## Workflow (how this project is run)

- Never commit to `main`; use a `v2/...` branch per task or wave.
- Parallelize independent work with subagents in isolated worktrees, after first merging a shared contract (stubs with exact signatures) to `main` via a PR.
- Every PR gets an **independent review agent**. Fix blockers (and cheap advisories), re-review, wait for green CI, then merge. Post the review summary as a PR comment.
- Clean up merged branches and worktrees afterwards.
- Reports must not overclaim: numbers in docs must match the generated data.

## Evaluation data hygiene

`data/gold/` holds held-out test instructions. **Never read `data/gold/` when writing or tuning training caption templates or caption generation code** (design §5). Real human items (`source: human`) are the headline test set once added; `simulated_human`, `llm_persona` and `llm_draft` items are LLM-written and must be reported separately and never relabeled.
