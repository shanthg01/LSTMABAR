# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LSTMABAR v2: text-driven guitar tone transformation grounded in pedal circuit physics. Guitar audio + instruction → parameters for a differentiable, physically grounded pedalboard → rendered audio + explanation. Solo side project; the main deliverable is a Gradio demo.

Read [docs/design.md](docs/design.md) (architecture, data, evaluation) and [docs/execution-plan.md](docs/execution-plan.md) (phases P0–P8, exit gates, milestones) before substantial work. Work happens on `v2/...` feature branches, not `main`.

`legacy/` holds the v1 W266 submission (tag `v1-w266`) and is read-only reference. Its design flaws are documented in design §1; do not reuse its labelling scheme, DDSP engine, or RLHF code.

## Commands

The environment is managed by `uv` (Python 3.11 pinned in `.python-version`; CPU torch from the PyTorch CPU index). On this Windows machine bare `python` is the Store stub, so use `uv run` or `py`.

```bash
uv sync --extra dev                      # add --extra dsp/analysis/clap/demo/quantum as needed
uv run pytest                            # all tests
uv run pytest tests/test_foundation.py::test_cli_smoke   # single test
uv run ruff check .                      # lint (legacy/ excluded)
uv run lstmabar info                     # versions, device, git state (CLI: src/lstmabar/cli.py)
uv run lstmabar smoke                    # config -> seed -> run-dir sanity check
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest on Ubuntu with CPU torch.

## Architecture conventions

- Code lives in `src/lstmabar/` subpackages: `dsp/` (differentiable pedalboard), `physics/` (pedal knowledge base, circuit → grey-box param derivations, `whitebox/` offline sims), `analysis/` (harmonics, descriptors, archetype readout), `data/`, `models/`, `train/`, `eval/`, `demo/`.
- Configs are YAML in `configs/`, loaded by `lstmabar.config.load_config`. Configs compose via a `defaults:` list of relative paths, then dotlist overrides (`seed=1 paths.runs=...`).
- Every experiment calls `seed_everything` and writes to a run dir from `lstmabar.runs.create_run_dir` (`runs/<name>/<timestamp>/` with resolved `config.yaml` + `meta.json` holding the git SHA). `runs/` and `data/{raw,rendered,cache}/` are gitignored.
- Heavy training runs on Colab via `colab/bootstrap.ipynb`, which only clones, installs and calls the CLI. Never put project logic in notebooks; notebooks import from `src/`.
- The "archetypes" (sine/triangle/square/saw + noise) are a *measured* harmonic readout of audio in v2, not a prediction target.
- Every result is reported against the baselines in design §7 with multiple seeds.

## Evaluation data hygiene

`data/gold/` holds held-out test instructions. **Never read `data/gold/` when writing or tuning training caption templates or caption generation code** (design §5). Human-written items (`source: human`) are the headline test set; `instructions_llm_draft.yaml` is an LLM-drafted supplement.
