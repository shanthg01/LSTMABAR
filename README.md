# LSTMABAR v2

Text-driven guitar tone transformation grounded in pedal circuit physics. Give it a guitar recording and an instruction like *"warm, mid-forward overdrive, not too fizzy"*; it sets the knobs of a differentiable, physically grounded pedalboard, renders the result, and explains it (nearest real pedal, harmonic/archetype shift).

v2 is a solo side project building on the W266 course project. The original version is archived in [legacy/](legacy/) (tag `v1-w266`).

- Design: [docs/design.md](docs/design.md)
- Execution plan: [docs/execution-plan.md](docs/execution-plan.md)

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev          # core + test tooling (CPU torch)
uv run pytest                # tests
uv run ruff check .          # lint
uv run lstmabar info         # versions, device, git state
uv run lstmabar smoke        # config -> seed -> run dir sanity check
```

Optional extras: `dsp`, `analysis`, `clap`, `demo`, `quantum`. Install them with, for example, `uv sync --extra dev --extra demo`.

## Demo

A local Gradio page with manual knobs on the differentiable pedalboard and dry/wet A/B playback:

```bash
uv sync --extra dev --extra demo
uv run lstmabar demo                 # http://127.0.0.1:7860 (--port N, --share for a public link)
```

Upload or record a clip (capped at 15 s), or pick a synthetic example riff. Knobs are generated from each effect block's parameter specs.

Heavy training runs on Colab via [colab/bootstrap.ipynb](colab/bootstrap.ipynb), using the same CLI.
