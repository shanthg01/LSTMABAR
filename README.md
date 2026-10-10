# LSTMABAR v2

Text-driven guitar tone transformation grounded in pedal circuit physics. Give it a guitar recording and an instruction like *"warm, mid-forward overdrive, not too fizzy"*; it sets the knobs of a differentiable, physically grounded pedalboard, renders the result, and explains it (nearest real pedal, harmonic/archetype shift).

v2 is a solo side project building on the W266 course project. The original version is archived in [legacy/](legacy/) (tag `v1-w266`).

- Design: [docs/design.md](docs/design.md)
- Execution plan and status: [docs/execution-plan.md](docs/execution-plan.md)
- Dataset licenses: [docs/data-licenses.md](docs/data-licenses.md)

**Status:**
- **Done:** M1, a playable differentiable pedalboard (compressor → drive → EQ) with a Gradio demo; parameter recovery passes as an audio match ([report](reports/param_recovery.md)).
- **Next:** M2, physics-grounded presets from real pedal circuits.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev          # core + test tooling (CPU torch)
uv run pytest                # tests (add -m slow for the ~7 min parameter-recovery check)
uv run ruff check .          # lint (CI also runs ruff format --check)
uv run lstmabar info         # versions, device, git state
uv run lstmabar smoke        # config -> seed -> run dir sanity check
```

Optional extras: `demo` (Gradio + PyAV), `spikes`, `clap`, `analysis`, `quantum`; `dsp` (dasp-pytorch, auraloss) is currently unused, since the pedalboard is pure PyTorch. Install them with, for example, `uv sync --extra dev --extra demo`.

## Demo

A local Gradio page with manual knobs on the differentiable pedalboard and dry/wet A/B playback:

```bash
uv sync --extra dev --extra demo
uv run lstmabar demo                 # http://127.0.0.1:7860 (--port N)
```

Upload or record a clip (trimmed to 15 s, uploads up to 25 MB), or pick a synthetic example riff. Knobs are generated from each effect block's parameter specs; log-taper knobs slide over their position and show the physical value underneath. Any common format works (WAV, FLAC, MP3, OGG, M4A/AAC, WebM, ...): files are decoded with libsndfile, falling back to PyAV, which bundles its own ffmpeg, so no system ffmpeg is needed.

M2 features (install `--extra analysis` for the archetype panel and `--extra whitebox` for white-box renders): the **Pedal** dropdown picks a pedal from the knowledge base (`pedals/`) and shows its own knobs on a 0–10 scale. Moving a knob derives the grey-box settings from the circuit and writes them into the board sliders, with the derivation's key quantities and approximation/clamp notes underneath. While a pedal is active the EQ (sliders and on/off) is locked, since it holds the pedal's fitted tone stack. The other sliders stay editable, and a render after you edit one is labelled "custom". Pedals with a white-box circuit simulation get a third player (C, first 10 s of the clip, loudness-matched like B), rendered with the same knob settings as B; it needs the `whitebox` extra (numba) and is disabled with a message otherwise. The archetype panel (off by default; tick the box, it adds a few seconds per render) shows the sine/triangle/square/saw + noise mix and H1–H10 levels (dBc) of dry vs wet (vs white-box) on the loudest 3 s. When the pitch tracker finds no stable pitch (e.g. silence or dense chords) it says so; it often locks onto the root of a chord, and noise shows up as a large noise share.

`--share` creates a public URL that runs anyone's uploads on your machine; add `--auth user:pass` to require a login.

Heavy training runs on Colab via [colab/bootstrap.ipynb](colab/bootstrap.ipynb), using the same CLI.
