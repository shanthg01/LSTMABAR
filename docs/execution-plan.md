# LSTMABAR v2 — Execution Plan

Companion to [design.md](design.md). This is a solo side project, assumed at ~8–12 h/week, with a demo as the main deliverable. Durations are relative sizing, not commitments.

Each phase ends with an **exit gate**: don't start dependent work until it passes. Each task is tagged with where it runs:
- **[L]** local CPU
- **[C]** Colab GPU

## Milestones

| Milestone | When (approx.) | What you can show |
|---|---|---|
| **M1 — Playable pedalboard** | end of P1 (~wk 4) | Gradio page with manual knobs on the differentiable Drive/EQ/Comp chain |
| **M2 — Physics-grounded presets** | end of P2 (~wk 7) | Pick "TS808 / RAT / Fuzz Face…" + knob positions; hear grey-box vs white-box; see the archetype shift |
| **M3 — Demo v0 (text, no model)** | end of P5 (~wk 13) | Type an instruction → best baseline (kNN retrieval / LLM zero-shot) sets knobs |
| **M4 — Demo v1 (model)** | end of P6 (~wk 17) | The trained model drives the knobs, with nearest-pedal explanation and comparison vs baseline |
| **M5 — Write-up + extensions** | wk 18+ | Blog/README write-up; P7 add-ons, including the quantum experiment |

The demo grows at every milestone, so there is always something working to play with.

## Timeline

```
Week:     1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18+
P0 Found. ████
P1 DSP          ████████                         M1 ▲(wk4)
P2 Physics               ███████                 M2 ▲(wk7)
P3 Readout               ████   (interleaved with P2)
P4 Data                         ██████████
P5 Eval/BL                            ████████   M3 ▲(wk13)
P6 Model                                    ████████████  M4 ▲(wk17)
P7 Ext.                                                  ──────►
```

Critical path: P0 → P1 → P2 → P4 → P6. Solo, so the "parallel" phases are interleaved, not concurrent.

---

## P0 — Foundation, gold set, risk spikes (weeks 1–2)

| # | Task | Where | Output |
|---|---|---|---|
| 0.1 | Tag current state `v1-w266`; move v1 `scripts/` + `notebooks/` to `legacy/` | L | Clean tree; v1 reproducible from the tag |
| 0.2 | `uv` env on Python 3.11/3.12; `pyproject.toml` + `uv.lock` with torch (CPU wheel locally), omegaconf, pytest, ruff; heavier deps (auraloss, dasp-pytorch, transformers for CLAP, librosa, gradio, pennylane) as optional extras, added to core only when a phase needs them | L | `uv sync` + `uv run pytest` work |
| 0.3 | `src/lstmabar/` skeleton (design §8); seed utility; config loading; run-dir convention; GitHub Actions running ruff + pytest | L | CI green |
| 0.4 | `colab/bootstrap.ipynb`: clone repo, `pip install -e .`, mount Drive for data/runs, run a CLI command on GPU | C | Smoke run on a Colab GPU |
| 0.5 | **Write the gold instructions now (~150), before any template exists.** For each: free text + a rough target description (pedal family, gain level, brightness). Store in `data/gold/instructions_raw.yaml`. Recruit ~5 guitarist friends to contribute ~50 more later | L | Uncontaminated gold text |
| 0.6 | **Spike A — CLAP vocabulary:** render ~5 tones (clean, soft OD, hard dist, fuzz, dark) with Spotify `pedalboard`; score against ~30 tone words with LAION-CLAP | L | Note: can CLAP rank them? Sets how much P6 leans on `L_clap` |
| 0.7 | **Spike B — licenses** for DI, real-pedal and capture sources | L | Done: [data-licenses.md](data-licenses.md). Follow-up emails to Fraunhofer, EGDB, TONE3000, GuitarML, EGFxSet authors |
| 0.8 | Update `CLAUDE.md` for the v2 layout and commands | L | — |

**Exit gate:** CI green; Colab smoke run works; ≥150 gold instructions committed; spike notes written.

**Status (2026-10-09): P0 exit gate passed.**
- CI green (Python 3.11 + 3.13).
- Colab smoke run OK on a Tesla T4 (Python 3.13, torch 2.11 + CUDA 13.0, commit 3bc4a7b).
- Gold set: 600 items across four files, all LLM-written; real `human` items are still to be added.
- Spike A ([CLAP findings](../spikes/clap_vocab/FINDINGS.md)) and spike B ([data licenses](data-licenses.md)) are done.
- Open follow-ups: license emails (see data-licenses.md), optional own DI recordings, rerun spike A on real DI.

## P1 — Differentiable DSP core (weeks 2–4) → M1

| # | Task | Where | Output |
|---|---|---|---|
| 1.1 | Biquad HPF/LPF/peak/shelf with differentiable coefficients; tilt EQ | L | `dsp/filters.py` |
| 1.2 | Oversampling wrapper (4×/8×, anti-alias) | L | `dsp/oversample.py` |
| 1.3 | Waveshaper family `f(x; softness, asymmetry, bias)`, smooth surrogate for hard clip | L | `dsp/waveshaper.py` |
| 1.4 | Blocks with `param_spec`: Drive, 3-band EQ, Compressor (wrap dasp where suitable), Gain | L | `dsp/blocks.py` |
| 1.5 | `Pedalboard` with gates, batched | L | `dsp/pedalboard.py` |
| 1.6 | Tests: filter response vs analytic; symmetric → odd harmonics only, asymmetric → evens appear; aliasing bound; every param gets non-zero gradient | L | `tests/dsp/` |
| 1.7 | **Parameter-recovery test:** random params → render → gradient-descend fresh params on MR-STFT → recover | L | `tests/dsp/test_recovery.py` |
| 1.8 | **M1 demo:** Gradio page, upload clip, manual knobs, A/B playback | L | `demo/app.py` (`lstmabar demo`) |

**Exit gate:** recovery ≥90% of trials on Drive + EQ; the M1 demo runs locally.

## P2 — Physics layer and pedal knowledge base (weeks 5–7) → M2

| # | Task | Where | Output |
|---|---|---|---|
| 2.1 | KB YAML schema + validator | L | `pedals/schema.yaml`, `physics/kb.py` |
| 2.2 | Author 5 pedals: TS808, RAT, DS-1, Fuzz Face (Si/Ge), Big Muff, using published circuit analyses as sources | L | `pedals/*.yaml` |
| 2.3 | Derivations: components + knob positions → grey-box params | L | `physics/derive.py` + tests |
| 2.4 | White-box sims for **2 circuits only** (diode clipper → TS/DS-1 class; Fuzz Face). Others use NAM captures if licensed, else grey-box only | L | `physics/whitebox/` |
| 2.5 | Fidelity check: fit grey-box to white-box; harmonic-spectrum dB error | L | `reports/greybox_fidelity.md` |
| 2.6 | **M2 demo:** pedal preset picker + KB knobs; grey-box vs white-box A/B | L | demo update |

**Exit gate:** mean harmonic error ≤ ~3 dB on the first 10 harmonics for the 2 white-box pedals (or deviations documented); M2 demo runs.

## P3 — Archetype readout and descriptors (interleaved, weeks 6–7)

| # | Task | Where | Output |
|---|---|---|---|
| 3.1 | f0 (pYIN) + harmonic amplitudes | L | `analysis/harmonics.py` |
| 3.2 | Odd/even ratio, harmonic slope, HNR, centroid, rolloff, crest | L | `analysis/descriptors.py` |
| 3.3 | NNLS projection → archetype vector + noise fraction | L | `analysis/archetypes.py` |
| 3.4 | Tests on pure oscillators; monotonic with drive gain | L | `tests/analysis/` |
| 3.5 | Demo panel: before/after archetype bars + harmonic plot | L | demo update |

**Exit gate:** oscillator tests pass; the demo shows the shift.

## P4 — Data pipeline (weeks 8–11)

| # | Task | Where | Output |
|---|---|---|---|
| 4.1 | Ingest licensed DI corpora (Guitar-TECHS, EGFxSet clean notes, ~30–60 min own DI recordings) → mono 44.1 kHz, 2–4 s onset-aligned clips + manifest | L | `data/sources/`, manifest |
| 4.2 | Renderer: dry × pedal × sampled knobs → wet (white-box for 2 pedals, NAM/grey-box for the rest); store params, pedal id, descriptors, archetype readout. Multiprocess; resumable shards | L (C if slow) | `data/render.py` |
| 4.3 | Caption generator: templates from params/descriptors/KB vocab + LLM paraphrase (Claude API), with no numeric leakage. **Do not look at the gold set while writing templates** | L | `data/captions.py` |
| 4.4 | Real-pedal test set (EGFxSet primary; pOD-set gain sweeps; ToneTwist internal only) | L | `test_real` |
| 4.5 | Pair each gold instruction with a target tone (pick the closest render / real recording); add friend-written instructions | L | `test_gold` |
| 4.6 | Splits disjoint by recording/performer + `test_unseen_pedal`; content hashes; data card | L | `data/splits.py`, `DATA_CARD.md` |
| 4.7 | Leakage tests: no shared recordings across splits, no param numbers in captions, gold-vs-template vocab overlap report | L | `tests/data/` |

**Exit gate:** dataset v1 frozen + hashed; leakage tests pass; target size ~20–50k synthetic pairs (fits on local disk and Drive).

## P5 — Evaluation harness and baselines (weeks 11–13) → M3

| # | Task | Where | Output |
|---|---|---|---|
| 5.1 | Metrics: param L1, gate F1, MR-STFT, descriptor/archetype error, CLAP score, gold 2AFC | L | `eval/metrics.py` |
| 5.2 | Stats: seeds, bootstrap CIs, paired tests, effect sizes | L | `eval/stats.py` |
| 5.3 | Baselines: no-op, mean params, kNN caption retrieval, LLM zero-shot params, Text2FX-style per-example CLAP optimization | L (C for Text2FX at scale) | `eval/baselines/` |
| 5.4 | `lstmabar eval` → markdown/JSON report with every baseline | L | `eval/report.py` |
| 5.5 | Precompute and cache CLAP text/audio embeddings for all splits | L (C if slow) | embedding cache |
| 5.6 | **M3 demo:** text box → best baseline sets the knobs | L | demo update |

**Exit gate:** baseline report reproducible from a clean checkout; M3 demo works.

## P6 — Model and experiments (weeks 14–17) → M4

| # | Task | Where | Output |
|---|---|---|---|
| 6.1 | Adapters on cached CLAP embeddings; parameter predictor + gates; physics mapping | L | `models/` |
| 6.2 | Stage 1: `L_param` only on cached embeddings (cheap) | L | first model |
| 6.3 | Stage 2: add `L_audio` (plus optional drive-amount-only `L_clap`, per spike A) through the oversampled pedalboard | C | full model |
| 6.4 | 5 seeds of the main config | C | `runs/main/` |
| 6.5 | Ablations: text-only, audio-only, shuffled audio, unconstrained vs physics mapping | C (stage-1 ablations L) | `runs/ablations/` |
| 6.6 | Small listening test (5–8 friends): model vs best baseline vs no-op on gold items | L | ratings |
| 6.7 | Results report with CIs, negative results included | L | `reports/v2_results.md` |
| 6.8 | **M4 demo:** model-driven knobs, nearest-pedal explanation, toggle vs baseline | L (CPU inference) | demo update |

**Exit gate:** the model beats the baselines on `test_gold` with non-overlapping 95% CIs, **or** a documented negative result. Either way, the M4 demo ships.

## P7 — Extensions (week 18+, pick in order)

1. **Quantum add-on experiment:** a *trainable* VQC bottleneck (PennyLane, CPU-simulated, 4–8 qubits) inserted after the fused embedding vs a classical bottleneck of identical width; same seeds, data and budget; reported with CIs. Runs on cached embeddings, so it stays cheap. Add it as a demo toggle if it holds up.
2. Modulation (chorus/phaser) and time effects (delay/reverb).
3. Grow the KB toward ~20 pedals; unseen-pedal generalization.
4. Review-text grounding for KB pedals.
5. Preference learning from demo A/B clicks: Bradley-Terry reward model → KL-regularized fine-tune.
6. Relative edits ("less fizz").
7. Host the demo on Hugging Face Spaces.

---

## Definition of done (every task)

- Code under `src/` with tests; CI green; notebooks import from `src/` only.
- Experiments launched from a config; run dir contains config, metrics, checkpoint and git SHA.
- Colab runs use the same CLI as local runs; no Colab-only code paths.
- Results always reported against baselines.

## First actions

1. P0.1–0.3: tag v1, move to `legacy/`, set up `uv` + skeleton + CI.
2. P0.5: write the gold instructions **before** touching captions. It's the cheapest step to do now and impossible to do uncontaminated later.
3. P0.6 CLAP spike and P0.7 license check.
