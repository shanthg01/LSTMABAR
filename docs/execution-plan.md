# LSTMABAR v2 — Execution Plan

Companion to [design.md](design.md). This is a solo side project, assumed at ~8–12 h/week, with a demo as the main deliverable. Durations are relative sizing, not commitments.

**Current status (2026-10-09):**
- **P0 and P1 are done; M1 is reached** (PRs #1–#17).
- **Next: P2**, together with P3, which is independent and can run in parallel. Start from the [P2 kickoff brief](#p2-kickoff-brief).

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

**Status (2026-10-09): P1 exit gate passed; M1 reached** (PRs #9–#15).
- Recovery ([report](../reports/param_recovery.md)): **23/24 audio match (96%)**. In parameter space, 9/24 trials have all knobs within ±0.05, or 16/24 with `drive.asymmetry`/`drive.bias` exempted (they trade off along a near-identical-audio direction). Most misses are the swept `eq.mid_hz`: likely local minima or coupled compensation.
- M1 demo: `uv run lstmabar demo`. Sliders are generated from the param specs; 15 s clips render in ~2 s on CPU.
- `default_pedalboard` forward+backward, B=8 × 3 s: ~2.8 s on CPU, dominated by the filter backward pass.

P1 follow-ups (non-blocking, fold into later phases):
- Speed: evaluate filter responses directly instead of zero-padded FFTs; move the compressor recursion off the CPU (parallel scan) if GPU training needs it; consider checkpointing the 4× shaper.
- Optimization: multi-start or coarse-to-fine search for the swept mid peak (P6 parameter loss / predictor init).
- Losses: crop the ~32-sample oversampler edge transients.
- Demo polish: put defaults on the slider step grid; accept auth from an env var; ":1" unit for ratio; friendlier knob labels.

Post-M1 fix (PR #17, found in manual testing): non-WAV uploads (e.g. `.m4a`) failed because Gradio needs a system ffmpeg. The demo now decodes uploads itself (libsndfile → PyAV). The manual browser test was then confirmed working by the owner.

## P2 — Physics layer and pedal knowledge base (weeks 5–7) → M2

| # | Task | Where | Output |
|---|---|---|---|
| 2.1 | KB YAML schema + validator (format per kickoff decision 1); validator test runs over every file in `pedals/` | L | `physics/kb.py` |
| 2.2 | Author 6 pedal files: TS808, RAT, DS-1, Fuzz Face Si, Fuzz Face Ge, Big Muff. Every component value cites a source URL; owner spot-checks values against the source | L | `pedals/*.yaml` |
| 2.3 | Derivations: components + knob positions → physical `Drive`/`EQ3` knob values (decision 2), with the level calibration of decision 3. Tests check the textbook numbers (e.g. TS808 HPF ≈ 720 Hz, drive-stage gain 1 + (51k + R_drive)/4.7k) | L | `physics/derive.py` + tests |
| 2.4 | White-box sims (decision 5): one generic one-capacitor diode-clipper solver in two configurations, **feedback** (TS808) and **shunt to ground** (DS-1 and RAT clipping stage), plus each pedal's linear stages from component values. Fuzz Face is a time-boxed stretch. Measure throughput (sizes P4) | L | `physics/whitebox/` |
| 2.5 | Fidelity check (decision 6): best-fit grey-box vs white-box, and derived (unfitted) grey-box vs white-box. `lstmabar fidelity` CLI, run dir, report (same pattern as `lstmabar recover`) | L | `reports/greybox_fidelity.md` |
| 2.6 | **M2 demo:** pedal picker + pedal knobs (Drive/Tone/Level…) → derived board knobs → existing sliders; grey-box vs white-box A/B for white-box pedals; archetype panel once P3 lands | L | demo update |

**Exit gate:**
- Best-fit grey-box harmonic error (decision 6 metric) ≤ ~3 dB mean for TS808 and DS-1, or the deviation is documented with its cause (e.g. a named topology gap from decision 4).
- Derived (unfitted) error reported for every white-box pedal; no gate on it.
- KB validator and derivation tests green in CI; M2 demo runs.

Not in P2: NAM/Proteus captures (they are a P4 rendering source, pending license emails), Big Muff white-box, rail clipping and bandwidth limits of op-amps. A RAT clipping-stage sim may be built from the shunt solver, but with ideal op-amps it is **diagnostic only** and not part of the gate.

### P2 kickoff brief

**What P2 builds on (already on `main`):**
- `Drive`, `EQ3`, `Compressor` and `Gain` blocks; their knob ranges are in design §4.3.
- `Pedalboard` / `default_pedalboard()` with gates, `describe()` and `to_vector`/`from_vector`.
- `dsp.losses.multi_resolution_stft_loss`.
- `dsp.signals` (synthetic riffs) and `lstmabar.audio` (decode, resample, loudness).
- The demo builds its sliders generically from `param_specs`, and `demo.render.knobs_to_params` turns physical knob values into board params. An M2 preset picker only needs to produce physical knob values.

**Kickoff decisions** (confirmed by the owner 2026-10-09, as recommended below):

1. **KB format.** One YAML per pedal under `pedals/`, validated by plain dataclasses in `physics/kb.py` (no new dependency, no separate schema file).
   - Fields: `id`, `name`, `family`, `topology` (enum, e.g. `opamp_feedback_clip`, `opamp_shunt_clip`, `two_transistor_fuzz`, `cascaded_transistor_clip`), clipping device(s) with type (Si/Ge/LED/MOSFET), component values, pots with value and taper, `sources` (URLs), optional `descriptors`.
   - Component values as schematic strings with SI suffixes (`"4.7k"`, `"0.047u"`, `"51p"`), parsed and range-checked by the validator.
   - Pot tapers `A` (audio), `B` (linear), `C` (reverse audio); `A`/`C` as the usual two-segment approximation. Knob position 0–1 → resistance is part of `kb.py`, not each derivation.
   - Fuzz Face Si and Ge are **separate pedal ids** (separate files), so P4's held-out-pedal split can treat them independently.
   - Load path comes from a `paths.pedals: pedals` config key, so a non-editable install can point elsewhere.
2. **Grey-box target.** `physics/derive.py` maps (pedal, knob positions) → physical values for the *existing* `Drive`/`EQ3` knobs, then `ParamSpec.normalize` → board params. That makes presets usable directly by `Pedalboard`, the renderer and the demo.
   - Nonlinear stage: closed-form circuit formulas (stage gain, pre-clip HPF corner, device → softness/asymmetry/bias).
   - Linear tone stacks (DS-1 and Big Muff LP/HP blends, RAT filter, TS tone): compute the analog magnitude response from components, then least-squares fit `Drive.tone_db` + `EQ3` to it. Deterministic and cheap, and it puts the "physics" in the response rather than in hand-tuned constants.
   - A preset uses `EQ3` for the pedal's tone stack. That's fine for M2 and P4; note it so the demo doesn't present `EQ3` as a separate user EQ while a preset is active.
3. **Level calibration.** The white-box works in volts and the grey-box clips at ±1, so fix a convention once, in config: **0 dBFS ↔ 1 V peak at the pedal input** (hot humbucker territory; typical DI peaks sit 6–20 dB lower).
   - Then `drive.gain_db` = stage gain + 20·log10(1 V / V_clip), where V_clip is the device's clipping voltage (≈0.6 V Si, ≈0.3 V Ge, ≈1.7 V LED; MOSFET devices take their value from the cited source). Without this, derived gains are off by up to ~10 dB and the fidelity check measures the calibration, not the model.
   - Both sims and the fidelity sweep use the same constant.
4. **Range and topology gaps.** Verify each against the sources before changing anything, and let the fidelity check decide whether a gap matters:
   - **Gain above 60 dB.** RAT is ≈ 1 + 100k/(47 Ω ∥ 560 Ω) ≈ 67 dB. Beyond ~45–50 dB the shaper output is already near-square, so the extra gain mainly changes sustain and noise. 67 dB is also only the high-frequency ceiling: the 47 Ω / 2.2 µF leg turns on above ~1.5 kHz, below that the 560 Ω leg gives ~45 dB, and the LM308 bandwidth caps it further. Decision: clamp at 60 dB and record it in the RAT file; `derive.py` must not apply 67 dB flat.
   - **Pre-clip low-pass.** The LM308's bandwidth (RAT) and the 51 pF feedback cap (TS808, ≈5.7 kHz at full drive) low-pass the signal *before* clipping. `Drive` has no pre-clip LP; its post-clip tilt only partly stands in.
   - **TS808 clean path.** Feedback clipping outputs the input *plus* the clipped gain path, so a unity, full-range clean component survives (part of the TS "transparency"). `Drive` has no clean blend. At 0 dBFS = 1 V the clean part (1 V) exceeds the clipped part (~0.6 V), so expect to need the `clean_db` fix for the TS808 gate; report TS808 error per input level to make the cause visible.
   - **Cascaded stages.** Big Muff has two clipping stages and DS-1 has a transistor booster before the op-amp. Use one `Drive` and accept the error (Big Muff has no white-box, so it is not measured).
   - Smallest-change fixes, only if fidelity misses the gate because of that gap: an optional `pre_lpf_hz` or a `clean_db` blend on `Drive` (default off). **Any change to Drive's ranges or chain must re-pass `lstmabar recover`** and update design §4.3.
5. **White-box solver.** Pure NumPy/SciPy, offline, no ngspice or other system dependencies on Windows.
   - One solver for a one-capacitor diode clipper (Shockley diodes, trapezoidal rule + Newton, 4–8× oversampled), configured as feedback (TS808) or shunt (DS-1, RAT clipping stage). Linear stages around it (input buffers, booster, op-amp gain networks, tone stacks) are analog transfer functions from component values, discretized with the bilinear transform. Op-amps are ideal (no rails, no slew); documented.
   - Vectorize over a batch of clips with the time loop in Python. Target ≥10× real time aggregate per core; if it misses, add `numba` as an optional `whitebox` extra rather than rewriting.
   - Validation tests: small-signal output matches the analytic linear response (≤0.1 dB); output bounded near V_clip at high drive; symmetric diodes → even harmonics below −80 dBc; halving the step changes the output negligibly (convergence).
   - Fuzz Face (two-transistor Ebers–Moll, bias-dependent asymmetry, input-impedance interaction) is a **stretch, time-boxed to ~1 week of P2 effort**. If it slips, Fuzz Face stays grey-box only and the gap is documented.
6. **Fidelity metric and fit.**
   - Inputs: sines at ~82, 165, 330, 660 Hz × −30, −20, −10, 0 dBFS (calibration of decision 3), 0.5 s each with onsets cropped; plus MR-STFT on a riff.
   - Metric: H1 level error in dB, plus H2–H10 in dBc (relative to H1), each floored at −60 dBc so harmonics both models put in the noise don't count. Report mean and max per pedal and per knob setting.
   - **One parameter set per (pedal, knob setting) must fit all pitches and levels at once.** Level dependence is the point of a clipper, so fitting per level would hide the gap.
   - Knob settings: drive ∈ {min, mid, max} × tone at mid (+ tone sweep at mid drive).
   - Fit with **multi-start** (P1 showed local minima), starting from the derived params plus random restarts.
   - Report both best-fit error (expressiveness of the grey-box family, the gate) and derived error (accuracy of `derive.py`). If they differ a lot, that motivates a per-pedal calibration in P4.
7. **Sources and data hygiene.**
   - Use published circuit analyses (e.g. ElectroSmash) for component values and cite URLs in each YAML. Don't copy schematic images or prose.
   - The KB's `descriptors` and pedal vocabulary feed P4 captions, so **gold-set hygiene applies: don't read `data/gold/` while writing them**. Write descriptors from circuit behaviour and the cited sources only.

**Suggested parallel waves** (same workflow as P1):
- wave 0 (main session): `kb.py` schema + validator, the calibration constant, `derive.py` contract (signatures + one worked pedal, TS808), whitebox solver interface.
- wave 1, three agents in parallel:
  - (a) author the other 5 pedal YAMLs against the schema (sources cited; owner spot-checks);
  - (b) diode-clipper white-box (both configurations) + validation tests + throughput benchmark;
  - (c) **P3 analysis** (harmonics, descriptors, archetype readout), independent of the KB. Wave 2 reuses its harmonic extraction for the fidelity metric.
- wave 2: derivations for all pedals + fidelity fit + report; topology-gap fixes only if the gate misses.
- wave 3: M2 demo (picker, A/B, archetype panel). Fuzz Face white-box in the remaining time box.

**P1 follow-ups that matter here:**
- Crop the ~32-sample oversampler edges in losses (the fidelity fit uses them).
- Multi-start for swept-frequency knobs (`eq.mid_hz` in the tone-stack fit and fidelity fit).
- Filter speed: only if the fidelity fit loop turns out slow; sine fits on 0.5 s clips are short.

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
| 4.2 | Renderer: dry × pedal × sampled knobs → wet (white-box for TS808 and DS-1, plus Fuzz Face if the P2 stretch lands; NAM/grey-box for the rest); store params, pedal id, descriptors, archetype readout. Multiprocess; resumable shards | L (C if slow) | `data/render.py` |
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
