# LSTMABAR v2 — Execution Plan

Companion to [design.md](design.md). This is a solo side project, assumed at ~8–12 h/week, with a demo as the main deliverable. Durations are relative sizing, not commitments.

**Current status (2026-10-13):**
- **P0 and P1 are done; M1 is reached** (PRs #1–#17).
- **P2 and P3 are done; M2 is reached** (PRs #19–#27; owner browser check of the M2 demo passed 2026-10-12). P2's gate passes for DS-1 and, via a documented deviation, for TS808 (see [P2 status](#p2-status)).
- **P4 is done: dataset v1 is frozen** (PRs #29–#37 and the freeze PR; see [P4 status](#p4-status)).
- **Next: P5** (evaluation harness and baselines → M3). P4 follow-ups are listed under the P4 status.

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

### P2 status

**Status (2026-10-10): P2 exit gate passed; M2 reached** (PRs #19–#26); owner browser check of the demo passed 2026-10-12.

| PR | Content |
|---|---|
| #19 | Plan refinement + confirmed kickoff decisions |
| #20 | Wave 0: KB schema/validator (`physics/kb.py`), calibration, tone fit (`analog.py`), derive contract, white-box interface |
| #21 | P3: harmonics, descriptors, archetype readout (`analysis/`) |
| #22 | Diode-clipper white-box (trapezoidal + Newton, feedback + shunt), device table, TS808 model; optional `whitebox` extra (numba) |
| #23 | Pedal KB: TS808, RAT, DS-1, Fuzz Face Si/Ge, Big Muff (`pedals/*.yaml`, sources cited) |
| #24 | Derivations for all 6 pedals (`derive.py`, `networks.py`) |
| #25 | DS-1 white-box, exact-network solver (`whitebox/port.py`), fidelity check (`lstmabar fidelity`) and [report](../reports/greybox_fidelity.md) |
| #26 | M2 demo: pedal picker, grey-box vs white-box A/B, archetype panel |

Gate ([report](../reports/greybox_fidelity.md); masked mean = H1 + dBc terms not floored on both sides, decision 6):

| pedal | best-fit | per setting | derived | verdict |
|---|---|---|---|---|
| DS-1 | 1.13 dB | 0.61–2.17 | 1.78 | PASS |
| TS808 | 3.48 dB | 2.54–4.02 | 5.35 | PASS via documented deviation (ceilings 4.1 dB mean / 4.5 dB per setting) |

TS808 deviation cause: feedback clipping's unity clean path (`Drive` has no clean blend; decision 4) plus upper odd harmonics of the lowest notes that the grey-box under-produces.

Other numbers: white-box 18–22× real time per core with numba at 4× oversampling (pure NumPy < 1×, so the `whitebox` extra is effectively required); the full fidelity run takes ~16–23 min on CPU.

P2/P3 follow-ups (non-blocking; fold into P4 or later):
- **TS808 `clean_db`** blend on `Drive` to close the clean-path gap (owner decision; requires an `lstmabar recover` re-pass and a design §4.3 update).
- **Per-pedal calibration (P4):** fits want softer clippers than the derived priors (fitted softness mostly 0.06–0.17) and TS808 fits raise `eq.low_db` by 3–7 dB.
- **Gain clamping at 60 dB** is common at 0 dBFS = 1 V (DS-1 above ~¼ Dist, Big Muff above ~½ Sustain, Fuzz Face Si at full, RAT at max); revisit if P4 renders need the extra range.
- **Source model:** add pickup inductance (`DeriveContext.source_henries`) — the resistive 10 kΩ source inflates the Fuzz Face Si pre-HPF (411 Hz vs ~143 Hz with a pickup-like source); flag presets whose gain peak lands on the search-band edge.
- **Finite transistor gain / interstage loading** (PR #24 A2): DS-1 booster −0.4…−1.2 dB, Big Muff ~−1.5 dB per stage plus interstage loading; hidden by the 60 dB clamp today.
- **Fuzz Face assumptions:** transistor gain (h_FE) and source impedance are not in the KB; Q1 one-sided saturation not modelled.
- **DS-1:** level-pot loading of the tone stack differs between white-box and derivation (0.2–0.8 dB); R11/R15 roles unconfirmed.
- **TS808 pot tapers:** sources disagree (Drive A vs linear, Tone "G", Level A vs B); currently A/B/A.
- **Fidelity:** switch the fit to the masked loss; parallelize across pedals/settings (full run ~16–23 min); RAT white-box (diagnostic) needs a solver without a capacitor at the diode node.
- **Analysis:** MFCC deferred; pYIN dominates analysis time (track f0 once per dry clip in P4).
- **Demo:** readout wording for the narrow edit-then-move race; debounce re-derives during dial drags if hosted with `--share`.
- **Sources:** electrosmash.com was down; values came from the ElectroSmash Archive mirror, cross-checked against Geofex, Kit Rae, Aion FX and Fuzz Central.

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

**Status (2026-10-10): P3 exit gate passed** (PR #21, demo panel in #26). Pure oscillators classify as themselves with weight > 0.9; harmonic amplitudes at known f0 are accurate to ~0.01 dB; a Drive gain sweep moves the readout sine → triangle-like → square monotonically, and asymmetry raises even harmonics. MFCC deferred.

## P4 — Data pipeline (weeks 8–11)

| # | Task | Where | Output |
|---|---|---|---|
| 4.1 | Ingest licensed DI corpora (Guitar-TECHS DI; optional own DI, test-only; EGFxSet `Clean` is test-only, see 4.4 and R2) → mono 44.1 kHz, 2–4 s onset-aligned clips + manifest; level policy per R3 | L | audio in `data/sources/` (gitignored), `data/ingest.py`, manifest in `data/manifests/` |
| 4.2 | Renderer: dry × (pedal preset \| free board) × sampled knobs → wet. White-box for TS808 and DS-1, grey-box for the rest (decision 3). Store pedal id, pedal knobs, board params + gates, engine, descriptors, archetype readout. Multiprocess, resumable shards | L (C if slow) | `data/render.py` |
| 4.3 | Caption generator: templates from params/descriptors/KB vocab + LLM paraphrase (Claude API), no numeric leakage. **Never read `data/gold/` while writing or tuning it** (decision 6) | L | `data/captions.py` |
| 4.4 | Real-pedal test set: EGFxSet (TS Mini, RAT2, BD-2 + clean), a few pOD-set pedals, ToneTwist analog subset internal only (decision 8) | L | audio in `data/test_real/` (gitignored), manifest in `data/manifests/` |
| 4.5 | Pair each gold instruction with a target tone, in a **separate session after the caption code is frozen** (decision 7); add friend-written instructions | L | `test_gold` |
| 4.6 | Splits disjoint by performer/recording + `test_unseen_pedal` (decision 11); content hashes; data card | L | `data/splits.py`, `DATA_CARD.md` |
| 4.7 | Leakage tests: no shared recordings across splits, no param numbers in captions, no gold path in caption code; one-shot gold-vs-template vocab overlap report at freeze | L | `tests/data/`, `reports/vocab_overlap.md` |

**Exit gate:** dataset v1 frozen + hashed; leakage tests pass; ~20–50k synthetic pairs (start at ~20k, decision 9), within the disk budget; data card written.

### P4 status

**Status (2026-10-13): P4 exit gate passed.** Dataset v1 is frozen: 26,812 pairs from 4,589 Guitar-TECHS DI clips (train 24,528 / val 1,444 / test 560 / `test_unseen_pedal` 280 items), 50,130 captions, 31,401 hashed audio files, dataset SHA-256 `c4987b4a…` ([DATA_CARD.md](../DATA_CARD.md), `data/manifests/v1/`). `lstmabar data freeze` re-checks split disjointness (test performer-disjoint, R11), holdout isolation, file hashes and every caption against the shared vocab rules; the leakage tests pass in CI. `test_real`: 2,789 items (EGFxSet 2,069; pOD-set 720, internal only). Disk: ~6 GB rendered + ~5.7 GB `test_real` + ~4 GB sources, inside the ~30 GB budget. Caption paraphrase spend: $0.31 (cap $2.50). The one-shot vocab audit is in [reports/vocab_overlap.md](../reports/vocab_overlap.md) (aggregate only; no real `human` gold items yet).

What the numbers say (all from the data card / reports):
- **Splits (R11):** val is a by-recording P1/P2 slice (not performer-disjoint); test is all of P3 (140 clips, out of domain in content).
- **Labels:** 8,934 items (33 %) have a *derived* label at the 60 dB Drive clamp (Big Muff 2,846, DS-1 2,254, Fuzz Face Si 1,672, Ge 1,160, TS808 891, rat 111). White-box labels use the calibration tables; 3,711 white-box items use the nearest-grid fallback (DS-1 2,311, TS808 1,400). Off-grid, calibrated TS808 labels halve the derived error (5.21 → 2.49 dB); DS-1 calibrated labels are level with derived (1.16 vs 1.18 dB) and worse at tone ≈ 0.25 ([reports/calibration.md](../reports/calibration.md)).
- **Analysis:** `analysis.max_frames` 32 (R8); vs 64, odd/even |Δ| median 0.21 dB, max 1.66 dB ([reports/render_bench.md](../reports/render_bench.md)).
- **Captions:** template vocabulary is narrow (227 word types; paraphrases 1,010); 3,494 items (13 %) have only a template caption (paraphrase identical to the template or rejected). Paraphrases stay close to the templates. Gold content-type coverage by all captions is ~30–36 % per LLM-written source.

Follow-ups (P5/P6 or a dataset v1.1, which must be labelled "tuned after gold audit" if templates change):
- Down-weight / dedupe clamped grey-box items in P6 (decision 10), or revisit option (b) (raise `drive.gain_db` max).
- DS-1 calibration: denser tone grid, or derived labels in the tone 0–0.5 cells.
- Real `human` gold items (friends) before P5 reports a headline number; gold pairing (4.5) in a separate session.
- Render throughput: the full render ran out of RAM at 7 and at 3 workers on this 16 GB machine with other apps open; it finished at 2 workers. Consider `maxtasksperchild` / per-shard memory caps.
- Unify the two remote-zip helpers (`ingest` streaming fetcher vs `remote_zip`).

### P4 kickoff brief

Prepared 2026-10-12 for a fresh session. Read this, CLAUDE.md, design §5 (data) and [data-licenses.md](data-licenses.md) first; confirm the decisions below with the owner before writing code. Same workflow as P2: `v2/...` branches, one PR per wave item, independent review agent per PR, fix blocking findings and re-review, merge on PASS + green CI.

**What P4 builds on (already on `main`):**
- `physics.kb.load_kb()` (6 pedals), `physics.derive.derive()` / `preset_params()` → normalized `BoardParams` for `dsp.pedalboard.default_pedalboard()` (Compressor → Drive → EQ3, per-block gates, `random_params`, `to_vector`/`from_vector`).
- `physics.whitebox.simulate()` for `ts808` and `ds1` (needs the `whitebox` extra; ~20× real time per core at 4× oversampling). `has_whitebox()` for runtime discovery.
- `physics.fidelity` (best-fit grey-box vs white-box machinery) and its report; `DeriveContext.from_config` (`physics.volts_per_full_scale`, `physics.source_ohms`).
- `analysis.archetypes.archetype_readout`, `analysis.harmonics.harmonic_profile` / `track_f0` (pYIN, `analysis` extra), `analysis.descriptors.describe`.
- `lstmabar.audio` (decode incl. PyAV fallback, resample, loudness), `config.load_config`, `runs.create_run_dir`, `seed_everything`.
- `data/raw/`, `data/rendered/`, `data/cache/`, `data/sources/` and `data/test_real/` are gitignored. **No audio is ever committed**: only code, manifests (`data/manifests/`), hashes and the data card. ToneTwist, pOD-set and IDMT audio must never leave the local machine/Drive. `src/lstmabar/data/` and `src/lstmabar/eval/` are empty packages.

**Owner actions (before or during P4):**
- **Disk:** the dev machine has ~57 GB free. Budget P4 at ≤ ~30 GB total (raw + rendered + cache); see decision 9.
- **Claude API key** (`ANTHROPIC_API_KEY` in the environment, never committed) for caption paraphrase, plus a spending cap you're comfortable with (decision 5).
- **Optional:** 30–60 min of your own DI recordings (always license-clean; also a natural extra test performer).
- **Friend-written instructions** for the gold set (task 4.5; they must not see the templates).
- **License emails** still open (data-licenses.md follow-ups): EGDB, TONE3000, GuitarML, Fraunhofer. P4 does not depend on them.

**Kickoff decisions** (each has a recommendation; confirm with the owner):

1. **DI sources and clipping into clips.**
   - Guitar-TECHS **DI tracks only** (CC BY 4.0, 3 players; 48 kHz float → 44.1 kHz) as primary; EGFxSet `Clean` (single notes, 5 pickup positions) as supplement; own DI if recorded. Not IDMT (NC-ND), not EGDB (no license yet), not GuitarSet (acoustic).
   - Download once by hand or with a small script that fetches only the needed Zenodo files; extract DI only, then delete the zips (disk).
   - EGFxSet `Clean` notes that serve as `test_real` inputs (the TS Mini / RAT2 / BD-2 note set) are **excluded from training DI** (or the whole EGFxSet note grid is test-only — decide at kickoff).
   - Clips: 3 s (config `audio.clip_seconds`), onset-aligned (onset detection on the DI, start ~20 ms before the onset), skip near-silent windows, keep source file + offset + performer + technique metadata. Mono 44.1 kHz, stored as FLAC (16-bit is enough for DI after level normalization; 24-bit if headroom matters).
2. **Input level policy (new; interacts with the 0 dBFS = 1 V calibration).** DI recordings are normalized arbitrarily, but pedal behaviour depends on input voltage. Recommendation: normalize each dry clip to a peak sampled from a pickup-like range (e.g. uniform −24…−6 dBFS, i.e. ~60–500 mV peak) and store the level; the model sees the level through the audio itself. Note: this does **not** mitigate the 60 dB `Drive` clamp (that limits the knob-derived `gain_db`, independent of input level; at low input the missing gain is *more* audible). The same level rule must apply at inference (demo/eval inputs get normalized into the same range).
3. **Engines and labels.**
   - TS808 and DS-1 render with the **white-box**; RAT, Fuzz Face Si/Ge and Big Muff with the **grey-box** (`derive` → pedalboard). Record `engine` per item so results can be split by it.
   - **Label for white-box renders** is the open question, since the board can't reproduce the circuit exactly (TS808 best-fit 3.5 dB). Options: (a) the derived board params (cheap; biased, derived TS808 error 5.4 dB); (b) a per-pedal **calibration table**: best-fit board params on a coarse knob grid (e.g. 5 × 3 settings per pedal, reusing `physics.fidelity`), interpolated to the sampled knobs; (c) a per-render fit (too slow). Recommendation: **(b)**, which is also the P2 per-pedal calibration follow-up; store (a) as well so the two can be compared.
   - NAM/TONE3000 captures are **not** used in v1 (license emails pending); design §5's "white-box primary" becomes: white-box for TS808/DS-1, grey-box majority. Update design §5 at kickoff.
   - Calibration table details to settle: grid = gain knob × tone knob (level knob handled analytically as output gain), sine levels re-chosen to cover the P4 input range (−24…−6 dBFS), interpolate in normalized-param space only between neighbouring fits started from the same start (fits are non-unique); budget ~1 h CPU per pedal.
   - Always store the pedal knob rotations too (pedal-space labels, for nearest-pedal explanation and `test_unseen_pedal`).
4. **What gets rendered (sampling mix).** Instructions also ask for clean, EQ-only and compressed tones, which no pedal preset covers. Recommendation: ~70 % pedal presets (6 pedals, knob rotations uniform in [0, 1], optional EQ3 tweak off) and ~30 % **free board** (`random_params` with sampled gates: clean/EQ only, compressor only, drive without a named pedal, bypass). One dry clip × several renders, so each dry clip appears with several tones (needed for kNN and for the audio-conditioned model). Free-board gates are sampled explicitly per scenario (`random_params` samples each gate independently, which is not the scenario mix). Pedal presets keep the derived EQ3 (it holds the pedal's tone stack); no extra user EQ on top in v1.
5. **Captions** (`data/captions.py`).
   - Inputs: pedal family/name (named in only a fraction of captions, e.g. 25 %), knob-derived intensities (drive amount, brightness from tilt/EQ, mids, compression), measured descriptors binned into words (centroid → dark/bright, odd/even + harmonic energy → smooth/gritty/fuzzy, HNR → noise), KB `descriptors`. **No numbers** in captions (no dB, Hz, knob values).
   - Templates first (deterministic, seeded), then LLM paraphrase for variety: Claude API via the **Message Batches API** (50 % cheaper, async), many captions per request, cached by (template text, model, prompt version) in `data/cache/`. Pick the current Claude model per the `claude-api` skill at kickoff; record model id + prompt in the data card. Default 2 captions per render (1 template, 1 paraphrase).
   - Absolute instructions only (relative edits are v2.x).
   - Paraphrases pass a **post-filter** (reject any digit, unit or knob value, with an allowlist for pedal names such as TS808, DS-1, RAT2, BD-2 — the 4.7 leakage test uses the same allowlist; reject mentions of a held-out pedal's name where that would leak `test_unseen_pedal`). Add an `llm` extra with the `anthropic` SDK. Estimate cost before the full run (captions × tokens × Batches price) and get the owner's OK.
   - **Analysis on polyphonic clips:** `describe`/`archetype_readout` take one f0, but many Guitar-TECHS clips are chords/phrases. Policy: extend `describe`/`archetype_readout` to accept a pitch track (`harmonic_profile` already takes `track=`) and pass the pYIN track from the dry clip; if the voiced fraction is low, mark harmonic descriptors NaN and caption from the board params + spectral descriptors only (centroid, rolloff, crest), never from NaNs.
6. **Gold-set hygiene, enforced.**
   - The session/agent writing captions **never reads `data/gold/`**. Add a test that fails if caption code references a `gold` path.
   - The vocab overlap audit (4.7) runs **once, at freeze**, by a separate step; it reports aggregate overlap only. Templates are not changed after seeing it; if they are, the dataset becomes v1.1 and is reported as "tuned after gold audit".
   - Real `human` gold items are the headline set; `simulated_human`, `llm_persona` and `llm_draft` are reported separately and never relabeled.
7. **Gold pairing (4.5), in a separate later session.** Each gold item has a rough target description (pedal family, gain level, brightness). Recommendation: turn each into a target **board-param region** (owner-reviewable YAML, one row per item) and render the target on held-out test DI clips; the P5 2AFC metric then compares the model's render against the target vs a distractor. This step necessarily reads `data/gold/`, so it happens only after the caption code is frozen and in a session that does not touch caption code.
8. **`test_real` (4.4).** EGFxSet units matching the KB: **Ibanez Mini Tube Screamer ↔ ts808**, **RAT2 ↔ rat**, plus BD-2 (no KB entry, family overdrive), each with the matching `Clean` note; settings from the EGFxSet CSV. pOD-set: download 3–5 pedals only (gain × tone grid, CC BY-NC; input is mixed material, not pure DI — use as a parametric drive/tone check). ToneTwist analog subset (has a Big Muff): internal numbers only, never republish audio. The P5 use: sim-to-real gap (KB render at the matched setting vs the real recording) and audio-target evaluation. EGFxSet and pOD-set audio is level-normalized, so the true input voltage is unknown: compare after loudness matching and/or sweep the assumed input level, and say so in the report.
9. **Size, storage and format.**
   - Start at **~20k pairs** (e.g. ~4k dry clips × 5 renders; superseded by R7), scale toward 50k only if disk and time allow (train-player DI is only ~3.5 h, so 50k means more renders per clip, not more clips). 3 s mono 44.1 kHz FLAC is ~150–250 KB → 20k wet ≈ 4–5 GB, plus dry clips and raw sources (Guitar-TECHS DI extracted ≈ a few GB), test sets (EGFxSet zips ≈ 2 GB for the needed units + Clean; pOD-set ≈ 1.7 GB per pedal) and transient space for zip + extraction at once. Keep the ≤ ~30 GB total.
   - Shards of ~1k items: FLAC files + one JSONL manifest per shard (no new dependency); a top-level index with per-file SHA-256 and a dataset hash.
   - Rendering: multiprocessing, deterministic per-item seeds, resumable (skip finished shards). Throughput estimate: white-box ~0.15 s per 3 s clip per core; grey-box batched in torch; analysis (pYIN once per dry clip, reused for its wets) dominates — budget for it.
10. **The 60 dB gain clamp.** Derived gain exceeds `Drive`'s 60 dB for DS-1 above ~¼ Dist, Big Muff above ~½ Sustain, Fuzz Face Si at full and RAT at max, so those knob ranges render identically on the grey-box pedals (DS-1 renders through the white-box, so its clamped settings still sound different; only the derived label collapses). Options: (a) keep 60 dB and down-weight / dedupe clamped settings, reporting clamped counts in the data card; (b) raise `drive.gain_db` max (e.g. 75 dB) — changes the P1 contract, needs an `lstmabar recover` re-pass and design §4.3 update. Recommendation: (a) for v1, revisit after P5 baselines.
11. **Splits (4.6; val/test rule revised by R11).** Guitar-TECHS has 3 players: train = 2 players, val/test = the third, split by recording (no shared source file between val and test). EGFxSet clean notes: test-only (R2). Own DI (if any): test only. Val and test share the third player, so consider rotating the held-out player across seeds in P6. `test_unseen_pedal`: hold out **one KB pedal** from train entirely — recommendation **rat** (its family is covered by DS-1, so it tests within-family generalization); alternative fuzz_face_ge (tests device change with the Si variant in train). Holding out `rat` also makes the EGFxSet RAT2 `test_real` items unseen-pedal items, and captions must not name it. Hashes are computed on content, not paths; they identify the frozen dataset, not bit-identical re-renders across machines (CPU vs Colab float differences).

**Kickoff resolutions (confirmed with the owner 2026-10-12; they supersede the recommendations above where they differ):**

- **R1. Caption model and budget.** `claude-haiku-5-5` through the Message Batches API, `thinking: {type: "disabled"}`, `output_config.effort: "low"`, structured JSON output keyed by caption id, ~25 captions per request. Estimate at batch prices ($0.05 / $0.25 per MTok in/out): ~$0.30 per 20k paraphrases, ~$0.75 per 50k; the planned v1 (R7: train + val/test + unseen renders) is ~28–34k paraphrases ≈ $0.45–0.55 (Sonnet 5.5 would be ~$5.80 for 20k, over budget). The owner's key has $4.50 in total, so the hard cap is `captions.max_usd: 2.50`. Guards:
  - `lstmabar data captions --estimate` prices the run and spends nothing: input from the free `count_tokens` endpoint, output from the pilot's measured tokens per caption (45 before the pilot);
  - a pilot batch of ~100 captions runs first, to measure real token counts;
  - a spend ledger (`data/cache/llm/ledger.jsonl`) is appended from each result's `usage`; submitted batches that have not returned count at their estimate; a submit is refused when ledger + in-flight + estimate > cap;
  - a paraphrase rejected by the post-filter falls back to its template caption (no paid retry); errored/expired requests are resubmitted at most once;
  - CI mocks the SDK; no test touches the network. The key comes only from `ANTHROPIC_API_KEY` (`.env` files are gitignored).
  - The paraphrase prompt is written fresh. It is not derived from the prompts that produced the gold `llm_draft` / `llm_persona` items.
- **R2. EGFxSet is entirely test-only.** Its clean notes are very likely the dry inputs of its pedal recordings, so any training use would leak into `test_real`. Training DI = Guitar-TECHS DI only (own DI, if recorded, is test-only).
- **R3. Levels and storage.**
  - Dry clips are stored peak-normalized to −1 dBFS (16-bit FLAC) with the source gain recorded.
  - The input level is sampled **per render**: `input_peak_dbfs ~ U(−24, −6)`. The render input is `dry × 10^((input_peak_dbfs + 1)/20)`. Each dry clip is seen at several levels.
  - Wet renders are stored peak-normalized to −1 dBFS as **24-bit FLAC**, with `wet_gain_db` in the manifest (true wet = file × 10^(wet_gain_db/20)). FLAC cannot hold values above full scale, and Drive's level reaches +6 dB.
  - Inference rule (demo/eval): normalize the input peak to `audio.inference_peak_dbfs` (default −15 dBFS, the middle of the training range). Wiring it into the demo and eval input paths is a P5 task (5.4/5.6).
  - Ingest skips dry clips below `ingest.min_rms_dbfs` = **−55 dBFS** RMS at the source level (−45 at kickoff). The Guitar-TECHS DI noise floor is ~−80 dBFS, and P1's quiet takes (harmonics: median clip RMS −43 dBFS) lost half their clips at −45. The data card must report the `source_gain_db` and clip-RMS distributions per performer.
- **R4. White-box labels (decision 3b).**
  - Calibration grid = gain knob × tone knob, 5 × 3, for `ts808` and `ds1`. The level knob is handled analytically as output gain.
  - One joint best fit per grid setting over sine levels spanning the input range (−24, −15, −6 dBFS peak) at a few f0s, reusing `physics.fidelity` at the render sample rate (44.1 kHz, not `FidelityConfig`'s 22.05 kHz default). Each gain chain starts from the derived params at gain = 0.
  - Fits are warm-started along the grid (each from its gain-axis neighbour). **Residual interpolation** (decided after the first run: neighbouring fits jumped > 0.25 along tone in 14/16 cells because the derived tone → EQ mapping itself switches branch): label = clip(derived(knobs) + r, 0, 1), with the correction r = fit − derived (normalized params) interpolated bilinearly over the cell; derived boards at the grid points are stored in the table. Where corner residuals differ by more than `calibration.max_jump` (0.25), the nearest grid point's residual is used (`calibrated_nearest`). The renderer passes the derived board it already has (`lookup(knobs, derived=...)`). The level knob's analytic offset is added to `drive.level_db` afterwards and clamped to Drive's range (a clamp is recorded like any other). Fallback cells and an off-grid held-out check (derived vs nearest fit vs residual lookup at the cell centres) are in `reports/calibration.md`.
  - Tables live in `data/calibration/<pedal>.json`. They are small, contain no audio, and are committed.
  - Every white-box item stores both the calibrated label (`board_params`, used for training) and the derived one (`board_params_derived`).
- **R5. Sampling mix.**
  - 70 % pedal presets: pedal uniform over the five non-holdout pedals (the same mix for train, val and test); knobs U(0, 1); derived EQ3 on; compressor off; no user EQ.
  - 30 % free board: bypass 5 %, EQ only 25 %, compressor only 15 %, compressor + EQ 15 %, generic drive (EQ and compressor each on with p = 0.5) 40 %.
- **R6. Holdout `rat`; clamp option (a).**
  - `rat` is never rendered for train or val. It is rendered only on test DI, as `test_unseen_pedal`. Captions never name it outside that split.
  - Items record `clamped` from `derive`'s `" clamped to "` notes only (other notes mention the clamp too, e.g. Big Muff's). For white-box items it describes the derived label; the calibrated label is in range by construction. The data card reports clamped counts.
  - Clamped grey-box settings are kept but flagged, so P6 can down-weight or dedupe them.
- **R7. Counts** (val/test clip counts revised by R11).
  - The two train players give at most ~4k non-overlapping 3 s clips (fewer after skipping silence).
  - Plan ~3–4k train clips × 6 renders (≥ 18k train pairs). Val and test clips (player 3, split by recording) get 4 renders each, and `test_unseen_pedal` 2 `rat` renders per test clip.
  - The data card reports the exact per-split counts.
- **R8. Throughput gate in wave 0.** Benchmark render + analysis per item on the fixtures and extrapolate to 20k items. If it exceeds ~12 h on the local CPU, cut the analysis cost first (`max_frames`, analysis on a subset), not the dataset. *Measured in task 4.2 (`reports/render_bench.md`, fixtures): the planned v1 from the ingest facts is 4,280 clips / 24,780 items (train 3700 × 6, val 450 × 4, test 130 × 4 + 260 unseen). On 4 workers that prices at ~11.0 h with `max_frames: 64`, ~6.9 h at 32 and ~4.4 h at 16. `analysis.max_frames` is 32: the largest option within 0.75 × the 12 h budget (`bench.headroom`; an earlier run on a heavily loaded CPU measured ~2× slower, and worker scaling is assumed linear). Change of the wet analysis vs 64 frames on the bench renders (median / max): odd/even 0.21 / 1.66 dB at 32 and 0.63 / 2.19 dB at 16; archetype L1 0.008 / 0.052 at 32 and 0.011 / 0.065 at 16. Wet analysis dominates (~2.6 s/item at 32 frames); pYIN is ~1.8 s per dry clip.*
- **R9. Downloads.** List the Zenodo record's files through its API first. Download one archive at a time, extract the DI tracks, and delete the archive before the next one, so the transient peak is one archive.
  - As implemented (4.1): Guitar-TECHS ingest streams only the DI members out of the remote zips via HTTP range requests instead of downloading archives. No archive is ever on disk, and the transient peak is one member.
- **R10. Gold set status.** Per its commit history, `instructions_human.yaml` holds *simulated*-human items; the per-item `source` field is authoritative. Real `human` items (friends, owner action) are needed before P5 reports a headline number.
- **R11. Splits revised (wave 2, supersedes decision 11's val/test rule).** Ingest found that the held-out player P3 recorded only the 12 musical excerpts (481 s of DI, ~130 clips), too little to split again and out of domain for model selection. So `test` = all of P3 (performer-disjoint; `test_unseen_pedal` renders are made on it), and `val` = an in-domain **by-recording** slice of P1/P2: within each (performer, technique) group, `splits.val_recording_fraction` (10 %) of recordings, chosen by hash (single-recording groups stay in train). Recordings are hashed on their content key (id without the performer prefix), so a take of shared material goes to val for both players and val measures unseen content. Only chords and scales have several takes per performer, so val holds chords and scales only, ~8 % of P1/P2 time (~400 clips, estimated; exact counts in the data card). Val therefore measures new recordings and content, not new players; only test is performer-disjoint. The data card states all of this. Rotating the held-out player in P6 remains possible (P1 or P2, ~2.4 h each).

**Suggested parallel waves** (same workflow as P2):
- wave 0 (main session): data contracts — item/manifest schema (dataclasses), `configs/data/*.yaml`, shard layout, CLI skeleton (`lstmabar data ingest|render|captions|split|freeze`), level policy, a tiny synthetic DI fixture set for tests.
- wave 1, in parallel:
  - (a) DI ingest (Guitar-TECHS DI only, R2): download, resample, onset-aligned clips, manifest;
  - (b) renderer: pedal/free-board sampling, white-box/grey-box engines, calibration table (decision 3b), descriptors/archetypes, sharded multiprocess output — built against the synthetic fixtures;
  - (c) caption generator: templates + Batches-API paraphrase + cache — **gold-blind agent**, briefed explicitly;
  - (d) `test_real` ingest (EGFxSet units + their `Clean` notes + a few pOD-set pedals).
- wave 2: full render of v1, splits, hashing, data card, leakage tests, one-shot vocab overlap report → freeze.
  - As implemented (4.6–4.7): `lstmabar data freeze` checks splits/items/files/captions (fails loudly, writes nothing on failure; `--allow-partial` marks the index partial), writes `files.sha256` (dry + wet; `test_real` hashes stay in its manifest), `index.json` and the generated `DATA_CARD.md`. `lstmabar data vocab-audit` is a separate one-shot step on the frozen captions → `reports/vocab_overlap.md` (aggregates per gold `source` only).
- wave 3 (separate session): gold pairing (4.5) + friend instructions.

**P2/P3 follow-ups that matter here:** per-pedal calibration (decision 3b), the 60 dB gain clamp (decision 10), pYIN cost (track f0 once per dry clip), Fuzz Face source-impedance assumption (renders inherit it), MFCC (add to descriptors if P5 needs it).

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
8. Real-time hardware (post-v2, unscheduled): a Daisy-based pedal with an iPad app replacing the pedalboard. Costed in [realtime-hardware.md](realtime-hardware.md).

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
