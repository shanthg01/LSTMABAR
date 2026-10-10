# LSTMABAR v2 — Overall Design

Status: approved direction · Owner: Shanth Gopalswamy (solo side project building on the W266 work with Zain Nasim) · Date: 2026-10-09

**Decisions (2026-10-09):**
- Scope: guitar only.
- Quantum layer: kept as an add-on experiment (v2.x), not on the critical path.
- Compute: local CPU for development, rendering and cached-embedding training; Colab GPU for heavy training.
- Team: solo.
- Goal: a side project whose main deliverable is an interactive demo.

## 1. Why an overhaul

An audit of v1 (the W266 submission) found that its headline results do not hold up:

- **Circular labels.** Archetype targets are produced by substring keyword counts over the same caption the text tower reads. The model re-learns a regex, the audio does not decide any label, and substring matches mislabel clips ("round" ⊂ "background", "thin" ⊂ "something").
- **No real lift over trivial baselines.** Test MSE is 0.064 vs 0.085 for predicting the mean vector; top-1 accuracy is 56.7% vs 50.5% for always predicting "sine".
- **Tuning compared training loss.** `validate_epoch()` is never called and every tuning JSON has an empty `val_loss_history`. The quantum hyperparameters (`circuit_depth`, `dropout_rate`, `noise_strength`) were never passed to the model, and the circuit has no trainable quantum parameters.
- **The transform is not DDSP.** HPSS runs in librosa/numpy, so no gradient flows. The output appears in no loss, so the "learnable" filters never train. On centroid error (5–6 kHz vs 0.9 kHz) and MFCC similarity, it scores worse than multiplying the input by 1.2.
- **RLHF is not policy gradient.** The loss is `-sum(log w) * reward`, which is independent of the action taken. Only 1–3 real ratings were ever collected.
- **Conceptual mismatch.** Oscillator archetypes (sine/square/saw/triangle/noise) describe sound *sources*, but the task is *transforming* existing audio. Words like "distorted" and "crunchy" were mapped to *noise*, but physically distortion is harmonic generation.

v1 is preserved as tag `v1-w266`. v2 is a redesign, not a patch.

## 2. Goal and scope

**Task:** given a guitar recording (DI or lightly processed) and a natural-language tone instruction, predict the settings of a physically grounded, differentiable effects chain, render the transformed audio, and explain the result.

> "give it a warm, mid-forward overdrive like a Tube Screamer, not too fizzy" + clean DI → `{drive: gain=0.62, asym=0.1, tone=0.4, pre_hpf=720 Hz …}` → audio + "≈ TS-style soft clipping; harmonic profile moved sine → triangle/square"

**In scope (v2.0):** single-instrument electric guitar; drive-family effects (overdrive, distortion, fuzz), EQ/tone, and compression. Absolute instructions ("make it sound X").

**Deliverable:** a local Gradio demo. Upload or record a DI clip, type an instruction, and hear the result alongside the predicted knob settings, the nearest real pedal, and the archetype/harmonic shift. Manual knob overrides let you tweak from there. Optionally hosted later on Hugging Face Spaces.

**Later (v2.x):** modulation (chorus, phaser, flanger), delay/reverb, relative edits ("a bit less fizz"), preference-based fine-tuning, and the quantum bottleneck add-on experiment.

**Out of scope:** full-mix music (MusicCaps), source separation, real-time plugins.

**Why narrow to guitar:** pedals act on a single instrument, and DI guitar data with known processing exists or can be synthesized. That gives ground-truth labels, which removes v1's core flaw.

## 3. Core idea: three layers of physics

| Layer | Role | Differentiable? | Used for |
|---|---|---|---|
| **White-box circuits** | Netlist / ODE simulation of real pedals (diode and transistor equations, component values) | No (offline) | Generating realistic training and test audio |
| **Grey-box pedalboard** | Filter → waveshaper → filter blocks (Wiener–Hammerstein) whose parameters map to circuit quantities (e.g., fc = 1/2πRC) | Yes | The instrument the model plays; gradients flow through it |
| **Archetype readout** | Measures the harmonic profile of any audio and projects it onto {sine, triangle, square, saw} + noise fraction | Deterministic, not learned | Explanation, auxiliary metric, link to v1 |

How the v1 archetypes survive: the archetypes are harmonic fingerprints, and pedal circuits are the operators that move a signal between them.
- Sine has no harmonics; triangle has odd harmonics at 1/n²; square has odd harmonics at 1/n; saw has all harmonics at 1/n.
- Symmetric clipping pushes sine → square. Asymmetric clipping adds even harmonics → saw-like. A low-pass tone control pulls back toward triangle/sine.

So the archetypes stop being a *control space* (v1) and become a *measurement space* (v2).

## 4. System architecture

```
                 ┌──────────────────────────┐
 text ──────────►│ Text encoder (frozen CLAP │──┐
                 │ text) + trainable adapter│  │
                 └──────────────────────────┘  │   ┌───────────────────────┐
                                               ├──►│ Parameter predictor   │
                 ┌──────────────────────────┐  │   │ (fusion MLP, per-block│
 input audio ───►│ Audio encoder (frozen    │──┘   │ heads, on/off gates)  │
     │           │ CLAP audio) + adapter    │      └──────────┬────────────┘
     │           └──────────────────────────┘                 │ normalized params ∈ [0,1]
     │                                                        ▼
     │           ┌──────────────────────────────────────────────────────────┐
     └──────────►│ Differentiable pedalboard (grey-box)                      │
                 │ Compressor → Drive [pre-HPF, gain, waveshaper(softness,   │
                 │ asymmetry), post-tone, level] → 3-band EQ  (oversampled)  │
                 └──────────────────────┬───────────────────────────────────┘
                                        ▼
                                  output audio ──► Archetype readout + descriptors
                                        │                    │
                                        └──► Explanation: nearest pedal in knowledge base,
                                             harmonic-profile shift, physical params
```

### 4.1 Encoders
- **Text:** frozen LAION-CLAP text encoder (shares an embedding space with audio, so CLAP scores can serve as a loss and a metric), plus a 2-layer adapter. A sentence-transformer variant is kept as an ablation.
- **Audio:** frozen CLAP audio encoder on the *input* clip, so the model can condition on the starting tone (an already-bright input needs less treble). An AST/PANNs alternative is kept behind a config flag.
- Rationale: v1 trained a ResNet/AST from scratch on about 3.7k clips. Pretrained and frozen encoders plus small heads fit the data scale.

### 4.2 Parameter predictor
- The fused embedding goes through an MLP, then per-block heads. Each head emits sigmoid-normalized parameters and a block-enable gate.
- Normalized parameters map to physical units through the **physics layer** (§4.4), e.g. `gain_norm → dB` using ranges taken from real pedal knob tapers.
- Optional v2.x: a distributional head (Beta/Gaussian per parameter) for uncertainty and for the preference-learning stage.

### 4.3 Differentiable pedalboard (`dsp/`) — implemented in P1
Pure PyTorch, with no torchaudio or dasp dependency. Contracts are in `dsp/base.py`:
- Audio is `(B, T)`; knobs are `(B,)`.
- Blocks take **normalized** knobs in [0, 1] and map them to physical units via `ParamSpec` (linear/log taper).
- Knobs are constant per clip, processing is zero-latency, and wet/dry gating lives in the pedalboard.

| Block | Chain | Knobs (physical range) |
|---|---|---|
| `Drive` | biquad HPF → gain → 4× oversampled waveshaper → tilt (1 kHz pivot) → level | pre_hpf_hz 20–1500 (log), gain_db 0–60, softness 0–1, asymmetry 0–1, bias ±0.25, tone_db ±12, level_db −36…+6 |
| `EQ3` | low shelf 120 Hz, peak (Q 0.9), high shelf 3 kHz | low/mid/high_db ±12, mid_hz 250–4000 (log) |
| `Compressor` | peak detect → soft-knee gain computer → attack/release → makeup | threshold_db −60…0, ratio 1–20 (log), attack 0.5–100 ms (log), release 10–1000 ms (log), makeup 0–24 dB |
| `Gain` | — | gain_db ±24 |

Module notes:
- **Filters** (`filters.py`): RBJ biquads applied by frequency sampling, an FIR approximation of the IIR filter with FFT size ≥ 2T; coefficients are computed in float64.
- **Waveshaper**: `x/(1+|x|^p)^(1/p)` with `p = 2·6^softness`. Asymmetry lowers the negative ceiling to −0.5. Bias is renormalized, so keep |bias| ≤ 0.25.
- **Oversampling**: Kaiser-windowed sinc, −90 dB stopband, zero phase.
- **Compressor**: control-rate recursion in NumPy behind a custom autograd function. It is fast on CPU but syncs the GPU on CUDA.
- **Pedalboard** (`pedalboard.py`): `default_pedalboard()` = compressor → drive → eq. Per-block `enabled` gate (`y = g·block(x) + (1−g)·x`), plus `default_params`, `random_params`, `layout`, `to_vector`/`from_vector` and `describe`.
- **Recovery check** (`recovery.py`): the P1 exit gate passed as an *audio-match* criterion (96%). In parameter space, `drive.asymmetry` and `drive.bias` trade off, and the swept `eq.mid_hz` is prone to local minima. See [reports/param_recovery.md](../reports/param_recovery.md).

### 4.4 Physics layer and pedal knowledge base (`physics/`, `pedals/`)
- One YAML per pedal: family, topology, clipping device (Si/Ge/LED/MOSFET), key component values, knob tapers, a source (e.g., an ElectroSmash analysis), and optional review-derived descriptors.
- Derivation functions turn component values and knob positions into grey-box parameters. Example: Tube Screamer clipping-stage HPF ≈ 720 Hz from 4.7 kΩ/0.047 µF, and gain = 1 + (51k + R_drive)/4.7k.
- Uses:
  1. ranges and priors for the parameter predictor;
  2. presets for synthetic rendering;
  3. nearest-pedal retrieval for explanations;
  4. a link from text (pedal names, reviews) to physics.
- Starter set: TS808, ProCo RAT, Boss DS-1, Fuzz Face (Si + Ge), Big Muff Pi. Grow to about 20 pedals.

### 4.5 White-box reference sims (`physics/whitebox/`)
- Offline, non-differentiable, for 2–3 circuits: one diode-clipper solver in feedback (TS808) and shunt (DS-1 / RAT clipping stage) configurations, plus a time-boxed Fuzz Face transistor model. Linear stages come from component values via the bilinear transform.
- Purpose: realistic data, and a check on grey-box fidelity (§7).
- Sims and grey-box share one level calibration (0 dBFS ↔ 1 V peak at the pedal input), so derived drive gains are comparable.
- Solver and calibration: P2 kickoff decisions 3 and 5 (pure NumPy/SciPy, no ngspice/system deps; 0 dBFS ↔ 1 V peak).
- Implemented (P2): `whitebox/diode_clipper.py` (trapezoidal rule + Newton per sample, 4–8× oversampled, numba-compiled when the `whitebox` extra is installed), `whitebox/port.py` (exact nodal solve when linear networks load the diode node), TS808 and DS-1 models sharing their linear stages with `physics/derive.py`. The Fuzz Face (time-boxed stretch) and RAT (diagnostic) white-boxes were not built in P2; they stay grey-box only. Fidelity: [reports/greybox_fidelity.md](../reports/greybox_fidelity.md).

### 4.6 Archetype readout and descriptors (`analysis/`)
- f0 tracking (pYIN or torchcrepe) → harmonic amplitudes → odd/even energy ratio, harmonic slope, harmonic-to-noise ratio.
- NNLS projection of the harmonic profile onto the four reference spectra, plus a noise fraction from the HNR, gives an archetype vector. Unit-tested on pure oscillators.
- Standard descriptors: spectral centroid, rolloff, crest factor, RMS, MFCC.

## 5. Data design

| Source | Content | Role |
|---|---|---|
| DI guitar: Guitar-TECHS (CC BY 4.0, primary), EGFxSet clean notes (CC BY 4.0), own DI recordings; IDMT-SMT-Guitar private-only (CC BY-NC-ND); EGDB pending license confirmation | Clean input audio | Dry inputs |
| Synthetic renders | dry × pedal preset × random knob settings → wet, via **white-box** sims (primary) and grey-box (secondary) | Training pairs with **ground-truth params** |
| Real pedal recordings: EGFxSet (CC BY 4.0, primary), pOD-set (CC BY-NC 4.0, gain sweeps), ToneTwist AFx analog subset (internal numbers only) | Real hardware outputs | Sim-to-real test set |
| TONE3000 (ex-ToneHunt) NAM captures, hand-picked (~20–50; site ToS forbids bulk/scripted download); manifest in repo, `.nam` files never redistributed | Named community captures of real pedals | Extra realistic renders + names for text grounding |
| Pedal knowledge base + reviews | Specs, schematics, descriptive text | Text vocabulary, priors, explanations |
| **Gold instruction set** | ~150 human-written instructions with matching target tones. Written by the owner **before** any caption template exists, plus ~50 from guitarist friends/volunteers who never see the templates | **Primary text test set** |

### Captions without circularity
- Training captions are generated *from the known processing* (pedal family, parameter values, measured descriptors) via templates plus LLM paraphrase. The label is the actual render and its parameters, never a keyword count over the input text.
- Template text is narrow, so the model could learn to invert the templates. Two guards:
  1. headline results are reported on the **gold set** written by humans who haven't seen the templates;
  2. caption-generator vocabulary is audited against the gold set.
- Splits are disjoint by DI recording/performer, and there is a **held-out pedal** split (unseen circuits) to test generalization.
- Every dataset version is frozen with a content hash and a data card.

Licensing details, conditions and open questions: [data-licenses.md](data-licenses.md).

## 6. Training

**Losses** (weights in config):
- `L_param` — L1 on normalized params (synthetic data only), with gating BCE.
- `L_audio` — multi-resolution STFT loss (auraloss) between rendered output and target wet audio.
- `L_clap` (optional, low weight) — CLAP distorted-vs-clean contrast on the render, for the drive-amount direction only. Spike A ([findings](../spikes/clap_vocab/FINDINGS.md)) showed that only the drive-amount contrast is robust across CLAP checkpoints. Brightness is model-dependent, and mids and guitar jargon fail, so it is not used as a general text-to-tone loss.
- `L_desc` (optional) — descriptor/archetype-shift consistency with the target.

**Procedure:**
- Curriculum: parameters only → add the audio loss → add the CLAP loss.
- AdamW + cosine schedule sized to the real step count.
- Early stopping on the validation **gold-proxy** metric. Test sets are touched once, at the end.

**Preference stage (v2.x):** collect pairwise A/B preferences, fit a Bradley-Terry reward model, then fine-tune a distributional parameter head with KL regularization to the supervised policy. This replaces v1's RLHF.

## 7. Evaluation

**Metrics:**
- Parameter: normalized L1 per parameter; block-gate F1.
- Audio: MR-STFT distance to target; errors in centroid, odd/even ratio, HNR and archetype vector.
- Text adherence: CLAP score; gold-set accuracy in 2AFC-vs-distractor form.
- Perceptual: listening test (pairwise preference + "matches description" Likert), ≥10 listeners.
- Grey-box fidelity: harmonic-spectrum dB error of the best-fit grey-box vs white-box (the P2 gate), and of the derived, unfitted grey-box (accuracy of the physics mapping).

**Baselines** (required, in every results table):
- no-op (dry passthrough);
- mean parameters;
- kNN retrieval (nearest training caption → its params);
- LLM zero-shot parameter prediction (LLM2Fx-style);
- per-example CLAP-gradient optimization of params (Text2FX-style);
- v1 model, where applicable.

**Ablations:**
- text-only / audio-only / shuffled audio;
- grey-box physics mapping vs unconstrained params;
- frozen vs fine-tuned encoders;
- (v2.x) quantum bottleneck vs a matched classical bottleneck of identical width.

**Statistics:** ≥5 seeds, bootstrap 95% CIs, paired tests on per-item metrics, and an effect size reported alongside every p-value.

## 8. Repository structure

```
pyproject.toml           # uv-managed, pinned deps
configs/                 # YAML (OmegaConf): data/, model/, train/, eval/, experiment/
pedals/                  # knowledge-base YAMLs (schema = dataclasses in physics/kb.py)
src/lstmabar/
  dsp/                   # filters, waveshapers, oversampling, blocks, pedalboard
  physics/               # param derivations, knob tapers, whitebox/ sims
  analysis/              # f0, harmonics, archetype readout, descriptors
  data/                  # source ingestion, renderer, captioner, datasets, splits
  models/                # encoders, adapters, predictor, (quantum/)
  train/                 # losses, trainer loop, checkpoints (config + git SHA embedded)
  eval/                  # metrics, baselines, stats, listening-test export, report
  demo/                  # Gradio app: upload/record → instruction → audio + knobs + explanation
  cli.py                 # render / train / eval / report / demo entry points
colab/                   # thin launchers: clone repo, pip install -e ., call CLI on GPU
tests/                   # unit + gradient-flow + param-recovery tests
notebooks/               # thin: import from src/, no copied classes
legacy/                  # v1 code, read-only reference for the paper
```

**Engineering rules:**
- Seeds everywhere.
- Checkpoints embed the resolved config + git SHA.
- One run directory per experiment (`runs/<name>/<timestamp>/`) holding config, metrics JSON, TensorBoard logs and the checkpoint.
- CI runs ruff + pytest on CPU with tiny fixtures.

## 9. Key risks

| Risk | Mitigation |
|---|---|
| CLAP barely knows guitar-tone words ("fizzy", "mid hump") | **Confirmed by spike A** (mean word AUC ≈ 0.6 in both checkpoints; jargon fails; brightness is model-dependent; only the drive-amount contrast is robust). Text side is learned from paired synthetic data into parameter space; `L_clap` is restricted to drive amount; evaluation leans on descriptors, the gold set and listening tests |
| Sim-to-real gap | White-box data + NAM captures; dedicated real-pedal test set |
| Template captions reintroduce circularity | Gold human set as the headline metric; held-out vocabulary audit |
| Aliasing / vanishing gradients in clipping | Oversampling, smooth surrogates, gradient-flow tests |
| Dataset licenses / scraping ToS | License check before ingestion; curated KB first, scraping only where permitted |
| Compute (local machine has no NVIDIA GPU) | Local CPU: DSP dev, tests, rendering (multiprocess), CLAP embedding caching, head-only training on cached embeddings, demo inference. Colab GPU: training with `L_audio`/`L_clap` through the oversampled pedalboard, multi-seed sweeps. The code stays device-agnostic, and Colab notebooks only `pip install` the repo and call the CLI |
| Solo bandwidth | Demo-first milestones (Demo v0 early on baselines, Demo v1 on the model); white-box sims limited to 2 circuits; NAM captures as the realism fallback; listening test sized to ~5–8 friends |

## 10. Decisions log

| Date | Decision |
|---|---|
| 2026-10-09 | Guitar-only scope |
| 2026-10-09 | Quantum bottleneck kept as an add-on experiment (P7), trainable VQC vs matched classical bottleneck |
| 2026-10-09 | Local CPU (Python 3.12 via `py`, `uv`-managed env) + Colab GPU for heavy training |
| 2026-10-09 | Solo project; the demo is the primary deliverable, with rigorous evaluation kept but right-sized |
| 2026-10-09 | Spike A: CLAP only robustly tracks the drive-amount contrast, so `L_clap` is optional and drive-only; the text side is learned from paired synthetic data ([findings](../spikes/clap_vocab/FINDINGS.md)) |
| 2026-10-09 | Spike B: Guitar-TECHS + EGFxSet (CC BY) are the primary DI and real-pedal sources; TONE3000 captures are hand-picked only; IDMT is private-eval only; GuitarSet is dropped (acoustic) ([licenses](data-licenses.md)) |
| 2026-10-09 | Gold set provenance: `llm_draft`, `llm_persona` and `simulated_human` (all LLM-written) stay separate from real `human` items; clipping conventions: Muff soft, Klon hard, last drive in a stack sets clipping |
| 2026-10-09 | P1 DSP: normalized-knob contract, pure-torch frequency-sampled biquads, NumPy compressor recursion behind custom autograd, 4× oversampled p-norm waveshaper, Drive bias range ±0.25 |
| 2026-10-09 | Python 3.11 pinned locally (`.python-version`, supersedes the 3.12 note above); CI also tests 3.13 to match Colab |
| 2026-10-09 | Demo decodes uploads itself (libsndfile → PyAV) instead of relying on a system ffmpeg; user-facing errors never include server paths; 25 MB upload cap, 192 kHz sample-rate cap |
| 2026-10-09 | P2 kickoff: one YAML per pedal (`pedals/`, SI-suffix strings, dataclass validator); 0 dBFS ↔ 1 V peak calibration; RAT gain clamped at Drive's 60 dB; white-box = one diode-clipper solver (feedback: TS808, shunt: DS-1/RAT) as the fidelity gate, Fuzz Face time-boxed stretch |
| 2026-10-10 | P2 closed: fidelity gate on the *masked* mean (both-floored dBc terms excluded); DS-1 passes (1.13 dB), TS808 passes via a documented deviation with ceilings (3.48 dB; clean-path gap); numba is the `whitebox` extra |
