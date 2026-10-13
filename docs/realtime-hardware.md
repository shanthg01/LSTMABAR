# Real-time hardware: a pedalboard substitute (post-v2 sketch)

**Status:** exploration only, not scheduled. Real-time plugins are out of scope for v2.0 (design §2). This note records what it would take to run the system live, as a box that sits between guitar and amp and is controlled from an iPad, so the option is costed if it is picked up after M4. Prices are rough 2026 street prices.

## Target

- Guitar → box → amp. No cabinet simulation, because the box feeds a real amp.
- Footswitch bypass and preset switching. Latency stays below what a player notices: at most ~3 ms round trip, ideally ~1 ms.
- An iPad app where you type a tone description, get knob values plus an explanation, adjust individual blocks, and save presets to the box.
- Decent, gig-usable quality at minimum cost.

The key split: **only the DSP has to be real-time.** Text → parameters runs once per request, so a few seconds on the iPad or a server is fine.

## What blocks real-time today

The `dsp/` pedalboard is clip-based and non-causal by design (it is a differentiable training renderer). It cannot be streamed as-is:

| Component | Current (offline) | Real-time requirement |
|---|---|---|
| Filters (`filters.py`) | RBJ biquads applied by FFT frequency sampling over the whole clip | Same RBJ coefficients, run per sample (transposed direct form II) |
| Oversampling (`oversample.py`) | Symmetric Kaiser sinc, applied centred (zero phase); `HALF_WIDTH = 32` | Causal polyphase filter. The linear-phase version adds ~32 samples per direction (~1.3 ms round trip at 48 kHz). A minimum-phase or IIR half-band design cuts this to well under 1 ms. |
| Compressor (`compressor.py`) | Control-rate frames, gain placed at frame centres, which looks ahead in time | Per-sample (or causal-block) peak detector and attack/release; no lookahead |
| Knobs | Constant per clip | Parameter smoothing, so live knob or preset changes don't click ("zipper noise") |
| Not present | — | Noise gate (needed at high gain), denormal flushing, output trim for amp input level |
| White-box TS808 (`whitebox/`) | Newton solve per sample, 4–8× oversampled | Too heavy at 8× on a microcontroller. Use precomputed lookup tables for the clipper (a standard technique), or ship grey-box only. |

**Phase mismatch.** The FFT-sampled biquads already carry the causal IIR phase, so a per-sample port should match them in phase too. The mismatch comes only from the zero-phase oversampler and the compressor's frame-centre look-ahead: there the causal ports reproduce magnitude but not phase. Validate them with magnitude-domain null tests against PyTorch renders, using the multi-resolution STFT loss already in `dsp/losses.py`. If the predictor is retrained later, consider rendering its data through the causal path.

## What to build

| # | Piece | Work | Difficulty | Rough effort (evenings) |
|---|---|---|---|---|
| 1 | Portable C++ DSP core | Dependency-free C++ port of `Compressor`, `Drive`, `EQ3` and `Gain`, with the `ParamSpec` tapers. Golden tests against PyTorch renders. | Medium (mostly translation) | 2–4 weeks |
| 2 | Desktop test host | Wrap the core in a JUCE plugin or a minimal host and play through a USB interface. Checks feel and latency before any hardware. | Easy–medium | ~1 week |
| 3 | Pedal firmware | Core on the target board, footswitch bypass and presets, physical knobs, presets in flash | Medium | ~2 weeks |
| 4 | Control link | Compact, versioned preset struct (16 knob floats plus 3 block enables for the default board; 17 floats with `Gain`) over BLE | Medium | 1–2 weeks |
| 5 | iPad app (SwiftUI) | Text box, sliders generated from the pedalboard `layout` (same rule as the demo: no hard-coded knob names), presets, explanation text, BLE | Medium (new stack) | 3–4 weeks |
| 6 | Text → params | Interim: an LLM + pedal-KB variant of the P5 LLM zero-shot baseline. Later: the trained predictor (P6), served from a small server or exported to CoreML. If the model needs the player's audio, capture ~5 s on the box and send it to the app (~0.5 MB, fine over BLE), or use the iPad mic. | Interim easy; the full model is the research project | — |
| 7 | Analog front end | Hi-Z input buffer, output level suited to an amp input, grounding, 9 V power | Easy with a kit carrier PCB; hard from scratch | — |

Total: roughly 3–4 months of evenings to a gig-usable box, assuming the interim LLM text control.

## What to buy

### Recommended: Daisy-based pedal

| Item | ~Cost |
|---|---|
| Electrosmith Daisy Seed (480 MHz Cortex-M7, 24-bit codec, ~1 ms latency, instant boot) | $30 |
| Pedal carrier PCB with I/O buffers, pots and footswitches (Terrarium/Hothouse-style kit) | $20–90 |
| Enclosure (125B), jacks, pots, footswitches if the kit lacks them | $20–30 |
| ESP32-S3 dev board as a BLE bridge (UART to the Daisy) | $10 |
| Apple Developer account (optional; free sideloading expires every 7 days) | $99/yr |
| LLM API for text → params | cents/month |
| Soldering iron and multimeter, if not already owned | $50–80 |

**Hardware total: about $80–160** (excluding tools and the developer account). For comparison, commercial programmable modelers cost $500+.

### Alternatives

| Option | ~Cost | Pros | Cons |
|---|---|---|---|
| **Daisy + ESP32 + native iPad app** (recommended) | $120 | A real pedal: instant boot, ~1 ms latency, rugged | Three codebases (firmware, ESP32, Swift) |
| **Raspberry Pi 4/5 + audio HAT + input buffer** | $150–220 | Wi-Fi built in. The web UI is served to iPad Safari, so no App Store. Can run the Python predictor on the box. | 20–30 s boot, ~3–6 ms latency, SD corruption if power is pulled (needs a read-only filesystem), less pedal-like |
| **iPad as the modeler** (USB interface into an AUv3 app) | $50–100 | No hardware build | iPad on stage and extra cabling; not a pedalboard replacement |

## Risks

1. **Noise.** The most common failure in DIY digital pedals. Start from a proven carrier PCB rather than a custom analog design.
2. **Coverage.** The pedalboard models only compressor, drive and EQ. A full board replacement also needs modulation, delay and reverb (P7 item 2). Off-the-shelf DaisySP blocks would cover these, without physics grounding.
3. **Into a real amp.** Drive voicings were validated on DI renders. An amp's front end changes how they respond, so expect to re-tune presets by ear.
4. **Text quality depends on the model.** Until P6 lands, text control is the interim LLM + KB mapping: good for preset-style requests ("TS808, a bit more bite"). The physics-grounded mapping arrives with the trained predictor.

## Suggested order

1. C++ core and desktop host. Gate: the C++ output matches the PyTorch renders in magnitude (loss threshold set when the port starts).
2. Daisy pedal with physical knobs and KB presets (TS808, DS-1). Usable at this point.
3. ESP32 bridge and iPad app with sliders and presets.
4. Text control: the interim LLM + KB mapping first, then the trained predictor.

Each step leaves a working device, so the project can stop at any step.
