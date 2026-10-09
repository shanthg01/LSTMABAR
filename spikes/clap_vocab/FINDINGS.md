# Spike A (P0.6): does CLAP understand guitar-tone vocabulary?

Date: 2026-10-09 · Script: [run.py](run.py) · Full numbers: [results/report.md](results/report.md), [results/results.json](results/results.json)

## Setup

- **Audio:** 2 synthetic riffs (Karplus-Strong single-note line + power chords, 6 s) × 11 tones (clean, dark, bright, soft OD, hard distortion, fuzz, scooped metal, compressed, chorus, reverb, lo-fi), loudness-matched. Plus a 6-step drive-gain sweep and a 6-step low-pass cutoff sweep per riff.
- **Text:** 30 tone words × 3 prompt templates. Expected word→tone matches were pre-registered in `EXPECTED` before any run.
- **Models (Hugging Face `transformers` 5.19):** `laion/clap-htsat-unfused` and `laion/larger_clap_general`. Both pass a pipeline sanity check (sine / white noise / plucked string → 3/3).
- `laion/larger_clap_music` returns exactly uniform probabilities even on the sanity check under this `transformers` version (a checkpoint/loader incompatibility), so it was **not evaluated**.

## Results

| | htsat-unfused | larger_clap_general |
|---|---|---|
| Mean word AUC (best template; 0.5 = chance) | 0.60 | 0.60 |
| Top-1 hit rate (best template) | 0.27 | 0.37 |
| Drive sweep, distorted−clean contrast (Spearman ρ, lead / chords) | **+1.00 / +1.00** | **+0.77 / +0.77** |
| Brightness sweep, bright−dark contrast | +0.94 / −0.31 | −0.09 / −0.60 |

What works (AUC ≥ 0.8 in at least one model):
- **Coarse drive amount:** "distorted" 0.96, "heavy" 0.88, "aggressive" 0.86, "saturated" 0.79 (general model).
- **Space:** "spacious" and "ambient" 0.93 (general model).
- **Some clean adjectives:** "glassy" and "twangy" 0.88, "warm" 0.86 (unfused model).
- The drive-gain contrast is monotonic in both models.

What fails:
- **Guitar jargon:** "scooped" 0.00 / 0.38, "fizzy" 0.06 / 0.62, "chugging metal" 0.25 / 0.03, "Tube Screamer" 0.80 / 0.60 (top match is the wrong tone in both).
- **Brightness and darkness:** "dark" 0.39 / 0.17, which is *inverted*. The cutoff sweep contrast changes sign between riffs and models.
- **Compression:** inconsistent, 0.80 / 0.33.
- The two models disagree word by word, and neither is reliable across the vocabulary.

## Conclusions

1. **CLAP knows "how distorted" and "how spacious", not guitar tone.** It cannot serve as the primary text-to-tone signal.
2. **The text side must be learned from our own paired data.** Use a frozen text encoder plus a trained adapter that maps into *effect-parameter space*, supervised by `L_param` / `L_audio` on synthetic renders with known settings. Do not rely on zero-shot alignment in the CLAP joint space.
3. **`L_clap` is demoted to an optional, low-weight auxiliary term** restricted to the drive-amount direction (distorted vs clean contrast). It is never used for brightness, mids or jargon.
4. **Evaluation should lean on measured descriptors** (centroid, odd/even ratio, gain/HNR, archetype readout), the gold set and the listening test. Report CLAP score only as a coarse secondary metric.
5. **The Text2FX-style baseline** (per-example CLAP optimization) is expected to be weak on jargon. Keep it; that contrast is part of the story.
6. **P6 ablation:** CLAP text encoder vs a sentence-transformer as the frozen text backbone.

## Caveats

- The audio is synthetic and likely out of distribution for CLAP. Rerun on real DI guitar (e.g. GuitarSet, pending the P0.7 license check) through the same tones before treating these numbers as final.
- Small sample: 2 riffs, 11 tones, one seed. AUCs per word rest on 22 clips.
- The expected word→tone mapping is one guitarist's judgment (pre-registered, not tuned after seeing results).

## Reproduce

```bash
uv sync --extra dev --extra spikes
uv run python spikes/clap_vocab/run.py --save-audio   # ~5 min on CPU; audio → results/audio/ (gitignored)
```
