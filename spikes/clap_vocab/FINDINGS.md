# Spike A (P0.6): does CLAP understand guitar-tone vocabulary?

Date: 2026-10-09 · Script: [run.py](run.py) · Full numbers: [results/report.md](results/report.md), [results/results.json](results/results.json)

## Setup

- **Audio:** 2 synthetic riffs (Karplus-Strong single-note line + power chords, 6 s) × 11 tones (clean, dark, bright, soft OD, hard distortion, fuzz, scooped metal, compressed, chorus, reverb, lo-fi). Every clip is RMS-matched to 0.030 with no peak limiting (verified: 0.02999–0.03001). Plus a 6-step drive-gain sweep and a 6-step low-pass cutoff sweep per riff.
- **Text:** 30 tone words × 3 prompt templates (`word`, `tone` = "a {w} electric guitar tone", `sound`). Expected word→tone matches were pre-registered in `EXPECTED` before any run.
- **Models (Hugging Face `transformers` 5.19):** `laion/clap-htsat-unfused` ("unfused") and `laion/larger_clap_general` ("general"). Both pass a pipeline sanity check (sine / white noise / plucked string → 3/3).
- `laion/larger_clap_music` returns exactly uniform probabilities even on the sanity check under this `transformers` version (a checkpoint/loader incompatibility), so it was **not evaluated**.

## Results

| | Unfused | General |
|---|---|---|
| Mean word AUC, best template (0.5 = chance) | 0.60 (`word`) | 0.61 (`tone`) |
| Top-1 hit rate, same template | 0.33 | 0.33 |
| Drive sweep, distorted−clean contrast (Spearman ρ, lead / chords; n = 6 steps each) | +1.00 / +0.94 | +0.83 / +0.77 |
| Brightness sweep, bright−dark contrast | +0.94 / +0.83 | −1.00 / +0.26 |

Per-word AUCs below use the `tone` template for both models.

**General model:**
- **Drive amount and space work:** "distorted" 1.00, "heavy" 0.97, "aggressive" 0.89, "saturated" 0.87, "fuzzy" 0.85, "gritty" 0.80, "spacious"/"ambient" 0.93.
- **Clean and dark adjectives fail:** "clean" 0.43, "dark" 0.12, "smooth" 0.25, "shimmering" 0.25.

**Unfused model:**
- **Distortion words fail:** "distorted" 0.56, "heavy" 0.08, "aggressive" 0.22, "fizzy" 0.15. Its single-word "distorted" sweep is mixed (+0.49 / −0.60); only the distorted−clean *contrast* tracks drive.
- **Some adjectives score high:** "warm" 0.94, "creamy" 0.95, "mid-heavy" 0.93, "smooth" 0.86, "Tube Screamer" 0.85 (also "twangy" 0.85, despite missing its top-1). However, this model ranks `soft_od` top for 27 of 30 words, a hub effect. Most of the high scores are for words whose expected tones include `soft_od`, so they are weak evidence of real vocabulary knowledge.

**Both models fail:**
- **Guitar jargon:** "scooped" 0.03 / 0.40, "chugging metal" 0.23 / 0.12, "compressed" 0.65 / 0.30.
- **"dark":** 0.40 / 0.12, below chance in both.
- **Agreement:** the two models agree on little beyond broad drive and space.

**Effect of loudness matching:** an earlier run with imperfect loudness matching (compressed clips ~6–8 dB, bright ~2–6 dB and one clean clip ~1 dB quieter) gave the unfused brightness contrast as +0.94 / −0.31. With exact matching it is +0.94 / +0.83, so level was confounding brightness judgements.

## Conclusions

1. **No CLAP checkpoint reliably understands guitar-tone vocabulary.** Mean word AUC is about 0.6 in both models, and each has a different, partial competence. CLAP cannot be the primary text-to-tone signal.
2. **The drive-amount contrast (distorted − clean) is the one robust signal.** It is positive and strong in both models (ρ 0.77–1.00, though with only 6 steps per riff). Brightness tracks in the unfused model only, so it is not reliable across models.
3. **The text side must be learned from our own paired data.** Use a frozen text encoder plus a trained adapter that maps into *effect-parameter space*, supervised by `L_param` / `L_audio` on synthetic renders with known settings, not zero-shot alignment in the CLAP joint space.
4. **`L_clap` is demoted to an optional, low-weight auxiliary term** restricted to the drive-amount contrast. Brightness, mids and jargon are left to `L_param` / `L_audio`.
5. **Evaluation should lean on measured descriptors** (centroid, odd/even ratio, gain/HNR, archetype readout), the gold set and the listening test. Report CLAP score only as a coarse secondary metric.
6. **The Text2FX-style baseline** (per-example CLAP optimization) is expected to be weak on jargon. Keep it, since that contrast is part of the story.
7. **P6 ablation:** CLAP text encoder vs a sentence-transformer as the frozen text backbone.

## Caveats

- The audio is synthetic and likely out of distribution for CLAP. Rerun on real DI guitar (e.g. GuitarSet, pending the P0.7 license check) through the same tones before treating these numbers as final.
- Small sample: 2 riffs, 11 tones, one seed. Each per-word AUC rests on 22 clips; each sweep ρ on 6 points.
- The expected word→tone mapping is one guitarist's judgment (pre-registered, not tuned after seeing results).
- Per-word AUC and top-1 interact with hub effects (one tone scoring high for most words). Top-1 is the more affected of the two.

## Reproduce

```bash
uv sync --extra dev --extra spikes
uv run python spikes/clap_vocab/run.py --save-audio   # ~3–5 min on CPU; audio → results/audio/ (gitignored)
```
