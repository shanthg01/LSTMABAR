# CLAP vocabulary spike: results

Pipeline sanity (sine / white noise / plucked string vs matching text) and
audio-embedding spread across the 22 tone clips (cosine; 1.0 = identical).

| Model | Sanity top-1 | Mean off-diag cos | Min off-diag cos |
|---|---|---|---|
| laion/clap-htsat-unfused | 1.00 | 0.712 | 0.437 |
| laion/larger_clap_general | 1.00 | 0.740 | 0.458 |

Mean ROC-AUC over 30 tone words (0.5 = chance) and top-1 hit rate, per prompt template.

| Model | Template | Mean AUC | Top-1 hit |
|---|---|---|---|
| laion/clap-htsat-unfused | word | 0.603 | 0.33 |
| laion/clap-htsat-unfused | tone | 0.540 | 0.30 |
| laion/clap-htsat-unfused | sound | 0.546 | 0.23 |
| laion/larger_clap_general | word | 0.510 | 0.10 |
| laion/larger_clap_general | tone | 0.607 | 0.33 |
| laion/larger_clap_general | sound | 0.570 | 0.33 |

Sweeps: Spearman rho of similarity vs sweep value, per riff (lead, chords). 
Expected sign: distorted +, clean −, bright +, dark −; contrasts (a-b) +.

| Model | gain:distorted | gain:clean | gain:distorted-clean | cutoff:bright | cutoff:dark | cutoff:bright-dark |
|---|---|---|---|---|---|---|
| laion/clap-htsat-unfused | +0.49 / -0.60 | -1.00 / -0.94 | +1.00 / +0.94 | -1.00 / +0.26 | -1.00 / -0.26 | +0.94 / +0.83 |
| laion/larger_clap_general | +1.00 / +1.00 | -0.54 / +0.43 | +0.83 / +0.77 | +0.49 / +1.00 | +0.60 / +0.83 | -1.00 / +0.26 |

## laion/clap-htsat-unfused: per-word AUC (template 'tone' for both models)

| Word | AUC | Top-1 tone | Hit |
|---|---|---|---|
| clean | 0.67 | soft_od | no |
| glassy | 0.79 | soft_od | no |
| sparkly | 0.54 | reverb | no |
| bright | 0.39 | reverb | no |
| twangy | 0.85 | soft_od | no |
| dark | 0.40 | soft_od | no |
| muddy | 0.65 | soft_od | no |
| warm | 0.94 | soft_od | yes |
| smooth | 0.86 | soft_od | yes |
| overdriven | 0.60 | soft_od | yes |
| crunchy | 0.60 | soft_od | yes |
| distorted | 0.56 | soft_od | no |
| heavy | 0.08 | soft_od | no |
| aggressive | 0.22 | soft_od | no |
| fuzzy | 0.38 | soft_od | no |
| fizzy | 0.15 | soft_od | no |
| gritty | 0.70 | soft_od | yes |
| saturated | 0.35 | soft_od | yes |
| creamy | 0.95 | soft_od | yes |
| mid-heavy | 0.93 | soft_od | yes |
| scooped | 0.03 | soft_od | no |
| chugging metal | 0.23 | soft_od | no |
| compressed | 0.65 | soft_od | no |
| squashed | 0.53 | soft_od | no |
| lo-fi | 0.28 | soft_od | no |
| shimmering | 0.33 | reverb | no |
| watery | 0.55 | soft_od | no |
| spacious | 0.55 | soft_od | no |
| ambient | 0.60 | soft_od | no |
| Tube Screamer | 0.85 | soft_od | yes |

## laion/clap-htsat-unfused: top-5 words per tone (z-scored per word)

- **clean**: clean, fizzy, bright, warm, smooth
- **dark**: ambient, smooth, spacious, lo-fi, warm
- **bright**: fizzy, twangy, scooped, sparkly, glassy
- **soft_od**: muddy, aggressive, squashed, dark, gritty
- **hard_dist**: squashed, distorted, Tube Screamer, overdriven, gritty
- **fuzz**: squashed, compressed, Tube Screamer, gritty, distorted
- **metal_scooped**: distorted, overdriven, crunchy, gritty, squashed
- **compressed**: scooped, saturated, chugging metal, overdriven, creamy
- **chorus**: chugging metal, muddy, gritty, watery, crunchy
- **reverb**: sparkly, shimmering, bright, glassy, watery
- **lofi**: scooped, clean, twangy, sparkly, creamy

## laion/larger_clap_general: per-word AUC (template 'tone' for both models)

| Word | AUC | Top-1 tone | Hit |
|---|---|---|---|
| clean | 0.43 | reverb | yes |
| glassy | 0.58 | reverb | no |
| sparkly | 0.47 | reverb | no |
| bright | 0.69 | reverb | no |
| twangy | 0.43 | fuzz | no |
| dark | 0.12 | reverb | no |
| muddy | 0.57 | hard_dist | no |
| warm | 0.46 | reverb | no |
| smooth | 0.25 | reverb | no |
| overdriven | 0.75 | hard_dist | yes |
| crunchy | 0.65 | hard_dist | yes |
| distorted | 1.00 | hard_dist | yes |
| heavy | 0.97 | fuzz | yes |
| aggressive | 0.89 | fuzz | no |
| fuzzy | 0.85 | fuzz | yes |
| fizzy | 0.76 | reverb | no |
| gritty | 0.80 | hard_dist | yes |
| saturated | 0.87 | hard_dist | yes |
| creamy | 0.60 | reverb | no |
| mid-heavy | 0.82 | reverb | no |
| scooped | 0.40 | chorus | no |
| chugging metal | 0.12 | chorus | no |
| compressed | 0.30 | fuzz | no |
| squashed | 0.55 | hard_dist | no |
| lo-fi | 0.75 | fuzz | no |
| shimmering | 0.25 | reverb | no |
| watery | 0.42 | reverb | no |
| spacious | 0.93 | reverb | yes |
| ambient | 0.93 | reverb | yes |
| Tube Screamer | 0.57 | hard_dist | no |

## laion/larger_clap_general: top-5 words per tone (z-scored per word)

- **clean**: clean, chugging metal, scooped, warm, lo-fi
- **dark**: mid-heavy, lo-fi, scooped, chugging metal, twangy
- **bright**: clean, glassy, sparkly, chugging metal, spacious
- **soft_od**: chugging metal, mid-heavy, scooped, creamy, twangy
- **hard_dist**: distorted, compressed, muddy, gritty, overdriven
- **fuzz**: compressed, distorted, overdriven, heavy, gritty
- **metal_scooped**: distorted, muddy, squashed, gritty, fuzzy
- **compressed**: chugging metal, Tube Screamer, scooped, squashed, overdriven
- **chorus**: chugging metal, scooped, Tube Screamer, watery, squashed
- **reverb**: smooth, creamy, shimmering, warm, sparkly
- **lofi**: lo-fi, saturated, creamy, ambient, overdriven
