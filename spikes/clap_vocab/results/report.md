# CLAP vocabulary spike: results

Pipeline sanity (sine / white noise / plucked string vs matching text) and
audio-embedding spread across the 22 tone clips (cosine; 1.0 = identical).

| Model | Sanity top-1 | Mean off-diag cos | Min off-diag cos |
|---|---|---|---|
| laion/clap-htsat-unfused | 1.00 | 0.710 | 0.451 |
| laion/larger_clap_general | 1.00 | 0.727 | 0.432 |

Mean ROC-AUC over 30 tone words (0.5 = chance) and top-1 hit rate, per prompt template.

| Model | Template | Mean AUC | Top-1 hit |
|---|---|---|---|
| laion/clap-htsat-unfused | word | 0.595 | 0.23 |
| laion/clap-htsat-unfused | tone | 0.534 | 0.27 |
| laion/clap-htsat-unfused | sound | 0.521 | 0.23 |
| laion/larger_clap_general | word | 0.523 | 0.20 |
| laion/larger_clap_general | tone | 0.595 | 0.37 |
| laion/larger_clap_general | sound | 0.569 | 0.37 |

Sweeps: Spearman rho of similarity vs sweep value, per riff (lead, chords). 
Expected sign: distorted +, clean −, bright +, dark −; contrasts (a-b) +.

| Model | gain:distorted | gain:clean | gain:distorted-clean | cutoff:bright | cutoff:dark | cutoff:bright-dark |
|---|---|---|---|---|---|---|
| laion/clap-htsat-unfused | +0.54 / -0.77 | -1.00 / -0.94 | +1.00 / +1.00 | -1.00 / +0.83 | -1.00 / +0.83 | +0.94 / -0.31 |
| laion/larger_clap_general | +1.00 / +1.00 | -0.54 / +0.43 | +0.77 / +0.77 | +0.03 / +0.94 | +0.31 / +0.94 | -0.09 / -0.60 |

## laion/clap-htsat-unfused: per-word AUC (template 'tone')

| Word | AUC | Top-1 tone | Hit |
|---|---|---|---|
| clean | 0.76 | soft_od | no |
| glassy | 0.88 | bright | yes |
| sparkly | 0.68 | reverb | no |
| bright | 0.50 | reverb | no |
| twangy | 0.88 | bright | yes |
| dark | 0.39 | soft_od | no |
| muddy | 0.51 | soft_od | no |
| warm | 0.86 | dark | yes |
| smooth | 0.85 | bright | no |
| overdriven | 0.58 | soft_od | yes |
| crunchy | 0.51 | soft_od | yes |
| distorted | 0.54 | soft_od | no |
| heavy | 0.06 | soft_od | no |
| aggressive | 0.22 | soft_od | no |
| fuzzy | 0.33 | soft_od | no |
| fizzy | 0.06 | bright | no |
| gritty | 0.53 | soft_od | yes |
| saturated | 0.36 | soft_od | yes |
| creamy | 0.82 | bright | no |
| mid-heavy | 0.90 | soft_od | yes |
| scooped | 0.00 | bright | no |
| chugging metal | 0.25 | soft_od | no |
| compressed | 0.80 | soft_od | no |
| squashed | 0.82 | soft_od | no |
| lo-fi | 0.38 | soft_od | no |
| shimmering | 0.33 | reverb | no |
| watery | 0.47 | soft_od | no |
| spacious | 0.47 | soft_od | no |
| ambient | 0.47 | soft_od | no |
| Tube Screamer | 0.80 | bright | no |

## laion/clap-htsat-unfused: top-5 words per tone (z-scored per word)

- **clean**: clean, warm, fizzy, bright, smooth
- **dark**: ambient, lo-fi, smooth, spacious, warm
- **bright**: fizzy, twangy, Tube Screamer, glassy, dark
- **soft_od**: aggressive, muddy, heavy, gritty, crunchy
- **hard_dist**: distorted, Tube Screamer, overdriven, squashed, gritty
- **fuzz**: distorted, squashed, Tube Screamer, gritty, overdriven
- **metal_scooped**: distorted, overdriven, crunchy, aggressive, gritty
- **compressed**: scooped, compressed, saturated, ambient, creamy
- **chorus**: chugging metal, distorted, squashed, watery, gritty
- **reverb**: sparkly, bright, shimmering, warm, watery
- **lofi**: scooped, clean, twangy, sparkly, lo-fi

## laion/larger_clap_general: per-word AUC (template 'tone')

| Word | AUC | Top-1 tone | Hit |
|---|---|---|---|
| clean | 0.40 | reverb | yes |
| glassy | 0.57 | reverb | no |
| sparkly | 0.57 | reverb | no |
| bright | 0.68 | reverb | no |
| twangy | 0.57 | reverb | no |
| dark | 0.17 | reverb | no |
| muddy | 0.50 | hard_dist | no |
| warm | 0.50 | reverb | no |
| smooth | 0.20 | reverb | no |
| overdriven | 0.72 | hard_dist | yes |
| crunchy | 0.69 | hard_dist | yes |
| distorted | 0.96 | hard_dist | yes |
| heavy | 0.88 | fuzz | yes |
| aggressive | 0.86 | fuzz | no |
| fuzzy | 0.75 | fuzz | yes |
| fizzy | 0.62 | reverb | no |
| gritty | 0.74 | hard_dist | yes |
| saturated | 0.79 | hard_dist | yes |
| creamy | 0.62 | reverb | no |
| mid-heavy | 0.80 | soft_od | yes |
| scooped | 0.38 | soft_od | no |
| chugging metal | 0.03 | chorus | no |
| compressed | 0.33 | fuzz | no |
| squashed | 0.57 | hard_dist | no |
| lo-fi | 0.68 | fuzz | no |
| shimmering | 0.38 | reverb | no |
| watery | 0.45 | reverb | no |
| spacious | 0.93 | reverb | yes |
| ambient | 0.93 | reverb | yes |
| Tube Screamer | 0.60 | chorus | no |

## laion/larger_clap_general: top-5 words per tone (z-scored per word)

- **clean**: chugging metal, mid-heavy, scooped, lo-fi, warm
- **dark**: mid-heavy, lo-fi, scooped, chugging metal, twangy
- **bright**: chugging metal, lo-fi, scooped, glassy, warm
- **soft_od**: chugging metal, mid-heavy, scooped, Tube Screamer, twangy
- **hard_dist**: distorted, compressed, overdriven, squashed, muddy
- **fuzz**: compressed, distorted, overdriven, gritty, squashed
- **metal_scooped**: squashed, distorted, muddy, gritty, overdriven
- **compressed**: chugging metal, Tube Screamer, scooped, mid-heavy, squashed
- **chorus**: chugging metal, Tube Screamer, watery, scooped, squashed
- **reverb**: shimmering, smooth, sparkly, creamy, warm
- **lofi**: lo-fi, saturated, creamy, ambient, compressed
