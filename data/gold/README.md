# Gold instruction set (held-out evaluation data)

Natural-language guitar-tone instructions paired with target effect settings
(family, gain, brightness, mids, compression, clipping, reference pedal, scope).
Used only to evaluate instruction-to-effect mapping.

## Files

| File | ids | `source` | Written by | Role |
|---|---|---|---|---|
| `instructions_human.yaml` | `h###` | `human` | Real guitarists (`author` required) | **Headline test set** |
| `instructions_llm_persona.yaml` | `p###` | `llm_persona` | An LLM role-playing 8 distinct guitarist personas (`author: persona-N`, personas listed in the file header) | Supplementary test set |
| `instructions_llm_draft.yaml` | `g###` | `llm_draft` | An LLM, single voice | Supplementary test set |

All LLM-written items were produced by isolated agents that did not read the project's docs, code or caption templates. LLM-written items must never be relabeled as `human`. Report results per file, because LLM-written test text may share phrasing habits with LLM-paraphrased training captions.

## Schema

Each item: `id`, `source`, `author` (human and persona files), `instruction`, `target`, `notes`.
`target` fields and allowed values:

- `in_scope`: bool. False if the request mainly needs modulation, delay, reverb, octave, pitch, wah or amp simulation. Mixed requests count as in scope when the drive part is the main ask.
- `family`: clean | overdrive | distortion | fuzz | none (none = EQ/compression only)
- `gain`: none | low | medium | high | extreme
- `brightness`: dark | slightly_dark | neutral | slightly_bright | bright
- `mids`: scooped | neutral | pushed
- `compression`: none | light | heavy
- `clipping`: soft | hard | asymmetric | unknown
- `reference_pedal`: drive/compression pedal name if named or clearly implied, else null

## Label conventions

- Clean and none families use `clipping: unknown`. A clean boost is `family: clean`, `gain: low`.
- Tube Screamer: soft, mids pushed. RAT / DS-1: hard. Fuzz Face / Tone Bender: asymmetric.
- Big Muff: soft (diodes in each stage's feedback loop). Klon: hard (Ge diodes to ground).
  `instructions_llm_draft.yaml` predates these two conventions and still labels Big Muff
  `hard` and Klon `soft`. Normalize it before using `clipping` in any metric.

## Rules

- **Do NOT read this directory when writing training caption templates or training data.**
  It is held-out test data; any leakage invalidates the evaluation.
- Keep ids unique within each file and use the file's prefix.
