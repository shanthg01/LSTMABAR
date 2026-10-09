# Gold instruction set (held-out evaluation data)

Natural-language guitar-tone instructions paired with target effect settings
(family, gain, brightness, mids, compression, clipping, reference pedal, scope).
Used only to evaluate instruction-to-effect mapping.

## Files

- `instructions_llm_draft.yaml`: 150 items drafted by an LLM (`source: llm_draft`),
  written without looking at the project's docs, code or caption templates.
  Labels are one reasonable interpretation; ambiguous items say so in `notes`.
- `instructions_human.yaml` (to be added): human-written items go in this separate file
  with `source: human`, using the same schema. Do not mix them into the LLM-drafted file.

## Rules

- **Do NOT read this directory when writing training caption templates or training data.**
  It is held-out test data; any leakage invalidates the evaluation.
- Keep ids unique within each file (`g###` for the LLM draft; use a distinct prefix such as `h###` for human items).
