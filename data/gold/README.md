# Gold instruction set (held-out evaluation data)

Natural-language guitar-tone instructions paired with target effect settings
(family, gain, brightness, mids, compression, clipping, reference pedal, scope).
Used only to evaluate instruction-to-effect mapping.

## Files

| File | ids | `source` | Written by | Role |
|---|---|---|---|---|
| `instructions_human.yaml` | `h###` | `simulated_human` (h001–h150); `human` for real people, appended after | `simulated_human`: 10 independent LLM agents, each role-playing one invented guitarist in an isolated context (`author: sim-NN`, 15 each). `human`: real guitarists (`author` = pseudonym) | Human-style test set. Real `human` items, once added, are the headline subset |
| `instructions_llm_persona.yaml` | `p###` | `llm_persona` | An LLM role-playing 8 distinct guitarist personas (`author: persona-N`, personas listed in the file header) | Supplementary test set |
| `instructions_llm_draft.yaml` | `g###` | `llm_draft` | An LLM, single voice | Supplementary test set |

`targets/targets_v1.yaml` (not an instruction file; kept in a subdirectory so `*.yaml` globs
here see only instructions) maps each labelled item to a target board-param region and
contradicting distractor regions, owner-reviewed (task 4.5, `lstmabar data gold-targets`).
`lstmabar data gold-pair` renders them on the `test` DI clips; the manifest is
`data/manifests/gold_v1.jsonl` (+ `.index.json`), audio under `data/rendered/gold_v1/`.

All LLM-written items (`llm_draft`, `llm_persona`, `simulated_human`) were produced by isolated agents that did not read the project's docs, code, caption templates or other gold files. LLM-written items must never be relabeled as `human`. Report results per file, because LLM-written test text may share phrasing habits with LLM-paraphrased training captions.

## Schema

Each item: `id`, `source`, `author` (human and persona files), `instruction`, `target`, `notes`,
and optionally `example_id` (`source: human` items written for a shared example clip, mode A
in [docs/gold-collection.md](../../docs/gold-collection.md): `ex01`, `ex02`, ...).
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
- Stacked drives: the last drive in the chain sets `clipping` (and its EQ character), e.g. "Muff into a RAT" → hard.
  As of 2026-10-09 all gold files, including `instructions_llm_draft.yaml`, follow these
  conventions.

## Real (`source: human`) items

- Added only by `lstmabar data gold-import --input <responses.csv>` (protocol and CSV
  columns: [docs/gold-collection.md](../../docs/gold-collection.md)). The import is
  append-only: it appends after the last id (h151, ...) and verifies that every existing
  item, including all `simulated_human` items, is byte-for-byte unchanged.
- `author` is an anonymised pseudonym `friend-NN`. Real names are never committed; the
  respondent → pseudonym mapping stays local in the gitignored cache
  (`data/cache/gold/authors.json`, keyed by a hash of the name).
- **Unlabelled targets:** imported items have every `target` field `null` until the owner
  labels them by hand with the conventions above. Downstream steps skip items whose target
  is unlabelled. `notes` holds the respondent's optional context (guitar / amp / song).

## Rules

- **Do NOT read this directory when writing training caption templates or training data.**
  It is held-out test data; any leakage invalidates the evaluation.
- Keep ids unique within each file and use the file's prefix.
