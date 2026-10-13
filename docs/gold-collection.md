# Collecting friend-written gold instructions (task 4.5)

Real people's instructions (`source: human`) are the **headline test set**: every
instruction-following result is reported on them first, and the LLM-written gold files
(`simulated_human`, `llm_persona`, `llm_draft`) are reported separately. They are held-out
test data and are never used for training or for tuning the caption templates
(design §5, [data/gold/README.md](../data/gold/README.md)).

Target: **about 50 instructions from at least 5 friends**, 5–10 each.

## What friends are told

Send friends only this section and the example WAV files. Do not send them the repo, the
docs, the training captions or any parameter values.

- Write in your own words, the way you would ask a bandmate, a tech or a plugin for a sound.
  **Don't use ChatGPT or any other AI tool**, and don't look at the project's code, docs or
  example captions.
- One tone per instruction. Any length or style is fine: a few words, a paragraph, slang,
  typos, other languages mixed in.
- You may name pedals, amps, artists, songs or records.
- Don't put your name, email, links or phone numbers in the instruction.

Two kinds of instruction:

- **(A) Describe this example** (`mode = example`). Listen to a dry clip (`exNN_dry.wav`) and a
  processed example (`exNN_wet.wav`). Write the instruction you would give to turn the dry
  sound into the example. Give the example's id (`ex01`, `ex02`, ...).
- **(B) Free** (`mode = free`). A tone you actually want, for a song, a gig or a recording.
  Optionally say what guitar, amp or song it is for.

## Example clips (owner)

`lstmabar data gold-examples` writes `data/rendered/gold_examples/<example_id>_dry.wav` and
`<example_id>_wet.wav` (ids `ex01`, `ex02`, ...) plus `examples.json`. Share **only the WAV
files**. `examples.json` holds the settings that made each example; never share it or any
parameter values. The dry audio is Guitar-TECHS DI (CC BY 4.0); credit it when you share
the clips: *Pedroza, Abreu, Corey, Roman, "Guitar-TECHS: An Electric Guitar Dataset Covering
Techniques, Musical Excerpts, Chords and Scales Using a Diverse Array of Hardware",
ICASSP 2025 (arXiv:2501.03720), CC BY 4.0.*

## Form / CSV columns

One row per instruction. A Google Form works: export the responses as CSV and rename the
headers to exactly these names (extra columns are ignored; the file is read as UTF-8).

| Column | Required | Values |
|---|---|---|
| `timestamp` | no | anything (not imported) |
| `respondent` | yes | name or handle; anonymised on import, never stored |
| `mode` | yes | `example` or `free` |
| `example_id` | iff `mode = example` | `ex01`, `ex02`, ... (must be empty for `free`; the column may be omitted from a free-only form) |
| `instruction` | yes | the instruction, at most 1000 characters |
| `context` | no | guitar / amp / song; stored in the item's `notes` |
| `own_words` | yes | must be `yes`: "I wrote this myself, without AI tools" |
| `consent` | yes | must be `yes`: "this may be used, anonymised, as test data in the project and its reports" |

For the two yes/no questions a form option whose text starts with "Yes" also counts as yes.

## Privacy

- Real names are never committed or printed. The import maps each respondent to
  `friend-NN` (first-seen order, stable across imports) through
  `data/cache/gold/authors.json` (`gold.authors_file`), keyed by the sha256 of the
  normalised name. `data/cache/` is gitignored; keep that file, or returning friends get new
  pseudonyms.
- Rows whose instruction or context contain an email address, a URL, a phone number or the
  respondent's own name are rejected; fix or drop them in the CSV and re-import.
- Keep the raw CSV outside the repo, or delete it after the import.

## Import and labelling

1. `uv run lstmabar data gold-import --input friends.csv --dry-run` prints the rows it would
   reject (`row N: <reason>`, never the content) and the ids it would append.
2. Run it again without `--dry-run`. Accepted rows are appended to
   `data/gold/instructions_human.yaml` (append-only; existing items are untouched) with ids
   after the last one (`h151`, ...), `source: human`, `author: friend-NN`, the instruction
   whitespace-normalised, `example_id` for mode A, and every `target` field `null`.
   Re-importing the same CSV appends nothing (exact duplicates of any gold instruction are
   rejected), so a growing form export can be re-imported.
3. The owner labels each new item's `target` fields by hand, following the conventions in
   [data/gold/README.md](../data/gold/README.md). Downstream steps skip items whose target is
   still unlabelled.
4. Target regions are then drafted with `lstmabar data gold-targets`.

Config: `configs/data/gold_v1.yaml` (`gold.*`; the `gold-*` steps use it by default).
