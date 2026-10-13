# Vocabulary overlap: held-out instructions vs dataset captions

Generated once at freeze by `lstmabar data vocab-audit` (src/lstmabar/data/vocab_audit.py). Aggregate numbers only: no instruction, caption or word list is shown. **Caption templates and the paraphrase prompt must not change after this report;** if they do, the dataset becomes v1.1 ("tuned after gold audit").

Dataset SHA-256: `c4987b4ab834d3f7b4da87fcfc72d42fd995ad634f1939c7dfb7635291c93cfa`.

Instruction files (sha256):

- `instructions_human.yaml`: `49099b3760b652c0c3369b704ad9328a77c4338b4a6415fbd737de95033196c2`
- `instructions_llm_draft.yaml`: `d358650691f390a054ed24cc8a55906f068b9826db85fb8f73a68bd36244b4cc`
- `instructions_llm_persona.yaml`: `384b287f8f011ac98dee992c676f0d0529eedf548a2905f730e8ab68fa37e17c`

## Caption vocabularies

| captions | texts | tokens | word types |
|---|---|---|---|
| all | 50130 | 555584 | 1013 |
| template | 26812 | 289532 | 227 |
| paraphrase | 23318 | 266052 | 1010 |

## Coverage by instruction source

Each source is reported separately (`human` is the headline set; `simulated_human`, `llm_persona` and `llm_draft` are LLM-written). *Token coverage* = share of instruction word occurrences whose word appears anywhere in the captions; *type coverage* = share of distinct instruction words; *content* rows drop a generic English stopword list; *bigram* = adjacent word pairs; *Jaccard* = shared / union word types.

No items yet for: `human`.

### Against all captions

| source | items | tokens | types | token cov | type cov | content token cov | content type cov | bigram token cov | bigram type cov | Jaccard |
|---|---|---|---|---|---|---|---|---|---|---|
| simulated_human | 150 | 2895 | 870 | 68.3 % | 35.4 % | 46.5 % | 30.2 % | 24.2 % | 15.6 % | 19.6 % |
| llm_persona | 150 | 2087 | 771 | 67.4 % | 36.3 % | 45.8 % | 30.4 % | 26.3 % | 19.1 % | 18.6 % |
| llm_draft | 150 | 1512 | 594 | 71.6 % | 42.4 % | 54.5 % | 36.3 % | 31.6 % | 24.7 % | 18.6 % |

### Against template captions

| source | items | tokens | types | token cov | type cov | content token cov | content type cov | bigram token cov | bigram type cov | Jaccard |
|---|---|---|---|---|---|---|---|---|---|---|
| simulated_human | 150 | 2895 | 870 | 47.4 % | 15.7 % | 27.3 % | 12.8 % | 9.5 % | 4.7 % | 14.3 % |
| llm_persona | 150 | 2087 | 771 | 48.0 % | 17.5 % | 27.1 % | 14.3 % | 10.2 % | 6.3 % | 15.6 % |
| llm_draft | 150 | 1512 | 594 | 51.7 % | 22.9 % | 37.2 % | 19.3 % | 16.2 % | 9.9 % | 19.9 % |

### Against paraphrase captions

| source | items | tokens | types | token cov | type cov | content token cov | content type cov | bigram token cov | bigram type cov | Jaccard |
|---|---|---|---|---|---|---|---|---|---|---|
| simulated_human | 150 | 2895 | 870 | 68.3 % | 35.3 % | 46.5 % | 30.1 % | 24.2 % | 15.5 % | 19.5 % |
| llm_persona | 150 | 2087 | 771 | 67.4 % | 36.3 % | 45.8 % | 30.4 % | 26.3 % | 19.1 % | 18.7 % |
| llm_draft | 150 | 1512 | 594 | 71.6 % | 42.4 % | 54.5 % | 36.3 % | 31.6 % | 24.7 % | 18.6 % |
