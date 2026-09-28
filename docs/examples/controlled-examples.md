# Controlled Examples (schema v0.2)

Original author: Mashrafi. Converted and maintained by: Saber.

## What they are
19 hand-written Banglish sentences, each built to test one specific case
(numeric ambiguity, transliteration, dialect, PII, intent, combinations).
They are for development, testing and the demo.

They are NOT the research dataset and NOT human gold data.
Every record has `label_source: "synthetic"` and `split: "none"`.
Never train on them and never report them as research results.

## Files
| File | What it is |
|---|---|
| `data/synthetic/controlled_examples_v02.jsonl` | The examples in schema v0.2 (use this one) |
| `data/synthetic/mashrafi_original/` | Mashrafi's original files, unchanged |
| `scripts/convert_controlled_examples.py` | Converts the original to v0.2 |
| `scripts/check_controlled_examples.py` | Checks counts and positions |

Check them with:

    PYTHONUTF8=1 python scripts/check_controlled_examples.py

## Coverage
| Category | Count |
|---|---|
| CLEAN_CONTROL | 3 |
| NUMERIC_AMBIGUITY | 3 |
| TRANSLITERATION_AMBIGUITY | 3 |
| DIALECT_MEANING | 2 (placeholders, see below) |
| INTENT_AMBIGUITY | 2 |
| PII_PHONE / PII_TRANSACTION_ID / PII_ID_BOUNDARY | 1 each |
| COMBINED / COMBINED_PII_BOUNDARY | 2 / 1 |

## What each field means here
- `routing`: the route the author EXPECTS. It is a design expectation, not a measured result.
- `aleatoric`, `epistemic`: always `null`. Scores must come from the teachers, never be written by hand.
- `human_ambiguous`: `null`, because these are not human-annotated.
- `sanitized_prompt`: the author's intended output. It contains markers like
  `<AMBIGUOUS_AMOUNT>` and `<AMBIGUOUS_TIME_REFERENCE>`; these are not PII placeholders.
- `metadata.placeholder_input: true`: the input is not a real sentence. Skip it when running models.

## Known limits
- The 2 dialect examples use `<CONFIRMED_DIALECT_TERM>` instead of a real word, because no verified
  dialect term and meaning were available. Real dialect examples should come from 5-Dialects-BN
  (proposal Section 3.1).
- Mashrafi's original marked `5oo tk` as aleatoric HIGH. The proposal (Section 2) says the opposite
  (a human reads 500, so aleatoric is low; teachers disagree, so epistemic is high). All hand-written
  signals were removed in v0.2.
- 12 of 19 examples expect ASK_USER; routes are unbalanced.
- Only 1 example each for phone, transaction ID and ID-number PII.
- All PII values are fake (e.g. 01712345678, 9X87K, 1987654321).

## Who uses them
- Sadat: test the evaluator end to end
- Himel: test PII detection and masking
- Zarif: small teacher smoke runs
- Demo (Day 3): show each uncertainty type on one clear sentence
