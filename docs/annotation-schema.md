# Data Schema (v0.2)

Owner: Saber. Every module reads and writes this format.
Example record: `data/schemas/example.json`.
Based on the proposal, Section 2, 2.1 and 3.3.

## Changes from v0.1
- Placeholders always follow `<TYPE_N>` (the ID_NUMBER example now uses `<ID_NUMBER_1>`).
- New optional field `human_ambiguous` on each uncertainty.
- "Not set yet" is `null`.
- Allowed `surface_form` values are listed.
- `ADDRESS` added to PII types; `ID_NUMBER` is the only generic ID type.

## Fields

| Field | Type | Meaning |
|---|---|---|
| id | string | Unique ID, e.g. BG_000001 (Banglish), HG_000001 (Hinglish) |
| schema_version | string | Currently "0.2" |
| input | string | Original raw text. Never changed. |
| clean_input | string | Input after light cleaning (spaces, dots, invisible characters). Words, digits and case unchanged. |
| normalized_text | string or null | Normalized text produced by teachers or student (evaluated with CER/WER) |
| sanitized_prompt | string or null | Clean, PII-masked prompt (teacher/student output) |
| pii | list | Detected PII spans |
| preserved_entities | list | Non-PII entities that must be kept, e.g. amounts |
| uncertainties | list | Uncertain spans |
| routing | string or null | One of the 4 routes below |
| metadata | object | language, surface_form, source, label_source, split |

## "Not set yet"

A field that has not been produced yet is `null` (for example `routing: null` before the decision layer runs).
Lists that are empty stay `[]`. Do not use empty strings `""` to mean "not set".

## Span positions

`start` and `end` are character positions in `clean_input` (Python style: `clean_input[start:end]` gives the span text).
Positions always refer to `clean_input` (the text PII detection runs on), not `input` or `normalized_text`.

## Uncertainty entry

| Key | Type | Meaning |
|---|---|---|
| span | string | The uncertain text |
| start, end | int | Position in `clean_input` |
| types | list | One or more uncertainty types |
| candidates | list | Possible readings |
| aleatoric | number or null | 0.0 to 1.0, two decimals |
| epistemic | number or null | 0.0 to 1.0, two decimals |
| human_ambiguous | bool or null | Gold records only: true if a native reader cannot decide the meaning from context. Ground truth for the aleatoric channel (proposal Section 3.3). `null` for silver/synthetic. |

## Allowed values

**Uncertainty types** (proposal Section 2.1). A span may have more than one:
TRANSLITERATION, NUMERIC_AMBIGUITY, PII_BOUNDARY, DIALECT_MEANING, INTENT

**Routing** (exactly one, or null): PROCEED, PROCEED_WITH_FLAGS, ASK_USER, ESCALATE

**PII types** (Himel to confirm): PHONE, NID, ID_NUMBER, TXN_ID, ACCOUNT, CARD, OTP, EMAIL, NAME, ADDRESS
- Use ID_NUMBER when a number is clearly an identifier but its exact type is unclear.

**Placeholders**: always `<TYPE_N>`, numbered per prompt, e.g. `<PHONE_1>`, `<PHONE_2>`, `<ID_NUMBER_1>`.
The same PII value gets the same placeholder within one prompt.

**surface_form** (results are reported per surface form, proposal Section 6.2):
banglish, romanized, code_mixed, dialect, hinglish

## metadata.label_source

| Value | Meaning | Allowed use |
|---|---|---|
| silver | Produced by the teacher ensemble | Training only |
| gold | Checked by human annotators | Evaluation only |
| synthetic | Written by the team with fake PII | Testing PII code |
| illustrative | Hand-made example, numbers not from experiments | Documentation only |
| prediction | Output of a model (e.g. scripts/infer_student.py) | Evaluation input only |
| none | No labels yet (raw text) | Input to teachers |

**Rule: never evaluate on silver data, never train on gold data.**

## metadata.split

train, dev, test, or none. Splits are made by source, so no gold sentence (or near-copy) appears in training (proposal Section 3.4).

## Safety

Never put real PII in any file committed to GitHub. Use fake values only.
