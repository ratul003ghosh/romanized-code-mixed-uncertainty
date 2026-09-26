# Data Schema (v0.1)

Owner: Saber. Every module reads and writes this format.
Example record: `data/schemas/example.json`.
Based on the proposal, Section 2 and Section 2.1.

## Fields

| Field | Type | Meaning |
|---|---|---|
| id | string | Unique ID, e.g. BG_000001 (Banglish), HG_000001 (Hinglish) |
| schema_version | string | Currently "0.1" |
| input | string | Original raw text. Never changed. |
| clean_input | string | Input after light cleaning (spaces, dots, invisible characters). Words, digits and case unchanged. |
| normalized_text | string | Normalized text produced by teachers or student (evaluated with CER/WER). Empty until then. |
| sanitized_prompt | string | Clean, PII-masked prompt (teacher/student output) |
| pii | list | Detected PII spans (see below) |
| preserved_entities | list | Non-PII entities that must be kept, e.g. amounts |
| uncertainties | list | Uncertain spans (see below) |
| routing | string | One of the 4 routes below |
| metadata | object | Language, source, label_source, split |

## Span positions

`start` and `end` are character positions in `clean_input` (Python style: `clean_input[start:end]` gives the span text).
Positions always refer to `clean_input` (the text PII detection runs on), not `input` or `normalized_text`.

## Allowed values

**Uncertainty types** (proposal Section 2.1). A span may have more than one:
TRANSLITERATION, NUMERIC_AMBIGUITY, PII_BOUNDARY, DIALECT_MEANING, INTENT

**Routing** (exactly one): PROCEED, PROCEED_WITH_FLAGS, ASK_USER, ESCALATE

**PII types** (draft, Himel to confirm): PHONE, NID, ID_NUMBER, TXN_ID, ACCOUNT, CARD, OTP, EMAIL, NAME
- Use ID_NUMBER when a number is clearly an identifier but its exact type is unclear.

**Placeholders**: `<TYPE_N>`, numbered per prompt, e.g. `<PHONE_1>`, `<PHONE_2>`.
The same PII value gets the same placeholder within one prompt.

**Scores**: `aleatoric` and `epistemic` are numbers from 0.0 to 1.0, two decimals.

## metadata.label_source

| Value | Meaning | Allowed use |
|---|---|---|
| silver | Produced by the teacher ensemble | Training only |
| gold | Checked by human annotators | Evaluation only |
| synthetic | Written by the team with fake PII | Testing PII code |
| illustrative | Hand-made example, numbers not from experiments | Documentation only |
| none | No labels yet (raw text) | Input to teachers |

**Rule: never evaluate on silver data, never train on gold data.**

## metadata.split

train, dev, test, or none. Splits are made by source, so no gold sentence (or near-copy) appears in training (proposal Section 3.4).

## Safety

Never put real PII in any file committed to GitHub. Use fake values only.
