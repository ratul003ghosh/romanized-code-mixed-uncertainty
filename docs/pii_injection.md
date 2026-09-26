# PII injection and noise (proposal v2, section 3.2)

Owner: himel (data lead). Code: `src/pii_inject/`, `src/noise/`, `scripts/make_silver_inputs.py`.
Tests: `tests/test_pii_inject.py`.

## What it produces

Silver **inputs** for the teacher ensemble: prompts with synthetic PII at exact character
offsets. They are not training targets; normalization, uncertainty channels and routing
come from the teachers (section 4). Nothing here is gold data.

Each JSONL record:

| Field | Meaning |
|---|---|
| `input` | the prompt text |
| `pii` | `type` (schema v1.0.0 names), `span`, `start`, `end`, `canonical`, `regime` |
| `preserved_entities` | `AMOUNT` / `SERVICE` with offsets; never masked |
| `hints` | generation facts: `PII_BOUNDARY` for bare IDs, `NUMERIC_AMBIGUITY` for amounts like `5oo tk` |
| `noise_applied` | word-level noise operations used |
| `meta` | generator version, seed, date, commit, member, shard; `synthetic`: `true` or `"pii_only"` |
| `pii_reference` | optional (`--with-reference`): schema-valid, PII-only output for PII F1, leakage and B1 |

Invariant (tested): `input[start:end] == span` for every entity.

## Regimes

- **canonical**: `01712345678`, `+8801712345678`, `4539 1488 0343 6467`
- **spoken**: digits as words, `zero one seven ...` or `shunno ek shat ...`
- **perturbed**: delimiters (`017-123-45678`), digit-letter swaps (`O1712...`), Bangla-script
  digits (`০১৭...`), attached postpositions (`01712345678e`, `Mitu Royke`)

Report leakage and PII F1 per regime (section 6.1).

## Two modes

1. **Template mode** (fully synthetic): Banglish finance templates in `templates.py`.
2. **Carrier mode** (real text + synthetic PII): PII clauses attached to sentences from
   BanglaTLit, 5-Dialects-BN etc. The carrier text is kept unchanged.

```bash
python scripts/make_silver_inputs.py --n-template 5000 --member himel --with-reference \
    --out data/silver/inputs_template_v0.jsonl
python scripts/make_silver_inputs.py --carrier /kaggle/input/<dataset>/banglatlit_romanized.txt \
    --carrier-source BanglaTLit --carrier-surface-form romanized --n-carrier 20000 \
    --member himel --out data/silver/inputs_banglatlit_v0.jsonl
```

## Before using carrier mode

Scan carrier text for real PII first (a regex detector is still to be written, Step 5 of
the plan) and drop or label hits; unlabeled real PII in silver inputs teaches the student
to leave PII unmasked.

## Open items (record decisions in docs/decisions.md)

- VERIFY transaction-ID formats (bKash, Nagad, Rocket) and bank account lengths.
- Native speakers review and extend `templates.py`, including dialect variants.
- Tune regime weights and noise rates on the pilot.
- Regex PII detector (baseline B1, runtime rule 1, real-PII scan).
- 8-gram / MinHash contamination filter against the gold sets (section 3.4).
