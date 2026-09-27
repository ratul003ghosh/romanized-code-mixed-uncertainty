# Architecture & Research Decisions

Record important decisions here:
* teacher model choice:
* student model choice:
* quantization:
* uncertainty formulation:
* dataset selection:
* PII strategy:
* routing thresholds:
* GPU limitations:

## 2026-09-27 · PII module (proposed by Himel, for Saber and the team to approve)
* PII types for schema v0.2 confirmed: PHONE, NID, ID_NUMBER, TXN_ID, ACCOUNT, CARD, OTP, EMAIL, NAME, ADDRESS
* PII detection: src/pii/detect.py is the pipeline PII Masking stage (proposal 4.6 rule 1); the regex baseline B1 (PR #24) stays the comparison system
* IDs: BG_SYN_000001 (template), BG_PII_000001 / HG_PII_000001 (real text + synthetic PII), to avoid collisions with BG_000001
* label_source: synthetic for template records; none for real text sent to the teachers (injected PII kept in pii as ground truth)
* optional pii[].regime (canonical / spoken / perturbed) for per-regime leakage (proposal 6.1)
| 2026-09-27 | Silver inputs v0: 5,000 template records (seed 1) + 20,000 BanglaTLit carrier records with synthetic PII (seed 2), generator pii_inject-0.1.0. Kaggle dataset banglish-silver-inputs v1. | First input set for the teacher pilot (section 3.2) | himel (draft) |
| 2026-09-27 | Default split: BanglaTLit test = gold pool, val = dev pool, train + PT = silver (overlaps with the pools removed) | Keeps gold sentences out of training (section 3.4) | himel (draft) |
| 2026-09-27 | Carrier lines with digit runs of 7+, Bangla digit runs, e-mails or +880 dropped before injection | Scraped text may contain real PII; stricter regex detector to follow | himel (draft) |
