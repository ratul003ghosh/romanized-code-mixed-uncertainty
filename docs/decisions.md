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

## 2026-09-27 · PII injection (proposed by Himel, for Saber to approve)
* PII types for schema v0.2 confirmed: PHONE, NID, ID_NUMBER, TXN_ID, ACCOUNT, CARD, OTP, EMAIL, NAME, ADDRESS
* IDs: BG_SYN_000001 (template), BG_PII_000001 / HG_PII_000001 (real text + synthetic PII), to avoid collisions with BG_000001
* label_source: synthetic for template records; none for real text sent to the teachers (injected PII kept in pii as ground truth)
* optional pii[].regime (canonical / spoken / perturbed) for per-regime leakage (proposal 6.1)
* carrier lines with 7+ digit runs, Bangla digit runs, e-mails or +880 are dropped before injection, until a regex detector replaces this filter
