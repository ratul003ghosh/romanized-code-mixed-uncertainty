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

## 2026-09-27 · PII injection and preprocessing (himel, draft, needs team review)
* dataset selection: silver inputs v0 = 5,000 template records (seed 1) + 20,000 BanglaTLit carrier records with synthetic PII (seed 2); generator pii_inject-0.1.0; stored in private Kaggle dataset banglish-silver-inputs
* split: BanglaTLit test = gold pool, val = dev pool, train + pre-training corpus = silver; silver lines matching the pools are removed (proposal 3.4)
* PII strategy: carrier lines with 7+ digit runs, Bangla digit runs, e-mails or +880 are dropped before injection, until the regex PII detector replaces this filter
