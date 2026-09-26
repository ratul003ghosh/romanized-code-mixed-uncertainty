# Decision log

| Date | Decision | Why | Agreed by |
|---|---|---|---|
| 2026-09-24 | Output schema v1.0.0 frozen: pii items carry `span` and optional `score`; placeholders are `<TYPE_n>` | Needed for PII span F1, answer restoration and conformal risk control | |
| 2026-09-24 | Compute on Kaggle T4 x2, fp16; QLoRA for 7B teachers | No bf16 on T4; 16 GB per card | |
| 2026-09-27 | Silver inputs v0: 5,000 template records (seed 1) + 20,000 BanglaTLit carrier records with synthetic PII (seed 2), generator pii_inject-0.1.0. Kaggle dataset banglish-silver-inputs v1. | First input set for the teacher pilot (section 3.2) | himel (draft) |
| 2026-09-27 | Default split: BanglaTLit test = gold pool, val = dev pool, train + PT = silver (overlaps with the pools removed) | Keeps gold sentences out of training (section 3.4) | himel (draft) |
| 2026-09-27 | Carrier lines with digit runs of 7+, Bangla digit runs, e-mails or +880 dropped before injection | Scraped text may contain real PII; stricter regex detector to follow | himel (draft) |
| 2026-09-27 | Silver inputs v0: 5,000 template records (seed 1) + 20,000 BanglaTLit carrier records with synthetic PII (seed 2), generator pii_inject-0.1.0. Kaggle dataset banglish-silver-inputs v1. | First input set for the teacher pilot (section 3.2) | himel (draft) |
| 2026-09-27 | Default split: BanglaTLit test = gold pool, val = dev pool, train + PT = silver (overlaps with the pools removed) | Keeps gold sentences out of training (section 3.4) | himel (draft) |
| 2026-09-27 | Carrier lines with digit runs of 7+, Bangla digit runs, e-mails or +880 dropped before injection | Scraped text may contain real PII; stricter regex detector to follow | himel (draft) |
| 2026-09-27 | Silver inputs v0: 5,000 template records (seed 1) + 20,000 BanglaTLit carrier records with synthetic PII (seed 2), generator pii_inject-0.1.0. Kaggle dataset banglish-silver-inputs v1. | First input set for the teacher pilot (section 3.2) | himel (draft) |
| 2026-09-27 | Default split: BanglaTLit test = gold pool, val = dev pool, train + PT = silver (overlaps with the pools removed) | Keeps gold sentences out of training (section 3.4) | himel (draft) |
| 2026-09-27 | Carrier lines with digit runs of 7+, Bangla digit runs, e-mails or +880 dropped before injection | Scraped text may contain real PII; stricter regex detector to follow | himel (draft) |
