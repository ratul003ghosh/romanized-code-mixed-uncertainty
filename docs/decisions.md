# Decision log

| Date | Decision | Why | Agreed by |
|---|---|---|---|
| 2026-09-24 | Output schema v1.0.0 frozen: pii items carry `span` and optional `score`; placeholders are `<TYPE_n>` | Needed for PII span F1, answer restoration and conformal risk control | |
| 2026-09-24 | Compute on Kaggle T4 x2, fp16; QLoRA for 7B teachers | No bf16 on T4; 16 GB per card | |
