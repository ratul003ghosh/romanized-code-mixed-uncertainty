
## 2026-09-27 · PII detector (Zarif, #10 #11)
- Code: src/pii/detector.py, masking.py, arbitrate.py; eval: scripts/eval_pii.py
- Hand-made set (34 cases, written with the detector): recall 1.00, leakage 0.0%. Regression check only.
- Hard held-out set (18 cases, not tuned on): recall 0.467, precision 0.875, leakage 46.7%, over-masking 0.0%.
- Misses: names without a keyword, free-form addresses, obfuscated e-mails, "double" in spoken numbers, OTP keyword after the number.
- Synthetic data only; real measurement needs the gold set.
