# Experiment Log

This file tracks experiment runs, models, splits, and evaluation metrics across the project.

---

## [PIPELINE SANITY CHECK] Run 001: Regex Baseline Dry-Run (DO NOT USE IN RESULTS TABLE)
- **Type**: Pipeline Sanity Check (Synthetic Dry-Run — Exclude from Paper Results)
- **Date**: 2026-09-26
- **Lead**: Sadat
- **Model**: `src/baselines/regex_baseline.py` (Regex & Rule-Based PII detector)
- **Dataset**: `data/synthetic/sample_synthetic.jsonl` (Hand-crafted synthetic smoke test set)
- **Predictions**: `experiments/baseline/sample_regex_predictions.jsonl`
- **Full Report**: `experiments/baseline/sample_report.md`
- **Output JSON**: `experiments/baseline/sample_eval.json`

> **Note**: This run is purely a code verification dry-run to ensure the evaluation metric pipeline, I/O formatting, and schema compatibility operate without runtime errors. It does NOT represent empirical research results.

### Summary Metrics (Pipeline Smoke Test)
| Metric | Value | Interpretation |
|---|---|---|
| **PII Span F1 (Exact)** | 0.6000 | Exact character span match against synthetic labels |
| **PII Span F1 (IoU >= 0.5)** | 1.0000 | Token-level overlap detection |
| **PII Leakage Rate** | 0.00% | Zero synthetic sensitive strings leaked |
| **Utility Preservation** | 0.7500 | Preserved entities kept |
| **Ambiguity AUROC** | 1.0000 | Sanity verification on synthetic sample |
| **Expected Calibration Error (ECE)** | 0.5000 | Uncalibrated baseline probabilities |
| **Routing Accuracy** | 0.5000 | Default baseline routing check |

### Pipeline Takeaway
Verified that all metric calculation functions (Span F1, IoU overlap, leakage, utility, AUROC, ECE, and routing) execute end-to-end without bugs on Schema v0.2 data.


## 2026-09-27 · PII detector (Zarif, #10 #11)
- Code: src/pii/detector.py, masking.py, arbitrate.py; eval: scripts/eval_pii.py
- Hand-made set (34 cases, written with the detector): recall 1.00, leakage 0.0%. Regression check only.
- Hard held-out set (18 cases, not tuned on): recall 0.467, precision 0.875, leakage 46.7%, over-masking 0.0%.
- Misses: names without a keyword, free-form addresses, obfuscated e-mails, "double" in spoken numbers, OTP keyword after the number.
- Synthetic data only; real measurement needs the gold set.

---

## [PIPELINE SANITY CHECK] Run 002: Day 2 Baselines & Routing Dry-Runs (DO NOT USE IN RESULTS TABLE)
- **Type**: Pipeline Sanity Check (Synthetic Dry-Run — Exclude from Paper Results)
- **Date**: 2026-09-27
- **Lead**: Sadat
- **Models**:
  - `src/baselines/translit_baseline.py` (Phonetic Transliteration-first Baseline)
  - `src/baselines/student_confidence.py` (Student Raw Softmax Confidence Baseline, RQ2 ablation)
  - `src/routing/decision.py` (Risk-Controlled Dynamic Routing Decision Engine)
- **Dataset**: `data/synthetic/sample_synthetic.jsonl`
- **Predictions**:
  - `experiments/baseline/sample_translit_predictions.jsonl`
  - `experiments/baseline/sample_student_conf_predictions.jsonl`

### Summary Metrics
| Metric | Translit Baseline | Student Conf Baseline | Notes / Interpretation |
|---|---|---|---|
| **PII Span F1 (Exact)** | 0.6000 | 0.6000 | Exact character span match against synthetic labels |
| **PII Span F1 (IoU >= 0.5)** | 1.0000 | 1.0000 | Token-level overlap detection |
| **PII Leakage Rate** | 0.00% | 0.00% | Zero synthetic sensitive strings leaked |
| **Utility Preservation** | 0.7500 | 0.7500 | Preserved entities kept |
| **Ambiguity AUROC** | 0.5000 | 0.0000 | Ambiguity discrimination (Single-conf assigns high conf uniformly) |
| **Channel Separation (CSI)** | 0.0000 | 0.0000 | Confirms single-channel models cannot decouple A from E |
| **Mean CER / WER** | 0.6768 / 0.8982 | 0.0000 / 0.0000 | Translit projects to Bengali script |

### Key Takeaways
1. **Transliteration-First Baseline**: Phonetic transliteration into Bengali script functions end-to-end and maps detected PII spans back to `clean_input`.
2. **Student Confidence Baseline (RQ2 ablation)**: Single confidence scores (`aleatoric == epistemic`) yield a cross-channel gap of 0.0000, confirming that without dual-channel decomposition, student models cannot separate data ambiguity from epistemic doubt.
3. **Routing Decision Engine**: Implements the 4-way decision space (`PROCEED`, `PROCEED_WITH_FLAGS`, `ASK_USER`, `ESCALATE`) with risk-controlled threshold tuning via conformal prediction grid search.

