# Experiment Log

This file tracks experiment runs, models, splits, and evaluation metrics across the project.

---

## Run 001: Rule-Based & Regex PII Baseline (Day 1 Sanity Benchmark)
- **Date**: 2026-09-26
- **Lead**: Sadat
- **Model**: `src/baselines/regex_baseline.py` (Regex & Rule-Based PII detector)
- **Dataset**: `data/synthetic/sample_gold.jsonl` (Banglish test prompts conforming to Schema v0.2)
- **Predictions**: `experiments/baseline/sample_regex_predictions.jsonl`
- **Full Report**: `experiments/baseline/sample_report.md`
- **Output JSON**: `experiments/baseline/sample_eval.json`

### Summary Results
| Metric | Value | Interpretation |
|---|---|---|
| **PII Span F1 (Exact)** | 0.6000 | Exact character span match against gold labels |
| **PII Span F1 (IoU >= 0.5)** | 1.0000 | 100% token-level overlap detection of PII spans |
| **PII Leakage Rate** | 0.00% | Zero ground-truth sensitive strings leaked into sanitized prompt |
| **Utility Preservation** | 0.7500 | 75% of preserved entities kept (demonstrates standard over-masking) |
| **Ambiguity AUROC** | 1.0000 | Baseline ambiguity discrimination on synthetic sample |
| **Expected Calibration Error (ECE)** | 0.5000 | Uncalibrated baseline probabilities |
| **Routing Accuracy** | 0.5000 | Regex defaults to `PROCEED_WITH_FLAGS`; misses `ASK_USER` ambiguity |

### Key Takeaway for RQ1–RQ3
Traditional regex masking is effective at finding explicit phone numbers and transaction patterns (0% leakage), but lacks context to detect numeric/dialect ambiguity (e.g. "5oo tk") or calibrate uncertainty, failing to route to `ASK_USER` (only 50% routing accuracy). This establishes the exact baseline the Student Model needs to surpass.

