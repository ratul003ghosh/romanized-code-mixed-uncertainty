# Evaluation Benchmark Report: Regex Baseline

- **Evaluated Samples**: 4
- **Total Gold PII Spans**: 5
- **Total Preserved Entities**: 4

## 1. PII Detection & Privacy Performance
| Metric | Precision | Recall | F1 Score | Notes |
|---|---|---|---|---|
| **PII Span (Exact)** | 0.6000 | 0.6000 | **0.6000** | Exact char boundaries |
| **PII Span (IoU >= 0.5)** | 1.0000 | 1.0000 | **1.0000** | Partial token overlap |
| **PII Leakage Rate** | - | - | **0.00%** | Lower is better (Privacy) |
| **Utility Preservation** | - | - | **0.7500** | Preserved entities kept |

## 2. Uncertainty & Calibration (RQ1 / RQ2)
| Metric | Value | Meaning |
|---|---|---|
| **Ambiguity AUROC** | 0.5000 | Discriminating human ambiguous spans |
| **Expected Calibration Error (ECE)** | 0.5000 | Calibration gap across confidence bins |
| **Brier Score** | 0.5000 | Mean squared uncertainty error |

## 2.1 Channel Separation (RQ1 / K5, Proposal Section 3.3)
| Channel Metric | Primary AUROC | Cross-Channel AUROC | Gap (Primary - Cross) |
|---|---|---|---|
| **Aleatoric Channel (Human Ambiguity)** | 0.5000 | 0.5000 | **0.0000** |
| **Epistemic Channel (Model Disagreement/Error)** | nan | nan | **N/A** |
- **Channel Separation Index (CSI)**: **0.0000** (Higher gap confirms decoupled channels)

## 3. Dynamic Routing & Risk Control (RQ3)
| Metric | Value | Meaning |
|---|---|---|
| **Routing Accuracy** | 0.5000 | Agreement with gold routing action |
| **Routing Macro F1** | 0.3750 | 4-way balanced decision quality |
| **Automatic Coverage** | 100.00% | Prompts routed without escalation |
| **Residual Leakage on Covered** | 0.00% | Privacy risk on automated outputs |

## 4. Text Normalization Quality
- **Mean CER (Character Error Rate)**: 1.0000
- **Mean WER (Word Error Rate)**: 1.0000
