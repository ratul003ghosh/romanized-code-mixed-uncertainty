
# Know Why You're Unsure: Distilling Decomposed, Typed Span Uncertainty into Small Models for Risk-Controlled Sanitization of Romanized Code-Mixed Prompts

## 1. Problem
Romanized code-mixed prompts (like Banglish and Hinglish) contain severe spelling variations, dialect differences, and ambiguous context, making PII detection and risk-controlled sanitization highly challenging.

## 2. Motivation
Traditional PII masking either over-redacts (reducing utility) or under-redacts (leaking privacy). Large models can detect nuance but are too expensive.

## 3. Core Idea
Distill the disagreement (uncertainty) of a teacher ensemble into a small 1.5B student model, predicting span-level typed uncertainty (aleatoric vs epistemic) to dynamically route uncertain cases.

## 4. Architecture
Input -> Preprocessing -> PII Masking -> Teacher Ensemble -> Uncertainty Estimation -> Silver Labels -> Student Model -> Routing

## 5. Research Questions
See `docs/research-questions.md`.

## 6. Dataset Strategy
Use existing datasets, augmented with synthetic examples for testing and teacher-generated silver data for student distillation.

## 7. PII Strategy
Rule-based + regex/NER for clear identifiers (phones, accounts), avoiding real PII in the repository.

## 8. Uncertainty Approach
Calculate aleatoric (data) and epistemic (model) uncertainty from teacher ensemble disagreement (entropy, JSD).

## 9. Routing
Threshold-based routing based on uncertainty scores: ASK_USER, ESCALATE, PROCEED, PROCEED_WITH_FLAGS.

## 10. Student Model
1.5B model (e.g., Qwen/Llama) fine-tuned with QLoRA to predict uncertainty without the full ensemble.

## 11. Baselines
Regex/NER, Transliteration, Direct Teacher, Student confidence.

## 12. Metrics
PII Leakage, Span F1, AUROC, Routing Accuracy, ECE.

## 13. Repository Structure
Standard src/docs/data/scripts layout.

## 14. Installation
```bash
pip install -r requirements.txt
```

15. Running the pipeline
    CPU-safe checks (tests + PII eval): python scripts/run_full_experiment.py
    GPU teacher run: python scripts/test_gpu.py, then see docs/faculty-run.md

## 16. GPU Instructions
See `docs/faculty-run.md`.

## 17. Current Status
See `docs/status.md`.

## 18. Limitations
See `docs/limitations.md`.

## 19. Team
- Ratul (Research Lead / Integration)
- Zarif (Teacher + Uncertainty Lead)
- Saber (Dataset + Student Lead)
- Himel (PII + Preprocessing)
- Sadat (Evaluation + Baselines)
- Mashrafi (Examples + Documentation)
