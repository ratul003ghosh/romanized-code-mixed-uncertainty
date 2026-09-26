"""
Evaluation package for the Romanized Code-Mixed Uncertainty Sanitization project.
"""

from evaluation.evaluator import MasterEvaluator, load_jsonl, save_json
from evaluation.pii_metrics import evaluate_pii_spans, compute_pii_leakage, compute_overmasking_rate
from evaluation.uncertainty_metrics import compute_roc_curve_and_auroc, compute_ece, compute_brier_score
from evaluation.routing_metrics import evaluate_routing_decisions, compute_risk_coverage
from evaluation.text_metrics import compute_cer, compute_wer, evaluate_text_normalization

__all__ = [
    "MasterEvaluator",
    "load_jsonl",
    "save_json",
    "evaluate_pii_spans",
    "compute_pii_leakage",
    "compute_overmasking_rate",
    "compute_roc_curve_and_auroc",
    "compute_ece",
    "compute_brier_score",
    "evaluate_routing_decisions",
    "compute_risk_coverage",
    "compute_cer",
    "compute_wer",
    "evaluate_text_normalization"
]

