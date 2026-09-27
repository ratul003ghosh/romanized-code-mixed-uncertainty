"""
Master Evaluation Harness for the Romanized Code-Mixed Uncertainty Sanitization project.
Combines PII span metrics, leakage, over-masking, uncertainty calibration, and routing metrics.
"""

from typing import List, Dict, Any, Optional
import json
import os

from evaluation.pii_metrics import (
    evaluate_pii_spans,
    compute_pii_leakage,
    compute_overmasking_rate
)
from evaluation.uncertainty_metrics import (
    evaluate_span_uncertainty,
    evaluate_channel_separation
)
from evaluation.routing_metrics import (
    evaluate_routing_decisions,
    compute_risk_coverage
)
from evaluation.text_metrics import evaluate_text_normalization


class MasterEvaluator:
    """Master benchmark evaluator comparing predictions against gold standard records."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        gold_records: List[Dict[str, Any]],
        pred_records: List[Dict[str, Any]],
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation suite across all project dimensions.
        """
        results = {
            "model_name": model_name,
            "num_gold_samples": len(gold_records),
            "num_pred_samples": len(pred_records),
            "pii_span_metrics": evaluate_pii_spans(gold_records, pred_records, self.iou_threshold),
            "pii_leakage": compute_pii_leakage(gold_records, pred_records),
            "overmasking_utility": compute_overmasking_rate(gold_records, pred_records),
            "uncertainty_calibration": evaluate_span_uncertainty(gold_records, pred_records),
            "channel_separation": evaluate_channel_separation(gold_records, pred_records, self.iou_threshold),
            "routing_decision": evaluate_routing_decisions(gold_records, pred_records),
            "risk_coverage": compute_risk_coverage(gold_records, pred_records),
            "text_normalization": evaluate_text_normalization(gold_records, pred_records)
        }
        return results

    @staticmethod
    def format_markdown_report(eval_results: Dict[str, Any]) -> str:
        """Formats the evaluation results into a clean GitHub-flavored markdown report."""
        model = eval_results.get("model_name", "Model")
        span = eval_results.get("pii_span_metrics", {})
        exact = span.get("exact_match", {})
        partial = span.get("partial_match", {})
        leak = eval_results.get("pii_leakage", {})
        util = eval_results.get("overmasking_utility", {})
        unc = eval_results.get("uncertainty_calibration", {})
        cs = eval_results.get("channel_separation", {})
        rout = eval_results.get("routing_decision", {})
        rc = eval_results.get("risk_coverage", {})
        txt = eval_results.get("text_normalization", {})

        report = []
        report.append(f"# Evaluation Benchmark Report: {model}\n")
        report.append(f"- **Evaluated Samples**: {eval_results.get('num_pred_samples', 0)}")
        report.append(f"- **Total Gold PII Spans**: {leak.get('total_gold_pii', 0)}")
        report.append(f"- **Total Preserved Entities**: {util.get('total_preserved_entities', 0)}\n")

        report.append("## 1. PII Detection & Privacy Performance")
        report.append("| Metric | Precision | Recall | F1 Score | Notes |")
        report.append("|---|---|---|---|---|")
        report.append(f"| **PII Span (Exact)** | {exact.get('precision', 0):.4f} | {exact.get('recall', 0):.4f} | **{exact.get('f1', 0):.4f}** | Exact char boundaries |")
        report.append(f"| **PII Span (IoU >= 0.5)** | {partial.get('precision', 0):.4f} | {partial.get('recall', 0):.4f} | **{partial.get('f1', 0):.4f}** | Partial token overlap |")
        report.append(f"| **PII Leakage Rate** | - | - | **{leak.get('leakage_percentage', 0):.2f}%** | Lower is better (Privacy) |")
        report.append(f"| **Utility Preservation** | - | - | **{util.get('utility_preservation_rate', 0):.4f}** | Preserved entities kept |\n")

        report.append("## 2. Uncertainty & Calibration (RQ1 / RQ2)")
        auroc = unc.get("ambiguity_auroc")
        auroc_str = f"{auroc:.4f}" if isinstance(auroc, (float, int)) else "N/A"
        ece = unc.get("ambiguity_ece")
        ece_str = f"{ece:.4f}" if isinstance(ece, (float, int)) else "N/A"
        brier = unc.get("ambiguity_brier_score")
        brier_str = f"{brier:.4f}" if isinstance(brier, (float, int)) else "N/A"

        report.append("| Metric | Value | Meaning |")
        report.append("|---|---|---|")
        report.append(f"| **Ambiguity AUROC** | {auroc_str} | Discriminating human ambiguous spans |")
        report.append(f"| **Expected Calibration Error (ECE)** | {ece_str} | Calibration gap across confidence bins |")
        report.append(f"| **Brier Score** | {brier_str} | Mean squared uncertainty error |\n")

        # Channel Separation table (RQ1 / K5, Proposal §3.3)
        if cs:
            ale_ch = cs.get("aleatoric_channel", {})
            epi_ch = cs.get("epistemic_channel", {})
            csi = cs.get("channel_separation_index")
            csi_str = f"{csi:.4f}" if isinstance(csi, (float, int)) else "N/A"
            a_gap = ale_ch.get("cross_channel_gap")
            a_gap_str = f"{a_gap:.4f}" if isinstance(a_gap, (float, int)) else "N/A"
            e_gap = epi_ch.get("cross_channel_gap")
            e_gap_str = f"{e_gap:.4f}" if isinstance(e_gap, (float, int)) else "N/A"

            report.append("## 2.1 Channel Separation (RQ1 / K5, Proposal Section 3.3)")
            report.append("| Channel Metric | Primary AUROC | Cross-Channel AUROC | Gap (Primary - Cross) |")
            report.append("|---|---|---|---|")
            a_pri = ale_ch.get("auroc_aleatoric_on_ambiguity")
            a_cross = ale_ch.get("auroc_epistemic_cross")
            a_pri_str = f"{a_pri:.4f}" if isinstance(a_pri, (float, int)) else "N/A"
            a_cross_str = f"{a_cross:.4f}" if isinstance(a_cross, (float, int)) else "N/A"
            report.append(f"| **Aleatoric Channel (Human Ambiguity)** | {a_pri_str} | {a_cross_str} | **{a_gap_str}** |")

            e_pri = epi_ch.get("auroc_epistemic_on_errors")
            e_cross = epi_ch.get("auroc_aleatoric_cross")
            e_pri_str = f"{e_pri:.4f}" if isinstance(e_pri, (float, int)) else "N/A"
            e_cross_str = f"{e_cross:.4f}" if isinstance(e_cross, (float, int)) else "N/A"
            report.append(f"| **Epistemic Channel (Model Disagreement/Error)** | {e_pri_str} | {e_cross_str} | **{e_gap_str}** |")
            report.append(f"- **Channel Separation Index (CSI)**: **{csi_str}** (Higher gap confirms decoupled channels)\n")

        report.append("## 3. Dynamic Routing & Risk Control (RQ3)")
        acc = rout.get("accuracy")
        acc_str = f"{acc:.4f}" if isinstance(acc, (float, int)) else "N/A"
        macro_f1 = rout.get("macro_f1")
        f1_str = f"{macro_f1:.4f}" if isinstance(macro_f1, (float, int)) else "N/A"
        cov = rc.get("coverage_rate", 0)
        c_leak = rc.get("covered_leakage_rate", 0)

        report.append("| Metric | Value | Meaning |")
        report.append("|---|---|---|")
        report.append(f"| **Routing Accuracy** | {acc_str} | Agreement with gold routing action |")
        report.append(f"| **Routing Macro F1** | {f1_str} | 4-way balanced decision quality |")
        report.append(f"| **Automatic Coverage** | {cov:.2%} | Prompts routed without escalation |")
        report.append(f"| **Residual Leakage on Covered** | {c_leak:.2%} | Privacy risk on automated outputs |\n")

        if "mean_cer" in txt:
            report.append("## 4. Text Normalization Quality")
            report.append(f"- **Mean CER (Character Error Rate)**: {txt.get('mean_cer'):.4f}")
            report.append(f"- **Mean WER (Word Error Rate)**: {txt.get('mean_wer'):.4f}\n")

        return "\n".join(report)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Loads a list of JSON records from a JSON or JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        if content.startswith("["):
            return json.loads(content)
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(data: Any, path: str):
    """Saves data to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
