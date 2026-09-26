"""
Unit tests for the evaluation metrics suite.
Tests span F1, leakage rate, over-masking rate, AUROC, ECE, and routing metrics.
"""

import unittest
import sys
import os

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.pii_metrics import evaluate_pii_spans, compute_pii_leakage, compute_overmasking_rate
from evaluation.uncertainty_metrics import compute_roc_curve_and_auroc, compute_ece, compute_brier_score
from evaluation.routing_metrics import evaluate_routing_decisions, compute_risk_coverage
from evaluation.text_metrics import compute_cer, compute_wer
from evaluation.evaluator import MasterEvaluator


class TestEvaluationMetrics(unittest.TestCase):

    def setUp(self):
        self.gold_records = [
            {
                "id": "BG_001",
                "clean_input": "kalk3 bkash e 5oo tk send korsi trans id 9X87K. 01712345678 check koren",
                "pii": [
                    {"type": "TXN_ID", "text": "9X87K", "start": 41, "end": 46},
                    {"type": "PHONE", "text": "01712345678", "start": 48, "end": 59}
                ],
                "preserved_entities": [
                    {"type": "AMOUNT", "value": "500 BDT"},
                    {"type": "SERVICE", "value": "bkash"}
                ],
                "uncertainties": [
                    {"span": "5oo tk", "aleatoric": 0.8, "epistemic": 0.2, "human_ambiguous": True},
                    {"span": "bkash", "aleatoric": 0.1, "epistemic": 0.1, "human_ambiguous": False}
                ],
                "routing": "ESCALATE",
                "normalized_text": "kal bkash e 500 taka send koresi transaction id 9X87K"
            }
        ]

    def test_pii_exact_and_partial_span_matching(self):
        # Perfect prediction
        pred_perfect = [
            {
                "id": "BG_001",
                "pii": [
                    {"type": "TXN_ID", "text": "9X87K", "start": 41, "end": 46},
                    {"type": "PHONE", "text": "01712345678", "start": 48, "end": 59}
                ]
            }
        ]
        res = evaluate_pii_spans(self.gold_records, pred_perfect)
        self.assertEqual(res["exact_match"]["f1"], 1.0)
        self.assertEqual(res["exact_match"]["tp"], 2)
        self.assertEqual(res["exact_match"]["fp"], 0)
        self.assertEqual(res["exact_match"]["fn"], 0)

        # Partial span prediction (offset by 1 character)
        pred_partial = [
            {
                "id": "BG_001",
                "pii": [
                    {"type": "TXN_ID", "text": "9X87", "start": 41, "end": 45}  # partial match
                ]
            }
        ]
        res_partial = evaluate_pii_spans(self.gold_records, pred_partial, iou_threshold=0.5)
        self.assertEqual(res_partial["exact_match"]["tp"], 0)
        self.assertEqual(res_partial["partial_match"]["tp"], 1)

    def test_pii_leakage(self):
        # Case 1: Phone leaked into sanitized prompt
        pred_leaked = [
            {
                "id": "BG_001",
                "sanitized_prompt": "Send money trans id <TXN_ID_1>. 01712345678 check koren"
            }
        ]
        leak_res = compute_pii_leakage(self.gold_records, pred_leaked)
        self.assertEqual(leak_res["total_gold_pii"], 2)
        self.assertEqual(leak_res["leaked_pii_count"], 1)  # Phone leaked!
        self.assertAlmostEqual(leak_res["leakage_rate"], 0.5)

        # Case 2: Clean sanitized prompt, no leaks
        pred_clean = [
            {
                "id": "BG_001",
                "sanitized_prompt": "Send money trans id <TXN_ID_1>. <PHONE_1> check koren"
            }
        ]
        clean_res = compute_pii_leakage(self.gold_records, pred_clean)
        self.assertEqual(clean_res["leaked_pii_count"], 0)
        self.assertEqual(clean_res["leakage_rate"], 0.0)

    def test_overmasking_utility(self):
        # Service "bkash" preserved in text
        pred_preserved = [
            {
                "id": "BG_001",
                "sanitized_prompt": "send money via bkash with 500 BDT",
                "pii": []
            }
        ]
        res = compute_overmasking_rate(self.gold_records, pred_preserved)
        self.assertEqual(res["erroneously_masked_count"], 0)
        self.assertEqual(res["utility_preservation_rate"], 1.0)

        # Service "bkash" masked as PII
        pred_masked = [
            {
                "id": "BG_001",
                "sanitized_prompt": "send money via <NAME_1> with 500 BDT",
                "pii": [{"type": "NAME", "text": "bkash"}]
            }
        ]
        res_masked = compute_overmasking_rate(self.gold_records, pred_masked)
        self.assertEqual(res_masked["erroneously_masked_count"], 1)
        self.assertAlmostEqual(res_masked["utility_preservation_rate"], 0.5)

    def test_auroc_and_ece(self):
        # Perfect discrimination test
        y_true = [1, 1, 0, 0]
        y_scores = [0.9, 0.8, 0.2, 0.1]
        roc_res = compute_roc_curve_and_auroc(y_true, y_scores)
        self.assertEqual(roc_res["auroc"], 1.0)

        # ECE test with perfect calibration
        y_true_ece = [1, 1, 0, 0]
        y_probs_ece = [1.0, 1.0, 0.0, 0.0]
        ece_res = compute_ece(y_true_ece, y_probs_ece, n_bins=10)
        self.assertAlmostEqual(ece_res["ece"], 0.0)

    def test_routing_metrics(self):
        pred_routing = [
            {
                "id": "BG_001",
                "routing": "ESCALATE"
            }
        ]
        rout_res = evaluate_routing_decisions(self.gold_records, pred_routing)
        self.assertEqual(rout_res["accuracy"], 1.0)
        self.assertEqual(rout_res["per_class"]["ESCALATE"]["f1"], 1.0)

    def test_text_distance_metrics(self):
        cer = compute_cer("bkash", "bkash")
        self.assertEqual(cer, 0.0)
        cer_diff = compute_cer("bkash", "bkast")
        self.assertAlmostEqual(cer_diff, 0.2)

        wer = compute_wer("amar taka pay nai", "amar taka pay nai")
        self.assertEqual(wer, 0.0)
        wer_diff = compute_wer("amar taka pay nai", "amar taka paini")
        self.assertAlmostEqual(wer_diff, 0.5)

    def test_master_evaluator_end_to_end(self):
        evaluator = MasterEvaluator()
        pred_records = [
            {
                "id": "BG_001",
                "sanitized_prompt": "Send money trans id <TXN_ID_1>. <PHONE_1> check koren",
                "pii": [
                    {"type": "TXN_ID", "text": "9X87K", "start": 41, "end": 46},
                    {"type": "PHONE", "text": "01712345678", "start": 48, "end": 59}
                ],
                "uncertainties": [
                    {"span": "5oo tk", "aleatoric": 0.85, "epistemic": 0.15},
                    {"span": "bkash", "aleatoric": 0.05, "epistemic": 0.05}
                ],
                "routing": "ESCALATE",
                "normalized_text": "kal bkash e 500 taka send koresi transaction id 9X87K"
            }
        ]
        results = evaluator.evaluate(self.gold_records, pred_records, model_name="TestModel")
        report = evaluator.format_markdown_report(results)
        self.assertIn("TestModel", report)
        self.assertIn("PII Span (Exact)", report)
        self.assertIn("Ambiguity AUROC", report)
        self.assertIn("Routing Accuracy", report)


if __name__ == "__main__":
    unittest.main()

