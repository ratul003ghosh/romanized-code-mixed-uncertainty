"""
Unit tests for the risk-controlled dynamic routing engine.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.routing.decision import RoutingDecisionEngine


class TestRoutingDecision(unittest.TestCase):

    def setUp(self):
        self.engine = RoutingDecisionEngine(tau_epistemic=0.60, tau_aleatoric=0.60, flag_threshold=0.30)

    def test_escalate_on_high_epistemic(self):
        rec = {
            "id": "R_1",
            "uncertainties": [
                {"span": "random_token", "epistemic": 0.85, "aleatoric": 0.10}
            ],
            "pii": []
        }
        self.assertEqual(self.engine.decide_route(rec), "ESCALATE")

    def test_ask_user_on_high_aleatoric(self):
        rec = {
            "id": "R_2",
            "uncertainties": [
                {"span": "5oo tk", "epistemic": 0.20, "aleatoric": 0.75}
            ],
            "pii": []
        }
        self.assertEqual(self.engine.decide_route(rec), "ASK_USER")

    def test_proceed_with_flags_on_masked_pii(self):
        rec = {
            "id": "R_3",
            "uncertainties": [],
            "pii": [{"type": "PHONE", "text": "01712345678"}]
        }
        self.assertEqual(self.engine.decide_route(rec), "PROCEED_WITH_FLAGS")

    def test_proceed_on_clean_prompt(self):
        rec = {
            "id": "R_4",
            "uncertainties": [
                {"span": "meeting", "epistemic": 0.05, "aleatoric": 0.05}
            ],
            "pii": []
        }
        self.assertEqual(self.engine.decide_route(rec), "PROCEED")

    def test_batch_apply(self):
        records = [
            {"id": "R_1", "uncertainties": [{"epistemic": 0.80, "aleatoric": 0.1}], "pii": []},
            {"id": "R_2", "uncertainties": [], "pii": []}
        ]
        routed = self.engine.apply(records)
        self.assertEqual(routed[0]["routing"], "ESCALATE")
        self.assertEqual(routed[1]["routing"], "PROCEED")


if __name__ == "__main__":
    unittest.main()
