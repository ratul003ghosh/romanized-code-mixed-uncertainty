"""
Unit tests for baseline models (Regex & Rule-based PII baseline).
"""

import unittest
import sys
import os

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines.regex_baseline import RegexPIIBaseline


class TestRegexBaseline(unittest.TestCase):

    def setUp(self):
        self.model = RegexPIIBaseline()

    def test_phone_detection(self):
        text = "Amar number 01712345678 and second num is +8801812345678"
        spans = self.model.detect_spans(text)
        types = [s["type"] for s in spans]
        self.assertEqual(types, ["PHONE", "PHONE"])
        self.assertEqual(spans[0]["text"], "01712345678")
        self.assertEqual(spans[1]["text"], "+8801812345678")

    def test_transaction_id_detection(self):
        text = "kalk3 bkash e trans id 9X87K taka pay nai"
        spans = self.model.detect_spans(text)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["type"], "TXN_ID")
        self.assertEqual(spans[0]["text"], "9X87K")

    def test_amount_exclusion(self):
        # 500 tk or 5000 BDT should NOT be detected as an ID_NUMBER
        text = "amar 500 tk send korsi, balance 5000 bdt"
        spans = self.model.detect_spans(text)
        self.assertEqual(len(spans), 0)

    def test_sanitization_placeholders(self):
        text = "Trans id 9X87K pathaisi 01712345678 eta te"
        spans = self.model.detect_spans(text)
        sanitized, pii_with_placeholders = self.model.sanitize(text, spans)
        
        self.assertIn("<TXN_ID_1>", sanitized)
        self.assertIn("<PHONE_1>", sanitized)
        self.assertNotIn("9X87K", sanitized)
        self.assertNotIn("01712345678", sanitized)
        self.assertEqual(len(pii_with_placeholders), 2)

    def test_full_record_prediction(self):
        record = {
            "id": "BG_TEST_1",
            "clean_input": "kalk3 bkash e 5oo tk send korsi trans id 9X87K, 01712345678",
            "metadata": {"language": "banglish"}
        }
        pred = self.model.predict_record(record)
        self.assertEqual(pred["id"], "BG_TEST_1")
        self.assertEqual(pred["schema_version"], "0.2")
        self.assertIn("<TXN_ID_1>", pred["sanitized_prompt"])
        self.assertIn("<PHONE_1>", pred["sanitized_prompt"])
        self.assertEqual(pred["routing"], "PROCEED_WITH_FLAGS")


if __name__ == "__main__":
    unittest.main()

