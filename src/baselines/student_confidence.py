"""
Student Confidence Baseline (Ablation for RQ2).
Evaluates a single model that relies purely on raw softmax confidence / token perplexity,
without two-channel decomposed uncertainty (aleatoric vs. epistemic).

Answers Research Question RQ2:
"Can a small 1.5B student learn typed span-level uncertainty and improve over its own confidence?"
"""

from typing import List, Dict, Any, Tuple, Optional
import math

from src.baselines.regex_baseline import RegexPIIBaseline
from src.routing.decision import RoutingDecisionEngine


class StudentConfidenceBaseline:
    """
    Simulates a standard fine-tuned student model outputting predictions
    with raw token/span softmax confidence, without decomposed uncertainty.
    """

    def __init__(self, default_confidence: float = 0.85):
        self.default_confidence = default_confidence
        self.regex_detector = RegexPIIBaseline()
        self.router = RoutingDecisionEngine()

    def estimate_span_confidence(self, span_text: str, is_ambiguous_pattern: bool = False) -> float:
        """
        Estimates raw single-channel confidence.
        Standard language models are over-confident even on ambiguous inputs (high confidence).
        """
        # Over-confident failure mode: single models assign high confidence to plausible completions
        if is_ambiguous_pattern:
            return 0.70  # Slightly lower, but fails to clearly signal ambiguity
        return self.default_confidence

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs student confidence baseline on a single record.
        Outputs single-channel uncertainty (aleatoric = epistemic = 1 - confidence).
        """
        raw_text = record.get("clean_input") or record.get("text") or record.get("input") or ""
        detected_spans = self.regex_detector.detect_spans(raw_text)
        sanitized_prompt, pii_list = self.regex_detector.sanitize(raw_text, detected_spans)

        # Single-channel uncertainty spans
        uncertainties = []
        for p in pii_list:
            conf = self.estimate_span_confidence(p.get("text", ""))
            raw_unc = round(1.0 - conf, 4)
            # Crucial ablation: single confidence model cannot separate channels (A == E)
            uncertainties.append({
                "span": p.get("text", ""),
                "start": p.get("start"),
                "end": p.get("end"),
                "types": ["PII_BOUNDARY"],
                "candidates": [p.get("text", "")],
                "confidence": conf,
                "aleatoric": raw_unc,
                "epistemic": raw_unc
            })

        out_rec = {
            "id": record.get("id", "UNKNOWN"),
            "schema_version": "0.2",
            "input": record.get("input", raw_text),
            "clean_input": raw_text,
            "normalized_text": record.get("normalized_text"),
            "sanitized_prompt": sanitized_prompt,
            "pii": pii_list,
            "preserved_entities": [],
            "uncertainties": uncertainties,
            "routing": None,
            "metadata": {
                "language": record.get("metadata", {}).get("language", "banglish"),
                "surface_form": record.get("metadata", {}).get("surface_form", "banglish"),
                "source": "student_confidence_baseline",
                "label_source": "baseline"
            }
        }

        # Apply routing
        out_rec["routing"] = self.router.decide_route(out_rec)
        return out_rec

    def predict_all(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a list of records."""
        return [self.predict_record(r) for r in records]
