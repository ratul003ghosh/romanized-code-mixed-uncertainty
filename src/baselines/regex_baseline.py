"""
Regex & Rule-Based PII Baseline Detector for Romanized Code-Mixed (Banglish / Hinglish) Text.
Conforms to Schema v0.2.
"""

from typing import List, Dict, Any, Tuple, Optional
import re


class RegexPIIBaseline:
    """
    Rule-based and regex baseline that identifies common PII patterns in Romanized prompts,
    replaces them with typed placeholders (<TYPE_N>), and determines routing.
    """

    def __init__(self):
        # Compile targeted patterns
        # 1. Phone numbers (Bangladeshi: +8801..., 01..., with optional spaces/dashes)
        self.phone_pattern = re.compile(
            r"(?:\+?880[\s-]?)?0?1[3-9]\d{2}[\s-]?\d{3}[\s-]?\d{3}\b"
        )
        
        # 2. Email addresses
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        
        # 3. Credit / Debit Card numbers (16 digits with optional spaces or dashes)
        self.card_pattern = re.compile(
            r"\b(?:\d{4}[\s-]?){3}\d{4}\b"
        )
        
        # 4. Transaction IDs (bKash, Nagad, Rocket, general bank transactions)
        # e.g., TrxID 9X87K, Trans ID: BL908123A, etc.
        self.txn_context_pattern = re.compile(
            r"(?:trans(?:action)?[\s_.-]*(?:id)?|trx(?:id)?|txid)[\s:]*([A-Za-z0-9]{5,16})\b",
            re.IGNORECASE
        )
        
        # 5. OTP / Verification PINs
        self.otp_pattern = re.compile(
            r"(?:otp|pin|code|password)[\s:]*(\d{4,6})\b",
            re.IGNORECASE
        )

        # 6. National IDs (NID: 10, 13, or 17 digits) with context or standalone long digits
        self.nid_pattern = re.compile(
            r"(?:nid|smart[\s-]?card)[\s:]*(\d{10}|\d{13}|\d{17})\b",
            re.IGNORECASE
        )

        # 7. Generic ID / Account Numbers (10 to 16 digits)
        self.id_number_pattern = re.compile(
            r"\b\d{10,16}\b"
        )

        # Monetary amount pattern to avoid masking currency as account/ID
        self.amount_pattern = re.compile(
            r"\b\d+[\s-]*(?:tk|taka|bdt|rs|rupee|usd|\$|৳)\b|\b(?:tk|taka|bdt|rs|rupee|usd|\$|৳)[\s-]*\d+\b",
            re.IGNORECASE
        )

    def detect_spans(self, text: str) -> List[Dict[str, Any]]:
        """
        Detects PII spans in the input text. Returns non-overlapping spans sorted by start position.
        """
        detected: List[Dict[str, Any]] = []
        occupied_spans: List[Tuple[int, int]] = []

        # Find amounts first to avoid false-positive ID masking
        excluded_spans = []
        for m in self.amount_pattern.finditer(text):
            excluded_spans.append((m.start(), m.end()))

        def _is_overlapping(start: int, end: int, span_list: List[Tuple[int, int]]) -> bool:
            return any(max(start, s) < min(end, e) for s, e in span_list)

        # 1. Emails
        for m in self.email_pattern.finditer(text):
            s, e = m.start(), m.end()
            if not _is_overlapping(s, e, occupied_spans):
                detected.append({"type": "EMAIL", "text": m.group(), "start": s, "end": e})
                occupied_spans.append((s, e))

        # 2. Cards
        for m in self.card_pattern.finditer(text):
            s, e = m.start(), m.end()
            if not _is_overlapping(s, e, occupied_spans) and not _is_overlapping(s, e, excluded_spans):
                detected.append({"type": "CARD", "text": m.group(), "start": s, "end": e})
                occupied_spans.append((s, e))

        # 3. Explicit Transaction IDs (with context)
        for m in self.txn_context_pattern.finditer(text):
            val = m.group(1)
            # Find position of the transaction code itself
            val_start = m.start(1)
            val_end = m.end(1)
            if not _is_overlapping(val_start, val_end, occupied_spans):
                detected.append({"type": "TXN_ID", "text": val, "start": val_start, "end": val_end})
                occupied_spans.append((val_start, val_end))

        # 4. Explicit OTPs
        for m in self.otp_pattern.finditer(text):
            val_start = m.start(1)
            val_end = m.end(1)
            if not _is_overlapping(val_start, val_end, occupied_spans):
                detected.append({"type": "OTP", "text": m.group(1), "start": val_start, "end": val_end})
                occupied_spans.append((val_start, val_end))

        # 5. Explicit NIDs
        for m in self.nid_pattern.finditer(text):
            val_start = m.start(1)
            val_end = m.end(1)
            if not _is_overlapping(val_start, val_end, occupied_spans):
                detected.append({"type": "NID", "text": m.group(1), "start": val_start, "end": val_end})
                occupied_spans.append((val_start, val_end))

        # 6. Phones
        for m in self.phone_pattern.finditer(text):
            s, e = m.start(), m.end()
            if not _is_overlapping(s, e, occupied_spans) and not _is_overlapping(s, e, excluded_spans):
                detected.append({"type": "PHONE", "text": m.group(), "start": s, "end": e})
                occupied_spans.append((s, e))

        # 7. Generic ID / Account Numbers (remaining 10-16 digits not part of amount)
        for m in self.id_number_pattern.finditer(text):
            s, e = m.start(), m.end()
            if not _is_overlapping(s, e, occupied_spans) and not _is_overlapping(s, e, excluded_spans):
                detected.append({"type": "ID_NUMBER", "text": m.group(), "start": s, "end": e})
                occupied_spans.append((s, e))

        # Sort detected spans by start position
        detected.sort(key=lambda x: x["start"])
        return detected

    def sanitize(self, text: str, detected_pii: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Replaces detected PII spans with <TYPE_N> placeholders and numbers each type sequentially.
        """
        type_counters: Dict[str, int] = {}
        pii_with_placeholders = []
        
        # Assign placeholders
        for item in detected_pii:
            pii_type = item["type"]
            count = type_counters.get(pii_type, 0) + 1
            type_counters[pii_type] = count
            placeholder = f"<{pii_type}_{count}>"
            
            entry = dict(item)
            entry["placeholder"] = placeholder
            pii_with_placeholders.append(entry)

        # Build sanitized string by replacing spans backwards to preserve character indices
        sanitized = text
        for item in sorted(pii_with_placeholders, key=lambda x: x["start"], reverse=True):
            s = item["start"]
            e = item["end"]
            sanitized = sanitized[:s] + item["placeholder"] + sanitized[e:]

        return sanitized, pii_with_placeholders

    def decide_routing(self, detected_pii: List[Dict[str, Any]]) -> str:
        """
        Baseline threshold routing:
        - If no PII: PROCEED
        - If PII detected and successfully masked: PROCEED_WITH_FLAGS
        - If multiple ambiguous ID_NUMBERs detected: ESCALATE
        """
        if not detected_pii:
            return "PROCEED"
        
        id_numbers = [p for p in detected_pii if p["type"] == "ID_NUMBER"]
        if len(id_numbers) >= 2:
            return "ESCALATE"
            
        return "PROCEED_WITH_FLAGS"

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs baseline detection and sanitization on a single record conforming to Schema v0.2.
        """
        raw_text = record.get("clean_input") or record.get("text") or record.get("input") or ""
        detected_spans = self.detect_spans(raw_text)
        sanitized_prompt, pii_list = self.sanitize(raw_text, detected_spans)
        routing = self.decide_routing(pii_list)

        output = {
            "id": record.get("id", "UNKNOWN"),
            "schema_version": "0.2",
            "input": record.get("input", raw_text),
            "clean_input": raw_text,
            "normalized_text": None,
            "sanitized_prompt": sanitized_prompt,
            "pii": pii_list,
            "preserved_entities": [],
            "uncertainties": [],
            "routing": routing,
            "metadata": {
                "language": record.get("metadata", {}).get("language", "banglish"),
                "source": "regex_baseline",
                "label_source": "baseline"
            }
        }
        return output

    def predict_all(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Runs baseline prediction across a list of records."""
        return [self.predict_record(rec) for rec in records]

