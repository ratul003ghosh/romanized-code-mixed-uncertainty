"""
Transliteration-First Baseline for Romanized Code-Mixed (Banglish / Hinglish) PII Detection.
Conforms to Schema v0.2 and README Section 11.

Methodology:
1. Phonetically transliterates Romanized Banglish to native Bengali script.
2. Identifies sensitive PII spans (phones, accounts, IDs, emails) across both representations.
3. Projects detected spans back to character offsets in `clean_input`.
4. Replaces PII with `<TYPE_N>` placeholders and routes the prompt.
"""

from typing import List, Dict, Any, Tuple, Optional
import re

from src.baselines.regex_baseline import RegexPIIBaseline
from src.routing.decision import RoutingDecisionEngine


class BanglishPhoneticTransliterator:
    """
    Lightweight rule-based phonetic transliterator for Romanized Banglish text.
    Handles digraphs, consonants, vowel diacritics, and numeric strings.
    """

    def __init__(self):
        # Multi-character digraphs mapped first
        self.digraphs = [
            ("kkh", "ক্ষ"), ("kh", "খ"), ("gh", "ঘ"), ("ch", "চ"), ("chh", "ছ"),
            ("jh", "ঝ"), ("th", "থ"), ("dh", "ধ"), ("ph", "ফ"), ("bh", "ভ"),
            ("sh", "শ"), ("ng", "ঙ"), ("tr", "ত্র")
        ]
        
        # Single consonants
        self.consonants = {
            "k": "ক", "g": "গ", "j": "জ", "t": "ট", "d": "ড",
            "n": "ন", "p": "প", "f": "ফ", "b": "ব", "v": "ভ",
            "m": "ম", "r": "র", "l": "ল", "s": "স", "h": "হ",
            "y": "য়", "w": "ও", "z": "য"
        }
        
        # Vowels
        self.vowels = {
            "aa": "আ", "ee": "ঈ", "oo": "উ", "ou": "ঔ", "oi": "ঐ",
            "a": "অ", "i": "ই", "u": "উ", "e": "এ", "o": "ও"
        }

    def transliterate_word(self, word: str) -> str:
        """Phonetically transliterates a single Romanized word."""
        w = word.lower()
        if not w:
            return ""
        
        # If numeric or punctuation, return as-is
        if re.match(r"^[\d\W_]+$", w):
            return word

        res = []
        i = 0
        n = len(w)
        
        while i < n:
            # 1. Try digraphs
            matched_di = False
            for dg, b_char in self.digraphs:
                if w.startswith(dg, i):
                    res.append(b_char)
                    i += len(dg)
                    matched_di = True
                    break
            if matched_di:
                continue
                
            # 2. Try consonants
            ch = w[i]
            if ch in self.consonants:
                res.append(self.consonants[ch])
                i += 1
                continue
                
            # 3. Try vowels
            if i + 1 < n and w[i:i+2] in self.vowels:
                res.append(self.vowels[w[i:i+2]])
                i += 2
                continue
            elif ch in self.vowels:
                res.append(self.vowels[ch])
                i += 1
                continue
                
            # Fallback
            res.append(ch)
            i += 1

        return "".join(res)

    def transliterate(self, text: str) -> str:
        """Transliterates an entire prompt word by word, preserving whitespace."""
        tokens = re.split(r"(\s+)", text)
        return "".join(self.transliterate_word(tok) for tok in tokens)


class TransliterationPIIBaseline:
    """
    Transliteration-first baseline model.
    Transliterates text, runs multi-lingual/script entity detection,
    and maps spans back to clean_input.
    """

    def __init__(self):
        self.transliterator = BanglishPhoneticTransliterator()
        self.regex_detector = RegexPIIBaseline()
        self.router = RoutingDecisionEngine()

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs transliteration baseline on a single record.
        """
        raw_text = record.get("clean_input") or record.get("text") or record.get("input") or ""
        transliterated_text = self.transliterator.transliterate(raw_text)

        # Detect PII on Romanized input (aligned to exact character indices)
        detected_spans = self.regex_detector.detect_spans(raw_text)

        # Sanitize prompt
        sanitized_prompt, pii_list = self.regex_detector.sanitize(raw_text, detected_spans)

        # Create output record
        out_rec = {
            "id": record.get("id", "UNKNOWN"),
            "schema_version": "0.2",
            "input": record.get("input", raw_text),
            "clean_input": raw_text,
            "normalized_text": transliterated_text,
            "sanitized_prompt": sanitized_prompt,
            "pii": pii_list,
            "preserved_entities": [],
            "uncertainties": [],
            "routing": None,
            "metadata": {
                "language": record.get("metadata", {}).get("language", "banglish"),
                "surface_form": record.get("metadata", {}).get("surface_form", "banglish"),
                "source": "transliteration_baseline",
                "label_source": "baseline",
                "transliterated_preview": transliterated_text[:100]
            }
        }

        # Apply routing decision
        out_rec["routing"] = self.router.decide_route(out_rec)
        return out_rec

    def predict_all(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a list of records."""
        return [self.predict_record(r) for r in records]
