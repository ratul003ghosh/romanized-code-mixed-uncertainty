"""PII detection + masking (issues #10, #11). All values are synthetic."""
import json
import re
from pathlib import Path

import pytest

from src.pii.detector import PII_TYPES, candidates, detect, digit_view, luhn_ok
from src.pii.masking import canonical, mask, sanitize

ROOT = Path(__file__).resolve().parents[1]


def types(text):
    return [(s["type"], s["text"]) for s in detect(text)]


def test_pii_types_match_schema_doc():
    doc = (ROOT / "docs" / "annotation-schema.md").read_text(encoding="utf-8")
    listed = re.search(r"\*\*PII types\*\*[^:]*:\s*([A-Z_, ]+)", doc).group(1)
    assert set(t.strip() for t in listed.split(",")) == set(PII_TYPES)


def test_digit_view_keeps_length_and_maps_lookalikes():
    for s in ["O17l2345678", "৫০০ tk", "5oo tk", "hello"]:
        assert len(digit_view(s)) == len(s)
    assert digit_view("O17l2345678") == "01712345678"
    assert digit_view("5oo tk") == "500 tk"
    assert digit_view("hello") == "hello"


def test_luhn():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")


@pytest.mark.parametrize("text,expected", [
    ("amar number 01712345678", [("PHONE", "01712345678")]),
    ("bkash 01812-345678 e", [("PHONE", "01812-345678")]),
    ("call +8801712345678", [("PHONE", "+8801712345678")]),
    ("amar phone O17l2345678", [("PHONE", "O17l2345678")]),
    ("amar phone ০১৭১২৩৪৫৬৭৮", [("PHONE", "০১৭১২৩৪৫৬৭৮")]),
    ("nid 1990123456789", [("NID", "1990123456789")]),
    ("account no 123456789012 te", [("ACCOUNT", "123456789012")]),
    ("card 4111 1111 1111 1111 diye", [("CARD", "4111 1111 1111 1111")]),
    ("OTP 482913 ashche", [("OTP", "482913")]),
    ("trx id 7GH2K91LX", [("TXN_ID", "7GH2K91LX")]),
    ("mail test.user@example.com e", [("EMAIL", "test.user@example.com")]),
    ("amar nam Karim Hossain, ami", [("NAME", "Karim Hossain")]),
    ("house 12, road 5, Dhanmondi te", [("ADDRESS", "house 12, road 5, Dhanmondi")]),
    ("1987654321 eta check koren", [("ID_NUMBER", "1987654321")]),
])
def test_detects(text, expected):
    assert types(text) == expected


@pytest.mark.parametrize("text", [
    "5oo tk send korsi", "tk 1500 cash out", "5000 taka pathao", "12oo taka recharge",
    "27.09.2026 tarikh", "2026 er budget", "what is the name of this app", "amar laptop slow",
])
def test_no_false_positives(text):
    assert detect(text) == []


def test_amount_next_to_pii_is_kept():
    assert types("nagad 01812345678 e 1200 taka") == [("PHONE", "01812345678")]


def test_short_leading_group_is_not_glued_to_phone():
    assert types("amar 2 01712345678 number") == [("PHONE", "01712345678")]


def test_spoken_numbers():
    assert types("amar number zero one seven one two three four five six seven eight")[0][0] == "PHONE"
    assert types("number ta shunno ek sat ek dui tin char pach choy sat aat")[0][0] == "PHONE"


def test_low_score_candidates_exist_but_are_not_masked_by_default():
    text = "ei 12345678 ta ki?"                       # bare 8 digits: unclear
    assert detect(text) == []
    low = [c for c in candidates(text) if c["type"] == "ID_NUMBER"]
    assert low and low[0]["score"] < 0.5
    assert detect(text, threshold=0.2)               # a calibrated (lower) threshold masks it


def test_every_span_has_score_and_valid_offsets():
    text = "amar nam Rahim, number 01712345678, nid 1990123456789, mail a.b@example.com"
    for s in detect(text):
        assert 0 < s["score"] <= 1 and text[s["start"]:s["end"]] == s["text"]


def test_mask_placeholders_follow_schema():
    out = sanitize("call 01712345678 or 01812345678, again 017-1234-5678")
    assert out["masked"] == "call <PHONE_1> or <PHONE_2>, again <PHONE_1>"
    assert [p["placeholder"] for p in out["pii"]] == ["<PHONE_1>", "<PHONE_2>", "<PHONE_1>"]
    for p in out["pii"]:
        assert set(p) >= {"type", "placeholder", "text", "start", "end"}


def test_repeated_name_is_masked_everywhere():
    out = sanitize("amar naam rahim, rahim er bkash e pathao")
    assert out["masked"] == "amar naam <NAME_1>, <NAME_1> er bkash e pathao"


def test_canonical_same_phone():
    assert canonical("PHONE", "+880 1712-345678") == canonical("PHONE", "O17l2345678") == "01712345678"


def test_mask_roundtrip_offsets():
    text = "trans id 9X87K, taka pay nai. 1987654321 eta check koren"
    masked, pii = mask(text, detect(text))
    assert masked == "trans id <TXN_ID_1>, taka pay nai. <ID_NUMBER_1> eta check koren"
    assert all(text[p["start"]:p["end"]] == p["text"] for p in pii)


def test_synthetic_set_regression():
    """Regression guard on the hand-made set. NOT a result: the detector was written alongside it."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_pii import evaluate
    recs = [json.loads(l) for l in (ROOT / "data/synthetic/pii_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    summary, _ = evaluate(recs)
    assert summary["leakage_rate"] == 0.0 and summary["over_masking_rate"] == 0.0
