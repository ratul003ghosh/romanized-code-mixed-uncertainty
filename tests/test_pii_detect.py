import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_pii_detection import evaluate  # noqa: E402
from src.pii.detect import detect, mask  # noqa: E402
from src.pii.inject import make_carrier_record, make_template_record  # noqa: E402


def types(text):
    return [(s["type"], s["text"]) for s in detect(text)]


@pytest.mark.parametrize("text, expected", [
    ("ami kal 01712345678 e 500 tk bkash korbo", [("PHONE", "01712345678")]),
    ("amar NID 1234567890, account e taka ashe nai", [("NID", "1234567890")]),
    ("account no 8234152879109 er balance", [("ACCOUNT", "8234152879109")]),
    ("card 4539 1488 0343 6467 theke kete nise", [("CARD", "4539 1488 0343 6467")]),
    ("OTP 4821 ashche kintu kaj korche na", [("OTP", "4821")]),
    ("trans id 8N7A6D5E4F, ekhono pay nai", [("TXN_ID", "8N7A6D5E4F")]),
    ("mail korsi rahim.khan77@gmail.com theke", [("EMAIL", "rahim.khan77@gmail.com")]),
    ("amar naam Rafi Khan, lock hoye gese", [("NAME", "Rafi Khan")]),
    ("parcel house 12, road 5, Mirpur, Dhaka e deliver", [("ADDRESS", "house 12, road 5, Mirpur, Dhaka")]),
])
def test_each_pii_type(text, expected):
    assert types(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("number shunno ek shat ek dui tin char pach choy shat aat", ("PHONE", "shunno ek shat ek dui tin char pach choy shat aat")),
    ("call zero one seven one two three four five six seven eight", ("PHONE", "zero one seven one two three four five six seven eight")),
    ("call dio ০১৭১২৩৪৫৬৭৮ e", ("PHONE", "০১৭১২৩৪৫৬৭৮")),
    ("O17l2345678 ei number", ("PHONE", "O17l2345678")),
    ("017-123-45678 number e", ("PHONE", "017-123-45678")),
    ("01712345678e 300 tk pathaisi", ("PHONE", "01712345678")),
    ("TxrID VR7YBHVW, ekhono", ("TXN_ID", "VR7YBHVW")),
    ("TrxID ১YP৭A৯CMUC, ami", ("TXN_ID", "১YP৭A৯CMUC")),
])
def test_spoken_and_perturbed_formats(text, expected):
    assert expected in types(text)


@pytest.mark.parametrize("text", [
    "bhai 5oo taka recharge dilam kintu ashe nai",
    "20000tk. pathaisi kal",
    "ajke onek gorom porse, bari jabo",
    "tk 1500 dite hobe",
])
def test_amounts_and_plain_text_are_not_pii(text):
    assert detect(text) == []


def test_proposal_example_is_masked_like_the_proposal():
    text = "kalk3 bkash e 5oo tk send korsi trans id 9X87K, taka pay nai. 1987654321 eta check koren"
    masked, pii = mask(text, detect(text))
    assert masked == ("kalk3 bkash e 5oo tk send korsi trans id <TXN_ID_1>, taka pay nai. "
                      "<ID_NUMBER_1> eta check koren")
    assert all(text[p["start"]:p["end"]] == p["text"] for p in pii)


def test_same_value_same_placeholder():
    text = "01712345678 e call den, na hole 01712345678 e sms, ba 01898765432"
    masked, _ = mask(text, detect(text))
    assert masked == "<PHONE_1> e call den, na hole <PHONE_1> e sms, ba <PHONE_2>"


def test_glued_name_is_masked_without_leaking():
    text = "amar naam Fatema Aktere, lock"
    (span,) = [s for s in detect(text) if s["type"] == "NAME"]
    assert "Fatema Akter" in span["text"]                  # may cover the glued letter, never less


def test_leakage_on_generated_records_stays_low():
    """Sanity check on our own synthetic data (optimistic; the real test is the gold set)."""
    rng = random.Random(3)
    records = [make_template_record(i, rng, seed=3) for i in range(300)]
    carrier = {"id": "c", "text": "ajke office e onek kaj chilo, pore kotha boli", "source": "BanglaTLit"}
    records += [make_carrier_record(i, carrier, rng, seed=3) for i in range(200)]
    by, _, _ = evaluate(records)
    overall = by[("all", "all")]
    assert overall["leaked"] / overall["n"] < 0.02
    for regime in ("canonical", "spoken"):
        s = by[("regime", regime)]
        assert s["found"] / s["n"] > 0.95
