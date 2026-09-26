import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.validate import load_validator, validate  # noqa: E402
from src.noise.noise import noise_amount, noise_text  # noqa: E402
from src.pii_inject import generators as G  # noqa: E402
from src.pii_inject.build import (make_carrier_record, make_template_record,  # noqa: E402
                                  pii_reference)

N = 400


@pytest.fixture(scope="module")
def records():
    rng = random.Random(0)
    return [make_template_record(i, rng, seed=0) for i in range(N)]


@pytest.fixture(scope="module")
def v():
    return load_validator()


def test_phone_format():
    rng = random.Random(1)
    for _ in range(200):
        assert re.fullmatch(r"01[3-9]\d{8}", G.gen_phone(rng))


def test_nid_lengths():
    rng = random.Random(2)
    assert {len(G.gen_nid(rng)) for _ in range(300)} == {10, 13, 17}


def test_card_numbers_are_luhn_valid():
    rng = random.Random(3)
    for _ in range(300):
        c = G.gen_card_number(rng)
        assert len(c) == 16 and G.luhn_valid(c)
    assert not G.luhn_valid("4539148803436468")


def test_offsets_match_spans(records):
    for r in records:
        for p in r["pii"]:
            assert r["input"][p["start"]:p["end"]] == p["span"], r["id"]
        for e in r["preserved_entities"]:
            assert r["input"][e["start"]:e["end"]] == e["value"], r["id"]


def test_all_regimes_and_types_occur(records):
    regimes = {p["regime"] for r in records for p in r["pii"]}
    types = {p["type"] for r in records for p in r["pii"]}
    assert regimes == set(G.REGIMES)
    assert types == set(G.PII_TYPES)


def test_types_match_schema(v):
    schema_types = set(v.schema["$defs"]["pii_type"]["enum"])
    assert set(G.PII_TYPES) == schema_types


def test_reference_outputs_are_schema_valid(records, v):
    bad = [(r["id"], errs) for r in records if (errs := validate(pii_reference(r), r["input"], v))]
    assert not bad, bad[:3]


def test_deterministic_by_seed():
    a = [make_template_record(i, random.Random(7), seed=7)["input"] for i in range(20)]
    b = [make_template_record(i, random.Random(7), seed=7)["input"] for i in range(20)]
    assert a == b


def test_id_number_gets_pii_boundary_hint(records):
    for r in records:
        if any(p["type"] == "ID_NUMBER" for p in r["pii"]):
            assert any(h["type"] == "PII_BOUNDARY" for h in r["hints"])


def test_carrier_text_kept_intact(v):
    rng = random.Random(4)
    carrier = {"text": "ajke office e onek kaj chilo, pore kotha boli", "source": "demo", "id": 1}
    for i in range(50):
        r = make_carrier_record(i, carrier, rng, seed=4)
        assert carrier["text"] in r["input"]
        assert r["pii"] and not validate(pii_reference(r), r["input"], v)


def test_spoken_regime_uses_words_only(records):
    for r in records:
        for p in r["pii"]:
            if p["regime"] == "spoken":
                assert not any(c in G.BN_SCRIPT_DIGITS for c in p["span"])


def test_noise_changes_text_but_not_length_wildly():
    rng = random.Random(5)
    text = "ami kal bkash e taka pathaisi kintu receiver bolche ashe nai"
    noisy, ops = noise_text(text, rng, p_word=1.0)
    assert noisy != text and ops
    assert abs(len(noisy) - len(text)) <= len(text.split())


def test_amount_noise():
    assert noise_amount("500", random.Random(0)) in {"5oo", "5OO"}
    assert noise_amount("20000", random.Random(1)) in {"2oooo", "2OOOO", "200oo", "200OO"}
    assert noise_amount("7", random.Random(0)) is None
