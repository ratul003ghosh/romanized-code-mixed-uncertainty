import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pii import generators as G  # noqa: E402
from src.pii import schema  # noqa: E402
from src.pii.detect import mask  # noqa: E402
from src.pii.inject import Segment, join, make_carrier_record, make_template_record  # noqa: E402
from src.preprocessing.clean import clean_text  # noqa: E402
from src.preprocessing.noise import noise_amount, noise_text  # noqa: E402

N = 400
SCHEMA_DOC = (ROOT / "docs" / "annotation-schema.md").read_text(encoding="utf-8")


def doc_list(label: str) -> set[str]:
    """Comma-separated values that follow a bold label in docs/annotation-schema.md."""
    m = re.search(r"\*\*" + re.escape(label) + r"\*\*[^\n]*?:\s*([A-Za-z_, ]+)\n", SCHEMA_DOC)
    if not m or not m.group(1).strip():
        m = re.search(r"\*\*" + re.escape(label) + r"\*\*[^\n]*\n([A-Za-z_, ]+)\n", SCHEMA_DOC)
    assert m, f"{label} not found in docs/annotation-schema.md"
    return {v.strip() for v in m.group(1).split(",") if v.strip()}


@pytest.fixture(scope="module")
def records():
    rng = random.Random(0)
    return [make_template_record(i, rng, seed=0) for i in range(N)]


@pytest.fixture(scope="module")
def carrier_records():
    rng = random.Random(1)
    carriers = [{"id": "c1", "text": "ajke office e onek kaj chilo, pore kotha boli", "source": "BanglaTLit"},
                {"id": "c2", "text": "net ashe na keno bujhtesi na...", "source": "BanglaTLit"},
                {"id": "c3", "text": "aaj mausam bahut achha hai yaar", "source": "COMI-LINGUA",
                 "language": "hinglish", "surface_form": "hinglish"}]
    return [make_carrier_record(i, carriers[i % 3], rng, seed=1, n_clauses=1 + i % 3) for i in range(150)]


# ---------------------------------------------------------------- agreement with schema v0.2
def test_schema_rules_are_read_from_the_team_document():
    """src/pii/schema.py holds no copy of the schema: every list must equal the document."""
    assert set(schema.PII_TYPES) == doc_list("PII types") and len(schema.PII_TYPES) == 10
    assert set(schema.SURFACE_FORMS) == doc_list("surface_form")
    assert set(schema.ROUTES) == doc_list("Routing")
    assert set(schema.UNCERTAINTY_TYPES) == doc_list("Uncertainty types")
    for value in schema.LABEL_SOURCES:
        assert f"| {value} |" in SCHEMA_DOC
    assert schema.SCHEMA_VERSION == "0.2"
    assert "id" in schema.TOP_LEVEL_KEYS and "metadata" in schema.TOP_LEVEL_KEYS and "span" not in schema.TOP_LEVEL_KEYS


def test_generators_cover_every_schema_type():
    assert set(G.GENERATORS) == set(schema.PII_TYPES)


def test_every_record_passes_schema_check(records, carrier_records):
    bad = [(r["id"], errs) for r in records + carrier_records if (errs := schema.check_record(r))]
    assert not bad, bad[:3]


def test_input_is_already_clean(records, carrier_records):
    for r in records + carrier_records:
        assert r["input"] == r["clean_input"] == clean_text(r["input"])


def test_offsets_match_text(records, carrier_records):
    for r in records + carrier_records:
        for p in r["pii"]:
            assert r["clean_input"][p["start"]:p["end"]] == p["text"], r["id"]


def test_ids_labels_and_nulls(records, carrier_records):
    for r in records:
        assert r["id"].startswith("BG_SYN_") and r["metadata"]["label_source"] == "synthetic"
    for r in carrier_records:
        assert r["metadata"]["label_source"] == "none"
        prefix = "HG_PII_" if r["metadata"]["language"] == "hinglish" else "BG_PII_"
        assert r["id"].startswith(prefix)
    for r in records + carrier_records:
        assert r["normalized_text"] is None and r["sanitized_prompt"] is None and r["routing"] is None
        assert r["uncertainties"] == []


def test_same_value_gets_same_placeholder():
    segs = [Segment("call "), Segment("01712345678", "pii", "PHONE", {"regime": "canonical"}),
            Segment(" ba "), Segment("01712345678", "pii", "PHONE", {"regime": "canonical"}),
            Segment(" or "), Segment("01898765432", "pii", "PHONE", {"regime": "canonical"})]
    _, pii, _, _ = join(segs, random.Random(0), 0.0)
    assert [p["placeholder"] for p in pii] == ["<PHONE_1>", "<PHONE_1>", "<PHONE_2>"]


def test_masking_gold_spans_gives_the_record_placeholders(records):
    for r in records:
        masked, pii = mask(r["clean_input"], r["pii"])
        assert [p["placeholder"] for p in pii] == [p["placeholder"] for p in r["pii"]]
        assert all(p["placeholder"] in masked for p in r["pii"])


def test_check_record_catches_mistakes(records):
    r = dict(records[0], pii=[dict(records[0]["pii"][0], type="ACCOUNT_NUMBER")])
    assert any("pii type" in e for e in schema.check_record(r))
    r = dict(records[0], pii=[dict(records[0]["pii"][0], start=0, end=1)])
    assert any("offsets" in e for e in schema.check_record(r))
    assert any("schema_version" in e for e in schema.check_record(dict(records[0], schema_version="1.0.0")))


# ---------------------------------------------------------------- generators
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


def test_all_regimes_and_types_occur(records):
    assert {p["regime"] for r in records for p in r["pii"]} == set(schema.REGIMES)
    assert {p["type"] for r in records for p in r["pii"]} == set(G.PII_TYPES)


def test_spoken_regime_uses_words_only(records):
    for r in records:
        for p in r["pii"]:
            if p["regime"] == "spoken":
                assert not any(c in G.BN_SCRIPT_DIGITS for c in p["text"])


def test_deterministic_by_seed():
    a = [make_template_record(i, random.Random(7), seed=7)["input"] for i in range(20)]
    b = [make_template_record(i, random.Random(7), seed=7)["input"] for i in range(20)]
    assert a == b


def test_id_number_gets_pii_boundary_hint(records):
    for r in records:
        if any(p["type"] == "ID_NUMBER" for p in r["pii"]):
            assert any(h["type"] == "PII_BOUNDARY" for h in r["metadata"]["hints"])


def test_carrier_text_kept_intact(carrier_records):
    for r in carrier_records:
        assert r["pii"]
        assert any(t in r["input"] for t in ("ajke office e onek kaj chilo, pore kotha boli",
                                              "net ashe na keno bujhtesi na...",
                                              "aaj mausam bahut achha hai yaar"))


# ---------------------------------------------------------------- noise
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
