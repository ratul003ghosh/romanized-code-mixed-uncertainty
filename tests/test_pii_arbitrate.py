"""Rule-1 check of model output (proposal §4.6). All values are synthetic."""
from src.pii.arbitrate import enforce
from src.pii.masking import sanitize

SRC = "amar number 01712345678, trx id 9X87KQ2M, mail test.user@example.com, 5oo tk pathaisi"


def source():
    return sanitize(SRC)["pii"]


def test_clean_output_unchanged():
    out = "My number is <PHONE_1>, transaction ID <TXN_ID_1>, email <EMAIL_1>. I sent 500 BDT."
    assert enforce(out, source()) == {"text": out, "leaked": False, "replacements": []}


def test_leaked_values_get_their_input_placeholder():
    res = enforce("Call 01712345678 about transaction 9X87KQ2M.", source())
    assert res["leaked"] and res["text"] == "Call <PHONE_1> about transaction <TXN_ID_1>."
    assert {r["reason"] for r in res["replacements"]} == {"source_value"}


def test_reformatted_values_are_caught():
    for leaked in ("017-1234-5678", "O1712345678", "০১৭১২৩৪৫৬৭৮", "+880 1712345678", "+8801712345678"):
        assert enforce(f"Please call {leaked} today.", source())["text"] == "Please call <PHONE_1> today.", leaked


def test_email_case_insensitive():
    assert enforce("Write to Test.User@Example.com", source())["text"] == "Write to <EMAIL_1>"


def test_new_pii_in_output_gets_next_number():
    res = enforce("Number <PHONE_1> and also 01898765432.", source())
    assert res["text"] == "Number <PHONE_1> and also <PHONE_2>."
    assert res["replacements"][0]["reason"] == "detected"


def test_amounts_and_placeholders_left_alone():
    assert not enforce("Send 5000 BDT to <ACCOUNT_12> by 2026.", source())["leaked"]


def test_source_pii_not_modified():
    pii = [{"type": "PHONE", "text": "01712345678", "start": 0, "end": 11}]
    assert enforce("call 01712345678", pii)["text"] == "call <PHONE_1>" and "placeholder" not in pii[0]
