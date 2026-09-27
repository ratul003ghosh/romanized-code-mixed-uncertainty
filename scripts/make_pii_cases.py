"""Write the hand-made synthetic PII test set data/synthetic/pii_cases.jsonl (schema v0.2).

All values are fake (dummy numbers, example.com, the public test card 4111 1111 1111 1111).
Offsets are computed from the text, so they are always exact. label_source = "synthetic":
use for testing the PII code only, never as a paper result.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing.clean import clean_text  # noqa: E402
from src.pii.masking import mask  # noqa: E402

# (text, [(pii value, type), ...], [(amount/entity that must NOT be masked, type), ...], regime)
CASES = [
    ("bkash e 5oo tk send korsi trx id 8N7A6D5E4F, taka ashe nai", [("8N7A6D5E4F", "TXN_ID")], [("5oo tk", "AMOUNT")], "canonical"),
    ("amar number 01712345678, call dio", [("01712345678", "PHONE")], [], "canonical"),
    ("nagad 01812-345678 e 1200 taka pathao", [("01812-345678", "PHONE")], [("1200 taka", "AMOUNT")], "perturbed"),
    ("amar phone O17l2345678 bondho", [("O17l2345678", "PHONE")], [], "perturbed"),
    ("amar mobile ০১৯১২৩৪৫৬৭৮ e sms dao", [("০১৯১২৩৪৫৬৭৮", "PHONE")], [], "perturbed"),
    ("amar number zero one seven one one two two three three four four", [("zero one seven one one two two three three four four", "PHONE")], [], "spoken"),
    ("number ta shunno ek sat ek dui tin char pach choy sat aat", [("shunno ek sat ek dui tin char pach choy sat aat", "PHONE")], [], "spoken"),
    ("whatsapp +8801612345678 e knock dio", [("+8801612345678", "PHONE")], [], "canonical"),
    ("amar nid 1990123456789 diye form fill koro", [("1990123456789", "NID")], [], "canonical"),
    ("NID no: 4512 3456 78 ta bhul ashe", [("4512 3456 78", "NID")], [], "perturbed"),
    ("jatiyo porichoy potro 19901234567890123 hariye geche", [("19901234567890123", "NID")], [], "canonical"),
    ("account no 123456789012 te 2000 taka joma dao", [("123456789012", "ACCOUNT")], [("2000 taka", "AMOUNT")], "canonical"),
    ("amar bank a/c 2050-1234-5678-9 block hoye geche", [("2050-1234-5678-9", "ACCOUNT")], [], "perturbed"),
    ("card 4111 1111 1111 1111 diye payment hocche na", [("4111 1111 1111 1111", "CARD")], [], "canonical"),
    ("4111111111111111 eta diye kinsi", [("4111111111111111", "CARD")], [], "canonical"),
    ("OTP 482913 ashche kintu kaj kore na", [("482913", "OTP")], [], "canonical"),
    ("pin code ta 7391 bolo na karo ke", [("7391", "OTP")], [], "canonical"),
    ("amar mail test.user@example.com e pathao", [("test.user@example.com", "EMAIL")], [], "canonical"),
    ("Rahim.Test99@example.co e cv dilam", [("Rahim.Test99@example.co", "EMAIL")], [], "canonical"),
    ("amar nam Karim Hossain, amar account e problem", [("Karim Hossain", "NAME")], [], "canonical"),
    ("amar naam rahim, rahim er bkash e pathao", [("rahim", "NAME"), ("rahim", "NAME")], [], "canonical"),
    ("Md. Jamal Uddin ke 500 tk dite hobe", [("Jamal Uddin", "NAME")], [("500 tk", "AMOUNT")], "canonical"),
    ("basa house 12, road 5, Dhanmondi te parcel pathao", [("house 12, road 5, Dhanmondi", "ADDRESS")], [], "canonical"),
    ("trans id 9X87K, taka pay nai. 1987654321 eta check koren", [("9X87K", "TXN_ID"), ("1987654321", "ID_NUMBER")], [], "canonical"),
    ("1987654321 number e call dite bolo", [("1987654321", "ID_NUMBER")], [], "canonical"),
    ("Rocket TxnID 12345678901 fail korse", [("12345678901", "TXN_ID")], [], "canonical"),
    ("amar 2 01712345678 number e call dao", [("01712345678", "PHONE")], [], "perturbed"),
    ("nid 1234567890e deya ache", [("1234567890", "NID")], [], "perturbed"),
    # negatives: nothing to mask
    ("5000 tk er moddhe ekta phone suggest koro", [], [("5000 tk", "AMOUNT")], "none"),
    ("27.09.2026 tarikh e 2026 er budget dekhbo", [], [], "none"),
    ("tk 1500 cash out charge koto?", [], [("tk 1500", "AMOUNT")], "none"),
    ("amar laptop onek slow, ki korbo?", [], [], "none"),
    ("what is the name of this app", [], [], "none"),
    ("12oo taka recharge korsi kintu balance ashe nai", [], [("12oo taka", "AMOUNT")], "none"),
]

# Harder cases written AFTER the detector, and not used to tune it. Report these numbers as the
# honest picture of the rule-based detector's weak spots.
HARD_CASES = [
    ("amar number 017 1234 5678 e call dio", [("017 1234 5678", "PHONE")], [], "perturbed"),
    ("01712 345678 ei number e bkash ache", [("01712 345678", "PHONE")], [], "perturbed"),
    ("rahim bhai ke bolo kal ashte", [("rahim", "NAME")], [], "canonical"),
    ("ami sumaiya, amar order ashe nai", [("sumaiya", "NAME")], [], "canonical"),
    ("mirpur 10, block c, road 7 er basay deliver korben", [("mirpur 10, block c, road 7", "ADDRESS")], [], "canonical"),
    ("mail: test dot user at example dot com", [("test dot user at example dot com", "EMAIL")], [], "perturbed"),
    ("verification e 5 8 2 9 1 4 dilam kaj hoy na", [("5 8 2 9 1 4", "OTP")], [], "perturbed"),
    ("4111-1111-1111-1111 card e taka kete nise", [("4111-1111-1111-1111", "CARD")], [], "perturbed"),
    ("smart card no 5512345678 update korte chai", [("5512345678", "NID")], [], "canonical"),
    ("trxid: bk7h2k9x1q ta dekhen", [("bk7h2k9x1q", "TXN_ID")], [], "perturbed"),
    ("zero one eight double seven one two three four five six e call den", [("zero one eight double seven one two three four five six", "PHONE")], [], "spoken"),
    ("1712345678 e bkash korsi", [("1712345678", "PHONE")], [], "perturbed"),
    ("amar acc 0012 3456 7890 te salary ashe", [("0012 3456 7890", "ACCOUNT")], [], "perturbed"),
    ("ei 553901 code ta kake dibo?", [("553901", "OTP")], [], "canonical"),
    ("Tanvir Ahmed er sathe kotha bolte chai", [("Tanvir Ahmed", "NAME")], [], "canonical"),
    ("price 12,500 tk, 2 ta nibo", [], [("12,500 tk", "AMOUNT")], "none"),
    ("model no RTX4060 er dam koto", [], [], "none"),
    ("1500 er jaygay 15000 kete nise", [], [], "none"),
]


def build(cases=CASES, prefix="BG_PIITEST"):
    out = []
    for i, (text, pii, keep, regime) in enumerate(cases, 1):
        text = clean_text(text)
        spans, pos = [], 0
        for value, t in pii:                      # search left to right so repeated values get both spans
            s = text.index(value, pos)
            spans.append({"type": t, "text": value, "start": s, "end": s + len(value)})
            pos = s + len(value)
        spans = mask(text, spans)[1]                 # adds schema placeholders (<TYPE_N>, same value = same N)
        entities = [{"type": t, "value": v, "start": text.index(v), "end": text.index(v) + len(v)} for v, t in keep]
        out.append({"id": f"{prefix}_{i:04d}", "schema_version": "0.2", "input": text, "clean_input": text,
                    "normalized_text": None, "sanitized_prompt": None, "pii": spans,
                    "preserved_entities": entities, "uncertainties": [], "routing": None,
                    "metadata": {"language": "banglish", "surface_form": "banglish", "source": "handmade_pii_cases",
                                 "label_source": "synthetic", "split": "none", "regime": regime}})
    return out


if __name__ == "__main__":
    for path, cases, prefix in (("data/synthetic/pii_cases.jsonl", CASES, "BG_PIITEST"),
                                ("data/synthetic/pii_cases_hard.jsonl", HARD_CASES, "BG_PIIHARD")):
        with open(path, "w", encoding="utf-8") as f:
            for r in build(cases, prefix):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(cases)} records to {path}")
