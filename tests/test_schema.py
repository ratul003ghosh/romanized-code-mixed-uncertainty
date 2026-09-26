import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.validate import load_validator, validate  # noqa: E402

INPUT = ("kalk3 bkash e 5oo tk send korsi trans id 9X87K, taka pay nai. "
         "1987654321 eta check koren")

# The worked example from section 2 of the proposal, with two v1.0.0 changes:
# pii items carry the original "span", and placeholders are <TYPE_n>.
GOOD = {
    "sanitized_prompt": ("Yesterday I sent 500 BDT via bKash (transaction ID <TXN_ID_1>), "
                         "but the money was not received. Please check <ID_NUMBER_1>."),
    "pii": [
        {"type": "TXN_ID", "placeholder": "<TXN_ID_1>", "span": "9X87K"},
        {"type": "ID_NUMBER", "placeholder": "<ID_NUMBER_1>", "span": "1987654321"},
    ],
    "preserved_entities": [
        {"type": "AMOUNT", "value": "500 BDT"},
        {"type": "SERVICE", "value": "bKash"},
    ],
    "uncertainties": [
        {"span": "5oo tk", "types": ["NUMERIC_AMBIGUITY"],
         "candidates": ["500 BDT", "5000 BDT"], "aleatoric": 0.21, "epistemic": 0.63},
        {"span": "1987654321", "types": ["PII_BOUNDARY"],
         "candidates": ["NID", "account number"], "aleatoric": 0.74, "epistemic": 0.12},
    ],
    "routing": "ESCALATE",
}


@pytest.fixture(scope="module")
def v():
    return load_validator()


def test_schema_itself_is_valid(v):
    assert v is not None


def test_proposal_example_passes(v):
    assert validate(GOOD, INPUT, v) == []


def broken(mutate):
    out = copy.deepcopy(GOOD)
    mutate(out)
    return out


@pytest.mark.parametrize("name,mutate", [
    ("missing routing", lambda o: o.pop("routing")),
    ("unknown routing", lambda o: o.update(routing="MAYBE")),
    ("extra top-level key", lambda o: o.update(notes="hi")),
    ("score above 1", lambda o: o["uncertainties"][0].update(aleatoric=1.3)),
    ("three decimals", lambda o: o["uncertainties"][0].update(epistemic=0.633)),
    ("unknown uncertainty type", lambda o: o["uncertainties"][0].update(types=["SPELLING"])),
    ("empty types", lambda o: o["uncertainties"][0].update(types=[])),
    ("duplicate types", lambda o: o["uncertainties"][0].update(types=["INTENT", "INTENT"])),
    ("placeholder/type mismatch", lambda o: o["pii"][0].update(type="OTP")),
    ("bad placeholder format", lambda o: o["pii"][0].update(placeholder="[TXN_ID_1]")),
    ("placeholder not in pii list",
     lambda o: o.update(sanitized_prompt=o["sanitized_prompt"] + " <PHONE_1>")),
    ("raw PII leaked", lambda o: o.update(
        sanitized_prompt=o["sanitized_prompt"].replace("<ID_NUMBER_1>", "1987654321 <ID_NUMBER_1>"))),
    ("span not in input", lambda o: o["uncertainties"][0].update(span="6oo tk")),
])
def test_broken_outputs_fail(v, name, mutate):
    assert validate(broken(mutate), INPUT, v), f"'{name}' should have failed"
