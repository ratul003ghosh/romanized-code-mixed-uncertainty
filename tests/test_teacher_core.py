"""CPU unit tests: decomposition math and schema/offset mapping. Run: python -m pytest -q tests/test_teacher_core.py"""
import math

import torch

from src.uncertainty.agreement import discordance
from src.teachers.output_schema import extract_json, serialize_with_offsets, validate
from src.uncertainty.decomposition import decompose


def lp(rows):
    return torch.log(torch.tensor(rows, dtype=torch.float32)).unsqueeze(1)  # [K, T=1, V]


def test_identity_A_plus_E_equals_total():
    logp = torch.log_softmax(torch.randn(4, 7, 50), -1)
    st = decompose(logp, [0] * 7, top_k=5)
    assert torch.allclose(st["A"] + st["E"], st["H_total"], atol=1e-5)
    assert st["jsd_min_before_clamp"] > -1e-5


def test_identical_teachers_zero_epistemic():
    row = torch.log_softmax(torch.randn(1, 5, 30), -1)
    st = decompose(row.repeat(3, 1, 1), [1] * 5, top_k=5)
    assert float(st["E"].max()) < 1e-5


def test_confident_disagreement_is_epistemic():
    # two teachers each certain of a different token -> A ~ 0, E ~ ln 2
    st = decompose(lp([[1 - 2e-6, 1e-6, 1e-6], [1e-6, 1 - 2e-6, 1e-6]]), [0], top_k=2)
    assert float(st["A"][0]) < 1e-3 and abs(float(st["E"][0]) - math.log(2)) < 1e-3


def test_shared_uncertainty_is_aleatoric():
    # both teachers split 50/50 the same way -> A ~ ln 2, E ~ 0
    st = decompose(lp([[0.5, 0.5 - 1e-6, 1e-6], [0.5, 0.5 - 1e-6, 1e-6]]), [0], top_k=2)
    assert abs(float(st["A"][0]) - math.log(2)) < 1e-3 and float(st["E"][0]) < 1e-5


def test_entropy_weights_favor_confident_teacher():
    st = decompose(lp([[0.98, 0.01, 0.01], [0.34, 0.33, 0.33]]), [0], top_k=1, tau=0.5)
    assert st["kd_top_ids"][0, 0] == 0 and float(st["kd_top_probs"][0, 0]) > (0.98 + 0.34) / 2


def test_serialize_offsets_roundtrip():
    obj, _ = validate({"sanitized_prompt": 'Send "500" BDT to <PHONE_1>', "pii": [
        {"type": "phone", "span": "01700000000", "placeholder": "<PHONE_1>"}],
        "preserved_entities": [{"type": "AMOUNT", "value": "500 BDT"}], "ambiguous_spans": []})
    y, leaves = serialize_with_offsets(obj)
    assert extract_json(y) == obj
    lf = {tuple(l["path"]): l for l in leaves}
    assert y[lf[("pii", 0, "type")]["start"]:lf[("pii", 0, "type")]["end"]] == "PHONE"
    assert y[lf[("preserved_entities", 0, "value")]["start"]:lf[("preserved_entities", 0, "value")]["end"]] == "500 BDT"


def test_extract_json_with_noise_and_validation_errors():
    raw = 'Sure! ```json\n{"sanitized_prompt": "call 01700000000", "pii": [{"type": "PHONE", "span": "01700000000", "placeholder": "<PHONE_1>"}], "preserved_entities": [], "ambiguous_spans": [{"span": "x", "types": ["BOGUS"]}]}``` done'
    obj, errors = validate(extract_json(raw))
    assert obj is not None
    assert any("copied" in e for e in errors) and any("without valid type" in e for e in errors)
    assert validate(extract_json("no json here"))[0] is None


def test_discordance():
    a = {"sanitized_prompt": "x", "pii": [{"type": "NID", "span": "1987654321", "placeholder": "<NID_1>"}],
         "preserved_entities": [], "ambiguous_spans": []}
    b = {**a, "pii": [{"type": "ACCOUNT", "span": "1987654321", "placeholder": "<ACCOUNT_1>"}]}
    assert discordance("pii", a["pii"][0], [a, b]) == 0.5
