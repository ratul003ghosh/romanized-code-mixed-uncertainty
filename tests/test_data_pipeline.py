"""CPU tests for the dataset side: teacher-input builder, JSONL validator, teacher interface,
silver -> student conversion and the student prediction record. No downloads, no GPU."""
import json
import os
import subprocess
import sys

from src.utils.jsonl_validate import validate_file
from scripts.build_student_data import check_record, make_target
from scripts.infer_student import to_prediction

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = [
    {"id": "a", "split": "train", "text_transliterated": "oi mia click korle subscribe hobe na??"},
    {"id": "b", "split": "train", "text_transliterated": "amar number 01712345678 e call dio  ...."},
    {"id": "c", "split": "train", "text_transliterated": "Tnx bro ato sundor akta gift"},   # also in test
    {"id": "d", "split": "train", "text_transliterated": "   "},
    {"id": "e", "split": "train", "text_transliterated": None},
    {"id": "f", "split": "train", "text_transliterated": "x" * 1200},
    {"id": "g", "split": "train", "text_transliterated": "bad \ufffd char"},
    {"id": "h", "split": "train", "text_transliterated": "oi mia click korle subscribe hobe na??"},
    {"id": "i", "split": "validation", "text_transliterated": "trx id 8N7A6D5E4F, bkash e 500 tk"},
    {"id": "j", "split": "test", "text_transliterated": "Tnx bro ato sundor akta gift"},
]


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


def build(tmp_path):
    src = tmp_path / "raw.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in RAW), encoding="utf-8")
    out = tmp_path / "out"
    res = run("scripts/build_teacher_inputs.py", "--source-file", str(src), "--out-dir", str(out),
              "--train", "0", "--dev", "0", "--test", "0")
    assert res.returncode == 0, res.stderr
    return out, json.loads((out / "teacher_inputs_stats.json").read_text())


def test_builder_counts_and_filters(tmp_path):
    out, stats = build(tmp_path)
    assert stats["raw_read"] == {"test": 1, "dev": 1, "train": 8}
    assert stats["final"] == {"train": 2, "dev": 1, "test": 1}
    assert stats["dropped"] == {"duplicate_across_splits": 1, "empty_after_cleaning": 1,
                                "malformed_or_missing_text": 1, "too_long": 1, "broken_unicode": 1,
                                "duplicate_same_split": 1}
    assert stats["records_with_pii_detected"] == {"train": 1, "dev": 1}
    assert stats["masking_failures"] == 0
    assert (out / "teacher_inputs.jsonl").read_text() == (out / "teacher_inputs_train.jsonl").read_text()


def test_builder_output_passes_validator_and_teacher_interface(tmp_path):
    out, _ = build(tmp_path)
    for split in ("train", "dev", "test"):
        assert validate_file(str(out / f"teacher_inputs_{split}.jsonl"), "teacher_input").ok
    res = run("scripts/check_teacher_interface.py", str(out / "teacher_inputs.jsonl"))
    assert res.returncode == 0 and "RESULT: PASS" in res.stdout, res.stdout + res.stderr


def test_teacher_input_keeps_pii_unmasked(tmp_path):
    out, _ = build(tmp_path)
    rec = [json.loads(line) for line in open(out / "teacher_inputs_train.jsonl", encoding="utf-8")][1]
    assert "01712345678" in rec["clean_input"]                 # teachers must see the PII
    assert rec["metadata"]["pii_detected"][0]["type"] == "PHONE"


def test_validator_finds_every_problem_type(tmp_path):
    good = {"id": "A", "schema_version": "0.2", "input": "hello", "clean_input": "hello",
            "metadata": {"source": "s", "label_source": "none", "split": "train"}}
    lines = [json.dumps(good), "{not json", "[1]",
             json.dumps({**good, "id": "B", "clean_input": ""}),
             json.dumps({**good, "clean_input": "other"}),
             json.dumps({**good, "id": "C"}),
             json.dumps({**good, "id": "D", "clean_input": None}),
             json.dumps({**good, "id": "E", "clean_input": 5}),
             json.dumps({**good, "id": "F", "clean_input": "y" * 2000}),
             json.dumps({**good, "id": "G", "schema_version": "0.1", "clean_input": "z\x07"})]
    p = tmp_path / "bad.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    rep = validate_file(str(p), "teacher_input")
    for cat in ("invalid_json", "malformed_record", "empty_input", "duplicate_id", "duplicate_text",
                "null_value", "wrong_type", "too_long", "schema_violation", "bad_unicode"):
        assert rep.counts[cat] >= 1, cat
    assert rep.valid == 1


def test_student_target_and_prediction_roundtrip():
    rec = json.load(open(os.path.join(ROOT, "data/schemas/example.json"), encoding="utf-8"))
    rec["metadata"].update({"label_source": "silver", "split": "train"})
    assert check_record(rec) is None
    target = make_target(rec)
    assert target["pii"][0]["span"] == "9X87K"
    row = {"id": rec["id"], "text": rec["clean_input"], "source": rec}
    pred = to_prediction(row, json.dumps(target), {"num_tokens": 1}, "ckpt")
    assert pred["metadata"]["json_valid"]
    assert [(p["start"], p["end"]) for p in pred["pii"]] == [(41, 46), (62, 72)]
    bad = to_prediction(row, "not json", {}, "ckpt")
    assert not bad["metadata"]["json_valid"] and bad["pii"] == []
