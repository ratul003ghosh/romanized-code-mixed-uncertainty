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


HG_A = ["Annotated by: Annotator 1", "Annotated by: Annotator 2", "Annotated by: Annotator 3"]


def test_hinglish_builder(tmp_path):
    rows = [{"split": "test", "Sentences": f"ABB KI BAAR {i} PAR YAAR", HG_A[0]: "Ab ki baar",
             HG_A[1]: "ab ki baar", HG_A[2]: "Ab ki paar"} for i in range(5)]
    rows += [{"split": "test", "Sentences": "देश (India) में"},
             {"split": "train", "Sentences": "ABB KI BAAR 0 PAR YAAR"},          # duplicate of a test row
             {"split": "train", "Sentences": "Tum se na ho paayega", HG_A[0]: "x", HG_A[1]: "x", HG_A[2]: "x"}]
    src = tmp_path / "hg.jsonl"
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    out = tmp_path / "out"
    res = run("scripts/build_hinglish_inputs.py", "--source-file", str(src), "--out-dir", str(out))
    assert res.returncode == 0, res.stderr
    stats = json.loads((out / "teacher_inputs_hinglish_stats.json").read_text())
    assert stats["final"] == {"dev": 1, "test": 5}
    assert stats["dropped"] == {"devanagari_script": 1, "duplicate": 1}
    assert stats["annotators_disagree"] == {"test": 5}
    for split in ("dev", "test"):
        assert validate_file(str(out / f"teacher_inputs_hinglish_{split}.jsonl"), "teacher_input").ok


def test_banglishrev_builder_keeps_banglish_only_and_is_eval_only(tmp_path):
    revs = ["খুব ভালো", "very good product and nice quality really", "ok",
            "amar number 01712345678 e call dile valo hoy", "product ta onek valo ache",
            "product ta onek valo ache", "Khob valo kinto selar shobidha jonok naa"]
    data = [{"Root Category": "Phones", "Reviews": [{"Buyer ID": 7, "Review Content": r} for r in revs]}]
    src = tmp_path / "br.json"
    src.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out"
    res = run("scripts/build_banglishrev_inputs.py", "--json-path", str(src), "--out-dir", str(out))
    assert res.returncode == 0, res.stderr
    stats = json.loads((out / "teacher_inputs_banglishrev_stats.json").read_text())
    assert stats["language_rule"] == {"bangla_script": 1, "english_or_unclear": 1, "too_short": 1, "banglish": 4}
    assert stats["final"] == {"test": 3, "pii_detected_group": 1, "random_group": 2,
                              "english": 1, "bangla_script": 0}
    assert validate_file(str(out / "teacher_inputs_banglishrev_english_test.jsonl"), "teacher_input").ok
    path = out / "teacher_inputs_banglishrev_test.jsonl"
    assert validate_file(str(path), "teacher_input").ok
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        assert rec["metadata"]["split"] == "test" and rec["metadata"]["eval_only"]
        assert "Buyer ID" not in json.dumps(rec)


def test_medical_builder(tmp_path):
    d = tmp_path / "Dataset"
    d.mkdir()
    (d / "test.csv").write_text("id,question,indices,summary\n1,আমার মাথা ব্যথা করে কি করব,0,মাথা ব্যথা\n"
                                "2,আমার মাথা ব্যথা করে কি করব,1,dup\n3,,2,empty\n", encoding="utf-8")
    out = tmp_path / "out"
    res = run("scripts/build_medical_inputs.py", "--data-dir", str(d), "--out-dir", str(out))
    assert res.returncode == 0, res.stderr
    stats = json.loads((out / "teacher_inputs_medical_stats.json").read_text())
    assert stats["final"] == 1 and stats["dropped"] == {"duplicate": 1, "empty_after_cleaning": 1}
    rec = json.loads((out / "teacher_inputs_medical_test.jsonl").read_text(encoding="utf-8"))
    assert rec["metadata"]["surface_form"] == "bangla_script" and rec["metadata"]["domain"] == "medical"
    assert validate_file(str(out / "teacher_inputs_medical_test.jsonl"), "teacher_input").ok


def _vashantor_fixture(root):
    import csv as _csv
    regions = ["barishal", "chittagong", "mymensingh", "noakhali", "sylhet"]
    rows = {"Test": [("tomar abbu kemon ache?", "{r} dialect A")],
            "Validation": [("ami bhat khabo", "{r} dialect B")],
            # same standard sentence as Test but different dialect wording -> must be dropped from train
            "Train": [("tomar abbu kemon ache?", "{r} dialect A2"), ("amar mon kharap", "{r} dialect C")]}
    for folder, items in rows.items():
        (root / folder).mkdir(parents=True)
        for r in regions:
            with open(root / folder / f"{r.capitalize()} {folder} Translation.csv", "w", encoding="utf-8-sig",
                      newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(["bangla_speech ", "banglish_speech ", f"{r}_bangla_speech ", f"{r}_banglish_speech ",
                            "region_name ", "english_speech"])
                for std, dia in items:
                    w.writerow(["বাংলা", std + " ", "ডায়ালেক্ট", dia.format(r=r) + " ", r, "English"])


def test_dialect_builder_balances_regions_and_blocks_cross_split_sentences(tmp_path):
    _vashantor_fixture(tmp_path / "csv")
    out = tmp_path / "out"
    res = run("scripts/build_dialect_inputs.py", "--data-dir", str(tmp_path / "csv"), "--out-dir", str(out),
              "--train", "10", "--dev", "5", "--test", "5")
    assert res.returncode == 0, res.stderr
    stats = json.loads((out / "teacher_inputs_dialect_stats.json").read_text())
    assert stats["final"] == {"test": 5, "dev": 5, "train": 5}
    assert stats["dropped"] == {"standard_sentence_in_other_split": 5}
    rec = json.loads(open(out / "teacher_inputs_dialect_test.jsonl", encoding="utf-8").readline())
    assert rec["metadata"]["surface_form"] == "dialect" and rec["metadata"]["reference_standard_banglish"]
    for split in ("train", "dev", "test"):
        assert validate_file(str(out / f"teacher_inputs_dialect_{split}.jsonl"), "teacher_input").ok


def test_finance_builder_tags_script(tmp_path):
    d = tmp_path / "fin"
    d.mkdir()
    (d / "data.csv").write_text(
        "label,message\n"
        "scam,Congratulations you have won the bKash offer and it is yours to claim now\n"
        "ham,আপনার বিকাশ একাউন্টে ৫০০ টাকা জমা হয়েছে\n"
        "scam,apnar bkash account block hobe ekhon amake OTP ta pathan\n"
        "ham,\n", encoding="utf-8")
    out = tmp_path / "out"
    res = run("scripts/build_finance_inputs.py", "--data-dir", str(d), "--out-dir", str(out))
    assert res.returncode == 0, res.stderr
    stats = json.loads((out / "teacher_inputs_finance_stats.json").read_text())
    assert stats["by_script"] == {"english": 1, "bangla_script": 1, "banglish": 1}
    assert stats["by_label"] == {"scam": 2, "ham": 1}
    assert validate_file(str(out / "teacher_inputs_finance_test.jsonl"), "teacher_input").ok


def test_hinglish_train_split_never_overlaps_dev(tmp_path):
    a = HG_A
    rows = [{"split": "test", "Sentences": f"test line {i} yaar", a[0]: "x", a[1]: "x", a[2]: "x"} for i in range(4)]
    rows += [{"split": "train", "Sentences": f"train line {i} yaar", a[0]: "x", a[1]: "x", a[2]: "x"} for i in range(8)]
    src = tmp_path / "hg.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "out"
    res = run("scripts/build_hinglish_inputs.py", "--source-file", str(src), "--out-dir", str(out),
              "--train", "5", "--dev", "3", "--test", "4")
    assert res.returncode == 0, res.stderr
    stats = json.loads((out / "teacher_inputs_hinglish_stats.json").read_text())
    assert stats["final"] == {"train": 5, "dev": 3, "test": 4}
    texts = {s: {json.loads(l)["clean_input"] for l in open(out / f"teacher_inputs_hinglish_{s}.jsonl", encoding="utf-8")}
             for s in ("train", "dev", "test")}
    assert not texts["train"] & texts["dev"] and not texts["train"] & texts["test"]
