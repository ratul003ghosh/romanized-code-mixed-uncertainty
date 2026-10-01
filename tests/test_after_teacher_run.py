"""check_teacher_run.py + finalize_silver.py on a real (tiny-model) teacher run of two splits."""
import json
import os
import subprocess
import sys

from src.teachers import generate, pivot, score
from src.teachers.common import out_path, read_jsonl, write_jsonl
from src.teachers.output_schema import validate
from src.uncertainty import spans
from test_teacher_pipeline import build_tiny

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VARIANTS = ["balanced", "dialect", "finance_pii", "transliteration"]


def fake_run(tmp, mdir, split, rows):
    import yaml
    inp = tmp / f"in_{split}.jsonl"
    write_jsonl(inp, [{**r, "split": split, "source": "test"} for r in rows])
    run = tmp / "teacher" / split
    cfg = {"run_name": "t", "input_file": str(inp), "output_dir": str(run), "heterogeneous": [],
           "same_tokenizer": {"model": mdir, "dtype": "float32", "variants": VARIANTS},
           "generation": {"max_new_tokens": 4, "batch_size": 4}, "scoring": {"kd_top_k": 3},
           "spans": {"flag_quantile": 0.5}}
    os.makedirs(run, exist_ok=True)
    with open(run / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    gens = read_jsonl(generate.run(cfg))
    for g in gens:                                      # a tiny random model writes no JSON: give plausible answers
        text = next(r["text"] for r in rows if r["id"] == g["id"])
        num = "01712345678" if "01712345678" in text else None
        obj, _ = validate({"sanitized_prompt": "Call <PHONE_1> today." if num else "Fix this, please.",
                           "pii": [{"type": "PHONE", "span": num, "placeholder": "<PHONE_1>"}] if num else [],
                           "preserved_entities": [], "ambiguous_spans": [
                               {"span": text.split()[0], "types": ["INTENT"], "candidates": ["a", "b"]}]})
        g.update(parsed=obj, parse_ok=True, valid=True)
    write_jsonl(out_path(cfg, "generations.jsonl"), gens)
    pivot.run(cfg)
    _, info = score.run(cfg)
    json.dump(info, open(out_path(cfg, "score_info.json"), "w"))
    spans.report(cfg, spans.run(cfg), out_path(cfg, "generations.jsonl"), info)
    return run


def run_script(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "PYTHONUTF8": "1"})


def test_check_and_finalize(tmp_path):
    mdir = str(tmp_path / "tiny")
    build_tiny(mdir)
    train = fake_run(tmp_path, mdir, "train", [{"id": f"tr{i}", "text": f"amar number 01712345678 ekhon {i}"}
                                               for i in range(4)] + [{"id": "tr9", "text": "eta thik koro"}])
    fake_run(tmp_path, mdir, "dev", [{"id": "dv1", "text": "kal ashbo"}, {"id": "dv2", "text": "eta dekho"}])

    # clean run: no FAIL
    res = run_script("scripts/check_teacher_run.py", str(tmp_path / "teacher"))
    assert res.returncode == 0, res.stdout + res.stderr
    report = json.load(open(tmp_path / "teacher" / "run_check.json"))
    assert report["fail"] == 0 and {r["split"] for r in report["runs"]} == {"train", "dev"}
    assert os.path.exists(tmp_path / "teacher" / "run_check.md")

    # break two train records: one copies the raw phone number, one has a stray placeholder
    sil = read_jsonl(str(train / "silver.jsonl"))
    sil[0]["sanitized_prompt"] = "Call 01712345678 today."
    sil[1]["sanitized_prompt"] = "Call <PHONE_1> and <EMAIL_1>."
    write_jsonl(str(train / "silver.jsonl"), sil)
    res = run_script("scripts/check_teacher_run.py", str(tmp_path / "teacher"))
    assert "raw PII copied" in res.stdout and "placeholders match" in res.stdout

    out = tmp_path / "processed"
    res = run_script("scripts/finalize_silver.py", str(tmp_path / "teacher"), "--out-dir", str(out))
    assert res.returncode == 0, res.stdout + res.stderr
    man = json.load(open(out / "silver_manifest.json"))
    assert man["splits"]["train"]["records"] == 3
    assert man["splits"]["train"]["dropped"] == {"pii_copied": 1, "placeholders": 1}
    assert man["splits"]["train"]["build_student_data_accepts"] == 3
    assert man["splits"]["dev"]["records"] == 2 and man["splits"]["dev"]["validator"] == "PASS"
    assert all(r["metadata"]["split"] == "dev" for r in read_jsonl(str(out / "silver_dev.jsonl")))

    # Saber's converter takes the train file as it is
    res = run_script("scripts/build_student_data.py", str(out / "silver_train.jsonl"), str(out / "student_train.jsonl"))
    assert res.returncode == 0, res.stdout + res.stderr
    assert len(read_jsonl(str(out / "student_train.jsonl"))) == 3


def test_cross_split_overlap_fails(tmp_path):
    mdir = str(tmp_path / "tiny")
    build_tiny(mdir)
    fake_run(tmp_path, mdir, "train", [{"id": "same", "text": "eta thik koro"}])
    fake_run(tmp_path, mdir, "test", [{"id": "same", "text": "eta thik koro"}])
    res = run_script("scripts/check_teacher_run.py", str(tmp_path / "teacher"))
    assert res.returncode == 1 and "cross-split" in res.stdout
