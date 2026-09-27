"""Prove, without a GPU, that teacher_inputs*.jsonl fits the existing teacher code.

    PYTHONUTF8=1 python scripts/check_teacher_interface.py data/processed/teacher_inputs.jsonl
    PYTHONUTF8=1 python scripts/check_teacher_interface.py data/processed/teacher_inputs.jsonl --tokenizer

Checks, using the teacher code itself (nothing re-implemented):
  1. src.teachers.common.load_inputs reads every line and picks clean_input as the text
  2. src.teachers.prompts.build_messages builds the prompt for all 4 teacher variants
  3. src.uncertainty.silver.to_record turns a (dummy) teacher result for each input into a silver
     record that keeps id / split / source, passes the schema validator, and is accepted by
     scripts/build_student_data.py (train only)
  4. optional (--tokenizer): the teacher tokenizer (Qwen2.5-7B-Instruct) counts prompt tokens, so
     over-long prompts are found before the GPU run. Needs `transformers` and internet.
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.teachers.common import load_inputs  # noqa: E402
from src.teachers.prompts import build_messages, VARIANT_FOCUS  # noqa: E402
from src.uncertainty.silver import to_record  # noqa: E402
from src.utils.jsonl_validate import validate_file  # noqa: E402
from scripts.build_student_data import check_record  # noqa: E402

TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_file")
    ap.add_argument("--tokenizer", action="store_true", help="also count prompt tokens with the teacher tokenizer")
    ap.add_argument("--max-prompt-tokens", type=int, default=6000)
    args = ap.parse_args()

    raw_lines = sum(1 for line in open(args.input_file, encoding="utf-8") if line.strip())
    inputs = load_inputs({"input_file": args.input_file, "limit": None})
    print(f"1. load_inputs: {len(inputs)} of {raw_lines} lines loaded")
    assert len(inputs) == raw_lines, "some lines were skipped by load_inputs (empty text?)"
    bad_text = [r["id"] for r in inputs if r["text"] != r.get("clean_input")]
    assert not bad_text, f"teacher would not read clean_input for: {bad_text[:5]}"
    print("   text given to the teachers = clean_input for every record")

    variants = list(VARIANT_FOCUS)
    for r in inputs:
        for v in variants:
            msgs = build_messages(v, r["text"])
            assert msgs[-1] == {"role": "user", "content": r["text"]}
    print(f"2. build_messages: prompts built for {len(inputs)} inputs x {len(variants)} variants {variants}")

    splits = {}
    silver = []
    for r in inputs:
        dummy = {"id": r["id"], "text": r["text"], "pivot_teacher": "same:balanced", "spans": [],
                 "pivot_json": {"sanitized_prompt": "dummy", "pii": [], "preserved_entities": []}}
        rec = to_record(dummy, r, run_name="interface_check")
        assert rec["id"] == r["id"] and rec["clean_input"] == r["text"]
        assert rec["metadata"]["split"] == r["metadata"]["split"]
        splits[rec["metadata"]["split"]] = splits.get(rec["metadata"]["split"], 0) + 1
        silver.append(rec)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for rec in silver:
            tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    rep = validate_file(tmp.name, "record")
    os.unlink(tmp.name)
    assert rep.ok, rep.summary()
    accepted = sum(check_record(rec) is None for rec in silver)
    print(f"3. silver.to_record: {len(silver)} records, schema PASS, splits kept {splits}; "
          f"build_student_data would accept {accepted} (train only)")

    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
        lengths = []
        for r in inputs:
            prompt = tok.apply_chat_template(build_messages("balanced", r["text"]), add_generation_prompt=True,
                                             tokenize=False)
            lengths.append(len(tok(prompt, add_special_tokens=False)["input_ids"]))
        over = sum(n > args.max_prompt_tokens for n in lengths)
        print(f"4. tokenizer: prompt tokens min {min(lengths)}, max {max(lengths)}, "
              f"mean {sum(lengths) / len(lengths):.0f}; over {args.max_prompt_tokens}: {over}")
        # the system prompt + few-shot examples alone are hundreds of tokens; a tiny number means
        # the count itself is broken, so fail instead of reporting a false PASS
        assert min(lengths) > 100, f"token count looks wrong (min {min(lengths)})"
        assert over == 0

    print("RESULT: PASS - input file is compatible with the teacher pipeline")


if __name__ == "__main__":
    main()
