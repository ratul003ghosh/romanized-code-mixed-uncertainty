"""Make a tiny student-format file from the controlled synthetic examples, ONLY to test
scripts/train_student.py and scripts/infer_student.py before real silver data exists.

    PYTHONUTF8=1 python scripts/make_student_smoke_data.py

Output: data/synthetic/student_smoke.jsonl (17 examples; the 2 placeholder dialect inputs are skipped).
Every line has metadata.label_source = "synthetic". A model trained on it is NOT a research result.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.build_student_data import SYSTEM_PROMPT, make_target  # noqa: E402

IN_PATH = "data/synthetic/controlled_examples_v02.jsonl"
OUT_PATH = "data/synthetic/student_smoke.jsonl"

kept = 0
with open(IN_PATH, encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
    for line in fin:
        r = json.loads(line)
        if r["metadata"].get("placeholder_input"):
            continue
        example = {
            "id": r["id"],
            "metadata": {"label_source": "synthetic", "split": "none", "source": "controlled_examples"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["clean_input"]},
                {"role": "assistant", "content": json.dumps(make_target(r), ensure_ascii=False)},
            ],
        }
        fout.write(json.dumps(example, ensure_ascii=False) + "\n")
        kept += 1
print(f"Wrote {kept} synthetic smoke examples -> {OUT_PATH} (for script testing only)")
