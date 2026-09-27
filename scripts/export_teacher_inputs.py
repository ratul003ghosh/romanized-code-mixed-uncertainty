"""Export cleaned records for Zarif's teacher code: {"id", "text"}, one file per split.

Teacher outputs on train become silver TRAINING data.
Teacher outputs on dev/test are for evaluation only (RQ1/K1) and must never be used for training.
"""
import json

IN_PATH = "data/processed/banglatlit_sample_clean.jsonl"
OUT_PATTERN = "data/processed/teacher_inputs_banglatlit_{split}.jsonl"
SPLITS = ["train", "dev", "test"]

files = {s: open(OUT_PATTERN.format(split=s), "w", encoding="utf-8") for s in SPLITS}
counts = {s: 0 for s in SPLITS}

with open(IN_PATH, encoding="utf-8") as fin:
    for line in fin:
        r = json.loads(line)
        split = r["metadata"]["split"]
        if split not in files:
            continue
        out = {
            "id": r["id"],
            "text": r["clean_input"],
            "source": r["metadata"]["source"],
            "split": split,
            "label_source": "none",
        }
        files[split].write(json.dumps(out, ensure_ascii=False) + "\n")
        counts[split] += 1

for s in SPLITS:
    files[s].close()
    print(f"{s}: exported {counts[s]} -> {OUT_PATTERN.format(split=s)}")
