"""Export our cleaned records in the format Zarif's teacher code reads: {"id", "text"}."""
import json

IN_PATH = "data/processed/banglatlit_sample_clean.jsonl"
OUT_PATH = "data/processed/teacher_inputs_banglatlit_train.jsonl"
KEEP_SPLIT = "train"   # silver labels are for training only; never export test here

count = 0
with open(IN_PATH, encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
    for line in fin:
        r = json.loads(line)
        if r["metadata"]["split"] != KEEP_SPLIT:
            continue
        out = {
            "id": r["id"],
            "text": r["clean_input"],
            "source": r["metadata"]["source"],
            "split": r["metadata"]["split"],
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        count += 1

print(f"Exported {count} records to {OUT_PATH}")
