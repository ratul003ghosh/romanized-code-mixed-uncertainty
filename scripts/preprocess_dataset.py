"""Add clean_input to every record. Reads one JSONL, writes another."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.clean import clean_text

IN_PATH = "data/processed/banglatlit_sample.jsonl"
OUT_PATH = "data/processed/banglatlit_sample_clean.jsonl"

changed = 0
examples = []
with open(IN_PATH, encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
    for line in fin:
        record = json.loads(line)
        record["clean_input"] = clean_text(record["input"])
        if record["clean_input"] != record["input"]:
            changed += 1
            if len(examples) < 3:
                examples.append((record["input"], record["clean_input"]))
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Records changed by cleaning: {changed}")
for before, after in examples:
    print("BEFORE:", repr(before))
    print("AFTER: ", repr(after))
print("Saved to", OUT_PATH)
