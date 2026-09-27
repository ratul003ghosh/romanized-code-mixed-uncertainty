"""Create a tiny TEST fixture that imitates teacher silver output (schema v0.2).

This is NOT data for training or results. It only tests build_student_data.py
before Zarif's real silver file exists. Based on data/schemas/example.json.
"""
import copy
import json

EXAMPLE = "data/schemas/example.json"
OUT = "data/synthetic/silver_fixture.jsonl"

with open(EXAMPLE, encoding="utf-8") as f:
    base = json.load(f)

good = copy.deepcopy(base)
good["id"] = "FIX_001"
good["metadata"].update({"label_source": "silver", "split": "train", "source": "fixture"})

bad_routing = copy.deepcopy(good)
bad_routing["id"] = "FIX_002"
bad_routing["routing"] = "MAYBE"            # not an allowed route -> must be rejected

test_split = copy.deepcopy(good)
test_split["id"] = "FIX_003"
test_split["metadata"]["split"] = "test"    # test data -> must never become training data

with open(OUT, "w", encoding="utf-8") as f:
    for r in (good, bad_routing, test_split):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote 3 fixture records to {OUT} (1 valid, 1 bad routing, 1 test split)")
