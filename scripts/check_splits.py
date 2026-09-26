"""Check that IDs are unique across splits and that no text appears in two splits."""
import json
from collections import defaultdict

PATTERN = "data/processed/teacher_inputs_banglatlit_{split}.jsonl"
SPLITS = ["train", "dev", "test"]

ids = defaultdict(list)     # id   -> splits it appears in
texts = defaultdict(set)    # text -> splits it appears in
for split in SPLITS:
    with open(PATTERN.format(split=split), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids[r["id"]].append(split)
            texts[r["text"].strip().lower()].add(split)

dup_ids = {i: s for i, s in ids.items() if len(s) > 1}
cross = {t: s for t, s in texts.items() if len(s) > 1}

print("Total IDs:", len(ids))
print("Duplicate IDs:", len(dup_ids))
print("Texts appearing in more than one split:", len(cross))
for t, s in list(cross.items())[:5]:
    print("  ", sorted(s), repr(t[:80]))
