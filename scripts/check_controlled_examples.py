"""Check the controlled examples (schema v0.2): counts per category and position correctness.

Usage: python scripts/check_controlled_examples.py
"""
import json
from collections import Counter

PATH = "data/synthetic/controlled_examples_v02.jsonl"

rows = [json.loads(line) for line in open(PATH, encoding="utf-8") if line.strip()]

errors = []
for r in rows:
    text = r["clean_input"]
    for p in r["pii"]:
        if text[p["start"]:p["end"]] != p["text"]:
            errors.append(f"{r['id']}: PII position wrong for {p['text']!r}")
    for u in r["uncertainties"]:
        if text[u["start"]:u["end"]] != u["span"]:
            errors.append(f"{r['id']}: uncertainty position wrong for {u['span']!r}")
    if r["metadata"]["label_source"] != "synthetic":
        errors.append(f"{r['id']}: label_source should be 'synthetic'")

print("=" * 50)
print("CONTROLLED EXAMPLES CHECK (schema v0.2)")
print("=" * 50)
print(f"Total examples: {len(rows)}")
print("\nBy category:")
for cat, n in sorted(Counter(r["metadata"]["category"] for r in rows).items()):
    print(f"  {cat}: {n}")
print("\nBy expected routing:")
for route, n in sorted(Counter(r["routing"] for r in rows).items()):
    print(f"  {route}: {n}")
skipped = [r["id"] for r in rows if r["metadata"]["placeholder_input"]]
print(f"\nPlaceholder inputs (skip when running models): {skipped}")
print(f"\nProblems found: {len(errors)}")
for e in errors:
    print("  -", e)
