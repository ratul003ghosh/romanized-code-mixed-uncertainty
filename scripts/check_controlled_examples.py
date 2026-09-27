import json
from pathlib import Path

DATA = Path(__file__).parent / "examples" / "controlled_examples.json"

with open(DATA, encoding="utf-8") as f:
    examples = json.load(f)

print("=" * 60)
print("MASRAFI DAY 1 - CONTROLLED EXAMPLES CHECK")
print("=" * 60)
print(f"Total examples: {len(examples)}")

categories = {}
for ex in examples:
    categories[ex["category"]] = categories.get(ex["category"], 0) + 1

print("\nCategories:")
for category, count in sorted(categories.items()):
    print(f"  {category}: {count}")

print("\nExamples:")
for ex in examples:
    print(f'- {ex["id"]}: {ex["category"]} -> {ex["expected_routing"]}')

print("\nValidation:")
required = ["id", "category", "input", "normalized", "pii_spans",
            "uncertainty_spans", "expected_routing", "notes"]

errors = []
for ex in examples:
    missing = [field for field in required if field not in ex]
    if missing:
        errors.append(f'{ex.get("id", "UNKNOWN")}: missing {missing}')

if errors:
    print("FAILED")
    for error in errors:
        print(" ", error)
else:
    print("PASSED - all examples contain the required fields.")

print("\nIMPORTANT:")
print("These are controlled synthetic development examples.")
print("They are NOT human-annotated gold data.")
print("No research metric/result is reported by this script.")
