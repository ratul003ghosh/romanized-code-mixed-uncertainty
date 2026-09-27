"""Convert a sample of BanglaTLit into our schema v0.2 (JSONL, one record per line)."""
import json
import os
from datasets import load_dataset

# rows to take per split (prototype sample, not the full dataset)
N_PER_SPLIT = {"train": 3000, "validation": 500, "test": 500}
OUT_PATH = "data/processed/banglatlit_sample.jsonl"
SPLIT_MAP = {"train": "train", "validation": "dev", "test": "test"}


def to_record(row, number, source_split):
    return {
        "id": f"BG_{number:06d}",
        "schema_version": "0.2",
        "input": row["text_transliterated"],
        "normalized_text": None,
        "sanitized_prompt": None,
        "pii": [],
        "preserved_entities": [],
        "uncertainties": [],
        "routing": None,
        "metadata": {
            "language": "banglish",
            "surface_form": "banglish",
            "source": "BanglaTLit",
            "source_id": row["id"],
            "source_split": source_split,
            "reference_bengali": row["text_bengali"],
            "label_source": "none",
            "split": SPLIT_MAP[source_split],
        },
    }


def main():
    ds = load_dataset("aplycaebous/BanglaTLit", streaming=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    number = 1
    skipped = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for source_split, limit in N_PER_SPLIT.items():
            kept = 0
            for row in ds[source_split]:
                text = (row["text_transliterated"] or "").strip()
                if not text:
                    skipped += 1
                    continue
                f.write(json.dumps(to_record(row, number, source_split), ensure_ascii=False) + "\n")
                number += 1
                kept += 1
                if kept >= limit:
                    break
            print(f"{source_split}: kept {kept}")

    print(f"Total records: {number - 1}, skipped empty: {skipped}")
    print("Saved to", OUT_PATH)


if __name__ == "__main__":
    main()
