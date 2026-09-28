"""Build the teacher inputs from the REAL BanglaTLit dataset (proposal Section 3.1), in one command.

    PYTHONUTF8=1 python scripts/build_teacher_inputs.py            # 3000 / 500 / 500 (default)
    PYTHONUTF8=1 python scripts/build_teacher_inputs.py --train 0   # 0 = every train row

Steps, in order (all existing modules are reused, nothing is re-implemented):
  1. raw        read BanglaTLit rows (Hugging Face, or --source-file for an offline copy)
  2. clean      src.preprocessing.clean.clean_text  -> clean_input
  3. filter     drop malformed / empty / too long / broken-unicode rows and exact duplicates
                (test and dev are read first, so a sentence shared with train is removed from train)
  4. PII check  src.pii.detector.detect finds PII; src.pii.masking.sanitize must mask all of it.
                The teacher input is NOT masked: the teachers must see the PII to learn to mask it
                (proposal Section 4.1). Detected spans are stored in metadata.pii_detected as a
                rule-based reference only.
  5. write      schema v0.2 records (label_source "none"), one file per split, plus
                teacher_inputs.jsonl (= train split, the file the teacher configs expect) and
                teacher_inputs_stats.json with every count.

Note: BanglaTLit contains no synthetic PII; PII injection (proposal Section 3.2) is a separate step.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pii.detector import detect  # noqa: E402
from src.pii.masking import sanitize  # noqa: E402
from src.preprocessing.clean import clean_text  # noqa: E402
from src.utils.jsonl_validate import BAD_CHARS_RE, DEFAULT_MAX_CHARS  # noqa: E402

DATASET = "aplycaebous/BanglaTLit"
SPLIT_MAP = {"test": "test", "validation": "dev", "train": "train"}   # read order: test, dev, train
ID_PREFIX = {"train": "BG_TR_", "dev": "BG_DV_", "test": "BG_TE_"}


def read_source(source_file):
    """Yield (source_split, row). Offline mode reads a JSONL with a `split` field per row."""
    if source_file:
        with open(source_file, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for split in SPLIT_MAP:
            for row in rows:
                if row.get("split") == split:
                    yield split, row
        return
    from datasets import load_dataset
    ds = load_dataset(DATASET, streaming=True)
    for split in SPLIT_MAP:
        for row in ds[split]:
            yield split, row


def to_record(rid, row, clean, split, pii_found):
    return {
        "id": rid,
        "schema_version": "0.2",
        "input": row["text_transliterated"],
        "clean_input": clean,
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
            "source_id": row.get("id"),
            "source_split": {v: k for k, v in SPLIT_MAP.items()}[split],
            "reference_bengali": row.get("text_bengali"),
            "pii_detected": pii_found,
            "label_source": "none",
            "split": split,
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=3000, help="rows to keep from train (0 = all)")
    ap.add_argument("--dev", type=int, default=500, help="rows to keep from validation (0 = all)")
    ap.add_argument("--test", type=int, default=500, help="rows to keep from test (0 = all)")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="drop inputs longer than this")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--source-file", default=None, help="offline JSONL copy of BanglaTLit (for tests)")
    args = ap.parse_args()
    limits = {"train": args.train, "dev": args.dev, "test": args.test}

    os.makedirs(args.out_dir, exist_ok=True)
    stats = {"dataset": DATASET if not args.source_file else args.source_file,
             "limits": limits, "max_chars": args.max_chars,
             "raw_read": Counter(), "after_preprocessing": Counter(), "dropped": Counter(),
             "final": Counter(), "records_with_pii_detected": Counter(), "pii_types": Counter(),
             "masking_failures": 0}
    kept = {s: [] for s in limits}
    seen_text = {}                 # normalized text -> split where it was first kept
    done = set()

    for source_split, row in read_source(args.source_file):
        split = SPLIT_MAP[source_split]
        if split in done:
            continue
        stats["raw_read"][split] += 1

        text = row.get("text_transliterated")
        if not isinstance(text, str):
            stats["dropped"]["malformed_or_missing_text"] += 1
            continue
        clean = clean_text(text)
        stats["after_preprocessing"][split] += 1
        if not clean:
            stats["dropped"]["empty_after_cleaning"] += 1
            continue
        if len(clean) > args.max_chars:
            stats["dropped"]["too_long"] += 1
            continue
        if BAD_CHARS_RE.search(clean):
            stats["dropped"]["broken_unicode"] += 1
            continue
        norm = " ".join(clean.lower().split())
        if norm in seen_text:
            key = "duplicate_same_split" if seen_text[norm] == split else "duplicate_across_splits"
            stats["dropped"][key] += 1
            continue

        found = detect(clean)
        if found:
            masked = sanitize(clean)["masked"]
            if any(p["text"] in masked for p in found):
                stats["masking_failures"] += 1
            stats["records_with_pii_detected"][split] += 1
            stats["pii_types"].update(p["type"] for p in found)
        pii_found = [{"type": p["type"], "text": p["text"], "start": p["start"], "end": p["end"],
                      "score": p.get("score")} for p in found]

        seen_text[norm] = split
        rid = f"{ID_PREFIX[split]}{len(kept[split]) + 1:06d}"
        kept[split].append(to_record(rid, row, clean, split, pii_found))
        if limits[split] and len(kept[split]) >= limits[split]:
            done.add(split)

    for split, recs in kept.items():
        stats["final"][split] = len(recs)
        with open(os.path.join(args.out_dir, f"teacher_inputs_{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # the file the teacher configs expect by default = the train split (silver training data)
    with open(os.path.join(args.out_dir, "teacher_inputs.jsonl"), "w", encoding="utf-8") as f:
        for r in kept["train"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()}
    stats["total_final"] = sum(stats["final"].values())
    with open(os.path.join(args.out_dir, "teacher_inputs_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nWrote teacher_inputs_{{train,dev,test}}.jsonl and teacher_inputs.jsonl (= train) to {args.out_dir}")


if __name__ == "__main__":
    main()
