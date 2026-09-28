"""Dialect teacher inputs from Vashantor (Romanized regional Bangla, proposal type DIALECT_MEANING).

    PYTHONUTF8=1 python scripts/build_dialect_inputs.py          # train 500 / dev 100 / test 300

Source: Vashantor (Faria et al., arXiv 2311.11142), Mendeley Data bj5jgk878b v2, CSV version,
unzipped to data/raw/vashantor/csv/ (git-ignored). Regions: Barishal, Chittagong, Mymensingh,
Noakhali, Sylhet. Each CSV row is one sentence in 6 parallel forms:
  <region>_banglish_speech  -> input (Romanized dialect: what the pipeline sees)
  banglish_speech, bangla_speech, <region>_bangla_speech, english_speech -> metadata references
Header names in the files carry trailing spaces; they are stripped.

Splits follow Vashantor (Train/Validation/Test -> train/dev/test), balanced over the five regions.
Test is read first, then dev, then train; a row is dropped if its dialect text OR its standard
sentence was already used, because the same standard sentence appears in several regions and a
sentence used in one split must not appear (in another dialect) in a different split.
Within one split the same sentence may appear in several regions (parallel, per-region comparison).
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pii.detector import detect  # noqa: E402
from src.preprocessing.clean import clean_text  # noqa: E402
from src.utils.jsonl_validate import BAD_CHARS_RE, DEFAULT_MAX_CHARS  # noqa: E402

REGIONS = ["Barishal", "Chittagong", "Mymensingh", "Noakhali", "Sylhet"]
PLAN = [("test", "Test", "DL_TE_"), ("dev", "Validation", "DL_DV_"), ("train", "Train", "DL_TR_")]


def read_rows(data_dir, folder, region):
    paths = glob.glob(os.path.join(data_dir, "**", folder, f"{region}*.csv"), recursive=True)
    rows = []
    for p in paths:
        for row in csv.DictReader(open(p, encoding="utf-8-sig")):
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw/vashantor/csv")
    ap.add_argument("--train", type=int, default=500, help="total over all regions (0 = none)")
    ap.add_argument("--dev", type=int, default=100)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    if not glob.glob(os.path.join(args.data_dir, "**", "*.csv"), recursive=True):
        sys.exit(f"ERROR: no CSV files under {args.data_dir}. Unzip Vashantor_CSV_Format.zip there first.")

    totals = {"train": args.train, "dev": args.dev, "test": args.test}
    stats = {"dataset": "Vashantor (CSV)", "seed": args.seed, "limits": totals, "raw_available": {},
             "dropped": Counter(), "final": {}, "final_per_region": {}, "records_with_pii_detected": 0}
    rng = random.Random(args.seed)
    used_dialect, standard_split = set(), {}     # standard sentence -> split where it was first used
    os.makedirs(args.out_dir, exist_ok=True)

    for split, folder, prefix in PLAN:
        per_region = totals[split] // len(REGIONS)
        records, region_counts = [], Counter()
        for region in REGIONS:
            rows = read_rows(args.data_dir, folder, region)
            stats["raw_available"][f"{split}:{region}"] = len(rows)
            rng.shuffle(rows)
            key = region.lower()
            for row in rows:
                if region_counts[region] >= per_region:
                    break
                text = row.get(f"{key}_banglish_speech", "")
                clean = clean_text(text)
                if not clean:
                    stats["dropped"]["empty_or_missing_dialect_text"] += 1
                    continue
                if len(clean) > args.max_chars or BAD_CHARS_RE.search(clean):
                    stats["dropped"]["too_long_or_broken_unicode"] += 1
                    continue
                d_norm = " ".join(clean.lower().split())
                s_norm = " ".join(row.get("banglish_speech", "").lower().split())
                if d_norm in used_dialect:
                    stats["dropped"]["duplicate_dialect_text"] += 1
                    continue
                if s_norm and standard_split.get(s_norm, split) != split:
                    # the same sentence (in another dialect) is already in an earlier split
                    stats["dropped"]["standard_sentence_in_other_split"] += 1
                    continue
                used_dialect.add(d_norm)
                if s_norm:
                    standard_split.setdefault(s_norm, split)
                found = detect(clean)
                stats["records_with_pii_detected"] += bool(found)
                region_counts[region] += 1
                records.append({
                    "id": f"{prefix}{len(records) + 1:06d}",
                    "schema_version": "0.2",
                    "input": text, "clean_input": clean,
                    "normalized_text": None, "sanitized_prompt": None,
                    "pii": [], "preserved_entities": [], "uncertainties": [], "routing": None,
                    "metadata": {
                        "language": "banglish", "surface_form": "dialect", "dialect_region": region,
                        "source": "Vashantor", "source_split": folder,
                        "reference_standard_banglish": row.get("banglish_speech"),
                        "reference_standard_bangla": row.get("bangla_speech"),
                        "reference_dialect_bangla": row.get(f"{key}_bangla_speech"),
                        "reference_english": row.get("english_speech"),
                        "pii_detected": [{"type": f["type"], "text": f["text"], "start": f["start"], "end": f["end"]}
                                         for f in found],
                        "label_source": "none", "split": split,
                    },
                })
        stats["final"][split] = len(records)
        stats["final_per_region"][split] = dict(region_counts)
        with open(os.path.join(args.out_dir, f"teacher_inputs_dialect_{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats["dropped"] = dict(stats["dropped"])
    stats["total_final"] = sum(stats["final"].values())
    with open(os.path.join(args.out_dir, "teacher_inputs_dialect_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nWrote teacher_inputs_dialect_{{train,dev,test}}.jsonl to {args.out_dir}")


if __name__ == "__main__":
    main()
