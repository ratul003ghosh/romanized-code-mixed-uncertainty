"""Medical-domain evaluation slice from BanglaCHQ-Summ (Bangla consumer health questions).

    git clone --depth 1 https://github.com/alvi-khan/BanglaCHQ-Summ.git data/raw/BanglaCHQ-Summ
    PYTHONUTF8=1 python scripts/build_medical_inputs.py

Source: BanglaCHQ-Summ (BLP workshop @ EMNLP 2023), CC-BY-NC-SA-4.0; 2,350 questions from a public
Bangla health forum, each with a human-written summary (train 1880 / valid 235 / test 235).
The questions are in Bangla SCRIPT, not Romanized, so this is a domain + script control set:
surface_form "bangla_script", domain "medical", eval-only (split "test"), taken from the dataset's
own test split. Health questions are sensitive: the files stay in data/processed/ (git-ignored).
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pii.detector import detect  # noqa: E402
from src.preprocessing.clean import clean_text  # noqa: E402
from src.utils.jsonl_validate import BAD_CHARS_RE, DEFAULT_MAX_CHARS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw/BanglaCHQ-Summ/Dataset")
    ap.add_argument("--n", type=int, default=235, help="questions to keep from the test split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    path = os.path.join(args.data_dir, "test.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run:\n"
                 "  git clone --depth 1 https://github.com/alvi-khan/BanglaCHQ-Summ.git data/raw/BanglaCHQ-Summ")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    stats = {"dataset": "BanglaCHQ-Summ (test split)", "raw_read": len(rows), "dropped": Counter(),
             "final": 0, "records_with_pii_detected": 0, "pii_types": Counter()}
    random.Random(args.seed).shuffle(rows)
    seen, records = set(), []
    for row in rows:
        if len(records) >= args.n:
            break
        text = row.get("question")
        if not isinstance(text, str):
            stats["dropped"]["malformed_or_missing_text"] += 1
            continue
        clean = clean_text(text)
        if not clean:
            stats["dropped"]["empty_after_cleaning"] += 1
            continue
        if len(clean) > args.max_chars:
            stats["dropped"]["too_long"] += 1
            continue
        if BAD_CHARS_RE.search(clean):
            stats["dropped"]["broken_unicode"] += 1
            continue
        if clean in seen:
            stats["dropped"]["duplicate"] += 1
            continue
        seen.add(clean)
        found = detect(clean)
        if found:
            stats["records_with_pii_detected"] += 1
            stats["pii_types"].update(f["type"] for f in found)
        records.append({
            "id": f"MD_TE_{len(records) + 1:06d}",
            "schema_version": "0.2",
            "input": text, "clean_input": clean,
            "normalized_text": None, "sanitized_prompt": None,
            "pii": [], "preserved_entities": [], "uncertainties": [], "routing": None,
            "metadata": {
                "language": "bangla", "surface_form": "bangla_script", "domain": "medical",
                "source": "BanglaCHQ-Summ", "source_id": row.get("id"),
                "reference_summary": row.get("summary"),
                "pii_detected": [{"type": f["type"], "text": f["text"], "start": f["start"], "end": f["end"]}
                                 for f in found],
                "eval_only": True, "label_source": "none", "split": "test",
            },
        })
    stats["final"] = len(records)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "teacher_inputs_medical_test.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()}
    with open(os.path.join(args.out_dir, "teacher_inputs_medical_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nWrote teacher_inputs_medical_test.jsonl to {args.out_dir} (sensitive health text: keep private)")


if __name__ == "__main__":
    main()
