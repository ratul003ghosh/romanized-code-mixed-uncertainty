"""Economic-domain evaluation slice: Bangla/English financial scam and normal messages.

    PYTHONUTF8=1 python scripts/build_finance_inputs.py

Source: "Financial scams detection dataset", Mendeley Data znsk27yk3h v1 (523 messages, columns
label = scam/ham and message), unzipped to data/raw/finance_scams/ (git-ignored). The authors state
that names, phone numbers, account numbers and e-mails were removed or anonymized.

Every message is kept (eval-only, split "test", domain "finance") and tagged by script with the same
word-list rule as BanglishRev: bangla_script / banglish / english (anything else: "unclear").
The scam/ham label is kept in metadata; it is not a sanitization label.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.build_banglishrev_inputs import classify, is_plain_english  # noqa: E402
from src.pii.detector import detect  # noqa: E402
from src.preprocessing.clean import clean_text  # noqa: E402
from src.utils.jsonl_validate import BAD_CHARS_RE, DEFAULT_MAX_CHARS  # noqa: E402

SURFACE = {"bangla_script": ("bangla", "bangla_script"), "banglish": ("banglish", "banglish"),
           "english": ("english", "english"), "unclear": ("unknown", "code_mixed")}


def script_of(clean):
    label = classify(clean, min_words=1)
    if label in ("bangla_script", "banglish"):
        return label
    return "english" if is_plain_english(clean) else "unclear"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw/finance_scams")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    paths = glob.glob(os.path.join(args.data_dir, "**", "*.csv"), recursive=True)
    if not paths:
        sys.exit(f"ERROR: no CSV under {args.data_dir}. Unzip the Mendeley download there first.")

    rows = list(csv.DictReader(open(paths[0], encoding="utf-8-sig")))
    stats = {"dataset": "Financial scams detection dataset (Mendeley znsk27yk3h)", "raw_read": len(rows),
             "dropped": Counter(), "final": 0, "by_script": Counter(), "by_label": Counter(),
             "records_with_pii_detected": 0, "pii_types": Counter()}
    seen, records = set(), []
    for row in rows:
        text = (row.get("message") or "")
        clean = clean_text(text)
        if not clean:
            stats["dropped"]["empty_after_cleaning"] += 1
            continue
        if len(clean) > args.max_chars or BAD_CHARS_RE.search(clean):
            stats["dropped"]["too_long_or_broken_unicode"] += 1
            continue
        norm = " ".join(clean.lower().split())
        if norm in seen:
            stats["dropped"]["duplicate"] += 1
            continue
        seen.add(norm)
        script = script_of(clean)
        language, surface = SURFACE[script]
        label = (row.get("label") or "").strip()
        stats["by_script"][script] += 1
        stats["by_label"][label] += 1
        found = detect(clean)
        if found:
            stats["records_with_pii_detected"] += 1
            stats["pii_types"].update(f["type"] for f in found)
        records.append({
            "id": f"FN_TE_{len(records) + 1:06d}",
            "schema_version": "0.2",
            "input": text, "clean_input": clean,
            "normalized_text": None, "sanitized_prompt": None,
            "pii": [], "preserved_entities": [], "uncertainties": [], "routing": None,
            "metadata": {
                "language": language, "surface_form": surface, "domain": "finance",
                "source": "FinancialScams-Mendeley", "scam_label": label,
                "pii_detected": [{"type": f["type"], "text": f["text"], "start": f["start"], "end": f["end"]}
                                 for f in found],
                "eval_only": True, "label_source": "none", "split": "test",
            },
        })
    stats["final"] = len(records)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "teacher_inputs_finance_test.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()}
    with open(os.path.join(args.out_dir, "teacher_inputs_finance_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nWrote teacher_inputs_finance_test.jsonl to {args.out_dir}")


if __name__ == "__main__":
    main()
