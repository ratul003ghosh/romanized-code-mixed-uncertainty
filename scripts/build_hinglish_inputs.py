"""Hinglish teacher inputs from COMI-LINGUA, TN part (proposal Section 3.3, RQ4 generality check).

    PYTHONUTF8=1 python scripts/build_hinglish_inputs.py            # dev 100 + test 300 (proposal size)

Source: LingoIITGN/COMI-LINGUA, config "TN" (text normalization), CC-BY-4.0.
  Sentences                      -> input (real, noisy Roman-script Hinglish)
  Annotated by: Annotator 1/2/3  -> metadata.reference_normalizations (all three kept; they can differ)
Only Roman-script rows are used (rows with any Devanagari character are skipped and counted),
because the pipeline input is Romanized text. dev comes from TN train, test from TN test.
Steps and checks are the same as build_teacher_inputs.py (clean_text, filters, PII check, no masking).
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pii.detector import detect  # noqa: E402
from src.pii.masking import sanitize  # noqa: E402
from src.preprocessing.clean import clean_text  # noqa: E402
from src.utils.jsonl_validate import BAD_CHARS_RE, DEFAULT_MAX_CHARS  # noqa: E402

DATASET, CONFIG = "LingoIITGN/COMI-LINGUA", "TN"
ANNOTATORS = ["Annotated by: Annotator 1", "Annotated by: Annotator 2", "Annotated by: Annotator 3"]
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
PLAN = [("test", "test", "HG_TE_"), ("dev", "train", "HG_DV_")]   # (our split, source split, id prefix)


def load_rows(source_file):
    if source_file:
        rows = [json.loads(line) for line in open(source_file, encoding="utf-8") if line.strip()]
        return {s: [r for r in rows if r.get("split") == s] for s in ("train", "test")}
    from datasets import load_dataset
    ds = load_dataset(DATASET, CONFIG)
    return {s: list(ds[s]) for s in ("train", "test")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", type=int, default=100)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--source-file", default=None, help="offline JSONL with a `split` field (for tests)")
    args = ap.parse_args()
    limits = {"dev": args.dev, "test": args.test}

    rows = load_rows(args.source_file)
    stats = {"dataset": f"{DATASET} ({CONFIG})" if not args.source_file else args.source_file,
             "limits": limits, "seed": args.seed, "raw_available": {}, "raw_read": Counter(),
             "dropped": Counter(), "final": Counter(), "records_with_pii_detected": Counter(),
             "pii_types": Counter(), "masking_failures": 0, "annotators_disagree": Counter()}
    seen, kept = set(), {"dev": [], "test": []}
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    for split, source_split, prefix in PLAN:          # test first, so test sentences never land in dev
        pool = list(rows[source_split])
        stats["raw_available"][source_split] = len(pool)
        rng.shuffle(pool)
        for row in pool:
            if len(kept[split]) >= limits[split]:
                break
            stats["raw_read"][split] += 1
            text = row.get("Sentences")
            if not isinstance(text, str):
                stats["dropped"]["malformed_or_missing_text"] += 1
                continue
            if DEVANAGARI_RE.search(text):
                stats["dropped"]["devanagari_script"] += 1
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
            norm = " ".join(clean.lower().split())
            if norm in seen:
                stats["dropped"]["duplicate"] += 1
                continue
            seen.add(norm)
            refs = [row.get(a) for a in ANNOTATORS]
            refs = [r for r in refs if isinstance(r, str) and r.strip()]
            if len({" ".join(r.lower().split()) for r in refs}) > 1:          # ignore case and spacing
                stats["annotators_disagree"][split] += 1
            found = detect(clean)
            if found:
                masked = sanitize(clean)["masked"]
                if any(p["text"] in masked for p in found):
                    stats["masking_failures"] += 1
                stats["records_with_pii_detected"][split] += 1
                stats["pii_types"].update(p["type"] for p in found)
            kept[split].append({
                "id": f"{prefix}{len(kept[split]) + 1:06d}",
                "schema_version": "0.2",
                "input": text,
                "clean_input": clean,
                "normalized_text": None, "sanitized_prompt": None,
                "pii": [], "preserved_entities": [], "uncertainties": [], "routing": None,
                "metadata": {
                    "language": "hinglish", "surface_form": "hinglish", "source": "COMI-LINGUA",
                    "source_config": CONFIG, "source_split": source_split,
                    "reference_normalizations": refs,
                    "pii_detected": [{"type": p["type"], "text": p["text"], "start": p["start"], "end": p["end"]}
                                     for p in found],
                    "label_source": "none", "split": split,
                },
            })

    for split, recs in kept.items():
        stats["final"][split] = len(recs)
        with open(os.path.join(args.out_dir, f"teacher_inputs_hinglish_{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()}
    stats["total_final"] = sum(stats["final"].values())
    with open(os.path.join(args.out_dir, "teacher_inputs_hinglish_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nWrote teacher_inputs_hinglish_{{dev,test}}.jsonl to {args.out_dir}")


if __name__ == "__main__":
    main()
