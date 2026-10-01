"""Silver part 2: turn teacher silver records (schema v0.2) into student training pairs.

Input : JSONL of silver records (label_source == "silver").
Output: JSONL, one training example per line:
        {"id": ..., "messages": [system, user, assistant]}
        user      = clean_input
        assistant = target JSON (proposal Section 2 format)

Safety rules:
- Only label_source == "silver" AND split == "train" become training data (proposal Section 3.4).
- Records that fail the schema check are discarded (proposal Section 3.2, step 5).

Usage:
    python scripts/build_student_data.py INPUT.jsonl OUTPUT.jsonl
"""
import json
import sys
from collections import Counter

SYSTEM_PROMPT = ("You sanitize Romanized Bangla (Banglish) prompts. Return only JSON with keys: "
                 "sanitized_prompt, pii, preserved_entities, uncertainties, routing.")

ROUTES = {"PROCEED", "PROCEED_WITH_FLAGS", "ASK_USER", "ESCALATE", None}
UNC_TYPES = {"TRANSLITERATION", "NUMERIC_AMBIGUITY", "PII_BOUNDARY", "DIALECT_MEANING", "INTENT"}


def check_record(r):
    """Return None if the record is usable, otherwise a short reason string."""
    meta = r.get("metadata", {})
    if meta.get("label_source") != "silver":
        return "not_silver"
    if meta.get("split") != "train":
        return "not_train_split"
    if not r.get("clean_input"):
        return "missing_clean_input"
    if not isinstance(r.get("sanitized_prompt"), str) or not r["sanitized_prompt"]:
        return "missing_sanitized_prompt"
    if r.get("routing") not in ROUTES:
        return "bad_routing"
    for key in ("pii", "preserved_entities", "uncertainties"):
        if not isinstance(r.get(key), list):
            return f"{key}_not_list"
    for u in r["uncertainties"]:
        if not u.get("types") or not set(u["types"]) <= UNC_TYPES:
            return "bad_uncertainty_type"
        for score in ("aleatoric", "epistemic"):
            v = u.get(score)
            if v is not None and not (0.0 <= v <= 1.0):
                return f"bad_{score}_score"
    return None


def round2(v):
    return None if v is None else round(float(v), 2)


def make_target(r, args=None):
    """Keep only what the student should output (proposal Section 2)."""
    uncertainties = []
    for u in r["uncertainties"]:
        unc = {"span": u["span"]}
        if not (args and args.ablate_types):
            unc["types"] = u["types"]
        unc["candidates"] = u.get("candidates", [])
        if not (args and args.ablate_aleatoric):
            unc["aleatoric"] = round2(u.get("aleatoric"))
        if not (args and args.ablate_epistemic):
            unc["epistemic"] = round2(u.get("epistemic"))
        
        # If any keys were ablated, we might want to keep the object structure if there's anything left.
        # But wait, if types, aleatoric, epistemic are all removed, we just have 'span'.
        uncertainties.append(unc)
        
    if args and args.ablate_uncertainty:
        uncertainties = []

    routing = None if (args and args.ablate_routing) else r["routing"]

    return {
        "sanitized_prompt": r["sanitized_prompt"],
        "pii": [{"type": p["type"], "placeholder": p["placeholder"], "span": p.get("text")} for p in r["pii"]],
        "preserved_entities": r["preserved_entities"],
        "uncertainties": uncertainties,
        "routing": routing,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build student training data with optional ablations.")
    ap.add_argument("in_path", help="Input silver.jsonl")
    ap.add_argument("out_path", help="Output student_train.jsonl")
    ap.add_argument("--ablate-aleatoric", action="store_true", help="Remove aleatoric scores")
    ap.add_argument("--ablate-epistemic", action="store_true", help="Remove epistemic scores")
    ap.add_argument("--ablate-types", action="store_true", help="Remove uncertainty types")
    ap.add_argument("--ablate-uncertainty", action="store_true", help="Remove all uncertainty")
    ap.add_argument("--ablate-routing", action="store_true", help="Remove routing decisions")
    args = ap.parse_args()

    reasons = Counter()
    kept = 0
    with open(args.in_path, encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            problem = check_record(r)
            if problem:
                reasons[problem] += 1
                continue
            example = {
                "id": r["id"],
                "metadata": {"label_source": "silver", "split": "train",
                             "source": r["metadata"].get("source")},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["clean_input"]},
                    {"role": "assistant", "content": json.dumps(make_target(r, args), ensure_ascii=False)},
                ],
            }
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            kept += 1
    print(f"Kept {kept} training examples -> {args.out_path}")
    print(f"Discarded {sum(reasons.values())}: {dict(reasons)}")


if __name__ == "__main__":
    main()
