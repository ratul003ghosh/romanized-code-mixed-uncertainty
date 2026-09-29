"""Convert Mashrafi's 19 controlled examples to schema v0.2 (JSONL).

Input : data/synthetic/mashrafi_original/controlled_examples.json  (kept unchanged)
Output: data/synthetic/controlled_examples_v02.jsonl

What changes:
- positions (start/end) are computed automatically in clean_input
- PII type IDENTIFIER -> ID_NUMBER; placeholders follow <TYPE_N>
- uncertainty "type" -> "types" (list)
- aleatoric/epistemic = null: scores must come from measurement, not be written by hand
- human_ambiguous = null: these are synthetic, not human-annotated
- label_source = "synthetic", split = "none" (never train or report results on these)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.clean import clean_text

IN_PATH = "data/synthetic/mashrafi_original/controlled_examples.json"
OUT_PATH = "data/synthetic/controlled_examples_v02.jsonl"
PII_TYPE_MAP = {"IDENTIFIER": "ID_NUMBER"}
PLACEHOLDER_MAP = {"<IDENTIFIER_1>": "<ID_NUMBER_1>", "<ID_1>": "<ID_NUMBER_1>"}


def find_span(text, span, record_id):
    start = text.find(span)
    if start == -1:
        raise ValueError(f"{record_id}: span {span!r} not found in clean_input {text!r}")
    return start, start + len(span)


def convert(ex):
    clean = clean_text(ex["input"])
    counters = {}
    pii = []
    for p in ex["pii_spans"]:
        ptype = PII_TYPE_MAP.get(p["type"], p["type"])
        counters[ptype] = counters.get(ptype, 0) + 1
        start, end = find_span(clean, p["span"], ex["id"])
        pii.append({"type": ptype, "placeholder": f"<{ptype}_{counters[ptype]}>",
                    "text": p["span"], "start": start, "end": end})

    uncertainties = []
    for u in ex["uncertainty_spans"]:
        start, end = find_span(clean, u["span"], ex["id"])
        uncertainties.append({"span": u["span"], "start": start, "end": end,
                              "types": [u["type"]], "candidates": u.get("candidates", []),
                              "aleatoric": None, "epistemic": None, "human_ambiguous": None})

    sanitized = ex.get("normalized")
    if sanitized:
        for old, new in PLACEHOLDER_MAP.items():
            sanitized = sanitized.replace(old, new)

    return {
        "id": ex["id"],
        "schema_version": "0.2",
        "input": ex["input"],
        "clean_input": clean,
        "normalized_text": None,
        "sanitized_prompt": sanitized,
        "pii": pii,
        "preserved_entities": [],
        "uncertainties": uncertainties,
        "routing": ex["expected_routing"],
        "metadata": {
            "language": "banglish",
            "surface_form": "dialect" if ex["category"] == "DIALECT_MEANING" else "banglish",
            "source": "controlled_examples",
            "author": "Mashrafi (converted by Saber)",
            "category": ex["category"],
            "notes": ex.get("notes", ""),
            "placeholder_input": "<CONFIRMED_DIALECT_TERM>" in ex["input"],
            "label_source": "synthetic",
            "split": "none",
        },
    }


def main():
    with open(IN_PATH, encoding="utf-8") as f:
        examples = json.load(f)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(convert(ex), ensure_ascii=False) + "\n")
    print(f"Converted {len(examples)} examples -> {OUT_PATH}")


if __name__ == "__main__":
    main()
