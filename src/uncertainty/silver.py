"""Teacher output -> team schema v0.2 records (docs/annotation-schema.md).

One flat record per input that had a usable pivot. These records feed
scripts/build_student_data.py (Saber), which keeps only label_source == "silver" AND
split == "train" for training. Dev/test records keep their split, so they are used for
evaluation only (RQ1/K1 teacher scores) and never for training.

Positions (`start`, `end`) refer to `clean_input`, the text the teachers received. A span the
teacher wrote that cannot be found verbatim in `clean_input` (paraphrased or reformatted) gets
start = end = null and is counted in `metadata.unlocated_spans`.
"""
from __future__ import annotations

SCHEMA_VERSION = "0.2"


def _meta(inp: dict, key: str, default=None):
    return inp.get(key, (inp.get("metadata") or {}).get(key, default))


def _locator(text: str):
    """Find spans left to right, so a value that appears twice gets both occurrences."""
    next_from: dict[str, int] = {}

    def locate(span: str):
        if not span:
            return None, None
        i = text.find(span, next_from.get(span, 0))
        if i < 0:
            i = text.find(span)                  # repeated mention of an earlier span
        if i < 0:
            return None, None
        next_from[span] = i + len(span)
        return i, i + len(span)
    return locate


def to_record(r: dict, inp: dict, run_name: str = "") -> dict:
    """r: one record of spans.jsonl; inp: the matching input line (carries id, split, source...)."""
    pj, clean = r["pivot_json"], r["text"]
    locate = _locator(clean)
    unlocated = 0

    pii = []
    for e in pj["pii"]:
        s, t = locate(e["span"])
        unlocated += s is None
        pii.append({"type": e["type"], "placeholder": e["placeholder"], "text": e["span"], "start": s, "end": t})

    uncertainties = []
    for sp in r["spans"]:
        if sp["kind"] != "listed" or not sp.get("flagged"):
            continue                              # output-side words stay in spans.jsonl (diagnostics)
        if sp["field"].startswith("preserved_entities"):
            s = t = None                          # value is the rewritten form (e.g. "500 BDT"), not input text
        else:
            s, t = _locator(clean)(sp["span"])
        unlocated += s is None
        uncertainties.append({"span": sp["span"], "start": s, "end": t, "types": sp["types"],
                              "candidates": sp["candidates"], "aleatoric": sp["aleatoric"],
                              "epistemic": sp["epistemic"], "human_ambiguous": None})

    return {
        "id": r["id"],
        "schema_version": SCHEMA_VERSION,
        "input": inp.get("input") or clean,
        "clean_input": clean,
        "normalized_text": None,
        "sanitized_prompt": pj["sanitized_prompt"],
        "pii": pii,
        "preserved_entities": pj["preserved_entities"],
        "uncertainties": uncertainties,
        "routing": None,                          # set by the decision layer (§4.6)
        "metadata": {
            "language": _meta(inp, "language", "banglish"),
            "surface_form": _meta(inp, "surface_form", "banglish"),
            "source": _meta(inp, "source"),
            "label_source": "silver",
            "split": _meta(inp, "split", "none"),
            "pivot_teacher": r["pivot_teacher"],
            "teacher_run": run_name,
            "unlocated_spans": unlocated,
        },
    }
