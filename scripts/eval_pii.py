"""Evaluate the PII detector on schema v0.2 records that have gold `pii` spans.

    python scripts/eval_pii.py data/synthetic/pii_cases.jsonl [--threshold 0.5] [--show-errors]

Metrics (proposal §6.1): span precision/recall/F1 (exact and IoU >= 0.5, type must match),
leakage rate (gold PII spans with at least one character left unmasked, any type counts as masked),
over-masking (preserved entities such as amounts that got masked), per type and per regime.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pii.masking import sanitize  # noqa: E402


def iou(a, b):
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union else 0.0


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 1.0
    r = tp / n_gold if n_gold else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(records, threshold=0.5):
    c = defaultdict(float)
    per_type = defaultdict(lambda: defaultdict(float))
    per_regime = defaultdict(lambda: defaultdict(float))
    errors = []
    for r in records:
        text = r.get("clean_input") or r["input"]
        pred = sanitize(text, threshold)["pii"]
        gold = r["pii"]
        regime = r.get("metadata", {}).get("regime", "unknown")
        covered = [False] * len(text)
        for p in pred:
            for i in range(p["start"], p["end"]):
                covered[i] = True
        c["pred"] += len(pred)
        c["gold"] += len(gold)
        used_exact, used_iou = set(), set()
        for g in gold:
            leaked = not all(covered[g["start"]:g["end"]])
            ex = next((j for j, p in enumerate(pred) if j not in used_exact and p["type"] == g["type"]
                       and (p["start"], p["end"]) == (g["start"], g["end"])), None)
            io = next((j for j, p in enumerate(pred) if j not in used_iou and p["type"] == g["type"]
                       and iou(p, g) >= 0.5), None)
            if ex is not None:
                used_exact.add(ex)
            if io is not None:
                used_iou.add(io)
            for bucket in (c, per_type[g["type"]], per_regime[regime]):
                bucket["n"] += 1
                bucket["exact"] += ex is not None
                bucket["iou"] += io is not None
                bucket["leaked"] += leaked
            if leaked or io is None:
                errors.append({"id": r["id"], "gold": g, "leaked": leaked})
        c["tp_exact"] += len(used_exact)
        c["tp_iou"] += len(used_iou)
        for p in pred:
            if not any(iou(p, g) > 0 for g in gold):
                errors.append({"id": r["id"], "false_positive": p})
        for e in r.get("preserved_entities", []):
            if "start" in e:
                c["entities"] += 1
                c["entities_masked"] += any(covered[e["start"]:e["end"]])
    ep, er, ef = prf(c["tp_exact"], c["pred"], c["gold"])
    ip, ir, i_f = prf(c["tp_iou"], c["pred"], c["gold"])
    summary = {
        "records": len(records), "gold_spans": int(c["gold"]), "pred_spans": int(c["pred"]),
        "exact": {"precision": round(ep, 3), "recall": round(er, 3), "f1": round(ef, 3)},
        "iou_0.5": {"precision": round(ip, 3), "recall": round(ir, 3), "f1": round(i_f, 3)},
        "leakage_rate": round(c["leaked"] / c["gold"], 3) if c["gold"] else 0.0,
        "over_masking_rate": round(c["entities_masked"] / c["entities"], 3) if c["entities"] else 0.0,
        "per_type": {t: {"n": int(v["n"]), "recall_iou": round(v["iou"] / v["n"], 3),
                         "leakage": round(v["leaked"] / v["n"], 3)} for t, v in sorted(per_type.items())},
        "per_regime": {t: {"n": int(v["n"]), "recall_iou": round(v["iou"] / v["n"], 3),
                           "leakage": round(v["leaked"] / v["n"], 3)} for t, v in sorted(per_regime.items())},
        "threshold": threshold,
    }
    return summary, errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--show-errors", action="store_true")
    a = ap.parse_args()
    with open(a.path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    summary, errors = evaluate(recs, a.threshold)
    print(json.dumps(summary, indent=2))
    if a.show_errors:
        for e in errors:
            print(json.dumps(e, ensure_ascii=False))
