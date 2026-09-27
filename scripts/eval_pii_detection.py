"""Measure the rule-based PII detector on PII-injected records (proposal v2, 6.1).

  python scripts/eval_pii_detection.py silver/inputs_template.jsonl silver/inputs_carrier.jsonl \
      --out docs/results/pii_detection_v0.md

Because the PII was injected, every record carries exact gold spans with their formatting regime,
so the detector can be scored per regime and per type:

  found      share of gold spans the detector returns with exactly the same start and end
  type ok    share of found spans that also get the right type
  leaked     share of gold spans with at least one character left unmasked (any type counts as
             masked); this is the leakage rate of proposal 6.1
  precision  share of predicted spans that exactly match a gold span
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pii.detect import detect  # noqa: E402


def evaluate(records):
    by = defaultdict(lambda: {"n": 0, "found": 0, "type_ok": 0, "leaked": 0})
    n_pred = n_pred_exact = 0
    for r in records:
        text = r["clean_input"]
        pred = detect(text)
        n_pred += len(pred)
        pred_pos = {(p["start"], p["end"]): p["type"] for p in pred}
        covered = [False] * len(text)
        for p in pred:
            for i in range(p["start"], p["end"]):
                covered[i] = True
        gold_pos = {(g["start"], g["end"]) for g in r["pii"]}
        n_pred_exact += sum(1 for k in pred_pos if k in gold_pos)
        for g in r["pii"]:
            key = (g["start"], g["end"])
            found = key in pred_pos
            leaked = not all(covered[g["start"]:g["end"]])
            for group in (("all", "all"), ("regime", g.get("regime", "unknown")), ("type", g["type"])):
                s = by[group]
                s["n"] += 1
                s["found"] += found
                s["type_ok"] += found and pred_pos[key] == g["type"]
                s["leaked"] += leaked
    return by, n_pred, n_pred_exact


def table(by, kind, order=None):
    keys = [k for (grp, k) in by if grp == kind]
    keys = [k for k in (order or sorted(keys)) if k in keys]
    lines = [f"| {kind} | spans | found | type ok | leaked |", "|---|---:|---:|---:|---:|"]
    for k in keys:
        s = by[(kind, k)]
        type_ok = s["type_ok"] / s["found"] if s["found"] else 0.0
        lines.append(f"| {k} | {s['n']:,} | {s['found'] / s['n']:.1%} | {type_ok:.1%} | {s['leaked'] / s['n']:.1%} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    records = [json.loads(line) for path in a.inputs for line in path.open(encoding="utf-8")]
    by, n_pred, n_pred_exact = evaluate(records)
    s = by[("all", "all")]
    report = "\n\n".join([
        f"Records: {len(records):,} · gold PII spans: {s['n']:,} · predicted spans: {n_pred:,}",
        f"**Overall:** found {s['found'] / s['n']:.1%} · leaked {s['leaked'] / s['n']:.1%} · "
        f"precision {n_pred_exact / max(n_pred, 1):.1%}",
        "### By formatting regime",
        table(by, "regime", ["canonical", "spoken", "perturbed"]),
        "### By PII type",
        table(by, "type"),
    ])
    print(report)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
