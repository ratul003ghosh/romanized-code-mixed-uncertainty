"""Stage B (CPU): choose the pivot y* per input (§4.2).

best_teacher: output of the configured teacher (fallback: first parse-ok same-tokenizer
              output, then any).
medoid:       the parse-ok output with the highest mean field agreement to all others
              (practical stand-in for field-wise majority that stays self-consistent).
"""
import logging
from collections import defaultdict

from src.uncertainty.agreement import json_agreement
from src.teachers.common import load_inputs, out_path, read_jsonl, write_jsonl
from src.teachers.output_schema import serialize_with_offsets

log = logging.getLogger(__name__)


def choose(recs, strategy, best):
    ok = [r for r in recs if r["parse_ok"]]
    if not ok:
        return None
    if strategy == "medoid" and len(ok) > 1:
        return max(ok, key=lambda r: sum(json_agreement(r["parsed"], o["parsed"]) for o in ok if o is not r))
    for cond in (lambda r: r["teacher_id"] == best and r["valid"],
                 lambda r: r["teacher_id"] == best,
                 lambda r: r["kind"] == "same_tokenizer" and r["valid"],
                 lambda r: r["valid"], lambda r: True):
        hit = [r for r in ok if cond(r)]
        if hit:
            return hit[0]


def run(cfg):
    pcfg = cfg.get("pivot", {})
    strategy, best = pcfg.get("strategy", "best_teacher"), pcfg.get("best_teacher", "same:balanced")
    by_id = defaultdict(list)
    for r in read_jsonl(out_path(cfg, "generations.jsonl")):
        by_id[r["id"]].append(r)
    out, n_fail = [], 0
    for x in load_inputs(cfg):
        r = choose(by_id[x["id"]], strategy, best)
        if r is None:
            n_fail += 1
            out.append({"id": x["id"], "text": x["text"], "pivot_ok": False})
            continue
        y, leaves = serialize_with_offsets(r["parsed"])
        out.append({"id": x["id"], "text": x["text"], "pivot_ok": True, "pivot_teacher": r["teacher_id"],
                    "pivot_strategy": strategy, "pivot_json": r["parsed"], "y_star": y, "leaves": leaves})
    path = out_path(cfg, "pivots.jsonl")
    write_jsonl(path, out)
    log.info("pivots: %d ok, %d failed (no parseable teacher output)", len(out) - n_fail, n_fail)
    return path
