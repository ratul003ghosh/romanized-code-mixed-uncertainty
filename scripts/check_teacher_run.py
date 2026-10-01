"""Check a finished teacher run BEFORE making the real silver data (owner: Zarif).

    PYTHONUTF8=1 python scripts/check_teacher_run.py experiments/teacher            # every split folder in it
    PYTHONUTF8=1 python scripts/check_teacher_run.py experiments/teacher/train experiments/teacher/dev
    PYTHONUTF8=1 python scripts/check_teacher_run.py experiments/teacher --examples 5

For each run folder (one per split, as in docs/faculty-run.md) it checks:
  1. files      - every file the pipeline writes is there
  2. counts     - input -> generations -> pivots -> token scores -> spans -> silver: no ids lost, no duplicates
  3. split      - every silver record carries the folder's split; no id or text shared across splits
  4. teachers   - JSON parse / schema-valid rate per teacher, error kinds, likely truncated answers
  5. scores     - A + E = H(p_bar), E >= 0, array lengths, GPU sanity check (identical teachers -> E = 0)
  6. silver     - schema v0.2 validator, Saber's build_student_data gate, raw PII copied into the
                  sanitized prompt, placeholders that do not match the pii list, unlocated spans
and prints a few examples (input, teacher answers, pivot, uncertain spans) to read by eye.

Writes <root>/run_check.md and <root>/run_check.json. Exit code 0 = no FAIL, 1 = at least one FAIL.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.teachers.common import load_inputs, read_jsonl  # noqa: E402
from src.utils.jsonl_validate import validate_file  # noqa: E402

SPLITS = ("train", "dev", "test")
EXPECTED_FILES = ("generations.jsonl", "pivots.jsonl", "token_scores.jsonl", "spans.jsonl", "silver.jsonl",
                  "summary.json", "config_used.yaml")
PLACEHOLDER = re.compile(r"<[A-Z_]+_\d+>")
DECOMP_TOL = 2e-3          # A, E, H are stored rounded to 4 decimals
PARSE_WARN = 0.8           # warn when fewer than 80% of a teacher's answers parse


class Check:
    def __init__(self, name):
        self.name, self.items = name, []

    def add(self, level, what, detail=""):
        self.items.append({"level": level, "what": what, "detail": str(detail)})

    @property
    def level(self):
        levels = [i["level"] for i in self.items]
        return "FAIL" if "FAIL" in levels else ("WARN" if "WARN" in levels else "OK")


def find_runs(paths):
    runs = []
    for p in paths:
        if os.path.exists(os.path.join(p, "generations.jsonl")):
            runs.append(p)
        elif os.path.isdir(p):
            runs += sorted(os.path.join(p, d) for d in os.listdir(p)
                           if os.path.exists(os.path.join(p, d, "generations.jsonl")))
    return runs


def load_config(run):
    path = os.path.join(run, "config_used.yaml")
    if not os.path.exists(path):
        return {}
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ids_of(recs):
    return [str(r["id"]) for r in recs]


def dupes(ids):
    return sorted(i for i, n in Counter(ids).items() if n > 1)


def error_kind(msg):
    m = msg.lower()
    for key, label in (("copied", "PII copied into sanitized_prompt"), ("placeholder", "placeholder not <TYPE_N>"),
                       ("unknown pii type", "unknown PII type"), ("bad pii", "malformed pii entry"),
                       ("bad entity", "malformed entity"), ("without valid type", "uncertainty without valid type"),
                       ("missing", "missing field"), ("not a list", "field not a list")):
        if key in m:
            return label
    return "other"


def check_run(run, n_examples):
    split = os.path.basename(os.path.normpath(run))
    split = split if split in SPLITS else None
    cfg = load_config(run)
    out = {"run": run, "split": split, "checks": [], "examples": [], "stats": {}}
    c = {k: Check(k) for k in ("files", "counts", "split", "teachers", "scores", "silver")}
    out["checks"] = list(c.values())

    # 1. files
    for f in EXPECTED_FILES:
        c["files"].add("OK" if os.path.exists(os.path.join(run, f)) else "FAIL", f,
                       "" if os.path.exists(os.path.join(run, f)) else "missing")
    score_info = {}
    if os.path.exists(os.path.join(run, "score_info.json")):
        score_info = json.load(open(os.path.join(run, "score_info.json"), encoding="utf-8"))
    else:
        c["files"].add("WARN", "score_info.json", "missing: GPU sanity result unknown")

    gens = read_jsonl(os.path.join(run, "generations.jsonl"))
    pivots = read_jsonl(os.path.join(run, "pivots.jsonl"))
    scores = read_jsonl(os.path.join(run, "token_scores.jsonl"))
    spans = read_jsonl(os.path.join(run, "spans.jsonl"))
    silver = read_jsonl(os.path.join(run, "silver.jsonl"))

    # 2. counts
    inputs = []
    inp_path = cfg.get("input_file")
    if inp_path and os.path.exists(inp_path):
        inputs = load_inputs({k: cfg.get(k) for k in ("input_file", "limit", "shard")})
        c["counts"].add("OK", "inputs", f"{len(inputs)} from {inp_path}")
    else:
        c["counts"].add("WARN", "inputs", f"input file {inp_path!r} not found here; counts checked against generations")
    teachers = sorted({g["teacher_id"] for g in gens})
    expected = set(ids_of(inputs)) if inputs else {str(g["id"]) for g in gens}
    for t in teachers:
        got = [str(g["id"]) for g in gens if g["teacher_id"] == t]
        missing = expected - set(got)
        c["counts"].add("FAIL" if missing else "OK", f"generations {t}",
                        f"{len(got)} answers" + (f"; {len(missing)} inputs missing, e.g. {sorted(missing)[:3]}" if missing else ""))
        if dupes(got):
            c["counts"].add("FAIL", f"generations {t}", f"duplicate ids {dupes(got)[:3]}")
    ok_pivots = [p for p in pivots if p.get("pivot_ok")]
    failed = [p["id"] for p in pivots if not p.get("pivot_ok")]
    c["counts"].add("OK" if len(pivots) == len(expected) else "FAIL", "pivots",
                    f"{len(pivots)} records, {len(ok_pivots)} usable, {len(failed)} without any parseable answer")
    for name, recs in (("token_scores", scores), ("spans", spans), ("silver", silver)):
        got = set(ids_of(recs))
        missing = {str(p["id"]) for p in ok_pivots} - got
        extra = got - expected
        lvl = "FAIL" if (missing or extra or dupes(ids_of(recs))) else "OK"
        detail = f"{len(recs)} records"
        if missing:
            detail += f"; {len(missing)} usable pivots missing, e.g. {sorted(missing)[:3]}"
        if extra:
            detail += f"; {len(extra)} ids not in the input"
        if dupes(ids_of(recs)):
            detail += f"; duplicate ids {dupes(ids_of(recs))[:3]}"
        c["counts"].add(lvl, name, detail)
    out["stats"]["counts"] = {"inputs": len(expected), "usable_pivots": len(ok_pivots), "no_pivot": len(failed),
                              "silver": len(silver)}

    # 3. split
    in_split = {str(r["id"]): (r.get("split") or (r.get("metadata") or {}).get("split")) for r in inputs}
    got_splits = Counter(r.get("metadata", {}).get("split") for r in silver)
    c["split"].add("OK" if (split is None or set(got_splits) <= {split}) else "FAIL", "silver split labels",
                   dict(got_splits) if split is None else f"{dict(got_splits)} (folder = {split})")
    wrong = [r["id"] for r in silver if in_split.get(str(r["id"])) not in (None, r.get("metadata", {}).get("split"))]
    if inputs:
        c["split"].add("FAIL" if wrong else "OK", "split kept from input", f"{len(wrong)} records changed split" if wrong else "all kept")
    labels = Counter(r.get("metadata", {}).get("label_source") for r in silver)
    c["split"].add("OK" if set(labels) <= {"silver"} else "FAIL", "label_source", dict(labels))

    # 4. teachers
    tstats = {}
    for t in teachers:
        g = [x for x in gens if x["teacher_id"] == t]
        parse = sum(x["parse_ok"] for x in g) / max(len(g), 1)
        valid = sum(x["valid"] for x in g) / max(len(g), 1)
        trunc = sum(1 for x in g if not x["parse_ok"] and "}" not in (x.get("raw") or "")[-20:] and len(x.get("raw") or "") > 400)
        kinds = Counter(error_kind(e) for x in g for e in x.get("errors", []))
        secs = [x.get("sec_per_item", 0) for x in g]
        tstats[t] = {"n": len(g), "parse_rate": round(parse, 3), "schema_valid_rate": round(valid, 3),
                     "likely_truncated": trunc, "errors": dict(kinds.most_common(6)),
                     "sec_per_item": round(sum(secs) / max(len(secs), 1), 2)}
        lvl = "OK" if parse >= PARSE_WARN else "WARN"
        c["teachers"].add(lvl, t, f"parse {parse:.0%}, schema-valid {valid:.0%}, truncated {trunc}, "
                                  f"errors {dict(kinds.most_common(3))}")
    out["stats"]["teachers"] = tstats

    # 5. scores
    sanity = score_info.get("sanity", {})
    if sanity:
        c["scores"].add("OK" if sanity.get("passed") else "FAIL", "GPU sanity (identical teachers -> E = 0)",
                        f"max E = {sanity.get('max_E_identical')}")
    worst_resid, neg_e, bad_len, trunc = 0.0, 0, 0, 0
    for s in scores:
        n = s["n_tokens"]
        if not all(len(s[k]) == n for k in ("A", "E", "H_total", "offsets", "tokens")):
            bad_len += 1
            continue
        neg_e += sum(1 for e in s["E"] if e < -1e-6)
        worst_resid = max([worst_resid] + [abs(a + e - h) for a, e, h in zip(s["A"], s["E"], s["H_total"])])
        trunc += bool(s.get("truncated"))
    c["scores"].add("OK" if worst_resid <= DECOMP_TOL else "FAIL", "A + E = H(p_bar)", f"worst gap {worst_resid:.4f} nats")
    c["scores"].add("OK" if not neg_e else "FAIL", "E >= 0", f"{neg_e} negative values")
    c["scores"].add("OK" if not bad_len else "FAIL", "array lengths", f"{bad_len} records with mismatched arrays")
    c["scores"].add("OK" if not trunc else "WARN", "pivot truncated at max_pivot_tokens", f"{trunc} records")
    if scores:
        allA = [a for s in scores for a in s["A"]]
        allE = [e for s in scores for e in s["E"]]
        out["stats"]["mean_A"], out["stats"]["mean_E"] = round(sum(allA) / len(allA), 4), round(sum(allE) / len(allE), 4)
        c["scores"].add("OK", "mean A / mean E per token (nats)", f"{out['stats']['mean_A']} / {out['stats']['mean_E']}")
        if out["stats"]["mean_E"] == 0:
            c["scores"].add("WARN", "epistemic channel", "E is zero everywhere: the teachers never disagree")

    # 6. silver
    sil_path = os.path.join(run, "silver.jsonl")
    if os.path.exists(sil_path):
        rep = validate_file(sil_path, "record")
        c["silver"].add("OK" if rep.ok else "FAIL", "schema v0.2 validator",
                        f"{rep.valid}/{rep.lines} valid" + ("" if rep.ok else f"; {dict(rep.counts)}"))
    if split == "train":
        from scripts.build_student_data import check_record
        reasons = Counter(check_record(r) for r in silver)
        acc = reasons.pop(None, 0)
        c["silver"].add("OK" if not reasons else "WARN", "build_student_data gate",
                        f"{acc} accepted" + (f", rejected {dict(reasons)}" if reasons else ""))
    leaks = [r["id"] for r in silver if any(len(p.get("text") or "") >= 4 and p["text"] in r["sanitized_prompt"]
                                            for p in r["pii"])]
    c["silver"].add("OK" if not leaks else "WARN", "raw PII copied into sanitized_prompt",
                    f"{len(leaks)} records" + (f", e.g. {leaks[:3]} (finalize_silver.py drops them)" if leaks else ""))
    mism = [r["id"] for r in silver
            if set(PLACEHOLDER.findall(r["sanitized_prompt"])) != {p["placeholder"] for p in r["pii"]}]
    c["silver"].add("OK" if not mism else "WARN", "placeholders match the pii list",
                    f"{len(mism)} records differ" + (f", e.g. {mism[:3]} (dropped)" if mism else ""))
    try:
        from src.pii.arbitrate import enforce
        maybe = [r["id"] for r in silver if enforce(r["sanitized_prompt"], r["pii"])["leaked"]]
        c["silver"].add("OK" if not maybe else "WARN", "detector finds PII-like text in sanitized_prompt",
                        f"{len(maybe)} records to review by eye" + (f", e.g. {maybe[:3]}" if maybe else ""))
    except ImportError:
        pass
    unloc = sum(r.get("metadata", {}).get("unlocated_spans", 0) for r in silver)
    n_spans = sum(len(r["pii"]) + len(r["uncertainties"]) for r in silver)
    c["silver"].add("OK" if not n_spans or unloc / n_spans < 0.2 else "WARN", "spans located in clean_input",
                    f"{n_spans - unloc}/{n_spans}")
    types = Counter(t for r in silver for u in r["uncertainties"] for t in u["types"])
    flagged = [len(r["uncertainties"]) for r in silver]
    out["stats"]["uncertainty_types"] = dict(types)
    out["stats"]["flags_per_record"] = round(sum(flagged) / max(len(flagged), 1), 2)
    c["silver"].add("OK", "uncertainty types", dict(types.most_common()))

    # examples to read by eye: first record, most uncertain, one with PII
    by_id = {str(r["id"]): r for r in silver}
    picks = []
    if silver:
        picks = [silver[0], max(silver, key=lambda r: len(r["uncertainties"]))]
        picks += [r for r in silver if r["pii"]][:1]
    seen = []
    for r in picks:
        if r["id"] in seen:
            continue
        seen.append(r["id"])
        raws = {g["teacher_id"]: (g.get("raw") or "")[:400] for g in gens if str(g["id"]) == str(r["id"])}
        out["examples"].append({"id": r["id"], "input": r["clean_input"], "sanitized": r["sanitized_prompt"],
                                "pii": r["pii"], "uncertainties": r["uncertainties"][:4],
                                "pivot_teacher": r["metadata"].get("pivot_teacher"), "teacher_raw": raws})
        if len(out["examples"]) >= n_examples:
            break
    out["_by_id"] = by_id
    return out


def cross_split(results):
    """No id and no identical clean_input may appear in two splits (proposal §3.4)."""
    chk = Check("cross-split")
    by_split = {}
    for res in results:
        if res["split"]:
            by_split.setdefault(res["split"], {}).update(res["_by_id"])
    names = sorted(by_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ids = set(by_split[a]) & set(by_split[b])
            texts = {r["clean_input"] for r in by_split[a].values()} & {r["clean_input"] for r in by_split[b].values()}
            chk.add("FAIL" if ids else "OK", f"shared ids {a}/{b}", len(ids))
            chk.add("WARN" if texts else "OK", f"identical clean_input {a}/{b}", len(texts))
    return chk


def to_markdown(results, xs):
    lines = ["# Teacher run check", ""]
    for res in results:
        lines += [f"## {res['run']}  (split: {res['split']})", "", "| Check | Item | Level | Detail |", "|---|---|---|---|"]
        for ch in res["checks"]:
            for it in ch.items:
                lines.append(f"| {ch.name} | {it['what']} | {it['level']} | {it['detail'].replace('|', '/')} |")
        lines += ["", "### Examples", ""]
        for ex in res["examples"]:
            lines += [f"**{ex['id']}** (pivot: {ex['pivot_teacher']})", "", f"- input: `{ex['input']}`",
                      f"- sanitized: `{ex['sanitized']}`",
                      f"- pii: {[(p['type'], p['text'], p['placeholder']) for p in ex['pii']]}",
                      f"- uncertain: {[(u['span'], u['types'], u['aleatoric'], u['epistemic']) for u in ex['uncertainties']]}"]
            for t, raw in ex["teacher_raw"].items():
                lines.append(f"- {t}: `{raw.replace(chr(10), ' ')[:300]}`")
            lines.append("")
    if xs.items:
        lines += ["## Across splits", "", "| Item | Level | Detail |", "|---|---|---|"]
        lines += [f"| {it['what']} | {it['level']} | {it['detail']} |" for it in xs.items]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run folders, or a folder that contains them")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--out", default=None, help="report path without extension (default <first path>/run_check)")
    a = ap.parse_args()
    runs = find_runs(a.paths)
    if not runs:
        print("no run folder (with generations.jsonl) found in", a.paths)
        sys.exit(2)
    results = [check_run(r, a.examples) for r in runs]
    xs = cross_split(results)
    n_fail = sum(ch.level == "FAIL" for res in results for ch in res["checks"]) + (xs.level == "FAIL")
    n_warn = sum(ch.level == "WARN" for res in results for ch in res["checks"]) + (xs.level == "WARN")
    for res in results:
        print(f"\n== {res['run']} (split {res['split']})")
        for ch in res["checks"]:
            print(f"[{ch.level:4}] {ch.name}")
            for it in ch.items:
                if it["level"] != "OK":
                    print(f"       {it['level']}: {it['what']}: {it['detail']}")
    print(f"\n[{xs.level:4}] cross-split")
    base = a.out or os.path.join(a.paths[0] if os.path.isdir(a.paths[0]) else ".", "run_check")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(to_markdown(results, xs))
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"runs": [{"run": r["run"], "split": r["split"], "stats": r["stats"],
                             "checks": {ch.name: {"level": ch.level, "items": ch.items} for ch in r["checks"]}}
                            for r in results],
                   "cross_split": {"level": xs.level, "items": xs.items}, "fail": n_fail, "warn": n_warn},
                  f, ensure_ascii=False, indent=2)
    print(f"\nRESULT: {n_fail} FAIL, {n_warn} WARN  -> report {base}.md")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
