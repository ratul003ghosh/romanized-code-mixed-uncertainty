"""Check a teacher output JSONL for completeness and compare sequential vs parallel runs.

Usage:
  python compare_teacher_outputs.py PARALLEL.jsonl --inputs teacher_inputs.jsonl --teachers 6
  python compare_teacher_outputs.py PARALLEL.jsonl --baseline SEQUENTIAL.jsonl --tol 1e-2

Key assumed from the team's description: each row has "id" and "teacher_id".
Exit code 0 = all checks passed, 1 = problems found.
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict


def load(path):
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
                print(f"  BAD JSON line {n} in {path} (truncated by a killed worker?)")
    return rows, bad


def close(a, b, tol, path="", diffs=None):
    """Recursive compare: floats within tol, everything else exact."""
    diffs = diffs if diffs is not None else []
    if isinstance(a, float) or isinstance(b, float):
        try:
            if not math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol):
                diffs.append((path, a, b))
        except (TypeError, ValueError):
            diffs.append((path, a, b))
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                diffs.append((f"{path}.{k}", a.get(k, "<missing>"), b.get(k, "<missing>")))
            else:
                close(a[k], b[k], tol, f"{path}.{k}", diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((f"{path}[len]", len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                close(x, y, tol, f"{path}[{i}]", diffs)
    elif a != b:
        diffs.append((path, a, b))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--inputs", help="teacher inputs JSONL (expects an 'id' field)")
    ap.add_argument("--teachers", type=int, help="expected teachers per input (6)")
    ap.add_argument("--baseline", help="sequential-run JSONL to compare against")
    ap.add_argument("--tol", type=float, default=1e-2, help="float tolerance (4-bit + batching differs slightly)")
    a = ap.parse_args()
    ok = True

    rows, bad = load(a.output)
    print(f"{a.output}: {len(rows)} rows, {bad} unparsable lines")
    ok &= bad == 0

    keys = Counter((r.get("id"), r.get("teacher_id")) for r in rows)
    dups = [k for k, c in keys.items() if c > 1]
    print(f"duplicate (id, teacher_id): {len(dups)}")
    ok &= not dups

    per_id = defaultdict(set)
    for r in rows:
        per_id[r.get("id")].add(r.get("teacher_id"))
    print(f"distinct ids: {len(per_id)}")

    if a.teachers:
        short = {i: len(t) for i, t in per_id.items() if len(t) != a.teachers}
        print(f"ids without exactly {a.teachers} teachers: {len(short)}  (e.g. {list(short.items())[:3]})")
        ok &= not short

    if a.inputs:
        want = {json.loads(l).get("id") for l in open(a.inputs, encoding="utf-8") if l.strip()}
        missing, extra = want - set(per_id), set(per_id) - want
        print(f"inputs: {len(want)} | missing from output: {len(missing)} | not in inputs: {len(extra)}")
        ok &= not missing and not extra

    if a.baseline:
        base, bbad = load(a.baseline)
        bm = {(r.get("id"), r.get("teacher_id")): r for r in base}
        pm = {(r.get("id"), r.get("teacher_id")): r for r in rows}
        print(f"baseline rows: {len(bm)} | parallel rows: {len(pm)}")
        only_b, only_p = set(bm) - set(pm), set(pm) - set(bm)
        print(f"only in baseline: {len(only_b)} | only in parallel: {len(only_p)}")
        ok &= not only_b and not only_p
        shown = 0
        for k in sorted(set(bm) & set(pm), key=str):
            d = close(bm[k], pm[k], a.tol)
            if d and shown < 10:
                print(f"DIFF {k}: {d[:3]}")
                shown += 1
        total = sum(1 for k in set(bm) & set(pm) if close(bm[k], pm[k], a.tol))
        print(f"rows differing beyond tol={a.tol}: {total}")
        ok &= total == 0

    print("RESULT:", "PASS" if ok else "PROBLEMS FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
