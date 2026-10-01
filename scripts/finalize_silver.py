"""Make the REAL silver dataset from finished teacher runs (owner: Zarif). Run check_teacher_run.py first.

    PYTHONUTF8=1 python scripts/finalize_silver.py experiments/teacher
    PYTHONUTF8=1 python scripts/finalize_silver.py experiments/teacher/train experiments/teacher/dev --out-dir data/processed

For every split it collects silver.jsonl from the run folder(s) of that split (several shard folders
are merged), keeps a record only if it is safe and consistent, and writes
    <out-dir>/silver_<split>.jsonl      one schema v0.2 record per line
    <out-dir>/silver_manifest.json      counts, every drop reason, sha256 of each file, source runs

A record is dropped when:
    schema       - it fails the schema v0.2 validator
    wrong_split  - its metadata.split differs from the run folder's split
    pii_copied   - a raw PII value from the input appears in sanitized_prompt (the student would learn to leak)
    placeholders - placeholders in sanitized_prompt do not match its pii list
    duplicate_id - the id was already taken (e.g. two shards overlap); the first one is kept
Records are never edited: a fixed-up target would no longer be what the teachers said.

The output is checked again (validator must PASS; for train, build_student_data must accept every record)
and the script exits 1 if that fails. Then hand silver_train.jsonl (+ manifest) to Saber:
    python scripts/build_student_data.py data/processed/silver_train.jsonl data/processed/student_train.jsonl
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, OrderedDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.teachers.common import read_jsonl  # noqa: E402
from src.utils.jsonl_validate import validate_file  # noqa: E402

SPLITS = ("train", "dev", "test")
PLACEHOLDER = re.compile(r"<[A-Z_]+_\d+>")


def find_runs(paths):
    runs = []
    for p in paths:
        if os.path.exists(os.path.join(p, "silver.jsonl")):
            runs.append(p)
        elif os.path.isdir(p):
            runs += sorted(os.path.join(p, d) for d in os.listdir(p)
                           if os.path.exists(os.path.join(p, d, "silver.jsonl")))
    return runs


def split_of(run):
    """Folder name decides the split (train / dev / test, or e.g. train_shard0)."""
    name = os.path.basename(os.path.normpath(run))
    for s in SPLITS:
        if name == s or name.startswith(s + "_") or name.startswith(s + "-"):
            return s
    return None


def bad_lines(path):
    """1-based line numbers that the schema validator rejects."""
    rep = validate_file(path, "record")
    return {line for line, _, _ in rep.errors}


def drop_reason(r, split):
    if r.get("metadata", {}).get("split") != split:
        return "wrong_split"
    if any(len(p.get("text") or "") >= 4 and p["text"] in r["sanitized_prompt"] for p in r["pii"]):
        return "pii_copied"
    if set(PLACEHOLDER.findall(r["sanitized_prompt"])) != {p["placeholder"] for p in r["pii"]}:
        return "placeholders"
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def finalize(paths, out_dir):
    runs = find_runs(paths)
    if not runs:
        raise SystemExit(f"no run folder with silver.jsonl in {paths}")
    os.makedirs(out_dir, exist_ok=True)
    manifest = OrderedDict(created=datetime.datetime.now().isoformat(timespec="seconds"), git_commit=git_commit(),
                           note="label_source=silver: training only for split=train; dev/test = teacher "
                                "predictions for evaluation (RQ1), never for training", splits={})
    ok = True
    for split in SPLITS:
        split_runs = [r for r in runs if split_of(r) == split]
        if not split_runs:
            continue
        kept, seen, drops, per_run = [], set(), Counter(), {}
        for run in split_runs:
            path = os.path.join(run, "silver.jsonl")
            rejected = bad_lines(path)
            recs = read_jsonl(path)
            per_run[run] = len(recs)
            for i, r in enumerate(recs, 1):
                reason = "schema" if i in rejected else drop_reason(r, split)
                if reason is None and str(r["id"]) in seen:
                    reason = "duplicate_id"
                if reason:
                    drops[reason] += 1
                    continue
                seen.add(str(r["id"]))
                kept.append(r)
        out = os.path.join(out_dir, f"silver_{split}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # check the output again
        rep = validate_file(out, "record")
        accepted = None
        if split == "train":
            sys.path.insert(0, os.path.join(ROOT, "scripts"))
            from build_student_data import check_record
            accepted = sum(check_record(r) is None for r in kept)
        good = rep.ok and (accepted is None or accepted == len(kept)) and len(kept) > 0
        ok &= good
        manifest["splits"][split] = OrderedDict(
            file=out, records=len(kept), read=sum(per_run.values()), dropped=dict(drops), source_runs=per_run,
            sha256=sha256(out), validator="PASS" if rep.ok else "FAIL",
            build_student_data_accepts=accepted,
            records_with_pii=sum(bool(r["pii"]) for r in kept),
            records_with_uncertainty=sum(bool(r["uncertainties"]) for r in kept),
            uncertainty_types=dict(Counter(t for r in kept for u in r["uncertainties"] for t in u["types"])))
        print(f"{split:5}: kept {len(kept)} of {sum(per_run.values())}  dropped {dict(drops) or 0}  "
              f"validator {'PASS' if rep.ok else 'FAIL'}" + (f"  student-data gate {accepted}/{len(kept)}"
                                                            if accepted is not None else "") + f"  -> {out}")
    with open(os.path.join(out_dir, "silver_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"manifest -> {os.path.join(out_dir, 'silver_manifest.json')}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run folders, or a folder that contains them")
    ap.add_argument("--out-dir", default="data/processed")
    a = ap.parse_args()
    ok, _ = finalize(a.paths, a.out_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
