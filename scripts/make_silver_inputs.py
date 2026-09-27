"""Generate PII-injected input records in schema v0.2 (proposal v2, section 3.2).

Template mode (fully synthetic, label_source "synthetic"):
  python scripts/make_silver_inputs.py --n-template 5000 --seed 1 --out silver/inputs_template.jsonl

Carrier mode (real sentences + synthetic PII, label_source "none", i.e. input to the teachers):
  python scripts/make_silver_inputs.py --carrier carriers/carrier_sample.jsonl --seed 2 \
      --out silver/inputs_carrier.jsonl

Carrier files come from convert_sources.py / sample_carriers.py (JSONL with "text"); a plain text
file with one sentence per line also works. Every record is checked against schema v0.2 with
src/pii_inject/v02.py; the run stops if any record fails.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pii_inject.build import make_carrier_record, make_template_record  # noqa: E402
from src.pii_inject.v02 import check_record  # noqa: E402


def read_carrier(path: Path, source: str, surface_form: str, dialect: str | None):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                text = row.get("text") or row.get("input")
                if not text:
                    continue
                yield {"text": text, "id": row.get("id", i), "source": row.get("source", source),
                       "language": row.get("language"), "surface_form": row.get("surface_form", surface_form),
                       "dialect": row.get("dialect", dialect), "split": row.get("split", "train")}
            else:
                yield {"text": line, "id": i, "source": source,
                       "surface_form": surface_form, "dialect": dialect, "split": "train"}


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-template", type=int, default=0)
    ap.add_argument("--carrier", type=Path)
    ap.add_argument("--carrier-source", default="carrier")
    ap.add_argument("--carrier-surface-form", default="banglish",
                    help="schema v0.2 surface_form for plain-text carriers")
    ap.add_argument("--carrier-dialect", default=None)
    ap.add_argument("--n-carrier", type=int, default=0, help="0 = use every carrier line")
    ap.add_argument("--clauses", type=int, default=1, help="PII clauses per carrier sentence")
    ap.add_argument("--word-noise", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--member", default=None)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    meta = {k: v for k, v in {"commit": git_commit(), "member": a.member, "shard": a.shard}.items() if v}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()

    with out.open("w", encoding="utf-8") as f:
        def write(rec):
            errors = check_record(rec)
            if errors:
                raise SystemExit(f"{rec['id']} breaks schema v0.2: {errors}")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["records"] += 1
            for p in rec["pii"]:
                stats[f"pii:{p['type']}"] += 1
                stats[f"regime:{p['regime']}"] += 1

        for i in range(a.n_template):
            write(make_template_record(i, rng, seed=a.seed, p_word_noise=a.word_noise, meta_extra=meta))

        if a.carrier:
            carriers = list(read_carrier(a.carrier, a.carrier_source, a.carrier_surface_form,
                                         a.carrier_dialect))
            rng.shuffle(carriers)
            if a.n_carrier:
                carriers = carriers[:a.n_carrier]
            for i, c in enumerate(carriers):
                write(make_carrier_record(i, c, rng, seed=a.seed, n_clauses=a.clauses, meta_extra=meta))

    print(f"Wrote {stats['records']} records to {out} (all pass the schema v0.2 check)")
    for k in sorted(k for k in stats if k != "records"):
        print(f"  {k:24s} {stats[k]}")


if __name__ == "__main__":
    main()
