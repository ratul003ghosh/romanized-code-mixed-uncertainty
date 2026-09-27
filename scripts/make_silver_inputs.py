"""Generate silver *input* records with synthetic PII (proposal v2, section 3.2).

Template-only (fully synthetic):
  python scripts/make_silver_inputs.py --n-template 5000 --out data/silver/inputs_v0.jsonl

With real carrier sentences (one sentence per line, or JSONL with a "text" field):
  python scripts/make_silver_inputs.py --carrier /kaggle/input/.../banglatlit.txt \
      --carrier-source BanglaTLit --carrier-surface-form romanized \
      --n-carrier 20000 --out data/silver/inputs_v0.jsonl

Every record carries meta.generator, meta.seed and, if given, meta.commit/member/shard
(CONTRIBUTING.md). A PII-only reference output is written with --with-reference.
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
from src.pii_inject.build import make_carrier_record, make_template_record, pii_reference  # noqa: E402


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
                       "surface_form": row.get("surface_form", surface_form),
                       "dialect": row.get("dialect", dialect), "split": row.get("split", "silver")}
            else:
                yield {"text": line, "id": i, "source": source,
                       "surface_form": surface_form, "dialect": dialect, "split": "silver"}


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
    ap.add_argument("--carrier-surface-form", default="unknown")
    ap.add_argument("--carrier-dialect", default=None)
    ap.add_argument("--n-carrier", type=int, default=0, help="0 = use every carrier line")
    ap.add_argument("--clauses", type=int, default=1, help="PII clauses per carrier sentence")
    ap.add_argument("--word-noise", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--member", default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--with-reference", action="store_true",
                    help="add a schema-valid PII-only reference under 'pii_reference'")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    meta = {k: v for k, v in {"commit": git_commit(), "member": a.member, "shard": a.shard}.items() if v}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()

    with out.open("w", encoding="utf-8") as f:
        def write(rec):
            if a.with_reference:
                rec["pii_reference"] = pii_reference(rec)
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
                write(make_carrier_record(i, c, rng, seed=a.seed, n_clauses=a.clauses,
                                          p_word_noise=0.0, meta_extra=meta))

    print(f"Wrote {stats['records']} records to {out}")
    for k in sorted(k for k in stats if k != "records"):
        print(f"  {k:24s} {stats[k]}")


if __name__ == "__main__":
    main()
