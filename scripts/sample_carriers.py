"""Filter possible real PII out of carrier text and draw a balanced sample (proposal v2, 3.2 step 1).

  python scripts/sample_carriers.py --inp /kaggle/working/carriers/merged_silver.jsonl \
      --out /kaggle/working/carriers/carrier_sample_v0.jsonl --quota BanglaTLit=20000,BIDWESH=5000

1. Real-PII filter: scraped text may already contain real phone numbers, IDs or e-mails.
   Such a line would enter training with *unlabeled* PII and teach the student to leave PII
   unmasked, so it is dropped. The pattern is deliberately strict; it will be replaced by the
   regex PII detector (baseline B1 / runtime rule 1) once that exists.
2. Balanced sample: a fixed quota per source, spread evenly over the dialects inside a
   source, so small dialect sources are not swamped by BanglaTLit.

With a single dialect group per source this reproduces the notebook sampling used for
silver inputs v0 exactly (same seed, same order of random calls).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SUSPECT_PII = re.compile(
    r"\d[\d\s\-.]{6,}\d"          # 7+ digits, possibly with separators (phones, NIDs, accounts)
    r"|[০-৯]{7,}"                 # the same in Bangla digits
    r"|[\w.+-]+@[\w-]+\.\w+"      # e-mail addresses
    r"|\+?880"                    # Bangladesh country code
)


def parse_quota(text: str) -> dict[str, int]:
    out = {}
    for part in text.split(","):
        if part.strip():
            name, n = part.split("=")
            out[name.strip()] = int(n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True, help="merged_silver.jsonl from convert_sources.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quota", default="BanglaTLit=20000,BIDWESH=5000",
                    help="per-source quota, e.g. BanglaTLit=20000,BIDWESH=5000")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    quota = parse_quota(a.quota)
    kept, dropped = [], Counter()
    with a.inp.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if SUSPECT_PII.search(r["text"]):
                dropped[r["source"]] += 1
            else:
                kept.append(r)
    print(f"Real-PII filter: kept {len(kept)}, dropped {sum(dropped.values())} {dict(dropped)}")

    by_source: dict[str, dict[str | None, list]] = defaultdict(lambda: defaultdict(list))
    for r in kept:
        by_source[r["source"]][r.get("dialect")].append(r)

    random.seed(a.seed)
    sample, summary = [], {}
    for source, groups in by_source.items():
        for rows in groups.values():
            random.shuffle(rows)
        want = quota.get(source, 0)
        picked: list = []
        iters = [iter(rows) for rows in groups.values()]
        while len(picked) < want and iters:              # round-robin over dialects
            alive = []
            for it in iters:
                if len(picked) >= want:
                    break
                row = next(it, None)
                if row is not None:
                    picked.append(row)
                    alive.append(it)
            iters = alive
        sample += picked
        summary[source] = {"picked": len(picked), "available": sum(len(g) for g in groups.values()),
                           "by_dialect": dict(Counter(r.get("dialect") for r in picked))}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sample)} carriers to {a.out}")
    for source, s in summary.items():
        print(f"  {source:12s} picked {s['picked']:>6} of {s['available']:>7}  {s['by_dialect']}")


if __name__ == "__main__":
    main()
