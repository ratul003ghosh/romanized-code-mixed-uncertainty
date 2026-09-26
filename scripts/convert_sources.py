"""Convert raw corpora into carrier JSONL for make_silver_inputs.py (proposal v2, 3.1-3.4).

  python scripts/convert_sources.py --raw /kaggle/working/raw_sources --out /kaggle/working/carriers

Writes one file per source and split:
  banglatlit_gold_pool.jsonl   BanglaTLit test  -> candidates for the human gold set (never silver)
  banglatlit_dev_pool.jsonl    BanglaTLit val   -> dev / calibration candidates
  banglatlit_silver.jsonl      BanglaTLit train + BanglaTLit-PT, minus anything in the pools
  comilingua_hinglish_pool.jsonl  COMI-LINGUA MT test, Romanized Hindi -> Hinglish gold candidates
  comilingua_silver.jsonl      COMI-LINGUA MT train, Romanized Hindi
  bidwesh_silver.jsonl         BIDWESH dialect sentences, non-hate rows only (section 3.1)
  merged_silver.jsonl          with --merge: all *_silver files, de-duplicated across sources and
                               against every gold/dev pool

Cleaning is minimal on purpose (NFC, zero-width chars, whitespace, URLs); spelling noise
such as "5oo" or "upddate" is kept, because the model has to handle it.
The split assignment is a default; confirm it with the team and record it in docs/decisions.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
URL = re.compile(r"https?://\S+|www\.\S+")
SPACES = re.compile(r"\s+")
BENGALI = re.compile(r"[\u0980-\u09FF]")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = ZERO_WIDTH.sub("", text)
    text = URL.sub("<URL>", text)
    return SPACES.sub(" ", text).strip()


def surface_form(text: str, lang: str) -> str:
    latin = len(LATIN.findall(text))
    native = len((BENGALI if lang == "bn" else DEVANAGARI).findall(text))
    if native == 0:
        return "romanized" if lang == "bn" else "hinglish"
    if latin == 0:
        return "native_script"
    return "mixed_script"


def key(text: str) -> str:
    """Normalization used for duplicate detection (case, punctuation and spacing ignored)."""
    return re.sub(r"[^\w]+", "", text.lower())


JUNK = {"#NAME?", "#VALUE!", "#REF!", "nan", "NaN", "None"}


def ok_length(text: str, lo: int = 3, hi: int = 60) -> bool:
    return text not in JUNK and lo <= len(text.split()) <= hi


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


class Writer:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.handles, self.stats = {}, Counter()

    def write(self, name: str, row: dict):
        if name not in self.handles:
            self.handles[name] = (self.out_dir / f"{name}.jsonl").open("w", encoding="utf-8")
        self.handles[name].write(json.dumps(row, ensure_ascii=False) + "\n")
        self.stats[name] += 1

    def close(self):
        for h in self.handles.values():
            h.close()


def convert_banglatlit(raw: Path, w: Writer, stats: Counter):
    d = raw / "BanglaTLit" / "data"
    protected: set[str] = set()
    for split_file, name, split in [("BanglaTLiT_test.csv", "banglatlit_gold_pool", "gold_pool"),
                                    ("BanglaTLiT_val.csv", "banglatlit_dev_pool", "dev_pool")]:
        for r in read_csv(d / split_file):
            t = clean(r["text_transliterated"])
            if not ok_length(t):
                stats["btl_short_or_long"] += 1
                continue
            protected.add(key(t))
            w.write(name, {"id": f"btl-{r['id']}", "text": t, "source": "BanglaTLit",
                           "surface_form": surface_form(t, "bn"), "dialect": None, "split": split,
                           "reference_bengali": clean(r["text_bengali"])})

    seen: set[str] = set()

    def silver(text: str, rid: str, ref: str | None, origin: str):
        t = clean(text)
        k = key(t)
        if not ok_length(t):
            stats["btl_short_or_long"] += 1
        elif k in protected:
            stats["btl_removed_overlap_with_pools"] += 1
        elif k in seen:
            stats["btl_duplicate"] += 1
        else:
            seen.add(k)
            row = {"id": rid, "text": t, "source": "BanglaTLit", "origin": origin,
                   "surface_form": surface_form(t, "bn"), "dialect": None, "split": "silver"}
            if ref:   # train.csv covers all 245k PT rows; only 42.7k pairs are human-verified
                row["reference_bengali_unverified"] = clean(ref)
            w.write("banglatlit_silver", row)

    for r in read_csv(d / "BanglaTLit_train.csv"):
        silver(r["text_transliterated"], f"btl-{r['id']}", r["text_bengali"], "train")
    pt = d / "BanglaTLit-PT.txt"
    if pt.exists():
        with pt.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    silver(line, f"btlpt-{i:06d}", None, "pt")


def convert_comilingua(raw: Path, w: Writer, stats: Counter):
    d = raw / "COMI-LINGUA"
    rh_cols = ["Annotator_1_RH_translation", "annotator2_RH_translation", "annotator3_RH_translation"]
    protected: set[str] = set()
    for split_file, name, split in [("MT_test.csv", "comilingua_hinglish_pool", "gold_pool"),
                                    ("MT_train.csv", "comilingua_silver", "silver")]:
        path = d / split_file
        if not path.exists():
            continue
        seen: set[str] = set()
        for i, r in enumerate(read_csv(path)):
            text = next((r[c] for c in rh_cols if r.get(c) and r[c].strip()), None)
            if not text:
                stats["cl_no_romanized"] += 1
                continue
            t = clean(text)
            k = key(t)
            if not ok_length(t):
                stats["cl_short_or_long"] += 1
            elif k in seen:
                stats["cl_duplicate"] += 1
            elif split == "silver" and k in protected:
                stats["cl_removed_overlap_with_pool"] += 1
            else:
                seen.add(k)
                if split == "gold_pool":
                    protected.add(k)
                w.write(name, {"id": f"cl-{split}-{i:06d}", "text": t, "source": "COMI-LINGUA",
                               "surface_form": surface_form(t, "hi"), "dialect": None, "split": split,
                               "register": "news_translation"})


DIALECTS = {"noakhali": "Noakhali", "chittagong": "Chittagong", "chattogram": "Chittagong",
            "ctg": "Chittagong", "barishal": "Barishal", "barisal": "Barishal"}
NON_HATE = {"0", "0.0", "non-hate", "non hate", "nonhate", "not hate", "no", "false", "non_hate"}


def _read_table(path: Path) -> list[dict]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        import pandas as pd
        return pd.read_excel(path, dtype=str).fillna("").to_dict("records")
    return list(read_csv(path))


def _dialect_of(name: str) -> str | None:
    n = name.lower()
    return next((d for k, d in DIALECTS.items() if k in n), None)


def convert_bidwesh(raw: Path, w: Writer, stats: Counter, text_col=None, dialect_col=None,
                    label_col=None, keep_hate=False):
    """BIDWESH layout is detected from column names and printed; override with --bidwesh-* flags.

    wide layout: one text column per dialect (e.g. 'Noakhali', 'Chittagong', 'Barishal')
    long layout: a text column plus a dialect/region column
    Hate rows are dropped unless keep_hate is set (proposal 3.1: toxic content filtered).
    """
    files = sorted(p for p in (raw / "BIDWESH").rglob("*")
                   if p.suffix.lower() in {".csv", ".xlsx", ".xls"})
    if not files:
        print("BIDWESH: no .csv/.xlsx files under", raw / "BIDWESH")
        return
    seen: set[str] = set()
    for path in files:
        rows = _read_table(path)
        if not rows:
            continue
        cols = list(rows[0].keys())
        lab = label_col or next((c for c in cols if "hate" in c.lower() and "type" not in c.lower()
                                 and "target" not in c.lower()), None) \
            or next((c for c in cols if c.lower() in {"label", "class"}), None)
        wide = {c: _dialect_of(c) for c in cols if _dialect_of(c) and c != dialect_col}
        if text_col:
            layout = [(text_col, None)]
        elif wide:
            layout = list(wide.items())
        else:
            tcol = next((c for c in cols if any(k in c.lower() for k in
                                                ("text", "sentence", "comment", "translat"))), None)
            if not tcol:
                raise SystemExit(f"BIDWESH {path.name}: cannot find a text column in {cols}; "
                                 f"use --bidwesh-text-col")
            layout = [(tcol, None)]
        dcol = dialect_col or next((c for c in cols if any(k in c.lower() for k in
                                                           ("dialect", "region"))), None)
        print(f"BIDWESH {path.name}: text={[c for c, _ in layout]} dialect_col={dcol} label={lab}")
        if lab is None and not keep_hate:
            raise SystemExit(f"BIDWESH {path.name}: no hate label column found in {cols}; "
                             f"use --bidwesh-label-col (hate rows must be filtered)")
        for i, r in enumerate(rows):
            if lab and not keep_hate and str(r.get(lab, "")).strip().lower() not in NON_HATE:
                stats["bid_dropped_hate"] += 1
                continue
            for col, dialect in layout:
                t = clean(r.get(col, ""))
                k = key(t)
                if not ok_length(t):
                    stats["bid_short_or_long"] += 1
                elif k in seen:
                    stats["bid_duplicate"] += 1
                else:
                    seen.add(k)
                    d = dialect or (_dialect_of(str(r.get(dcol, ""))) if dcol else None)
                    w.write("bidwesh_silver", {
                        "id": f"bid-{path.stem[:20]}-{i:05d}-{(d or 'x')[:3].lower()}", "text": t,
                        "source": "BIDWESH", "origin": "BD-SHS",   # BanglaVeilGuard quarantine, 3.4
                        "surface_form": surface_form(t, "bn"), "dialect": d, "split": "silver"})


def merge_silver(out_dir: Path, stats: Counter) -> int:
    """Union of all *_silver.jsonl, de-duplicated across sources and against every pool."""
    protected = set()
    for pool in out_dir.glob("*_pool.jsonl"):
        with pool.open(encoding="utf-8") as f:
            protected |= {key(json.loads(l)["text"]) for l in f}
    seen, n, by_source = set(), 0, Counter()
    with (out_dir / "merged_silver.jsonl").open("w", encoding="utf-8") as out:
        for part in sorted(out_dir.glob("*_silver.jsonl")):
            if part.name == "merged_silver.jsonl":
                continue
            with part.open(encoding="utf-8") as f:
                for line in f:
                    k = key(json.loads(line)["text"])
                    if k in protected:
                        stats["merge_removed_overlap_with_pools"] += 1
                    elif k in seen:
                        stats["merge_cross_source_duplicate"] += 1
                    else:
                        seen.add(k)
                        out.write(line)
                        n += 1
                        by_source[part.stem] += 1
    print("merged_silver.jsonl composition:", dict(by_source))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sources", default="banglatlit,comilingua,bidwesh",
                    help="comma-separated subset to convert")
    ap.add_argument("--merge", action="store_true", help="also write merged_silver.jsonl")
    ap.add_argument("--bidwesh-text-col")
    ap.add_argument("--bidwesh-dialect-col")
    ap.add_argument("--bidwesh-label-col")
    ap.add_argument("--bidwesh-keep-hate", action="store_true", help="not recommended")
    a = ap.parse_args()
    use = set(a.sources.split(","))
    a.out.mkdir(parents=True, exist_ok=True)
    w, stats = Writer(a.out), Counter()
    try:
        if "banglatlit" in use and (a.raw / "BanglaTLit" / "data").exists():
            convert_banglatlit(a.raw, w, stats)
        if "comilingua" in use and (a.raw / "COMI-LINGUA").exists():
            convert_comilingua(a.raw, w, stats)
        if "bidwesh" in use and (a.raw / "BIDWESH").exists():
            convert_bidwesh(a.raw, w, stats, a.bidwesh_text_col, a.bidwesh_dialect_col,
                            a.bidwesh_label_col, a.bidwesh_keep_hate)
    finally:
        w.close()
    if a.merge:
        w.stats["merged_silver"] = merge_silver(a.out, stats)
    print("Written:")
    for name, n in sorted(w.stats.items()):
        print(f"  {name + '.jsonl':34s} {n:>8}")
    print("Dropped:")
    for k, n in sorted(stats.items()):
        print(f"  {k:34s} {n:>8}")


if __name__ == "__main__":
    main()
