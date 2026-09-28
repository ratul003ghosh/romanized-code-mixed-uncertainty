"""Banglish real-PII evaluation slice from BanglishRev (proposal Section 3.1).

    PYTHONUTF8=1 python scripts/build_banglishrev_inputs.py                        # downloads ~2 GB JSON
    PYTHONUTF8=1 python scripts/build_banglishrev_inputs.py --json-path "reviews v1.json"

Source: BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset (NeurIPS 2024 D&B),
CC-BY-NC-SA-4.0, file "reviews v1.json" only (the 10k+ image files are never downloaded).

The proposal uses BanglishRev ONLY as an evaluation slice because it contains organic PII:
  * every record is split "test" (never training data) with metadata.eval_only = true
  * only "Review Content" is kept; Buyer ID, seller reply, dates, likes and images are dropped
  * output files stay in data/processed/ (git-ignored) and must never be committed or shared publicly
  * an ethics OK from the team lead is required before this slice is used (proposal Section 3.1)

Reviews are in Bangla script, English, and Banglish. A transparent word-list rule keeps Banglish only:
  - any Bangla-script character          -> dropped (bangla_script)
  - fewer than --min-words words         -> dropped (too_short)
  - at least 2 Banglish words from BANGLISH_WORDS and at least as many Banglish as English
    function words                       -> kept as Banglish; otherwise dropped (english_or_unclear)
This is a heuristic, not a language identifier; its counts are reported, and a private spot check
of kept reviews is recommended.

Sampling (seeded): all reviews where the existing PII detector finds PII are candidates for the
PII part (--n-pii), the rest for the random part (--n-random), so the slice really tests PII masking.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pii.detector import detect  # noqa: E402
from src.pii.masking import sanitize  # noqa: E402
from src.preprocessing.clean import clean_text  # noqa: E402
from src.utils.jsonl_validate import BAD_CHARS_RE, DEFAULT_MAX_CHARS  # noqa: E402

REPO = "BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset"
FILENAME = "reviews v1.json"
BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
WORD_RE = re.compile(r"[a-z]+")
BANGLISH_WORDS = set("""
ami amar amake amr apni apnar apnader tumi tomar se tar tara oder ache ase asey achhe nai nei na hoy hoi
hoyse hoise hoyeche hoiche hocche hosse hobe holo hoilo korsi korchi korlam korlo kore korte korben korbo
koren dilam dise dilo dey deya dia diye pelam paisi peyechi pai paini pelam valo bhalo vallo balo valoi
kharap onek khub kintu abar jonno theke kichu sob shob ekta ekdom taka dam pore age keno kemon kotha bole
bolse chilo silo cilo janen jani dhonnobad thik vai bhai apu sathe shathe ta ti gula gulo eta oita eita ei
oi tai tahole mone lage laglo pacchi pachhi chai chaile kom beshi besi ache aro emon jeta seta hoye hoya
kaj korche korse pabo pawa pawar dekhe dekhte deklam mal maal dilen dile nibo nilam kinsi kinlam kinechi
khob kinto naa hoyni hoynai pailam paichi paici dicche disi deyni dey nai valoi vlo vallage bhallage onk
""".split())
ENGLISH_WORDS = set("""
the is are was were very good product and it this i my not but with for of to in nice quality really
great bad have has had so also be will would can could its that as on at an you your they them we our
""".split())


def classify(text, min_words):
    if BANGLA_RE.search(text):
        return "bangla_script"
    words = WORD_RE.findall(text.lower())
    if len(words) < min_words:
        return "too_short"
    bn = sum(w in BANGLISH_WORDS for w in words)
    en = sum(w in ENGLISH_WORDS for w in words)
    return "banglish" if bn >= 2 and bn >= en else "english_or_unclear"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-pii", type=int, default=100, help="reviews with detected PII to include")
    ap.add_argument("--n-random", type=int, default=400, help="other Banglish reviews to include")
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    ap.add_argument("--json-path", default=None, help="local copy of 'reviews v1.json' (skips download)")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    path = args.json_path
    if not path:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(REPO, FILENAME, repo_type="dataset")
    products = json.load(open(path, encoding="utf-8"))

    stats = {"dataset": f"{REPO} / {FILENAME}", "seed": args.seed, "min_words": args.min_words,
             "products": len(products), "reviews_total": 0, "reviews_with_text": 0,
             "language_rule": Counter(), "dropped": Counter(), "banglish_candidates": 0,
             "banglish_with_pii_detected": 0, "final": {}, "pii_types_in_final": Counter(),
             "masking_failures": 0}
    seen = set()
    with_pii, without_pii = [], []
    for p in products:
        category = p.get("Root Category") or p.get("Category")
        for rev in p.get("Reviews") or []:
            stats["reviews_total"] += 1
            text = rev.get("Review Content")
            if not isinstance(text, str) or not text.strip():
                continue
            stats["reviews_with_text"] += 1
            clean = clean_text(text)
            label = classify(clean, args.min_words)
            stats["language_rule"][label] += 1
            if label != "banglish":
                continue
            if len(clean) > args.max_chars:
                stats["dropped"]["too_long"] += 1
                continue
            if BAD_CHARS_RE.search(clean):
                stats["dropped"]["broken_unicode"] += 1
                continue
            norm = " ".join(clean.lower().split())
            if norm in seen:
                stats["dropped"]["duplicate"] += 1
                continue
            seen.add(norm)
            item = (text, clean, category)
            (with_pii if detect(clean) else without_pii).append(item)
    stats["banglish_candidates"] = len(with_pii) + len(without_pii)
    stats["banglish_with_pii_detected"] = len(with_pii)

    rng = random.Random(args.seed)
    chosen = rng.sample(with_pii, min(args.n_pii, len(with_pii))) + \
        rng.sample(without_pii, min(args.n_random, len(without_pii)))
    records = []
    for i, (text, clean, category) in enumerate(chosen, start=1):
        found = detect(clean)
        if found:
            masked = sanitize(clean)["masked"]
            if any(f["text"] in masked for f in found):
                stats["masking_failures"] += 1
            stats["pii_types_in_final"].update(f["type"] for f in found)
        records.append({
            "id": f"BR_TE_{i:06d}",
            "schema_version": "0.2",
            "input": text,
            "clean_input": clean,
            "normalized_text": None, "sanitized_prompt": None,
            "pii": [], "preserved_entities": [], "uncertainties": [], "routing": None,
            "metadata": {
                "language": "banglish", "surface_form": "banglish", "source": "BanglishRev",
                "product_category": category,
                "sample_group": "pii_detected" if found else "random",
                "pii_detected": [{"type": f["type"], "text": f["text"], "start": f["start"], "end": f["end"]}
                                 for f in found],
                "eval_only": True, "contains_organic_pii": True,
                "label_source": "none", "split": "test",
            },
        })
    stats["final"] = {"test": len(records),
                      "pii_detected_group": sum(r["metadata"]["sample_group"] == "pii_detected" for r in records),
                      "random_group": sum(r["metadata"]["sample_group"] == "random" for r in records)}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "teacher_inputs_banglishrev_test.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()}
    with open(os.path.join(args.out_dir, "teacher_inputs_banglishrev_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nWrote teacher_inputs_banglishrev_test.jsonl to {args.out_dir} "
          "(PRIVATE: organic PII, never commit or share publicly)")


if __name__ == "__main__":
    main()
