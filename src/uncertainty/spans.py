"""Stage D (CPU): token scores -> span features a(s), e(s), d(s), types, candidates (§4.3-4.4).

Two kinds of spans:
  listed     - each pii / preserved_entities / ambiguous_spans entry of the pivot, scored over
               all tokens of its JSON subtree
  discovered - individual words of sanitized_prompt whose token scores are high

PROTOTYPE NOTE: the proposal maps (a, e, d) to calibrated scores with two logistic heads
fitted on the gold dev split (Eq. 4). Until gold labels exist, `aleatoric`/`epistemic`
are the empirical percentile of a(s)/e(s) within this run, and spans are flagged above a
quantile. Replace with the fitted heads + dev-tuned thresholds once labels exist.
"""
import bisect
import json
import logging
import re
from collections import defaultdict

from src.uncertainty.agreement import discordance, overlaps
from src.teachers.common import load_inputs, out_path, read_jsonl, write_jsonl
from src.uncertainty.silver import to_record

log = logging.getLogger(__name__)
PLACEHOLDER = re.compile(r"<[A-Z_]+_\d+>")
GROUP_OF = {"pii": "pii", "preserved_entities": "preserved_entities", "ambiguous_spans": "ambiguous_spans"}


def load_lexicon(path):
    if not path:
        return set()
    with open(path, encoding="utf-8") as f:
        return {w.strip().lower() for w in f if w.strip()}


def heuristic_type(text, group, lexicon):
    if group == "pii" or PLACEHOLDER.search(text):
        return "PII_BOUNDARY"
    if re.search(r"\d", text):
        return "NUMERIC_AMBIGUITY"
    if any(w in lexicon for w in re.findall(r"\w+", text.lower())):
        return "DIALECT_MEANING"
    return "TRANSLITERATION" if group == "sanitized_prompt" else "INTENT"


def tokens_in(offsets, start, end):
    return [i for i, (s, e) in enumerate(offsets) if s < end and e > start]


def stats(ts, idx):
    if not idx:
        return None
    arg = max(idx, key=lambda i: ts["H_total"][i])
    return {"a_raw": max(ts["A"][i] for i in idx), "e_raw": max(ts["E"][i] for i in idx),
            "h_raw": max(ts["H_total"][i] for i in idx), "peak_token": arg}


def candidates_for(group, entry, others):
    c = []
    if group == "pii":
        c = [entry["type"]] + [o["type"] for oj in others for o in oj["pii"] if overlaps(o["span"], entry["span"])]
    elif group == "preserved_entities":
        c = [entry["value"]] + [o["value"] for oj in others for o in oj["preserved_entities"] if o["type"] == entry["type"]]
    elif group == "ambiguous_spans":
        c = list(entry["candidates"]) + [x for oj in others for o in oj["ambiguous_spans"]
                                         if overlaps(o["span"], entry["span"]) for x in o["candidates"]]
    seen, out = set(), []
    for x in c:
        if x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


def ecdf(values):
    s = sorted(values)
    return lambda v: round(bisect.bisect_right(s, v) / len(s), 2) if s else 0.0


def run(cfg):
    scfg = cfg.get("spans", {})
    lexicon = load_lexicon(scfg.get("dialect_lexicon"))
    pivots = {p["id"]: p for p in read_jsonl(out_path(cfg, "pivots.jsonl"))}
    gens = defaultdict(list)
    for g in read_jsonl(out_path(cfg, "generations.jsonl")):
        if g["parse_ok"]:
            gens[g["id"]].append(g)

    records = []
    for ts in read_jsonl(out_path(cfg, "token_scores.jsonl")):
        p = pivots[ts["id"]]
        pj, y, offs = p["pivot_json"], p["y_star"], ts["offsets"]
        others = [g for g in gens[ts["id"]] if g["teacher_id"] != p["pivot_teacher"]]
        het = [g["parsed"] for g in others if g["kind"] == "heterogeneous"]
        d_set, d_source = (het, "heterogeneous") if het else ([g["parsed"] for g in others], "same_tokenizer")
        all_others = [g["parsed"] for g in others]

        spans = []
        for group in GROUP_OF:
            for i, entry in enumerate(pj[group]):
                leaves = [lf for lf in p["leaves"] if lf["path"][:2] == [group, i]]
                idx = sorted({t for lf in leaves for t in tokens_in(offs, lf["start"], lf["end"])})
                st = stats(ts, idx)
                if st is None:
                    continue  # truncated away
                text = entry.get("span") or entry.get("value")
                types = entry.get("types") or [heuristic_type(text, group, lexicon)]
                spans.append({"kind": "listed", "field": f"{group}[{i}]", "span": text, "types": types,
                              "candidates": candidates_for(group, entry, all_others),
                              "d": discordance(group, entry, d_set), "teacher_flagged": group == "ambiguous_spans",
                              **st, "peak_token_str": ts["tokens"][st["peak_token"]]})

        sp_leaf = next(lf for lf in p["leaves"] if lf["path"] == ["sanitized_prompt"])
        d_sp = discordance("sanitized_prompt", pj["sanitized_prompt"], d_set)
        for m in re.finditer(r"\S+", y[sp_leaf["start"]:sp_leaf["end"]]):
            s0 = sp_leaf["start"] + m.start()
            st = stats(ts, tokens_in(offs, s0, s0 + len(m.group())))
            if st is None:
                continue
            k = st["peak_token"]
            alts = [a.strip() for a in ts["alt_tokens"][k] if a.strip()]
            spans.append({"kind": "discovered", "field": "sanitized_prompt", "span": m.group(),
                          "types": [heuristic_type(m.group(), "sanitized_prompt", lexicon)],
                          "candidates": list(dict.fromkeys(alts)), "d": d_sp, "teacher_flagged": False,
                          **st, "peak_token_str": ts["tokens"][k]})
        records.append({"id": ts["id"], "text": p["text"], "pivot_teacher": p["pivot_teacher"],
                        "d_source": d_source, "n_teachers_compared": len(d_set),
                        "pivot_json": pj, "spans": spans})

    # prototype normalization + flagging (replace with Eq. 4 logistic heads later)
    q_listed, q_word = scfg.get("flag_quantile", 0.8), scfg.get("word_flag_quantile", 0.95)
    min_raw = scfg.get("min_raw_nats", 0.05)
    for kind, q in (("listed", q_listed), ("discovered", q_word)):
        pool = [s for r in records for s in r["spans"] if s["kind"] == kind]
        if not pool:
            continue
        fa, fe = ecdf([s["a_raw"] for s in pool]), ecdf([s["e_raw"] for s in pool])
        for s in pool:
            s["aleatoric"], s["epistemic"] = fa(s["a_raw"]), fe(s["e_raw"])
            s["flagged"] = ((s["aleatoric"] >= q and s["a_raw"] >= min_raw) or
                            (s["epistemic"] >= q and s["e_raw"] >= min_raw))

    write_jsonl(out_path(cfg, "spans.jsonl"), records)
    inputs = {x["id"]: x for x in load_inputs(cfg)}
    silver = [to_record(r, inputs.get(r["id"], {}), cfg.get("run_name", "")) for r in records]
    write_jsonl(out_path(cfg, "silver.jsonl"), silver)   # team schema v0.2, label_source "silver"
    log.info("spans: %d inputs, %d spans, %d flagged", len(records),
             sum(len(r["spans"]) for r in records), sum(s.get("flagged", False) for r in records for s in r["spans"]))
    return records


def report(cfg, records, gen_path, score_info):
    gens = read_jsonl(gen_path)
    per_t = defaultdict(list)
    for g in gens:
        per_t[g["teacher_id"]].append(g)
    teachers = {t: {"n": len(v), "parse_ok_rate": round(sum(g["parse_ok"] for g in v) / len(v), 3),
                    "schema_valid_rate": round(sum(g["valid"] for g in v) / len(v), 3),
                    "sec_per_item": round(sum(g["sec_per_item"] for g in v) / len(v), 2)} for t, v in per_t.items()}
    ts = read_jsonl(out_path(cfg, "token_scores.jsonl"))
    allA = [a for r in ts for a in r["A"]]
    allE = [e for r in ts for e in r["E"]]
    summary = {"teachers": teachers, "scoring": score_info, "n_inputs_scored": len(ts),
               "mean_A_nats": round(sum(allA) / max(len(allA), 1), 4),
               "mean_E_nats": round(sum(allE) / max(len(allE), 1), 4),
               "n_spans": sum(len(r["spans"]) for r in records),
               "n_flagged": sum(s.get("flagged", False) for r in records for s in r["spans"])}
    with open(out_path(cfg, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    lines = [f"# Teacher uncertainty run: {cfg.get('run_name', '')}", "", "```json", json.dumps(summary, indent=2), "```", ""]
    for r in records:
        lines += [f"## {r['id']}", f"**Input:** {r['text']}", "",
                  f"**Sanitized (pivot = {r['pivot_teacher']}):** {r['pivot_json']['sanitized_prompt']}", "",
                  "| span | kind | types | a_raw | e_raw | d | ale% | epi% | flag | candidates |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        top = sorted(r["spans"], key=lambda s: -(s["a_raw"] + s["e_raw"]))
        top = [s for s in top if s["kind"] == "listed"] + [s for s in top if s["kind"] == "discovered"][:5]
        for s in top:
            d = "-" if s["d"] is None else f"{s['d']:.2f}"
            lines.append(f"| {s['span'].replace('|', '/')} | {s['kind']} | {','.join(s['types'])} | {s['a_raw']:.3f} | "
                         f"{s['e_raw']:.3f} | {d} | {s.get('aleatoric', '')} | {s.get('epistemic', '')} | "
                         f"{'Y' if s.get('flagged') else ''} | {' / '.join(s['candidates'][:4]).replace('|', '/')} |")
        lines.append("")
    with open(out_path(cfg, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return summary
