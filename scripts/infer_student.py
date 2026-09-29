"""Run the trained student on inputs and save schema v0.2 predictions with confidence scores.

    python scripts/infer_student.py --checkpoint experiments/student/run1 \
        --input data/processed/teacher_inputs_test.jsonl \
        --output experiments/student/run1/predictions_test.jsonl

--checkpoint is the --output-dir of scripts/train_student.py (it holds adapter/ and run_metadata.json).
--input accepts schema v0.2 records (clean_input is used) or student-format lines (the user message).

Speed (docs/student-inference-speed.md): the LoRA adapter is merged into a 16-bit model and prompts
are generated in batches; greedy decoding, as in proposal Section 4.6.

Confidence (for baseline B5, "student's own confidence", proposal Section 5), per prediction:
  mean_token_logprob, min_token_logprob, sequence_logprob  - log-probability of the chosen tokens
  mean_entropy, max_entropy                                - entropy (nats) of the next-token distribution
  num_tokens
--save-token-scores also stores the per-token lists (tokens, logprobs, entropies) for span-level use.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.build_student_data import SYSTEM_PROMPT  # noqa: E402

TARGET_KEYS = ("sanitized_prompt", "pii", "preserved_entities", "uncertainties", "routing")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="training output dir (adapter/ + run_metadata.json)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="only the first N inputs")
    ap.add_argument("--save-token-scores", action="store_true")
    ap.add_argument("--no-merge", action="store_true", help="keep the adapter separate (slower)")
    return ap.parse_args()


def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def read_inputs(path, limit):
    rows = []
    for n, line in enumerate(open(path, encoding="utf-8"), start=1):
        if not line.strip():
            continue
        r = json.loads(line)
        if "messages" in r:
            text = r["messages"][1]["content"]
        else:
            text = r.get("clean_input") or r.get("text") or r.get("input")
        if not isinstance(text, str) or not text.strip():
            fail(f"line {n}: no input text")
        rows.append({"id": str(r.get("id", n)), "text": text, "source": r})
    return rows[:limit] if limit else rows


def locate(text, span, used):
    """Character position of span in text (left to right, so repeats get separate positions)."""
    if not isinstance(span, str) or not span:
        return None, None
    i = text.find(span, used.get(span, 0))
    if i < 0:
        i = text.find(span)
    if i < 0:
        return None, None
    used[span] = i + len(span)
    return i, i + len(span)


def to_prediction(row, raw, conf, checkpoint):
    text = row["text"]
    try:
        out = json.loads(raw)
        valid = isinstance(out, dict) and all(k in out for k in TARGET_KEYS)
    except json.JSONDecodeError:
        out, valid = None, False
    rec = {"id": row["id"], "schema_version": "0.2",
           "input": row["source"].get("input", text), "clean_input": text, "normalized_text": None,
           "sanitized_prompt": None, "pii": [], "preserved_entities": [], "uncertainties": [], "routing": None}
    if valid:
        used = {}
        rec["sanitized_prompt"] = out.get("sanitized_prompt")
        for p in out.get("pii") or []:
            if isinstance(p, dict):
                s, e = locate(text, p.get("span"), used)
                rec["pii"].append({"type": p.get("type"), "placeholder": p.get("placeholder"),
                                   "text": p.get("span"), "start": s, "end": e})
        rec["preserved_entities"] = out.get("preserved_entities") or []
        for u in out.get("uncertainties") or []:
            if isinstance(u, dict):
                s, e = locate(text, u.get("span"), {})
                rec["uncertainties"].append({"span": u.get("span"), "start": s, "end": e,
                                             "types": u.get("types"), "candidates": u.get("candidates", []),
                                             "aleatoric": u.get("aleatoric"), "epistemic": u.get("epistemic"),
                                             "human_ambiguous": None})
        rec["routing"] = out.get("routing")
    meta = row["source"].get("metadata") or {}
    rec["metadata"] = {"language": meta.get("language", "banglish"), "surface_form": meta.get("surface_form", "banglish"),
                       "source": meta.get("source"), "split": meta.get("split", "none"),
                       "label_source": "prediction", "model": "student", "checkpoint": checkpoint,
                       "json_valid": valid, "raw_output": raw, "confidence": conf}
    return rec


def main():
    args = parse_args()
    meta_path = os.path.join(args.checkpoint, "run_metadata.json")
    adapter_dir = os.path.join(args.checkpoint, "adapter")
    if not os.path.exists(meta_path) or not os.path.isdir(adapter_dir):
        fail(f"{args.checkpoint} is not a train_student.py output (needs run_metadata.json and adapter/)")
    if not os.path.exists(args.input):
        fail(f"input not found: {args.input}")
    run_meta = json.load(open(meta_path, encoding="utf-8"))
    system_prompt = run_meta.get("system_prompt", SYSTEM_PROMPT)
    rows = read_inputs(args.input, args.limit)
    print(f"{len(rows)} inputs; base model {run_meta['base_model']}; research_result={run_meta.get('research_result')}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(run_meta["base_model"], dtype=dtype)
    model.to("cuda" if use_cuda else "cpu")
    model = PeftModel.from_pretrained(model, adapter_dir)
    if not args.no_merge:
        model = model.merge_and_unload()
    model.eval()
    device = next(model.parameters()).device

    preds, start = [], time.time()
    for b in range(0, len(rows), args.batch_size):
        batch = rows[b:b + args.batch_size]
        prompts = [tokenizer.apply_chat_template([{"role": "system", "content": system_prompt},
                                                  {"role": "user", "content": r["text"]}],
                                                 add_generation_prompt=True, tokenize=False) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 return_dict_in_generate=True, output_logits=True,
                                 pad_token_id=tokenizer.pad_token_id)
        new_tokens = gen.sequences[:, enc["input_ids"].shape[1]:]
        # one step at a time, so memory stays small (batch x vocab, not batch x steps x vocab)
        step_lp = torch.zeros(new_tokens.shape, dtype=torch.float32)
        step_ent = torch.zeros(new_tokens.shape, dtype=torch.float32)
        for t, scores in enumerate(gen.logits):          # raw logits, before any logits processor
            lp = torch.log_softmax(scores.float(), dim=-1)
            step_lp[:, t] = lp.gather(1, new_tokens[:, t:t + 1]).squeeze(1).cpu()
            step_ent[:, t] = (-(lp.exp() * lp).sum(dim=-1)).cpu()
        for i, row in enumerate(batch):
            toks, lps, ents = [], [], []
            for t in range(new_tokens.shape[1]):
                tid = int(new_tokens[i, t])
                if t > 0 and int(new_tokens[i, t - 1]) == tokenizer.eos_token_id:
                    break                                   # everything after the end token is padding
                toks.append(tokenizer.decode([tid]))
                lps.append(float(step_lp[i, t]))
                ents.append(float(step_ent[i, t]))
            raw = tokenizer.decode(new_tokens[i], skip_special_tokens=True).strip()
            conf = {"num_tokens": len(lps),
                    "mean_token_logprob": round(sum(lps) / len(lps), 4) if lps else None,
                    "min_token_logprob": round(min(lps), 4) if lps else None,
                    "sequence_logprob": round(sum(lps), 4) if lps else None,
                    "mean_entropy": round(sum(ents) / len(ents), 4) if ents else None,
                    "max_entropy": round(max(ents), 4) if ents else None}
            if args.save_token_scores:
                conf["tokens"], conf["token_logprobs"], conf["token_entropies"] = toks, lps, ents
            preds.append(to_prediction(row, raw, conf, args.checkpoint))
        print(f"{min(b + args.batch_size, len(rows))}/{len(rows)} done, {time.time() - start:.0f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    valid = sum(p["metadata"]["json_valid"] for p in preds)
    secs = time.time() - start
    print(f"saved {len(preds)} predictions to {args.output}")
    print(f"valid JSON: {valid}/{len(preds)}; {secs:.0f}s total, {secs / max(len(preds), 1):.2f}s per input")


if __name__ == "__main__":
    main()
