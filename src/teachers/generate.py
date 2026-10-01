"""Stage A (GPU): every teacher generates the full JSON for every input (greedy)."""
import logging
import os
import time

import torch

from src.teachers import models
from src.teachers.common import append_jsonl, load_inputs, out_path, read_jsonl, teacher_groups, teacher_id
from src.teachers.prompts import build_messages
from src.teachers.output_schema import extract_json, validate

log = logging.getLogger(__name__)


def run(cfg):
    inputs = load_inputs(cfg)
    
    shard_info = cfg.get("worker_shard")
    shard_suffix = ""
    if shard_info:
        s_idx, s_total = map(int, shard_info.split("/"))
        inputs = [x for i, x in enumerate(inputs) if i % s_total == s_idx]
        shard_suffix = f"_shard{s_idx}"
        
    path = out_path(cfg, f"generations{shard_suffix}.jsonl")
    
    # Read done from shard file, and also from main file if resuming a mixed run
    done = {(r["id"], r["teacher_id"]) for r in read_jsonl(path)}
    if shard_info and os.path.exists(out_path(cfg, "generations.jsonl")):
        done.update((r["id"], r["teacher_id"]) for r in read_jsonl(out_path(cfg, "generations.jsonl")))
        
    gcfg = cfg.get("generation", {})
    bs, max_new = gcfg.get("batch_size", 4), gcfg.get("max_new_tokens", 512)
    adapters = cfg["same_tokenizer"].get("adapters")

    for group in teacher_groups(cfg):
        for variant in group["variants"]:
            tid = teacher_id(group, variant)
            todo = [x for x in inputs if (x["id"], tid) not in done]
            if not todo:
                continue
            if "_loaded" not in group:
                tok, model = models.load(group["model"], group["dtype"], group["load_in_4bit"])
                if group["kind"] == "same_tokenizer":
                    models.setup_adapters(model, adapters)
                group["_loaded"] = (tok, model)
            tok, model = group["_loaded"]
            if group["kind"] == "same_tokenizer":
                models.activate(model, variant, adapters)
            log.info("generating %s on %d inputs", tid, len(todo))
            t_start = time.time()
            for b in range(0, len(todo), bs):
                batch = todo[b:b + bs]
                texts = [models.chat_text(tok, build_messages(variant, x["text"])) for x in batch]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                t0 = time.time()
                with torch.inference_mode():
                    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                         temperature=None, top_p=None, top_k=None,
                                         pad_token_id=tok.pad_token_id)
                dt = (time.time() - t0) / len(batch)
                decoded = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
                for x, raw in zip(batch, decoded):
                    parsed, errors = validate(extract_json(raw))
                    append_jsonl(path, {"id": x["id"], "teacher_id": tid, "model": group["model"],
                                        "kind": group["kind"], "variant": variant, "raw": raw,
                                        "parsed": parsed, "valid": parsed is not None and not errors,
                                        "parse_ok": parsed is not None, "errors": errors,
                                        "sec_per_item": round(dt, 3)})
                done_n = min(b + bs, len(todo))
                left = (time.time() - t_start) / done_n * (len(todo) - done_n)
                log.info("%s: %d/%d done, %.1f s per input, about %.0f min left for this teacher",
                         tid, done_n, len(todo), dt, left / 60)
        if "_loaded" in group:
            models.unload(group.pop("_loaded")[1])
    log.info("generation done; peak GPU mem %s GB", models.gpu_mem_gb())
    return path
