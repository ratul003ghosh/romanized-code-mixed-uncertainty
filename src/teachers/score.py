"""Stage C (GPU): teacher-forced scoring of y* by the same-tokenizer teachers (§4.2-4.3).

For each input and each teacher variant k:
    logits of y* given (teacher-k prompt, x), restricted to the real vocabulary
Then A_t, E_t, H(p_bar_t), per-teacher NLL, and top-k of the entropy-weighted
KD target (Eq. 6) are saved per pivot token.
"""
import logging
import os
import time

import torch

from src.teachers import models
from src.teachers.common import append_jsonl, out_path, read_jsonl
from src.teachers.prompts import build_messages
from src.uncertainty.decomposition import decompose

log = logging.getLogger(__name__)


def _r(t, nd=4):
    return [round(v, nd) for v in t.tolist()] if t.dim() == 1 else [_r(row, nd) for row in t]


@torch.inference_mode()
def teacher_logp(model, tok, variant, adapters, text, y_ids, V):
    models.activate(model, variant, adapters)
    prompt_ids = models.chat_ids(tok, build_messages(variant, text))
    ids = torch.tensor([prompt_ids + y_ids], device=model.device)
    P, T = len(prompt_ids), len(y_ids)
    logits = model(input_ids=ids).logits[0, P - 1:P - 1 + T, :V].float()
    return torch.log_softmax(logits, dim=-1)


def sanity_identical_teachers(model, tok, adapters, variant, pivot, V):
    """Scoring the same teacher twice must give E_t ~ 0 everywhere."""
    y_ids = tok(pivot["y_star"], add_special_tokens=False).input_ids[:64]
    lp = teacher_logp(model, tok, variant, adapters, pivot["text"], y_ids, V)
    st = decompose(torch.stack([lp, lp]), y_ids)
    return {"max_E_identical": float(st["E"].max()), "passed": float(st["E"].max()) < 1e-3}


def run(cfg):
    s, sc = cfg["same_tokenizer"], cfg.get("scoring", {})
    variants, adapters = s["variants"], s.get("adapters")
    max_tok, tau = sc.get("max_pivot_tokens", 768), sc.get("tau", 1.0)
    top_k, alt_k = sc.get("kd_top_k", 10), sc.get("alt_k", 3)
    shard_info = cfg.get("worker_shard")
    shard_suffix = ""
    if shard_info:
        s_idx, s_total = map(int, shard_info.split("/"))
        shard_suffix = f"_shard{s_idx}"
        
    path = out_path(cfg, f"token_scores{shard_suffix}.jsonl")
    done = {r["id"] for r in read_jsonl(path)}
    if shard_info and os.path.exists(out_path(cfg, "token_scores.jsonl")):
        done.update(r["id"] for r in read_jsonl(out_path(cfg, "token_scores.jsonl")))
        
    pivots = [p for p in read_jsonl(out_path(cfg, "pivots.jsonl")) if p["pivot_ok"]]
    if shard_info:
        pivots = [p for i, p in enumerate(pivots) if i % s_total == s_idx]
    
    pivots = [p for p in pivots if p["id"] not in done]
    if not pivots:
        log.info("nothing to score")
        return path, {}

    tok, model = models.load(s["model"], s.get("dtype", "bfloat16"), s.get("load_in_4bit", False))
    models.setup_adapters(model, adapters)
    V, n_emb = models.shared_vocab(tok, model)
    info = {"vocab_tokenizer": len(tok), "embedding_rows": n_emb, "vocab_used": V}
    log.info("vocab: %s", info)
    if sc.get("sanity_check", True):
        info["sanity"] = sanity_identical_teachers(model, tok, adapters, variants[0], pivots[0], V)
        log.info("sanity (identical teachers -> E=0): %s", info["sanity"])

    times, worst_resid = [], 0.0
    for p in pivots:
        enc = tok(p["y_star"], add_special_tokens=False, return_offsets_mapping=True)
        y_ids, offs = enc.input_ids[:max_tok], [list(o) for o in enc.offset_mapping[:max_tok]]
        t0 = time.time()
        logp = torch.stack([teacher_logp(model, tok, v, adapters, p["text"], y_ids, V) for v in variants])
        st = decompose(logp, y_ids, tau=tau, top_k=top_k, alt_k=alt_k)
        del logp
        times.append(time.time() - t0)
        worst_resid = min(worst_resid, st["jsd_min_before_clamp"])
        append_jsonl(path, {
            "id": p["id"], "variants": variants, "n_tokens": len(y_ids), "truncated": len(enc.input_ids) > max_tok,
            "token_ids": y_ids, "offsets": offs,
            "tokens": [tok.decode([i]) for i in y_ids],
            "A": _r(st["A"]), "E": _r(st["E"]), "H_total": _r(st["H_total"]),
            "H_k": _r(st["H_k"]), "nll_k": _r(st["nll_k"]), "p_bar_y": _r(st["p_bar_y"]),
            "kd_top_ids": st["kd_top_ids"].tolist(), "kd_top_probs": _r(st["kd_top_probs"]),
            "alt_tokens": [[tok.decode([i]) for i in row] for row in st["alt_ids"].tolist()],
            "alt_probs": _r(st["alt_probs"], 3),
            "jsd_min_before_clamp": st["jsd_min_before_clamp"],
        })
    models.unload(model)
    info.update({"n_scored": len(pivots), "sec_per_item": round(sum(times) / len(times), 3),
                 "worst_jsd_before_clamp": worst_resid, "peak_gpu_gb": models.gpu_mem_gb()})
    log.info("scoring done: %s", info)
    return path, info
