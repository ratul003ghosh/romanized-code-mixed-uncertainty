import json
import os


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_jsonl(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_inputs(cfg):
    """Accepts {"id", "text"} lines or schema v0.2 records (clean_input preferred, since PII
    offsets refer to it); extra keys are carried along."""
    out = []
    for i, r in enumerate(read_jsonl(cfg["input_file"])):
        text = r.get("clean_input") or r.get("text") or r.get("input") or r.get("prompt")
        if not text:
            continue
        out.append({**r, "id": str(r.get("id", i)), "text": text})
    lim = cfg.get("limit")
    return out[:lim] if lim else out


def teacher_groups(cfg):
    """Each group = one model load. Same-tokenizer group: several variants (prompt or LoRA)."""
    s = cfg["same_tokenizer"]
    groups = [{"alias": "same", "kind": "same_tokenizer", "model": s["model"],
               "dtype": s.get("dtype", "bfloat16"), "load_in_4bit": s.get("load_in_4bit", False),
               "variants": s["variants"]}]
    for h in cfg.get("heterogeneous") or []:
        groups.append({"alias": "het_" + h["alias"], "kind": "heterogeneous", "model": h["model"],
                       "dtype": h.get("dtype", "bfloat16"), "load_in_4bit": h.get("load_in_4bit", False),
                       "variants": [h.get("variant", "balanced")]})
    return groups


def teacher_id(group, variant):
    return f"{group['alias']}:{variant}"


def out_path(cfg, name):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    return os.path.join(cfg["output_dir"], name)
