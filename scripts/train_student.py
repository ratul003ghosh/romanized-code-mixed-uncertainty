"""Train the student (Qwen2.5-1.5B-Instruct) with QLoRA on student training pairs.

Real run (after the teacher run, on silver data built by scripts/build_student_data.py):
    python scripts/train_student.py --data data/processed/student_train.jsonl \
        --output-dir experiments/student/run1 --epochs 1

Smoke test (checks the script only, on synthetic data; NOT a research result):
    python scripts/train_student.py --data data/synthetic/student_smoke.jsonl \
        --output-dir experiments/student/smoke --smoke

Method (same as notebooks/03_student_training.ipynb, now reusable):
  4-bit NF4 base model (bitsandbytes) + LoRA adapters (peft); loss on the assistant JSON only
  (prompt tokens are masked with -100). This is proposal Section 4.5 Stage 1 (sequence-level
  distillation). Stage 2 (token-level KD) is not implemented here.

Output directory:
  adapter/            LoRA adapter + tokenizer (load with scripts/infer_student.py)
  run_metadata.json   everything needed to reproduce the run (args, data hash, versions, losses)
"""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.jsonl_validate import validate_file  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="student training JSONL (messages format)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=1024, help="longer examples are skipped, not cut")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="tiny run: at most 8 examples and 10 steps")
    ap.add_argument("--no-4bit", action="store_true", help="load the base model without quantization "
                    "(needed on CPU; the real run uses 4-bit)")
    return ap.parse_args()


def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def encode(tokenizer, example, max_length):
    """Prompt (system + user) is masked; the model learns only the assistant JSON + end token."""
    msgs = example["messages"]
    prompt = tokenizer.apply_chat_template(msgs[:2], add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(msgs[2]["content"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + target_ids
    if len(ids) > max_length:
        return None
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + target_ids}


def collate(batch, pad_id, torch):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return ids, labels, mask


def main():
    args = parse_args()

    # 1-2. data must exist and pass the schema check before anything heavy is loaded
    if not os.path.exists(args.data):
        fail(f"training data not found: {args.data}\n"
             "Build it from teacher silver output with scripts/build_student_data.py, or for a smoke "
             "test run scripts/make_student_smoke_data.py first.")
    rep = validate_file(args.data, "student")
    if not rep.ok:
        print(rep.summary())
        fail("training data failed validation (see above)")
    examples = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    if not examples:
        fail("training data is empty")
    label_sources = sorted({(e.get("metadata") or {}).get("label_source", "unknown") for e in examples})
    if label_sources != ["silver"]:
        print(f"WARNING: label_source = {label_sources}, not silver. This run tests the script only "
              "and is NOT a research result.")
    if args.smoke:
        examples = examples[:8]
        args.max_steps = min(args.max_steps or 10, 10)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    if not use_cuda and not args.no_4bit:
        fail("no CUDA GPU found. 4-bit training needs a GPU (e.g. Colab T4). "
             "Use --no-4bit only for CPU tests with a tiny model.")

    # 3-5. model: 4-bit base + LoRA
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode(tokenizer, e, args.max_length) for e in examples]
    skipped_long = sum(x is None for x in encoded)
    encoded = [x for x in encoded if x is not None]
    if not encoded:
        fail(f"every example is longer than --max-length {args.max_length}")
    print(f"examples: {len(examples)} read, {len(encoded)} used, {skipped_long} skipped (too long)")

    if args.no_4bit:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
        model.to("cuda" if use_cuda else "cpu")
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False
    device = next(model.parameters()).device

    # 6. training loop
    steps_per_epoch = math.ceil(len(encoded) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps or max(1, math.ceil(steps_per_epoch * args.epochs))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    model.train()
    losses, step, micro, start = [], 0, 0, time.time()
    order = []
    while step < total_steps:
        if not order:
            order = list(range(len(encoded)))
            random.shuffle(order)
        batch = [encoded[order.pop()] for _ in range(min(args.batch_size, len(order)))]
        ids, labels, mask = collate(batch, tokenizer.pad_token_id, torch)
        loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)).loss
        if not torch.isfinite(loss):
            fail(f"loss became {loss.item()} at step {step}")
        (loss / args.grad_accum).backward()
        micro += 1
        if micro % args.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            losses.append(round(loss.item(), 4))
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(f"step {step}/{total_steps}  loss {loss.item():.4f}  {time.time() - start:.0f}s")
    train_seconds = time.time() - start

    # 7-8. save adapter + metadata
    adapter_dir = os.path.join(args.output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    import peft
    import transformers
    meta = {
        "base_model": args.model,
        "args": vars(args),
        "data": {"path": args.data, "sha256": sha256(args.data), "label_sources": label_sources,
                 "examples_read": len(examples), "examples_used": len(encoded), "skipped_too_long": skipped_long},
        "training": {"optimizer_steps": step, "first_loss": losses[0], "last_loss": losses[-1],
                     "losses": losses, "seconds": round(train_seconds, 1),
                     "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if use_cuda else None},
        "environment": {"gpu": torch.cuda.get_device_name(0) if use_cuda else "cpu",
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "peft": peft.__version__, "python": sys.version.split()[0], "git_commit": git_commit()},
        "system_prompt": examples[0]["messages"][0]["content"],
        "research_result": label_sources == ["silver"] and not args.smoke,
    }
    with open(os.path.join(args.output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"saved adapter to {adapter_dir}")
    print(f"saved run metadata to {os.path.join(args.output_dir, 'run_metadata.json')}")
    print(f"loss {losses[0]} -> {losses[-1]} in {step} steps, {train_seconds:.0f}s")


if __name__ == "__main__":
    main()
