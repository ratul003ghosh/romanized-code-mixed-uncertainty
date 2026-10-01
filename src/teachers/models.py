"""Model loading and chat templating. One teacher model on the GPU at a time."""
import gc
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


def load(name, dtype="bfloat16", load_in_4bit=False, padding_side="left"):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # transformers >= 4.56 uses `dtype`; older versions silently ignore it and load fp32
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dkey = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    kwargs = {dkey: getattr(torch, dtype)}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype))
    if not torch.cuda.is_available():
        log.warning("no CUDA GPU: %s runs on the CPU (only sensible for tests with tiny models)", name)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs).eval()
    placed = set(map(str, (getattr(model, "hf_device_map", None) or {}).values()))
    if placed & {"cpu", "disk"}:
        # device_map="auto" silently moves layers that do not fit on the GPU to CPU RAM or disk.
        # Generation then takes hours per batch and looks like "nothing happens".
        raise RuntimeError(
            f"{name} does not fit in GPU memory ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB): "
            f"parts were placed on {sorted(placed & {'cpu', 'disk'})}. Use a *_small_gpu config "
            "(4-bit, about 5.5 GB for the 7B model), e.g. configs/teacher_full_small_gpu.yaml.")
    if torch.cuda.is_available():
        log.info("loaded %s on %s (%s, 4-bit=%s), GPU memory in use %.1f GB", name, torch.cuda.get_device_name(0),
                 dtype, load_in_4bit, torch.cuda.memory_allocated() / 1e9)
    else:
        log.info("loaded %s on %s", name, model.device)
    return tok, model


def unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def shared_vocab(tok, model):
    """Real vocab vs padded embedding rows (Qwen pads 151665 -> 152064). §4.1:
    distributions are restricted to the shared (real) token set."""
    n_emb = model.get_output_embeddings().weight.shape[0]
    return min(len(tok), n_emb), n_emb


def chat_text(tok, messages):
    """Apply the chat template; fold the system prompt into the first user turn for
    models whose template rejects a system role (e.g. Gemma)."""
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        if messages[0]["role"] != "system":
            raise
        sys_msg, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
        rest[0]["content"] = sys_msg + "\n\n" + rest[0]["content"]
        return tok.apply_chat_template(rest, tokenize=False, add_generation_prompt=True)


def chat_ids(tok, messages):
    return tok(chat_text(tok, messages), add_special_tokens=False).input_ids


def gpu_mem_gb():
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0


def setup_adapters(model, adapters):
    """Optional LoRA teachers (§4.1): adapters = {variant: path}. Requires `peft`."""
    for name, path in (adapters or {}).items():
        model.load_adapter(path, adapter_name=name)
        log.info("loaded adapter %s from %s", name, path)


def activate(model, variant, adapters):
    if adapters:
        if variant not in adapters:
            raise ValueError(f"variant {variant} has no adapter; give every variant an adapter")
        model.set_adapter(variant)
