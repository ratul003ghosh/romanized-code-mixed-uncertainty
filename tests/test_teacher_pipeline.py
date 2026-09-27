"""End-to-end CPU test of all four stages with a tiny random Qwen2 model + tiny BPE tokenizer.
No downloads. Generation output is garbage, so valid teacher JSON is injected before pivoting;
everything else (chat templating, teacher forcing, vocab slicing, offsets, spans) runs for real.
"""
import json
import os

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from src.teachers import generate, pivot, score
from src.uncertainty import spans
from src.teachers.common import out_path, read_jsonl, write_jsonl
from src.teachers.prompts import EXAMPLES, TASK
from src.teachers.output_schema import serialize_with_offsets, validate

CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def build_tiny(d):
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    corpus = [TASK] + [e[0] + json.dumps(e[1]) for e in EXAMPLES] * 5
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = CHATML
    tok.save_pretrained(d)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok) + 37, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    Qwen2ForCausalLM(cfg).save_pretrained(d)   # padded vocab tests the slicing path


def test_pipeline(tmp_path):
    mdir = str(tmp_path / "tiny")
    build_tiny(mdir)
    inp = tmp_path / "in.jsonl"
    rows = [json.loads(l) for l in open("data/synthetic/teacher_smoke.jsonl")][:2]
    rows[0]["split"], rows[1]["split"] = "train", "dev"          # like Saber's per-split export
    write_jsonl(inp, rows)
    cfg = {"input_file": str(inp), "output_dir": str(tmp_path / "out"),
           "same_tokenizer": {"model": mdir, "dtype": "float32",
                              "variants": ["balanced", "dialect", "finance_pii", "transliteration"]},
           "heterogeneous": [], "generation": {"max_new_tokens": 8, "batch_size": 2},
           "scoring": {"kd_top_k": 5, "max_pivot_tokens": 2000}, "spans": {"flag_quantile": 0.5}}

    gen_path = generate.run(cfg)
    gens = read_jsonl(gen_path)
    assert len(gens) == 8 and not any(g["valid"] for g in gens)   # random model -> invalid JSON

    # inject plausible teacher outputs that disagree on the PII type
    for g in gens:
        typ = "NID" if g["variant"] in ("balanced", "dialect") else "ACCOUNT"
        obj, _ = validate({"sanitized_prompt": f"Please check <{typ}_1> for {g['id']}.",
                           "pii": [{"type": typ, "span": "01900000000", "placeholder": f"<{typ}_1>"}],
                           "preserved_entities": [{"type": "AMOUNT", "value": "1200 BDT"}],
                           "ambiguous_spans": [{"span": "12oo", "types": ["NUMERIC_AMBIGUITY"],
                                                "candidates": ["1200 BDT"]}]})
        g.update(parsed=obj, parse_ok=True, valid=True)
    write_jsonl(gen_path, gens)

    pivot.run(cfg)
    p = read_jsonl(out_path(cfg, "pivots.jsonl"))
    assert all(x["pivot_ok"] and x["pivot_teacher"] == "same:balanced" for x in p)

    _, info = score.run(cfg)
    assert info["sanity"]["passed"] and info["vocab_used"] < info["embedding_rows"]
    ts = read_jsonl(out_path(cfg, "token_scores.jsonl"))
    for r in ts:
        tok_text = "".join(r["tokens"])
        assert tok_text == p[0]["y_star"] or r["id"] != p[0]["id"]   # offsets consistent
        assert len(r["A"]) == len(r["E"]) == r["n_tokens"] and min(r["E"]) >= 0

    recs = spans.run(cfg)
    s = spans.report(cfg, recs, gen_path, info)
    pii = [x for x in recs[0]["spans"] if x["field"] == "pii[0]"][0]
    assert abs(pii["d"] - 2 / 3) < 1e-9 and set(pii["candidates"]) == {"NID", "ACCOUNT"}
    assert s["n_spans"] > 0
    check_silver_v02(read_jsonl(out_path(cfg, "silver.jsonl")))
    print(open(out_path(cfg, "report.md")).read()[:1500])


def check_silver_v02(silver):
    """Silver records follow team schema v0.2 and pass Saber's converter gate."""
    keys = {"id", "schema_version", "input", "clean_input", "normalized_text", "sanitized_prompt", "pii",
            "preserved_entities", "uncertainties", "routing", "metadata"}
    assert [r["metadata"]["split"] for r in silver] == ["train", "dev"]
    for r in silver:
        assert keys <= set(r) and r["schema_version"] == "0.2" and r["metadata"]["label_source"] == "silver"
        for p in r["pii"]:
            if p["start"] is not None:
                assert r["clean_input"][p["start"]:p["end"]] == p["text"]
        for u in r["uncertainties"]:
            assert u["types"] and 0 <= u["aleatoric"] <= 1 and 0 <= u["epistemic"] <= 1
            if u["start"] is not None:
                assert r["clean_input"][u["start"]:u["end"]] == u["span"]
    assert silver[0]["pii"][0]["start"] is not None       # smoke-1 contains 01900000000
    assert silver[1]["metadata"]["unlocated_spans"] >= 1  # smoke-2 does not: counted, not invented
    conv = os.path.join(os.path.dirname(__file__), "..", "scripts", "build_student_data.py")
    if os.path.exists(conv):                                   # Saber's converter (PR #27)
        import importlib.util
        spec = importlib.util.spec_from_file_location("bsd", conv)
        bsd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bsd)
        assert [bsd.check_record(r) for r in silver] == [None, "not_train_split"]
        assert json.loads(json.dumps(bsd.make_target(silver[0])))["sanitized_prompt"]
