"""§4.1 check: do the 7B teacher and 1.5B student share the tokenizer? CPU only, downloads tokenizers/configs."""
import json
import sys

from transformers import AutoConfig, AutoTokenizer

teacher = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
student = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-1.5B-Instruct"
tt, ts = AutoTokenizer.from_pretrained(teacher), AutoTokenizer.from_pretrained(student)
ct, cs = AutoConfig.from_pretrained(teacher), AutoConfig.from_pretrained(student)
same_vocab = tt.get_vocab() == ts.get_vocab()
res = {"teacher": teacher, "student": student,
       "tokenizer_len": [len(tt), len(ts)], "config_vocab_size (padded)": [ct.vocab_size, cs.vocab_size],
       "identical_token_to_id_map": same_vocab,
       "shared_vocab_for_KD": min(len(tt), len(ts)) if same_vocab else None}
print(json.dumps(res, indent=2))
sys.exit(0 if same_vocab else 1)
