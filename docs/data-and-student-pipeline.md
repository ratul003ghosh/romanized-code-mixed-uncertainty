# Data + Student Pipeline (Saber)

How the real data becomes teacher inputs, and how the student is trained and run.
Everything here runs without the faculty GPU except the real student training.

## 1. Teacher inputs from the real dataset

Source: BanglaTLit (`aplycaebous/BanglaTLit`, EMNLP Findings 2024, proposal Section 3.1, MIT licence).

    PYTHONUTF8=1 python scripts/build_teacher_inputs.py                   # 3000 / 500 / 500
    PYTHONUTF8=1 python scripts/build_teacher_inputs.py --train 0 --dev 0 --test 0   # every row

Output in `data/processed/` (not in git):

| File | Use |
|---|---|
| `teacher_inputs.jsonl` | = train split; the file `configs/teacher_full*.yaml` reads by default |
| `teacher_inputs_{train,dev,test}.jsonl` | one file per split (per-split teacher runs, see docs/gpu-run.md) |
| `teacher_inputs_stats.json` | every count: raw, after cleaning, dropped (by reason), final, PII |

Steps: raw -> `clean_text` -> filter (malformed, empty, too long, broken unicode, exact duplicates;
test and dev are read first, so a sentence shared with train is removed from train) -> PII check
(`src.pii.detector.detect` + `src.pii.masking.sanitize`) -> schema v0.2 records, `label_source: none`.

**The teacher input is not masked.** The teachers must see the PII to learn to mask it
(proposal Section 4.1); detected PII is stored in `metadata.pii_detected` as a rule-based reference.
BanglaTLit has almost no PII, so synthetic PII injection (proposal Section 3.2) is still needed.

## 2. Checks before the GPU run

    python scripts/validate_jsonl.py --kind teacher_input data/processed/teacher_inputs*.jsonl
    PYTHONUTF8=1 python scripts/check_teacher_interface.py data/processed/teacher_inputs.jsonl --tokenizer
    python -m pytest -q tests/

`validate_jsonl.py` reports invalid JSON, malformed records, missing fields, nulls, wrong types,
empty inputs, too-long inputs, broken unicode, duplicate IDs, duplicate texts and schema violations
(`--kind record` for silver/predictions, `--kind student` for training pairs).
`check_teacher_interface.py` runs the teacher's own `load_inputs`, `build_messages` (4 variants) and
`silver.to_record` on every input; `--tokenizer` also counts prompt tokens with the Qwen2.5-7B tokenizer.

## 3. After the teacher run: silver -> student data

    PYTHONUTF8=1 python scripts/build_student_data.py experiments/teacher/train/silver.jsonl data/processed/student_train.jsonl
    python scripts/validate_jsonl.py --kind student data/processed/student_train.jsonl

## 4. Student training (GPU)

    python scripts/train_student.py --data data/processed/student_train.jsonl --output-dir experiments/student/run1

QLoRA: 4-bit NF4 Qwen2.5-1.5B-Instruct + LoRA (r 16, alpha 32), loss on the JSON target only.
Saves `adapter/` and `run_metadata.json` (args, data sha256, losses, versions, GPU, git commit).
Checkpoints are git-ignored. Stage 2 token-level KD (proposal Section 4.5) is not implemented.

## 5. Student inference

    python scripts/infer_student.py --checkpoint experiments/student/run1 \
        --input data/processed/teacher_inputs_test.jsonl --output experiments/student/run1/pred_test.jsonl

Merges the adapter into a 16-bit model, generates in batches (docs/student-inference-speed.md), and
writes schema v0.2 records (`label_source: prediction`) with `metadata.confidence`: mean/min token
log-probability, sequence log-probability, mean/max entropy (baseline B5). `--save-token-scores`
adds per-token values.

## 6. Smoke test of the student scripts (synthetic, NOT a research result)

    PYTHONUTF8=1 python scripts/make_student_smoke_data.py
    python scripts/train_student.py --data data/synthetic/student_smoke.jsonl --output-dir experiments/student/smoke --smoke
    python scripts/infer_student.py --checkpoint experiments/student/smoke --input data/synthetic/controlled_examples_v02.jsonl --output experiments/student/smoke/pred.jsonl --limit 4
    python scripts/validate_jsonl.py --kind record experiments/student/smoke/pred.jsonl

## Troubleshooting

| Problem | Fix |
|---|---|
| `ImportError: Found an incompatible version of torchao` in infer_student.py | `pip uninstall -y torchao` (not used by this project; Colab pre-installs an old version that newer peft rejects in 16-bit mode) |
| `4. tokenizer: prompt tokens min 2` | fixed: the check now counts tokens, not dictionary keys, and fails if counts look impossibly small |

## Verified on 2026-09-27 (Colab T4)
- 70/70 tests pass; teacher inputs 3000/500/500 rebuild identically; tokenizer: prompts 1158-1362 tokens
- train_student.py --smoke: 10 steps, loss 1.3313 -> 0.0407, adapter saved (synthetic data, not a result)
- infer_student.py: checkpoint loaded, 4/4 valid JSON, 0.58 s per input, predictions pass the validator
