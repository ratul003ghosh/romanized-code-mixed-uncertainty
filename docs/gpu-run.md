# GPU run: teacher ensemble

Owner: Zarif. Issues #7 (smoke test), #17 (handoff doc). Code: `src/teachers/`, `src/uncertainty/`.
This page is for whoever runs the teacher models on an external GPU. You do not need to read the code.

## 0. What you need

| | Big GPU (≥ 20 GB with bfloat16, e.g. A100, L4 24 GB, RTX 3090/4090) | Small GPU (< 20 GB or no bfloat16, e.g. Colab/Kaggle T4) |
|---|---|---|
| Config | `configs/teacher_*.yaml` (bfloat16) | `configs/teacher_*_small_gpu.yaml` (4-bit, float16) |
| Qwen2.5-7B memory | ~15 GB | ~5.5 GB |
| Free disk | ~20 GB per model (downloads) | same |

`scripts/test_gpu.py` picks the right smoke-test config automatically. Python 3.10+ is required.

## 1. Setup

```bash
git clone https://github.com/ratul003ghosh/romanized-code-mixed-uncertainty.git
cd romanized-code-mixed-uncertainty
pip install -r requirements.txt
```
On Colab/Kaggle, prefix shell commands with `!` and run `%cd romanized-code-mixed-uncertainty` once.

## 2. Smoke test (always first, ~10–15 min)

```bash
python scripts/test_gpu.py
```
It prints the GPU, runs the CPU tests, checks the teacher/student tokenizer, then runs 5 synthetic
prompts × 4 teachers.

Passing looks like this in the output:
- `using config: ...` (which config it picked)
- tests: `9 passed`
- `sanity (identical teachers -> E=0): {... 'passed': True}`
- `DONE. Send back: experiments/teacher/smoke_results.tar.gz`

**Send back** `experiments/teacher/smoke_results.tar.gz`. If any step fails, send
`experiments/teacher/smoke/smoke_console.log` instead. Both contain only synthetic data.

## 3. Full run (only after the team confirms the smoke results)

The input files are securely stored on a private Hugging Face repository and are not in git (`data/processed/` is ignored). To download and prepare them:

```bash
# 1. Log in with an authorized token
hf auth login

# 2. Download and prepare splits locally
python scripts/prepare_teacher_data.py --dataset mlpaper/teacher-inputs --split train --output data/processed/teacher_inputs_train.jsonl
python scripts/prepare_teacher_data.py --dataset mlpaper/teacher-inputs --split dev --output data/processed/teacher_inputs_dev.jsonl
python scripts/prepare_teacher_data.py --dataset mlpaper/teacher-inputs --split test --output data/processed/teacher_inputs_test.jsonl
```
Note: Ensure you are logged into an account that has access to `mlpaper/teacher-inputs`.

The heterogeneous teachers (Gemma-2-9B, Llama-3.1-8B) are gated on Hugging Face: log in, accept both
licences, create a read token, then:
```bash
export HF_TOKEN=hf_...                     # Colab: use the Secrets panel instead of pasting it
CONFIG=configs/teacher_full.yaml           # small GPU: configs/teacher_full_small_gpu.yaml
for SPLIT in train dev test; do
  python scripts/run_teacher.py --config $CONFIG \
    --input data/processed/teacher_inputs_${SPLIT}.jsonl \
    --output-dir experiments/teacher/${SPLIT}
done
```
- One folder per split, so dev/test outputs can never be mixed into training data.
- **Resumable:** if the session stops, run the same command again; finished (input, teacher) pairs are skipped.
- On Colab/Kaggle, sessions end after some hours. Copy `experiments/teacher/` to Google Drive or Kaggle
  output regularly, or run one split per session.
- Time estimate: take `sec_per_item` from the smoke run's `summary.json` × number of prompts × 6 teachers
  (generation), plus scoring. Tell the team the estimate before starting a long run.

**Send back** the `experiments/teacher/train`, `dev` and `test` folders as one zip, privately
(Drive link or direct message). Do not upload them anywhere public.

## 4. Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Use the `_small_gpu` config, or lower `generation.batch_size` (8 → 4 → 2) |
| `401` / `403` / `gated repo` on download | Accept the model licence on Hugging Face and set `HF_TOKEN` |
| Error about `bfloat16` on a T4 | Use the `_small_gpu` config (float16) |
| Gemma-2 produces empty or broken output in float16 | Known Gemma-2/float16 issue: remove the `gemma2_9b` line from the config and note it |
| Sanity check `'passed': False` or `nan` values | Stop and send the log; do not start the full run |
| Session disconnected mid-run | Rerun the same command; it resumes |

## What not to do
- Do not change code or prompts; if something breaks, send the log.
- Do not commit or upload outputs, tokens, or model files.
- Record which config and GPU were used (the smoke log shows both).
