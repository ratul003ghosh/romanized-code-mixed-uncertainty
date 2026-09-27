# GPU run: teacher ensemble

Owner: Adham. Issues #7 (smoke test), #17 (handoff doc).

## Requirements
- One NVIDIA GPU, at least 20 GB (32 GB planned). Qwen2.5-7B in bf16 is about 15 GB.
- Python 3.10+, about 20 GB free disk for the model download.

## 1. Smoke test (run this first, ~10-15 min)
```bash
git clone https://github.com/ratul003ghosh/romanized-code-mixed-uncertainty.git
cd romanized-code-mixed-uncertainty
git checkout feat/teacher-uncertainty      # until it is merged into main
pip install -r requirements.txt
python scripts/test_gpu.py
```
Send back `experiments/teacher/smoke_results.tar.gz`. If a step fails, send
`experiments/teacher/smoke/smoke_console.log` instead.

What passing looks like:
- tests: `9 passed`
- `sanity ... 'passed': True` in the log (the same teacher scored twice gives E = 0)
- `experiments/teacher/smoke/report.md` exists with 5 inputs

## 2. Full run (one run per split, after we send the input files)
```bash
export HF_TOKEN=...    # gemma-2-9b-it and Llama-3.1-8B are gated: accept their licences on Hugging Face first
for SPLIT in train dev test; do
  python scripts/run_teacher.py --config configs/teacher_full.yaml \
    --input data/processed/teacher_inputs_banglatlit_${SPLIT}.jsonl \
    --output-dir experiments/teacher/${SPLIT}
done
```
Each split gets its own folder, so dev/test teacher outputs can never be mixed into training data.
- Resumable: rerunning skips finished (input, teacher) pairs.
- Stages can be run separately: `--stage generate | pivot | score | spans`.
- Send back the `experiments/teacher/train`, `dev` and `test` folders (zip them). Do not upload it anywhere public.

## Troubleshooting
- Out of memory: set `load_in_4bit: true` in the config (needs bitsandbytes), or lower `generation.batch_size`.
- 401/403 on download: the model is gated; accept the licence and set `HF_TOKEN`.
