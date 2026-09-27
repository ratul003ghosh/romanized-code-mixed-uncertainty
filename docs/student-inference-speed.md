# Student Inference Speed Test

Owner: Saber. Date: 2026-09-27. Notebook: `notebooks/04_student_inference_speed.ipynb`

## Why
In the feasibility test, generation took about 37 s per prompt (about 7 tokens/s).
Evaluating 500 test prompts would take about 5 hours. This test measures what makes it faster.

## Setup
- Google Colab, Tesla T4. transformers 5.16.1, bitsandbytes 0.50.2
- Model: Qwen/Qwen2.5-1.5B-Instruct, BASE model (no LoRA adapters)
- Same prompt as the feasibility test; every run generates exactly 200 new tokens; greedy decoding
- One untimed warm-up before each timed run

## Results

Single runs:

| Setup | Memory | Time (200 tokens) | Tokens/s |
|---|---|---|---|
| A. 4-bit (NF4) | ~1 GB | 12.8 s | 15.6 |
| B. 16-bit | 2.88 GB | 7.7 s | 26.1 |

Batching, 16-bit, median of 3 runs in mixed order (peak memory 3.11 GB):

| Batch | Median total | Tokens/s | Time per prompt | Spread |
|---|---|---|---|---|
| 1 | 8.2 s | 24.4 | 8.19 s | 0.4 s |
| 4 | 8.5 s | 93.6 | 2.14 s | 6.7 s |
| 8 | 8.6 s | 185.9 | 1.08 s | 0.1 s |
| 16 | 8.9 s | 358.2 | 0.56 s | 3.0 s |

A first single-pass batching run gave inconsistent numbers (batch 1 took 20.8 s), probably
because the shared GPU changed clock speed. It was repeated as above; only the medians are reported.

## Findings
1. The training setup slows generation: the feasibility test's model (LoRA attached +
   prepare_model_for_kbit_training) ran at about 7 tokens/s; the plain 4-bit model at 15.6.
2. 16-bit is about 1.7x faster than 4-bit and uses only 2.88 GB.
3. Batching is nearly free: total time is about the same for 1 to 16 prompts.
   At batch 16: 0.56 s per prompt, about 60x faster than the feasibility test.

## Recommendation
Train with QLoRA (4-bit). For evaluation, merge the LoRA adapters into a 16-bit model and
generate in batches of 16. Estimated time for 500 test prompts: about 5 minutes (estimate).

## Limits
- Base model only; the merged trained student is not tested yet (merging adds no layers,
  so speed should be similar, but this is not measured).
- Identical prompts in each batch and fixed 200-token outputs; real prompts and outputs vary.
- Batch-16 speed is throughput for evaluation. Single-user latency on this GPU is the batch-1
  number (8.2 s for 200 tokens), relevant to proposal Section 6.1.
- Shared Colab GPU: some runs were outliers (see spread).
