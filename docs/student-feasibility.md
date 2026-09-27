# Student Feasibility Test: Qwen2.5-1.5B-Instruct + QLoRA

Owner: Saber. Date: 2026-09-26. Notebook: `notebooks/03_student_training.ipynb`

## Setup
- Hardware: Google Colab, Tesla T4 (14.6 GB usable)
- Libraries: PyTorch 2.11.0+cu128, transformers 5.16.1, peft 0.21.0, bitsandbytes 0.50.2
- Model: Qwen/Qwen2.5-1.5B-Instruct (proposal Section 4.5 names Qwen2.5-1.5B; Instruct chosen because it already follows instructions)
- 4-bit: NF4, double quantization, compute dtype float16 (T4 has no bfloat16)
- LoRA: r=16, alpha=32, dropout=0.05, all attention + MLP projections (prototype values, not tuned)
- Training: AdamW, lr 2e-4, batch 4, 30 steps, loss on target tokens only

## Results (measured)

| Check | Result |
|---|---|
| Model loads in 4-bit | Yes, 1.07 GB GPU memory |
| Trainable parameters | 18,464,768 of 1,562,179,072 (1.18%) |
| Memory with LoRA | 1.59 GB |
| Training runs | Yes, no out-of-memory error |
| Peak memory during training | 4.26 GB (sequence 339 tokens, batch 4) |
| Speed | 2.14 s per step |
| Loss | 1.3672 (step 1) -> 0.0001 (step 30) |
| Output after training | Valid JSON with all 5 schema keys |
| Generation speed | 36.9 s for about 250 tokens |

Vocabulary: tokenizer 151,665 tokens, embedding matrix 151,936 (padded). Relevant for the teacher-student vocabulary check (proposal Section 4.1).

## What this does and does not show
- Shows: the QLoRA training pipeline works on a free T4 and fits comfortably in memory.
- Does NOT show that the student learned the task. Training used ONE illustrative example (from `data/schemas/example.json`) repeated, as an "overfit one example" sanity check. The loss near 0 means memorization.
- Memory and speed are real measurements, since they depend on sequence length, not content.

## Estimates (not measured)
At 2.14 s per step, batch 4, similar sequence length, 1 epoch:
- 3,000 examples: about 27 minutes
- 20,000 examples: about 3 hours
- 50,000 examples: about 7.5 hours

## Before training (base model, no fine-tuning)
Input: "amar bkash e 5oo tk pathaisi kintu pay nai"
Output: "I have sent 500 Tk via Bkash." It silently chose 500 (no uncertainty signal) and dropped "kintu pay nai" (but it was not received).

## Open issues
- Generation is slow (about 7 tokens/s). Evaluating 500 prompts would take about 5 hours. To try: merge LoRA before inference, batched generation, abbreviated JSON keys (proposal Section 6.3).
- Not tested: training on real silver data; Stage 2 masked token-level KD (needs teacher distributions); multiple epochs; longer inputs.

## Verdict
1.5B + QLoRA is feasible for the prototype on a free T4. Inference speed is the main risk for evaluation time.
