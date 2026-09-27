# Teacher ensemble + two-channel uncertainty

Owner: Adham. Proposal v2 §4.1-4.4. Issues #2-#7.

| Issue | Where |
|---|---|
| #2 Teacher model loading | `src/teachers/models.py` |
| #3 Teacher inference | `src/teachers/generate.py`, `src/teachers/prompts.py` |
| #4 Probability/logit extraction | `src/teachers/score.py` (teacher-forced, restricted to the real vocab) |
| #5 Entropy calculation | `src/uncertainty/decomposition.py` |
| #6 Teacher disagreement | `src/uncertainty/decomposition.py` (JSD), `src/uncertainty/agreement.py` (field-level d(s)) |
| #7 GPU smoke test | `scripts/test_gpu.py`, `configs/teacher_smoke.yaml` |

## Pipeline
| Stage | Where | Output (in `experiments/teacher/<run>/`) |
|---|---|---|
| A generate: every teacher writes the full JSON | GPU | `generations.jsonl` |
| B pivot: choose y* (best teacher or medoid) | CPU | `pivots.jsonl` |
| C score: K teachers score y* under teacher forcing | GPU | `token_scores.jsonl` |
| D spans: a(s), e(s), d(s), types, candidates | CPU | `spans.jsonl`, `silver.jsonl`, `report.md`, `summary.json` |

Decomposition in nats with uniform weights: `H(p_bar) = A + E`, `A = mean_k H(p_k)`, `E = JSD(p_1..p_K)`.
Entropy weights (Eq. 6) are used only for the KD target stored in `token_scores.jsonl`.

## Interfaces
- Input (from dataset pipeline): JSONL, one `{"id": ..., "text": ...}` per line; extra keys are carried along.
- Student / silver (Saber): `silver.jsonl` with `input` and `target` in the proposal §2 schema;
  `token_scores.jsonl` has `kd_top_ids` / `kd_top_probs` per pivot token for masked KD.
- Evaluation (Rafiuzzaman): per-span `a_raw`, `e_raw`, `d`, `aleatoric`, `epistemic`, `flagged` in `spans.jsonl`;
  `pii_spans_for_eval` in `silver.jsonl`; schema pass rates per teacher in `summary.json`.
- Routing (Ratul): `routing` is `null`; the decision layer reads `aleatoric`, `epistemic`, `types`.

## Prototype vs. proposal (document, do not hide)
| Proposal | Prototype | Upgrade |
|---|---|---|
| 4 LoRA teachers on Qwen2.5-7B | 4 prompt variants mirroring the LoRA data mixes | `adapters:` per variant in config |
| Logistic heads on gold dev (Eq. 4) | percentile of a(s)/e(s) within the run; flag above a quantile | fit heads once gold dev exists |
| Lexicon-based typing | heuristics + dialect lexicon hook | lexicon from 5-Dialects-BN / BIDWESH |
| Spans on the input | listed spans are input spans; discovered spans are output words | align output words to input |
