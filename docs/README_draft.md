# Know Why You're Unsure: Romanized Code-Mixed Prompt Sanitization

A small on-device model that rewrites Banglish/Romanized prompts into a
clean, PII-masked form, flagging uncertain spans as aleatoric (input is
ambiguous) or epistemic (model lacks knowledge), and routes accordingly.

## Pipeline
Input → Preprocessing → PII Detection → Teacher Ensemble → Aleatoric/Epistemic
→ Student (1.5B) → Routing → Evaluation

## Routing (first matching rule wins)
1. Raw PII in output → placeholder + override
2. Blocking span, high aleatoric → ASK_USER
3. Any span, high epistemic → ESCALATE
4. Non-blocking flags remain → PROCEED_WITH_FLAGS
5. No flags → PROCEED

## Setup
git clone <repo>
pip install -r requirements.txt
export HF_TOKEN=...   # never commit this

## Running the demo
python scripts/run_demo.py --all

## Status
Dataset [DONE] | Teacher code [DONE] | Teacher GPU run [PENDING] | Silver [PENDING] | Student [PENDING] |
Routing [PARTIAL] | Evaluation [PARTIAL] | Demo [PENDING] (currently using placeholder inputs;
PII detection/masking/arbitration are real and verified)

## Safety rules
- No tokens committed
- No raw PII JSONL committed
- No fabricated teacher/student results - placeholders clearly labeled as such
