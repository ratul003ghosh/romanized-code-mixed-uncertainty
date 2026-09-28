# Demo Walkthrough

⚠️ **Note:** Example inputs below are author-written placeholders, pending
Hugging Face dataset access (`ml-paper-group`/`mlpaper` teacher-inputs).
They will be replaced with real dataset records once access is confirmed.
PII detection, masking, and rule-1 arbitration shown here run on the REAL
code (`src/pii/masking.py`, `src/pii/arbitrate.py`) — only the input text
is a stand-in.

Run: `python scripts/run_demo.py --all`

## Example 1: Safe / clear input → PROCEED
- Input (placeholder): "kemon acho? ajke amar meeting ache biket"
- Why chosen: no PII, no ambiguity
- PII detected: none
- Routing rule: rule 5 (no flags) → PROCEED
- Results: ⏳ Pending real teacher/student run

## Example 2: Ambiguous input → ASK_USER
- Input (placeholder): "eta thik koro"
- Why chosen: request has no clear object → INTENT ambiguity
- PII detected: none
- Expected channel: aleatoric (every teacher unsure)
- Routing rule: rule 2 → ASK_USER
- Results: ⏳ Pending

## Example 3: Model/knowledge uncertainty → ESCALATE
- Input (placeholder): "5oo tk pathaisi kalke, taka pay nai"
- Why chosen: ambiguous amount, teachers likely disagree
- PII detected: none
- Expected channel: epistemic (teachers confident but disagree)
- Routing rule: rule 3 → ESCALATE
- Results: ⏳ Pending

## Example 4: PII-containing input → safe output
- Input (placeholder): "amar number 01712345678, bkash trans id 9X87K, OTP 4821 eta check koren"
- PII detected: PHONE (01712345678, score 0.95), TXN_ID (9X87K, score 0.95), OTP (4821, score 0.9)
- Masked output: "amar number <PHONE_1>, bkash trans id <TXN_ID_1>, OTP <OTP_1> eta check koren"
- Rule 1 check: leaked = false, replacements = [] (nothing raw remained)
- Note: teacher input keeps PII visible by design; masking applies to the final output only
- Results: ⏳ Pending for teacher/student parts
