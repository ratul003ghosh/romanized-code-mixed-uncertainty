# Demo Inputs

This directory contains carefully crafted sample inputs (`demo_inputs.jsonl`) intended to demonstrate the end-to-end capabilities of the student model. 

These inputs are **fictional** test cases representing various challenges in Romanized Bangla/Banglish processing. No real PII or private data is used here.

## Demo Categories Covered

1. **demo_01 (normal Banglish)**: Evaluates the model's ability to seamlessly proceed on standard conversational text without hallucinating PII or flagging false positives.
2. **demo_02 (obvious PII)**: Tests baseline PII detection (phone numbers). The model should accurately extract and mask this.
3. **demo_03 (ambiguous PII boundary)**: Tests epistemic uncertainty and boundary identification ("Rahman bhaiyer mudi dokaner pashe" vs. "Gulshan 2"). The model should assign uncertainty to the span boundary of the location.
4. **demo_04 (numeric ambiguity)**: Tests whether a simple numeric value is marked as a preserved entity, an ambiguous code, or left alone.
5. **demo_05 (dialect/meaning ambiguity)**: Tests aleatoric/human uncertainty handling. "Chhera" can mean "torn" or "boy" (Sylheti dialect). This should trigger `DIALECT_MEANING` uncertainty.
6. **demo_06 (multiple PII in one sentence)**: Tests robust multi-entity extraction (Names, Account Number, Email Address) within a single code-mixed utterance.
7. **demo_07 (ambiguous number)**: Evaluates semantic disambiguation where the same number ("1000") could act as an ID code (sensitive) or a monetary amount (preserved).
8. **demo_08 (uncertainty/routing)**: Tests risk assessment. The inclusion of a PIN code ("4321") in a troubleshooting context should ideally trigger an `ESCALATE` or `PROCEED_WITH_FLAGS` routing decision due to privacy risks.
9. **demo_09 (code-mixed English/Banglish)**: Assesses cross-lingual parsing and OTP detection in heavily code-mixed text.
10. **demo_10 (spelling/transliteration ambiguity)**: Tests `TRANSLITERATION` uncertainty (brak vs. brac), identifying ambiguity in phonetic Romanization.

## Usage
These inputs are meant strictly for demonstration (e.g., via `scripts/demo_student.py`) to observe the predicted outputs (`sanitized_prompt`, `uncertainties`, `routing`, etc.) *after* the model is trained. No predictions or evaluation results are pre-computed here.
