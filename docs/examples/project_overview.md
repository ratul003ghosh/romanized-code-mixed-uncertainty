# Project Overview — Beginner Version

## What is this project?

The project studies whether a small model can safely sanitize Banglish/Romanized code-mixed prompts while also explaining why it is uncertain.

The system is designed around two kinds of uncertainty:

- **Aleatoric uncertainty:** the input itself is genuinely ambiguous.
- **Epistemic uncertainty:** the model is uncertain because of a knowledge/model limitation.

The proposal also uses typed uncertainty spans such as:

- TRANSLITERATION
- NUMERIC_AMBIGUITY
- PII_BOUNDARY
- DIALECT_MEANING
- INTENT

The final decision layer can use routes such as:

- PROCEED
- PROCEED_WITH_FLAGS
- ASK_USER
- ESCALATE

## Important for Day 1

The examples in this folder are **controlled synthetic development/demo examples**.

They are NOT:

- human-annotated gold data
- a research dataset
- experimental results
- evidence that the model already works

Actual research results must come from real experiments and evaluation.
