# Pipeline Explanation — Beginner Version

## 1. Input

The user writes a Banglish/Romanized prompt.

Example:

`ami 5oo tk send korsi`

## 2. Preprocessing

The system cleans/normalizes the input while preserving information needed for PII and uncertainty analysis.

## 3. PII detection and masking

Potential personal information is detected.

Example:

`01712345678` -> `<PHONE_1>`

## 4. Teacher models

Multiple teacher models analyze the task. Their behavior is compared to estimate uncertainty.

## 5. Uncertainty

The project separates uncertainty into:

### Aleatoric

The input itself is ambiguous.

Example:

`kal` may require context to know the intended time reference.

### Epistemic

The model is uncertain because of its own knowledge/model limitations.

## 6. Typed uncertainty

The project uses uncertainty types including:

- TRANSLITERATION
- NUMERIC_AMBIGUITY
- PII_BOUNDARY
- DIALECT_MEANING
- INTENT

## 7. Routing

Depending on the detected risk/uncertainty, the system can choose:

- PROCEED
- PROCEED_WITH_FLAGS
- ASK_USER
- ESCALATE

## 8. Evaluation

The evaluation team will later compare actual model outputs against appropriate labels and metrics.

Day 1 controlled examples only provide known development cases. They do not prove model performance.
