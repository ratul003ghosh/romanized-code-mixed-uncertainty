# Uncertainty Types

## 1. TRANSLITERATION

A Romanized Bangla form can have multiple possible interpretations.

Example:

`kal jabo`

Expected issue:

`kal` can require context.

Expected Day 1 signal:

Aleatoric = HIGH (expected test condition, not measured)

## 2. NUMERIC_AMBIGUITY

A noisy numeric expression has multiple plausible readings.

Example:

`5oo tk`

Controlled candidates:

- 500 BDT
- 5000 BDT

Expected Day 1 signal:

Aleatoric = HIGH (expected test condition, not measured)

## 3. PII_BOUNDARY

A span appears to contain personal/identifying information, but its exact subtype is unclear.

Example:

`1987654321`

Controlled candidates:

- NID
- account number
- other identifier

The safe prototype behavior is to mask the identifier rather than expose it.

## 4. DIALECT_MEANING

A dialect-specific expression is difficult to interpret.

Day 1 uses placeholders until a dialect term is verified from an approved source or annotation.

## 5. INTENT

The user's intended action/request is unclear.

A verified example should be added later rather than inventing a fake research label.

## Important distinction

`HIGH`, `LOW`, and `PENDING_MEASUREMENT` in controlled examples are expected test conditions. They are NOT teacher scores, accuracy, AUROC, or other measured research results.


## 5. INTENT

The user's requested action can be unclear even when the words are understandable.

Example:

`bkash er taka niye ki korbo?`

Possible development interpretations include checking a transaction, sending money, or reporting a problem.

This is a controlled example. The candidate intents are not human-annotated gold labels.
