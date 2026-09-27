# Controlled Example Cases

The main examples are in:

`examples/controlled_examples.json`

The simplified expected labels for Sadat are in:

`examples/expected_outputs.json`

## Coverage

- 3 clean control examples
- 3 numeric ambiguity examples
- 3 transliteration ambiguity examples
- 2 dialect placeholders
- 3 PII examples
- 3 combined examples
- 2 intent ambiguity examples

Total: 19 examples.


## How to use

Run:

`python run_examples.py`

This checks that the JSON structure is readable and prints the categories.

It does NOT run the research model and does NOT calculate research metrics.
