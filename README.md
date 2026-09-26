# Know Why You're Unsure — code

Setup on Kaggle: open `kaggle_block1.ipynb` (import it into a new Kaggle notebook) and follow its cells.
Local setup (Linux/WSL2, NVIDIA GPU): `bash scripts/setup_env.sh`

Output contract: `configs/schema.json` (v1.0.0, frozen; tag `schema-v1.0.0`).
Changing it requires a version bump in `$id`, a new tag, and passing tests.

Validate outputs: `python -m src.common.validate FILE.jsonl`
Tests: `python -m pytest -q tests/`

| Folder | Purpose (proposal section) |
|---|---|
| data/raw, silver, gold | source corpora, teacher targets, human gold (3.1–3.3); never committed |
| src/common | schema validation, shared utilities |
| src/pii_inject | synthetic PII in three regimes (3.2) |
| src/noise | typos, digit–letter swaps, OCR confusions (3.2) |
| src/teachers | LoRA teachers, teacher-forced scoring (4.1–4.2) |
| src/decomp | aleatoric/epistemic decomposition, logistic heads (4.3–4.4) |
| annotation | guidelines and annotation files (2.1, 3.3) |
| configs | schema and experiment configs |
