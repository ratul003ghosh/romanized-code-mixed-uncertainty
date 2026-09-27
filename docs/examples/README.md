# Masrafi — Day 1: Synthetic Examples + Documentation

## What this folder contains

This is the Day 1 deliverable for the Synthetic Examples + Documentation role.

### Main files

- `examples/controlled_examples.json` — controlled synthetic examples
- `examples/expected_outputs.json` — simplified expected labels for evaluation
- `experiments/experiment_log.csv` — experiment/setup log
- `docs/project_overview.md` — beginner project explanation
- `docs/pipeline_explanation.md` — pipeline explanation
- `docs/uncertainty_types.md` — uncertainty categories
- `docs/example_cases.md` — example documentation
- `docs/figures_tables.md` — figure/table placeholders
- `run_examples.py` — simple JSON validation/check script

## Important research rule

The examples are controlled synthetic development/demo examples.

They are NOT human-annotated gold data and are NOT research results.

The checker only validates the file structure and prints coverage. It does not run the model and does not calculate research metrics.

## Run in VS Code

1. Open this folder in VS Code.
2. Open the VS Code terminal.
3. Run:

```bash
python run_examples.py
```

If your computer uses `python3`, run:

```bash
python3 run_examples.py
```

Expected result includes:

`PASSED - all examples contain the required fields.`

## GitHub

Use a branch for this work. Do not push directly to `main` if your team workflow requires PRs.

Typical commands:

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
git checkout -b masrafi-day1-examples
```

Copy these Day 1 files into the repository.

Then:

```bash
git status
git add examples docs experiments run_examples.py
git commit -m "Add Day 1 controlled examples and documentation"
git push -u origin masrafi-day1-examples
```

Then open a Pull Request on GitHub and ask Ratul to review/merge it.

Never commit passwords, API keys, private tokens, model checkpoints, or real PII.
