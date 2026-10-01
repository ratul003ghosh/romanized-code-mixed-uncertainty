# After the faculty teacher run: from raw outputs to the real silver data

Owner: Zarif. Hand-off to Saber at the end. Scripts: `scripts/check_teacher_run.py`, `scripts/finalize_silver.py`.

## 0. Put the returned folders in place

The faculty sends back `experiments/teacher/train`, `dev` and `test` (one folder per split, see `docs/faculty-run.md`).
Unzip them so the layout is:

```
experiments/teacher/train/{generations,pivots,token_scores,spans,silver}.jsonl, summary.json, score_info.json, config_used.yaml
experiments/teacher/dev/...
experiments/teacher/test/...
```

Shards from a two-GPU run go in folders named `train_shard0`, `train_shard1`, ...; they are merged automatically.
The input files named in each `config_used.yaml` (`data/processed/teacher_inputs_<split>.jsonl`) should be present
too, so the counts can be checked against the real inputs.

## 1-3. Inspect, validate, confirm counts and splits

```bash
PYTHONUTF8=1 python scripts/check_teacher_run.py experiments/teacher --examples 5
```

| Check | FAIL when | WARN when |
| --- | --- | --- |
| files | a pipeline output is missing | `score_info.json` missing |
| counts | an input has no answer from a teacher; ids lost or duplicated between stages | input file not found |
| split | a silver record has another split than its folder or its input; label_source not `silver` | |
| teachers | | a teacher parses under 80% of its answers (error kinds and likely truncated answers are listed) |
| scores | GPU sanity failed; A + E differs from H(p_bar); negative E; arrays of different length | pivots cut at `max_pivot_tokens`; E is zero everywhere |
| silver | schema v0.2 validator fails | raw PII copied into `sanitized_prompt`; placeholders differ from the pii list; detector finds PII-like text; many spans not located |
| cross-split | the same id in two splits | the same `clean_input` text in two splits |

Read the examples at the end of `experiments/teacher/run_check.md` by eye: does the sanitized prompt keep the
meaning, are the placeholders right, do the uncertain spans make sense?

`run_check.md` contains example prompts: do not commit it if the inputs are private. `run_check.json` has
only counts and ids and can go into the experiment log.

## 4-5. Generate and validate the real silver data

```bash
PYTHONUTF8=1 python scripts/finalize_silver.py experiments/teacher --out-dir data/processed
```

Writes `data/processed/silver_{train,dev,test}.jsonl` and `silver_manifest.json`. A record is dropped (never
edited) if it fails the schema, has the wrong split, copies raw PII into `sanitized_prompt`, has placeholders
that do not match its pii list, or repeats an id. The output is validated again; for train every record must
pass `build_student_data.py`'s gate. Exit code 1 = something is wrong; do not hand it over.

## 6. Hand-off to Saber

Give Saber `data/processed/silver_train.jsonl` and `silver_manifest.json` (privately, like the inputs). He runs:

```bash
python scripts/build_student_data.py data/processed/silver_train.jsonl data/processed/student_train.jsonl
python scripts/validate_jsonl.py --kind student data/processed/student_train.jsonl
```

`silver_dev.jsonl` / `silver_test.jsonl` are teacher predictions on evaluation data: for RQ1 (do teacher scores
match human ambiguity) with Sadat's evaluator, never for training.

Log the numbers from `silver_manifest.json` (kept, dropped by reason, sha256) in `docs/experiment-log.md`.
