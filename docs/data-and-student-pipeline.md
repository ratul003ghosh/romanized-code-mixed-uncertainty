# Data + Student Pipeline (Saber)

How the real data becomes teacher inputs, and how the student is trained and run.
Everything here runs without the faculty GPU except the real student training.

## 1. Teacher inputs from the real dataset

Source: BanglaTLit (`aplycaebous/BanglaTLit`, EMNLP Findings 2024, proposal Section 3.1, MIT licence).

    PYTHONUTF8=1 python scripts/build_teacher_inputs.py                   # 3000 / 500 / 500
    PYTHONUTF8=1 python scripts/build_teacher_inputs.py --train 0 --dev 0 --test 0   # every row

Output in `data/processed/` (not in git):

| File | Use |
|---|---|
| `teacher_inputs.jsonl` | = train split; the file `configs/teacher_full*.yaml` reads by default |
| `teacher_inputs_{train,dev,test}.jsonl` | one file per split (per-split teacher runs, see docs/gpu-run.md) |
| `teacher_inputs_stats.json` | every count: raw, after cleaning, dropped (by reason), final, PII |

Steps: raw -> `clean_text` -> filter (malformed, empty, too long, broken unicode, exact duplicates;
test and dev are read first, so a sentence shared with train is removed from train) -> PII check
(`src.pii.detector.detect` + `src.pii.masking.sanitize`) -> schema v0.2 records, `label_source: none`.

**The teacher input is not masked.** The teachers must see the PII to learn to mask it
(proposal Section 4.1); detected PII is stored in `metadata.pii_detected` as a rule-based reference.
BanglaTLit has almost no PII, so synthetic PII injection (proposal Section 3.2) is still needed.

## 2. Checks before the GPU run

    python scripts/validate_jsonl.py --kind teacher_input data/processed/teacher_inputs*.jsonl
    PYTHONUTF8=1 python scripts/check_teacher_interface.py data/processed/teacher_inputs.jsonl --tokenizer
    python -m pytest -q tests/

`validate_jsonl.py` reports invalid JSON, malformed records, missing fields, nulls, wrong types,
empty inputs, too-long inputs, broken unicode, duplicate IDs, duplicate texts and schema violations
(`--kind record` for silver/predictions, `--kind student` for training pairs).
`check_teacher_interface.py` runs the teacher's own `load_inputs`, `build_messages` (4 variants) and
`silver.to_record` on every input; `--tokenizer` also counts prompt tokens with the Qwen2.5-7B tokenizer.

## 3. After the teacher run: silver -> student data

    PYTHONUTF8=1 python scripts/build_student_data.py experiments/teacher/train/silver.jsonl data/processed/student_train.jsonl
    python scripts/validate_jsonl.py --kind student data/processed/student_train.jsonl

## 4. Student training (GPU)

    python scripts/train_student.py --data data/processed/student_train.jsonl --output-dir experiments/student/run1

QLoRA: 4-bit NF4 Qwen2.5-1.5B-Instruct + LoRA (r 16, alpha 32), loss on the JSON target only.
Saves `adapter/` and `run_metadata.json` (args, data sha256, losses, versions, GPU, git commit).
Checkpoints are git-ignored. Stage 2 token-level KD (proposal Section 4.5) is not implemented.

## 5. Student inference

    python scripts/infer_student.py --checkpoint experiments/student/run1 \
        --input data/processed/teacher_inputs_test.jsonl --output experiments/student/run1/pred_test.jsonl

Merges the adapter into a 16-bit model, generates in batches (docs/student-inference-speed.md), and
writes schema v0.2 records (`label_source: prediction`) with `metadata.confidence`: mean/min token
log-probability, sequence log-probability, mean/max entropy (baseline B5). `--save-token-scores`
adds per-token values.

## 6. Smoke test of the student scripts (synthetic, NOT a research result)

    PYTHONUTF8=1 python scripts/make_student_smoke_data.py
    python scripts/train_student.py --data data/synthetic/student_smoke.jsonl --output-dir experiments/student/smoke --smoke
    python scripts/infer_student.py --checkpoint experiments/student/smoke --input data/synthetic/controlled_examples_v02.jsonl --output experiments/student/smoke/pred.jsonl --limit 4
    python scripts/validate_jsonl.py --kind record experiments/student/smoke/pred.jsonl

## Troubleshooting

| Problem | Fix |
|---|---|
| `ImportError: Found an incompatible version of torchao` in infer_student.py | `pip uninstall -y torchao` (not used by this project; Colab pre-installs an old version that newer peft rejects in 16-bit mode) |
| `4. tokenizer: prompt tokens min 2` | fixed: the check now counts tokens, not dictionary keys, and fails if counts look impossibly small |

## Verified on 2026-09-27 (Colab T4)
- 70/70 tests pass; teacher inputs 3000/500/500 rebuild identically; tokenizer: prompts 1158-1362 tokens
- train_student.py --smoke: 10 steps, loss 1.3313 -> 0.0407, adapter saved (synthetic data, not a result)
- infer_student.py: checkpoint loaded, 4/4 valid JSON, 0.58 s per input, predictions pass the validator

## 7. More datasets (requested by the faculty)

| Dataset | Script | Output (data/processed/, git-ignored) | Role |
|---|---|---|---|
| COMI-LINGUA TN (Hinglish, CC-BY-4.0) | `scripts/build_hinglish_inputs.py` | `teacher_inputs_hinglish_{dev,test}.jsonl` (100 / 300) | RQ4 generality check (proposal Section 3.3) |
| BanglishRev (CC-BY-NC-SA-4.0) | `scripts/build_banglishrev_inputs.py` | `teacher_inputs_banglishrev_test.jsonl` (100 PII + 400 random) | Real-PII evaluation slice only (Section 3.1) |

    PYTHONUTF8=1 python scripts/build_hinglish_inputs.py
    PYTHONUTF8=1 python scripts/build_banglishrev_inputs.py
    python scripts/validate_jsonl.py --kind teacher_input data/processed/teacher_inputs_hinglish_*.jsonl data/processed/teacher_inputs_banglishrev_test.jsonl

- Hinglish: only Roman-script rows (Devanagari rows skipped and counted); all three annotator
  normalizations kept in `metadata.reference_normalizations`.
- BanglishRev: only the review text is kept (no buyer ID, reply, dates or images). Banglish is chosen by a
  transparent word-list rule (see the script); counts per rule are in the stats file. Every record is
  `split: test`, `eval_only: true`. **Contains organic PII: never commit, never share publicly, and use
  only after the team lead's ethics OK.**
- Dialects (5-Dialects-BN) are still pending: its download link has not been found.

### Results of the real build (Colab, 2026-09-28)
- Hinglish: 400 inputs (dev 100 / test 300); 5 Devanagari rows skipped; 0 PII detected, so the Hinglish
  set tests normalization and uncertainty, not PII masking (needs synthetic PII injection).
  The `annotators_disagree` count treats any punctuation difference as disagreement; it is NOT a measure
  of ambiguity and should not be reported as one.
- BanglishRev: 1,746,943 reviews with text; the word-list rule marked 153,139 as Banglish (146,504 after
  duplicates). This is a lower bound: some Banglish with unlisted spellings falls into "english_or_unclear".
  Only 60 Banglish reviews had detected PII, so the slice is 460 (60 PII + 400 random), not 500.
  The 60 PII detections are rule-based and need a private manual check before this is called a real-PII slice.
- All three files: validator 0 problems; teacher interface PASS.

## 8. Control sets: English, Bangla script, medical (faculty request)

| Set | Script | Output | Notes |
|---|---|---|---|
| BanglishRev English-only | `build_banglishrev_inputs.py` (same run) | `teacher_inputs_banglishrev_english_test.jsonl` (200) | strict rule: no Banglish word, at least 3 English function words |
| BanglishRev Bangla script | same | `teacher_inputs_banglishrev_bangla_test.jsonl` (200) | reviews containing Bangla characters, at least 4 words |
| BanglaCHQ-Summ (medical, CC-BY-NC-SA-4.0) | `build_medical_inputs.py` | `teacher_inputs_medical_test.jsonl` (235) | Bangla-script health questions + reference summary |

All are eval-only (`split: test`) control sets with new `surface_form` values `english` and `bangla_script`;
the method itself targets Romanized input. BanglishRev and medical text are private (git-ignored).
The rule-based PII detector is built for Latin letters and ASCII digits, so "0 PII detected" in
Bangla-script text does not mean the text contains no PII.

## 9. Dialects and finance (faculty request)

| Set | Script | Output | Notes |
|---|---|---|---|
| Vashantor (Mendeley bj5jgk878b v2) | `build_dialect_inputs.py` | `teacher_inputs_dialect_{train,dev,test}.jsonl` (500 / 100 / 300) | Romanized dialect text, 5 regions: Barishal, Chittagong, Mymensingh, Noakhali, Sylhet (no Rangpur); balanced per region |
| Financial scams (Mendeley znsk27yk3h v1) | `build_finance_inputs.py` | `teacher_inputs_finance_test.jsonl` (all 523 minus drops) | Bangla / Banglish / English messages, scam or ham; eval-only |

Setup (raw files are git-ignored):

    python -m zipfile -e "<Vashantor zip>" data/raw/vashantor
    python -m zipfile -e data/raw/vashantor/*/Vashantor_CSV_Format.zip data/raw/vashantor/csv
    python -m zipfile -e "<Financial scams zip>" data/raw/finance_scams
    PYTHONUTF8=1 python scripts/build_dialect_inputs.py
    PYTHONUTF8=1 python scripts/build_finance_inputs.py

Vashantor gives each sentence in standard Banglish, standard Bangla, dialect Bangla, dialect Banglish
and English; all are kept as references. A standard sentence used in one split is blocked from the
other splits (in any dialect). The standard Banglish column contains some typing noise
(e.g. "ab bbu"), so treat the references as noisy.

### Results of the real build (laptop, 2026-09-28)
- Dialects (Vashantor): 900 inputs (train 500 / dev 100 / test 300), exactly balanced over 5 regions
  (100 / 20 / 60 per region); 0 dropped; 0 PII detected.
- Finance: 519 inputs (4 duplicates dropped); scam 309 / ham 210; by script: english 250,
  bangla_script 100, unclear 169, banglish 0. The script rule could not decide 169 messages, so the
  script split is approximate. 9 PII detections despite the authors' anonymization: probably
  placeholders or false alarms; check privately before reporting.
- All 4 files: validator 0 problems.

### Results: control sets (Colab, 2026-09-28)
- BanglishRev English-only: 200 (from 246,097 candidates); Bangla script: 200 (from 388,745 candidates).
  Banglish slice rebuilt with the same seed: identical 460 (reproducible).
- Medical (BanglaCHQ-Summ test split): 235; 0 PII detected (detector is not built for Bangla script).
- All three files: validator 0 problems.

## 10. Dataset licences and citations

| Dataset | Version used | Licence | What it allows |
|---|---|---|---|
| BanglaTLit (aplycaebous/BanglaTLit) | Hugging Face | MIT | research + commercial, keep notice |
| Vashantor | Mendeley V2, doi:10.17632/bj5jgk878b.2 | CC BY 4.0 | research + commercial, cite |
| COMI-LINGUA (LingoIITGN/COMI-LINGUA) | Hugging Face | CC BY 4.0 | research + commercial, cite |
| BanglishRev | Hugging Face, "reviews v1.json" | CC BY-NC-SA 4.0 | non-commercial, share-alike, cite |
| BanglaCHQ-Summ | GitHub alvi-khan/BanglaCHQ-Summ | CC BY-NC-SA 4.0 | non-commercial, share-alike, cite |
| Financial scams detection dataset | Mendeley V1, doi:10.17632/znsk27yk3h.1 (a V2 exists, not used) | CC BY 4.0 | research + commercial, cite |

Citations:
- Fahim et al. BanglaTLit: a benchmark dataset for back-transliteration of Romanized Bangla. Findings of EMNLP 2024.
- Faria, F. T. J.; Bin Moin, M.; Al Wase, A.; Sani, M. R.; Ahmmed, M.; Muhammad, T. (2024). Vashantor: A Large-scale
  Multilingual Benchmark Dataset for Automated Translation of Bangla Regional Dialects to Bangla Language.
  Mendeley Data, V2, doi:10.17632/bj5jgk878b.2
- Sheth, R.; Beniwal, H.; Singh, M. COMI-LINGUA: Expert Annotated Large-Scale Dataset for Multitask NLP in
  Hindi-English Code-Mixing. Findings of EMNLP 2025.
- Shamael, M. N. et al. BanglishRev: A Large-Scale Bangla-English and Code-mixed Dataset of Product Reviews in
  E-Commerce. NeurIPS 2024 Datasets and Benchmarks (arXiv 2412.13161).
- Khan, A. et al. BanglaCHQ-Summ: An Abstractive Summarization Dataset for Medical Queries in Bangla
  Conversational Speech. BLP Workshop, EMNLP 2023.
- Financial scams detection dataset (2025). International University of Business Agriculture and Technology.
  Mendeley Data, V1, doi:10.17632/znsk27yk3h.1

Three datasets are non-commercial (NC). This is fine for research; derived data from them must not be
released commercially, and share-alike (SA) applies to anything redistributed from them.

## 11. Scaled to 10,000 records (2026-09-28)
Rebuilt with larger sizes from the same real sources (no new data sources, no invented data):
BanglaTLit 5,886 (train 4,886 / dev 500 / test 500; 1 duplicate + 1 too-long row replaced),
dialects 1,900 (train 1,500 = 300 per region / dev 100 / test 300; 1 duplicate replaced),
Hinglish 400, BanglishRev 460, English 300, Bangla script 300, finance 519, medical 235 = 10,000.
Test sets keep their proposal sizes; the growth is in training data. Earlier records keep their IDs.
Commands: build_teacher_inputs.py --train 4886; build_dialect_inputs.py --train 1500;
build_banglishrev_inputs.py --n-english 300 --n-bangla 300. All files: validator 0 problems.
Before the one-time faculty GPU run, the teacher time for 10,000 inputs must be estimated from the
smoke test's sec_per_item.
