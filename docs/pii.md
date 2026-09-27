# PII detection and masking

Owner: Zarif (issues #10, #11). Proposal v2 §2.1 (precedence rules), §4.6 (rule 1, conformal risk control), §6.1 (metrics).

| Part | Code | What it does |
|---|---|---|
| Detection | `src/pii/detector.py` | Finds the 10 schema PII types in `clean_input`; every span has a confidence `score` |
| Masking | `src/pii/masking.py` | Replaces spans with `<TYPE_N>`; same value = same placeholder, also when re-formatted; a detected name/e-mail is masked everywhere it repeats |
| Output check | `src/pii/arbitrate.py` | Proposal §4.6 rule 1: re-masks PII that survives in teacher/student output and returns `leaked` for routing |
| Evaluation | `scripts/eval_pii.py` | Span P/R/F1 (exact and IoU >= 0.5), leakage rate, over-masking, per type and per regime |
| Test data | `scripts/make_pii_cases.py` → `data/synthetic/pii_cases*.jsonl` | Hand-made synthetic cases, schema v0.2, `label_source: synthetic` |

## Usage

```python
from src.pii.masking import sanitize
out = sanitize("amar nam rahim, rahim ke 01712345678 e call dao")
out["masked"]   # "amar nam <NAME_1>, <NAME_1> ke <PHONE_1> e call dao"
out["pii"]      # [{"type", "placeholder", "text", "start", "end", "score"}, ...]

from src.pii.arbitrate import enforce
res = enforce("Call 017-1234-5678 about <TXN_ID_1>", source_pii=out["pii"])
res["text"], res["leaked"]   # "Call <PHONE_1> about <TXN_ID_1>", True
```

## Detection rules

- **Normalization:** Bangla digits and digit look-alikes next to digits (`5oo`, `O17l...`) are read as digits; offsets stay on the original text.
- **Amounts are never PII:** a number with tk / taka / BDT / ৳ / Rs before or after it is skipped (§2.1).
- **Context decides the type (§2.1):** a keyword in the same clause (nid, account, a/c, otp, pin, code, trx id, card, ...). A BD mobile pattern is PHONE; a Luhn-valid 13–19 digit number is CARD.
- **Conservative default:** a bare digit string of 10+ digits is ID_NUMBER (score 0.7, masked). 7–9 digits without context get score 0.3 and are **not** masked by default.
- **Formats:** separators (`017-1234-5678`, `4111 1111 1111 1111`), `+880`, spoken numbers in English and Banglish (`zero one seven…`, `shunno ek sat…`), glued postpositions (`1234567890e`).
- **Names:** after "amar nam/naam", "nam holo", "my name is", or a title (Md., Mr., Dr.). **Addresses:** "house/basa/flat N, road N, Area".

### Scores and conformal risk control

`detect(text, threshold=0.5)` masks spans with `score >= threshold`; `candidates(text)` returns all spans with scores. The scores are hand-set per rule (0.3–0.99), not learned. Proposal §4.6 picks the masking threshold with conformal risk control on a calibration set, so the threshold here is a placeholder until that calibration runs.

## Results (synthetic, hand-made; not paper results)

| Set | Cases | Gold spans | Recall (IoU ≥ 0.5) | Precision | Leakage | Over-masking |
|---|---:|---:|---:|---:|---:|---:|
| `pii_cases.jsonl` (written together with the detector) | 34 | 30 | 1.00 | 1.00 | 0.0% | 0.0% |
| `pii_cases_hard.jsonl` (written afterwards, not tuned on) | 18 | 15 | 0.47 | 0.88 | 46.7% | 0.0% |

The first row is a regression test only: the detector was written with those cases in view.
The hard set is the honest picture. Missed there:
- names without a keyword (`rahim bhai ke bolo`, `ami sumaiya`, `Tanvir Ahmed er sathe`)
- addresses not in "house N, road N" form (`mirpur 10, block c, road 7`)
- obfuscated e-mail (`test dot user at example dot com`), "double seven" in spoken numbers
- an OTP whose keyword comes after the number (`ei 553901 code ta`)
- `1712345678` (phone without the leading 0) is masked, but typed ID_NUMBER instead of PHONE

These are the cases a rule-based system cannot cover well and where the generative sanitizer has to do better (kill criterion K4). Real measurements come from the human gold set and the BanglishRev real-PII slice.

## Reproduce

```bash
python scripts/make_pii_cases.py
python scripts/eval_pii.py data/synthetic/pii_cases.jsonl
python scripts/eval_pii.py data/synthetic/pii_cases_hard.jsonl --show-errors
python -m pytest -q tests/test_pii_detector.py tests/test_pii_arbitrate.py
```
