"""PII masking with schema v0.2 placeholders (issue #11).

    from src.pii.masking import sanitize
    out = sanitize("amar nam rahim, rahim ke 01712345678 e call dao")
    out["masked"]  # "amar nam <NAME_1>, <NAME_1> ke <PHONE_1> e call dao"
    out["pii"]     # [{"type", "placeholder", "text", "start", "end", "score"}, ...]  (schema v0.2)

Rules (docs/annotation-schema.md):
  - placeholders are <TYPE_N>, numbered per prompt in order of first appearance;
  - the same PII value gets the same placeholder, also when written differently
    (017-1234-5678 and 01712345678 are the same phone number);
  - a detected value is masked everywhere it appears in the prompt (a name found once after
    "amar nam" is also masked where it appears again without the keyword).
"""
from __future__ import annotations

import re

from src.pii.detector import DEFAULT_THRESHOLD, detect, digit_view

_NUMERIC = {"PHONE", "NID", "ID_NUMBER", "ACCOUNT", "CARD", "OTP"}
_MIN_PROPAGATE = 3          # do not propagate values shorter than this (would hit ordinary words)


def canonical(pii_type: str, value: str) -> str:
    """Key that identifies the same PII value across different spellings."""
    if pii_type in _NUMERIC:
        digits = re.sub(r"\D", "", digit_view(value))
        if pii_type == "PHONE" and len(digits) == 13 and digits.startswith("880"):
            digits = digits[2:]
        return digits
    return re.sub(r"\s+", " ", value.strip().lower())


def _propagate(text: str, spans: list[dict]) -> list[dict]:
    """Add further occurrences of detected non-numeric values (names, emails, IDs, addresses)."""
    extra = []
    for sp in spans:
        if sp["type"] in _NUMERIC or len(sp["text"]) < _MIN_PROPAGATE:
            continue
        rx = re.compile(r"(?<!\w)" + re.escape(sp["text"]) + r"(?!\w)", re.I)
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            if not any(s < o["end"] and o["start"] < e for o in spans + extra):
                extra.append({**sp, "text": text[s:e], "start": s, "end": e, "rule": "repeat_of_" + sp["rule"]})
    return sorted(spans + extra, key=lambda x: x["start"])


def mask(text: str, spans: list[dict]) -> tuple[str, list[dict]]:
    """Replace non-overlapping spans with placeholders. Returns (masked_text, schema v0.2 pii list)."""
    numbers: dict[str, int] = {}
    placeholder_of: dict[tuple[str, str], str] = {}
    pii = []
    for sp in sorted(spans, key=lambda x: x["start"]):
        key = (sp["type"], canonical(sp["type"], sp["text"]))
        if key not in placeholder_of:
            numbers[sp["type"]] = numbers.get(sp["type"], 0) + 1
            placeholder_of[key] = f"<{sp['type']}_{numbers[sp['type']]}>"
        item = {"type": sp["type"], "placeholder": placeholder_of[key], "text": sp["text"],
                "start": sp["start"], "end": sp["end"]}
        if "score" in sp:
            item["score"] = sp["score"]
        pii.append(item)
    masked = text
    for p in sorted(pii, key=lambda p: p["start"], reverse=True):
        masked = masked[:p["start"]] + p["placeholder"] + masked[p["end"]:]
    return masked, pii


def sanitize(text: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Detect + mask in one call. Offsets in `pii` refer to `text` (schema v0.2: clean_input)."""
    masked, pii = mask(text, _propagate(text, detect(text, threshold)))
    return {"masked": masked, "pii": pii}
