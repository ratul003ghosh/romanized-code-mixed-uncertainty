"""Runtime PII check on MODEL OUTPUT (proposal §4.6, routing rule 1).

The teacher/student writes a sanitized prompt. Before it is sent to a cloud model, this
deterministic check makes sure no raw PII survived:

  1. every PII value known from the input is searched for in the output, also when re-formatted
     (017-1234-5678, O1712345678, Bangla digits, +880...), and replaced by its input placeholder;
  2. the detector runs on the output itself, to catch PII the input step missed.

Anything found is replaced by a deterministic <TYPE_N> placeholder and reported. `leaked=True`
tells the decision layer to override the route (ASK_USER or PROCEED_WITH_FLAGS) and log the case.

    from src.pii.arbitrate import enforce
    res = enforce("Call 017-1234-5678 about <TXN_ID_1>", source_pii=record["pii"])
    # {"text": "Call <PHONE_1> about <TXN_ID_1>", "leaked": True, "replacements": [...]}
"""
from __future__ import annotations

import re

from src.pii.detector import detect, digit_view
from src.pii.masking import _NUMERIC, canonical

PLACEHOLDER = re.compile(r"<([A-Z_]+)_([1-9][0-9]*)>")
_MIN_LEN = 4                # shorter known values would match ordinary words


def _pattern(pii_type: str, value: str) -> tuple[re.Pattern, bool] | None:
    """(regex, search_in_digit_view) for a known input PII value."""
    if pii_type in _NUMERIC:
        digits = canonical(pii_type, value)
        if len(digits) < _MIN_LEN:
            return None
        prefix = r"(?:\+?88[\s.\-]?)?" if pii_type == "PHONE" else ""
        return re.compile(r"(?<!\d)" + prefix + r"[\s.\-]?".join(digits) + r"(?!\d)"), True
    if len(value.strip()) < _MIN_LEN:
        return None
    return re.compile(r"(?<!\w)" + re.escape(value.strip()) + r"(?!\w)", re.I), False


def _used_numbers(output: str, source_pii: list[dict]) -> dict[str, int]:
    used: dict[str, int] = {}
    for t, n in PLACEHOLDER.findall(output) + [
            PLACEHOLDER.fullmatch(p["placeholder"]).groups() for p in source_pii
            if PLACEHOLDER.fullmatch(p.get("placeholder") or "")]:
        used[t] = max(used.get(t, 0), int(n))
    return used


def enforce(output: str, source_pii: list[dict] | None = None) -> dict:
    """Mask raw PII left in `output`. Returns {"text", "leaked", "replacements"}; each replacement is
    {"type", "placeholder", "text", "reason"} with reason "source_value" or "detected"."""
    source_pii = source_pii or []
    used = _used_numbers(output, source_pii)
    view = digit_view(output)                           # same length as output
    blocked = [(m.start(), m.end()) for m in PLACEHOLDER.finditer(output)]
    hits: list[tuple[int, int, str, str, str]] = []

    def free(s, e):
        return not any(s < b and a < e for a, b in blocked + [(h[0], h[1]) for h in hits])

    def placeholder_for(pii_type, value, given=None):
        key = (pii_type, canonical(pii_type, value))
        if key not in assigned:
            if given:
                assigned[key] = given
            else:
                used[pii_type] = used.get(pii_type, 0) + 1
                assigned[key] = f"<{pii_type}_{used[pii_type]}>"
        return assigned[key]

    assigned: dict[tuple[str, str], str] = {}
    for p in source_pii:                                # 1. known input values
        found = _pattern(p["type"], p.get("text", ""))
        if not found:
            continue
        rx, in_view = found
        for m in rx.finditer(view if in_view else output):
            if free(m.start(), m.end()):
                hits.append((m.start(), m.end(), p["type"],
                             placeholder_for(p["type"], p["text"], p.get("placeholder")), "source_value"))
    for sp in detect(output):                           # 2. anything else the detector finds
        if free(sp["start"], sp["end"]):
            hits.append((sp["start"], sp["end"], sp["type"], placeholder_for(sp["type"], sp["text"]), "detected"))

    text = output
    for s, e, _, ph, _ in sorted(hits, reverse=True):
        text = text[:s] + ph + text[e:]
    reps = [{"type": t, "placeholder": ph, "text": output[s:e], "reason": r} for s, e, t, ph, r in sorted(hits)]
    return {"text": text, "leaked": bool(reps), "replacements": reps}
