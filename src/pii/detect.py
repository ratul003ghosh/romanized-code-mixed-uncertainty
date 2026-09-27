"""Rule-based PII detection and masking (README pipeline stage "PII Masking"; proposal 4.6 rule 1).

    from src.pii.detect import detect, mask
    spans = detect("amar NID 1234567890, TrxID 8N7A6D5E4F")
    masked, pii = mask(text, spans)      # "amar NID <NID_1>, TrxID <TXN_ID_1>", schema v0.2 pii items

Covers the ten schema PII types and the formats that Romanized Bangla actually uses:
  - Bangla digits (০১৭...), digit-letter look-alikes next to digits (O1712..., 0171234567l)
  - delimiters inside numbers (017-123-45678, 4539 1488 0343 6467)
  - spoken numbers, English or Banglish (zero one seven ..., shunno ek shat ...)
  - postpositions glued to the value (01712345678e, TrxID 8N7A6D5E4Fta)
Amounts (500 tk, 5oo taka) are never masked.

Classification follows the precedence rules of proposal 2.1: a context word decides the type
(NID, account, OTP, transaction ID, card); a phone-shaped number is PHONE; a Luhn-valid 16-digit
number is CARD; any other digit string of 10+ digits is ID_NUMBER and is masked conservatively.

This is the pipeline's deterministic detector, owned with the PII module. The regex baseline B1
(src/baselines/regex_baseline.py, PR #24) is the separate *comparison* system in the evaluation.
"""
from __future__ import annotations

import re

from src.pii.generators import luhn_valid
from src.pii.schema import PII_TYPES

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
LOOKALIKE = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8"}
NUMBER_WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine".split())}
NUMBER_WORDS.update({w: str(i) for i, w in enumerate("shunno ek dui tin char pach choy shat aat noy".split())})
NUMBER_WORDS.update({"shunyo": "0", "sunno": "0", "panch": "5", "chhoy": "6", "sat": "7", "at": "8", "nay": "9"})

CTX = {
    "NID": re.compile(r"\b(nid|n\.i\.d|national id|voter id|smart ?card|jatiyo porichoy)\b"),
    "ACCOUNT": re.compile(r"\b(account|acc|a/c|ac no|acct|bank)\b"),
    "OTP": re.compile(r"\b(otp|pin|code|verification)\b"),
    "TXN_ID": re.compile(r"\b(t(?:rx|xr|nx|rnx|xn|x)\s?id|tran?s?(?:action)?\s?id|reference|ref no)\b"),
    "CARD": re.compile(r"\b(card|visa|master ?card)\b"),
}
CURRENCY_AFTER = re.compile(r"\s{0,2}(tk\b|tk\.|taka|bdt|৳|rs\b|rupee)", re.I)
CURRENCY_BEFORE = re.compile(r"(tk|taka|bdt|৳|rs)\.?\s{0,2}$", re.I)

NUM_RUN = re.compile(r"(?<![0-9])\+?[0-9](?:(?:\s?-\s?|[ .])?[0-9])+")
SPOKEN = re.compile(r"\b(?:%s)(?:\s+(?:%s)){3,}\b" % (("|".join(sorted(NUMBER_WORDS, key=len, reverse=True)),) * 2),
                    re.I)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*?"
                   r"\.(?:com\.bd|ac\.bd|gov\.bd|org\.bd|com|net|org|edu|info|io|bd|me|co)")
TXN_TOKEN = re.compile(r"(?i:t(?:rx|xr|nx|rnx|xn|x)\s?id|tran?s?(?:action)?\s?id)\s*[:#-]?\s*"
                       r"([A-Z0-9]{3,12}(?:(?:\s?-\s?|[ .])[A-Z0-9]{1,6}){0,3})")
NAME_CTX = re.compile(r"\b(?:(?:amar\s+)?na+(?:m|rn)a?|my name is|ami)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")
NAME_BEFORE_KE = re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+?)(?=\s?ke\b)")
ADDRESS = re.compile(r"\b(?:house|house no\.?|h#|basa|bari)\s*\d+[A-Za-z]?,?\s*(?:road|rd|lane)\s*\d+[A-Za-z]?"
                     r"(?:,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)?){0,2}", re.I)
# Only "ke" is stripped: for -e/-re/-te/-ta the word boundary is ambiguous (Akter+e vs Akte+re), and a
# masker should rather cover one extra letter than leave part of a name visible.
GLUED_SUFFIX = re.compile(r"(?<=[a-z]{3})(ke)$")

PRIORITY = {"EMAIL": 0, "ADDRESS": 1, "TXN_ID": 2, "spoken": 3, "number": 4, "NAME": 5}


def _digit_view(text: str) -> str:
    """Same length as text: Bangla digits -> ASCII, look-alike letters next to digits -> digits."""
    chars = list(text.translate(BN_DIGITS))
    for _ in range(2):                                   # second pass catches OO1..., ...1ll
        for i, c in enumerate(chars):
            if c in LOOKALIKE:
                left = chars[i - 1] if i else ""
                right = chars[i + 1] if i + 1 < len(chars) else ""
                if left.isdigit() or right.isdigit():
                    chars[i] = LOOKALIKE[c]
    return "".join(chars)


def _context(low: str, start: int) -> str:
    """Up to 30 characters before a value, cut at the last clause boundary (. ! ? ; , and the danda)."""
    window = low[max(0, start - 30):start]
    return re.split(r"[.!?;,\u0964]", window)[-1]


def _classify(digits: str, context: str) -> str | None:
    n = len(digits)
    if re.fullmatch(r"(?:88)?01[3-9]\d{8}", digits):
        return "PHONE"
    if CTX["OTP"].search(context) and 4 <= n <= 6:
        return "OTP"
    if CTX["TXN_ID"].search(context) and 6 <= n <= 12:
        return "TXN_ID"
    if CTX["NID"].search(context) and n in (10, 13, 17):
        return "NID"
    if n == 16 and (luhn_valid(digits) or CTX["CARD"].search(context)):
        return "CARD"
    if CTX["ACCOUNT"].search(context) and 10 <= n <= 18:
        return "ACCOUNT"
    if n >= 10:
        return "ID_NUMBER"                               # proposal 2.1: identifier-like, mask conservatively
    return None


def _strip_glued(text: str, start: int, end: int) -> int:
    """Drop a glued "ke" from the last word of a name or address (Mitu Royke -> Mitu Roy)."""
    m = GLUED_SUFFIX.search(text[start:end])
    return end - len(m.group(1)) if m else end


def _candidates(text: str) -> list[tuple[int, int, int, str]]:
    low = text.lower()
    view = _digit_view(text)
    out: list[tuple[int, int, int, str]] = []

    for m in EMAIL.finditer(text):
        out.append((PRIORITY["EMAIL"], m.start(), m.end(), "EMAIL"))
    for m in ADDRESS.finditer(text):
        out.append((PRIORITY["ADDRESS"], m.start(), _strip_glued(text, m.start(), m.end()), "ADDRESS"))
    for m in TXN_TOKEN.finditer(text.translate(BN_DIGITS)):
        core = re.sub(r"[^A-Z0-9]", "", m.group(1))
        if len(core) >= 5 and (re.search(r"\d", core) or len(core) >= 8):
            out.append((PRIORITY["TXN_ID"], m.start(1), m.end(1), "TXN_ID"))
    for m in SPOKEN.finditer(text):
        digits = "".join(NUMBER_WORDS[w.lower()] for w in m.group(0).split())
        t = _classify(digits, _context(low, m.start()))
        if t:
            out.append((PRIORITY["spoken"], m.start(), m.end(), t))
    for m in NUM_RUN.finditer(view):
        s, e = m.start(), m.end()
        if CURRENCY_AFTER.match(text, e) or CURRENCY_BEFORE.search(text[max(0, s - 6):s]):
            continue                                     # an amount, never PII
        digits = re.sub(r"\D", "", m.group(0))
        t = _classify(digits, _context(low, s))
        if t:
            out.append((PRIORITY["number"], s, e, t))
    for rx in (NAME_CTX, NAME_BEFORE_KE):
        for m in rx.finditer(text):
            out.append((PRIORITY["NAME"], m.start(1), _strip_glued(text, m.start(1), m.end(1)), "NAME"))
    return out


def detect(text: str, types: set[str] | None = None) -> list[dict]:
    """Non-overlapping PII spans, sorted by position: [{type, text, start, end}]."""
    chosen: list[tuple[int, int, str]] = []
    for _, s, e, t in sorted(_candidates(text)):
        if (types is None or t in types) and not any(s < ce and cs < e for cs, ce, _ in chosen):
            chosen.append((s, e, t))
    return [{"type": t, "text": text[s:e], "start": s, "end": e} for s, e, t in sorted(chosen)]


def mask(text: str, spans: list[dict]) -> tuple[str, list[dict]]:
    """Replace spans with <TYPE_N> placeholders; the same value gets the same placeholder.

    Returns (masked_text, pii) where pii items follow schema v0.2 (type, placeholder, text, start, end).
    """
    counters: dict[str, int] = {}
    placeholder_of: dict[tuple[str, str], str] = {}
    pii = []
    for sp in sorted(spans, key=lambda x: x["start"]):
        key = (sp["type"], sp["text"])
        if key not in placeholder_of:
            counters[sp["type"]] = counters.get(sp["type"], 0) + 1
            placeholder_of[key] = f"<{sp['type']}_{counters[sp['type']]}>"
        pii.append({"type": sp["type"], "placeholder": placeholder_of[key], "text": sp["text"],
                    "start": sp["start"], "end": sp["end"]})
    masked = text
    for p in sorted(pii, key=lambda p: p["start"], reverse=True):
        masked = masked[:p["start"]] + p["placeholder"] + masked[p["end"]:]
    return masked, pii


assert set(t for t in PRIORITY if t.isupper()) | {"PHONE", "NID", "ID_NUMBER", "ACCOUNT", "CARD", "OTP"} == set(PII_TYPES)
