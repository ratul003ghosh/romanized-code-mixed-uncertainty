"""PII detection for Romanized / code-mixed Bangla prompts (issue #10).

    from src.pii.detector import detect
    detect("amar bkash 01712345678, trx id 8N7A6D5E4F")
    # [{"type": "PHONE", "text": "01712345678", "start": 11, "end": 22, "score": 0.95, "rule": "bd_mobile"},
    #  {"type": "TXN_ID", "text": "8N7A6D5E4F", "start": 31, "end": 41, "score": 0.95, "rule": "txn_keyword"}]

Every span carries a confidence `score` in [0, 1]. `detect()` returns spans with score >= threshold
(default 0.5). Lower-scored candidates (e.g. a bare 8-digit number) are still produced by
`candidates()`, so the masking threshold can later be calibrated with conformal risk control
(proposal §4.6) instead of being fixed by hand.

Type decisions follow proposal §2.1: a keyword in the same clause decides the type; a digit string
in an identity or account context is PII; a number with a currency marker (tk, taka, BDT, ৳) is an
AMOUNT and is never masked; a long bare digit string is ID_NUMBER and masked conservatively.

Offsets are character positions in the text passed in (schema v0.2: `clean_input`).
"""
from __future__ import annotations

import re

PII_TYPES = ("PHONE", "NID", "ID_NUMBER", "TXN_ID", "ACCOUNT", "CARD", "OTP", "EMAIL", "NAME", "ADDRESS")
DEFAULT_THRESHOLD = 0.5

# ---------------------------------------------------------------- normalization
_BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_LOOKS_LIKE_DIGIT = {"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1", "Z": "2", "S": "5", "B": "8"}


def digit_view(text: str) -> str:
    """Same length as `text`, so offsets are shared: Bangla digits become ASCII digits, and letters
    that look like digits become digits when they touch a digit (5oo -> 500, O17l... -> 0171...)."""
    chars = list(text.translate(_BANGLA_DIGITS))
    changed = True
    while changed:                      # repeat so runs like "OO1" or "1ll" are fully converted
        changed = False
        for i, c in enumerate(chars):
            if c in _LOOKS_LIKE_DIGIT and (
                    (i > 0 and chars[i - 1].isdigit()) or (i + 1 < len(chars) and chars[i + 1].isdigit())):
                chars[i] = _LOOKS_LIKE_DIGIT[c]
                changed = True
    return "".join(chars)


def luhn_ok(digits: str) -> bool:
    if not digits.isdigit() or len(digits) < 12:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


# ---------------------------------------------------------------- keyword context
_KEYWORDS = {
    "NID": r"nid|n\.i\.d|national\s*id|voter\s*id|smart\s*card|jatiyo\s*porichoy(?:\s*potro)?|porichoy\s*potro",
    "ACCOUNT": r"account|acc|acct|a/c|ac\s*no|hisab|bank",
    "OTP": r"otp|pin|code|verification|verify|secret",
    "TXN_ID": r"trx|trnx|txn|txr|tnx|trans(?:action)?|tran|reference|ref(?:\s*no)?",
    "CARD": r"card|visa|master\s*card|mastercard|debit|credit",
    "PHONE": r"phone|mobile|mob|number|nombor|num|nmbr|call|fon|whats\s*app|imo|bkash|bikash|nagad|rocket|recharge",
}
_KW = {t: re.compile(r"(?<![a-z])(?:%s)(?![a-z])" % p) for t, p in _KEYWORDS.items()}
_CLAUSE_BREAK = re.compile(r"[.!?;,\u0964\n]")


def _clause_before(low: str, start: int, width: int = 40) -> str:
    return _CLAUSE_BREAK.split(low[max(0, start - width):start])[-1]


def _has(kind: str, ctx: str) -> bool:
    return bool(_KW[kind].search(ctx))


# ---------------------------------------------------------------- amounts (never PII)
_CURRENCY_AFTER = re.compile(r"\s{0,2}(?:tk|taka|bdt|৳|rs|rupee|rupees|k)(?![a-z])", re.I)
_CURRENCY_BEFORE = re.compile(r"(?:tk|taka|bdt|৳|rs)\.?\s{0,2}$", re.I)


def _is_amount(text: str, start: int, end: int) -> bool:
    return bool(_CURRENCY_AFTER.match(text, end) or _CURRENCY_BEFORE.search(text[max(0, start - 8):start]))


# ---------------------------------------------------------------- numeric classification
def classify_digits(digits: str, ctx: str) -> tuple[str, float, str] | None:
    """(type, score, rule) for a digit string given the clause before it, or None."""
    n = len(digits)
    if re.fullmatch(r"(?:880|88)?01[3-9]\d{8}", digits) or (re.fullmatch(r"1[3-9]\d{8}", digits) and _has("PHONE", ctx)):
        return "PHONE", 0.95, "bd_mobile"
    if _has("OTP", ctx) and 4 <= n <= 8:
        return "OTP", 0.9, "otp_keyword"
    if _has("TXN_ID", ctx) and 6 <= n <= 20:
        return "TXN_ID", 0.9, "txn_keyword"
    if _has("NID", ctx) and n in (10, 13, 17):
        return "NID", 0.95, "nid_keyword"
    if _has("CARD", ctx) and 13 <= n <= 19:
        return "CARD", 0.95, "card_keyword"
    if 13 <= n <= 19 and luhn_ok(digits):
        return "CARD", 0.85, "luhn"
    if _has("ACCOUNT", ctx) and 8 <= n <= 20:
        return "ACCOUNT", 0.9, "account_keyword"
    if _has("NID", ctx) and n >= 8:
        return "NID", 0.75, "nid_keyword_len"
    if n >= 10:
        return "ID_NUMBER", 0.7, "long_digits"          # §2.1: identifier-like, mask conservatively
    if 7 <= n <= 9:
        return "ID_NUMBER", 0.3, "medium_digits"        # below default threshold; kept for calibration
    return None


# ---------------------------------------------------------------- patterns
# digit runs, allowing one separator (space, dash, dot) between digits: 017-1234-5678, 4111 1111 1111 1111
_DIGIT_RUN = re.compile(r"(?<!\d)\+?\d(?:[ \-.]?\d)+(?!\d)")
_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_TXN_AFTER_KW = re.compile(
    r"(?<![A-Za-z])(?:%s)\s*(?:id|no|number|nombor)?\s*(?:is|holo|:|#|-)?\s*([A-Za-z0-9]{5,20})(?![A-Za-z0-9])" % _KEYWORDS["TXN_ID"],
    re.I)
_CODE_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{8,14}(?![A-Za-z0-9])")
_NUMBER_WORDS = {
    **{w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine".split())},
    **{w: str(i) for i, w in enumerate("shunno ek dui tin char pach choy sat aat noy".split())},
    "shunyo": "0", "sunno": "0", "oh": "0", "panch": "5", "chhoy": "6", "shat": "7", "saat": "7", "at": "8", "nay": "9",
}
_WORD_ALT = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_SPOKEN = re.compile(r"(?<![a-z])(?:%s)(?:[\s,-]+(?:%s)){4,}(?![a-z])" % (_WORD_ALT, _WORD_ALT), re.I)
_NAME_STOP = r"(?:ami|amar|ar|and|er|ke|theke|number|nombor|nid|phone|account|bkash|nagad|ache|holo|is|ta|te|e)"
_NAME_AFTER_KW = re.compile(
    r"(?<![a-z])(?:amar\s+(?:nam|naam|name)|(?:nam|naam)\s+holo|my\s+name\s+is|name\s*:)(?:\s+(?:holo|is))?\s*:?\s+"
    r"((?!%s\b)[A-Za-z][a-z]{2,}(?:\s+(?!%s\b)[A-Za-z][a-z]{2,})?)" % (_NAME_STOP, _NAME_STOP), re.I)
_NAME_TITLE = re.compile(r"(?<![A-Za-z])(?:Md|Mohammad|Muhammad|Mr|Mrs|Ms|Dr)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")
_ADDRESS = re.compile(
    r"(?<![A-Za-z])(?i:house|house\s*no|h#|holding|basa|bari|flat)\s*[#:]?\s*\d+[A-Za-z]?"
    r"(?:\s*,?\s*(?i:road|rd|lane|goli|sector|block|avenue)\s*[#:]?\s*\d+[A-Za-z]?)*"
    r"(?:\s*,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)?){0,2}")   # area names only when capitalised


# ---------------------------------------------------------------- detection
def _run_variants(run: str, offset: int) -> list[tuple[int, int]]:
    """The whole digit run, plus the run without a short (1-2 digit) first or last group, so
    "2 01712345678" still yields the phone number; detect() keeps the best-scoring variant."""
    out = [(offset, offset + len(run))]
    groups = [(m.start(), m.end()) for m in re.finditer(r"\+?\d+", run)]
    if len(groups) > 1 and groups[0][1] - groups[0][0] <= 2:
        out.append((offset + groups[1][0], offset + len(run)))
    if len(groups) > 1 and groups[-1][1] - groups[-1][0] <= 2:
        out.append((offset, offset + groups[-2][1]))
    return out


def candidates(text: str) -> list[dict]:
    """All PII candidates with scores, possibly overlapping."""
    low, view = text.lower(), digit_view(text)
    out: list[dict] = []

    def add(t, s, e, score, rule):
        out.append({"type": t, "text": text[s:e], "start": s, "end": e, "score": score, "rule": rule})

    for m in _EMAIL.finditer(text):
        add("EMAIL", m.start(), m.end(), 0.99, "email")
    for m in _ADDRESS.finditer(text):
        add("ADDRESS", m.start(), m.end(), 0.85, "address")
    for m in _TXN_AFTER_KW.finditer(text):
        tok = m.group(1)
        if re.search(r"\d", tok) and (re.search(r"[A-Za-z]", tok) or len(tok) >= 6):
            add("TXN_ID", m.start(1), m.end(1), 0.95, "txn_keyword")
    for m in _CODE_TOKEN.finditer(text):                    # e.g. 8N7A6D5E4F with no keyword
        add("TXN_ID", m.start(), m.end(), 0.4, "code_token")
    for m in _SPOKEN.finditer(text):
        digits = "".join(_NUMBER_WORDS[w.lower()] for w in re.split(r"[\s,-]+", m.group(0)))
        hit = classify_digits(digits, _clause_before(low, m.start()))
        if hit:
            add(hit[0], m.start(), m.end(), hit[1], "spoken_" + hit[2])
    for m in _DIGIT_RUN.finditer(view):
        for s, e in _run_variants(m.group(0), m.start()):
            if _is_amount(text, s, e):
                continue
            digits = re.sub(r"\D", "", view[s:e])
            hit = classify_digits(digits, _clause_before(low, s))
            if hit:
                add(hit[0], s, e, hit[1], hit[2])
    for rx, score, rule in ((_NAME_AFTER_KW, 0.85, "name_keyword"), (_NAME_TITLE, 0.8, "name_title")):
        for m in rx.finditer(text):
            add("NAME", m.start(1), m.end(1), score, rule)
    return out


def detect(text: str, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Non-overlapping PII spans with score >= threshold, sorted by position.

    Overlaps are resolved by higher score, then longer span."""
    chosen: list[dict] = []
    for c in sorted(candidates(text), key=lambda c: (-c["score"], -(c["end"] - c["start"]), c["start"])):
        if c["score"] >= threshold and not any(c["start"] < k["end"] and k["start"] < c["end"] for k in chosen):
            chosen.append(c)
    return sorted(chosen, key=lambda c: c["start"])
