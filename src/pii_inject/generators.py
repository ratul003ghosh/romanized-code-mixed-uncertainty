"""Synthetic PII values and their surface renderings (proposal v2, section 3.2).

Every value is random and synthetic. Type names follow the team schema v0.2
(docs/annotation-schema.md): PHONE, NID, ID_NUMBER, TXN_ID, ACCOUNT, CARD, OTP, EMAIL, NAME, ADDRESS.

Each generator returns a canonical value. `render()` turns it into the text that
appears in the prompt, in one of three regimes:
  canonical  01712345678, 4539 1488 0343 6467
  spoken     digits as words: "zero one seven ...", "shunno ek shat ..."
  perturbed  delimiters, digit-letter swaps, Bangla-script digits, an attached postposition

Formats marked VERIFY are assumptions; check them against real examples and record
the final choice in docs/decisions.md.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass

PII_TYPES = ("PHONE", "NID", "ID_NUMBER", "TXN_ID", "ACCOUNT", "CARD",
             "OTP", "EMAIL", "NAME", "ADDRESS")
REGIMES = ("canonical", "spoken", "perturbed")

# Types whose value is (mostly) digits, so a spoken rendering makes sense
DIGIT_TYPES = {"PHONE", "NID", "ID_NUMBER", "ACCOUNT", "CARD", "OTP"}

EN_DIGITS = "zero one two three four five six seven eight nine".split()
BN_ROMAN_DIGITS = "shunno ek dui tin char pach choy shat aat noy".split()
BN_SCRIPT_DIGITS = "০১২৩৪৫৬৭৮৯"
DIGIT_TO_LETTER = {"0": ["O", "o"], "1": ["l", "I"], "5": ["S"], "8": ["B"]}
POSTPOSITIONS = ["e", "te", "ta", "r", "re"]

MALE_FIRST = ["Rahim", "Karim", "Tanvir", "Sabbir", "Rafi", "Arif", "Hasan", "Sakib",
              "Mahmud", "Nayeem", "Rakib", "Shuvo", "Imran", "Sourav", "Jewel", "Fahim"]
FEMALE_FIRST = ["Fatema", "Ayesha", "Nusrat", "Mitu", "Sumaiya", "Jannat", "Priya", "Tania",
                "Shirin", "Farhana", "Moushumi", "Sadia", "Rumana", "Lima", "Tasnim", "Puja"]
SHARED_LAST = ["Hossain", "Rahman", "Ahmed", "Islam", "Chowdhury", "Khan", "Das", "Saha",
               "Sarker", "Roy", "Talukder", "Sheikh"]
MALE_LAST = SHARED_LAST + ["Uddin", "Miah"]
FEMALE_LAST = SHARED_LAST + ["Akter", "Begum", "Khatun"]
FIRST_NAMES = MALE_FIRST + FEMALE_FIRST
LAST_NAMES = sorted(set(MALE_LAST + FEMALE_LAST))
AREAS = ["Mirpur", "Dhanmondi", "Uttara", "Mohammadpur", "Agrabad", "Zindabazar", "Khulshi",
         "Shaheb Bazar", "Boalia", "Sonadanga", "Gulshan", "Banani", "Motijheel", "Kazir Dewri"]
CITIES = ["Dhaka", "Chattogram", "Sylhet", "Rajshahi", "Khulna", "Barishal", "Rangpur", "Cumilla"]
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]


@dataclass
class Rendered:
    surface: str            # exact text inserted into the prompt
    canonical: str          # underlying value, e.g. digits only
    regime: str             # regime actually used (may fall back to canonical)
    attach_suffix: str = ""  # postposition glued directly after the span (not part of it)


# ---------------------------------------------------------------- value generators

def _digits(rng: random.Random, n: int, first_nonzero: bool = True) -> str:
    first = rng.choice("123456789") if first_nonzero else rng.choice(string.digits)
    return first + "".join(rng.choice(string.digits) for _ in range(n - 1))


def luhn_check_digit(partial: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(partial)):
        d = int(ch)
        if i % 2 == 0:      # these positions are doubled once the check digit is appended
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return len(digits) > 1 and total % 10 == 0


def gen_phone(rng):          # BD mobile: 01[3-9] + 8 digits
    return "01" + rng.choice("3456789") + _digits(rng, 8, first_nonzero=False)


def gen_nid(rng):            # 10 (smart card), 13, or 17 digits (older, starts with birth year)
    kind = rng.choice([10, 13, 17])
    if kind == 17:
        return str(rng.randint(1950, 2005)) + _digits(rng, 13, first_nonzero=False)
    return _digits(rng, kind)


def gen_id_number(rng):      # bare identifier of unclear type (PII_BOUNDARY), 10-13 digits
    return _digits(rng, rng.choice([10, 11, 12, 13]))


def gen_account_number(rng):  # bank account, VERIFY typical lengths per bank
    return _digits(rng, rng.choice([13, 16, 17]))


def gen_card_number(rng):    # Visa (4) or Mastercard (51-55), Luhn-valid
    prefix = "4" if rng.random() < 0.5 else "5" + rng.choice("12345")
    body = prefix + _digits(rng, 15 - len(prefix), first_nonzero=False)
    return body + luhn_check_digit(body)


def gen_txn_id(rng):         # VERIFY: bKash ~10 upper alnum, Nagad ~8 upper alnum, Rocket digits
    style = rng.choice(["bkash", "nagad", "rocket"])
    alnum = string.ascii_uppercase + string.digits
    if style == "bkash":
        return "".join(rng.choice(alnum) for _ in range(10))
    if style == "nagad":
        return "".join(rng.choice(alnum) for _ in range(8))
    return _digits(rng, 10)


def gen_otp(rng):
    return _digits(rng, rng.choice([4, 5, 6]), first_nonzero=False)


def gen_email(rng):
    first, last = gen_person_name(rng).lower().split()
    name = first + rng.choice(["", ".", "_"]) + last
    return f"{name}{rng.randint(1, 999)}@{rng.choice(EMAIL_DOMAINS)}"


def gen_person_name(rng):
    if rng.random() < 0.5:
        return f"{rng.choice(MALE_FIRST)} {rng.choice(MALE_LAST)}"
    return f"{rng.choice(FEMALE_FIRST)} {rng.choice(FEMALE_LAST)}"


def gen_address(rng):
    return (f"house {rng.randint(1, 120)}, road {rng.randint(1, 40)}, "
            f"{rng.choice(AREAS)}, {rng.choice(CITIES)}")


GENERATORS = {
    "PHONE": gen_phone, "NID": gen_nid, "ID_NUMBER": gen_id_number,
    "ACCOUNT": gen_account_number, "CARD": gen_card_number,
    "TXN_ID": gen_txn_id, "OTP": gen_otp, "EMAIL": gen_email,
    "NAME": gen_person_name, "ADDRESS": gen_address,
}


def generate(pii_type: str, rng: random.Random) -> str:
    return GENERATORS[pii_type](rng)


# ---------------------------------------------------------------- regime renderers

def _canonical_surface(pii_type: str, value: str, rng: random.Random) -> str:
    if pii_type == "PHONE" and rng.random() < 0.25:
        return "+88" + value
    if pii_type == "CARD" and rng.random() < 0.5:
        return " ".join(value[i:i + 4] for i in range(0, len(value), 4))
    return value


def _spoken_surface(value: str, rng: random.Random) -> str:
    words = EN_DIGITS if rng.random() < 0.5 else BN_ROMAN_DIGITS
    return " ".join(words[int(c)] if c.isdigit() else c for c in value)


def _perturbed_surface(value: str, rng: random.Random) -> tuple[str, str]:
    ops = rng.sample(["delimit", "swap", "attach", "bn_digits"], k=rng.choice([1, 2]))
    if "bn_digits" in ops and "swap" in ops:     # a swapped letter cannot also be a Bangla digit
        ops.remove("swap")
    s = value
    if "delimit" in ops and len(s) >= 6:
        sep = rng.choice(["-", " ", ".", " - "])
        size = rng.choice([3, 4, 5])
        s = sep.join(s[i:i + size] for i in range(0, len(s), size))
    if "swap" in ops:
        idx = [i for i, c in enumerate(s) if c in DIGIT_TO_LETTER]
        for i in rng.sample(idx, k=min(len(idx), rng.choice([1, 2]))):
            s = s[:i] + rng.choice(DIGIT_TO_LETTER[s[i]]) + s[i + 1:]
    if "bn_digits" in ops:
        s = "".join(BN_SCRIPT_DIGITS[int(c)] if c.isdigit() else c for c in s)
    suffix = rng.choice(POSTPOSITIONS) if "attach" in ops else ""
    if s == value and not suffix:    # make sure something actually changed
        suffix = rng.choice(POSTPOSITIONS)
    return s, suffix


def render(pii_type: str, value: str, regime: str, rng: random.Random) -> Rendered:
    """Render a canonical value in the requested regime.

    Non-digit types have no spoken form; they fall back to canonical. Names, e-mails
    and addresses are only perturbed by an attached postposition.
    """
    if regime == "spoken" and pii_type in DIGIT_TYPES:
        return Rendered(_spoken_surface(value, rng), value, "spoken")
    if regime == "perturbed":
        if pii_type in {"NAME", "EMAIL", "ADDRESS"}:
            return Rendered(value, value, "perturbed", rng.choice(POSTPOSITIONS))
        surface, suffix = _perturbed_surface(value, rng)
        return Rendered(surface, value, "perturbed", suffix)
    return Rendered(_canonical_surface(pii_type, value, rng), value, "canonical")
