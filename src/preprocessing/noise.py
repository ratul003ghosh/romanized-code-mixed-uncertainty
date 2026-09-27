"""Input noise for Banglish text (proposal v2, section 3.2 step 4).

Noise is applied per text segment *before* segments are joined, so character
offsets of PII spans are computed afterwards and never need remapping.

  typo            drop, duplicate, or swap adjacent characters inside a word
  banglish_vowel  common Romanization variation: o<->u, i<->e, sh<->s, ph<->f
  ocr             rn<->m, cl<->d, l<->1, O<->0 (only inside words)
  digit_letter    digits turned into look-alike letters in amounts: 500 -> 5oo
"""
from __future__ import annotations

import random
import re

VOWEL_SWAPS = [("o", "u"), ("u", "o"), ("i", "e"), ("e", "i"), ("sh", "s"), ("ph", "f"), ("oo", "u")]
OCR_SWAPS = [("rn", "m"), ("m", "rn"), ("cl", "d"), ("l", "1"), ("O", "0")]
AMOUNT_DIGIT_LETTER = {"1": "l", "5": "S"}
WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _typo(word: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice(["drop", "dup", "swap"])
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "dup":
        return word[:i] + word[i] + word[i:]
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def _replace_once(word: str, pairs, rng: random.Random) -> str:
    options = [(a, b) for a, b in pairs if a in word]
    if not options:
        return word
    a, b = rng.choice(options)
    starts = [m.start() for m in re.finditer(re.escape(a), word)]
    s = rng.choice(starts)
    return word[:s] + b + word[s + len(a):]


def noise_text(text: str, rng: random.Random, p_word: float = 0.08) -> tuple[str, list[str]]:
    """Apply word-level noise with probability p_word per word. Returns (text, ops applied)."""
    applied: list[str] = []

    def one(m: re.Match) -> str:
        word = m.group(0)
        if rng.random() >= p_word:
            return word
        op = rng.choice(["typo", "banglish_vowel", "ocr"])
        new = {"typo": lambda w: _typo(w, rng),
               "banglish_vowel": lambda w: _replace_once(w, VOWEL_SWAPS, rng),
               "ocr": lambda w: _replace_once(w, OCR_SWAPS, rng)}[op](word)
        if new != word:
            applied.append(op)
        return new

    return WORD_RE.sub(one, text), applied


def noise_amount(number_text: str, rng: random.Random) -> str | None:
    """Look-alike letters in an amount, the way people type it: 500 -> 5oo, 20000 -> 2oooo.

    Zeros after the first digit are replaced together (sometimes only the last two);
    otherwise a single 1 or 5 is swapped. Returns None when nothing can be swapped.
    """
    zeros = [i for i, c in enumerate(number_text) if c == "0" and i > 0]
    if zeros:
        if len(zeros) > 2 and rng.random() < 0.3:
            zeros = zeros[-2:]
        letter = "o" if rng.random() < 0.85 else "O"
        return "".join(letter if i in zeros else c for i, c in enumerate(number_text))
    idx = [i for i, c in enumerate(number_text) if c in AMOUNT_DIGIT_LETTER and i > 0]
    if not idx:
        return None
    i = rng.choice(idx)
    return number_text[:i] + AMOUNT_DIGIT_LETTER[number_text[i]] + number_text[i + 1:]
