"""Light, safe cleaning. Never changes words, spelling, digits or case."""
import re
import unicodedata


def clean_text(text):
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)            # one standard form for Bangla letters
    text = text.replace("\u200b", "").replace("\ufeff", "")  # invisible junk characters
    # NOTE: we keep \u200c and \u200d, because Bangla script uses them
    text = text.replace("\u2026", "...")                  # the single "…" character
    text = re.sub(r"\.{4,}", "...", text)                # "......." -> "..."
    text = re.sub(r"\s+", " ", text)                      # many spaces/tabs/newlines -> one space
    return text.strip()
