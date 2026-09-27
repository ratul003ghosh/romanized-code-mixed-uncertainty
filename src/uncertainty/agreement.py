"""Field-level agreement between teacher outputs (pivot medoid and discordance d(s))."""
import re
from difflib import SequenceMatcher

SIM_THRESHOLD = 0.8  # sanitized_prompt counts as agreeing at or above this similarity


def norm(s):
    s = re.sub(r"[^\w<>@.\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def overlaps(a, b):
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a in b or b in a)


def text_sim(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def entry_agrees(group, entry, other):
    """Does `other` (a validated teacher JSON) agree with pivot `entry` of `group`?"""
    if other is None:
        return False
    if group == "sanitized_prompt":
        return text_sim(entry, other["sanitized_prompt"]) >= SIM_THRESHOLD
    if group == "pii":
        return any(overlaps(entry["span"], o["span"]) and entry["type"] == o["type"] for o in other["pii"])
    if group == "preserved_entities":
        return any(entry["type"] == o["type"] and norm(entry["value"]) == norm(o["value"])
                   for o in other["preserved_entities"])
    if group == "ambiguous_spans":
        return any(overlaps(entry["span"], o["span"]) for o in other["ambiguous_spans"])
    raise ValueError(group)


def json_agreement(a, b):
    """Symmetric mean agreement over all units of a and b (used to pick the medoid)."""
    scores = [float(entry_agrees("sanitized_prompt", a["sanitized_prompt"], b))]
    for g in ("pii", "preserved_entities", "ambiguous_spans"):
        scores += [float(entry_agrees(g, e, b)) for e in a[g]]
        scores += [float(entry_agrees(g, e, a)) for e in b[g]]
    return sum(scores) / len(scores)


def discordance(group, entry, others):
    """d(s) = 1 - (1/M) * sum_m 1[teacher m agrees on the field]   (proposal Eq. 3)."""
    if not others:
        return None
    return 1.0 - sum(entry_agrees(group, entry, o) for o in others) / len(others)
