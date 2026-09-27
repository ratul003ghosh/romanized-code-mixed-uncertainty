"""Build PII-injected input records in the team schema v0.2 (proposal v2, section 3.2).

A prompt is assembled from segments (plain text, PII, preserved entities). Noise is applied to
segments before joining and offsets are computed while joining, so every PII span satisfies
`clean_input[start:end] == text` by construction. Generated text is already clean
(`clean_text(input) == input`), so `input` and `clean_input` are identical and offsets hold for both.

Records are *inputs* for the teacher ensemble: normalized_text, sanitized_prompt and routing are
null and uncertainties are empty; the teachers fill them in (section 4). The injected PII is
stored in `pii` because it is known ground truth.

Proposed conventions, pending approval by Saber (dataset and schema owner):
  ids           BG_SYN_000001 (template), BG_PII_000001 / HG_PII_000001 (real text + synthetic PII)
  label_source  "synthetic" for template records, "none" for real text sent to the teachers
  pii.regime    optional key: canonical | spoken | perturbed (needed for per-regime leakage, 6.1)
"""
from __future__ import annotations

import datetime
import random
import re
from dataclasses import dataclass, field

from src.noise.noise import noise_amount, noise_text
from src.pii_inject import templates as T
from src.pii_inject.generators import PII_TYPES, generate, render
from src.pii_inject.v02 import SCHEMA_VERSION
from src.preprocessing.clean import clean_text

GENERATOR_VERSION = "pii_inject-0.2.0"
SLOT_RE = re.compile(r"\{([A-Z_]+)\}")
SERVICE_CANONICAL = {"bkash": "bKash", "bikash": "bKash", "nagad": "Nagad", "rocket": "Rocket", "upay": "Upay"}
DEFAULT_REGIMES = {"canonical": 0.4, "spoken": 0.25, "perturbed": 0.35}
MAX_TRIES = 20


@dataclass
class Segment:
    text: str
    kind: str = "plain"          # plain | pii | preserved
    etype: str | None = None     # PII type or preserved-entity type
    info: dict = field(default_factory=dict)


def _amount_segment(rng: random.Random, p_noise: float) -> tuple[Segment, dict | None]:
    value = rng.choice(T.AMOUNT_VALUES)
    number = f"{value:,}" if value >= 1000 and rng.random() < 0.3 else str(value)
    hint = None
    if rng.random() < p_noise:
        noised = noise_amount(number.replace(",", ""), rng)
        if noised:
            number, hint = noised, {"type": "NUMERIC_AMBIGUITY"}
    unit = rng.choice(T.AMOUNT_UNITS)
    text = f"{number}{'' if rng.random() < 0.15 else ' '}{unit}"
    seg = Segment(text, "preserved", "AMOUNT", {"value": f"{value} BDT"})
    return seg, (dict(hint, span=text) if hint else None)


def fill_template(template: str, rng: random.Random, regime_weights: dict[str, float],
                  p_amount_noise: float = 0.15) -> tuple[list[Segment], list[dict]]:
    """Turn a template into segments; returns (segments, hints)."""
    segs: list[Segment] = []
    hints: list[dict] = []
    pos = 0
    regimes, weights = zip(*regime_weights.items())
    for m in SLOT_RE.finditer(template):
        if m.start() > pos:
            segs.append(Segment(template[pos:m.start()]))
        slot = m.group(1)
        if slot in PII_TYPES:
            r = render(slot, generate(slot, rng), rng.choices(regimes, weights)[0], rng)
            segs.append(Segment(r.surface, "pii", slot, {"regime": r.regime}))
            if r.attach_suffix:
                segs.append(Segment(r.attach_suffix, "plain", info={"attached": True}))
            if slot == "ID_NUMBER":
                hints.append({"type": "PII_BOUNDARY", "span": r.surface})
        elif slot == "AMOUNT":
            seg, hint = _amount_segment(rng, p_amount_noise)
            segs.append(seg)
            if hint:
                hints.append(hint)
        elif slot == "SERVICE":
            name = rng.choice(T.SERVICES)
            segs.append(Segment(name, "preserved", "SERVICE", {"value": SERVICE_CANONICAL[name.lower()]}))
        else:
            raise ValueError(f"unknown slot {{{slot}}} in template: {template}")
        pos = m.end()
    if pos < len(template):
        segs.append(Segment(template[pos:]))
    return _fix_attached_spacing(segs), hints


def _fix_attached_spacing(segs: list[Segment]) -> list[Segment]:
    """An attached postposition replaces a following ' e ' / ' ke ' etc. instead of doubling it."""
    out: list[Segment] = []
    for i, s in enumerate(segs):
        if s.info.get("attached") and i + 1 < len(segs) and segs[i + 1].kind == "plain":
            nxt = segs[i + 1]
            m = re.match(r"\s+(e|te|ta|r|re|ke|er|theke)\b", nxt.text)
            if m:
                s = Segment(m.group(1), "plain", info={"attached": True})
                segs[i + 1] = Segment(nxt.text[m.end():])
        out.append(s)
    return out


def join(segs: list[Segment], rng: random.Random, p_word_noise: float):
    """Apply noise to plain segments, join, compute offsets and assign <TYPE_N> placeholders."""
    text, pii, preserved, noise_ops = "", [], [], []
    counters: dict[str, int] = {}
    placeholder_of: dict[tuple[str, str], str] = {}
    for s in segs:
        piece = s.text
        if s.kind == "plain" and not s.info.get("attached") and p_word_noise > 0:
            piece, ops = noise_text(piece, rng, p_word_noise)
            noise_ops += ops
        start = len(text)
        text += piece
        if s.kind == "pii":
            key = (s.etype, piece)
            if key not in placeholder_of:            # same value -> same placeholder within a prompt
                counters[s.etype] = counters.get(s.etype, 0) + 1
                placeholder_of[key] = f"<{s.etype}_{counters[s.etype]}>"
            pii.append({"type": s.etype, "placeholder": placeholder_of[key], "text": piece,
                        "start": start, "end": start + len(piece), "regime": s.info["regime"]})
        elif s.kind == "preserved":
            preserved.append({"type": s.etype, "value": s.info["value"]})
    return text, pii, preserved, noise_ops


def _record(rid, text, pii, preserved, *, language, surface_form, source, label_source, split,
            hints, noise_ops, seed, extra_meta) -> dict:
    return {
        "id": rid,
        "schema_version": SCHEMA_VERSION,
        "input": text,
        "clean_input": text,
        "normalized_text": None,
        "sanitized_prompt": None,
        "pii": pii,
        "preserved_entities": preserved,
        "uncertainties": [],
        "routing": None,
        "metadata": {
            "language": language, "surface_form": surface_form, "source": source,
            "label_source": label_source, "split": split,
            "generator": GENERATOR_VERSION, "seed": seed, "date": datetime.date.today().isoformat(),
            "hints": hints, "noise_applied": sorted(set(noise_ops)), **extra_meta,
        },
    }


def _build_clean(make_segments, rng, p_word_noise):
    """Retry until the joined text is already clean, so input == clean_input and offsets hold."""
    for _ in range(MAX_TRIES):
        segs, hints = make_segments()
        text, pii, preserved, ops = join(segs, rng, p_word_noise)
        if clean_text(text) == text:
            return text, pii, preserved, ops, hints
    raise ValueError("could not build a clean record")


def make_template_record(i: int, rng: random.Random, *, seed: int,
                         regime_weights: dict[str, float] | None = None,
                         p_word_noise: float = 0.08, meta_extra: dict | None = None) -> dict:
    """Fully synthetic Banglish finance prompt (label_source 'synthetic')."""
    weights = regime_weights or DEFAULT_REGIMES
    text, pii, preserved, ops, hints = _build_clean(
        lambda: fill_template(rng.choice(T.FINANCE_TEMPLATES), rng, weights), rng, p_word_noise)
    return _record(f"BG_SYN_{i:06d}", text, pii, preserved, language="banglish", surface_form="banglish",
                   source="synthetic_template", label_source="synthetic", split="train", hints=hints,
                   noise_ops=ops, seed=seed, extra_meta=meta_extra or {})


def make_carrier_record(i: int, carrier: dict, rng: random.Random, *, seed: int, n_clauses: int = 1,
                        regime_weights: dict[str, float] | None = None,
                        meta_extra: dict | None = None) -> dict:
    """A real sentence with synthetic PII clauses attached (label_source 'none': input to teachers).

    `carrier` needs "text" and may carry "id", "source", "language", "surface_form", "dialect",
    "split". The carrier text itself is only passed through clean_text.
    """
    weights = regime_weights or DEFAULT_REGIMES
    base = clean_text(carrier["text"])

    def make_segments():
        segs: list[Segment] = [Segment(base)]
        hints: list[dict] = []
        for _ in range(n_clauses):
            clause, h = fill_template(rng.choice(T.PII_CLAUSES), rng, weights)
            hints += h
            if rng.random() < 0.5:                       # clause after the sentence
                last = segs[-1].text
                sep = " " if last[-1:] in ".!?\u0964" else rng.choice([", ", ". ", " "])
                segs = segs + [Segment(sep)] + clause
            else:                                        # clause before the sentence
                segs = clause + [Segment(rng.choice([", ", ". ", " "]))] + segs
        return segs, hints

    text, pii, preserved, ops, hints = _build_clean(make_segments, rng, 0.0)
    language = carrier.get("language") or ("hinglish" if carrier.get("surface_form") == "hinglish" else "banglish")
    prefix = "HG" if language == "hinglish" else "BG"
    extra = {"source_id": carrier.get("id"),
             **({"dialect": carrier["dialect"]} if carrier.get("dialect") else {}),
             **(meta_extra or {})}
    return _record(f"{prefix}_PII_{i:06d}", text, pii, preserved, language=language,
                   surface_form=carrier.get("surface_form") or "banglish",
                   source=carrier.get("source", "carrier"), label_source="none",
                   split=carrier.get("split", "train"), hints=hints, noise_ops=ops, seed=seed,
                   extra_meta=extra)
