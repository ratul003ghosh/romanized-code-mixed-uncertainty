"""Build silver input records with exact character offsets (proposal v2, section 3.2).

A prompt is assembled from segments (plain text, PII, preserved entities). Noise is
applied to segments before joining, and offsets are computed while joining, so
`record["input"][start:end] == span` holds for every entity by construction.

Records are *inputs* for the teacher ensemble. They are not training targets:
normalization, uncertainty channels and routing come from the teachers (section 4).
`pii_reference()` gives a schema-valid, PII-only reference used for PII span F1,
leakage checks and baseline B1.
"""
from __future__ import annotations

import datetime
import random
import re
from dataclasses import dataclass, field

from src.noise.noise import noise_amount, noise_text
from src.pii_inject import templates as T
from src.pii_inject.generators import PII_TYPES, REGIMES, generate, render

GENERATOR_VERSION = "pii_inject-0.1.0"
SLOT_RE = re.compile(r"\{([A-Z_]+)\}")


@dataclass
class Segment:
    text: str
    kind: str = "plain"          # plain | pii | preserved
    etype: str | None = None     # schema type, e.g. PHONE or AMOUNT
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
    seg = Segment(text, "preserved", "AMOUNT", {"canonical": f"{value} BDT"})
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
            value = generate(slot, rng)
            r = render(slot, value, rng.choices(regimes, weights)[0], rng)
            segs.append(Segment(r.surface, "pii", slot,
                                {"canonical": r.canonical, "regime": r.regime}))
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
            segs.append(Segment(rng.choice(T.SERVICES), "preserved", "SERVICE"))
        else:
            raise ValueError(f"unknown slot {{{slot}}} in template: {template}")
        pos = m.end()
    if pos < len(template):
        segs.append(Segment(template[pos:]))
    return _fix_attached_spacing(segs), hints


def _fix_attached_spacing(segs: list[Segment]) -> list[Segment]:
    """An attached postposition replaces a following ' e ' / ' te ' etc. instead of doubling it."""
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


def join(segs: list[Segment], rng: random.Random, p_word_noise: float) -> tuple[str, list, list, list]:
    """Apply noise to plain segments, join, and compute offsets."""
    text, pii, preserved, noise_ops = "", [], [], []
    for s in segs:
        piece = s.text
        if s.kind == "plain" and not s.info.get("attached") and p_word_noise > 0:
            piece, ops = noise_text(piece, rng, p_word_noise)
            noise_ops += ops
        start = len(text)
        text += piece
        if s.kind == "pii":
            pii.append({"type": s.etype, "span": piece, "start": start, "end": start + len(piece),
                        "canonical": s.info["canonical"], "regime": s.info["regime"]})
        elif s.kind == "preserved":
            ent = {"type": s.etype, "value": piece, "start": start, "end": start + len(piece)}
            if "canonical" in s.info:
                ent["canonical"] = s.info["canonical"]
            preserved.append(ent)
    return text, pii, preserved, noise_ops


def _record(rid, source, split, surface_form, dialect, text, pii, preserved,
            noise_ops, hints, seed, meta_extra) -> dict:
    return {
        "id": rid, "source": source, "split": split,
        "surface_form": surface_form, "dialect": dialect,
        "input": text, "pii": pii, "preserved_entities": preserved,
        "hints": hints, "noise_applied": sorted(set(noise_ops)),
        "meta": {"generator": GENERATOR_VERSION, "seed": seed, "synthetic": True,
                 "date": datetime.date.today().isoformat(), **meta_extra},
    }


def make_template_record(i: int, rng: random.Random, *, seed: int,
                         regime_weights: dict[str, float] | None = None,
                         p_word_noise: float = 0.08, meta_extra: dict | None = None) -> dict:
    regime_weights = regime_weights or {"canonical": 0.4, "spoken": 0.25, "perturbed": 0.35}
    template = rng.choice(T.FINANCE_TEMPLATES)
    segs, hints = fill_template(template, rng, regime_weights)
    text, pii, preserved, ops = join(segs, rng, p_word_noise)
    return _record(f"synth-template-{i:06d}", "synthetic_template", "silver", "banglish",
                   None, text, pii, preserved, ops, hints, seed, meta_extra or {})


def make_carrier_record(i: int, carrier: dict, rng: random.Random, *, seed: int,
                        n_clauses: int = 1, regime_weights: dict[str, float] | None = None,
                        p_word_noise: float = 0.0, meta_extra: dict | None = None) -> dict:
    """Attach PII clauses to a sentence from a real corpus (the carrier stays unchanged
    unless p_word_noise > 0). `carrier` needs "text" and may carry "source", "id",
    "surface_form", "dialect"."""
    regime_weights = regime_weights or {"canonical": 0.4, "spoken": 0.25, "perturbed": 0.35}
    segs: list[Segment] = [Segment(carrier["text"].strip())]
    hints: list[dict] = []
    for _ in range(n_clauses):
        clause, h = fill_template(rng.choice(T.PII_CLAUSES), rng, regime_weights)
        hints += h
        if rng.random() < 0.5:
            segs = segs + [Segment(rng.choice([", ", ". ", " "]))] + clause
        else:
            segs = clause + [Segment(rng.choice([", ", ". ", " "]))] + segs
    text, pii, preserved, ops = join(segs, rng, p_word_noise)
    src = carrier.get("source", "carrier")
    rid = f"{src}-{carrier.get('id', i)}-pii{i:06d}"
    rec = _record(rid, src, carrier.get("split", "silver"), carrier.get("surface_form", "unknown"),
                  carrier.get("dialect"), text, pii, preserved, ops, hints, seed, meta_extra or {})
    rec["meta"]["synthetic"] = "pii_only"      # the carrier text is real, the PII is synthetic
    return rec


def pii_reference(record: dict) -> dict:
    """Schema-valid output containing only the PII masking (for PII F1, leakage, B1).

    sanitized_prompt is the *input* with PII replaced by placeholders; it is not a
    normalized target. uncertainties are left empty and routing is PROCEED.
    """
    text = record["input"]
    counters: dict[str, int] = {}
    placeholder_of: dict[tuple[str, str], str] = {}
    pii_items = []
    for p in sorted(record["pii"], key=lambda p: p["start"]):
        key = (p["type"], p["span"])
        if key not in placeholder_of:
            counters[p["type"]] = counters.get(p["type"], 0) + 1
            placeholder_of[key] = f"<{p['type']}_{counters[p['type']]}>"
            pii_items.append({"type": p["type"], "placeholder": placeholder_of[key], "span": p["span"]})
    masked = text
    for p in sorted(record["pii"], key=lambda p: p["start"], reverse=True):
        masked = masked[:p["start"]] + placeholder_of[(p["type"], p["span"])] + masked[p["end"]:]
    return {
        "sanitized_prompt": masked,
        "pii": pii_items,
        "preserved_entities": [{"type": e["type"], "value": e["value"]}
                               for e in record["preserved_entities"]],
        "uncertainties": [],
        "routing": "PROCEED",
    }
