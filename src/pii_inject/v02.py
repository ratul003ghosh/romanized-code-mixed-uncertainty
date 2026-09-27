"""Schema v0.2 rules for PII-injected input records (docs/annotation-schema.md, owner: Saber).

The team schema v0.2 is the single source of truth. This module only restates the parts
that PII injection produces, and `check_record()` reports anything that breaks them.
tests/test_pii_inject.py reads docs/annotation-schema.md and fails if these lists drift from it.
"""
from __future__ import annotations

import re

from src.preprocessing.clean import clean_text

SCHEMA_VERSION = "0.2"
PII_TYPES = ("PHONE", "NID", "ID_NUMBER", "TXN_ID", "ACCOUNT", "CARD", "OTP", "EMAIL", "NAME", "ADDRESS")
SURFACE_FORMS = {"banglish", "romanized", "code_mixed", "dialect", "hinglish"}
LABEL_SOURCES = {"silver", "gold", "synthetic", "illustrative", "none"}
SPLITS = {"train", "dev", "test", "none"}
ROUTES = {"PROCEED", "PROCEED_WITH_FLAGS", "ASK_USER", "ESCALATE", None}
REGIMES = {"canonical", "spoken", "perturbed"}
PLACEHOLDER_RE = re.compile(r"^<([A-Z_]+)_([1-9][0-9]*)>$")

TOP_LEVEL_KEYS = ("id", "schema_version", "input", "clean_input", "normalized_text", "sanitized_prompt",
                  "pii", "preserved_entities", "uncertainties", "routing", "metadata")
METADATA_KEYS = ("language", "surface_form", "source", "label_source", "split")


def check_record(r: dict) -> list[str]:
    """Return a list of problems; an empty list means the record follows v0.2."""
    errors: list[str] = []
    for k in TOP_LEVEL_KEYS:
        if k not in r:
            errors.append(f"missing key {k}")
    if errors:
        return errors
    if r["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version {r['schema_version']!r}")
    if not isinstance(r["input"], str) or not r["input"]:
        errors.append("input must be a non-empty string")
    elif r["clean_input"] != clean_text(r["input"]):
        errors.append("clean_input != clean_text(input)")
    if r["routing"] not in ROUTES:
        errors.append(f"routing {r['routing']!r}")
    for k in ("pii", "preserved_entities", "uncertainties"):
        if not isinstance(r[k], list):
            errors.append(f"{k} must be a list")

    meta = r["metadata"] if isinstance(r["metadata"], dict) else {}
    for k in METADATA_KEYS:
        if k not in meta:
            errors.append(f"metadata.{k} missing")
    if meta.get("surface_form") not in SURFACE_FORMS:
        errors.append(f"metadata.surface_form {meta.get('surface_form')!r}")
    if meta.get("label_source") not in LABEL_SOURCES:
        errors.append(f"metadata.label_source {meta.get('label_source')!r}")
    if meta.get("split") not in SPLITS:
        errors.append(f"metadata.split {meta.get('split')!r}")

    text = r["clean_input"] if isinstance(r["clean_input"], str) else ""
    placeholder_of: dict[tuple[str, str], str] = {}
    for p in r["pii"] if isinstance(r["pii"], list) else []:
        m = PLACEHOLDER_RE.match(str(p.get("placeholder", "")))
        if p.get("type") not in PII_TYPES:
            errors.append(f"pii type {p.get('type')!r}")
        if not m or m.group(1) != p.get("type"):
            errors.append(f"placeholder {p.get('placeholder')!r} does not match type {p.get('type')!r}")
        s, e = p.get("start"), p.get("end")
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text)
                and text[s:e] == p.get("text")):
            errors.append(f"pii offsets do not point at {p.get('text')!r} in clean_input")
        if "regime" in p and p["regime"] not in REGIMES:
            errors.append(f"pii regime {p['regime']!r}")
        key = (p.get("type"), p.get("text"))
        if key in placeholder_of and placeholder_of[key] != p.get("placeholder"):
            errors.append(f"same value {p.get('text')!r} has two placeholders")
        placeholder_of[key] = p.get("placeholder")
    for ent in r["preserved_entities"] if isinstance(r["preserved_entities"], list) else []:
        if not {"type", "value"} <= set(ent):
            errors.append("preserved entity needs type and value")
    return errors


def mask(r: dict) -> str:
    """clean_input with every PII span replaced by its placeholder (for leakage checks and baselines)."""
    text = r["clean_input"]
    for p in sorted(r["pii"], key=lambda p: p["start"], reverse=True):
        text = text[:p["start"]] + p["placeholder"] + text[p["end"]:]
    return text
