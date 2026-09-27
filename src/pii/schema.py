"""Schema v0.2 rules, read from docs/annotation-schema.md (owner: Saber) at import time.

The schema document is the single source of truth: nothing here restates its allowed values.
If the document changes, the checks change with it; if a list cannot be found in the
document, importing this module fails loudly instead of guessing.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.preprocessing.clean import clean_text

SCHEMA_DOC = Path(__file__).resolve().parents[2] / "docs" / "annotation-schema.md"
_DOC = SCHEMA_DOC.read_text(encoding="utf-8")


def _doc_list(label: str) -> tuple[str, ...]:
    """Comma-separated values after a bold label, on the same line or the next one."""
    m = re.search(r"\*\*" + re.escape(label) + r"\*\*[^\n]*?:\s*([A-Za-z_, ]+?)\s*\n", _DOC)
    if not m:
        raise RuntimeError(f"'{label}' list not found in {SCHEMA_DOC}")
    return tuple(v.strip() for v in m.group(1).split(",") if v.strip())


def _doc_table_values(heading: str) -> tuple[str, ...]:
    """First-column values of the table under a '## heading'."""
    section = _DOC.split(f"## {heading}", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\|\s*([a-z_]+)\s*\|", section, flags=re.M)
    if not rows:
        raise RuntimeError(f"table under '{heading}' not found in {SCHEMA_DOC}")
    return tuple(rows)


SCHEMA_VERSION = re.search(r'schema_version \| string \| Currently "([^"]+)"', _DOC).group(1)
PII_TYPES = _doc_list("PII types")
UNCERTAINTY_TYPES = _doc_list("Uncertainty types")
ROUTES = _doc_list("Routing")
SURFACE_FORMS = _doc_list("surface_form")
LABEL_SOURCES = _doc_table_values("metadata.label_source")
SPLITS = tuple(re.search(r"## metadata\.split\s+([a-z, ]+?), or none", _DOC).group(1).replace(" ", "").split(",")) + ("none",)
_FIELDS = _DOC.split("## Fields", 1)[1].split("\n## ", 1)[0]
TOP_LEVEL_KEYS = tuple(re.findall(r"^\| ([a-z_]+) \|", _FIELDS, flags=re.M))   # header row "Field" is capitalised, so it is not matched
METADATA_KEYS = tuple(k.strip() for k in re.search(r"\| metadata \| object \| ([a-z_, ]+) \|", _DOC).group(1).split(","))
PLACEHOLDER_RE = re.compile(r"^<([A-Z_]+)_([1-9][0-9]*)>$")

# Not in the schema document yet: proposed to Saber as an optional key on pii items.
REGIMES = ("canonical", "spoken", "perturbed")


def check_record(r: dict) -> list[str]:
    """Return a list of problems; an empty list means the record follows the schema document."""
    errors = [f"missing key {k}" for k in TOP_LEVEL_KEYS if k not in r]
    if errors:
        return errors
    if r["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version {r['schema_version']!r} (document says {SCHEMA_VERSION!r})")
    if not isinstance(r["input"], str) or not r["input"]:
        errors.append("input must be a non-empty string")
    elif r["clean_input"] != clean_text(r["input"]):
        errors.append("clean_input != clean_text(input)")
    if r["routing"] is not None and r["routing"] not in ROUTES:
        errors.append(f"routing {r['routing']!r}")
    for k in ("pii", "preserved_entities", "uncertainties"):
        if not isinstance(r[k], list):
            errors.append(f"{k} must be a list")

    meta = r["metadata"] if isinstance(r["metadata"], dict) else {}
    errors += [f"metadata.{k} missing" for k in METADATA_KEYS if k not in meta]
    for key, allowed in (("surface_form", SURFACE_FORMS), ("label_source", LABEL_SOURCES), ("split", SPLITS)):
        if key in meta and meta[key] not in allowed:
            errors.append(f"metadata.{key} {meta[key]!r}")

    text = r["clean_input"] if isinstance(r["clean_input"], str) else ""
    placeholder_of: dict[tuple, str] = {}
    for p in r["pii"] if isinstance(r["pii"], list) else []:
        m = PLACEHOLDER_RE.match(str(p.get("placeholder", "")))
        if p.get("type") not in PII_TYPES:
            errors.append(f"pii type {p.get('type')!r}")
        if not m or m.group(1) != p.get("type"):
            errors.append(f"placeholder {p.get('placeholder')!r} does not match type {p.get('type')!r}")
        s, e = p.get("start"), p.get("end")
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text) and text[s:e] == p.get("text")):
            errors.append(f"pii offsets do not point at {p.get('text')!r} in clean_input")
        if "regime" in p and p["regime"] not in REGIMES:
            errors.append(f"pii regime {p['regime']!r}")
        key = (p.get("type"), p.get("text"))
        if key in placeholder_of and placeholder_of[key] != p.get("placeholder"):
            errors.append(f"same value {p.get('text')!r} has two placeholders")
        placeholder_of[key] = p.get("placeholder")
    for u in r["uncertainties"] if isinstance(r["uncertainties"], list) else []:
        if not set(u.get("types", [])) <= set(UNCERTAINTY_TYPES):
            errors.append(f"uncertainty types {u.get('types')!r}")
    for ent in r["preserved_entities"] if isinstance(r["preserved_entities"], list) else []:
        if not {"type", "value"} <= set(ent):
            errors.append("preserved entity needs type and value")
    return errors
