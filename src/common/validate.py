"""Validate sanitizer outputs against configs/schema.json plus semantic rules.

Used for three things in the project:
  1. filtering silver targets (section 3.2, step 5),
  2. the JSON-validity metric (section 6.1),
  3. catching drift whenever the schema or the prompts change.

CLI:
  python -m src.common.validate path/to/file.jsonl
Each line must be {"input": "...", "output": {...}} or just the output object.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "configs" / "schema.json"
PLACEHOLDER_RE = re.compile(r"<[A-Z_]+_[1-9][0-9]*>")


def load_validator(path: Path = SCHEMA_PATH) -> Draft202012Validator:
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)  # fails loudly if the schema itself is broken
    return Draft202012Validator(schema)


def semantic_errors(out: dict, source: str | None = None) -> list[str]:
    """Rules JSON Schema cannot express (or that constrained decoders may ignore)."""
    errs: list[str] = []
    listed = [p["placeholder"] for p in out["pii"]]
    used = set(PLACEHOLDER_RE.findall(out["sanitized_prompt"]))

    if len(listed) != len(set(listed)):
        errs.append("duplicate placeholder in pii list")
    for ph in sorted(used - set(listed)):
        errs.append(f"{ph} appears in sanitized_prompt but not in pii list")
    for ph in sorted(set(listed) - used):
        errs.append(f"{ph} is in pii list but not used in sanitized_prompt")

    for p in out["pii"]:
        if not p["placeholder"][1:-1].rsplit("_", 1)[0] == p["type"]:
            errs.append(f"placeholder {p['placeholder']} does not match type {p['type']}")
        if p["span"] in out["sanitized_prompt"]:
            errs.append(f"raw PII span '{p['span']}' leaked into sanitized_prompt")

    for u in out["uncertainties"]:
        for k in ("aleatoric", "epistemic"):
            if abs(round(u[k], 2) - u[k]) > 1e-9:
                errs.append(f"{k}={u[k]} has more than two decimals")

    if source is not None:
        for p in out["pii"]:
            if p["span"] not in source:
                errs.append(f"pii span '{p['span']}' not found in input")
        for u in out["uncertainties"]:
            if u["span"] not in source:
                errs.append(f"uncertainty span '{u['span']}' not found in input")
    return errs


def validate(out: dict, source: str | None = None,
             validator: Draft202012Validator | None = None) -> list[str]:
    """Return a list of error messages; an empty list means valid."""
    validator = validator or load_validator()
    schema_errs = [f"schema: {'/'.join(map(str, e.path)) or '<root>'}: {e.message}"
                   for e in validator.iter_errors(out)]
    if schema_errs:  # semantic checks assume the structure is right
        return schema_errs
    return semantic_errors(out, source)


def main(path: str) -> None:
    validator = load_validator()
    total = passed = 0
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"line {i}: not JSON ({e})")
            continue
        out, src = (rec["output"], rec.get("input")) if "output" in rec else (rec, None)
        errs = validate(out, src, validator)
        if errs:
            print(f"line {i}: " + "; ".join(errs))
        else:
            passed += 1
    rate = passed / total if total else 0.0
    print(f"\n{passed}/{total} valid ({rate:.1%})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.common.validate FILE.jsonl")
    main(sys.argv[1])
