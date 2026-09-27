"""JSON schema for teacher outputs: parsing, validation, canonical serialization.

The canonical serializer records the character range of every leaf value, so
token-level uncertainty on the pivot sequence y* can be mapped back to JSON fields.
"""
import json
import re

UNC_TYPES = ["TRANSLITERATION", "NUMERIC_AMBIGUITY", "PII_BOUNDARY", "DIALECT_MEANING", "INTENT"]
PII_TYPES = ["PHONE", "NID", "TXN_ID", "ACCOUNT", "CARD", "OTP", "EMAIL", "NAME", "ADDRESS", "ID"]
TOP_KEYS = ["sanitized_prompt", "pii", "preserved_entities", "ambiguous_spans"]


def extract_json(text):
    """Return the first balanced {...} object in text parsed as JSON, or None."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _str(x):
    return x.strip() if isinstance(x, str) and x.strip() else None


def validate(obj):
    """Normalize a parsed teacher output into canonical key order.

    Returns (normalized_or_None, errors). Schema-valid means: not None and no errors.
    Malformed list entries are dropped and reported as errors.
    """
    if not isinstance(obj, dict):
        return None, ["not a JSON object"]
    errors = []
    sp = _str(obj.get("sanitized_prompt"))
    if not sp:
        return None, ["missing sanitized_prompt"]
    out = {"sanitized_prompt": sp, "pii": [], "preserved_entities": [], "ambiguous_spans": []}

    for k in ("pii", "preserved_entities", "ambiguous_spans"):
        if k not in obj:
            errors.append(f"missing {k}")
        elif not isinstance(obj[k], list):
            errors.append(f"{k} is not a list")

    def entries(k):
        v = obj.get(k)
        return v if isinstance(v, list) else []

    for e in entries("pii"):
        d = e if isinstance(e, dict) else {}
        t, s, ph = _str(d.get("type")), _str(d.get("span")), _str(d.get("placeholder"))
        if t and s and ph:
            out["pii"].append({"type": t.upper(), "span": s, "placeholder": ph})
        else:
            errors.append(f"bad pii entry: {e!r}"[:120])

    for e in entries("preserved_entities"):
        d = e if isinstance(e, dict) else {}
        t, v = _str(d.get("type")), _str(d.get("value"))
        if t and v:
            out["preserved_entities"].append({"type": t.upper(), "value": v})
        else:
            errors.append(f"bad entity: {e!r}"[:120])

    for e in entries("ambiguous_spans"):
        d = e if isinstance(e, dict) else {}
        s = _str(d.get("span"))
        if not s:
            errors.append(f"bad ambiguous span: {e!r}"[:120])
            continue
        raw_types = d.get("types") or []
        raw_types = [raw_types] if isinstance(raw_types, str) else raw_types
        types = [t.upper() for t in raw_types if isinstance(t, str) and t.upper() in UNC_TYPES]
        cands = [c.strip() for c in (d.get("candidates") or []) if isinstance(c, str) and c.strip()]
        if not types:
            errors.append(f"ambiguous span without valid type: {s}")
        out["ambiguous_spans"].append({"span": s, "types": types, "candidates": cands})

    # PII must not be copied into the sanitized prompt
    for e in out["pii"]:
        if len(e["span"]) >= 4 and e["span"] in sp:
            errors.append(f"pii span copied into sanitized_prompt ({e['type']})")
    return out, errors


def serialize_with_offsets(obj):
    """Serialize obj to compact JSON (this string IS the pivot y*) and record leaf ranges.

    Each leaf is {"path": [...], "start": int, "end": int}; y[start:end] is the value as
    written (inside the quotes for strings).
    """
    parts, leaves, pos = [], [], [0]

    def emit(s):
        parts.append(s)
        pos[0] += len(s)

    def rec(v, path):
        if isinstance(v, dict):
            emit("{")
            for i, (k, val) in enumerate(v.items()):
                if i:
                    emit(", ")
                emit(json.dumps(k, ensure_ascii=False) + ": ")
                rec(val, path + [k])
            emit("}")
        elif isinstance(v, list):
            emit("[")
            for i, val in enumerate(v):
                if i:
                    emit(", ")
                rec(val, path + [i])
            emit("]")
        else:
            s = json.dumps(v, ensure_ascii=False)
            q = 1 if isinstance(v, str) else 0
            leaves.append({"path": path, "start": pos[0] + q, "end": pos[0] + len(s) - q})
            emit(s)

    rec(obj, [])
    return "".join(parts), leaves
