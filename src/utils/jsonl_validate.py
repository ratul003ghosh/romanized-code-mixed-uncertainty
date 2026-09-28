"""Line-by-line validation of the project's JSONL files (schema v0.2, docs/annotation-schema.md).

Three kinds of file are supported:
  teacher_input - input to the teacher pipeline (data/processed/teacher_inputs*.jsonl)
  record        - a full schema v0.2 record, e.g. silver.jsonl from the teachers
  student       - student training pairs from scripts/build_student_data.py

validate_file() never stops at the first problem: it reads every line and returns a report
with one entry per problem, so a single run shows everything that is wrong.
"""
from __future__ import annotations

import json
import re
from collections import Counter

PII_TYPES = {"PHONE", "NID", "ID_NUMBER", "TXN_ID", "ACCOUNT", "CARD", "OTP", "EMAIL", "NAME", "ADDRESS"}
UNC_TYPES = {"TRANSLITERATION", "NUMERIC_AMBIGUITY", "PII_BOUNDARY", "DIALECT_MEANING", "INTENT"}
ROUTES = {"PROCEED", "PROCEED_WITH_FLAGS", "ASK_USER", "ESCALATE"}
LABEL_SOURCES = {"silver", "gold", "synthetic", "illustrative", "none", "prediction"}
SPLITS = {"train", "dev", "test", "none"}
SURFACE_FORMS = {"banglish", "romanized", "code_mixed", "dialect", "hinglish", "bangla_script", "english"}
TARGET_KEYS = ("sanitized_prompt", "pii", "preserved_entities", "uncertainties", "routing")
PLACEHOLDER_RE = re.compile(r"^<([A-Z_]+)_(\d+)>$")
BAD_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]")   # control chars + replacement char
DEFAULT_MAX_CHARS = 1000


class Report:
    def __init__(self, path, kind):
        self.path, self.kind = path, kind
        self.lines = 0
        self.valid = 0
        self.errors = []            # (line_no, record_id, message)
        self.counts = Counter()     # problem category -> count

    def add(self, line_no, rid, category, message):
        self.errors.append((line_no, rid, f"{category}: {message}"))
        self.counts[category] += 1

    @property
    def ok(self):
        return not self.errors

    def summary(self, max_errors=15):
        out = [f"File: {self.path}  (kind: {self.kind})",
               f"Lines read: {self.lines}",
               f"Valid records: {self.valid}",
               f"Problems: {len(self.errors)}"]
        for cat, n in sorted(self.counts.items()):
            out.append(f"  {cat}: {n}")
        for line_no, rid, msg in self.errors[:max_errors]:
            out.append(f"  line {line_no} (id={rid}): {msg}")
        if len(self.errors) > max_errors:
            out.append(f"  ... and {len(self.errors) - max_errors} more")
        out.append("RESULT: " + ("PASS" if self.ok else "FAIL"))
        return "\n".join(out)


def _check_text(rep, n, rid, field, value, max_chars, allow_empty=False):
    """A text field must be a string, non-empty, not too long, and free of broken characters."""
    if value is None:
        rep.add(n, rid, "null_value", f"{field} is null")
        return False
    if not isinstance(value, str):
        rep.add(n, rid, "wrong_type", f"{field} must be a string, got {type(value).__name__}")
        return False
    if not value.strip() and not allow_empty:
        rep.add(n, rid, "empty_input", f"{field} is empty")
        return False
    if len(value) > max_chars:
        rep.add(n, rid, "too_long", f"{field} has {len(value)} characters (limit {max_chars})")
    if BAD_CHARS_RE.search(value):
        rep.add(n, rid, "bad_unicode", f"{field} contains control or replacement characters")
    return True


def _check_metadata(rep, n, rid, meta, require_label=None):
    if not isinstance(meta, dict):
        rep.add(n, rid, "missing_field", "metadata must be an object")
        return
    for key in ("source", "label_source", "split"):
        if meta.get(key) in (None, ""):
            rep.add(n, rid, "missing_field", f"metadata.{key} is missing")
    if meta.get("label_source") not in LABEL_SOURCES:
        rep.add(n, rid, "schema_violation", f"metadata.label_source {meta.get('label_source')!r} not allowed")
    if require_label and meta.get("label_source") != require_label:
        rep.add(n, rid, "schema_violation", f"metadata.label_source should be {require_label!r}")
    if meta.get("split") not in SPLITS:
        rep.add(n, rid, "schema_violation", f"metadata.split {meta.get('split')!r} not allowed")
    sf = meta.get("surface_form")
    if sf is not None and sf not in SURFACE_FORMS:
        rep.add(n, rid, "schema_violation", f"metadata.surface_form {sf!r} not allowed")


def _check_position(rep, n, rid, text, item, span_key):
    s, e = item.get("start"), item.get("end")
    if s is None and e is None:
        return                                   # allowed: span could not be located
    if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text)):
        rep.add(n, rid, "schema_violation", f"bad start/end ({s}, {e}) for {item.get(span_key)!r}")
    elif text[s:e] != item.get(span_key):
        rep.add(n, rid, "schema_violation",
                f"clean_input[{s}:{e}] = {text[s:e]!r} but span is {item.get(span_key)!r}")


def _check_score(rep, n, rid, name, v):
    if v is None:
        return
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0.0 <= v <= 1.0:
        rep.add(n, rid, "schema_violation", f"{name} must be a number in [0, 1] or null, got {v!r}")


def _check_record_body(rep, n, rid, r, max_chars):
    text = r.get("clean_input") if isinstance(r.get("clean_input"), str) else ""
    for key in ("pii", "preserved_entities", "uncertainties"):
        if not isinstance(r.get(key), list):
            rep.add(n, rid, "wrong_type", f"{key} must be a list")
            return
    if r.get("routing") is not None and r["routing"] not in ROUTES:
        rep.add(n, rid, "schema_violation", f"routing {r['routing']!r} not allowed")
    sp = r.get("sanitized_prompt")
    if sp is not None:
        _check_text(rep, n, rid, "sanitized_prompt", sp, max_chars * 2)
    for p in r["pii"]:
        if not isinstance(p, dict):
            rep.add(n, rid, "wrong_type", "pii entry must be an object")
            continue
        if p.get("type") not in PII_TYPES:
            rep.add(n, rid, "schema_violation", f"pii type {p.get('type')!r} not allowed")
        m = PLACEHOLDER_RE.match(p.get("placeholder") or "")
        if not m or m.group(1) != p.get("type"):
            rep.add(n, rid, "schema_violation", f"placeholder {p.get('placeholder')!r} must be <TYPE_N>")
        if "text" in p:
            _check_position(rep, n, rid, text, p, "text")
    for u in r["uncertainties"]:
        if not isinstance(u, dict):
            rep.add(n, rid, "wrong_type", "uncertainty entry must be an object")
            continue
        types = u.get("types")
        if not isinstance(types, list) or not types or not set(types) <= UNC_TYPES:
            rep.add(n, rid, "schema_violation", f"uncertainty types {types!r} not allowed")
        _check_score(rep, n, rid, "aleatoric", u.get("aleatoric"))
        _check_score(rep, n, rid, "epistemic", u.get("epistemic"))
        _check_position(rep, n, rid, text, u, "span")


def _check_student(rep, n, rid, r, max_chars):
    msgs = r.get("messages")
    if not isinstance(msgs, list) or [m.get("role") for m in msgs if isinstance(m, dict)] != \
            ["system", "user", "assistant"]:
        rep.add(n, rid, "schema_violation", "messages must be [system, user, assistant]")
        return False
    for m in msgs:
        if not _check_text(rep, n, rid, f"{m['role']} content", m.get("content"), max_chars * 4):
            return False
    try:
        target = json.loads(msgs[2]["content"])
    except json.JSONDecodeError as e:
        rep.add(n, rid, "invalid_json", f"assistant content is not JSON: {e}")
        return False
    missing = [k for k in TARGET_KEYS if k not in target]
    if missing:
        rep.add(n, rid, "missing_field", f"assistant JSON lacks {missing}")
        return False
    return True


def validate_file(path, kind, max_chars=DEFAULT_MAX_CHARS):
    rep = Report(path, kind)
    seen_ids, seen_texts = {}, {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rep.lines += 1
            before = len(rep.errors)
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                rep.add(n, None, "invalid_json", str(e))
                continue
            if not isinstance(r, dict):
                rep.add(n, None, "malformed_record", "line is not a JSON object")
                continue
            rid = r.get("id")
            if not isinstance(rid, str) or not rid:
                rep.add(n, rid, "missing_field", "id must be a non-empty string")
            elif rid in seen_ids:
                rep.add(n, rid, "duplicate_id", f"also on line {seen_ids[rid]}")
            else:
                seen_ids[rid] = n

            if kind == "student":
                text_ok = _check_student(rep, n, rid, r, max_chars)
                key_text = r["messages"][1]["content"] if text_ok else None
            else:
                if r.get("schema_version") != "0.2":
                    rep.add(n, rid, "schema_violation", f"schema_version {r.get('schema_version')!r}, expected '0.2'")
                _check_text(rep, n, rid, "input", r.get("input"), max_chars)
                text_ok = _check_text(rep, n, rid, "clean_input", r.get("clean_input"), max_chars)
                key_text = r.get("clean_input") if text_ok else None
                if kind == "teacher_input":
                    _check_metadata(rep, n, rid, r.get("metadata"), require_label="none")
                    if r.get("metadata", {}).get("split") == "none":
                        rep.add(n, rid, "schema_violation", "teacher inputs need split train/dev/test")
                else:
                    _check_metadata(rep, n, rid, r.get("metadata"))
                    _check_record_body(rep, n, rid, r, max_chars)

            if key_text is not None:
                norm = " ".join(key_text.lower().split())
                if norm in seen_texts:
                    rep.add(n, rid, "duplicate_text", f"same text as line {seen_texts[norm]}")
                else:
                    seen_texts[norm] = n
            if len(rep.errors) == before:
                rep.valid += 1
    return rep
