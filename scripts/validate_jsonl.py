"""Validate every line of a JSONL file against schema v0.2.

    python scripts/validate_jsonl.py --kind teacher_input data/processed/teacher_inputs.jsonl
    python scripts/validate_jsonl.py --kind record  experiments/teacher/train/silver.jsonl
    python scripts/validate_jsonl.py --kind student data/processed/student_train.jsonl

Exit code 0 = PASS, 1 = problems found, 2 = file missing.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.jsonl_validate import DEFAULT_MAX_CHARS, validate_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="one or more JSONL files")
    ap.add_argument("--kind", required=True, choices=["teacher_input", "record", "student"])
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help=f"maximum length of an input text (default {DEFAULT_MAX_CHARS})")
    args = ap.parse_args()

    all_ok = True
    for path in args.paths:
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}")
            sys.exit(2)
        rep = validate_file(path, args.kind, args.max_chars)
        print(rep.summary())
        print()
        all_ok &= rep.ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
