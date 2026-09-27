"""
CLI runner for baseline models.
Runs a chosen baseline on an input dataset and exports predictions conforming to Schema v0.2.
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines.regex_baseline import RegexPIIBaseline
from src.baselines.translit_baseline import TransliterationPIIBaseline
from src.baselines.student_confidence import StudentConfidenceBaseline
from evaluation.evaluator import load_jsonl, save_json


def main():
    parser = argparse.ArgumentParser(description="Run baseline models on Romanized code-mixed prompts.")
    parser.add_argument("--baseline", type=str, choices=["regex", "translit", "student_conf"], default="regex",
                        help="Baseline model to run (default: regex).")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input dataset (JSON or JSONL).")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save baseline predictions.")
    args = parser.parse_args()

    output_path = args.output or f"experiments/baseline/{args.baseline}_predictions.jsonl"

    print(f"Loading input data from: {args.input}")
    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records.")

    if args.baseline == "regex":
        print("Running Regex & Rule-Based PII baseline...")
        model = RegexPIIBaseline()
        predictions = model.predict_all(records)
    elif args.baseline == "translit":
        print("Running Transliteration-First PII baseline...")
        model = TransliterationPIIBaseline()
        predictions = model.predict_all(records)
    elif args.baseline == "student_conf":
        print("Running Student Raw Confidence Baseline (RQ2 ablation)...")
        model = StudentConfidenceBaseline()
        predictions = model.predict_all(records)
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        import json
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"Saved {len(predictions)} predictions to: {output_path}")


if __name__ == "__main__":
    main()

