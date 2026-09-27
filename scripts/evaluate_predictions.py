"""
CLI script to evaluate model predictions against a Gold dataset.
Computes PII span metrics, leakage, over-masking, uncertainty calibration, and routing metrics.
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.evaluator import MasterEvaluator, load_jsonl, save_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions against gold ground truth.")
    parser.add_argument("--gold", type=str, required=True,
                        help="Path to gold dataset (JSON or JSONL).")
    parser.add_argument("--pred", type=str, required=True,
                        help="Path to prediction file (JSON or JSONL).")
    parser.add_argument("--model-name", type=str, default="Evaluated Model",
                        help="Display name of the model being evaluated.")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU overlap threshold for partial span matching.")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional path to save full evaluation results as JSON.")
    parser.add_argument("--output-report", type=str, default=None,
                        help="Optional path to save formatted Markdown report.")
    args = parser.parse_args()

    print(f"Loading gold records from: {args.gold}")
    gold_records = load_jsonl(args.gold)
    print(f"Loading pred records from: {args.pred}")
    pred_records = load_jsonl(args.pred)

    evaluator = MasterEvaluator(iou_threshold=args.iou_thresh)
    results = evaluator.evaluate(gold_records, pred_records, model_name=args.model_name)
    report = evaluator.format_markdown_report(results)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60 + "\n")

    if args.output_json:
        save_json(results, args.output_json)
        print(f"Full metrics JSON saved to: {args.output_json}")

    if args.output_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Markdown report saved to: {args.output_report}")


if __name__ == "__main__":
    main()

