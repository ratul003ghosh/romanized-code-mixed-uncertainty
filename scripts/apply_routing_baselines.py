import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser(description="Apply routing baselines to student predictions.")
    parser.add_argument("--input", required=True, help="Input predictions.jsonl from student model")
    parser.add_argument("--output", required=True, help="Output predictions.jsonl with modified routing")
    parser.add_argument("--strategy", required=True, choices=["confidence", "fixed_threshold"],
                        help="Routing baseline strategy")
    parser.add_argument("--confidence-metric", default="mean_token_logprob", 
                        help="Confidence metric to use for 'confidence' strategy")
    parser.add_argument("--threshold", type=float, default=-0.5,
                        help="Threshold for routing (e.g. logprob > thresh -> PROCEED, else ESCALATE)")
    
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:
         
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            
            if args.strategy == "fixed_threshold":
                # E.g. purely aleatoric threshold routing if needed, or always PROCEED. 
                # A simple fixed threshold on aleatoric uncertainty from the first span:
                unc = rec.get("uncertainties") or []
                max_aleatoric = max([float(u.get("aleatoric") or 0.0) for u in unc], default=0.0)
                if max_aleatoric > args.threshold:
                    rec["routing"] = "ESCALATE"
                else:
                    rec["routing"] = "PROCEED"
                    
            elif args.strategy == "confidence":
                # Route based purely on generation confidence
                meta = rec.get("metadata", {})
                conf = meta.get("confidence", {})
                val = conf.get(args.confidence_metric, -float('inf'))
                if val is not None and val < args.threshold:
                    rec["routing"] = "ESCALATE"
                else:
                    rec["routing"] = "PROCEED"

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {args.strategy} routing baseline to {args.output}")

if __name__ == "__main__":
    main()
