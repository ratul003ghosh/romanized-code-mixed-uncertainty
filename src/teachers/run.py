"""CLI: python scripts/run_teacher.py --config configs/teacher_smoke.yaml [--stage all|generate|pivot|score|spans]"""
import argparse
import json
import logging
import os
import time

import yaml

from src.teachers import generate, pivot, score
from src.uncertainty import spans
from src.teachers.common import out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=["all", "generate", "pivot", "score", "spans"])
    ap.add_argument("--limit", type=int, default=None, help="override number of inputs")
    ap.add_argument("--input", default=None, help="override input_file (e.g. one split)")
    ap.add_argument("--output-dir", default=None, help="override output_dir (use one folder per split)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.limit:
        cfg["limit"] = args.limit
    if args.input:
        cfg["input_file"] = args.input
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
        cfg["run_name"] = f"{cfg.get('run_name', 'run')}:{args.output_dir.rstrip('/').split('/')[-1]}"
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_path(cfg, "run.log"))])
    log = logging.getLogger("run")
    with open(out_path(cfg, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    t0 = time.time()
    info_path = out_path(cfg, "score_info.json")
    if args.stage in ("all", "generate"):
        generate.run(cfg)
    if args.stage in ("all", "pivot"):
        pivot.run(cfg)
    if args.stage in ("all", "score"):
        _, info = score.run(cfg)
        if info:
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
    if args.stage in ("all", "spans"):
        records = spans.run(cfg)
        info = json.load(open(info_path)) if os.path.exists(info_path) else {}
        summary = spans.report(cfg, records, out_path(cfg, "generations.jsonl"), info)
        log.info("summary: %s", json.dumps(summary))
    log.info("stage %s finished in %.1f s", args.stage, time.time() - t0)


if __name__ == "__main__":
    main()
