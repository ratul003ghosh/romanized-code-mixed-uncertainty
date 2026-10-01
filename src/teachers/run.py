"""CLI: python scripts/run_teacher.py --config configs/teacher_smoke.yaml [--stage all|generate|pivot|score|spans]"""
import argparse
import json
import logging
import os
import sys
import time

import yaml

from src.teachers import generate, pivot, score
from src.uncertainty import spans
from src.teachers.common import out_path


import subprocess
import sys
import shutil

def run_workers(args, stage, num_workers=2):
    log = logging.getLogger("run")
    log.info("launching %d workers for stage %s", num_workers, stage)
    procs = []
    for i in range(num_workers):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        cmd = [sys.executable, "-m", "src.teachers.run", "--config", args.config, "--stage", stage, "--worker-shard", f"{i}/{num_workers}"]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.input:
            cmd.extend(["--input", args.input])
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)
    
    success = True
    for p in procs:
        p.wait()
        if p.returncode != 0:
            success = False
    if not success:
        log.error("one or more workers failed in stage %s", stage)
        sys.exit(1)
    log.info("stage %s workers finished", stage)

def merge_jsonl(cfg, basename, num_workers=2):
    import json
    out_file = out_path(cfg, basename)
    
    # Read existing IDs from main file to prevent duplication during merge
    existing = set()
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if "teacher_id" in r:
                    existing.add((r["id"], r["teacher_id"]))
                else:
                    existing.add(r["id"])
                    
    with open(out_file, "a", encoding="utf-8") as f_out:
        for i in range(num_workers):
            shard_file = out_path(cfg, f"{basename.replace('.jsonl', '')}_shard{i}.jsonl")
            if os.path.exists(shard_file):
                with open(shard_file, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        if not line.strip(): continue
                        r = json.loads(line)
                        key = (r["id"], r["teacher_id"]) if "teacher_id" in r else r["id"]
                        if key not in existing:
                            f_out.write(line)
                            existing.add(key)
                os.remove(shard_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=["all", "generate", "pivot", "score", "spans"])
    ap.add_argument("--limit", type=int, default=None, help="override number of inputs")
    ap.add_argument("--input", default=None, help="override input_file (e.g. one split)")
    ap.add_argument("--output-dir", default=None, help="override output_dir (use one folder per split)")
    ap.add_argument("--parallel", action="store_true", help="run generation and scoring on 2 GPUs")
    ap.add_argument("--worker-shard", default=None, help="internal: e.g. 0/2")
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
<<<<<<< HEAD
    if args.worker_shard:
        cfg["worker_shard"] = args.worker_shard
        
=======
    # Fail loudly before any GPU work: a missing or empty input file used to finish "successfully"
    # with nothing generated.
    inp = cfg.get("input_file")
    if args.stage in ("all", "generate", "pivot", "spans"):
        if not inp or not os.path.exists(inp):
            sys.exit(f"ERROR: input file not found: {inp!r} (working folder: {os.getcwd()}).\n"
                     "Prepare the inputs first (docs/faculty-run.md, step 3).")
        n_lines = sum(1 for line in open(inp, encoding="utf-8") if line.strip())
        if n_lines == 0:
            sys.exit(f"ERROR: input file is empty: {inp}")
>>>>>>> origin/main
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out_path(cfg, "run.log"), encoding="utf-8")])
    log = logging.getLogger("run")
<<<<<<< HEAD
    
    # Only dump config from the main process
    if not args.worker_shard:
        with open(out_path(cfg, "config_used.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)
            
=======
    log.info("config %s | input %s | output %s", args.config, inp, os.path.abspath(cfg["output_dir"]))
    with open(out_path(cfg, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
>>>>>>> origin/main
    t0 = time.time()
    info_path = out_path(cfg, "score_info.json")
    
    if args.stage in ("all", "generate"):
        if args.parallel and not args.worker_shard:
            run_workers(args, "generate")
            merge_jsonl(cfg, "generations.jsonl")
        else:
            generate.run(cfg)
            
    if args.stage in ("all", "pivot"):
        if not args.worker_shard:
            pivot.run(cfg)
            
    if args.stage in ("all", "score"):
        if args.parallel and not args.worker_shard:
            run_workers(args, "score")
            merge_jsonl(cfg, "token_scores.jsonl")
            # Create a dummy info file so spans can run
            if not os.path.exists(info_path):
                with open(info_path, "w") as f:
                    json.dump({"merged_from_shards": True}, f, indent=2)
        else:
            _, info = score.run(cfg)
            if info and not args.worker_shard:
                with open(info_path, "w") as f:
                    json.dump(info, f, indent=2)
                    
    if args.stage in ("all", "spans"):
        if not args.worker_shard:
            records = spans.run(cfg)
            info = json.load(open(info_path)) if os.path.exists(info_path) else {}
            summary = spans.report(cfg, records, out_path(cfg, "generations.jsonl"), info)
            log.info("summary: %s", json.dumps(summary))
            
    log.info("stage %s finished in %.1f s", args.stage, time.time() - t0)


if __name__ == "__main__":
    main()
