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
from src.teachers.common import out_path, load_inputs, read_jsonl, teacher_groups


import subprocess
import sys
import shutil

def _worker_index(cfg):
    shard = cfg.get("worker_shard")
    if not shard:
        return None
    try:
        idx, total = map(int, shard.split("/"))
    except Exception as exc:
        raise RuntimeError(
            f"invalid worker_shard={shard!r}; expected i/n"
        ) from exc
    if total != 2 or idx not in (0, 1):
        raise RuntimeError(
            f"parallel execution requires 0/2 or 1/2, got {shard!r}"
        )
    return idx


def _validate_worker_cuda():
    # Fail early instead of silently falling back to CPU.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in ("0", "1"):
        raise RuntimeError(
            f"unexpected CUDA_VISIBLE_DEVICES={visible!r}; expected '0' or '1'"
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is unavailable in worker {visible}; refusing CPU fallback."
        )

    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(
            f"worker {visible} sees {count} CUDA devices; expected exactly 1."
        )

    logging.getLogger("run").info(
        "GPU isolation OK: CUDA_VISIBLE_DEVICES=%s, logical device_count=%d, device=%s",
        visible,
        count,
        torch.cuda.get_device_name(0),
    )


def run_workers(args, stage, num_workers=2):
    log = logging.getLogger("run")

    if num_workers != 2:
        raise RuntimeError("parallel teacher execution currently requires exactly 2 workers")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Parallel mode requested, but CUDA is unavailable in the parent."
        )
    visible_gpus = torch.cuda.device_count()
    if visible_gpus < num_workers:
        raise RuntimeError(
            f"Parallel mode requires at least {num_workers} GPUs; parent sees {visible_gpus}."
        )

    log.info("launching %d workers for stage %s", num_workers, stage)
    procs = []

    for i in range(num_workers):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)

        cmd = [
            sys.executable,
            "-m",
            "src.teachers.run",
            "--config",
            args.config,
            "--stage",
            stage,
            "--worker-shard",
            f"{i}/{num_workers}",
        ]

        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.input:
            cmd.extend(["--input", args.input])
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])

        log.info("worker %d CUDA_VISIBLE_DEVICES=%s", i, env["CUDA_VISIBLE_DEVICES"])
        procs.append((i, subprocess.Popen(cmd, env=env)))

    failed = []
    for i, p in procs:
        rc = p.wait()
        if rc != 0:
            failed.append((i, rc))

    if failed:
        for i, rc in failed:
            log.error("worker %d failed in stage %s with exit code %s", i, stage, rc)
        log.error("NO MERGE WILL BE PERFORMED.")
        raise SystemExit(1)

    log.info("stage %s: all workers finished successfully", stage)

def _record_key(r):
    if "id" not in r:
        raise ValueError("record is missing required 'id'")
    if "teacher_id" in r:
        return (str(r["id"]), str(r["teacher_id"]))
    return str(r["id"])


def _read_keys(path):
    keys = []
    seen = set()
    if not os.path.exists(path):
        return keys, seen

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON in {path}:{lineno}: {exc}") from exc
            key = _record_key(r)
            keys.append(key)
            seen.add(key)
    return keys, seen


def _expected_generation_keys(cfg):
    inputs = load_inputs(cfg)
    expected = set()
    for x in inputs:
        for group in teacher_groups(cfg):
            for variant in group["variants"]:
                expected.add((str(x["id"]), f"{group['alias']}:{variant}"))
    return expected


def _expected_score_ids(cfg):
    return {
        str(r["id"])
        for r in read_jsonl(out_path(cfg, "pivots.jsonl"))
        if r.get("pivot_ok")
    }


def _validate_generation_output(cfg):
    expected = _expected_generation_keys(cfg)
    actual = set(_read_keys(out_path(cfg, "generations.jsonl"))[1])

    missing = expected - actual
    extra = actual - expected

    if missing:
        raise RuntimeError(
            f"generation output incomplete: {len(missing)} missing; "
            f"sample={sorted(missing, key=str)[:10]}"
        )
    if extra:
        raise RuntimeError(
            f"generation output has {len(extra)} unexpected records; "
            f"sample={sorted(extra, key=str)[:10]}"
        )

    logging.getLogger("run").info(
        "generation output validation OK: %d/%d records",
        len(actual),
        len(expected),
    )


def _validate_score_output(cfg):
    expected = _expected_score_ids(cfg)
    actual = {
        str(r["id"])
        for r in read_jsonl(out_path(cfg, "token_scores.jsonl"))
    }

    missing = expected - actual
    extra = actual - expected

    if missing:
        raise RuntimeError(
            f"score output incomplete: {len(missing)} missing; "
            f"sample={sorted(missing)[:10]}"
        )
    if extra:
        raise RuntimeError(
            f"score output has {len(extra)} unexpected records; "
            f"sample={sorted(extra)[:10]}"
        )

    logging.getLogger("run").info(
        "score output validation OK: %d/%d pivot records",
        len(actual),
        len(expected),
    )


def merge_jsonl(cfg, basename, num_workers=2):
    # Merge atomically; delete worker shards only after validation.
    log = logging.getLogger("run")
    out_file = out_path(cfg, basename)
    tmp_file = out_file + ".merge_tmp"

    existing_records = []
    _, existing_keys = _read_keys(out_file)

    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            existing_records = [line for line in f if line.strip()]

    shard_paths = [
        out_path(cfg, f"{basename.replace('.jsonl', '')}_shard{i}.jsonl")
        for i in range(num_workers)
    ]

    for path in shard_paths:
        if not os.path.exists(path):
            raise RuntimeError(
                f"expected worker shard is missing: {path}; refusing partial merge."
            )
        _read_keys(path)

    merged_keys = set(existing_keys)
    added = 0

    try:
        with open(tmp_file, "w", encoding="utf-8") as f_out:
            for line in existing_records:
                f_out.write(line)

            for shard_file in shard_paths:
                with open(shard_file, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        key = _record_key(r)
                        if key not in merged_keys:
                            f_out.write(line)
                            merged_keys.add(key)
                            added += 1

        os.replace(tmp_file, out_file)

        # Validate before deleting any shard.
        _read_keys(out_file)

        if basename == "generations.jsonl":
            _validate_generation_output(cfg)
        elif basename == "token_scores.jsonl":
            _validate_score_output(cfg)

    except Exception:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        raise

    for shard_file in shard_paths:
        os.remove(shard_file)

    log.info(
        "merged %s atomically: %d new records; shards removed after validation",
        out_file,
        added,
    )


def merge_score_info(cfg, num_workers=2):
    # Preserve per-worker score sanity/timing/GPU metadata.
    infos = []

    for i in range(num_workers):
        path = out_path(cfg, f"score_info_shard{i}.json")
        if not os.path.exists(path):
            raise RuntimeError(f"missing score metadata shard: {path}")
        with open(path, "r", encoding="utf-8") as f:
            infos.append(json.load(f))

    if not any(infos):
        raise RuntimeError("all score workers returned empty score metadata")

    merged = {
        "parallel": True,
        "workers": num_workers,
        "worker_info": {str(i): info for i, info in enumerate(infos)},
        "n_scored": sum(int(info.get("n_scored", 0)) for info in infos),
    }

    times = [
        float(info["sec_per_item"])
        for info in infos
        if info.get("sec_per_item") is not None
    ]
    if times:
        merged["sec_per_item_mean_worker"] = sum(times) / len(times)

    residuals = [
        float(info["worst_jsd_before_clamp"])
        for info in infos
        if info.get("worst_jsd_before_clamp") is not None
    ]
    if residuals:
        merged["worst_jsd_before_clamp"] = min(residuals)

    peaks = [
        float(info["peak_gpu_gb"])
        for info in infos
        if info.get("peak_gpu_gb") is not None
    ]
    if peaks:
        merged["peak_gpu_gb_max_worker"] = max(peaks)

    merged["sanity"] = [
        info.get("sanity")
        for info in infos
        if info.get("sanity") is not None
    ]

    with open(out_path(cfg, "score_info.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    for i in range(num_workers):
        os.remove(out_path(cfg, f"score_info_shard{i}.json"))

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
    if args.worker_shard:
        cfg["worker_shard"] = args.worker_shard

    # Fail loudly before any GPU work: a missing or empty input file used to finish
    # successfully with nothing generated.
    inp = cfg.get("input_file")
    if args.stage in ("all", "generate", "pivot", "spans"):
        if not inp or not os.path.exists(inp):
            sys.exit(
                f"ERROR: input file not found: {inp!r} (working folder: {os.getcwd()}).\n"
                "Prepare the inputs first (docs/faculty-run.md, step 3)."
            )
        with open(inp, encoding="utf-8") as _f:
            n_lines = sum(1 for line in _f if line.strip())
        if n_lines == 0:
            sys.exit(f"ERROR: input file is empty: {inp}")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out_path(cfg, "run.log"), encoding="utf-8")])
    log = logging.getLogger("run")
    log.info(
        "config %s | input %s | output %s",
        args.config,
        inp,
        os.path.abspath(cfg["output_dir"]),
    )

    # Only the parent writes the shared config file.
    if not args.worker_shard:
        with open(out_path(cfg, "config_used.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
    if args.worker_shard:
        _worker_index(cfg)
        _validate_worker_cuda()

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
            merge_score_info(cfg)
        else:
            _, info = score.run(cfg)
            if args.worker_shard:
                worker_idx = _worker_index(cfg)
                worker_info_path = out_path(cfg, f"score_info_shard{worker_idx}.json")
                with open(worker_info_path, "w", encoding="utf-8") as f:
                    json.dump(info or {}, f, indent=2)
            elif info:
                with open(info_path, "w", encoding="utf-8") as f:
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
