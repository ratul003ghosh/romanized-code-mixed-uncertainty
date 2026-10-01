#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

BR=feature/parallel-teacher-inference
BASE=${BASE:-origin/main}

# ---- 0. safety: backup + branch ------------------------------------------------
git fetch origin
git checkout "$BR"
git branch -f "backup/parallel-before-fix" HEAD
echo ">> backup branch: backup/parallel-before-fix"

# ---- 1. bring in the missing base commits ---------------------------------------
if [ -z "${SKIP_MERGE:-}" ]; then
  if ! git merge --no-edit "$BASE"; then
    echo "!! MERGE CONFLICT. Resolve it (keep BOTH the new fail-on-offload/missing-input code"
    echo "   and the parallel code in run.py), 'git add' + 'git commit', then rerun: SKIP_MERGE=1 bash fix_parallel.sh"
    exit 1
  fi
fi

# ---- 2. new helper block for run.py ---------------------------------------------
cat > /tmp/new_helpers.py <<'PYEOF'
# ===== parallel helpers (verified merge, supervised workers) =====
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections import Counter

from src.teachers.common import out_path, read_jsonl, load_inputs, teacher_groups

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gpu_ids(num_workers):
    """Pick GPU ids for the workers; respect an existing CUDA_VISIBLE_DEVICES."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        ids = [x.strip() for x in env.split(",") if x.strip()]
    else:
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        except Exception as e:
            raise SystemExit(f"--parallel needs nvidia-smi to count GPUs: {e}")
        ids = [str(i) for i in range(sum(1 for l in out.splitlines() if l.startswith("GPU ")))]
    if len(ids) < num_workers:
        raise SystemExit(f"--parallel needs {num_workers} GPUs but found {len(ids)}: {ids}")
    return ids[:num_workers]


def run_workers(args, stage, num_workers=2):
    log = logging.getLogger("run")
    gpus = gpu_ids(num_workers)
    log.info("launching %d workers for stage %s on GPUs %s", num_workers, stage, gpus)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    procs = []
    try:
        for i, gpu in enumerate(gpus):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            cmd = [sys.executable, "-m", "src.teachers.run", "--config", os.path.abspath(args.config),
                   "--stage", stage, "--worker-shard", f"{i}/{num_workers}"]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            if args.input:
                cmd += ["--input", os.path.abspath(args.input)]
            if args.output_dir:
                cmd += ["--output-dir", os.path.abspath(args.output_dir)]
            procs.append(subprocess.Popen(cmd, env=env, cwd=REPO_ROOT))
        while True:  # stop as soon as one worker fails
            codes = [p.poll() for p in procs]
            if any(c not in (None, 0) for c in codes) or all(c == 0 for c in codes):
                break
            time.sleep(5)
    finally:  # Ctrl-C / SIGTERM / failure: never leave workers holding the GPUs
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
    if any(p.returncode != 0 for p in procs):
        log.error("worker failure in stage %s, exit codes %s; shards kept, rerun to resume",
                  stage, [p.returncode for p in procs])
        sys.exit(1)
    log.info("stage %s workers finished", stage)


def merge_jsonl(cfg, basename, num_workers=2, expected_ids=None, uniform_teachers=False, n_teachers=None):
    """Atomic, verified merge of <stem>_shardN.jsonl into <basename>."""
    log = logging.getLogger("run")
    out_file = out_path(cfg, basename)
    stem = basename[: -len(".jsonl")]
    shards = [out_path(cfg, f"{stem}_shard{i}.jsonl") for i in range(num_workers)]
    tmp = out_file + ".tmp"
    seen = set()
    with open(tmp, "w", encoding="utf-8") as f_out:
        for path in [out_file] + shards:  # existing main file first: resumes keep old rows
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f_in:
                for n, line in enumerate(f_in, 1):
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                        key = (str(r["id"]), r["teacher_id"]) if "teacher_id" in r else str(r["id"])
                    except (json.JSONDecodeError, KeyError):
                        log.warning("skipping unreadable line %d in %s", n, path)
                        continue
                    if key not in seen:
                        seen.add(key)
                        f_out.write(line if line.endswith("\n") else line + "\n")
        f_out.flush()
        os.fsync(f_out.fileno())

    ids = {k[0] if isinstance(k, tuple) else k for k in seen}
    problems = []
    if expected_ids is not None:
        exp = {str(x) for x in expected_ids}
        missing, extra = exp - ids, ids - exp
        if missing:
            problems.append(f"{len(missing)} expected ids missing, e.g. {sorted(missing)[:3]}")
        if extra:
            log.warning("%d ids in %s are not in the current inputs (stale rows?)", len(extra), basename)
    if uniform_teachers:
        counts = Counter(k[0] for k in seen if isinstance(k, tuple))
        want = n_teachers or (max(counts.values()) if counts else 0)
        short = [i for i, c in counts.items() if c != want]
        if short:
            problems.append(f"{len(short)} ids do not have exactly {want} teacher rows, e.g. {short[:3]}")
    if problems:
        os.remove(tmp)
        raise RuntimeError("merge of %s refused: %s (shards kept; rerun to resume)" % (basename, "; ".join(problems)))

    os.replace(tmp, out_file)
    for s in shards:
        if os.path.exists(s):
            os.remove(s)
    log.info("merged %s: %d records", basename, len(seen))


def merge_score_info(cfg, num_workers=2):
    """bool -> all(); int -> sum(); float -> max(); else shard 0. Per-shard values kept under 'shards'."""
    infos = []
    for i in range(num_workers):
        p = out_path(cfg, f"score_info_shard{i}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                infos.append(json.load(f))
    merged = {"shards": infos}
    for k in {k for d in infos for k in d if k != "shards"}:
        vals = [d[k] for d in infos if k in d]
        if all(isinstance(v, bool) for v in vals):
            merged[k] = all(vals)
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            merged[k] = sum(vals)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            merged[k] = max(vals)
        else:
            merged[k] = vals[0]
    with open(out_path(cfg, "score_info.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)  # always overwrite: no stale file
    for i in range(num_workers):
        p = out_path(cfg, f"score_info_shard{i}.json")
        if os.path.exists(p):
            os.remove(p)
    return merged
# ===== end parallel helpers =====
PYEOF

# ---- 3. patch common.py and run.py (aborts, changing nothing, if an anchor is missing) ----
python - <<'PYEOF'
import re, sys

# ---- common.py: resume-safe read_jsonl ----
p = "src/teachers/common.py"
s = open(p, encoding="utf-8").read()
new_read = '''def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    out = []
    for n, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if n != len(lines) - 1:
                raise  # corruption in the middle is a real error
            logging.getLogger(__name__).warning("dropping truncated last line in %s", path)
            with open(path, "w", encoding="utf-8") as f:  # repair so later appends stay clean
                f.writelines(lines[:-1])
    return out
'''
if "dropping truncated last line" not in s:
    s2, n = re.subn(r"def read_jsonl\(path\):.*?(?=\n\ndef )", new_read.rstrip("\n"), s, count=1, flags=re.S)
    if n != 1:
        sys.exit("!! could not find read_jsonl in common.py")
    if not re.search(r"^import logging", s2, re.M):
        s2 = re.sub(r"^import json", "import json\nimport logging", s2, count=1, flags=re.M)
        if "import logging" not in s2:
            sys.exit("!! could not add 'import logging' to common.py")
    open(p, "w", encoding="utf-8").write(s2)
    print(">> common.py patched")

# ---- run.py ----
p = "src/teachers/run.py"
s = open(p, encoding="utf-8").read()
if "def gpu_ids" in s:
    print(">> run.py already patched, skipping")
    sys.exit(0)
helpers = open("/tmp/new_helpers.py", encoding="utf-8").read()

s, n = re.subn(r"\n(?:import subprocess\nimport sys\nimport shutil\n\n)?def run_workers.*?(?=\ndef main\()",
               "\n" + helpers + "\n\n", s, count=1, flags=re.S)
if n != 1:
    sys.exit("!! could not locate old run_workers/merge_jsonl block before main()")

def swap(s, old, new):
    if old not in s:
        sys.exit("!! anchor not found in main():\n" + old)
    return s.replace(old, new, 1)

s = swap(s,
'''            run_workers(args, "generate")
            merge_jsonl(cfg, "generations.jsonl")
''',
'''            run_workers(args, "generate")
            _inputs = load_inputs(cfg)
            _n_teachers = sum(len(g["variants"]) for g in teacher_groups(cfg))
            merge_jsonl(cfg, "generations.jsonl", expected_ids={x["id"] for x in _inputs},
                        uniform_teachers=True, n_teachers=_n_teachers)
''')
s = swap(s,
'''            run_workers(args, "score")
            merge_jsonl(cfg, "token_scores.jsonl")
            # Create a dummy info file so spans can run
            if not os.path.exists(info_path):
                with open(info_path, "w") as f:
                    json.dump({"merged_from_shards": True}, f, indent=2)
''',
'''            run_workers(args, "score")
            _ok_ids = {p["id"] for p in read_jsonl(out_path(cfg, "pivots.jsonl")) if p["pivot_ok"]}
            merge_jsonl(cfg, "token_scores.jsonl", expected_ids=_ok_ids)
            merge_score_info(cfg)
''')
s = swap(s,
'''            if info and not args.worker_shard:
                with open(info_path, "w") as f:
                    json.dump(info, f, indent=2)
''',
'''            if info:
                dest = (out_path(cfg, f"score_info_shard{args.worker_shard.split('/')[0]}.json")
                        if args.worker_shard else info_path)
                with open(dest, "w") as f:
                    json.dump(info, f, indent=2)
''')
open(p, "w", encoding="utf-8").write(s)
print(">> run.py patched")
PYEOF

# ---- 4. tests -----------------------------------------------------------------
cat > tests/test_parallel_merge.py <<'PYEOF'
import json
import signal
import types

import pytest

from src.teachers import run as R


def _write(path, rows, trunc=None):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        if trunc:
            f.write(trunc)


def _rows(ids, teachers):
    return [{"id": i, "teacher_id": t} for i in ids for t in teachers]


def test_sharding_complete_and_disjoint():
    items = list(range(11))
    s0 = [x for i, x in enumerate(items) if i % 2 == 0]
    s1 = [x for i, x in enumerate(items) if i % 2 == 1]
    assert sorted(s0 + s1) == items and not set(s0) & set(s1)


def test_merge_complete_dedupes_and_cleans(tmp_path):
    cfg = {"output_dir": str(tmp_path)}
    T = ["a", "b", "c"]
    _write(tmp_path / "generations_shard0.jsonl", _rows(["0", "2"], T))
    _write(tmp_path / "generations_shard1.jsonl", _rows(["1", "3"], T) + _rows(["1"], T))
    R.merge_jsonl(cfg, "generations.jsonl", expected_ids={"0", "1", "2", "3"},
                  uniform_teachers=True, n_teachers=3)
    rows = [json.loads(l) for l in open(tmp_path / "generations.jsonl")]
    assert len(rows) == 12
    assert len({(r["id"], r["teacher_id"]) for r in rows}) == 12
    assert not list(tmp_path.glob("*_shard*"))


def test_merge_refuses_incomplete_and_keeps_shards(tmp_path):
    cfg = {"output_dir": str(tmp_path)}
    _write(tmp_path / "generations_shard0.jsonl", _rows(["0", "2"], ["a", "b"]))
    _write(tmp_path / "generations_shard1.jsonl", _rows(["1"], ["a", "b"]))
    with pytest.raises(RuntimeError):
        R.merge_jsonl(cfg, "generations.jsonl", expected_ids={"0", "1", "2", "3"},
                      uniform_teachers=True, n_teachers=2)
    assert not (tmp_path / "generations.jsonl").exists()
    assert (tmp_path / "generations_shard0.jsonl").exists()


def test_merge_rejects_partial_teacher_rows(tmp_path):
    cfg = {"output_dir": str(tmp_path)}
    _write(tmp_path / "generations_shard0.jsonl", _rows(["0"], ["a", "b"]) + _rows(["2"], ["a"]))
    with pytest.raises(RuntimeError):
        R.merge_jsonl(cfg, "generations.jsonl", expected_ids={"0", "2"},
                      uniform_teachers=True, n_teachers=2)


def test_merge_skips_truncated_line(tmp_path):
    cfg = {"output_dir": str(tmp_path)}
    _write(tmp_path / "generations_shard0.jsonl", _rows(["0"], ["a", "b"]), trunc='{"id": "2", "teac')
    R.merge_jsonl(cfg, "generations.jsonl", expected_ids={"0"}, uniform_teachers=True, n_teachers=2)
    assert len(open(tmp_path / "generations.jsonl").readlines()) == 2


def test_read_jsonl_repairs_truncated_tail(tmp_path):
    from src.teachers.common import read_jsonl, append_jsonl
    p = str(tmp_path / "x.jsonl")
    _write(p, [{"id": "0"}], trunc='{"id": "1')
    assert read_jsonl(p) == [{"id": "0"}]
    append_jsonl(p, {"id": "1"})
    assert read_jsonl(p) == [{"id": "0"}, {"id": "1"}]


def test_gpu_ids_respects_env_and_requires_two(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert R.gpu_ids(2) == ["2", "3"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(SystemExit):
        R.gpu_ids(2)


def test_worker_failure_exits_and_keeps_shards(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    old = signal.getsignal(signal.SIGTERM)
    shard = tmp_path / "generations_shard0.jsonl"
    _write(shard, _rows(["0"], ["a"]))

    class FakeProc:
        returncode = 1
        def poll(self): return 1
        def terminate(self): pass
        def wait(self, timeout=None): return 1
        def kill(self): pass

    monkeypatch.setattr(R.subprocess, "Popen", lambda *a, **k: FakeProc())
    args = types.SimpleNamespace(config="c.yaml", limit=None, input=None, output_dir=None)
    try:
        with pytest.raises(SystemExit):
            R.run_workers(args, "generate")
    finally:
        signal.signal(signal.SIGTERM, old)
    assert shard.exists()
PYEOF

# ---- 5. verify -------------------------------------------------------------------
python -m py_compile src/teachers/run.py src/teachers/generate.py src/teachers/score.py src/teachers/common.py
python -m src.teachers.run --help | grep -q -- "--parallel" && echo ">> --parallel flag present"
python -m pytest tests -q
echo ">> parallel tests selected:"
python -m pytest tests -q -k "merge or shard or gpu or worker or read_jsonl" -v | tail -n 15

# ---- 6. commit (push only if PUSH=1) ----------------------------------------------
git add src tests
git commit -m "Harden parallel teacher run: atomic verified merge, worker supervision, resume-safe read_jsonl, score_info merge"
if [ "${PUSH:-0}" = "1" ]; then git push origin "$BR"; else echo ">> not pushed. Review, then: git push origin $BR"; fi
echo ">> HANDOVER COMMIT FOR FACULTY:"
git log --oneline -1
