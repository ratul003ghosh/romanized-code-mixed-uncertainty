"""GPU smoke test for the teacher ensemble (issue #7). Run from the repo root:

    python scripts/test_gpu.py

Steps: environment report -> CPU unit/tiny end-to-end tests -> teacher/student tokenizer
check -> 5 synthetic inputs x 4 teachers on Qwen2.5-7B -> experiments/teacher/smoke_results.tar.gz
"""
import os
import subprocess
import sys
import tarfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
OUT = "experiments/teacher/smoke"
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "smoke_console.log"), "w")


def step(title, cmd, required=True):
    print(f"\n== {title}", flush=True)
    log.write(f"\n== {title}\n")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout, flush=True)
    log.write(p.stdout)
    log.flush()
    if p.returncode and required:
        print(f"FAILED: {title}. Send back {OUT}/smoke_console.log")
        sys.exit(1)
    return p.returncode


py = sys.executable
step("environment", [py, "-c",
     "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__); "
     "print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); "
     "print('gpu mem GB', round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else 0)"])
step("unit + tiny end-to-end tests", [py, "-m", "pytest", "-q", "tests/test_teacher_core.py", "tests/test_teacher_pipeline.py"])
step("tokenizer check teacher vs student (proposal 4.1)", [py, "scripts/vocab_check.py"], required=False)
step("smoke run: 5 inputs x 4 teachers", [py, "scripts/run_teacher.py", "--config", "configs/teacher_smoke.yaml"])
log.close()

tar_path = "experiments/teacher/smoke_results.tar.gz"
with tarfile.open(tar_path, "w:gz") as t:
    t.add(OUT, arcname="smoke")
print(f"\nDONE. Send back: {tar_path}")
