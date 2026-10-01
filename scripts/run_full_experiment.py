"""End-to-end experiment runner.

Runs every step that works on any machine (tests, PII evaluation).
GPU-only steps (teacher generation) are skipped with a clear message
when no NVIDIA GPU is available, instead of crashing.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run(name, cmd):
    print(f"\n=== {name} ===")
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode == 0


def main() -> int:
    py = sys.executable
    results = {}

    results["unit tests"] = run("Unit tests", [py, "-m", "pytest", "-q", "tests"])

    pii = ROOT / "data/synthetic/pii_cases_hard.jsonl"
    if pii.exists():
        results["PII evaluation"] = run(
            "PII evaluation", [py, "scripts/eval_pii.py", str(pii)]
        )
    else:
        print(f"\nSKIPPED PII evaluation: {pii} not found")
        results["PII evaluation"] = None

    print("\n=== GPU teacher steps ===")
    if has_gpu():
        print("GPU found. Run: python scripts/test_gpu.py, then see docs/faculty-run.md")
        results["GPU teacher run"] = None
    else:
        print("SKIPPED: no NVIDIA GPU here. Run test_gpu.py and run_teacher.py "
              "on Kaggle or the faculty GPU machine (see docs/faculty-run.md).")
        results["GPU teacher run"] = None

    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"{name:20s} {'PASS' if ok else 'SKIPPED' if ok is None else 'FAIL'}")
    return 0 if all(v is not False for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
