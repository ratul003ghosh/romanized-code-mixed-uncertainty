import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sample_carriers.py"


def run(tmp_path, rows, quota):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    inp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    subprocess.run([sys.executable, str(SCRIPT), "--inp", str(inp), "--out", str(out), "--quota", quota],
                   check=True, capture_output=True, text=True)
    return [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]


def test_lines_with_possible_real_pii_are_dropped(tmp_path):
    rows = [{"id": 1, "text": "call me at 01712345678 bhai", "source": "S"},
            {"id": 2, "text": "mail dao rahim@gmail.com e", "source": "S"},
            {"id": 3, "text": "number +8801712 dilam", "source": "S"},
            {"id": 4, "text": "ajke onek gorom porse", "source": "S"}]
    assert [r["id"] for r in run(tmp_path, rows, "S=10")] == [4]


def test_quota_per_source_and_dialect_balance(tmp_path):
    rows = [{"id": f"a{i}", "text": "kal dekha hobe", "source": "A"} for i in range(50)]
    rows += [{"id": f"n{i}", "text": "tui kene aso", "source": "B", "dialect": "Noakhali"} for i in range(30)]
    rows += [{"id": f"c{i}", "text": "tui ken aso", "source": "B", "dialect": "Chittagong"} for i in range(5)]
    out = run(tmp_path, rows, "A=20,B=10")
    by_source = {s: [r for r in out if r["source"] == s] for s in ("A", "B")}
    assert len(by_source["A"]) == 20 and len(by_source["B"]) == 10
    dialects = [r["dialect"] for r in by_source["B"]]
    assert dialects.count("Chittagong") == 5 and dialects.count("Noakhali") == 5


def test_source_without_quota_is_left_out(tmp_path):
    rows = [{"id": 1, "text": "ekta kotha bolar chilo", "source": "X"}]
    assert run(tmp_path, rows, "Y=5") == []
