"""Demo script.
Inputs are currently author-written placeholders (dataset access pending).
PII detection/masking/arbitration are REAL — run on whatever input is provided.
Teacher and student outputs are PENDING until real runs exist.
"""
import sys, os, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.pii import masking, arbitrate

TEACHER_OUT = os.path.join(ROOT, "outputs/teacher/teacher_outputs.jsonl")
STUDENT_OUT = os.path.join(ROOT, "outputs/student/student_outputs.jsonl")

def load_by_id(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {r.get("id"): r for r in map(json.loads, f)}

def run_pii(text):
    sanitized = masking.sanitize(text, 0.5)
    arbitration = arbitrate.enforce(sanitized["masked"], sanitized["pii"])
    return sanitized, arbitration

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--example"); p.add_argument("--all", action="store_true")
    a = p.parse_args()
    examples = json.load(open(os.path.join(ROOT, "examples/demo_examples.json")))
    teacher, student = load_by_id(TEACHER_OUT), load_by_id(STUDENT_OUT)

    for ex in examples:
        if not (a.all or ex["id"] == a.example):
            continue
        sanitized, arbitration = run_pii(ex["input"])
        print("=" * 70)
        print("EXAMPLE      :", ex["id"])
        print("INPUT SOURCE :", ex["input_source"])
        print("INPUT        :", ex["input"])
        print("PII DETECTED :", sanitized["pii"])
        print("TEACHER INPUT: PII kept visible (by design)")
        print("MASKED OUTPUT:", sanitized["masked"])
        print("RULE 1 CHECK :", arbitration)
        print("TEACHER UNC. :", teacher.get(ex["id"], "not found") if teacher else "⏳ PENDING teacher GPU run")
        print("STUDENT CONF.:", student.get(ex["id"], "not found") if student else "⏳ PENDING student training")
        print("ROUTE        : ⏳ PENDING | expected:", ex["expected"]["route"])
        print("WHY          :", ex["expected"]["why"])

if __name__ == "__main__":
    main()
