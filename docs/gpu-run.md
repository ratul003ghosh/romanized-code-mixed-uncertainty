# GPU Handoff Instructions

This code must be run on the external GPU instance.

## Setup
1. Clone the repository.
2. Install requirements: `pip install -r requirements.txt`
3. Check GPU status: `python scripts/test_gpu.py`

## Running Teachers
Use `scripts/run_teacher.py` to generate silver labels and uncertainty scores.

All outputs will be saved to `results/teacher/`.
Return the `results/` folder to the main team.
