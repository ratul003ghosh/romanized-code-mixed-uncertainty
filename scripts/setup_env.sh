#!/usr/bin/env bash
# Creates the Python 3.11 environment. Linux + NVIDIA GPU (vllm does not run natively on Windows; use WSL2).
set -euo pipefail
cd "$(dirname "$0")/.."

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install vllm                 # brings a matching torch + CUDA build
pip install -r requirements.txt  # everything else

python - <<'PY'
import torch, transformers, vllm, jsonschema, datasketch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print("transformers", transformers.__version__, "| vllm", vllm.__version__)
PY

pip freeze > requirements.lock.txt
python -m pytest -q tests/
echo "Environment ready. Versions frozen in requirements.lock.txt"
