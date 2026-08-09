#!/usr/bin/env bash
# Cross-platform convenience runner: venv + deps + full training run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PYTHON:-python3}"

if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    "$PY" -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "Running tests..."
python -m pytest tests/ -q

echo "Training (this takes ~15-25 minutes on Apple Silicon MPS / a modern CPU)..."
python train.py "$@"
