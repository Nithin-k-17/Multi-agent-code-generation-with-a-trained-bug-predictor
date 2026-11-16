#!/usr/bin/env bash
# One-time setup on macOS / Linux: creates a virtual environment and installs the Python libraries.
# Run from the project folder:  bash setup_unix.sh
set -e
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
$PY --version || { echo "Python 3.10+ is required: https://www.python.org/downloads/"; exit 1; }

if [ ! -d .venv ]; then
    echo "Creating virtual environment .venv ..."
    $PY -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
python main.py check || true
echo
echo "Setup finished. Next time, activate the environment with:  source .venv/bin/activate"
echo "Then start the website with:  python main.py serve"
