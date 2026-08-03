#!/usr/bin/env bash
set -euo pipefail

repo=${1:?usage: setup_wsl_triton.sh REPO [VENV]}
venv=${2:-$HOME/.cache/raggedroute/topk-triton-venv}

python3 -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install --requirement "$repo/benchmarks/triton/requirements-wsl.txt"
"$venv/bin/python" -m pip freeze
