#!/usr/bin/env bash
# Baixa as TCs públicas do laboratório do MVP (catálogo em samples.py).
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  .venv/bin/python samples.py
else
  python3 samples.py
fi
