#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  DEMO_PYTHON=".venv/bin/python"
elif [[ -x ../../work/venv/bin/python ]]; then
  DEMO_PYTHON="../../work/venv/bin/python"
else
  python3 -m venv .venv
  DEMO_PYTHON=".venv/bin/python"
  "$DEMO_PYTHON" -m pip install -e .
fi
if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
fi
exec "$DEMO_PYTHON" -m voice_demo.cli web
