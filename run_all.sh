#!/usr/bin/env bash
# Run the full analysis pipeline in order.
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=python3
fi

for script in src/0*.py; do
  echo "== running $script"
  "$PY" "$script"
done

echo "Done. Dashboard: outputs/figures/dashboard.png"
