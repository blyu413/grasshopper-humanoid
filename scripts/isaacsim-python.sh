#!/usr/bin/env bash
set -euo pipefail

NEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACSIM_DIR="${NEST_ISAACSIM_DIR:-$(dirname -- "$NEST_DIR")/isaac-sim-6.1.0}"
NEST_PYTHON="$NEST_DIR/.venv/bin/python"

if [[ ! -x "$NEST_PYTHON" ]]; then
    echo "NEST .venv is missing. Run 'uv sync --locked' in $NEST_DIR." >&2
    exit 1
fi
if [[ ! -x "$ISAACSIM_DIR/python.sh" ]]; then
    echo "Isaac Sim is missing at $ISAACSIM_DIR. Set NEST_ISAACSIM_DIR to its installation directory." >&2
    exit 1
fi

NEST_SITE_PACKAGES="$($NEST_PYTHON -c 'import site; print(site.getsitepackages()[0])')"
export VIRTUAL_ENV="$NEST_DIR/.venv"
export PYTHONEXE="$NEST_PYTHON"
export PYTHONPATH="$NEST_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

exec "$ISAACSIM_DIR/python.sh" "$@"
