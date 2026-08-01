#!/usr/bin/env bash
# Snapshot a workspace and run the two-process judge on it.
#
# usage: run_judge.sh <workspace_dir> <nonce>
#
# The snapshot (workspace minus the shipped starter/ tree, judge substitutes
# pristine) is taken by judge.py itself via --workspace; files are copied and
# made read-only before any agent code can run.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
export OMP_NUM_THREADS=2
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3
exec "$PY" "$HERE/judge.py" --workspace "$1" --nonce "$2"
