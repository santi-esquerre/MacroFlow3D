#!/usr/bin/env bash
# SF-43 campaign launcher: one detached V100 job runs the whole study (sf43camp/README.md).  Modelled on run_sf39.sh (script-relative
# paths, no `git -C`, bash 4.2 safe); all logic is in the Python package.
#
#   bash run_sf43.sh <build_dir> all [--resume] [--dry-run] [--smoke] [--plan <file>]
#
# Run from anywhere; the script enters the repository root (the mirror root on the V100), so <build_dir>, the plan's campaign_dir and the
# plan's other relative paths are relative to it.  --plan defaults to <artifact>/plan.json.
# Exit codes: 0 DONE, 10 DONE_PARTIAL (cutoff or halted arm or lane crash), 11 STOPPED (budget or disk), 12 analysis failed,
# 13 refused to start (md5 mismatch, invalid plan, stale state, usage), 143 cancelled by SIGTERM/SIGINT.  Never 70, 75, 97, 130.
# PYTHON overrides the interpreter.
set -uo pipefail

SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=$(dirname "$SCRIPTS")
REPO=$(cd "$ART/../../../.." && pwd)
SF42=$ART/../2026-10-30-sf42-confirmatory-tooling/scripts
PY=${PYTHON:-python3}

if [ $# -lt 2 ]; then
    echo "usage: run_sf43.sh <build_dir> all [--resume] [--dry-run] [--smoke] [--plan <file>]" >&2
    exit 13
fi
cd "$REPO" || exit 13
export PYTHONPATH="$SCRIPTS:$SF42${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -W error -m sf43camp "$@" --artifact "$ART"
