#!/usr/bin/env bash
# post hoc (2026-11-04), SF-43 addendum 1, N1: the FROZEN sf43ana exactly as scripts/analyse_sf43.sh runs it in production (same arguments,
# Gate 5 `auto`, coverage summary and qualification of the pre-registration), but with prereg_posthoc_g7.json (g7.responses only changed).
#   bash run_n1.sh SCRATCH      # SCRATCH made by build_scratch.py; output in SCRATCH/camp/analysis/summary.json
set -uo pipefail
SCRATCH=$1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPTS=$(dirname "$HERE")
ART=$(dirname "$SCRIPTS")
REPO=$(cd "$ART/../../../.." && pwd)
SF42=$ART/../2026-10-30-sf42-confirmatory-tooling/scripts
cd "$REPO" || exit 1
export PYTHONPATH="$SCRIPTS:$SF42"
CAMP=$SCRATCH/camp
mkdir -p "$CAMP/analysis"
python3 -W default -m sf43ana run --prereg "$HERE/prereg_posthoc_g7.json" --campaign "$CAMP" --reduced "$CAMP/reduced" --gate5 auto --out "$CAMP/analysis"
RC=$?
[ "$RC" -eq 0 ] && [ -f "$CAMP/analysis/summary.json" ] && python3 -W default -m sf43ana check-finite --summary "$CAMP/analysis/summary.json"
exit $RC
