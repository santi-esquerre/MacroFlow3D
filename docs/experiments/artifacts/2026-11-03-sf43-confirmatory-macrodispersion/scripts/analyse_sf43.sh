#!/usr/bin/env bash
# SF-43 final analysis: the engine's `analysis.command` (plan.json; the smoke overlay adds --smoke).  Frozen with the plan by sha256.
#
#   bash analyse_sf43.sh <campaign_dir> <build_dir> [--smoke]
#
# Production: `python3 -W default -m sf43ana run` (warnings are recorded in summary.json process.warnings, never fatal) with the committed pre-registration <A>/prereg.json, Gate 5 `auto` with the committed coverage
# summary (<A>/coverage/summary.json, the path in the pre-registration) and the qualification document of the robust sign (<A>/coverage/qualification.json,
# the pre-registration's `qualification`; without it every robust sign has p = 1), figures; outputs in <campaign_dir>/analysis/ (summary.json, figures/); then `check-finite` is run and its output printed (counts only, incl. warnings_count / errors_count; its exit status
# does not change the exit code, so the recompute / reconcile steps after this command still run).
# Smoke (--smoke): <A>/prereg_smoke.json, Gate 5 `auto` with the smoke job's OWN coverage output <campaign_dir>/coverage/summary.json if it
# exists, else Gate 5 descriptive (`--coverage-summary none`); likewise <campaign_dir>/coverage/qualification.json, else `--qualification none` (every robust
# sign p = 1, status "no qualification document": allowed in the smoke).  The analysis stdout / stderr go to FILES in <campaign_dir>/analysis/ and the
# ONLY thing printed is the output of `python3 -m sf43ana check-finite` (structure and counts): no estimand, bound, p-value, decision or
# class may reach stdout, stderr or the job log (the smoke test must not read alpha beyond finiteness).
# Exit code: 0 ok (also with per-hypothesis errors, which are inside summary.json); 1 the analysis (or, in smoke mode, check-finite) failed; 2 usage.  Runs from the repository root (relative paths of the plan).
set -uo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ] || { [ $# -eq 3 ] && [ "$3" != "--smoke" ]; }; then
    echo "usage: analyse_sf43.sh <campaign_dir> <build_dir> [--smoke]" >&2
    exit 2
fi
CAMP=$1
SMOKE=0
[ $# -eq 3 ] && SMOKE=1

SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=$(dirname "$SCRIPTS")
REPO=$(cd "$ART/../../../.." && pwd)
SF42=$ART/../2026-10-30-sf42-confirmatory-tooling/scripts
PY=${PYTHON:-python3}
cd "$REPO" || exit 1
export PYTHONPATH="$SCRIPTS:$SF42${PYTHONPATH:+:$PYTHONPATH}"
OUT="$CAMP/analysis"
mkdir -p "$OUT" || exit 1

if [ "$SMOKE" -eq 0 ]; then
    "$PY" -W default -m sf43ana run --prereg "$ART/prereg.json" --campaign "$CAMP" --reduced "$CAMP/reduced" --gate5 auto --out "$OUT"
    RC=$?
    [ "$RC" -eq 0 ] && [ -f "$OUT/summary.json" ] && { "$PY" -W default -m sf43ana check-finite --summary "$OUT/summary.json" || true; }
    exit $RC
fi

COV=none
[ -f "$CAMP/coverage/summary.json" ] && COV="$CAMP/coverage/summary.json"
QUAL=none
[ -f "$CAMP/coverage/qualification.json" ] && QUAL="$CAMP/coverage/qualification.json"
"$PY" -W default -m sf43ana run --prereg "$ART/prereg_smoke.json" --campaign "$CAMP" --reduced "$CAMP/reduced" --gate5 auto \
    --coverage-summary "$COV" --qualification "$QUAL" --out "$OUT" > "$OUT/run.stdout.txt" 2> "$OUT/run.stderr.txt"
RC=$?
if [ "$RC" -ne 0 ] || [ ! -f "$OUT/summary.json" ]; then
    echo "ANALYSIS_RUN_FAILED rc=$RC (details in analysis/run.stderr.txt of the campaign directory)"
    exit 1
fi
"$PY" -W default -m sf43ana check-finite --summary "$OUT/summary.json"
exit $?
