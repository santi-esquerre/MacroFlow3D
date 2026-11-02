#!/bin/bash
# SF-42 R4 host job: bash job.sh J1|J2|J3 [--reduce]   (run through `scripts/remote --increment SF-42 run`, one job at a time).
# Host outputs ONLY under ~/sf42_campaign/pilots/.  ~/sf39_campaign is read-only (hostref.py reads solver_b.json); the last lines
# print every file under it newer than ~/sf42_campaign/r3/.start (must be empty: nothing of SF-39 is touched).
# The job exit is 0 when the driver ran to its end; failed runs are RESULTS (RUN.json exit codes, "failed_runs=" line), not job errors.
set -u
JOB=${1:?usage: job.sh J1|J2|J3 [--reduce]}
shift
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/.." || exit 2          # scripts/ (the unittest root; sf42red, mcstat, seedlattice, pilots importable)
OUT=~/sf42_campaign/pilots
BUILD=~/MacroFlow3D-SF-42/build/v100-release
mkdir -p "$OUT"
echo "JOB_START $JOB $(date -u +%FT%TZ) tree=$(cat ~/MacroFlow3D-SF-42/.mf3d_source_rev 2> /dev/null)"
if [ ! -f "$OUT/hostref.json" ]; then
    python3 -B -m pilots.hostref --sf39 ~/sf39_campaign --r3 ~/sf42_campaign/r3 --out "$OUT/hostref.json" || echo "HOSTREF_FAILED"
fi
python3 -B -m pilots.run "$JOB" --build "$BUILD" --out "$OUT" "$@"
echo "DRIVER_EXIT=$?"
find ~/sf39_campaign -newer ~/sf42_campaign/r3/.start -print | head -20
echo SF39_UNTOUCHED_CHECK_DONE
echo "JOB_END $JOB $(date -u +%FT%TZ)"
exit 0
