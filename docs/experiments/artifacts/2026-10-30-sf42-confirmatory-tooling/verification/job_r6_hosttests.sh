#!/usr/bin/env bash
# SF-42 R6 job sf42-r6-hosttests (CPU): the whole Python suite on the host stack, the host-only Par2 alpha_x test included
# (it must RUN, not skip), and the SF-39 tree untouched check.  Run from the mirror root.
set -u
R=docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling
ALPHAX=$HOME/sf42_campaign/r3/par2_alphax_s025.txt
LOG=$HOME/sf42_campaign/r6/hosttests.log
mkdir -p "$HOME/sf42_campaign/r6"
date -u +START_UTC=%FT%TZ
echo "TREE=$(cat .mf3d_source_rev)"
python3 -c 'import platform,numpy,scipy; print("python", platform.python_version(), "numpy", numpy.__version__, "scipy", scipy.__version__)'
echo "nproc_all=$(nproc --all)"
[ -f "$ALPHAX" ] || { echo "ALPHAX_FILE_MISSING $ALPHAX"; exit 1; }
(cd "$R/scripts" && OMP_NUM_THREADS=1 MCSTAT_PAR2_ALPHAX_S025=$ALPHAX python3 -W error -B -m unittest discover -s . -t . -v) > "$LOG" 2>&1
rc=$?
tail -n 6 "$LOG"
echo "unittest_rc=$rc"
echo "skipped_tests=$(grep -c ' \.\.\. skipped' "$LOG")"
grep ' \.\.\. skipped' "$LOG"
echo "SF39_NEWER_THAN_R3_START (empty expected):"
find "$HOME/sf39_campaign" -newer "$HOME/sf42_campaign/r3/.start" -print | head -20
echo SF39_UNTOUCHED_CHECK_DONE
date -u +END_UTC=%FT%TZ
exit $rc
