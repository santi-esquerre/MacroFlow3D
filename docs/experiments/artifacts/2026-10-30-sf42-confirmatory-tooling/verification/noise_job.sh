#!/bin/bash
# SF-42 R3 J4 (host job): estimator-noise pilot, no new runs.  Reads the SF-39 consumer arrays read-only; writes only under
# ~/sf42_campaign/r3/noise.  Production set of an amplitude = DONE realizations of <a>_A (pass hmax_8) + <a>_B (single pass).
# The A arms also hold hmax_16/hmax_32 passes: not reduced here (the production pass is hmax_8).
set -eu
V="$(cd "$(dirname "$0")" && pwd)"
SC="$V/../scripts"
RUNS=~/sf39_campaign/runs
OUT=~/sf42_campaign/r3/noise
mkdir -p "$OUT"
cd "$SC"
for a in s025 s1 s225 s4; do
  passes=()
  for d in "$RUNS"/${a}_A/realizations/r_*/; do
    [ -f "$d/DONE" ] && passes+=("${d}hmax_8")
  done
  echo "SET $a: ${#passes[@]} A passes + arm ${a}_B"
  python3 -B -m sf42red reduce --consumer "$RUNS/${a}_B" "${passes[@]}" --root ~/sf39_campaign --out "$OUT/$a.csv"
done
python3 -B -W error "$V/noise_summary.py" "$SC" "$OUT" s025 s1 s225 s4 | tee "$OUT/noise_summary.csv"
find ~/sf39_campaign -newer ~/sf42_campaign/r3/.start -print | head -20
echo SF39_UNTOUCHED_CHECK_DONE
