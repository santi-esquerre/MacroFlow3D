#!/usr/bin/env bash
# SF-33 N5 campaign script, step 9b (claim (b): production refinement ladder 32 / 64 / 128).
#
# Usage (from the repository root; on the V100 host as a detached job, one eps per job):
#   bash docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/run_ladder.sh <build_dir> <eps> [<out_root>]
#   scripts/remote --increment SF-33 run sf33-ladder-0.5 -- \
#       "bash docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/run_ladder.sh build/v100-release 0.5"
#
# One SF-18 continuum field (sigma2 = 1, ell = 1/4, seed 3001; stage field k = exp(eps Y)), grids N = 32, 64, 128
# (ell/h = 8, 16, 32), one case after another: `inlet_slab --production --n N --eps <eps> --sigma2 1 --ell 0.25
# --seed 3001 --oracle-ladder --threads 32`.  Logs <out_root>/logs/ladder_<eps>/N<N>.log, JSON
# <out_root>/raw/ladder_<eps>/N<N>.json, then ladder_orders.py -> <out_root>/raw/ladder_<eps>/ladder_orders.md.
# Env: PYTHON (python3), NS (grids, default "32 64 128"), THREADS (32), EXTRA_ARGS (appended to every driver call).
set -euo pipefail

BUILD_DIR=${1:?usage: run_ladder.sh <build_dir> <eps> [<out_root>]}
EPS=${2:?usage: run_ladder.sh <build_dir> <eps> [<out_root>]}
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=$(dirname "$SCRIPTS")
OUT=${3:-$ART}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
BIN=$(cd "$BUILD_DIR" && pwd)/inlet_slab
PY=${PYTHON:-python3}
LOGS=$OUT/logs/ladder_$EPS
RAW=$OUT/raw/ladder_$EPS
mkdir -p "$LOGS" "$RAW"
if [ ! -x "$BIN" ]; then
    echo "run_ladder.sh: $BIN not found (build the inlet_slab target first)" >&2
    exit 2
fi
NS=${NS:-"32 64 128"}
THREADS=${THREADS:-32}
read -r -a extra <<< "${EXTRA_ARGS:-}"

run() {
    echo "+ $*"
    "$@"
}

echo "run_ladder.sh: build=$BUILD_DIR bin=$BIN eps=$EPS grids=($NS) threads=$THREADS out=$OUT host=$(hostname)" \
     "start=$(date -u +%FT%TZ)"
logs=()
for n in $NS; do
    t0=$(date +%s)
    set +e
    echo "+ $BIN --production --n $n --eps $EPS --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder" \
         "--threads $THREADS ${extra[*]:-} --summary $RAW/N$n.json > $LOGS/N$n.log 2>&1"
    "$BIN" --production --n "$n" --eps "$EPS" --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder \
        --threads "$THREADS" ${extra[@]+"${extra[@]}"} --summary "$RAW/N$n.json" > "$LOGS/N$n.log" 2>&1
    rc=$?
    set -e
    echo "  N=$n rc=$rc $(grep '^STATUS ' "$LOGS/N$n.log" | tail -1 || true) wall=$(( $(date +%s) - t0 ))s"
    grep -E '^(PATH|CASE|ORACLE_SUMMARY|TIMING|MEMORY|STATUS_DETAIL)' "$LOGS/N$n.log" || true
    logs+=("$LOGS/N$n.log")
done
run "$PY" "$SCRIPTS/ladder_orders.py" "${logs[@]}" --out "$RAW/ladder_orders.md"
echo "run_ladder.sh: end=$(date -u +%FT%TZ)"
