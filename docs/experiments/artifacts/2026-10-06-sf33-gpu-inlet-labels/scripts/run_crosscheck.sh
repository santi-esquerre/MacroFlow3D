#!/usr/bin/env bash
# SF-33 N5 campaign script, step 8 (SF-19 cross-check on the SF-29 `gauss` field at 16 / 24 / 32).
#
# Usage (from the repository root; on the V100 host as a detached job):
#   bash docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/run_crosscheck.sh <build_dir> [<out_root>]
#   scripts/remote --increment SF-33 run sf33-crosscheck -- \
#       "bash docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/run_crosscheck.sh build/v100-release"
#
# Exports `export_proto.py --crosscheck gauss:0.25:{16,24,32}` when absent, runs `inlet_slab --sf19-crosscheck`
# on each (logs <out_root>/logs/crosscheck/N<N>.log, JSON <out_root>/raw/crosscheck/N<N>.json) and tabulates the
# differences with their observed orders (compare_proto.py --crosscheck -> <out_root>/raw/crosscheck/crosscheck.md).
# Env: PYTHON (python3), NS (grids, default "16 24 32").
set -euo pipefail

BUILD_DIR=${1:?usage: run_crosscheck.sh <build_dir> [<out_root>]}
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=$(dirname "$SCRIPTS")
OUT=${2:-$ART}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
BIN=$(cd "$BUILD_DIR" && pwd)/inlet_slab
PY=${PYTHON:-python3}
EXPORTS=$OUT/exports
LOGS=$OUT/logs/crosscheck
RAW=$OUT/raw/crosscheck
mkdir -p "$EXPORTS" "$LOGS" "$RAW"
if [ ! -x "$BIN" ]; then
    echo "run_crosscheck.sh: $BIN not found (build the inlet_slab target first)" >&2
    exit 2
fi
NS=${NS:-"16 24 32"}

run() {
    echo "+ $*"
    "$@"
}

echo "run_crosscheck.sh: build=$BUILD_DIR bin=$BIN out=$OUT host=$(hostname) start=$(date -u +%FT%TZ)"
logs=()
for n in $NS; do
    d="$EXPORTS/crosscheck_gauss_0.25_$n"
    if [ ! -f "$d/Y_cells.npy" ]; then
        (cd "$SCRIPTS" && run "$PY" export_proto.py --crosscheck "gauss:0.25:$n" --out "$EXPORTS")
    else
        echo "  export $d exists (reused)"
    fi
    set +e
    echo "+ $BIN --sf19-crosscheck $d --summary $RAW/N$n.json > $LOGS/N$n.log 2>&1"
    "$BIN" --sf19-crosscheck "$d" --summary "$RAW/N$n.json" > "$LOGS/N$n.log" 2>&1
    rc=$?
    set -e
    echo "  N=$n rc=$rc $(grep '^STATUS ' "$LOGS/N$n.log" | tail -1 || true)"
    grep '^CROSSCHECK N=' "$LOGS/N$n.log" || true
    logs+=("$LOGS/N$n.log")
done
run "$PY" "$SCRIPTS/compare_proto.py" --crosscheck "${logs[@]}" --out "$RAW/crosscheck.md"
echo "run_crosscheck.sh: end=$(date -u +%FT%TZ)"
