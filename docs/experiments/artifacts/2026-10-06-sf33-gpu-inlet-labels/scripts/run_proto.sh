#!/usr/bin/env bash
# SF-33 N5 campaign script, step 9a (claim (a): prototype reproduction on the GPU).
#
# Usage (from the repository root; on the V100 host as a detached job):
#   bash docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/run_proto.sh <build_dir> [<out_root>]
#   scripts/remote --increment SF-33 run sf33-proto -- \
#       "bash docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/run_proto.sh build/v100-release"
#
# Matrix (SF-33 spec step 9a; prototype i1o4 CASE lines exist in raw/sweep2 of the SF-29 artifact):
#   gauss, gauss_ch, control2d x eps 0.25, 0.5 x N 16, 24;  N 32 for gauss:0.25 and gauss_ch:0.25;
#   generic3d eps 1 at N 16, 20, 24 and eps 0.25 / 0.5 at N 16.
# Per case: export_proto.py if <out_root>/exports/<case>/case.json is absent (24^3 / 32^3 exports compute the
# prototype oracle: minutes of CPU each), conversion of the saved 16^3 prototype solution when it exists, then
# `inlet_slab --proto ... [--solution ...]`.  Logs: <out_root>/logs/proto/<case>.log; JSON summaries:
# <out_root>/raw/proto/<case>.json; finally compare_proto.py over all logs -> <out_root>/raw/proto/compare_proto.md.
# A non-zero driver exit (distinct per status) is recorded and the matrix continues.
# Env: PYTHON (python3), CASES (override the matrix: space-separated field:eps:N), EXTRA_ARGS (extra
# driver options, word-split, e.g. EXTRA_ARGS="--forcing fixed"). Extra driver options may also follow a
# literal `--` after the positional arguments:
#   run_proto.sh <build_dir> [<out_root>] [-- <driver options>...]   (SF-33 N7a)
# Both are appended to every `inlet_slab --proto` call (EXTRA_ARGS first) and printed in the header line.
set -euo pipefail

USAGE="usage: run_proto.sh <build_dir> [<out_root>] [-- <driver options>...]"
POS=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then
        shift
        break
    fi
    POS+=("$1")
    shift
done
DRIVER_EXTRA=()
if [ -n "${EXTRA_ARGS:-}" ]; then
    read -r -a DRIVER_EXTRA <<< "$EXTRA_ARGS"
fi
DRIVER_EXTRA+=("$@")
BUILD_DIR=${POS[0]:?$USAGE}
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=$(dirname "$SCRIPTS")
OUT=${POS[1]:-$ART}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
BIN=$(cd "$BUILD_DIR" && pwd)/inlet_slab
PY=${PYTHON:-python3}
SF29=$(cd "$ART/../2026-10-02-sf29-inlet-labels" && pwd)
EXPORTS=$OUT/exports
LOGS=$OUT/logs/proto
RAW=$OUT/raw/proto
mkdir -p "$EXPORTS/solutions" "$LOGS" "$RAW"
if [ ! -x "$BIN" ]; then
    echo "run_proto.sh: $BIN not found (build the inlet_slab target first)" >&2
    exit 2
fi

DEFAULT_CASES="gauss:0.25:16 gauss:0.25:24 gauss:0.25:32 gauss:0.5:16 gauss:0.5:24
gauss_ch:0.25:16 gauss_ch:0.25:24 gauss_ch:0.25:32 gauss_ch:0.5:16 gauss_ch:0.5:24
control2d:0.25:16 control2d:0.25:24 control2d:0.5:16 control2d:0.5:24
generic3d:1:16 generic3d:1:20 generic3d:1:24 generic3d:0.25:16 generic3d:0.5:16"
CASES=${CASES:-$DEFAULT_CASES}

run() {
    echo "+ $*"
    "$@"
}

echo "run_proto.sh: build=$BUILD_DIR bin=$BIN out=$OUT host=$(hostname) start=$(date -u +%FT%TZ)" \
    "extra_args=[${DRIVER_EXTRA[*]:-}]"
"$PY" -c 'import sys, numpy; print("run_proto.sh: python", sys.version.split()[0], "numpy", numpy.__version__)'
summary=()
for spec in $CASES; do
    IFS=: read -r field eps n <<< "$spec"
    name="${field}_${eps}_${n}"
    t0=$(date +%s)
    if [ ! -f "$EXPORTS/$name/case.json" ]; then
        (cd "$SCRIPTS" && run "$PY" export_proto.py "$spec" --out "$EXPORTS")
    else
        echo "  export $EXPORTS/$name exists (reused)"
    fi
    sol_args=()
    npz="$SF29/raw/sweep2/solutions/${name}_i1o4.npz"
    if [ -f "$npz" ]; then
        if [ ! -f "$EXPORTS/solutions/${name}_i1o4/solution.json" ]; then
            (cd "$SCRIPTS" && run "$PY" export_proto.py --solutions "$npz" --out "$EXPORTS/solutions")
        fi
        sol_args=(--solution "$EXPORTS/solutions/${name}_i1o4")
    fi
    set +e
    echo "+ $BIN --proto $EXPORTS/$name ${sol_args[*]:-} --summary $RAW/$name.json ${DRIVER_EXTRA[*]:-}" \
        "> $LOGS/$name.log 2>&1"
    "$BIN" --proto "$EXPORTS/$name" ${sol_args[@]+"${sol_args[@]}"} --summary "$RAW/$name.json" \
        ${DRIVER_EXTRA[@]+"${DRIVER_EXTRA[@]}"} > "$LOGS/$name.log" 2>&1
    rc=$?
    set -e
    st=$(grep '^STATUS ' "$LOGS/$name.log" | tail -1 || true)
    echo "  CASE_RESULT $name rc=$rc ${st:-STATUS ?} wall=$(( $(date +%s) - t0 ))s"
    summary+=("$name rc=$rc ${st:-STATUS ?}")
done

echo "run_proto.sh: driver results"
for s in "${summary[@]}"; do
    echo "  $s"
done
set +e
run "$PY" "$SCRIPTS/compare_proto.py" "$LOGS"/*.log --summary "$SF29/raw/sweep2/summary.md" \
    --exports "$EXPORTS" --out "$RAW/compare_proto.md"
crc=$?
set -e
echo "run_proto.sh: compare_proto.py exit $crc; table $RAW/compare_proto.md; end=$(date -u +%FT%TZ)"
