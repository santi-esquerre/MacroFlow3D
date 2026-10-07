#!/usr/bin/env bash
# SF-33 N8' campaign B (gate-reduced; see analysis/README.md, N8' section). Three parts, one per detached job:
#
#   run_campaign_b.sh <build_dir> ladder  [NS]   eps 0.25 production ladder (default NS = "32 64 128"),
#                                                logs/ladder_0.25/N<N>.log, raw/ladder_0.25/N<N>.json,
#                                                /usr/bin/time -v in logs/ladder_0.25/N<N>.time (host max RSS),
#                                                then ladder_orders.py -> raw/ladder_0.25/ladder_orders.md
#   run_campaign_b.sh <build_dir> prod32-05      production 32^3 at eps 0.5 (same SF-18 field), logs/prod32_05/
#   run_campaign_b.sh <build_dir> oracle32       --cells <exports>/crosscheck_gauss_0.25_N/Y_cells.npy --eps 1
#                                                at N = 16, 24, 32 with --save-oracle (exports/oracle_gpu/N<N>),
#                                                then compare_oracle_sf29.py vs exports/gauss_0.25_N/psi_or_*.npy
#
# Everything the driver does is its default (SF-33 C3 solver: P-A + coarse mult/2 colored banded, Psi-tc on,
# EW forcing, restart 100, inner cap 6000, max-newton 120; printed on the SOLVER line of every log), plus the
# production options stated below.  Run from the repository root.  Every command is printed.
set -uo pipefail

BUILD_DIR=${1:?usage: run_campaign_b.sh <build_dir> ladder|prod32-05|oracle32 [NS]}
PART=${2:?usage: run_campaign_b.sh <build_dir> ladder|prod32-05|oracle32 [NS]}
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=$(dirname "$SCRIPTS")
BIN=$(cd "$BUILD_DIR" && pwd)/inlet_slab
PY=${PYTHON:-python3}
THREADS=${THREADS:-32}
EXPORTS=$ART/exports
FIELD=(--sigma2 1 --ell 0.25 --seed 3001)
[ -x "$BIN" ] || { echo "run_campaign_b.sh: $BIN not found" >&2; exit 2; }
echo "run_campaign_b.sh: part=$PART bin=$BIN threads=$THREADS host=$(hostname) start=$(date -u +%FT%TZ)"

drive() { # drive <log> <time_file> args...
    local log=$1 tf=$2
    shift 2
    echo "+ /usr/bin/time -v -o $tf $BIN $* > $log 2>&1"
    local t0
    t0=$(date +%s)
    /usr/bin/time -v -o "$tf" "$BIN" "$@" > "$log" 2>&1
    local rc=$?
    echo "  CASE_RESULT log=$log rc=$rc $(grep '^STATUS ' "$log" | tail -1) wall=$(( $(date +%s) - t0 ))s" \
         "max_rss_kB=$(grep 'Maximum resident' "$tf" | awk '{print $NF}') end=$(date -u +%FT%TZ)"
    grep -E '^(SOLVER|PATH|CASE|ORACLE_SUMMARY|TIMING|MEMORY|STATUS_DETAIL)' "$log" | sed 's/^/    /'
}

case "$PART" in
ladder)
    NS=${3:-"32 64 128"}
    LOGS=$ART/logs/ladder_0.25
    RAW=$ART/raw/ladder_0.25
    mkdir -p "$LOGS" "$RAW"
    logs=()
    for n in $NS; do
        drive "$LOGS/N$n.log" "$LOGS/N$n.time" --production --n "$n" --eps 0.25 "${FIELD[@]}" --oracle-ladder \
            --threads "$THREADS" --summary "$RAW/N$n.json"
        logs+=("$LOGS/N$n.log")
    done
    echo "+ $PY $SCRIPTS/ladder_orders.py ${logs[*]} --out $RAW/ladder_orders.md"
    "$PY" "$SCRIPTS/ladder_orders.py" "${logs[@]}" --out "$RAW/ladder_orders.md"
    echo "  ladder_orders rc=$?"
    echo "+ $PY $SCRIPTS/digest_newton.py $RAW/N*.json > $RAW/digest_newton.txt"
    "$PY" "$SCRIPTS/digest_newton.py" "$RAW"/N*.json > "$RAW/digest_newton.txt"
    echo "  digest_newton rc=$?"
    ;;
prod32-05)
    LOGS=$ART/logs/prod32_05
    RAW=$ART/raw/prod32_05
    mkdir -p "$LOGS" "$RAW"
    drive "$LOGS/N32.log" "$LOGS/N32.time" --production --n 32 --eps 0.5 "${FIELD[@]}" --oracle-ladder \
        --threads "$THREADS" --summary "$RAW/N32.json"
    echo "+ $PY $SCRIPTS/digest_newton.py $RAW/N32.json > $RAW/digest_newton.txt"
    "$PY" "$SCRIPTS/digest_newton.py" "$RAW/N32.json" > "$RAW/digest_newton.txt"
    echo "  digest_newton rc=$?"
    ;;
oracle32)
    LOGS=$ART/logs/oracle32
    RAW=$ART/raw/oracle32
    mkdir -p "$LOGS" "$RAW"
    cases=()
    for n in 16 24 32; do
        drive "$LOGS/N$n.log" "$LOGS/N$n.time" --production --n "$n" --eps 1 \
            --cells "$EXPORTS/crosscheck_gauss_0.25_$n/Y_cells.npy" --oracle-ladder --threads "$THREADS" \
            --save-oracle "$EXPORTS/oracle_gpu/N$n" --summary "$RAW/N$n.json"
        cases+=(--case "$n" "$EXPORTS/oracle_gpu/N$n" "$EXPORTS/gauss_0.25_$n")
    done
    for suf in "" _h16_tol1e-08 _h16_tol1e-10; do
        out=$RAW/oracle_vs_sf29${suf}.md
        echo "+ $PY $SCRIPTS/compare_oracle_sf29.py ${cases[*]} --suffix '$suf' --out $out"
        "$PY" "$SCRIPTS/compare_oracle_sf29.py" "${cases[@]}" --suffix "$suf" --out "$out"
        echo "  compare rc=$?"
    done
    ;;
*)
    echo "run_campaign_b.sh: unknown part '$PART'" >&2
    exit 2
    ;;
esac
echo "run_campaign_b.sh: part=$PART end=$(date -u +%FT%TZ)"
