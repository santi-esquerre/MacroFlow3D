#!/usr/bin/env bash
# SF-35 campaign runner (one part per detached V100 job).  Run from the repository root:
#
#   bash <A>/scripts/run_campaign.sh <build_dir> controls [NS]       positive controls A1-A3 (+ A4 via ctest), NS default "16 32 64"
#   bash <A>/scripts/run_campaign.sh <build_dir> xcons               cross-construction (--solve) at (eps,N) = (0.25,32) (0.25,64) (0.5,32)
#   bash <A>/scripts/run_campaign.sh <build_dir> regress             G: --production 32^3 eps 0.25 vs the SF-33 log (timing fields masked)
#   bash <A>/scripts/run_campaign.sh <build_dir> matrix <05|10|15> [NS]   eps = 0.5 / 1.0 / 1.5, NS default "32 64 128 256"
#   bash <A>/scripts/run_campaign.sh <build_dir> info                informative: sigma2 2.25 eps 1 vs sigma2 1 eps 1.5 at 32^3
#   bash <A>/scripts/run_campaign.sh <build_dir> gpu [NS]            host vs gpu tracer (only if built), NS default "128 256"
#   bash <A>/scripts/run_campaign.sh <build_dir> smoke               LOCAL ONLY: all of the above at 16^3 (not campaign evidence)
#
# <A> = docs/experiments/artifacts/2026-10-07-sf35-label-transport.
# Every driver run uses the driver defaults for the integration settings (--hmax-div 8 --tol 1e-8
# --max-roundtrip 1e-8; the ladder settings are fixed inside the driver) and the SF-18 field
# --sigma2 1 --ell 0.25 --seed 3001.  Every command is printed, every exit code is recorded in
# logs/<part>/exit_codes.tsv, a failing run is NEVER retried with other settings, and the
# pre-registered verdicts are computed by the digests (scripts/*.py).
# Environment: THREADS (default 32), PYTHON (default python3), NS (overrides the part default),
# LADDER_NS (matrix: grids that get --ladder, default "32 64"), INFO_N (info: grid, default 32).
# The synced tree has no .git: N7 writes the file <A>/COMMIT before syncing; absent -> "unknown".
# Exit status: 0 unless a run returned a code outside the part's allowed set or a digest failed.
set -uo pipefail

BUILD_DIR=${1:?usage: run_campaign.sh <build_dir> controls|xcons|regress|matrix|info|gpu|smoke [args]}
PART=${2:?usage: run_campaign.sh <build_dir> controls|xcons|regress|matrix|info|gpu|smoke [args]}
shift 2
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=$(dirname "$SCRIPTS")
SF33=$(dirname "$ART")/2026-10-06-sf33-gpu-inlet-labels
BIN=$(cd "$BUILD_DIR" && pwd)/inlet_slab
PY=${PYTHON:-python3}
THREADS=${THREADS:-32}
FIELD=(--sigma2 1 --ell 0.25 --seed 3001)
TAG=""   # "smoke/" under the smoke part
APFX=""  # "smoke_" under the smoke part
BAD=0

[ -x "$BIN" ] || { echo "run_campaign.sh: $BIN not found" >&2; exit 2; }
echo "run_campaign.sh: part=$PART bin=$BIN threads=$THREADS host=$(hostname) start=$(date -u +%FT%TZ)"

write_env() { # write_env <part_dir>
    mkdir -p "$ART/logs/$1"
    {
        echo "COMMIT=$(cat "$ART/COMMIT" 2>/dev/null || echo unknown)"
        echo "HOST=$(hostname)"
        echo "DATE_UTC=$(date -u +%FT%TZ)"
        echo "BIN=$BIN"
        echo "BUILD_DIR=$BUILD_DIR"
        echo "THREADS=$THREADS"
        echo "PART=$PART"
        echo "GPU=$( (nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) || true)"
    } > "$ART/logs/$1/run.env"
    echo "  run.env: $(tr '\n' ' ' < "$ART/logs/$1/run.env")"
}

in_set() { # in_set <value> <allowed...>
    local v=$1
    shift
    local a
    for a in "$@"; do [ "$v" = "$a" ] && return 0; done
    return 1
}

# /usr/bin/time -v is the campaign instrument (host max RSS).  Where it does not exist (a local WSL smoke) the
# wall time is measured with date and the RSS is recorded as n/a, with a loud warning.
if [ -x /usr/bin/time ]; then
    timed() { /usr/bin/time -v -o "$@"; }
else
    echo "WARNING: /usr/bin/time not found: host max RSS will be n/a (acceptable for a local smoke only)"
    timed() { # timed <time_file> cmd...
        local tf=$1 t0 rc
        shift
        t0=$(date +%s)
        "$@"
        rc=$?
        printf '\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:%02d\n\tMaximum resident set size (kbytes): n/a\n' \
            $(( $(date +%s) - t0 )) > "$tf"
        return $rc
    }
fi

LAST_RC=0
ALLOWED=(0)
drive() { # drive <part_dir> <name> driver-args...   (adds --summary raw/<part_dir>/<name>.json)
    local pd=$1 name=$2
    shift 2
    local log=$ART/logs/$pd/$name.log tf=$ART/logs/$pd/$name.time sj=$ART/raw/$pd/$name.json
    mkdir -p "$ART/logs/$pd" "$ART/raw/$pd"
    echo "+ /usr/bin/time -v -o $tf $BIN $* --summary $sj > $log 2>&1"
    local t0
    t0=$(date +%s)
    timed "$tf" "$BIN" "$@" --summary "$sj" > "$log" 2>&1
    local rc=$?
    LAST_RC=$rc
    printf '%s\t%s\n' "$name" "$rc" >> "$ART/logs/$pd/exit_codes.tsv"
    echo "  CASE_RESULT log=$log rc=$rc $(grep '^STATUS ' "$log" | tail -1) wall=$(( $(date +%s) - t0 ))s" \
         "max_rss_kB=$(grep 'Maximum resident' "$tf" 2>/dev/null | awk '{print $NF}') end=$(date -u +%FT%TZ)"
    grep -E '^(CASE|BACKFLOW|GROWTH|RESIDUAL|TRANSPORT_STATUS|LABELDIFF|TIMING|MEMORY|STATUS)' "$log" | sed 's/^/    /'
    if ! in_set "$rc" ${ALLOWED[@]+"${ALLOWED[@]}"}; then
        echo "  UNEXPECTED_EXIT_CODE name=$name rc=$rc allowed=${ALLOWED[*]} (recorded, not retried)"
        BAD=$((BAD + 1))
    fi
}

digest() { # digest <script> args...   (prints the command, counts a failure)
    echo "+ $PY $SCRIPTS/$1 ${*:2}"
    "$PY" "$SCRIPTS/$1" "${@:2}"
    local rc=$?
    echo "  $1 rc=$rc"
    [ "$rc" -eq 0 ] || BAD=$((BAD + 1))
}

# ------------------------------------------------------------------------------------------
part_controls() {
    local ns=${1:-${NS:-"16 32 64"}}
    local pd=${TAG}controls
    local EXP=$ART/exports/${TAG}controls
    write_env "$pd"
    mkdir -p "$EXP" "$ART/logs/$pd"
    ALLOWED=(0 21) # 21 = roundtrip_exceeded is a recorded result of a control run (shown in the digest status column)
    # A4 + A1-A3 fast tests (ctest), timing recorded (spec: each <= 60 s)
    echo "+ /usr/bin/time -v -o $ART/logs/$pd/ctest.time ctest --test-dir $BUILD_DIR --output-on-failure -R 'inlet_slab_(controls|trace16)' > $ART/logs/$pd/ctest.log 2>&1"
    timed "$ART/logs/$pd/ctest.time" ctest --test-dir "$BUILD_DIR" --output-on-failure \
        -R 'inlet_slab_(controls|trace16)' > "$ART/logs/$pd/ctest.log" 2>&1
    local rc=$?
    printf '%s\t%s\n' ctest "$rc" >> "$ART/logs/$pd/exit_codes.tsv"
    echo "  CTEST rc=$rc"
    grep -E 'Test +#|tests passed|tests failed' "$ART/logs/$pd/ctest.log" | sed 's/^/    /'
    [ "$rc" -eq 0 ] || BAD=$((BAD + 1))
    local n maxn=0
    for n in $ns; do
        [ "$n" -gt "$maxn" ] && maxn=$n
        echo "+ $PY $SCRIPTS/make_cos_x2_cells.py $n $EXP/cos_x2_N$n.npy"
        "$PY" "$SCRIPTS/make_cos_x2_cells.py" "$n" "$EXP/cos_x2_N$n.npy" || BAD=$((BAD + 1))
        drive "$pd" "cosx2_N$n" --construct trace --cells "$EXP/cos_x2_N$n.npy" --n "$n" --eps 0.5 \
            --threads "$THREADS" --save-labels "$EXP/cosx2_N$n"
    done
    # k = 1 (Y = 0 cells) at the largest grid of NS (campaign: 64)
    echo "+ $PY $SCRIPTS/make_cos_x2_cells.py $maxn $EXP/zeros_N$maxn.npy --zeros"
    "$PY" "$SCRIPTS/make_cos_x2_cells.py" "$maxn" "$EXP/zeros_N$maxn.npy" --zeros || BAD=$((BAD + 1))
    drive "$pd" "k1_N$maxn" --construct trace --cells "$EXP/zeros_N$maxn.npy" --n "$maxn" --eps 0.5 \
        --threads "$THREADS" --save-labels "$EXP/k1_N$maxn"
    # control2d (psi2 = x3) at 32 (smoke: the largest grid of NS)
    local c2=32
    case " $ns " in *" 32 "*) ;; *) c2=$maxn ;; esac
    drive "$pd" "control2d_N$c2" --construct trace --analytic control2d --n "$c2" --eps 0.5 \
        --threads "$THREADS" --save-labels "$EXP/control2d_N$c2"
    digest controls_digest.py --raw "$ART/raw/$pd" --exports "$EXP" --ctest-log "$ART/logs/$pd/ctest.log" \
        --out "$ART/analysis/${APFX}controls.md"
}

part_xcons() {
    local pairs=${1:-"0.25:32 0.25:64 0.5:32"}
    local pd=${TAG}xcons
    write_env "$pd"
    ALLOWED=(0 21) # the --solve outcome is in SOLVE_STATUS / the digest; 21 = trace roundtrip_exceeded, recorded
    local p e n
    for p in $pairs; do
        e=${p%%:*}
        n=${p##*:}
        drive "$pd" "eps${e}_N$n" --construct trace --solve --n "$n" --eps "$e" "${FIELD[@]}" --threads "$THREADS"
    done
    digest xcons_digest.py "$ART"/raw/"$pd"/eps*_N*.json --sf33 "$SF33" --out "$ART/analysis/${APFX}xcons.md"
}

part_regress() {
    local pd=${TAG}regress
    write_env "$pd"
    ALLOWED=(0)
    drive "$pd" prod32 --production --n 32 --eps 0.25 --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder \
        --threads "$THREADS"
    digest regress_diff.py "$ART/logs/$pd/prod32.log" "$SF33/logs/ladder_0.25_c4/N32.log" \
        --out "$ART/analysis/${APFX}regress.md"
}

part_matrix() {
    local lab=${1:?matrix needs 05|10|15}
    local ns=${2:-${NS:-"32 64 128 256"}}
    local ladder_ns=${LADDER_NS:-"32 64"}
    local eps
    case "$lab" in
    05) eps=0.5 ;;
    10) eps=1 ;;
    15) eps=1.5 ;;
    *) echo "run_campaign.sh: matrix amplitude must be 05|10|15" >&2; exit 2 ;;
    esac
    local pd=${TAG}matrix_eps$lab
    local EXP=$ART/exports/${TAG}matrix_eps$lab
    write_env "$pd"
    ALLOWED=(0 21 19 17 18) # ok, roundtrip_exceeded, trace_fail, inlet_backflow, darcy_failed: all recorded
    local n extra
    for n in $ns; do
        extra=()
        case " $ladder_ns " in *" $n "*) extra=(--ladder) ;; esac
        drive "$pd" "N$n" --construct trace --n "$n" --eps "$eps" "${FIELD[@]}" --threads "$THREADS" \
            --save-labels "$EXP/N$n" ${extra[@]+"${extra[@]}"}
    done
    digest orders_digest.py --eps "$eps" "$ART"/raw/"$pd"/N*.json --out "$ART/analysis/${APFX}matrix_eps$lab.md"
    cost_digest_part
}

cost_digest_part() { # H: whichever N256 points exist so far (re-run after each amplitude)
    local pre=${TAG}
    local js=() ts=() f
    for f in "$ART"/raw/"${pre}"matrix_eps*/N256.json; do
        [ -e "$f" ] || continue
        js+=("$f")
        ts+=("${f/\/raw\//\/logs\/}")
    done
    ts=(${ts[@]+"${ts[@]/%.json/.time}"})
    if [ "${#js[@]}" -gt 0 ]; then
        digest cost_digest.py ${js[@]+"${js[@]}"} ${ts[@]+"${ts[@]}"} --out "$ART/analysis/${APFX}cost_256.md"
    else
        echo "  cost_digest: no N256 point yet (skipped)"
    fi
}

part_info() {
    local n=${INFO_N:-32}
    local pd=${TAG}info
    local EXP=$ART/exports/${TAG}info
    write_env "$pd"
    ALLOWED=(0 21 19 17 18)
    drive "$pd" "sigma2.25_eps1_N$n" --construct trace --n "$n" --eps 1 --sigma2 2.25 --ell 0.25 --seed 3001 \
        --threads "$THREADS" --save-labels "$EXP/sigma2.25_eps1"
    drive "$pd" "sigma1_eps1.5_N$n" --construct trace --n "$n" --eps 1.5 "${FIELD[@]}" \
        --threads "$THREADS" --save-labels "$EXP/sigma1_eps1.5"
    digest labels_diff.py "$EXP/sigma2.25_eps1" "$EXP/sigma1_eps1.5" --out "$ART/analysis/${APFX}info_sigma_eps.md"
}

part_gpu() {
    local ns=${1:-${NS:-"128 256"}}
    local pd=${TAG}gpu
    local EXP=$ART/exports/${TAG}gpu
    write_env "$pd"
    ALLOWED=(0 21 19 17 18 1) # exit 1 is acceptable only when the log says "not built" (checked below)
    local n cases=() not_built=0
    for n in $ns; do
        drive "$pd" "gpu_N$n" --construct trace --n "$n" --eps 0.5 "${FIELD[@]}" --threads "$THREADS" \
            --tracer gpu --save-labels "$EXP/gpu_N$n"
        if [ "$LAST_RC" -eq 1 ] && grep -qi 'not built' "$ART/logs/$pd/gpu_N$n.log"; then
            echo "  GPU_TRACER_NOT_BUILT (exit 1, 'not built'): recorded; the part exits 0"
            not_built=1
            break
        fi
        if [ "$LAST_RC" -eq 1 ]; then
            echo "  UNEXPECTED_EXIT_CODE name=gpu_N$n rc=1 without 'not built' (recorded, not retried)"
            BAD=$((BAD + 1))
        fi
        drive "$pd" "host_N$n" --construct trace --n "$n" --eps 0.5 "${FIELD[@]}" --threads "$THREADS" \
            --tracer host --save-labels "$EXP/host_N$n"
        cases+=(--case "$n" "$ART/raw/$pd/host_N$n.json" "$ART/raw/$pd/gpu_N$n.json" "$EXP/host_N$n" "$EXP/gpu_N$n")
    done
    if [ "$not_built" -eq 1 ]; then
        digest gpu_equiv.py --not-built --out "$ART/analysis/${APFX}gpu.md"
    else
        digest gpu_equiv.py ${cases[@]+"${cases[@]}"} --out "$ART/analysis/${APFX}gpu.md"
    fi
}

case "$PART" in
controls) part_controls "$@" ;;
xcons) part_xcons ;;
regress) part_regress ;;
matrix) part_matrix "$@" ;;
info) part_info ;;
gpu) part_gpu "$@" ;;
smoke)
    TAG="smoke/"
    APFX="smoke_"
    LADDER_NS=16
    INFO_N=16
    part_controls "16"
    part_xcons "0.25:16"
    part_matrix 05 "16"
    part_info
    part_gpu "16"
    ;;
*)
    echo "run_campaign.sh: unknown part '$PART'" >&2
    exit 2
    ;;
esac
echo "run_campaign.sh: part=$PART end=$(date -u +%FT%TZ) unexpected_failures=$BAD"
[ "$BAD" -eq 0 ]
