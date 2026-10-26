#!/usr/bin/env bash
# Regression byte-compare harness, producer side: run the core case set with the four binaries of <bin_dir> into <out_dir>.
#
#   scripts/regress/run.sh [--list] <bin_dir> <out_dir>
#
# <bin_dir> holds macroflow3d_pipeline, inlet_slab, lagrangian_ensemble, lagrangian_medium_ab.  Configs always come from THIS tree's
# apps/ (the tree this script lives in).  <out_dir> must not exist.  --list prints every command and runs nothing.  Compare two outputs
# with scripts/regress/compare.sh.  See scripts/regress/README.md.
#
# Provenance: case definitions (r_items, r_args, r_expected, g_args, constants RC_*, TRACE_THREADS, PROD_N, PROD_EXTRA, SEED0) and the
# per-case bookkeeping (exit_codes.tsv, walls.tsv) are adapted from
# docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/scripts/run_sf41.sh at commit 3452a7d (non-dry settings only).
#
# Environment: REGRESS_SOURCE_TREE (free-text source tree id recorded in env.txt), REGRESS_TIMEOUT (per-case seconds, default 3600),
# PYTHON (default python3).  bash 4.2 compatible (the V100 host): no empty-array expansion under `set -u` without the ${a[@]+...} idiom.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LIST=0
if [ "${1:-}" = "--list" ]; then
    LIST=1
    shift
fi
if [ $# -ne 2 ]; then
    echo "usage: run.sh [--list] <bin_dir> <out_dir>" >&2
    exit 2
fi
BIN_DIR=$(cd "$1" 2> /dev/null && pwd) || { echo "run.sh: bin_dir '$1' not found" >&2; exit 2; }
OUT=$2
case "$OUT" in /*) ;; *) OUT=$PWD/$OUT ;; esac
TMO=${REGRESS_TIMEOUT:-3600}
BINS="macroflow3d_pipeline inlet_slab lagrangian_ensemble lagrangian_medium_ab"

# constants ----------------------------------------------------------------------------------------------------------
SEED0=4001
RC_FWD=" 0 18 22 "
RC_POL=" 0 18 22 23 "
RC_TRACE=" 0 17 19 21 "
TRACE_THREADS=32
PROD_N=32
PROD_EXTRA=()
# heterogeneous/homogeneous pipeline configs (apps/<name>.yaml), one case each in its own empty cwd; edit to add or retire members
PIPE_CASES=(config_par2_small config_streamfunctions_homogeneous config_streamfunctions_continuation)

if [ "$LIST" = "0" ]; then
    for b in $BINS; do
        [ -x "$BIN_DIR/$b" ] || { echo "run.sh: missing executable $BIN_DIR/$b" >&2; exit 2; }
    done
    [ -e "$OUT" ] && { echo "run.sh: refusing existing out_dir $OUT" >&2; exit 2; }
    mkdir -p "$OUT" || exit 2
fi

BAD=0
T0=$(date +%s)

# case_run <id> <expected rc list> <cwd> -- cmd...   (cwd is created empty; stdout+stderr -> <out>/log/<id>.log)
case_run() {
    local id=$1 expected=$2 cwd=$3 rc t0 wall
    shift 4
    if [ "$LIST" = "1" ]; then
        echo "[$id] (cwd $cwd) $*"
        return 0
    fi
    mkdir -p "$cwd" "$OUT/log"
    echo "+ [$id] $*"
    t0=$(date +%s)
    (cd "$cwd" && timeout -k 60 "$TMO" "$@") > "$OUT/log/$id.log" 2>&1
    rc=$?
    wall=$(($(date +%s) - t0))
    printf '%s\t%s\n' "$id" "$rc" >> "$OUT/exit_codes.tsv"
    printf '%s\t%s\t%s\n' "$id" "$wall" "$(date -u +%FT%TZ)" >> "$OUT/walls.tsv"
    echo "  CASE_RESULT run=$id rc=$rc wall=${wall}s"
    case "$expected" in
    *" $rc "*) ;;
    *)
        echo "  UNEXPECTED_EXIT_CODE run=$id rc=$rc (expected$expected)"
        BAD=$((BAD + 1))
        ;;
    esac
    return 0
}

# ---- the 20-run R set (SF-40 r_items / r_args / r_expected = SF-37 RR13) ------------------------------------------
r_items() { # lines `kind name arg1 arg2 arg3`
    local N e t c
    for N in 32 64; do
        for e in 0.5 1.0; do
            for t in host gpu; do
                echo "trace N${N}_e${e}_${t} $N $e $t"
            done
        done
    done
    echo "trace N64_e2.0_gpu 64 2 gpu"
    echo "prod prod32 - - -"
    for c in smoke_g3d_n16 g3d_n32_cov gauss_n64_e1_cov gauss_n64_e2 gauss_n64_e2_tol gauss_n40_e1 gauss_n80_e2; do
        echo "fwd $c - - -"
    done
    for c in gauss_n64_e1_cov gauss_n64_e2 gauss_n64_e2_tol; do
        echo "neutral $c - - -"
    done
}

R_ARGS=()
# r_args <kind> <name> <a1> <a2> <a3> <base>: R_ARGS = the SF-37 RR13 command (without the binary)
r_args() {
    local kind=$1 name=$2 base=$6
    local G=(--sigma2 1 --ell 0.25 --seed 3001)
    case "$kind" in
    trace) R_ARGS=(--construct trace --sigma2 1 --ell 0.25 --seed 3001 --threads "$TRACE_THREADS" --n "$3" --eps "$4" --tracer "$5"
        --save-labels "$base" --summary "$base.json") ;;
    prod) R_ARGS=(--production --n "$PROD_N" --eps 0.25 --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder ${PROD_EXTRA[@]+"${PROD_EXTRA[@]}"}) ;;
    fwd | neutral)
        case "$name" in
        smoke_g3d_n16) R_ARGS=(--track forward --analytic generic3d --n 16 --eps 0.25 --np-side 16 --planes "0.25,0.5,1" --times "0.5,1" --covectors) ;;
        g3d_n32_cov) R_ARGS=(--track forward --analytic generic3d --n 32 --eps 0.25 --np-side 32 --planes "0.5,1" --times 1 --covectors) ;;
        gauss_n64_e1_cov) R_ARGS=(--track forward "${G[@]}" --n 64 --eps 1 --np-side 64 --planes "0.25,0.5,1,2" --times "0.5,1,2" --covectors) ;;
        gauss_n64_e2) R_ARGS=(--track forward "${G[@]}" --n 64 --eps 2 --np-side 64 --planes "0.25,0.5,1,2,4" --times "0.5,1,2,4") ;;
        gauss_n64_e2_tol) R_ARGS=(--track forward "${G[@]}" --n 64 --eps 2 --np-side 64 --planes "0.5,1,2" --times "1,2" --tol 1e-10 --hmax-div 32 --max-arclength 1000) ;;
        gauss_n40_e1) R_ARGS=(--track forward "${G[@]}" --n 40 --eps 1 --np-side 32 --planes "0.5,1" --times 1) ;;
        gauss_n80_e2) R_ARGS=(--track forward "${G[@]}" --n 80 --eps 2 --np-side 32 --planes "0.5,1" --times 1) ;;
        *) return 1 ;;
        esac
        if [ "$kind" = "neutral" ]; then
            R_ARGS+=(--comparator pollock --save-vd "$base/vd" --save-seed-velocity)
        fi
        R_ARGS+=(--save-particles "$base" --summary "$base.json")
        ;;
    *) return 1 ;;
    esac
}

r_expected() { # r_expected <kind> -> expected rc list
    case "$1" in
    trace) echo "$RC_TRACE" ;;
    prod) echo " 0 " ;;
    fwd) echo "$RC_FWD" ;;
    *) echo "$RC_POL" ;;
    esac
}

# ---- G2 / G3 (SF-41 RR9 (ii), (iii)) and runner_b (SF-41 medium b, small) -----------------------------------------
# shellcheck disable=SC2054
G2_ARGS=(--grid 512,64,64 --eps 2 --ell 0.125 --seed0 "$SEED0" --nr 2 --window 0.1,0.9,0.1,0.9 --np-side 100 --planes auto --times auto
    --tol 1e-8 --hmax-div 8 --fused-splines --single-launch --twin on --pollock-bridge on)
# shellcheck disable=SC2054
G3_ARGS=(--grid 512,64,64 --ell 0.125 --eps 2 --seed0 "$SEED0" --nr 2)
# medium (b): same option set as the SF-41 E/F runs (run_sf41.sh e_cmd), 512x64x64 box (L1 = 8, 8 cells per ell), 2 realizations
# shellcheck disable=SC2054
RB_ARGS=(--medium b --grid 512,64,64 --eps 1 --ell 0.125 --seed0 "$SEED0" --nr 2 --window 0.1,0.9,0.1,0.9 --np-side 30 --planes auto --times auto
    --tol 1e-8 --hmax-div 8 --fused-splines --single-launch --twin on --pollock-bridge on)

# ---- run ----------------------------------------------------------------------------------------------------------
if [ "$LIST" = "0" ]; then
    {
        echo "DATE_UTC=$(date -u +%FT%TZ)"
        echo "HOST=$(hostname)"
        echo "REGRESS_SOURCE_TREE=${REGRESS_SOURCE_TREE:-unset}"
        echo "REPO=$REPO"
        echo "BIN_DIR=$BIN_DIR"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        echo "GPU: $(nvidia-smi -L 2> /dev/null | tr '\n' ';')"
        echo "NVCC: $(nvcc --version 2> /dev/null | tail -n 1)"
        echo "GCC: $(gcc --version 2> /dev/null | head -n 1)"
        echo "KERNEL: $(uname -sr)"
    } > "$OUT/env.txt"
    # shellcheck disable=SC2086
    (cd "$BIN_DIR" && md5sum $BINS) > "$OUT/binaries.md5"
fi

for cfg in "${PIPE_CASES[@]}"; do
    case_run "pipe_$cfg" " 0 " "$OUT/pipe/$cfg" -- "$BIN_DIR/macroflow3d_pipeline" "$REPO/apps/$cfg.yaml"
done

while read -r kind name a1 a2 a3; do
    [ "$LIST" = "0" ] && mkdir -p "$OUT/r/$kind"
    if [ "$kind" = "prod" ]; then
        r_args prod "$name" "$a1" "$a2" "$a3" "$OUT/r/prod32" || { echo "no argument set for $kind $name"; BAD=$((BAD + 1)); continue; }
        case_run r_prod "$(r_expected prod)" "$OUT/work/r_prod" -- "$BIN_DIR/inlet_slab" "${R_ARGS[@]}"
    else
        r_args "$kind" "$name" "$a1" "$a2" "$a3" "$OUT/r/$kind/$name" || { echo "no argument set for $kind $name"; BAD=$((BAD + 1)); continue; }
        case_run "r_${kind}_$name" "$(r_expected "$kind")" "$OUT/work/r_${kind}_$name" -- "$BIN_DIR/inlet_slab" "${R_ARGS[@]}"
    fi
done < <(r_items)

case_run g_runner_a " 0 " "$OUT/work/g_runner_a" -- "$BIN_DIR/lagrangian_ensemble" "${G2_ARGS[@]}" --out "$OUT/g_runner_a"
case_run g_ab " 0 " "$OUT/work/g_ab" -- "$BIN_DIR/lagrangian_medium_ab" "${G3_ARGS[@]}" --out "$OUT/g_ab"
case_run runner_b " 0 " "$OUT/work/runner_b" -- "$BIN_DIR/lagrangian_ensemble" "${RB_ARGS[@]}" --out "$OUT/runner_b"

# analysis-only AnalysisRunner on the runner_b tree: the runner writes analysis_config.yaml (run_mode analysis_only, output_dir ".") and
# documents `cd DIR && macroflow3d_pipeline analysis_config.yaml`; run it on a COPY so runner_b stays exactly what the runner wrote.
if [ "$LIST" = "1" ]; then
    echo "[runner_b_analysis] cp -r runner_b runner_b_analysis; (cwd runner_b_analysis) macroflow3d_pipeline analysis_config.yaml"
else
    cp -r "$OUT/runner_b" "$OUT/runner_b_analysis" 2> /dev/null || mkdir -p "$OUT/runner_b_analysis"
    case_run runner_b_analysis " 0 " "$OUT/runner_b_analysis" -- "$BIN_DIR/macroflow3d_pipeline" analysis_config.yaml
fi

if [ "$LIST" = "0" ]; then
    echo "REGRESS_DONE unexpected_rc=$BAD wall_s=$(($(date +%s) - T0))" | tee "$OUT/REGRESS_DONE"
fi
[ "$BAD" -eq 0 ]
