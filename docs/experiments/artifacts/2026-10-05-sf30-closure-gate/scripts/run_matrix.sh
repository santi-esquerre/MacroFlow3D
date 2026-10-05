#!/usr/bin/env bash
# SF-30 streamline-closure gate: matrix launcher.
#
# Run from the repository root on the execution host:
#
#   bash docs/experiments/artifacts/2026-10-05-sf30-closure-gate/scripts/run_matrix.sh <group> [--list] [--force]
#        [--bin build/v100-release] [--out output_sf30]
#
#   groups: controls | matched | matrix128 | ladder | manyperiod | all
#
# Each run is `<bin>/closure_gate <options> --out <out>/<group>/<run-id>`.
#   --list   print one line per run (`<group> <run-id> <full command>`) and execute nothing.
#   --force  re-run runs whose summary.json already exists (default: skip them, i.e. resume).
#
# A failing run never aborts the launcher. Every executed run appends
# `<run-id>\t<exit code>\t<wall seconds>` to `<out>/<group>/status.tsv`. The launcher exits 0
# iff every run of the requested group(s) exited 0 (a skipped run counts with the last exit
# code recorded for it in status.tsv, if any), otherwise 1. Usage errors exit 2.
#
# No GPU or thread option is passed: the job environment selects the device and the
# executable's default thread count applies.
#
# The run list is the pre-registered SF-30 matrix (103 runs); do not edit it.

set -u

usage() {
    sed -n '4,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
SEEDS16="${SCRIPT_DIR}/probe_seeds16.csv"

GROUP=""
LIST=0
FORCE=0
BIN="build/v100-release"
OUT="output_sf30"

while [ $# -gt 0 ]; do
    case "$1" in
        --list) LIST=1 ;;
        --force) FORCE=1 ;;
        --bin)
            [ $# -ge 2 ] || usage
            BIN="$2"
            shift
            ;;
        --out)
            [ $# -ge 2 ] || usage
            OUT="$2"
            shift
            ;;
        controls | matched | matrix128 | ladder | manyperiod | all)
            [ -z "$GROUP" ] || usage
            GROUP="$1"
            ;;
        *) usage ;;
    esac
    shift
done
[ -n "$GROUP" ] || usage

# ---------------------------------------------------------------------------
# Run list
# ---------------------------------------------------------------------------

COMMON_GAUSS="--tols 1e-6,1e-8,1e-10,1e-12 --working-tol 1e-8"

# The six cases: "<sigma2 label> <sigma2> <ell label> <ell>".
CASES=(
    "s025 0.25 l8 0.125"
    "s025 0.25 l16 0.0625"
    "s1 1 l8 0.125"
    "s1 1 l16 0.0625"
    "s4 4 l8 0.125"
    "s4 4 l16 0.0625"
)

RUN_IDS=()
RUN_ARGS=()

add_run() { # <run-id> <options without --out>
    RUN_IDS+=("$1")
    RUN_ARGS+=("$2")
}

build_controls() {
    local n spec field eps
    for n in 64 128 256; do
        for spec in "lester2021 1" "lester_brk 1" "control2d 1" "two_mode 0.5" "generic3d 0.5"; do
            read -r field eps <<<"$spec"
            add_run "a_${field}_n${n}_p16" \
                "--field ${field} --n ${n} --eps ${eps} --pcg-rtol 1e-12 --seeds-file ${SEEDS16} --tols 1e-6,1e-8,1e-10,1e-12 --working-tol 1e-10"
            add_run "a_${field}_n${n}" \
                "--field ${field} --n ${n} --eps ${eps} --pcg-rtol 1e-12 --tols 1e-6,1e-8,1e-10,1e-12 --working-tol 1e-8"
        done
    done
}

gauss_run() { # <prefix> <field> <case spec> <seed> <n> [<periods>]
    local prefix="$1" field="$2" case_spec="$3" seed="$4" n="$5" periods="${6:-}"
    local sl s l ll id opts
    read -r sl s ll l <<<"$case_spec"
    id="${prefix}_${sl}_${ll}_r${seed}_n${n}"
    opts="--field ${field} --n ${n} --sigma2 ${s} --ell ${l} --seed ${seed}"
    if [ -n "$periods" ]; then
        id="${id}_p${periods}"
        opts="${opts} --periods ${periods}"
    fi
    add_run "$id" "${opts} ${COMMON_GAUSS}"
}

CASE_S4_L16="s4 4 l16 0.0625"

build_matched() {
    local c n seed
    for c in "${CASES[@]}"; do
        for n in 64 128 256; do
            gauss_run m gaussian2d "$c" 3001 "$n"
        done
    done
    for seed in 3002 3003; do
        for n in 128 256; do
            gauss_run m gaussian2d "$CASE_S4_L16" "$seed" "$n"
        done
    done
}

build_matrix128() {
    local c seed
    for c in "${CASES[@]}"; do
        for seed in 3001 3002 3003 3004 3005; do
            gauss_run g gaussian "$c" "$seed" 128
        done
    done
}

build_ladder() {
    local c n seed
    for c in "${CASES[@]}"; do
        for n in 64 256; do
            gauss_run g gaussian "$c" 3001 "$n"
        done
    done
    for seed in 3002 3003; do
        gauss_run g gaussian "$CASE_S4_L16" "$seed" 256
    done
}

build_manyperiod() {
    local c
    for c in "${CASES[@]}"; do
        gauss_run p gaussian "$c" 3001 128 64
    done
    gauss_run p gaussian "$CASE_S4_L16" 3001 256 16
}

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

last_status() { # <status.tsv> <run-id>: last recorded exit code, empty if none
    [ -f "$1" ] || return 0
    awk -F'\t' -v id="$2" '$1 == id { c = $2 } END { if (c != "") print c }' "$1"
}

run_group() { # <group>; returns 0 iff every run of the group exited 0
    local g="$1" i id dir cmd rc t0 t1 secs prev failed=0
    RUN_IDS=()
    RUN_ARGS=()
    "build_${g}"
    for i in "${!RUN_IDS[@]}"; do
        id="${RUN_IDS[$i]}"
        dir="${OUT}/${g}/${id}"
        cmd="${BIN}/closure_gate ${RUN_ARGS[$i]} --out ${dir}"
        if [ "$LIST" -eq 1 ]; then
            echo "${g} ${id} ${cmd}"
            continue
        fi
        mkdir -p "${OUT}/${g}"
        if [ "$FORCE" -eq 0 ] && [ -f "${dir}/summary.json" ]; then
            prev="$(last_status "${OUT}/${g}/status.tsv" "$id")"
            echo "== [${g}] ${id}: summary.json exists, skipped (last recorded exit code: ${prev:-none})"
            if [ -n "$prev" ] && [ "$prev" != "0" ]; then failed=1; fi
            continue
        fi
        echo "== [${g}] ${id}: ${cmd}"
        t0="$(date +%s.%N)"
        # shellcheck disable=SC2086  # options are whitespace-separated words by construction
        ${BIN}/closure_gate ${RUN_ARGS[$i]} --out "${dir}"
        rc=$?
        t1="$(date +%s.%N)"
        secs="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')"
        printf '%s\t%s\t%s\n' "$id" "$rc" "$secs" >>"${OUT}/${g}/status.tsv"
        echo "== [${g}] ${id}: exit ${rc}, ${secs} s"
        if [ "$rc" -ne 0 ]; then failed=1; fi
    done
    return "$failed"
}

if [ "$GROUP" = "all" ]; then
    GROUPS_TO_RUN=(controls matched matrix128 ladder manyperiod)
else
    GROUPS_TO_RUN=("$GROUP")
fi

if [ "$LIST" -eq 0 ] && [ ! -f "$SEEDS16" ]; then
    echo "run_matrix.sh: seeds file not found: ${SEEDS16}" >&2
    exit 2
fi

overall=0
for g in "${GROUPS_TO_RUN[@]}"; do
    run_group "$g" || overall=1
done
exit "$overall"
