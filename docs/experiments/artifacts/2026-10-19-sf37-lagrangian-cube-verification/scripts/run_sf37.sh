#!/usr/bin/env bash
# SF-37 campaign runner (one part per detached V100 job).  Run from the repository root (the synced mirror):
#
#   bash <A>/scripts/run_sf37.sh <bin_dir> labels-eps10   N6a: gridded labels eps = 1 (V3a criterion C input), seeds 3101..3103, N 64/128/256
#   bash <A>/scripts/run_sf37.sh <bin_dir> labels-o3      N6a: gridded labels eps = 1.5 and 2 (blind O3 input), N 64/128/256, then the six 512^3
#   (parts appended by later nodes: v2, v3a, v1, v3b, bytecmp, digest; each is a function part_<name>, '-' -> '_')
#
# <A> = docs/experiments/artifacts/2026-10-19-sf37-lagrangian-cube-verification.
# <bin_dir> = a directory containing the executable `inlet_slab` (label parts: the BASE-commit binary, e.g.
#   ~/sf37_base_refs/bin; later parts: build/v100-release).  Resolved to an absolute path.
#
# Environment (all optional):
#   SF37_OUT_ROOT   output root on the host, NOT in the repository tree           (default $HOME/sf37_campaign)
#   SF37_BASE_REFS  base-commit references (job sf37-base-refs)                   (default $HOME/sf37_base_refs)
#   SF37_COMMIT     commit recorded in run_<part>.env                             (default: <A>/COMMIT, else "unknown")
#   SF37_DRYRUN     1 = local smoke, NOT campaign evidence (see below)            (default 0)
#   SF37_EPS_FILTER labels-o3 only: 15 or 20 restricts the part to eps 1.5 / 2    (default all)
#   PYTHON          interpreter for lines_cmp.py                                  (default python3)
#
# Layout of the label parts (contract C-LAYOUT), E in {10, 15, 20} for eps {1, 1.5, 2}, S the seed, N the grid:
#   $OUT_ROOT/labels/exports/eps{E}_s{S}/N{N}/     --save-labels directory
#   $OUT_ROOT/labels/raw/eps{E}_s{S}/N{N}.json     driver --summary JSON
#   $OUT_ROOT/labels/logs/eps{E}_s{S}/N{N}.log     stdout + stderr
#   $OUT_ROOT/labels/logs/eps{E}_s{S}/N{N}.time    /usr/bin/time -v file (the SF-35 orders_digest.py finds it from the
#                                                  JSON path by replacing raw/ -> logs/ and .json -> .time)
#   $OUT_ROOT/labels/exit_codes_<part>.tsv         case, rc, wall_s, end_utc (case = eps{E}_s{S}/N{N})
#   $OUT_ROOT/labels/run_<part>.env                COMMIT, binary path + md5sum, host, date, part, dryrun, GPU
#   $OUT_ROOT/labels/sha256_<part>.txt             sha256 of every export/summary/log/time file of each finished case
#   $OUT_ROOT/labels/resumed_<part>.txt            cases re-run after an interruption (RR12), if any
#   $OUT_ROOT/labels/preflight_<part>/             preflight reference case (outputs + result.txt)
# Every finished case prints `CASE_RESULT case= rc= wall= end=`.
#
# Rules (spec RR11/RR12): the label cases use ONLY the SF-35 defaults (h/8, tol 1e-8, max_roundtrip 1e-8; none is passed);
# exit codes 0, 21 (roundtrip_exceeded, labels saved), 19 (trace_fail) and 17 (inlet_backflow, no export) are EXPECTED
# outcomes (recorded; 17/19 make the point "not evaluable" downstream); any other code is a failure: recorded, the part
# continues with the next case, the final exit status is non-zero.  A failing case is NEVER retried with other settings.
# Resume: a case whose rc is recorded in the part's exit_codes file is finished and is skipped (SKIP_DONE; the summary
# JSON is expected, its absence is reported as SKIP_DONE_NOJSON and is not a re-run because a finished scientific
# outcome is never re-run); a case with partial outputs and NO recorded rc (interrupted) is re-run with identical
# settings after removing its partial outputs (RESUMED, listed in resumed_<part>.txt).
# No digest, metric or analysis is computed or printed by the label parts (the O3 metric is applied blind later); only
# the STATUS line of each case is echoed.
#
# Preflight (both label parts, before any campaign case): the base reference case
#   --construct trace --tracer gpu --threads 32 --sigma2 1 --ell 0.25 --seed 3001 --n 32 --eps 0.5
# is run with the same binary and compared with $SF37_BASE_REFS/trace/N32_e0.5_gpu{/psi1.npy,/psi2.npy,.log,.time}
# (cmp; lines CASE PLANE BACKFLOW GROWTH RESIDUAL TRANSPORT_STATUS STATUS via lines_cmp.py with the timing masks; exit
# code).  Any difference -> PREFLIGHT_DIFF and exit 3 before any campaign case.  Reference absent: PREFLIGHT_SKIP in a
# dry run, PREFLIGHT_DIFF + exit 3 otherwise.  Preflight is always repeated on a re-invocation.
#
# SF37_DRYRUN=1 (local smoke only; NOT campaign evidence): seeds become 3001 only; labels-eps10 grids {16, 32};
# labels-o3 phase 1 grids {16, 32}, phase 2 grid {32} (instead of 512; its cases coincide with phase 1 and therefore print
# SKIP_DONE); campaign cases run with --threads 8 (the preflight keeps --threads 32 to be comparable with the
# reference).  Unset, the switch changes nothing.
#
# bash 4.2 (the V100 host): no empty-array expansion under `set -u` (use ${a[@]+"${a[@]}"}); no backslashes in
# `${var//pat/rep}` replacements (none are used).  `scripts/remote run` exports CUDA_VISIBLE_DEVICES=<physical gpu>.
# Exit status: 0 unless a blocking item failed (unexpected exit code, preflight 3, usage 2).
#
# STRUCTURE.  Dispatch is deferred to an EXIT trap that fires when the shell reaches the end of this file, so that
# part functions APPENDED below this line by later nodes are already defined when the dispatch runs: append new
# `part_<name>` functions at the END of the file; nothing above needs editing.
set -uo pipefail

SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=$(dirname "$SCRIPTS")
ARG_BIN_DIR=${1:-}
ARG_PART=${2:-}
PY=${PYTHON:-python3}
OUT_ROOT=${SF37_OUT_ROOT:-$HOME/sf37_campaign}
BASE_REFS=${SF37_BASE_REFS:-$HOME/sf37_base_refs}
DRY=${SF37_DRYRUN:-0}
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
BAD=0
BIN_DIR=""
BIN=""
CHILD=""

# ============================================================================================================
# common helpers (generic; used by every part)
# ============================================================================================================
usage() {
    echo "usage: run_sf37.sh <bin_dir> <part>   (parts: $(declare -F | awk '{print $3}' | grep '^part_' | sed 's/^part_//' | tr '_' '-' | tr '\n' ' '))" >&2
}

# /usr/bin/time -v is the instrument for host max RSS; where it does not exist (a local WSL dry run) the wall time is
# measured with date and the RSS is recorded as n/a (acceptable for a local dry run only; refused in a campaign).
timed() { # timed <time_file> cmd...
    if [ -x /usr/bin/time ]; then
        /usr/bin/time -v -o "$@"
        return $?
    fi
    local tf=$1 t0 rc
    shift
    t0=$(date +%s)
    "$@"
    rc=$?
    printf '\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:%02d\n\tMaximum resident set size (kbytes): n/a\n\tExit status: %d\n' \
        $(( $(date +%s) - t0 )) "$rc" > "$tf"
    return $rc
}

kill_tree() { # kill_tree <pid>: TERM the process and all its descendants
    local c
    for c in $(pgrep -P "$1" 2> /dev/null); do
        kill_tree "$c"
    done
    kill -TERM "$1" 2> /dev/null
}

on_signal() {
    echo "run_sf37.sh: interrupted (signal); stopping the running case (no rc is recorded; a re-invocation resumes it)"
    [ -n "$CHILD" ] && kill_tree "$CHILD"
    exit 143
}

tsv_rc() { # tsv_rc <tsv> <case> -> the recorded rc of <case> (last record), empty if none
    [ -e "$1" ] || return 0
    awk -F'\t' -v c="$2" '$1 == c { r = $2 } END { if (r != "") print r }' "$1"
}

write_env() { # write_env <env_file> <part>   (appended as a new block if the file exists: a resumed invocation)
    local f=$1 part=$2 commit
    commit=${SF37_COMMIT:-$(cat "$ART/COMMIT" 2> /dev/null || echo unknown)}
    [ -e "$f" ] && echo "# --- resumed/repeated invocation ---" >> "$f"
    {
        [ "$DRY" = "1" ] && echo "# DRYRUN=1: local smoke, NOT campaign evidence"
        echo "COMMIT=$commit"
        echo "HOST=$(hostname)"
        echo "DATE_UTC=$(date -u +%FT%TZ)"
        echo "BIN=$BIN"
        echo "BIN_MD5=$(md5sum "$BIN" | awk '{print $1}')"
        echo "PART=$part"
        echo "DRYRUN=$DRY"
        echo "OUT_ROOT=$OUT_ROOT"
        echo "BASE_REFS=$BASE_REFS"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        echo "GPU=$( (nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | head -1) || true)"
    } >> "$f"
    echo "  env: $(grep -v '^#' "$f" | tail -n 12 | tr '\n' ' ')"
}

append_sha() { # append_sha <sha_file> <root> <path>...  (existing files/dirs only; paths recorded relative to <root>)
    local sf=$1 root=$2 p
    shift 2
    for p in "$@"; do
        [ -e "$p" ] || continue
        (cd "$root" && find "${p#"$root"/}" -type f | LC_ALL=C sort | xargs -r sha256sum) >> "$sf"
    done
}

# ============================================================================================================
# label parts (frozen after N6a): labels-eps10, labels-o3
# ============================================================================================================
EXPECTED_LABEL_RC=" 0 17 19 21 "

# labels_preflight <part>: base reference case N32 eps 0.5 gpu vs $BASE_REFS/trace (PREFLIGHT_OK | PREFLIGHT_SKIP | exit 3)
labels_preflight() {
    local part=$1 name=N32_e0.5_gpu
    local ref=$BASE_REFS/trace PF=$OUT_ROOT/labels/preflight_$part
    rm -rf "$PF"
    mkdir -p "$PF"
    if [ ! -d "$ref/$name" ] || [ ! -e "$ref/$name.log" ]; then
        if [ "$DRY" = "1" ]; then
            echo "PREFLIGHT_SKIP reference $ref/$name{,.log} absent (dry run)"
            echo "PREFLIGHT_SKIP reference absent (dry run)" > "$PF/result.txt"
            return 0
        fi
        echo "PREFLIGHT_DIFF reference $ref/$name{,.log} absent: no campaign case may run without the criterion-R preflight"
        echo "PREFLIGHT_DIFF reference absent" > "$PF/result.txt"
        exit 3
    fi
    local args=(--construct trace --tracer gpu --threads 32 --sigma2 1 --ell 0.25 --seed 3001 --n 32 --eps 0.5)
    echo "+ preflight: /usr/bin/time -v -o $PF/$name.time $BIN ${args[*]} --save-labels $PF/$name --summary $PF/$name.json > $PF/$name.log 2>&1"
    CHILD=""
    timed "$PF/$name.time" "$BIN" "${args[@]}" --save-labels "$PF/$name" --summary "$PF/$name.json" > "$PF/$name.log" 2>&1
    local rc=$?
    local ok=1 f npy="IDENTICAL" lines lrc brc
    for f in psi1.npy psi2.npy; do
        cmp -s "$PF/$name/$f" "$ref/$name/$f" || npy="DIFF($f)"
    done
    [ "$npy" = "IDENTICAL" ] || ok=0
    lines=$("$PY" "$SCRIPTS/lines_cmp.py" "$PF/$name.log" "$ref/$name.log" CASE PLANE BACKFLOW GROWTH RESIDUAL TRANSPORT_STATUS STATUS)
    lrc=$?
    [ "$lrc" -eq 0 ] || ok=0
    brc=$(grep 'Exit status' "$ref/$name.time" 2> /dev/null | awk '{print $NF}')
    [ "$rc" -eq 0 ] || ok=0
    [ -z "$brc" ] || [ "$brc" = "$rc" ] || ok=0
    {
        echo "preflight $name npy=$npy lines=$([ "$lrc" -eq 0 ] && echo IDENTICAL || echo DIFF) rc=$rc ref_rc=${brc:-n/a(expected 0)}"
        echo "$lines"
    } > "$PF/result.txt"
    sed 's/^/    /' "$PF/result.txt"
    if [ "$ok" -eq 1 ]; then
        echo "PREFLIGHT_OK $name"
        echo "PREFLIGHT_OK" >> "$PF/result.txt"
    else
        echo "PREFLIGHT_DIFF $name (no campaign case was run)"
        echo "PREFLIGHT_DIFF" >> "$PF/result.txt"
        exit 3
    fi
}

# label_case <part> <elab> <eps> <seed> <N>
label_case() {
    local part=$1 elab=$2 eps=$3 seed=$4 n=$5
    local tag=eps${elab}_s${seed} case_id
    case_id=$tag/N$n
    local L=$OUT_ROOT/labels
    local exp=$L/exports/$tag/N$n json=$L/raw/$tag/N$n.json log=$L/logs/$tag/N$n.log tf=$L/logs/$tag/N$n.time
    local tsv=$L/exit_codes_$part.tsv sha=$L/sha256_$part.txt
    local rc0
    rc0=$(tsv_rc "$tsv" "$case_id")
    if [ -n "$rc0" ]; then
        if [ -e "$json" ]; then
            echo "SKIP_DONE case=$case_id rc=$rc0"
        else
            echo "SKIP_DONE_NOJSON case=$case_id rc=$rc0 (rc recorded, summary JSON absent: finished outcome, not re-run)"
            case "$rc0" in 0 | 19 | 21) BAD=$((BAD + 1)) ;; esac
        fi
        case "$EXPECTED_LABEL_RC" in *" $rc0 "*) ;; *) BAD=$((BAD + 1)) ;; esac
        return 0
    fi
    if [ -e "$exp" ] || [ -e "$json" ] || [ -e "$log" ] || [ -e "$tf" ]; then
        echo "RESUMED case=$case_id (partial outputs and no recorded rc: removed, re-run with identical settings)"
        printf '%s\t%s\t%s\n' "$case_id" "$(date -u +%FT%TZ)" "partial outputs removed" >> "$L/resumed_$part.txt"
        rm -rf "$exp" "$json" "$log" "$tf"
    fi
    mkdir -p "$L/exports/$tag" "$L/raw/$tag" "$L/logs/$tag"
    local args=(--construct trace --tracer gpu --threads "$THREADS" --sigma2 1 --ell 0.25 --seed "$seed" --n "$n" --eps "$eps")
    echo "+ /usr/bin/time -v -o $tf $BIN ${args[*]} --save-labels $exp --summary $json > $log 2>&1"
    local t0 rc
    t0=$(date +%s)
    timed "$tf" "$BIN" "${args[@]}" --save-labels "$exp" --summary "$json" > "$log" 2>&1 &
    CHILD=$!
    wait "$CHILD"
    rc=$?
    CHILD=""
    local wall=$(( $(date +%s) - t0 )) end
    end=$(date -u +%FT%TZ)
    printf '%s\t%s\t%s\t%s\n' "$case_id" "$rc" "$wall" "$end" >> "$tsv"
    append_sha "$sha" "$L" "$exp" "$json" "$log" "$tf"
    echo "  CASE_RESULT case=$case_id rc=$rc wall=${wall}s end=$end"
    grep -E '^STATUS' "$log" | cut -c1-200 | sed 's/^/    /'
    case "$EXPECTED_LABEL_RC" in
    *" $rc "*) ;;
    *)
        echo "  UNEXPECTED_EXIT_CODE case=$case_id rc=$rc (recorded, not retried)"
        BAD=$((BAD + 1))
        ;;
    esac
}

# labels_run_part <part> <elab:eps:seed:N>...
labels_run_part() {
    local part=$1 tok elab eps seed n
    shift
    local L=$OUT_ROOT/labels
    mkdir -p "$L"
    [ -e "$L/exit_codes_$part.tsv" ] || printf 'case\trc\twall_s\tend_utc\n' > "$L/exit_codes_$part.tsv"
    write_env "$L/run_$part.env" "$part"
    labels_preflight "$part"
    for tok in "$@"; do
        IFS=: read -r elab eps seed n <<< "$tok"
        label_case "$part" "$elab" "$eps" "$seed" "$n"
    done
}

part_labels_eps10() {
    local seeds="3101 3102 3103" ns="64 128 256" s n
    if [ "$DRY" = "1" ]; then
        seeds="3001"
        ns="16 32"
    fi
    local cases=()
    for s in $seeds; do
        for n in $ns; do
            cases+=("10:1:$s:$n")
        done
    done
    labels_run_part labels-eps10 ${cases[@]+"${cases[@]}"}
}

part_labels_o3() {
    local filter=${SF37_EPS_FILTER:-all}
    case "$filter" in all | 15 | 20) ;; *)
        echo "run_sf37.sh: SF37_EPS_FILTER must be unset, all, 15 or 20 (got '$filter')" >&2
        exit 2
        ;;
    esac
    local seeds="3101 3102 3103" ns1="64 128 256" ns2="512" ed elab eps s n
    if [ "$DRY" = "1" ]; then
        seeds="3001"
        ns1="16 32"
        ns2="32"
    fi
    local cases=()
    for ed in 15:1.5 20:2; do # phase 1
        elab=${ed%%:*}
        eps=${ed#*:}
        [ "$filter" = "all" ] || [ "$filter" = "$elab" ] || continue
        for s in $seeds; do
            for n in $ns1; do
                cases+=("$elab:$eps:$s:$n")
            done
        done
    done
    for ed in 15:1.5 20:2; do # phase 2
        elab=${ed%%:*}
        eps=${ed#*:}
        [ "$filter" = "all" ] || [ "$filter" = "$elab" ] || continue
        for s in $seeds; do
            for n in $ns2; do
                cases+=("$elab:$eps:$s:$n")
            done
        done
    done
    echo "labels-o3: eps filter=$filter, ${#cases[@]} cases"
    labels_run_part labels-o3 ${cases[@]+"${cases[@]}"}
}

# ============================================================================================================
# dispatch (deferred to the EXIT trap: see STRUCTURE in the header).  Part functions appended below this point
# (by later nodes) are visible to it.
# ============================================================================================================
sf37_dispatch() {
    trap - EXIT
    trap on_signal INT TERM
    if [ -z "$ARG_BIN_DIR" ] || [ -z "$ARG_PART" ]; then
        usage
        exit 2
    fi
    local fn=part_${ARG_PART//-/_}
    if ! declare -F "$fn" > /dev/null; then
        echo "run_sf37.sh: unknown part '$ARG_PART'" >&2
        usage
        exit 2
    fi
    BIN_DIR=$(cd "$ARG_BIN_DIR" 2> /dev/null && pwd) || {
        echo "run_sf37.sh: bin_dir '$ARG_BIN_DIR' not found" >&2
        exit 2
    }
    BIN=$BIN_DIR/inlet_slab
    [ -x "$BIN" ] || { echo "run_sf37.sh: $BIN not found or not executable" >&2; exit 2; }
    THREADS=32
    if [ "$DRY" = "1" ]; then
        THREADS=8
        echo "run_sf37.sh: SF37_DRYRUN=1 (local smoke: seed 3001 only, small grids, --threads 8; NOT campaign evidence)"
    fi
    if [ ! -x /usr/bin/time ]; then
        if [ "$DRY" = "1" ]; then
            echo "WARNING: /usr/bin/time not found: host max RSS will be n/a (acceptable for a local dry run only)"
        else
            echo "run_sf37.sh: /usr/bin/time not found (host max RSS is required in a campaign)" >&2
            exit 2
        fi
    fi
    echo "run_sf37.sh: part=$ARG_PART bin=$BIN host=$(hostname) gpu_id=$GPU_ID out_root=$OUT_ROOT start=$(date -u +%FT%TZ)"
    "$fn"
    echo "run_sf37.sh: part=$ARG_PART end=$(date -u +%FT%TZ) blocking_failures=$BAD"
    [ "$BAD" -eq 0 ]
    exit $?
}
trap sf37_dispatch EXIT

# ============================================================================================================
# APPEND new part_<name> functions BELOW this line (later nodes: v2, v3a, v1, v3b, bytecmp, digest).
# ============================================================================================================
