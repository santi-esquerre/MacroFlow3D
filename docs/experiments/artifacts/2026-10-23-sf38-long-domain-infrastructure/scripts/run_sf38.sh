#!/usr/bin/env bash
# SF-38 campaign runner (one part per detached V100 job).  Run from the synced mirror (the repository root is derived from
# this script's path); every part is restartable.  Modelled on the SF-40 runner (run_sf40.sh): same helpers (timed,
# kill_tree, RR2 samplers, case_run, bc_* byte-compare helpers), same resume rules.
#
#   bash <A>/scripts/run_sf38.sh <build_dir> baserefs    BASE binaries (inlet_slab, macroflow3d_pipeline): RR9 references (R set, SF-27 trio)
#   bash <A>/scripts/run_sf38.sh <build_dir> controls    HEAD: A (long_box_tests --only A, --full), dirichlet_medium_tests, fast ctests, runner smoke
#   bash <A>/scripts/run_sf38.sh <build_dir> prodgrid    HEAD: B1, B2, B3 production grids with the runner (RR2 sampler) -> B_VERDICT
#   bash <A>/scripts/run_sf38.sh <build_dir> bpilot      HEAD: medium (b) solver pilot (RR4, RR2 sampler) -> C0_VERDICT
#   bash <A>/scripts/run_sf38.sh <build_dir> ab          HEAD: A/B, 10 realizations at sigma^2 = 1 and 4, ab_decision.py -> C_VERDICT selected=a|b
#   bash <A>/scripts/run_sf38.sh <build_dir> ensemble    HEAD: D-A uninterrupted, D-B SIGKILL + restart, D-C one realization, checks -> D_VERDICT
#   bash <A>/scripts/run_sf38.sh <build_dir> fieldonly   HEAD: G, field-only runs at the three production geometries -> G_* lines
#   bash <A>/scripts/run_sf38.sh <build_dir> par2pilot   HEAD: E, Par2 baseline pilot (three configs, dt/2 copies, field stats, K stats) -> E_VERDICT
#   bash <A>/scripts/run_sf38.sh <build_dir> final       HEAD: full ctest, SF-27 trio and the 20-run R set vs the base references -> F_CTEST F_TRIO F_RSET
#
# <A> = docs/experiments/artifacts/2026-10-23-sf38-long-domain-infrastructure.  <build_dir> holds the executables of the
# part (HEAD: build/v100-release; baserefs: the directory of the BASE binaries, $HOME/sf38_base_refs/bin).  `ctest` of
# `final` runs on <build_dir>.  `par2-pilot` is accepted as an alias of `par2pilot`.
#
# Environment (all optional):
#   SF38_OUT_ROOT   output root, NOT in the repository tree           (default $HOME/sf38_campaign; dry run $HOME/sf38_dryrun)
#   SF38_BASE_REFS  base-commit references (part baserefs)            (default $HOME/sf38_base_refs; dry run $OUT_ROOT/base_refs)
#   SF38_COMMIT     commit recorded in run.env                        (default <A>/COMMIT, else "unknown")
#   SF38_DRYRUN     1 = tiny settings, NOT campaign evidence          (default 0)
#   SF38_DIAG_PIPE  diagnostics-build pipeline for the K-stats check  (default $REPO/build/v100-diag/macroflow3d_pipeline, built
#                   by the part when absent; dry run: <build_dir>/macroflow3d_pipeline)
#   SF38_CTEST_ARGS extra arguments of the `final` ctest              (default empty; dry run: -R ^lagrangian_dirichlet_medium$)
#   SF38_AB_EXTRA_ARGS extra arguments appended to the `ab` runs      (default empty; the fallback tolerance of C0, if used)
#   SF38_AB_SKIP_C0_GUARD 1 = `ab` does not require a C0 PASS/PASS_FALLBACK in $OUT_ROOT/bpilot/results.txt (default 0)
#   PYTHON          interpreter                                       (default python3)
#
# Per-part bookkeeping in $OUT_ROOT/<part>/ (baserefs: in $SF38_BASE_REFS itself):
#   run.env         COMMIT, binaries + md5, host, CUDA_VISIBLE_DEVICES, nvidia-smi banner, nvcc and gcc versions
#   exit_codes.tsv  EXACTLY two tab-separated columns `run<TAB>rc`, no header    walls.tsv  `run<TAB>wall_s<TAB>end_utc`
#   results.txt     the verdict / comparison / RECORDED lines of the part (recomputed on every invocation from the finished runs)
#   sha256.txt      sha256 of every output file of each finished run (+ derived files)     resumed.txt  runs re-run after an interruption
#   PART_EXIT       0 iff the part ended with no blocking item (absent when the invocation was interrupted by a signal);
#                   results.txt ends with `RECORDED part_seconds=<this invocation> part_runs_seconds=<sum of the run walls>`
# Run artifacts are <base>/ (output dir), <base>.log (stdout+stderr), <base>.time (/usr/bin/time -v), <base>.free (free -b
# before), and for sampled runs <base>.apps.csv, <base>.dev.csv, <base>.pid_tree, <base>.precheck (RR2).
#
# Resume rules (SF-37 RR12): a run whose rc is recorded in exit_codes.tsv is finished and is never re-run (nothing is re-run
# because of its result); a run with partial outputs and NO recorded rc (infrastructure interruption) has them removed and is
# re-run with identical settings (listed in resumed.txt).  A SIGINT/SIGTERM records no rc.  Comparisons are recomputed.
# An exit code is a recorded outcome, not a retry trigger; unexpected codes make the part's exit status non-zero.
#
# SF38_DRYRUN=1 (tiny settings; NOT evidence; separate roots; the exclusive-GPU precheck is skipped because a workstation
# is shared): production grids replaced by 64x16x16 boxes, the ensemble part by 64x16x16 with the same kill/restart logic,
# field-only by 64x16x16, Par2 by a 64x16x16 / 2400-step / 1000-particle copy of each config, the R set reduced to 7 runs
# (4 threads, `--production --n 16`; the host-tracer and N >= 32 oracle runs exceed the 60 s local limit), the full
# ctest replaced by one fast case, `long_box_tests --only A` and `--full` by the fast tiers.  Unset, nothing changes.
#
# bash 4.2 (the V100 host): no empty-array expansion under `set -u` (use ${a[@]+"${a[@]}"}), no ${var@Q}, no negative
# subscripts, no backslashes in `${var//pat/rep}` replacements.  `scripts/remote run` exports CUDA_VISIBLE_DEVICES.
set -uo pipefail

SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=$(dirname "$SCRIPTS")
REPO=$(cd "$ART/../../../.." && pwd)
ARG_BIN_DIR=${1:-}
ARG_PART=${2:-}
PY=${PYTHON:-python3}
DRY=${SF38_DRYRUN:-0}
if [ "$DRY" = "1" ]; then
    OUT_ROOT=${SF38_OUT_ROOT:-$HOME/sf38_dryrun}
    BASE_REFS=${SF38_BASE_REFS:-$OUT_ROOT/base_refs}
else
    OUT_ROOT=${SF38_OUT_ROOT:-$HOME/sf38_campaign}
    BASE_REFS=${SF38_BASE_REFS:-$HOME/sf38_base_refs}
fi
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
GPU_ID=${GPU_ID%%,*}
BAD=0
BIN_DIR=""
CHILD=""
SAMP_APPS_PID=""
SAMP_DEV_PID=""
PD=""
RES=""
BC_N=0
PART=""
PART_T0=0

# constants ----------------------------------------------------------------------------------------------------------
SEED0=4001
# manifest.json keys dropped in the trio comparison (SF-40 RR7): the run time and the build identity (differs between a base
# and a head build by construction).
MANIFEST_DROPS=(timestamp build.git_hash)
BC_DROPS=('**.timings_s')
TRIO="config_pspta_small config_streamfunctions_homogeneous config_streamfunctions_continuation"
RC_FWD=" 0 18 22 "
RC_POL=" 0 18 22 23 "
RC_TRACE=" 0 17 19 21 "
PAR2_CFGS="s1 s225 s4"
PAR2_TMO=14400
LONG_TMO=43200
TRACE_THREADS=32
PROD_N=32
PROD_EXTRA=()

# ============================================================================================================
# common helpers (SF-40 run_sf40.sh)
# ============================================================================================================
usage() {
    echo "usage: run_sf38.sh <build_dir> <part>   (parts: baserefs controls prodgrid bpilot ab ensemble fieldonly par2pilot final)" >&2
}

# /usr/bin/time -v is the instrument for host max RSS; absent only in a local dry run (RSS n/a, refused in a campaign).
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

samplers_kill() {
    [ -n "$SAMP_APPS_PID" ] && kill "$SAMP_APPS_PID" 2> /dev/null
    [ -n "$SAMP_DEV_PID" ] && kill "$SAMP_DEV_PID" 2> /dev/null
    SAMP_APPS_PID=""
    SAMP_DEV_PID=""
    return 0
}

on_signal() {
    echo "run_sf38.sh: interrupted (signal); stopping the running case (no rc is recorded; a re-invocation resumes it)"
    [ -n "$CHILD" ] && kill_tree "$CHILD"
    samplers_kill
    exit 143
}

tsv_rc() { # tsv_rc <tsv> <run> -> the recorded rc of <run> (last record), empty if none
    [ -e "$1" ] || return 0
    awk -F'\t' -v c="$2" '$1 == c { r = $2 } END { if (r != "") print r }' "$1"
}

tsv_col() { # tsv_col <tsv> <run> <col> -> that column of the last record of <run>
    [ -e "$1" ] || return 0
    awk -F'\t' -v c="$2" -v k="$3" '$1 == c { r = $k } END { if (r != "") print r }' "$1"
}

safe_rm() { # safe_rm <path>: rm -rf, refusing anything that is not a deep absolute path
    case "$1" in
    /?*/?*/?*) rm -rf "$1" ;;
    *)
        echo "run_sf38.sh: refusing to remove '$1'" >&2
        exit 2
        ;;
    esac
}

has_flag() { # has_flag <flags> <letter>
    case "$1" in *"$2"*) return 0 ;; esac
    return 1
}

bin() { # bin <name> -> $BIN_DIR/<name>
    echo "$BIN_DIR/$1"
}

write_env() { # write_env <pd> <part>   (appended as a new block when the file exists: a resumed invocation)
    local pd=$1 part=$2 f commit b
    f=$pd/run.env
    commit=${SF38_COMMIT:-$(cat "$ART/COMMIT" 2> /dev/null || echo unknown)}
    [ -e "$f" ] && echo "# --- resumed/repeated invocation ---" >> "$f"
    {
        [ "$DRY" = "1" ] && echo "# DRYRUN=1: tiny settings, NOT campaign evidence"
        echo "COMMIT=$commit"
        echo "HOST=$(hostname)"
        echo "DATE_UTC=$(date -u +%FT%TZ)"
        echo "PART=$part"
        echo "BIN_DIR=$BIN_DIR"
        for b in lagrangian_ensemble lagrangian_medium_ab macroflow3d_pipeline inlet_slab par2_field_stats long_box_tests \
            dirichlet_medium_tests lagrangian_ensemble_tests; do
            [ -x "$BIN_DIR/$b" ] && echo "BIN_MD5_$b=$(md5sum "$BIN_DIR/$b" | awk '{print $1}')"
        done
        echo "DRYRUN=$DRY"
        echo "OUT_ROOT=$OUT_ROOT"
        echo "BASE_REFS=$BASE_REFS"
        echo "REPO=$REPO"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        echo "GPU_NAME=$( (nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_ID" 2> /dev/null | head -1) || true)"
        echo "NVIDIA_DRIVER=$( (nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$GPU_ID" 2> /dev/null | head -1) || true)"
        echo "CUDA_VERSION_NVIDIA_SMI=$( (nvidia-smi 2> /dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1) || true)"
        echo "NVCC=$( (nvcc --version 2> /dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1) || true)"
        echo "GCC=$( (gcc --version 2> /dev/null | head -1) || true)"
        nvidia-smi 2> /dev/null | head -n 12 | sed 's/^/# nvidia-smi banner: /'
    } >> "$f"
    echo "  env: $(grep -v '^#' "$f" | tail -n 16 | tr '\n' ' ' | cut -c1-900)"
}

append_sha() { # append_sha <pd> <base>: sha256 of <base>/** and <base>.* (paths relative to <pd>)
    local pd=$1 base=$2 rel f
    rel=${base#"$pd"/}
    (
        cd "$pd" || exit 0
        {
            [ -d "$rel" ] && find "$rel" -type f
            for f in "$rel".*; do
                [ -f "$f" ] && echo "$f"
            done
        } | LC_ALL=C sort | xargs -r sha256sum
    ) >> "$pd/sha256.txt"
}

sha_file() { # sha_file <pd> <file...>: sha256 of derived files (paths relative to <pd>)
    local pd=$1 f
    shift
    for f in "$@"; do
        [ -f "$pd/$f" ] && (cd "$pd" && sha256sum "$f") >> "$pd/sha256.txt"
    done
    return 0
}

find_desc() { # find_desc <pid> <comm> -> first descendant of <pid> whose comm is <comm> (15 chars)
    local c
    for c in $(pgrep -P "$1" 2> /dev/null); do
        if [ "$(ps -o comm= -p "$c" 2> /dev/null)" = "$2" ]; then
            echo "$c"
            return 0
        fi
        find_desc "$c" "$2" && return 0
    done
    return 1
}

samplers_start() { # samplers_start <base>   (RR2: all processes + device-wide memory.used of the assigned GPU, 200 ms)
    SAMP_APPS_PID=""
    SAMP_DEV_PID=""
    if ! command -v nvidia-smi > /dev/null 2>&1; then
        echo "WARNING: nvidia-smi not found: no samples (empty $1.apps.csv / $1.dev.csv)"
        : > "$1.apps.csv"
        : > "$1.dev.csv"
        return 0
    fi
    nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 200 > "$1.apps.csv" 2> /dev/null &
    SAMP_APPS_PID=$!
    nvidia-smi --query-gpu=timestamp,memory.used -i "$GPU_ID" --format=csv,noheader,nounits -lms 200 > "$1.dev.csv" 2> /dev/null &
    SAMP_DEV_PID=$!
    sleep 1
}

samplers_stop() {
    sleep 1
    if [ -n "$SAMP_APPS_PID" ]; then
        kill "$SAMP_APPS_PID" 2> /dev/null
        wait "$SAMP_APPS_PID" 2> /dev/null
    fi
    if [ -n "$SAMP_DEV_PID" ]; then
        kill "$SAMP_DEV_PID" 2> /dev/null
        wait "$SAMP_DEV_PID" 2> /dev/null
    fi
    SAMP_APPS_PID=""
    SAMP_DEV_PID=""
    return 0
}

run_in() { # run_in <cwd|-> <time_file> <timeout_s|0> cmd...   (runs in a background subshell)
    local cwd=$1 tf=$2 tmo=$3
    shift 3
    if [ "$cwd" != "-" ]; then
        cd "$cwd" || return 97
    fi
    if [ "$tmo" -gt 0 ]; then
        timed "$tf" timeout -k 60 "$tmo" "$@"
    else
        timed "$tf" "$@"
    fi
}

rc_expected_check() { # rc_expected_check <run> <rc> <expected list | any>
    [ "$3" = "any" ] && return 0
    case "$3" in
    *" $2 "*) ;;
    *)
        echo "  UNEXPECTED_EXIT_CODE run=$1 rc=$2 (expected$3; recorded, not retried)"
        BAD=$((BAD + 1))
        ;;
    esac
    return 0
}

precheck_gpu() { # precheck_gpu <id> <base>: 0 = no other compute process on the GPU (RR2); dry run: always 0
    if [ "$DRY" = "1" ] || ! command -v nvidia-smi > /dev/null 2>&1; then
        return 0
    fi
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$GPU_ID" > "$2.precheck" 2>&1
    if [ "$(grep -Ec '^[0-9]+,' "$2.precheck")" -gt 0 ]; then
        echo "GPU_BUSY run=$1: other compute process(es) hold GPU $GPU_ID (RR2); not run, no rc recorded:"
        sed 's/^/    /' "$2.precheck"
        BAD=$((BAD + 1))
        return 1
    fi
    echo "precheck: no compute process on GPU $GPU_ID" >> "$2.precheck"
    return 0
}

CASE_RC=""
CASE_STATE=""
CASE_WALL=""
# case_run <pd> <id> <base> <expected rc list | any> <flags: S sampler, X exclusive GPU> <cwd | -> <timeout_s | 0> -- cmd...
# Sets CASE_RC / CASE_STATE (done | ran | busy) / CASE_WALL.  Run artifacts live at <base> and <base>.*.
case_run() {
    local pd=$1 id=$2 base=$3 expected=$4 flags=$5 cwd=$6 tmo=$7
    shift 8
    CASE_RC=""
    CASE_STATE=""
    CASE_WALL=""
    local tsv=$pd/exit_codes.tsv rc0 f
    mkdir -p "$pd" "$(dirname "$base")"
    rc0=$(tsv_rc "$tsv" "$id")
    if [ -n "$rc0" ]; then
        echo "SKIP_DONE run=$id rc=$rc0"
        CASE_RC=$rc0
        CASE_STATE="done"
        CASE_WALL=$(tsv_col "$pd/walls.tsv" "$id" 2)
        rc_expected_check "$id" "$rc0" "$expected"
        return 0
    fi
    if [ -e "$base" ] || compgen -G "$base.*" > /dev/null; then
        echo "RESUMED run=$id (partial outputs and no recorded rc: removed, re-run with identical settings)"
        printf '%s\t%s\t%s\n' "$id" "$(date -u +%FT%TZ)" "partial outputs removed" >> "$pd/resumed.txt"
        safe_rm "$base"
        for f in "$base".*; do
            [ -e "$f" ] && safe_rm "$f"
        done
    fi
    if has_flag "$flags" X; then
        if ! precheck_gpu "$id" "$base"; then
            CASE_STATE=busy
            return 3
        fi
    fi
    command -v free > /dev/null 2>&1 && free -b > "$base.free" 2>&1
    if [ "$cwd" != "-" ]; then
        safe_rm "$cwd"
        mkdir -p "$cwd"
    fi
    echo "+ [$id] ${cwd/#-/.}: $* > $base.log 2>&1   (timeout=${tmo}s flags=$flags)"
    has_flag "$flags" S && samplers_start "$base"
    local t0 rc pidt="" exe
    exe=$(basename "$1")
    t0=$(date +%s)
    run_in "$cwd" "$base.time" "$tmo" "$@" > "$base.log" 2>&1 &
    CHILD=$!
    if has_flag "$flags" S; then
        local k
        for k in $(seq 1 40); do
            pidt=$(find_desc "$CHILD" "${exe:0:15}") && [ -n "$pidt" ] && break
            kill -0 "$CHILD" 2> /dev/null || break
            sleep 0.25
        done
        echo "${pidt:-none}" > "$base.pid_tree"
    fi
    wait "$CHILD"
    rc=$?
    CHILD=""
    has_flag "$flags" S && samplers_stop
    CASE_WALL=$(( $(date +%s) - t0 ))
    local end
    end=$(date -u +%FT%TZ)
    printf '%s\t%s\n' "$id" "$rc" >> "$tsv"
    printf '%s\t%s\t%s\n' "$id" "$CASE_WALL" "$end" >> "$pd/walls.tsv"
    append_sha "$pd" "$base"
    CASE_RC=$rc
    CASE_STATE=ran
    echo "  CASE_RESULT run=$id rc=$rc wall=${CASE_WALL}s end=$end"
    grep -E '^(R=[0-9]+ )?(STATUS|RUN_DONE)' "$base.log" 2> /dev/null | cut -c1-200 | sed 's/^/    /'
    rc_expected_check "$id" "$rc" "$expected"
    return 0
}

part_begin() { # part_begin <part>: sets PD/RES, bookkeeping
    PART=$1
    if [ "$PART" = "baserefs" ]; then
        PD=$BASE_REFS
    else
        PD=$OUT_ROOT/$PART
    fi
    mkdir -p "$PD"
    rm -f "$PD/PART_EXIT"
    RES=$PD/results.txt
    : > "$RES"
    [ "$DRY" = "1" ] && echo "# DRYRUN=1: tiny settings, NOT campaign evidence" >> "$RES"
    BC_N=0
    PART_T0=$(date +%s)
    write_env "$PD" "$PART"
}

part_finish() { # part_finish: RECORDED part_seconds (decision D16) + PART_EXIT
    local secs runs
    secs=$(( $(date +%s) - PART_T0 ))
    runs=$(awk -F'\t' '{ s += $2 } END { printf "%d", s }' "$PD/walls.tsv" 2> /dev/null)
    res "RECORDED part_seconds=$secs part_runs_seconds=${runs:-0} part=$PART (D16: this invocation / sum of the recorded run walls)"
    if [ "$BAD" -eq 0 ]; then
        echo 0 > "$PD/PART_EXIT"
    else
        echo 1 > "$PD/PART_EXIT"
    fi
}

# ============================================================================================================
# comparison helpers (SF-37 pattern; RES = results file of the part)
# ============================================================================================================
res() { # res <line...>: a result line (verdict / recorded), echoed and appended to results.txt
    echo "$*"
    echo "$*" >> "$RES"
}

res_block() { # res_block: every stdin line through res
    local ln
    while IFS= read -r ln; do
        res "$ln"
    done
}

bc_res() { # bc_res <IDENTICAL|DIFF|MISSING|SKIP> <item...>
    res "$@"
    case "$1" in IDENTICAL | SKIP) ;; *) BC_N=$((BC_N + 1)) ;; esac
}

bc_cmp() { # bc_cmp <item> <head_file> <ref_file>
    if [ ! -e "$2" ] || [ ! -e "$3" ]; then
        bc_res MISSING "$1 (head: $2 ref: $3)"
    elif cmp -s "$2" "$3"; then
        bc_res IDENTICAL "$1"
    else
        bc_res DIFF "$1"
    fi
}

bc_json() { # bc_json <item> <head_json> <ref_json> <drop spec>...
    local item=$1 a=$2 b=$3 out rc d
    shift 3
    if [ ! -e "$a" ] || [ ! -e "$b" ]; then
        bc_res MISSING "$item (head: $a ref: $b)"
        return 0
    fi
    local dargs=()
    for d in "$@"; do
        dargs+=(--drop "$d")
    done
    out=$("$PY" "$SCRIPTS/json_strip_cmp.py" "$a" "$b" ${dargs[@]+"${dargs[@]}"})
    rc=$?
    if [ "$rc" -eq 0 ]; then
        bc_res IDENTICAL "$item"
    elif [ "$rc" -eq 1 ]; then
        bc_res DIFF "$item: $out"
    else
        bc_res MISSING "$item: $out"
    fi
}

bc_lines() { # bc_lines <item> <head_log> <ref_log> <token>...   (lines_cmp.py: timing masks only)
    local item=$1 a=$2 b=$3 out rc
    shift 3
    if [ ! -e "$a" ] || [ ! -e "$b" ]; then
        bc_res MISSING "$item (head: $a ref: $b)"
        return 0
    fi
    out=$("$PY" "$SCRIPTS/lines_cmp.py" "$a" "$b" "$@")
    rc=$?
    if [ "$rc" -eq 0 ]; then
        bc_res IDENTICAL "$item: $out"
    elif [ "$rc" -eq 1 ]; then
        bc_res DIFF "$item: $(echo "$out" | head -n 1)"
        echo "$out" | sed 's/^/    /' | head -n 8
    else
        bc_res MISSING "$item: $out"
    fi
}

bc_rc_of() { # bc_rc_of <time_file> -> recorded exit status or empty
    grep 'Exit status' "$1" 2> /dev/null | tail -n 1 | awk '{print $NF}'
}

bc_rc_check() { # bc_rc_check <item> <head_rc> <ref_time_file>
    local brc
    brc=$(bc_rc_of "$3")
    if [ -z "$brc" ]; then
        bc_res MISSING "$1:rc (no Exit status in $3; head rc=$2)"
    elif [ "$brc" = "$2" ]; then
        bc_res IDENTICAL "$1:rc base=$brc head=$2"
    else
        bc_res DIFF "$1:rc base=$brc head=$2"
    fi
}

# bc_dir_files <item> <head_dir> <ref_dir> <strict 0|1>: every file of the reference directory must exist in the head
# directory and be cmp-identical; *.json by json_strip_cmp.py dropping BC_DROPS only; strict = no extra head file.
bc_dir_files() {
    local item=$1 hd=$2 rd=$3 strict=$4 f
    if [ ! -d "$rd" ] || [ ! -d "$hd" ]; then
        bc_res MISSING "$item (head dir: $hd ref dir: $rd)"
        return 0
    fi
    while IFS= read -r f; do
        case "$f" in
        *.json) bc_json "$item:$f" "$hd/$f" "$rd/$f" "${BC_DROPS[@]}" ;;
        *) bc_cmp "$item:$f" "$hd/$f" "$rd/$f" ;;
        esac
    done < <(cd "$rd" && find . -type f | sed 's#^\./##' | LC_ALL=C sort)
    if [ "$strict" = "1" ]; then
        local extra
        extra=$(comm -13 <(cd "$rd" && find . -type f | LC_ALL=C sort) <(cd "$hd" && find . -type f | LC_ALL=C sort) | tr '\n' ' ')
        if [ -z "$extra" ]; then
            bc_res IDENTICAL "$item:fileset"
        else
            bc_res DIFF "$item:fileset (extra head files: $extra)"
        fi
    fi
}

# bc_tree <item> <head_dir> <ref_dir>: SF-40 RR7 rule: every file cmp-identical, no extra file, except manifest.json compared
# by json_strip_cmp.py dropping the explicit MANIFEST_DROPS key list (no line-regex masking).
bc_tree() {
    local item=$1 hd=$2 rd=$3 f
    if [ ! -d "$rd" ] || [ ! -d "$hd" ]; then
        bc_res MISSING "$item (head dir: $hd ref dir: $rd)"
        return 0
    fi
    while IFS= read -r f; do
        case "$f" in
        manifest.json | */manifest.json) bc_json "$item:$f(drop ${MANIFEST_DROPS[*]})" "$hd/$f" "$rd/$f" "${MANIFEST_DROPS[@]}" ;;
        *) bc_cmp "$item:$f" "$hd/$f" "$rd/$f" ;;
        esac
    done < <(cd "$rd" && find . -type f | sed 's#^\./##' | LC_ALL=C sort)
    local extra
    extra=$(comm -13 <(cd "$rd" && find . -type f | LC_ALL=C sort) <(cd "$hd" && find . -type f | LC_ALL=C sort) | tr '\n' ' ')
    if [ -z "$extra" ]; then
        bc_res IDENTICAL "$item:fileset"
    else
        bc_res DIFF "$item:fileset (extra head files: $extra)"
    fi
}

# ============================================================================================================
# the 20-run R set (SF-40 r_items / r_args / r_expected = SF-37 RR13; RR9 of SF-38)
# ============================================================================================================
r_items() { # lines `kind name arg1 arg2 arg3`
    local N e t c
    if [ "$DRY" = "1" ]; then
        for e in 0.5 1.0; do
            echo "trace N32_e${e}_gpu 32 $e gpu"
        done
        echo "prod prod32 - - -"
        for c in smoke_g3d_n16 g3d_n32_cov gauss_n40_e1; do
            echo "fwd $c - - -"
        done
        echo "neutral gauss_n64_e1_cov - - -"
        return 0
    fi
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
# r_args <kind> <name> <a1> <a2> <a3> <base>: R_ARGS = the SF-37 RR13 base-reference command (without the binary)
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
    prod) [ "$DRY" = "1" ] && echo " 0 19 " || echo " 0 " ;; # dry: --n 16 may end in oracle_roundtrip_fail (19)
    fwd) echo "$RC_FWD" ;;
    *) echo "$RC_POL" ;;
    esac
}

# ============================================================================================================
# part baserefs (BASE binaries): RR9 references
# ============================================================================================================
part_baserefs() {
    part_begin baserefs
    mkdir -p "$BASE_REFS/bin"
    local b
    for b in inlet_slab macroflow3d_pipeline; do
        if [ "$BIN_DIR" != "$BASE_REFS/bin" ]; then
            cp -f "$BIN_DIR/$b" "$BASE_REFS/bin/"
        fi
    done
    echo "  base binaries in $BASE_REFS/bin: $(md5sum "$BASE_REFS/bin/inlet_slab" "$BASE_REFS/bin/macroflow3d_pipeline" | tr '\n' ' ')"
    res "RECORDED base_binaries_md5 $(md5sum "$BASE_REFS/bin/inlet_slab" "$BASE_REFS/bin/macroflow3d_pipeline" | awk '{printf "%s ", $1}')"
    local kind name a1 a2 a3 n=0
    echo "== R set (RR9: SF-35/36/37 default path) -> $BASE_REFS/r"
    while read -r kind name a1 a2 a3; do
        r_args "$kind" "$name" "$a1" "$a2" "$a3" "$BASE_REFS/r/$kind/$name" || { echo "no argument set for $kind $name"; BAD=$((BAD + 1)); continue; }
        n=$((n + 1))
        if [ "$kind" = "prod" ]; then
            case_run "$BASE_REFS" "r_$kind" "$BASE_REFS/r/prod32" "$(r_expected prod)" - - 0 -- "$BASE_REFS/bin/inlet_slab" "${R_ARGS[@]}"
        else
            case_run "$BASE_REFS" "r_${kind}_$name" "$BASE_REFS/r/$kind/$name" "$(r_expected "$kind")" - - 0 -- "$BASE_REFS/bin/inlet_slab" "${R_ARGS[@]}"
        fi
    done < <(r_items)
    echo "== SF-27 trio -> $BASE_REFS/trio/<cfg> (each from its own empty working directory)"
    local cfg
    for cfg in $TRIO; do
        case_run "$BASE_REFS" "trio_$cfg" "$BASE_REFS/trio/$cfg" " 0 " - "$BASE_REFS/trio/$cfg" 0 -- "$BASE_REFS/bin/macroflow3d_pipeline" "$REPO/apps/$cfg.yaml"
    done
    local nrec
    nrec=$(awk -F'\t' '{ n++ } END { print n + 0 }' "$PD/exit_codes.tsv")
    res "BASEREFS_RECORDED r_runs=$n trio_runs=3 recorded_in_exit_codes=$nrec unexpected_exit_codes=$BAD"
}

# ============================================================================================================
# part controls (HEAD)
# ============================================================================================================
ctl_exe_verdict() { # ctl_exe_verdict <id>: CHECK lines + ALL_PASS + exit code of a test executable
    local id=$1 log=$PD/$1.log rc np nf allp v
    rc=$(tsv_rc "$PD/exit_codes.tsv" "$id")
    np=$(grep -c '^CHECK .* PASS' "$log" 2> /dev/null || true)
    nf=$(grep -c '^CHECK .* FAIL' "$log" 2> /dev/null || true)
    np=${np:-0}
    nf=${nf:-0}
    grep -q '^ALL_PASS' "$log" 2> /dev/null && allp=1 || allp=0
    v=FAIL
    [ "$rc" = "0" ] && [ "$allp" = "1" ] && [ "$nf" = "0" ] && v=PASS
    res "CONTROL_VERDICT run=$id exit=$rc checks_pass=$np checks_fail=$nf all_pass=$allp $v"
    [ "$v" = "PASS" ] || BAD=$((BAD + 1))
}

a_verdict() { # a_verdict <id>: threshold A (RR10) from the A_VERDICT line printed by long_box_tests --only A
    local id=$1 log=$PD/$1.log rc line allp np nf v
    rc=$(tsv_rc "$PD/exit_codes.tsv" "$id")
    line=$(grep '^A_VERDICT ' "$log" 2> /dev/null | tail -n 1)
    allp=$(echo "$line" | sed -n 's/.*all_pass=\([0-9]*\).*/\1/p')
    np=$(echo "$line" | sed -n 's/.*checks_pass=\([0-9]*\).*/\1/p')
    nf=$(echo "$line" | sed -n 's/.*checks_fail=\([0-9]*\).*/\1/p')
    v=FAIL
    [ "$rc" = "0" ] && [ "$allp" = "1" ] && [ "${nf:-1}" = "0" ] && v=PASS
    res "A_VERDICT run=$id exit=$rc all_pass=${allp:--} checks_pass=${np:--} checks_fail=${nf:--} reported=[${line:-none}] $v"
    [ "$v" = "PASS" ] || BAD=$((BAD + 1))
}

smoke_verdict() { # smoke_verdict <id> <nr>: the runner smoke finished every realization
    local id=$1 nr=$2 rc nst nd v
    rc=$(tsv_rc "$PD/exit_codes.tsv" "$id")
    nst=$(grep -c '^R=[0-9]* STATUS ok' "$PD/$id.log" 2> /dev/null || true)
    nd=$(find "$PD/$id/realizations" -name DONE 2> /dev/null | wc -l | tr -d ' ')
    v=FAIL
    [ "$rc" = "0" ] && [ "${nst:-0}" = "$nr" ] && [ "$nd" = "$nr" ] && grep -q '^RUN_DONE' "$PD/$id.log" 2> /dev/null &&
        [ -f "$PD/$id/analysis_config.yaml" ] && [ -f "$PD/$id/analysis/macrodispersion.csv" ] && v=PASS
    res "SMOKE_VERDICT run=$id exit=$rc status_ok=${nst:-0}/$nr done_files=$nd/$nr $v"
    [ "$v" = "PASS" ] || BAD=$((BAD + 1))
}

sampler_verdict() { # sampler_verdict <id>: the RR2 sampler finds the PID (gates >= 5 samples, PID <= device)
    local out rc
    out=$("$PY" "$SCRIPTS/mem_verdict.py" peak "$PD" "$1")
    rc=$?
    res "SAMPLER_SMOKE run=$1 $(echo "$out" | head -n 1) $( [ "$rc" -eq 0 ] && echo OK || echo NOT_EVALUABLE)"
    [ "$rc" -eq 0 ] || BAD=$((BAD + 1))
}

part_controls() {
    part_begin controls
    local ENS PIPE full="--full" onlyA="--only A"
    ENS=$(bin lagrangian_ensemble)
    PIPE=$(bin macroflow3d_pipeline)
    if [ "$DRY" = "1" ]; then
        full=""
        onlyA="--only F1"
        res "# DRY: long_box_tests without --full, '--only A' replaced by '--only F1' (fast sizes), dirichlet_medium_tests without --full"
    fi
    # RR10: threshold A
    case_run "$PD" long_box_only_A "$PD/long_box_only_A" any - - 0 -- "$(bin long_box_tests)" $onlyA
    if [ "$DRY" = "1" ]; then
        ctl_exe_verdict long_box_only_A
    else
        a_verdict long_box_only_A
    fi
    case_run "$PD" long_box_full "$PD/long_box_full" any - - 0 -- "$(bin long_box_tests)" $full
    ctl_exe_verdict long_box_full
    # medium (b) library
    case_run "$PD" dirichlet_medium_full "$PD/dirichlet_medium_full" any - - 0 -- "$(bin dirichlet_medium_tests)" $full
    ctl_exe_verdict dirichlet_medium_full
    # the fast ctests, executables directly (what `ctest -R 'lagrangian_dirichlet_medium|lagrangian_ensemble'` runs)
    case_run "$PD" dirichlet_medium_fast "$PD/dirichlet_medium_fast" any - - 0 -- "$(bin dirichlet_medium_tests)"
    ctl_exe_verdict dirichlet_medium_fast
    case_run "$PD" lagrangian_ensemble_fast "$PD/lagrangian_ensemble_fast" any - - 0 -- \
        "$(bin lagrangian_ensemble_tests)" "$ENS" "$PIPE"
    ctl_exe_verdict lagrangian_ensemble_fast
    # runner smoke 64x16x16, two realizations, twin and bridge on
    case_run "$PD" runner_smoke "$PD/runner_smoke" any X - 0 -- "$ENS" --grid 64,16,16 --ell 0.25 --eps 1 --seed0 $SEED0 --nr 2 \
        --np-side 10 --twin on --pollock-bridge on --out "$PD/runner_smoke"
    smoke_verdict runner_smoke 2
    # sampler smoke: the RR2 PID filter must find samples (a box large enough to hold the context for > 1 s)
    case_run "$PD" sampler_smoke "$PD/sampler_smoke" any SX - 0 -- "$ENS" --grid 256,32,32 --ell 0.25 --eps 1 --seed0 $SEED0 --nr 1 \
        --np-side 20 --twin on --pollock-bridge on --out "$PD/sampler_smoke"
    smoke_verdict sampler_smoke 1
    sampler_verdict sampler_smoke
}

# ============================================================================================================
# production inputs (C-PROD) and the production-grid table
# ============================================================================================================
# prod_cmd <grid> <eps> <ell> <np_side> <seed0> <nr> <out>  ->  PROD_CMD = the runner command line (RR3 / C-PROD)
PROD_CMD=()
prod_cmd() {
    PROD_CMD=("$(bin lagrangian_ensemble)" --grid "$1" --eps "$2" --ell "$3" --seed0 "$5" --nr "$6" --x0 0 --window 0.1,0.9,0.1,0.9
        --np-side "$4" --planes auto --times auto --tol 1e-8 --hmax-div 8 --fused-splines --single-launch --twin on
        --pollock-bridge on --out "$7")
}

# b_table: lines `name grid eps ell np_side`
b_table() {
    if [ "$DRY" = "1" ]; then
        echo "B1 64,16,16 1 0.25 10"
        echo "B2 64,16,16 1.5 0.25 10"
        echo "B3 64,16,16 2 0.25 10"
    else
        echo "B1 2048,256,256 1 0.0390625 100"
        echo "B2 3328,384,384 1.5 0.041666666666666664 100"
        echo "B3 3328,384,384 2 0.08333333333333333 100"
    fi
}

part_prodgrid() {
    part_begin prodgrid
    local name grid eps ell np
    while read -r name grid eps ell np; do
        prod_cmd "$grid" "$eps" "$ell" "$np" $SEED0 1 "$PD/$name"
        case_run "$PD" "$name" "$PD/$name" any SX - "$LONG_TMO" -- "${PROD_CMD[@]}"
    done < <(b_table)
    "$PY" "$SCRIPTS/b_verdict.py" "$PD" | res_block
    grep -q '^B_ALL PASS' "$RES" || BAD=$((BAD + 1))
}

# ============================================================================================================
# part bpilot (RR4) and ab (RR5, RR6)
# ============================================================================================================
AB_S1=(--grid 2048,256,256 --ell 0.0390625 --eps 1)
AB_S4=(--grid 3328,256,256 --ell 0.125 --eps 2)
if [ "$DRY" = "1" ]; then
    AB_S1=(--grid 64,16,16 --ell 0.25 --eps 1)
    AB_S4=(--grid 64,16,16 --ell 0.125 --eps 2)
fi

part_bpilot() {
    part_begin bpilot
    # RR4: the pilot is the A/B application at the sigma^2 = 4 A/B grid, one realization (seed 4001); only the (b) solver record
    # (B_SOLVER line) and the memory are read by c0_verdict.py; nothing else of its output is inspected here.
    case_run "$PD" bpilot "$PD/bpilot" any SX - "$LONG_TMO" -- "$(bin lagrangian_medium_ab)" "${AB_S4[@]}" --seed0 $SEED0 --nr 1 --out "$PD/bpilot"
    "$PY" "$SCRIPTS/c0_verdict.py" "$PD" bpilot | res_block
    grep -Eq '^C0_VERDICT .* (PASS|PASS_FALLBACK)( \[|$)' "$RES" || BAD=$((BAD + 1))
}

part_ab() {
    part_begin ab
    local guard_ok=0
    if [ "${SF38_AB_SKIP_C0_GUARD:-0}" = "1" ]; then
        guard_ok=1
        res "RECORDED c0_guard=skipped (SF38_AB_SKIP_C0_GUARD=1)"
    elif grep -Eqs '^C0_VERDICT .* (PASS|PASS_FALLBACK)( \[|$)' "$OUT_ROOT/bpilot/results.txt"; then
        guard_ok=1
        res "RECORDED c0_guard=ok ($(grep -E '^C0_VERDICT' "$OUT_ROOT/bpilot/results.txt" | cut -c1-200))"
    fi
    if [ "$guard_ok" -ne 1 ]; then
        res "AB_REFUSED no C0 PASS/PASS_FALLBACK in $OUT_ROOT/bpilot/results.txt (failure policy: C0 not met -> stop before the A/B; owner)"
        BAD=$((BAD + 1))
        return 0
    fi
    local extra=()
    if [ -n "${SF38_AB_EXTRA_ARGS:-}" ]; then
        read -r -a extra <<< "$SF38_AB_EXTRA_ARGS"
        res "RECORDED ab_extra_args=$SF38_AB_EXTRA_ARGS"
    fi
    case_run "$PD" ab_s1 "$PD/ab_s1" " 0 " SX - "$LONG_TMO" -- "$(bin lagrangian_medium_ab)" "${AB_S1[@]}" --seed0 $SEED0 --nr 10 \
        --out "$PD/ab_s1" ${extra[@]+"${extra[@]}"}
    case_run "$PD" ab_s4 "$PD/ab_s4" " 0 " SX - "$LONG_TMO" -- "$(bin lagrangian_medium_ab)" "${AB_S4[@]}" --seed0 $SEED0 --nr 10 \
        --out "$PD/ab_s4" ${extra[@]+"${extra[@]}"}
    local l1=0.0390625 l4=0.125
    if [ "$DRY" = "1" ]; then
        l1=0.25
        l4=0.125
    fi
    local out rc
    out=$("$PY" "$SCRIPTS/ab_decision.py" "$PD/ab_s1/ab_primary.csv" "$PD/ab_s4/ab_primary.csv" --lambda1 $l1 --lambda4 $l4)
    rc=$?
    echo "$out" > "$PD/ab_decision.out"
    sha_file "$PD" ab_decision.out
    echo "$out" | res_block
    if [ "$rc" -ne 0 ] || ! grep -q '^C_VERDICT selected=' "$RES"; then
        res "C_VERDICT selected=none (ab_decision.py exit $rc: no selection computed)"
        BAD=$((BAD + 1))
    fi
    for r in ab_s1 ab_s4; do
        [ "$(tsv_rc "$PD/exit_codes.tsv" $r)" = "0" ] || BAD=$((BAD + 1))
        mem_records_for $r
    done
    # determinism check (RECORDED): the pilot's realization 4001 and the A/B's realization 4001 are the same pure function
    local bp="$OUT_ROOT/bpilot/bpilot"
    "$PY" "$SCRIPTS/d_checks.py" tree-cmp --name AB_4001_vs_BPILOT --ref "$bp/realizations/r_0000" --cand "$PD/ab_s4/realizations/r_0000" \
        --exclude timings.json | sed 's/^D_BYTES/RECORDED AB_DETERMINISM_4001 D_BYTES/' | res_block
}

mem_records_for() { # mem_records_for <run>: RECORDED memory / wall lines of a sampled run in $PD
    "$PY" - "$SCRIPTS" "$PD" "$1" << 'EOF' | res_block
import sys
sys.path.insert(0, sys.argv[1])
import mem_verdict as mv
print("\n".join(mv.mem_records(sys.argv[2], sys.argv[3])))
EOF
}

# ============================================================================================================
# part ensemble (D, RR7)
# ============================================================================================================
part_ensemble() {
    part_begin ensemble
    local grid=3328,384,384 eps=2 ell=0.08333333333333333 np=100 nr=3
    if [ "$DRY" = "1" ]; then
        grid=64,16,16
        eps=2
        ell=0.25
        np=10
    fi
    local PIPE
    PIPE=$(bin macroflow3d_pipeline)
    # D-A: uninterrupted, seeds 4001-4003
    prod_cmd "$grid" "$eps" "$ell" "$np" $SEED0 $nr "$PD/DA"
    res "RECORDED D_INVOCATION DA: ${PROD_CMD[*]}"
    case_run "$PD" DA "$PD/DA" " 0 " X - "$LONG_TMO" -- "${PROD_CMD[@]}"
    # D-B: the same command, SIGKILL of the runner after `R=1 STAGE` appears in its stdout, restarted with the identical command
    prod_cmd "$grid" "$eps" "$ell" "$np" $SEED0 $nr "$PD/DB"
    local DB_CMD=("${PROD_CMD[@]}")
    res "RECORDED D_INVOCATION DB_run1 (SIGKILL after '^R=1 STAGE '): ${DB_CMD[*]}"
    res "RECORDED D_INVOCATION DB_run2 (restart, identical command): ${DB_CMD[*]}"
    if [ -z "$(tsv_rc "$PD/exit_codes.tsv" DB_run2)" ] && [ -z "$(tsv_rc "$PD/exit_codes.tsv" DB_run1)" ]; then
        db_run1 "${DB_CMD[@]}"
    elif [ -n "$(tsv_rc "$PD/exit_codes.tsv" DB_run1)" ]; then
        echo "SKIP_DONE run=DB_run1 rc=$(tsv_rc "$PD/exit_codes.tsv" DB_run1)"
    fi
    if [ -n "$(tsv_rc "$PD/exit_codes.tsv" DB_run1)" ]; then
        case_run "$PD" DB_run2 "$PD/DB_run2" " 0 " X - "$LONG_TMO" -- "${DB_CMD[@]}"
    else
        res "D_REFUSED DB_run1 was not run (GPU busy): the restart is meaningless without the killed invocation; re-invoke the part"
        BAD=$((BAD + 1))
    fi
    # D-C: realization 4001 alone
    prod_cmd "$grid" "$eps" "$ell" "$np" $SEED0 1 "$PD/DC"
    res "RECORDED D_INVOCATION DC: ${PROD_CMD[*]}"
    case_run "$PD" DC "$PD/DC" " 0 " X - "$LONG_TMO" -- "${PROD_CMD[@]}"
    # in-process vs analysis_only: a copy of D-A reduced to stats/ + analysis_config.yaml is analysed by macroflow3d_pipeline
    local cp=$PD/DA_analysis_copy
    if [ -z "$(tsv_rc "$PD/exit_codes.tsv" DA_analysis)" ]; then
        safe_rm "$cp"
        mkdir -p "$cp"
        cp -a "$PD/DA/stats" "$PD/DA/analysis_config.yaml" "$cp/" 2> /dev/null
    fi
    case_run "$PD" DA_analysis "$PD/DA_analysis" " 0 " X - 3600 -- bash -c 'cd "$1" && exec "$2" analysis_config.yaml' _ "$cp" "$PIPE"
    sha_file "$PD" DA_analysis_copy/analysis/macrodispersion.csv DA_analysis_copy/ensemble/ensemble_timeseries.csv
    d_verdicts
}

# db_run1 <runner command...>: the first invocation of D-B, killed (SIGKILL) once `R=1 STAGE` is in its stdout
db_run1() {
    local base=$PD/DB_run1 out=$PD/DB f exe pidt killed=0 waited=0 poll=0.2 rc t0 end
    exe=$(basename "$1")
    if [ -e "$out" ] || compgen -G "$base.*" > /dev/null; then
        echo "RESUMED run=DB_run1 (partial outputs and no recorded rc: removed, re-run with identical settings)"
        printf '%s\t%s\t%s\n' DB_run1 "$(date -u +%FT%TZ)" "partial outputs removed" >> "$PD/resumed.txt"
        safe_rm "$out"
        for f in "$base".*; do
            [ -e "$f" ] && safe_rm "$f"
        done
    fi
    precheck_gpu DB_run1 "$base" || return 3
    command -v free > /dev/null 2>&1 && free -b > "$base.free" 2>&1
    [ "$DRY" = "1" ] && poll=0.05
    echo "+ [DB_run1] $*   > $base.log 2>&1   (SIGKILL once '^R=1 STAGE ' appears)"
    t0=$(date +%s)
    run_in - "$base.time" "$LONG_TMO" "$@" > "$base.log" 2>&1 &
    CHILD=$!
    while kill -0 "$CHILD" 2> /dev/null; do
        if grep -q '^R=1 STAGE ' "$base.log" 2> /dev/null; then
            pidt=$(find_desc "$CHILD" "${exe:0:15}")
            if [ -n "$pidt" ]; then
                kill -KILL "$pidt" 2> /dev/null && killed=1
            fi
            break
        fi
        sleep "$poll"
        waited=$((waited + 1))
    done
    wait "$CHILD"
    rc=$?
    CHILD=""
    end=$(date -u +%FT%TZ)
    printf '%s\t%s\n' DB_run1 "$rc" >> "$PD/exit_codes.tsv"
    printf '%s\t%s\t%s\n' DB_run1 "$(( $(date +%s) - t0 ))" "$end" >> "$PD/walls.tsv"
    append_sha "$PD" "$base"
    echo "  CASE_RESULT run=DB_run1 rc=$rc killed_by_script=$killed end=$end"
    echo "$killed" > "$base.killed"
}

d_verdicts() {
    local nfail=0 out rc r f
    local rcA rc1 rc2 rcC rcP killed r0 r1
    rcA=$(tsv_rc "$PD/exit_codes.tsv" DA)
    rc1=$(tsv_rc "$PD/exit_codes.tsv" DB_run1)
    rc2=$(tsv_rc "$PD/exit_codes.tsv" DB_run2)
    rcC=$(tsv_rc "$PD/exit_codes.tsv" DC)
    rcP=$(tsv_rc "$PD/exit_codes.tsv" DA_analysis)
    killed=$(cat "$PD/DB_run1.killed" 2> /dev/null || echo 0)
    res "RECORDED D_INVOCATIONS DA rc=$rcA; DB_run1 (killed) rc=$rc1 killed_by_script=$killed; DB_run2 (restart, identical command) rc=$rc2; DC rc=$rcC; DA_analysis rc=$rcP"
    # kill state: realization 0 completed, realization 1 not
    r0=$(grep -c '^R=0 STATUS ok' "$PD/DB_run1.log" 2> /dev/null || true)
    r1=$(grep -c '^R=1 STAGE ' "$PD/DB_run1.log" 2> /dev/null || true)
    local kok=0
    [ "$killed" = "1" ] && [ "${r0:-0}" -ge 1 ] && [ "${r1:-0}" -ge 1 ] && [ "$rc1" != "0" ] && ! grep -q '^R=1 STATUS' "$PD/DB_run1.log" && kok=1
    res "D_KILL run=DB_run1 rc=$rc1 killed_by_script=$killed r0_status_ok=${r0:-0} r1_stage_line=${r1:-0} r1_status_line=$(grep -c '^R=1 STATUS' "$PD/DB_run1.log" 2> /dev/null || true) $( [ "$kok" = 1 ] && echo CONFIRMED || echo NOT_CONFIRMED)"
    [ "$kok" = "1" ] || nfail=$((nfail + 1))
    # D-B tree vs D-A (RR7: exactly timings.json excluded)
    out=$("$PY" "$SCRIPTS/d_checks.py" tree-cmp --name DB_vs_DA --ref "$PD/DA" --cand "$PD/DB" --exclude timings.json)
    rc=$?
    echo "$out" | res_block
    [ "$rc" -eq 0 ] || nfail=$((nfail + 1))
    # D-C: realization 0000 of D-A (run-level aggregates of a one-realization run differ by construction)
    out=$("$PY" "$SCRIPTS/d_checks.py" tree-cmp --name DC_vs_DA_r0000 --ref "$PD/DA" --cand "$PD/DC" --exclude timings.json \
        --path realizations/r_0000 --path stats/realization_0000_timeseries.csv --path stats_twin/realization_0000_timeseries.csv)
    rc=$?
    echo "$out" | res_block
    [ "$rc" -eq 0 ] || nfail=$((nfail + 1))
    # analysis_only vs in-process (<= 1e-12 relative; byte equality reported)
    for f in analysis/macrodispersion.csv ensemble/ensemble_timeseries.csv; do
        out=$("$PY" "$SCRIPTS/d_checks.py" csv-cmp --name "analysis_only_vs_in_process:$f" --ref "$PD/DA/$f" --cand "$PD/DA_analysis_copy/$f" \
            --rtol 1e-12 --atol 1e-300)
        rc=$?
        echo "$out" | res_block
        [ "$rc" -eq 0 ] || nfail=$((nfail + 1))
    done
    # stats CSVs (read as read_timeseries_csv does) vs the TTIME lines of the D-A log
    out=$("$PY" "$SCRIPTS/d_checks.py" ttime-cmp --name DA_stats_vs_TTIME --log "$PD/DA.log" --stats-dir "$PD/DA/stats" --rtol 1e-12 --t-rtol 1e-9)
    rc=$?
    echo "$out" | res_block
    [ "$rc" -eq 0 ] || nfail=$((nfail + 1))
    for r in DA DB DC; do
        "$PY" "$SCRIPTS/d_checks.py" walls --name "$r" --root "$PD/$r" | res_block
    done
    for r in "$rcA" "$rc2" "$rcC" "$rcP"; do
        [ "$r" = "0" ] || nfail=$((nfail + 1))
    done
    if [ "$nfail" -eq 0 ]; then
        res "D_VERDICT PASS (D-A, D-B restart, D-C, analysis_only reproduction all met)"
    else
        res "D_VERDICT FAIL failing_items=$nfail"
        BAD=$((BAD + 1))
    fi
    for r in DA DB_run2 DC; do
        mem_records_for_unsampled "$r"
    done
}

mem_records_for_unsampled() { # RECORDED wall / RSS of a run without sampler files
    local r=$1
    res "RECORDED run=$r exit=$(tsv_rc "$PD/exit_codes.tsv" "$r") wall_s=$(tsv_col "$PD/walls.tsv" "$r" 2) host_max_rss_kb=$(grep 'Maximum resident' "$PD/$r.time" 2> /dev/null | awk '{print $NF}')"
}

# ============================================================================================================
# part fieldonly (G, RR11)
# ============================================================================================================
part_fieldonly() {
    part_begin fieldonly
    local name grid eps ell np args=()
    while read -r name grid eps ell np; do
        case_run "$PD" "F_$name" "$PD/F_$name" " 0 " X - "$LONG_TMO" -- "$(bin lagrangian_ensemble)" --field-only --grid "$grid" --ell "$ell" \
            --seed0 $SEED0 --nr 10 --out "$PD/F_$name"
        args+=(--geom "$name=$PD/F_$name")
    done < <(b_table)
    local abargs=()
    [ -d "$OUT_ROOT/ab/ab_s1" ] && abargs+=(--ab "AB_s1=$OUT_ROOT/ab/ab_s1")
    [ -d "$OUT_ROOT/ab/ab_s4" ] && abargs+=(--ab "AB_s4=$OUT_ROOT/ab/ab_s4")
    [ "${#abargs[@]}" -eq 0 ] && res "RECORDED G_AB_GRIDS not available (no $OUT_ROOT/ab/ab_s1 or ab_s4: run part ab first; rerun this part to add them)"
    local out rc
    out=$("$PY" "$SCRIPTS/g_table.py" "${args[@]}" ${abargs[@]+"${abargs[@]}"} --seed0 $SEED0 --nr 10)
    rc=$?
    echo "$out" | res_block
    if [ "$rc" -eq 0 ]; then
        res "G_VERDICT RECORDED_COMPLETE (10 seeds at each of the three production geometries; G has no threshold)"
    else
        res "G_VERDICT INCOMPLETE"
        BAD=$((BAD + 1))
    fi
}

# ============================================================================================================
# part par2pilot (E, RR8)
# ============================================================================================================
epoch_start_of() { # epoch_start_of <pd> <run> -> start epoch (end_utc - wall) of a finished run
    local end wall
    end=$(tsv_col "$1/walls.tsv" "$2" 3)
    wall=$(tsv_col "$1/walls.tsv" "$2" 2)
    [ -n "$end" ] && echo $(( $(date -u -d "$end" +%s) - ${wall:-0} ))
}

yaml_val() { # yaml_val <file> <key> -> first scalar value of `key:`
    sed -n "s/^[[:space:]]*$2:[[:space:]]*\\([-+0-9.eE]*\\).*/\\1/p" "$1" | head -n 1
}

part_par2pilot() {
    part_begin par2pilot
    local PIPE FS DIAG c cfg cfgdir=$PD/cfg
    PIPE=$(bin macroflow3d_pipeline)
    FS=$(bin par2_field_stats)
    DIAG=${SF38_DIAG_PIPE:-$REPO/build/v100-diag/macroflow3d_pipeline}
    [ "$DRY" = "1" ] && DIAG=${SF38_DIAG_PIPE:-$PIPE}
    mkdir -p "$cfgdir"
    # config files: the committed ones; dry run: tiny copies
    for c in $PAR2_CFGS; do
        if [ "$DRY" = "1" ]; then
            "$PY" "$SCRIPTS/par2_pilot.py" make-copy --src "$REPO/apps/config_pipeline_par2_sf38_$c.yaml" --dst "$cfgdir/$c.yaml" \
                --kind dry --outdir "sf38_par2_$c" || { BAD=$((BAD + 1)); return 0; }
        else
            cp -f "$REPO/apps/config_pipeline_par2_sf38_$c.yaml" "$cfgdir/$c.yaml"
        fi
        "$PY" "$SCRIPTS/par2_pilot.py" make-copy --src "$cfgdir/$c.yaml" --dst "$cfgdir/dt2_$c.yaml" --kind dt2 --outdir "sf38_par2_${c}_dt2" ||
            { BAD=$((BAD + 1)); return 0; }
        "$PY" "$SCRIPTS/par2_pilot.py" make-copy --src "$cfgdir/$c.yaml" --dst "$cfgdir/ks_$c.yaml" --kind kstats --outdir "sf38_par2_${c}_kstats" ||
            { BAD=$((BAD + 1)); return 0; }
    done
    sha_file "$PD" cfg/s1.yaml cfg/s225.yaml cfg/s4.yaml cfg/dt2_s1.yaml cfg/dt2_s225.yaml cfg/dt2_s4.yaml cfg/ks_s1.yaml cfg/ks_s225.yaml cfg/ks_s4.yaml
    res "RECORDED par2 configs: $(for c in $PAR2_CFGS; do printf '%s ' "$c"; done)(dt2 copies: dt halved, n_steps / snapshots.every / sample_every / output_every doubled, NR 3; see par2_pilot.py)"
    # D17 tool: realized ln K statistics, 10 realizations per config
    for c in $PAR2_CFGS; do
        case_run "$PD" "fs_$c" "$PD/fs_$c" " 0 " X - "$PAR2_TMO" -- "$FS" "$cfgdir/$c.yaml"
        "$PY" "$SCRIPTS/par2_pilot.py" fieldstat --cfg-name "$c" --log "$PD/fs_$c.log" | res_block
    done
    # pilot runs, N_R = 10, each from its own working directory (output_dir is relative)
    local etot=0
    for c in $PAR2_CFGS; do
        case_run "$PD" "run_$c" "$PD/run_$c" " 0 " SX "$PD/run_$c" "$PAR2_TMO" -- "$PIPE" "$cfgdir/$c.yaml"
        local od="$PD/run_$c/output/sf38_par2_$c"
        local line ok
        line=$("$PY" "$SCRIPTS/par2_pilot.py" outputs --cfg-name "$c" --out-dir "$od" --nr 10)
        ok=$?
        res "$line"
        [ "$ok" -eq 0 ] && [ "$(tsv_rc "$PD/exit_codes.tsv" "run_$c")" = "0" ] || etot=$((etot + 1))
        res "RECORDED run=run_$c exit=$(tsv_rc "$PD/exit_codes.tsv" "run_$c") wall_s=$(tsv_col "$PD/walls.tsv" "run_$c" 2) time_per_realization_mean_s=$(awk -v w="$(tsv_col "$PD/walls.tsv" "run_$c" 2)" 'BEGIN { printf "%.3f", w / 10 }') (total wall / 10, includes setup)"
        local st
        st=$(epoch_start_of "$PD" "run_$c")
        [ -n "$st" ] && "$PY" "$SCRIPTS/par2_pilot.py" rtimes --cfg-name "$c" --stats-dir "$od/stats" --start-epoch "$st" --nr 10 | res_block
        "$PY" "$SCRIPTS/par2_pilot.py" tb --cfg "$cfgdir/$c.yaml" --cfg-name "$c" --out-dir "$od" --nr 10 | res_block
        mkdir -p "$PD/series"
        cp -f "$od/analysis/macrodispersion.csv" "$PD/series/${c}_macrodispersion.csv" 2> /dev/null
        cp -f "$od/ensemble/ensemble_timeseries.csv" "$PD/series/${c}_ensemble_timeseries.csv" 2> /dev/null
        sha_file "$PD" "series/${c}_macrodispersion.csv" "series/${c}_ensemble_timeseries.csv"
        res "RECORDED PAR2_PEAK run_$c: $("$PY" "$SCRIPTS/mem_verdict.py" peak "$PD" "run_$c" 2> /dev/null | head -n 1)"
    done
    # paired dt/2 check on the first 3 realizations
    for c in $PAR2_CFGS; do
        case_run "$PD" "dt2_$c" "$PD/dt2_$c" " 0 " X "$PD/dt2_$c" "$PAR2_TMO" -- "$PIPE" "$cfgdir/dt2_$c.yaml"
        local od="$PD/run_$c/output/sf38_par2_$c" od2="$PD/dt2_$c/output/sf38_par2_${c}_dt2" pc="$PD/pair_$c.in"
        res "RECORDED run=dt2_$c exit=$(tsv_rc "$PD/exit_codes.tsv" "dt2_$c") wall_s=$(tsv_col "$PD/walls.tsv" "dt2_$c" 2) time_per_realization_mean_s=$(awk -v w="$(tsv_col "$PD/walls.tsv" "dt2_$c" 2)" 'BEGIN { printf "%.3f", w / 3 }')"
        if [ -z "$(tsv_rc "$PD/exit_codes.tsv" "pair_$c")" ]; then
            safe_rm "$pc"
            mkdir -p "$pc/stats"
            cp -f "$od/stats/realization_0000_timeseries.csv" "$od/stats/realization_0001_timeseries.csv" "$od/stats/realization_0002_timeseries.csv" "$pc/stats/" 2> /dev/null
            {
                echo "run_mode: analysis_only"
                echo "output:"
                echo "  output_dir: \".\""
                echo "analysis:"
                echo "  macrodispersion:"
                echo "    enabled: true"
                echo "    NR: 3"
                echo "    lambda: $(yaml_val "$cfgdir/$c.yaml" lambda)"
                echo "    vmean_norm: $(yaml_val "$cfgdir/$c.yaml" vmean_norm)"
            } > "$pc/analysis_config.yaml"
        fi
        case_run "$PD" "pair_$c" "$PD/pair_$c.analysis" " 0 " - - 3600 -- bash -c 'cd "$1" && exec "$2" analysis_config.yaml' _ "$pc" "$PIPE"
        "$PY" "$SCRIPTS/par2_pilot.py" paired --cfg-name "$c" --a "$pc/analysis/macrodispersion.csv" --b "$od2/analysis/macrodispersion.csv" \
            --out "$PD/series/${c}_paired_dt_minus_dt2.csv" | res_block
        sha_file "$PD" "series/${c}_paired_dt_minus_dt2.csv" "pair_$c.in/analysis/macrodispersion.csv"
    done
    # diagnostics build (K stats lines) and the K-stats agreement at %.4e
    if [ "$DRY" != "1" ] && [ ! -x "$DIAG" ]; then
        res "RECORDED diag_build: configuring and building $(dirname "$DIAG") (bare -j allowed on the V100)"
        ( cd "$REPO" && cmake -S . -B build/v100-diag -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
            -DMACROFLOW3D_ENABLE_DIAGNOSTICS=ON -DMACROFLOW3D_ENABLE_PETSC=OFF > "$PD/diag_build.log" 2>&1 &&
            cmake --build build/v100-diag --target macroflow3d_pipeline -j >> "$PD/diag_build.log" 2>&1 )
        printf '%s\t%s\n' diag_build "$?" >> "$PD/exit_codes.tsv"
        sha_file "$PD" diag_build.log
    fi
    if [ -x "$DIAG" ]; then
        res "RECORDED diag_pipeline=$DIAG md5=$(md5sum "$DIAG" | awk '{print $1}')"
        for c in $PAR2_CFGS; do
            case_run "$PD" "ks_$c" "$PD/ks_$c" " 0 " X "$PD/ks_$c" "$PAR2_TMO" -- "$DIAG" "$cfgdir/ks_$c.yaml"
            "$PY" "$SCRIPTS/par2_pilot.py" kstats --cfg-name "$c" --fieldstat-log "$PD/fs_$c.log" --pipeline-log "$PD/ks_$c.log" | res_block
        done
    else
        res "RECORDED diag_pipeline=unavailable ($DIAG): K-stats agreement not checked"
    fi
    if [ "$etot" -eq 0 ]; then
        res "E_VERDICT PASS (the three N_R = 10 runs exit 0 with outputs present)"
    else
        res "E_VERDICT FAIL runs_failing=$etot"
        BAD=$((BAD + 1))
    fi
}

# ============================================================================================================
# part final (HEAD build dir): full ctest + SF-27 trio + R set
# ============================================================================================================
bc_r_case() { # bc_r_case <kind> <name> <a1> <a2> <a3>   (SF-40 bc_r_case)
    local kind=$1 name=$2 id dir ref toks
    if [ "$kind" = "prod" ]; then
        id=r_prod
        dir=$PD/r/prod32
        ref=$BASE_REFS/r/prod32
        r_args prod "$name" "$3" "$4" "$5" "$dir"
        case_run "$PD" "$id" "$dir" any - - 0 -- "$(bin inlet_slab)" "${R_ARGS[@]}"
        [ "$CASE_STATE" = "busy" ] && return 0
        bc_lines "r:prod32:lines" "$dir.log" "$ref.log" CASE ORACLE ORACLE_SUMMARY EXTRA
        bc_rc_check "r:prod32" "$CASE_RC" "$ref.time"
        return 0
    fi
    id=r_${kind}_$name
    dir=$PD/r/$kind/$name
    ref=$BASE_REFS/r/$kind/$name
    if ! r_args "$kind" "$name" "$3" "$4" "$5" "$dir"; then
        bc_res MISSING "r:$kind:$name (no argument set)"
        return 0
    fi
    case_run "$PD" "$id" "$dir" any - - 0 -- "$(bin inlet_slab)" "${R_ARGS[@]}"
    [ "$CASE_STATE" = "busy" ] && return 0
    if [ "$kind" = "trace" ]; then
        BC_DROPS=('**.timings_s' 'transport.seconds_*')
        bc_dir_files "r:$kind:$name" "$dir" "$ref" 1
        bc_lines "r:$kind:$name:lines" "$dir.log" "$ref.log" CASE PLANE BACKFLOW GROWTH RESIDUAL TRANSPORT_STATUS STATUS
    else
        BC_DROPS=('**.timings_s')
        bc_dir_files "r:$kind:$name" "$dir" "$ref" 1
        toks="SETUP FIELD TRACK BACKFLOW TPLANE TTIME TSTATUS STATUS"
        [ "$kind" = "neutral" ] && toks="$toks PTRACK PPLANE PTIME PSTATUS PFAIL SEEDV"
        bc_lines "r:$kind:$name:lines" "$dir.log" "$ref.log" $toks
    fi
    bc_rc_check "r:$kind:$name" "$CASE_RC" "$ref.time"
}

part_final() {
    part_begin final
    local PIPE
    PIPE=$(bin macroflow3d_pipeline)
    res "RECORDED binaries md5: $(md5sum "$PIPE" "$(bin inlet_slab)" 2> /dev/null | tr '\n' ' ')"
    # ---- F: full ctest (dry run: one fast case)
    local cargs=()
    if [ -n "${SF38_CTEST_ARGS:-}" ]; then
        read -r -a cargs <<< "$SF38_CTEST_ARGS"
    elif [ "$DRY" = "1" ]; then
        cargs=(-R '^lagrangian_dirichlet_medium$')
    fi
    case_run "$PD" ctest "$PD/ctest" any - - 0 -- ctest --test-dir "$BIN_DIR" --output-on-failure ${cargs[@]+"${cargs[@]}"}
    local crc summ ntot nfail
    crc=$(tsv_rc "$PD/exit_codes.tsv" ctest)
    summ=$(grep -E '^[0-9]+% tests passed' "$PD/ctest.log" | tail -n 1) # "100% tests passed out of N" or "NN% tests passed, K tests failed out of N"
    ntot=$(echo "$summ" | sed -n 's/.*out of \([0-9]*\).*/\1/p')
    nfail=$(echo "$summ" | sed -n 's/.*, \([0-9]*\) tests failed.*/\1/p')
    nfail=${nfail:-0}
    grep -E '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$PD/ctest.log" | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: +([^ ]+) +\.+ *(\*\*\*)?([A-Za-z ]+[a-z]) +([0-9.]+) sec.*/\1\t\3\t\4/' > "$PD/ctest_times.tsv"
    sha_file "$PD" ctest_times.tsv
    res "RECORDED ctest_per_test_times written ($PD/ctest_times.tsv, $(wc -l < "$PD/ctest_times.tsv" | tr -d ' ') tests: name<TAB>status<TAB>seconds)"
    if [ "$crc" = "0" ] && [ -n "$ntot" ] && [ "$ntot" -ge 1 ] && [ "$nfail" = "0" ]; then
        res "F_CTEST exit=0 tests=$ntot failed=0 wall_s=$(tsv_col "$PD/walls.tsv" ctest 2) PASS (expected content change: lagrangian_tracker_gpu_host only; summary: $summ)"
    else
        res "F_CTEST exit=$crc tests=${ntot:--} failed=$nfail wall_s=$(tsv_col "$PD/walls.tsv" ctest 2) FAIL (summary: ${summ:-none})"
        BAD=$((BAD + 1))
    fi
    # ---- F: SF-27 trio vs the base references
    local cfg
    BC_N=0
    for cfg in $TRIO; do
        case_run "$PD" "trio_$cfg" "$PD/trio/$cfg" any - "$PD/trio/$cfg" 0 -- "$PIPE" "$REPO/apps/$cfg.yaml"
        bc_tree "trio:$cfg" "$PD/trio/$cfg" "$BASE_REFS/trio/$cfg"
        bc_rc_check "trio:$cfg" "$CASE_RC" "$BASE_REFS/trio/$cfg.time"
    done
    if [ "$BC_N" -eq 0 ]; then
        res "F_TRIO IDENTICAL (SF-27 trio byte-compare clean vs $BASE_REFS/trio; manifest keys ${MANIFEST_DROPS[*]} dropped)"
    else
        res "F_TRIO DIFF n=$BC_N"
        BAD=$((BAD + 1))
    fi
    # ---- F: the 20-run R set vs the base references
    BC_N=0
    local kind name a1 a2 a3
    if [ -d "$BASE_REFS/r" ]; then
        while read -r kind name a1 a2 a3; do
            bc_r_case "$kind" "$name" "$a1" "$a2" "$a3"
        done < <(r_items)
    else
        bc_res MISSING "base references $BASE_REFS/r (part baserefs not run)"
    fi
    local hmd5 bmd5
    hmd5=$(md5sum "$(bin inlet_slab)" | awk '{print $1}')
    bmd5=$(md5sum "$BASE_REFS/bin/inlet_slab" 2> /dev/null | awk '{print $1}')
    if [ "$hmd5" != "$bmd5" ]; then
        bc_res IDENTICAL "binary:md5_differs head=$hmd5 base=${bmd5:-absent}"
    else
        bc_res DIFF "binary:md5_differs (head and base inlet_slab are the SAME file content $hmd5)"
    fi
    if [ "$BC_N" -eq 0 ]; then
        res "F_RSET IDENTICAL (R set byte-identical to $BASE_REFS/r; SF-40 compare rules)"
    else
        res "F_RSET DIFF n=$BC_N"
        BAD=$((BAD + 1))
    fi
}

# ============================================================================================================
# dispatch
# ============================================================================================================
main() {
    trap on_signal INT TERM
    if [ -z "$ARG_BIN_DIR" ] || [ -z "$ARG_PART" ]; then
        usage
        exit 2
    fi
    [ "$ARG_PART" = "par2-pilot" ] && ARG_PART=par2pilot
    local fn need f
    case "$ARG_PART" in
    baserefs | controls | prodgrid | bpilot | ab | ensemble | fieldonly | par2pilot | final) fn=part_$ARG_PART ;;
    *)
        echo "run_sf38.sh: unknown part '$ARG_PART'" >&2
        usage
        exit 2
        ;;
    esac
    BIN_DIR=$(cd "$ARG_BIN_DIR" 2> /dev/null && pwd) || {
        echo "run_sf38.sh: build_dir '$ARG_BIN_DIR' not found" >&2
        exit 2
    }
    case "$ARG_PART" in
    baserefs) need="inlet_slab macroflow3d_pipeline" ;;
    controls) need="long_box_tests dirichlet_medium_tests lagrangian_ensemble_tests lagrangian_ensemble macroflow3d_pipeline" ;;
    prodgrid | fieldonly) need="lagrangian_ensemble" ;;
    bpilot | ab) need="lagrangian_medium_ab" ;;
    ensemble) need="lagrangian_ensemble macroflow3d_pipeline" ;;
    par2pilot) need="macroflow3d_pipeline par2_field_stats" ;;
    final) need="macroflow3d_pipeline inlet_slab" ;;
    esac
    for f in $need; do
        [ -x "$BIN_DIR/$f" ] || { echo "run_sf38.sh: $BIN_DIR/$f not found or not executable" >&2; exit 2; }
    done
    mkdir -p "$OUT_ROOT" && OUT_ROOT=$(cd "$OUT_ROOT" && pwd)
    if [ -d "$BASE_REFS" ]; then
        BASE_REFS=$(cd "$BASE_REFS" && pwd)
    elif [ "$ARG_PART" = "baserefs" ]; then
        mkdir -p "$BASE_REFS" && BASE_REFS=$(cd "$BASE_REFS" && pwd)
    fi
    if [ ! -x /usr/bin/time ]; then
        if [ "$DRY" = "1" ]; then
            echo "WARNING: /usr/bin/time not found: host max RSS will be n/a (acceptable for a local dry run only)"
        else
            echo "run_sf38.sh: /usr/bin/time not found (host max RSS is required in a campaign)" >&2
            exit 2
        fi
    fi
    if [ "$DRY" = "1" ]; then
        echo "run_sf38.sh: SF38_DRYRUN=1 (tiny settings; NOT campaign evidence)"
        TRACE_THREADS=4
        PROD_N=16
        PROD_EXTRA=(--threads 4)
        PAR2_TMO=1800
        LONG_TMO=1800
    fi
    echo "run_sf38.sh: part=$ARG_PART build_dir=$BIN_DIR host=$(hostname) gpu_id=$GPU_ID out_root=$OUT_ROOT base_refs=$BASE_REFS repo=$REPO start=$(date -u +%FT%TZ)"
    "$fn"
    part_finish
    echo "run_sf38.sh: part=$ARG_PART end=$(date -u +%FT%TZ) blocking_failures=$BAD"
    [ "$BAD" -eq 0 ]
    exit $?
}

main
