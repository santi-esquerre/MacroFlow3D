#!/usr/bin/env bash
# SF-40 campaign runner (one part per detached V100 job).  Run from the synced mirror (the repository root is derived
# from this script's path); every part is restartable.
#
#   bash <A>/scripts/run_sf40.sh <bin_dir> baserefs   BASE binaries: reference outputs (RR4 A set, RR8 R set, SF-27 trio, Par2 x2)
#   bash <A>/scripts/run_sf40.sh <bin_dir> bytecmp    HEAD: LEAN and L6 sets (RR4) + RR8 default-path set vs the base references
#   bash <A>/scripts/run_sf40.sh <bin_dir> l6         HEAD: four tracker variants at N 256 (RR9), byte-compared with the default run
#   bash <A>/scripts/run_sf40.sh <bin_dir> controls   HEAD: --full test executables, driver smoke, sampler smoke, early-warning run
#   bash <A>/scripts/run_sf40.sh <bin_dir> mem        HEAD: RR5 (probes, 512^3 LEAN, 3328x384x384 x2) + closed_form/pid_peak verdicts
#   bash <A>/scripts/run_sf40.sh <bin_dir> par2       HEAD: Par2 config once vs base par2_a (RR7) + PID/device peaks
#   bash <A>/scripts/run_sf40.sh <bin_dir> final      HEAD: full ctest + SF-27 trio vs the base trio
#
# <A> = docs/experiments/artifacts/2026-10-22-sf40-lean-lagrangian-stage.  <bin_dir> holds `inlet_slab` and
# `macroflow3d_pipeline` (baserefs: the BASE build; other parts: build/v100-release of the audited head).  The test
# executables and ctest of `controls` / `final` are taken from $SF40_BUILD_DIR (default <bin_dir>).
#
# Environment (all optional):
#   SF40_OUT_ROOT   output root, NOT in the repository tree          (default $HOME/sf40_campaign; dry run $HOME/sf40_dryrun)
#   SF40_BASE_REFS  base-commit references (part baserefs)           (default $HOME/sf40_base_refs; dry run $OUT_ROOT/base_refs)
#   SF40_COMMIT     commit recorded in run.env                       (default <A>/COMMIT, else "unknown")
#   SF40_BUILD_DIR  directory of the test executables / ctest        (default <bin_dir>)
#   SF40_DRYRUN     1 = tiny settings, NOT campaign evidence         (default 0)
#   PYTHON          interpreter                                      (default python3)
#
# Per-part bookkeeping in $OUT_ROOT/<part>/ (baserefs: in $SF40_BASE_REFS itself):
#   exit_codes.tsv  EXACTLY two tab-separated columns `run<TAB>rc`, no header     walls.tsv  `run<TAB>wall_s<TAB>end_utc`
#   run.env         COMMIT, BIN + md5 of both binaries, host, GPU (CUDA_VISIBLE_DEVICES + nvidia-smi name), driver/CUDA
#   sha256.txt      sha256 of every output file of each finished run          resumed.txt  runs re-run after an interruption
#   results.txt     the verdict / comparison lines of the part (recomputed on every invocation from the finished runs)
# Run artifacts are <base>/ (output dir), <base>.json (summary), <base>.log (stdout+stderr), <base>.time (/usr/bin/time -v),
# <base>.free (free -b before), and for sampled runs <base>.apps.csv, <base>.dev.csv, <base>.pid_tree (RR5).
#
# Resume rules (SF-37 RR12): a run whose rc is recorded in exit_codes.tsv is finished and is never re-run (nothing is re-run
# because of its result); a run with partial outputs and NO recorded rc (infrastructure interruption) has them removed and is
# re-run with identical settings (listed in resumed.txt).  A SIGINT/SIGTERM records no rc.  Comparisons are recomputed.
# An exit code is a recorded outcome, not a retry trigger; unexpected codes make the part's exit status non-zero.
#
# SF40_DRYRUN=1 (tiny settings; NOT evidence; separate roots): seed 3001 only, N {16,32}, --np-side 16, planes/times
# 0.25,0.5,1,2; no N=512 cases; RR8 set reduced (trace N32 x eps 0.5/1.0 x host/gpu, prod32, 3 forward cases, 1 opt-in case);
# D runs replaced by a 64x16x16 lean --grid run; Par2 replaced by apps/config_pspta_small.yaml; test executables without
# --full; the early-warning and cufft-size runs shrunk.  Unset, nothing changes.
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
DRY=${SF40_DRYRUN:-0}
if [ "$DRY" = "1" ]; then
    OUT_ROOT=${SF40_OUT_ROOT:-$HOME/sf40_dryrun}
    BASE_REFS=${SF40_BASE_REFS:-$OUT_ROOT/base_refs}
else
    OUT_ROOT=${SF40_OUT_ROOT:-$HOME/sf40_campaign}
    BASE_REFS=${SF40_BASE_REFS:-$HOME/sf40_base_refs}
fi
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
GPU_ID=${GPU_ID%%,*}
BAD=0
BIN_DIR=""
BIN=""
PIPE=""
TEST_DIR=""
CHILD=""
SAMP_APPS_PID=""
SAMP_DEV_PID=""
PD=""
RES=""
BC_N=0
PART=""

# RR4 / RR8 / RR7 constants -------------------------------------------------------------------------------------------
SEEDS="3101 3102 3103"
NS="64 128 256"
NPSIDE=256
PLANES="0.25,0.5,1,2,4,8,16"
TIMES="0.25,0.5,1,2,4,8,16"
LEAN_FLAGS=(--stage lean --fused-splines --single-launch)
L6_FLAGS=(--fused-splines --single-launch)
SETUP_PREFIXES=("SETUP   SF-18:" "SETUP   SF-19:" "SETUP   SF-19 corrector")
# manifest.json keys dropped in the Par2 / trio comparison (RR7): the keys the pipeline writes with the time of the run
# (`timestamp`, src/io/writers/ManifestWriter.hpp: std::time(nullptr)) and the build identity (`build.git_hash`, the
# `git rev-parse --short HEAD` at configure time, differs between a base and a head build by construction).
MANIFEST_DROPS=(timestamp build.git_hash)
BC_DROPS=('**.timings_s')
PAR2_CFG=config_pipeline_par2
PAR2_TMO=14400
TRIO="config_pspta_small config_streamfunctions_homogeneous config_streamfunctions_continuation"
RC_FWD=" 0 18 22 "
RC_POL=" 0 18 22 23 "
RC_TRACE=" 0 17 19 21 "

# ============================================================================================================
# common helpers
# ============================================================================================================
usage() {
    echo "usage: run_sf40.sh <bin_dir> <part>   (parts: baserefs bytecmp l6 controls mem par2 final)" >&2
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
    echo "run_sf40.sh: interrupted (signal); stopping the running case (no rc is recorded; a re-invocation resumes it)"
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
        echo "run_sf40.sh: refusing to remove '$1'" >&2
        exit 2
        ;;
    esac
}

has_flag() { # has_flag <flags> <letter>
    case "$1" in *"$2"*) return 0 ;; esac
    return 1
}

write_env() { # write_env <pd> <part>   (appended as a new block when the file exists: a resumed invocation)
    local pd=$1 part=$2 f commit
    f=$pd/run.env
    commit=${SF40_COMMIT:-$(cat "$ART/COMMIT" 2> /dev/null || echo unknown)}
    [ -e "$f" ] && echo "# --- resumed/repeated invocation ---" >> "$f"
    {
        [ "$DRY" = "1" ] && echo "# DRYRUN=1: tiny settings, NOT campaign evidence"
        echo "COMMIT=$commit"
        echo "HOST=$(hostname)"
        echo "DATE_UTC=$(date -u +%FT%TZ)"
        echo "PART=$part"
        echo "BIN_DIR=$BIN_DIR"
        [ -x "$BIN" ] && echo "BIN=$BIN" && echo "BIN_MD5=$(md5sum "$BIN" | awk '{print $1}')"
        [ -x "$PIPE" ] && echo "PIPE=$PIPE" && echo "PIPE_MD5=$(md5sum "$PIPE" | awk '{print $1}')"
        echo "DRYRUN=$DRY"
        echo "OUT_ROOT=$OUT_ROOT"
        echo "BASE_REFS=$BASE_REFS"
        echo "REPO=$REPO"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        echo "GPU_NAME=$( (nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_ID" 2> /dev/null | head -1) || true)"
        echo "NVIDIA_DRIVER=$( (nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$GPU_ID" 2> /dev/null | head -1) || true)"
        echo "CUDA_VERSION_NVIDIA_SMI=$( (nvidia-smi 2> /dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1) || true)"
        echo "NVCC=$( (nvcc --version 2> /dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1) || true)"
    } >> "$f"
    echo "  env: $(grep -v '^#' "$f" | tail -n 14 | tr '\n' ' ')"
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

samplers_start() { # samplers_start <base>   (RR5: all processes + device-wide memory.used of the assigned GPU, 200 ms)
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
    if has_flag "$flags" X && command -v nvidia-smi > /dev/null 2>&1; then
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "$GPU_ID" > "$base.precheck" 2>&1
        if [ "$(grep -Ec '^[0-9]+,' "$base.precheck")" -gt 0 ]; then
            echo "GPU_BUSY run=$id: other compute process(es) hold GPU $GPU_ID (RR5); not run, no rc recorded:"
            sed 's/^/    /' "$base.precheck"
            BAD=$((BAD + 1))
            CASE_STATE=busy
            return 3
        fi
        echo "precheck: no compute process on GPU $GPU_ID" >> "$base.precheck"
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
    grep -E '^(STATUS|PSTATUS)' "$base.log" 2> /dev/null | cut -c1-200 | sed 's/^/    /'
    rc_expected_check "$id" "$rc" "$expected"
    return 0
}

part_begin() { # part_begin <part>: sets PD/RES, bookkeeping, results.txt rebuilt for compare/verdict parts
    PART=$1
    if [ "$PART" = "baserefs" ]; then
        PD=$BASE_REFS
    else
        PD=$OUT_ROOT/$PART
    fi
    mkdir -p "$PD"
    RES=$PD/results.txt
    : > "$RES"
    [ "$DRY" = "1" ] && echo "# DRYRUN=1: tiny settings, NOT campaign evidence" >> "$RES"
    BC_N=0
    write_env "$PD" "$PART"
}

# ============================================================================================================
# comparison helpers (SF-37 pattern; RES = results file of the part)
# ============================================================================================================
res() { # res <line...>: a result line (verdict / recorded), echoed and appended to results.txt
    echo "$*"
    echo "$*" >> "$RES"
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

bc_setup() { # bc_setup <item> <head_log> <ref_log>   (LEAN: the three SETUP prefixes only, setup_lines_cmp.py)
    local item=$1 a=$2 b=$3 out rc
    if [ ! -e "$a" ] || [ ! -e "$b" ]; then
        bc_res MISSING "$item (head: $a ref: $b)"
        return 0
    fi
    out=$("$PY" "$SCRIPTS/setup_lines_cmp.py" "$a" "$b" "${SETUP_PREFIXES[@]}")
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

# bc_tree <item> <head_dir> <ref_dir>: RR7 rule: every file cmp-identical, no extra file, except manifest.json compared by
# json_strip_cmp.py dropping the explicit MANIFEST_DROPS key list (no line-regex masking).
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

rec_git_hash() { # rec_git_hash <label> <dir>: RECORDED build.git_hash of the manifest under <dir>
    local m
    m=$(find "$2" -name manifest.json 2> /dev/null | head -n 1)
    res "RECORDED manifest_git_hash $1: $( [ -n "$m" ] && sed -n 's/.*"git_hash": *"\([^"]*\)".*/\1/p' "$m" | head -n 1 || echo n/a)"
}

# ============================================================================================================
# case tables
# ============================================================================================================
settings() {
    if [ "$DRY" = "1" ]; then
        SEEDS="3001"
        NS="16 32"
        NPSIDE=16
        PLANES="0.25,0.5,1,2"
        TIMES="0.25,0.5,1,2"
        PAR2_CFG=config_pspta_small
        PAR2_TMO=1800
    fi
}

a_cases() { # RR4: lines `name N eps seed extra`
    local ed elab eps s n
    for ed in 10:1 15:1.5 20:2; do
        elab=${ed%%:*}
        eps=${ed#*:}
        for s in $SEEDS; do
            for n in $NS; do
                echo "N${n}_e${elab}_s${s} $n $eps $s -"
            done
        done
    done
    if [ "$DRY" = "1" ]; then
        echo "cov 16 1 3001 --covectors"
        echo "pol 16 2 3001 --comparator pollock"
    else
        echo "N512_e10_s3102 512 1 3102 -"
        echo "N512_e20_s3102 512 2 3102 -"
        echo "cov 64 1 3101 --covectors"
        echo "pol 64 2 3102 --comparator pollock"
    fi
}

# a_args <N> <eps> <seed> <base> <extra | ->: A_ARGS = the RR4 common arguments (summary outside the particles directory)
A_ARGS=()
a_args() {
    A_ARGS=(--track forward --sigma2 1 --ell 0.25 --seed "$3" --n "$1" --eps "$2" --np-side "$NPSIDE" --planes "$PLANES"
        --times "$TIMES" --tol 1e-8 --hmax-div 8 --max-arclength 1000 --save-particles "$4/" --summary "$4.json")
    if [ "$5" != "-" ]; then
        # the extra field may hold several words (`--comparator pollock`): one argv element per word (bash 4.2-safe on empty arrays)
        local ex=()
        read -r -a ex <<< "$5"
        A_ARGS+=(${ex[@]+"${ex[@]}"})
    fi
}

r_items() { # RR8 (= SF-37 RR13): lines `kind name arg1 arg2 arg3`
    local N e t c
    if [ "$DRY" = "1" ]; then
        for e in 0.5 1.0; do
            for t in host gpu; do
                echo "trace N32_e${e}_${t} 32 $e $t"
            done
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
    trace) R_ARGS=(--construct trace --sigma2 1 --ell 0.25 --seed 3001 --threads 32 --n "$3" --eps "$4" --tracer "$5"
        --save-labels "$base" --summary "$base.json") ;;
    prod) R_ARGS=(--production --n 32 --eps 0.25 --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder) ;;
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

# ============================================================================================================
# part baserefs (BASE binaries)
# ============================================================================================================
par2_ref_run() { # par2_ref_run <id>
    case_run "$BASE_REFS" "$1" "$BASE_REFS/$1" " 0 " S "$BASE_REFS/$1" "$PAR2_TMO" -- "$PIPE" "$REPO/apps/$PAR2_CFG.yaml"
}

part_baserefs() {
    part_begin baserefs
    mkdir -p "$BASE_REFS/bin"
    if [ "$BIN_DIR" != "$BASE_REFS/bin" ]; then
        cp -f "$BIN" "$PIPE" "$BASE_REFS/bin/"
    fi
    echo "  base binaries copied to $BASE_REFS/bin: $(md5sum "$BASE_REFS/bin/inlet_slab" "$BASE_REFS/bin/macroflow3d_pipeline" | tr '\n' ' ')"
    local name n e s extra
    echo "== RR4 A set (31 cases, default stage) -> $BASE_REFS/a"
    while read -r name n e s extra; do
        a_args "$n" "$e" "$s" "$BASE_REFS/a/$name" "$extra"
        case_run "$BASE_REFS" "a_$name" "$BASE_REFS/a/$name" "$RC_POL" - - 0 -- "$BIN" "${A_ARGS[@]}"
    done < <(a_cases)
    echo "== RR8 default-path set -> $BASE_REFS/r"
    local kind a1 a2 a3 fam
    while read -r kind name a1 a2 a3; do
        fam=$kind
        r_args "$kind" "$name" "$a1" "$a2" "$a3" "$BASE_REFS/r/$kind/$name" || { echo "no argument set for $kind $name"; BAD=$((BAD + 1)); continue; }
        if [ "$kind" = "prod" ]; then
            case_run "$BASE_REFS" "r_$kind" "$BASE_REFS/r/prod32" " 0 " - - 0 -- "$BIN" "${R_ARGS[@]}"
        else
            case_run "$BASE_REFS" "r_${kind}_$name" "$BASE_REFS/r/$kind/$name" "$(r_expected "$fam")" - - 0 -- "$BIN" "${R_ARGS[@]}"
        fi
    done < <(r_items)
    echo "== SF-27 trio -> $BASE_REFS/trio/<cfg> (each from its own empty working directory)"
    local cfg
    for cfg in $TRIO; do
        case_run "$BASE_REFS" "trio_$cfg" "$BASE_REFS/trio/$cfg" " 0 " - "$BASE_REFS/trio/$cfg" 0 -- "$PIPE" "$REPO/apps/$cfg.yaml"
    done
    echo "== Par2 config twice (determinism control; $PAR2_CFG)"
    par2_ref_run par2_a
    par2_ref_run par2_b
    res "# base-reference bookkeeping above; determinism control par2_a vs par2_b (RR7 rule):"
    BC_N=0
    bc_tree "determinism:par2_b_vs_par2_a" "$BASE_REFS/par2_b" "$BASE_REFS/par2_a"
    bc_rc_check "determinism:par2" "$(tsv_rc "$BASE_REFS/exit_codes.tsv" par2_b)" "$BASE_REFS/par2_a.time"
    rec_git_hash par2_a "$BASE_REFS/par2_a"
    if [ "$BC_N" -eq 0 ]; then
        res "BASEREFS_PAR2_DETERMINISTIC (base vs base identical under the RR7 rule)"
    else
        res "BASEREFS_PAR2_NONDETERMINISTIC n=$BC_N (G is NOT met on those files: BLOCKING per RR7, owner only may waive)"
        BAD=$((BAD + 1))
    fi
    res "RECORDED par2_wall_s a=$(tsv_col "$BASE_REFS/walls.tsv" par2_a 2) b=$(tsv_col "$BASE_REFS/walls.tsv" par2_b 2)"
    pid_peaks par2_a "$BASE_REFS/par2_a"
    pid_peaks par2_b "$BASE_REFS/par2_b"
}

pid_peaks() { # pid_peaks <label> <base>: RECORDED PID/device peaks of a sampled run (pid from the pgrep tree)
    local tree
    tree=$(cat "$2.pid_tree" 2> /dev/null || echo none)
    if [ ! -s "$2.apps.csv" ] || [ ! -s "$2.dev.csv" ] || [ "$tree" = "none" ]; then
        res "RECORDED PAR2_PEAK $1: not sampled (apps=$( [ -s "$2.apps.csv" ] && echo yes || echo no) dev=$( [ -s "$2.dev.csv" ] && echo yes || echo no) pid_tree=$tree)"
        return 0
    fi
    res "RECORDED PAR2_PEAK $1: $("$PY" "$SCRIPTS/pid_peak.py" --apps "$2.apps.csv" --dev "$2.dev.csv" --pid "$tree" --tree-pid "$tree")"
}

# ============================================================================================================
# part bytecmp (HEAD binaries)
# ============================================================================================================
bc_refs_present() {
    [ -d "$BASE_REFS/a" ] && [ -d "$BASE_REFS/r" ] && [ -x "$BASE_REFS/bin/inlet_slab" ]
}

# bc_a_case <set lean|l6> <name> <N> <eps> <seed> <extra>
bc_a_case() {
    local set=$1 name=$2 id dir ref toks
    id=${set}_$name
    dir=$PD/$set/$name
    ref=$BASE_REFS/a/$name
    a_args "$3" "$4" "$5" "$dir" "$6"
    if [ "$set" = "lean" ]; then
        case_run "$PD" "$id" "$dir" any - - 0 -- "$BIN" "${A_ARGS[@]}" "${LEAN_FLAGS[@]}"
    else
        case_run "$PD" "$id" "$dir" any - - 0 -- "$BIN" "${A_ARGS[@]}" "${L6_FLAGS[@]}"
    fi
    [ "$CASE_STATE" = "busy" ] && return 0
    BC_DROPS=('**.timings_s')
    bc_dir_files "$id" "$dir" "$ref" 1
    toks="FIELD TRACK BACKFLOW TPLANE TTIME TSTATUS STATUS"
    [ "$name" = "pol" ] && toks="$toks PTRACK PPLANE PTIME PSTATUS PFAIL"
    if [ "$set" = "lean" ]; then
        bc_lines "$id:lines" "$dir.log" "$ref.log" $toks
        bc_setup "$id:setup_lines" "$dir.log" "$ref.log"
    else
        bc_lines "$id:lines" "$dir.log" "$ref.log" SETUP $toks
    fi
    bc_rc_check "$id" "$CASE_RC" "$ref.time"
}

bc_r_case() { # bc_r_case <kind> <name> <a1> <a2> <a3>
    local kind=$1 name=$2 id dir ref toks
    if [ "$kind" = "prod" ]; then
        id=r_prod
        dir=$PD/r/prod32
        ref=$BASE_REFS/r/prod32
        r_args prod "$name" "$3" "$4" "$5" "$dir"
        case_run "$PD" "$id" "$dir" any - - 0 -- "$BIN" "${R_ARGS[@]}"
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
    case_run "$PD" "$id" "$dir" any - - 0 -- "$BIN" "${R_ARGS[@]}"
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

part_bytecmp() {
    part_begin bytecmp
    if ! bc_refs_present; then
        if [ "$DRY" = "1" ]; then
            bc_res SKIP "base_refs $BASE_REFS absent or incomplete (dry run: nothing compared)"
            res "BYTECMP_SKIPPED"
            return 0
        fi
        bc_res MISSING "base_refs $BASE_REFS (need a/, r/, bin/inlet_slab): criteria A and R cannot be evaluated"
        res "BYTECMP_DIFF n=$BC_N"
        BAD=$((BAD + 1))
        return 0
    fi
    local name n e s extra set
    for set in lean l6; do
        echo "== RR4 set $set"
        while read -r name n e s extra; do
            bc_a_case "$set" "$name" "$n" "$e" "$s" "$extra"
        done < <(a_cases)
    done
    echo "== RR8 default-path set"
    local kind a1 a2 a3
    while read -r kind name a1 a2 a3; do
        bc_r_case "$kind" "$name" "$a1" "$a2" "$a3"
    done < <(r_items)
    local hmd5 bmd5
    hmd5=$(md5sum "$BIN" | awk '{print $1}')
    bmd5=$(md5sum "$BASE_REFS/bin/inlet_slab" | awk '{print $1}')
    if [ "$hmd5" != "$bmd5" ]; then
        bc_res IDENTICAL "binary:md5_differs head=$hmd5 base=$bmd5"
    else
        bc_res DIFF "binary:md5_differs (head and base binaries are the SAME file content $hmd5)"
    fi
    if [ "$BC_N" -eq 0 ]; then
        res "BYTECMP_ALL_IDENTICAL"
    else
        res "BYTECMP_DIFF n=$BC_N"
        BAD=$((BAD + 1))
    fi
}

# ============================================================================================================
# part l6 (HEAD, current stage)
# ============================================================================================================
part_l6() {
    part_begin l6
    local n=256 s=3102 v vf
    if [ "$DRY" = "1" ]; then
        n=32
        s=3001
    fi
    for v in default fused single both; do
        case "$v" in
        default) vf="" ;;
        fused) vf="--fused-splines" ;;
        single) vf="--single-launch" ;;
        both) vf="--fused-splines --single-launch" ;;
        esac
        a_args "$n" 2 "$s" "$PD/$v" -
        # shellcheck disable=SC2086
        case_run "$PD" "l6_$v" "$PD/$v" any - - 0 -- "$BIN" "${A_ARGS[@]}" $vf
    done
    "$PY" "$SCRIPTS/l6_timings.py" "$PD" > "$PD/l6_timings.tsv"
    res "RECORDED l6_timings.tsv written ($PD/l6_timings.tsv)"
    grep '^RECORDED' "$PD/l6_timings.tsv" | tee -a "$RES"
    BC_DROPS=('**.timings_s')
    for v in fused single both; do
        bc_dir_files "l6:$v-vs-default" "$PD/$v" "$PD/default" 1
        bc_lines "l6:$v-vs-default:lines" "$PD/$v.log" "$PD/default.log" SETUP FIELD TRACK BACKFLOW TPLANE TTIME TSTATUS STATUS
        bc_rc_check "l6:$v-vs-default" "$(tsv_rc "$PD/exit_codes.tsv" "l6_$v")" "$PD/default.time"
    done
    if [ "$BC_N" -eq 0 ]; then
        res "L6_VARIANTS_ALL_IDENTICAL"
    else
        res "L6_VARIANTS_DIFF n=$BC_N"
        BAD=$((BAD + 1))
    fi
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

ctl_lean_closed_form() { # ctl_lean_closed_form <id> <label>
    local out geoms phs
    geoms=$(grep -c '^MEM_GEOM ' "$PD/$1.log" 2> /dev/null || true)
    phs=$(grep -c '^MEM_PHASE ' "$PD/$1.log" 2> /dev/null || true)
    geoms=${geoms:-0}
    phs=${phs:-0}
    out=$("$PY" "$SCRIPTS/closed_form.py" "$PD/$1.log" --run "$1")
    local rc=$?
    echo "$out" | grep -E '^C_(VERDICT|ITEM|LARGEST|RUN)' | tee -a "$RES"
    local v=FAIL
    [ "$rc" -eq 0 ] && [ "$geoms" -ge 1 ] && [ "$phs" -eq 5 ] && v=PASS
    res "CONTROL_VERDICT run=$1 $2 mem_geom_lines=$geoms mem_phase_lines=$phs closed_form_exit=$rc $v"
    [ "$v" = "PASS" ] || BAD=$((BAD + 1))
}

part_controls() {
    part_begin controls
    local full="--full" cuf="3328,384,384" t
    if [ "$DRY" = "1" ]; then
        full=""
        cuf="128,16,16"
        res "# DRY: test executables run without --full, cufft-size shrunk"
    fi
    for t in long_box_tests:long_box_full projected_pcg_external_tests:projected_pcg_external_full \
        affine_periodic_flow_lean_tests:affine_periodic_flow_lean_full; do
        if [ ! -x "$TEST_DIR/${t%%:*}" ]; then
            res "CONTROL_VERDICT run=${t#*:} exit=- MISSING_EXECUTABLE $TEST_DIR/${t%%:*} FAIL"
            BAD=$((BAD + 1))
            continue
        fi
        # shellcheck disable=SC2086
        case_run "$PD" "${t#*:}" "$PD/${t#*:}" any - - 0 -- "$TEST_DIR/${t%%:*}" $full
        ctl_exe_verdict "${t#*:}"
    done
    if [ -x "$TEST_DIR/long_box_tests" ]; then
        case_run "$PD" long_box_cufft3328 "$PD/long_box_cufft3328" any - - 0 -- "$TEST_DIR/long_box_tests" --cufft-size "$cuf"
        ctl_exe_verdict long_box_cufft3328
    fi
    # driver smoke: lean stage on 128x16x16 with a zeros cells file (k = 1)
    local zeros=$PD/zeros_128x16x16.npy
    "$PY" "$SCRIPTS/make_zeros_npy.py" 128 16 16 "$zeros"
    case_run "$PD" smoke_lean "$PD/smoke_lean" any - - 0 -- "$BIN" --track forward --stage lean --grid 128,16,16 --cells "$zeros" \
        --eps 1 --np-side 8 --planes 0.25,0.5,1 --times 0.5,1 --tol 1e-8 --hmax-div 8 --max-arclength 1000 \
        --save-particles "$PD/smoke_lean/" --summary "$PD/smoke_lean.json"
    [ "$CASE_STATE" = "busy" ] || ctl_lean_closed_form smoke_lean "driver_smoke(128x16x16,zeros)"
    # sampler smoke: PID filtering must find samples (probe 5 s; small lean run)
    case_run "$PD" sampler_probe "$PD/sampler_probe" any SX - 0 -- "$BIN" --context-probe 5
    sampler_verdict sampler_probe
    local grid="256,32,32"
    case_run "$PD" sampler_lean "$PD/sampler_lean" any SX - 0 -- "$BIN" --track forward --stage lean --fused-splines --single-launch \
        --grid "$grid" --sigma2 1 --ell 0.25 --seed 3001 --eps 1 --np-side 16 --planes 0.5,1 --tol 1e-8 --hmax-div 8 \
        --max-arclength 1000 --summary "$PD/sampler_lean.json"
    sampler_verdict sampler_lean
    # early-warning lean run (RR9; recorded)
    local ewgrid="1664,192,192" ewn=10
    if [ "$DRY" = "1" ]; then
        ewgrid="256,32,32"
        ewn=8
    fi
    case_run "$PD" early_warning "$PD/early_warning" any SX - 0 -- "$BIN" --track forward --stage lean --fused-splines --single-launch \
        --grid "$ewgrid" --sigma2 1 --eps 2 --ell 0.08333333333333333 --seed 4001 --np-side "$ewn" --planes 0.5 \
        --max-arclength 1000 --summary "$PD/early_warning.json"
    res "RECORDED early_warning grid=$ewgrid exit=$(tsv_rc "$PD/exit_codes.tsv" early_warning)"
    grep -E '^SETUP {3}SF-19|^MEM_GEOM|^MEM_PHASE' "$PD/early_warning.log" 2> /dev/null | cut -c1-400 | sed 's/^/RECORDED early_warning: /' | tee -a "$RES"
    if [ "$(tsv_rc "$PD/exit_codes.tsv" early_warning)" = "0" ]; then
        "$PY" "$SCRIPTS/closed_form.py" "$PD/early_warning.log" --run early_warning | grep -E '^C_(VERDICT|RUN)' | sed 's/^/RECORDED /' | tee -a "$RES"
    fi
}

sampler_verdict() { # sampler_verdict <id>
    local out rc tree
    tree=$(cat "$PD/$1.pid_tree" 2> /dev/null || echo none)
    out=$("$PY" "$SCRIPTS/pid_peak.py" --apps "$PD/$1.apps.csv" --dev "$PD/$1.dev.csv" --log "$PD/$1.log" --tree-pid "$tree")
    rc=$?
    res "SAMPLER_SMOKE run=$1 $out $( [ "$rc" -eq 0 ] && echo OK || echo NOT_EVALUABLE)"
    [ "$rc" -eq 0 ] || BAD=$((BAD + 1))
}

# ============================================================================================================
# part mem (HEAD): RR5
# ============================================================================================================
D3328_PLANES_I="0.041666666666666664,0.08333333333333333,0.16666666666666666,0.3333333333333333,0.6666666666666666,1.3333333333333333,2.6666666666666665,5.333333333333333,8.666666666666666"
D3328_PLANES_II="0.08333333333333333,0.16666666666666666,0.3333333333333333,0.6666666666666666,1.3333333333333333,2.6666666666666665,5.333333333333333,8.666666666666666"

# d_run <id> <kind: 512 | 512dev | 3328i | 3328ii>: RR5 arguments (dry: 64x16x16 lean --grid)
D_ARGS=()
d_args() {
    local id=$1 kind=$2
    if [ "$DRY" = "1" ]; then
        local eps=1.5
        [ "$kind" = "3328ii" ] && eps=2
        D_ARGS=(--track forward --stage lean --fused-splines --single-launch --grid 64,16,16 --sigma2 1 --seed 3001 --np-side 8
            --tol 1e-8 --hmax-div 8 --max-arclength 1000 --eps "$eps" --ell 0.25 --planes 0.25,0.5,1,2 --summary "$PD/$id.json")
        [ "$kind" = "512dev" ] && D_ARGS+=(--bhat device)
        return 0
    fi
    case "$kind" in
    512 | 512dev)
        # RR4 args of N 512, E 2, S 3102 + LEAN flags
        D_ARGS=(--track forward --sigma2 1 --ell 0.25 --seed 3102 --n 512 --eps 2 --np-side 256 --planes "0.25,0.5,1,2,4,8,16"
            --times "0.25,0.5,1,2,4,8,16" --tol 1e-8 --hmax-div 8 --max-arclength 1000 --save-particles "$PD/$id/"
            --summary "$PD/$id.json" "${LEAN_FLAGS[@]}")
        [ "$kind" = "512dev" ] && D_ARGS+=(--bhat device)
        ;;
    3328i)
        D_ARGS=(--track forward --stage lean --fused-splines --single-launch --grid 3328,384,384 --sigma2 1 --seed 4001
            --np-side 100 --tol 1e-8 --hmax-div 8 --max-arclength 1000 --eps 1.5 --ell 0.041666666666666664
            --planes "$D3328_PLANES_I" --summary "$PD/$id.json")
        ;;
    3328ii)
        D_ARGS=(--track forward --stage lean --fused-splines --single-launch --grid 3328,384,384 --sigma2 1 --seed 4001
            --np-side 100 --tol 1e-8 --hmax-div 8 --max-arclength 1000 --eps 2 --ell 0.08333333333333333
            --planes "$D3328_PLANES_II" --summary "$PD/$id.json")
        ;;
    esac
    local a
    case "$kind" in
    3328*)
        for a in "${D_ARGS[@]}"; do # RR2: never --bhat device at 3328x384x384 (level S exceeds the card)
            if [ "$a" = "--bhat" ]; then
                echo "run_sf40.sh: refusing --bhat at 3328x384x384" >&2
                exit 2
            fi
        done
        ;;
    esac
}

part_mem() {
    part_begin mem
    local probe_s=10
    [ "$DRY" = "1" ] && probe_s=3
    local k
    for k in 1 2 3; do
        case_run "$PD" "probe$k" "$PD/probe$k" any SX - 0 -- "$BIN" --context-probe "$probe_s"
    done
    case_run "$PD" probe_warm "$PD/probe_warm" any SX - 0 -- "$BIN" --context-probe "$probe_s" --warm
    if [ "$(tsv_rc "$PD/exit_codes.tsv" probe_warm)" != "0" ]; then
        res "RECORDED warm probe unavailable (the binary rejected --warm: exit $(tsv_rc "$PD/exit_codes.tsv" probe_warm))"
    fi
    local pair id kind
    for pair in d512:512 d512_bhatdev:512dev d3328_i:3328i d3328_ii:3328ii; do
        id=${pair%%:*}
        kind=${pair#*:}
        d_args "$id" "$kind"
        case_run "$PD" "$id" "$PD/$id" any SX - 0 -- "$BIN" "${D_ARGS[@]}"
    done
    local dryarg=""
    [ "$DRY" = "1" ] && dryarg="--dry"
    # shellcheck disable=SC2086
    "$PY" "$SCRIPTS/mem_verdict.py" "$PD" $dryarg | tee -a "$RES"
}

# ============================================================================================================
# part par2 (HEAD)
# ============================================================================================================
part_par2() {
    part_begin par2
    case_run "$PD" par2 "$PD/par2" any S "$PD/par2" "$PAR2_TMO" -- "$PIPE" "$REPO/apps/$PAR2_CFG.yaml"
    [ "$CASE_STATE" = "busy" ] && return 0
    if [ ! -d "$BASE_REFS/par2_a" ]; then
        bc_res MISSING "base reference $BASE_REFS/par2_a (part baserefs not run)"
        res "PAR2_DIFF n=$BC_N"
        BAD=$((BAD + 1))
        return 0
    fi
    bc_tree "par2:head_vs_base_par2_a" "$PD/par2" "$BASE_REFS/par2_a"
    bc_rc_check "par2" "$CASE_RC" "$BASE_REFS/par2_a.time"
    rec_git_hash base "$BASE_REFS/par2_a"
    rec_git_hash head "$PD/par2"
    res "RECORDED par2_wall_s head=$CASE_WALL base_a=$(tsv_col "$BASE_REFS/walls.tsv" par2_a 2) base_b=$(tsv_col "$BASE_REFS/walls.tsv" par2_b 2)"
    pid_peaks base_par2_a "$BASE_REFS/par2_a"
    pid_peaks head_par2 "$PD/par2"
    if [ "$BC_N" -eq 0 ]; then
        res "PAR2_ALL_IDENTICAL"
    else
        res "PAR2_DIFF n=$BC_N"
        BAD=$((BAD + 1))
    fi
}

# ============================================================================================================
# part final (HEAD build dir): full ctest + SF-27 trio
# ============================================================================================================
part_final() {
    part_begin final
    res "RECORDED binaries md5: $(md5sum "$PIPE" "$BIN" 2> /dev/null | tr '\n' ' ')"
    case_run "$PD" ctest "$PD/ctest" any - - 0 -- ctest --test-dir "$TEST_DIR" --output-on-failure
    local crc
    crc=$(tsv_rc "$PD/exit_codes.tsv" ctest)
    res "CTEST_EXIT=$crc CTEST_WALL_S=$(tsv_col "$PD/walls.tsv" ctest 2)"
    [ "$crc" = "0" ] || BAD=$((BAD + 1))
    local cfg
    BC_N=0
    for cfg in $TRIO; do
        case_run "$PD" "trio_$cfg" "$PD/trio/$cfg" any - "$PD/trio/$cfg" 0 -- "$PIPE" "$REPO/apps/$cfg.yaml"
        bc_tree "trio:$cfg" "$PD/trio/$cfg" "$BASE_REFS/trio/$cfg"
        bc_rc_check "trio:$cfg" "$CASE_RC" "$BASE_REFS/trio/$cfg.time"
    done
    if [ "$BC_N" -eq 0 ]; then
        res "TRIO_ALL_IDENTICAL"
    else
        res "TRIO_DIFF n=$BC_N"
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
    local fn need
    case "$ARG_PART" in
    baserefs | bytecmp | l6 | controls | mem | par2 | final) fn=part_$ARG_PART ;;
    *)
        echo "run_sf40.sh: unknown part '$ARG_PART'" >&2
        usage
        exit 2
        ;;
    esac
    BIN_DIR=$(cd "$ARG_BIN_DIR" 2> /dev/null && pwd) || {
        echo "run_sf40.sh: bin_dir '$ARG_BIN_DIR' not found" >&2
        exit 2
    }
    BIN=$BIN_DIR/inlet_slab
    PIPE=$BIN_DIR/macroflow3d_pipeline
    TEST_DIR=${SF40_BUILD_DIR:-$BIN_DIR}
    case "$ARG_PART" in
    baserefs) need="$BIN $PIPE" ;;
    par2 | final) need="$PIPE" ;;
    *) need="$BIN" ;;
    esac
    for f in $need; do
        [ -x "$f" ] || { echo "run_sf40.sh: $f not found or not executable" >&2; exit 2; }
    done
    mkdir -p "$OUT_ROOT" && OUT_ROOT=$(cd "$OUT_ROOT" && pwd)
    if [ -d "$BASE_REFS" ]; then
        BASE_REFS=$(cd "$BASE_REFS" && pwd)
    elif [ "$ARG_PART" = "baserefs" ]; then
        mkdir -p "$BASE_REFS" && BASE_REFS=$(cd "$BASE_REFS" && pwd)
    fi
    settings
    if [ ! -x /usr/bin/time ]; then
        if [ "$DRY" = "1" ]; then
            echo "WARNING: /usr/bin/time not found: host max RSS will be n/a (acceptable for a local dry run only)"
        else
            echo "run_sf40.sh: /usr/bin/time not found (host max RSS is required in a campaign)" >&2
            exit 2
        fi
    fi
    [ "$DRY" = "1" ] && echo "run_sf40.sh: SF40_DRYRUN=1 (tiny settings; NOT campaign evidence)"
    echo "run_sf40.sh: part=$ARG_PART bin_dir=$BIN_DIR host=$(hostname) gpu_id=$GPU_ID out_root=$OUT_ROOT base_refs=$BASE_REFS repo=$REPO start=$(date -u +%FT%TZ)"
    "$fn"
    echo "run_sf40.sh: part=$ARG_PART end=$(date -u +%FT%TZ) blocking_failures=$BAD"
    [ "$BAD" -eq 0 ]
    exit $?
}

main
