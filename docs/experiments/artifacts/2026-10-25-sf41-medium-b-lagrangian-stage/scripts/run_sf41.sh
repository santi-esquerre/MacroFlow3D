#!/usr/bin/env bash
# SF-41 campaign runner (one part per detached V100 job).  Run from the synced mirror (the repository root is derived from this
# script's path); every part is restartable.  Modelled on the SF-38 runner (run_sf38.sh): same helpers (timed, kill_tree, RR2
# samplers, case_run, bc_* byte-compare helpers) and resume rules.  Pre-registration: RR0-RR12 of
# docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/preregistration.md (activation commit fa3f253).
#
#   bash <A>/scripts/run_sf41.sh <build_dir> build       HEAD: CLEAN build (rm -rf build dir, cmake --preset, cmake --build), echo tree id, ctest -N,
#                                                        md5 of every binary -> $SF41_OUT_ROOT/build/binaries.md5 (checked by every other head part)
#   bash <A>/scripts/run_sf41.sh <build_dir> baserefs    BASE binaries ($HOME/sf41_base_refs/bin): SF-27 trio, SF-38 20-run R set, G(ii) runner run, G(iii) A/B run
#   bash <A>/scripts/run_sf41.sh <build_dir> controls    HEAD: A (--full, memcheck of the domain group), C1-C4, C5, D (RR2 sampler), memory-model records
#   bash <A>/scripts/run_sf41.sh <build_dir> cross       HEAD: B (RR4)
#   bash <A>/scripts/run_sf41.sh <build_dir> prodgrid    HEAD: E1-E4 (RR7) with the RR2 sampler; mapped-host re-run iff the PID peak exceeds 30000 MiB
#   bash <A>/scripts/run_sf41.sh <build_dir> evladder    HEAD: F (RR8), 24 runs
#   bash <A>/scripts/run_sf41.sh <build_dir> abrerun     HEAD: H (RR10), the two SF-38 A/B commands
#   bash <A>/scripts/run_sf41.sh <build_dir> final       HEAD: G (RR9): full ctest, SF-27 trio, 20-run R set, G(ii), G(iii) against the base references
#
# <A> = docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage.  <build_dir> = the head build directory
# (build/v100-release; baserefs: the directory of the BASE binaries, $HOME/sf41_base_refs/bin).
#
# Environment (all optional):
#   SF41_OUT_ROOT   output root, NOT in the repository tree           (default $HOME/sf41_campaign; dry run $HOME/sf41_dryrun)
#   SF41_BASE_REFS  base-commit references                            (default $HOME/sf41_base_refs; dry run $OUT_ROOT/base_refs)
#   SF41_COMMIT     commit recorded in run.env                        (default "unknown")
#   SF41_PRESET     cmake preset of the `build` part                  (default basename of <build_dir>, i.e. v100-release)
#   SF41_DRYRUN     1 = tiny settings, NOT campaign evidence          (default 0)
#   SF41_DRY_FORCE_MAPPED 1 = (dry run only) run the mapped-host fallback of E1 even though the PID peak is small (exercises the rule)
#   SF41_CTEST_ARGS extra arguments of the `final` ctest              (default empty; dry run: -R ^interpolation_nonperiodic_x$)
#   PYTHON          interpreter                                       (default python3)
#
# Per-part bookkeeping in $OUT_ROOT/<part>/ (baserefs: in $SF41_BASE_REFS itself):
#   run.env         DATE_UTC, host, GPU, driver, nvcc, gcc, binary md5s, SF41_COMMIT, tree echo, binaries.md5 check, nvidia-smi banner
#   exit_codes.tsv  EXACTLY two tab-separated columns `run<TAB>rc`, no header    walls.tsv  `run<TAB>wall_s<TAB>end_utc`
#   results.txt     the verdict / comparison / RECORDED lines of the part (recomputed on every invocation from the finished runs), ending with
#                   `RECORDED part_seconds=...` and `PART_DONE part=<p> rc=<0|1>`
#   sha256.txt      sha256 of every output file of each finished run (+ derived files)     resumed.txt  runs re-run after an interruption
#   PART_EXIT       0 iff the part ended with no blocking item (absent when the invocation was interrupted by a signal)
# Run artifacts are <base>/ (output dir), <base>.log (stdout+stderr), <base>.time (/usr/bin/time -v), <base>.free (free -b before),
# <base>.precheck (no compute process on the GPU, before every sampled/exclusive run) and, for sampled runs, <base>.apps.csv,
# <base>.dev.csv, <base>.pid_tree (RR2).
#
# Binaries (orchestrator rule, stale-object hazard of the shared mirror): the head build is a CLEAN build made by the `build` part, which records
# the md5 of every executable of <build_dir> in $OUT_ROOT/build/binaries.md5.  Every other head part REFUSES TO RUN (exit 2, PART_DONE rc=1) if
# that file is absent or any listed binary differs; the verification is echoed into run.env.  The base binaries in $BASE_REFS/bin come from a
# fresh mirror and are used as they are (their md5 is recorded).
#
# Resume rules (SF-37 RR12): a run whose rc is recorded in exit_codes.tsv is finished and is never re-run (nothing is re-run because of its
# result); a run with partial outputs and NO recorded rc (infrastructure interruption) has them removed and is re-run with identical settings
# (listed in resumed.txt).  A SIGINT/SIGTERM records no rc.  Comparisons are recomputed.  An exit code is a recorded outcome, not a retry
# trigger; unexpected codes make the part's exit status non-zero.  Every pre-registered run of a part is made whatever the intermediate
# verdicts (RR0: no outcome-dependent stopping); the only conditional run is the pre-authorized mapped-host fallback of E (RR7).
#
# SF41_DRYRUN=1 (tiny settings; NOT evidence; separate roots; the exclusive-GPU precheck is skipped because a workstation is shared):
# every grid <= 32^3-like, the fast tiers of the test programs, E on 128x16x16 / 64x16x16 boxes, F on 24x4x4 ... 96x16x16 boxes (3 rungs, 2 seeds),
# the H runs on 64x16x16, G(ii)/G(iii) on 64x16x16, the base references from the same binaries, a reduced R set.  Unset, nothing changes.
#
# bash 4.2 (the V100 host): no empty-array expansion under `set -u` (use ${a[@]+"${a[@]}"}), no ${var@Q}, no negative subscripts, no backslashes
# in `${var//pat/rep}` replacements.  `scripts/remote run` exports CUDA_VISIBLE_DEVICES.
set -uo pipefail

SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=$(dirname "$SCRIPTS")
REPO=$(cd "$ART/../../../.." && pwd)
ARG_BIN_DIR=${1:-}
ARG_PART=${2:-}
PY=${PYTHON:-python3}
DRY=${SF41_DRYRUN:-0}
if [ "$DRY" = "1" ]; then
    OUT_ROOT=${SF41_OUT_ROOT:-$HOME/sf41_dryrun}
    BASE_REFS=${SF41_BASE_REFS:-$OUT_ROOT/base_refs}
else
    OUT_ROOT=${SF41_OUT_ROOT:-$HOME/sf41_campaign}
    BASE_REFS=${SF41_BASE_REFS:-$HOME/sf41_base_refs}
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
BIN_CHECK=""

# constants ----------------------------------------------------------------------------------------------------------
SEED0=4001
# manifest.json keys dropped in the trio comparison (SF-40 RR7): the run time and the build identity (differs between a base and a head build).
MANIFEST_DROPS=(timestamp build.git_hash)
BC_DROPS=('**.timings_s')
TRIO="config_pspta_small config_streamfunctions_homogeneous config_streamfunctions_continuation"
RC_FWD=" 0 18 22 "
RC_POL=" 0 18 22 23 "
RC_TRACE=" 0 17 19 21 "
TRACE_THREADS=32
PROD_N=32
PROD_EXTRA=()
LONG_TMO=43200
MEM_LIMIT_MIB=30000

# ============================================================================================================
# common helpers (SF-40 run_sf40.sh)
# ============================================================================================================
usage() {
    echo "usage: run_sf41.sh <build_dir> <part>   (parts: build baserefs controls cross prodgrid evladder abrerun final)" >&2
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
    echo "run_sf41.sh: interrupted (signal); stopping the running case (no rc is recorded; a re-invocation resumes it)"
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
        echo "run_sf41.sh: refusing to remove '$1'" >&2
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

tree_id() { # tree echo of the synced mirror: git head/tree when .git exists, plus a digest of the sources (rsync carries no history guarantee)
    if git -C "$REPO" rev-parse --git-dir > /dev/null 2>&1; then
        echo "git_head=$(git -C "$REPO" rev-parse HEAD 2> /dev/null) git_tree=$(git -C "$REPO" rev-parse 'HEAD^{tree}' 2> /dev/null) git_dirty_files=$(git -C "$REPO" status --porcelain 2> /dev/null | wc -l | tr -d ' ')"
    else
        echo "git=absent"
    fi
    echo "src_digest=$( (cd "$REPO" && find src tests apps CMakeLists.txt CMakePresets.json -type f -not -path '*/external/petsc/*' -not -path '*/external/slepc/*' 2> /dev/null |
        LC_ALL=C sort | xargs -r sha256sum | sha256sum | cut -c1-64) )"
}

write_env() { # write_env <pd> <part>   (appended as a new block when the file exists: a resumed invocation)
    local pd=$1 part=$2 f b
    f=$pd/run.env
    [ -e "$f" ] && echo "# --- resumed/repeated invocation ---" >> "$f"
    {
        [ "$DRY" = "1" ] && echo "# DRYRUN=1: tiny settings, NOT campaign evidence"
        echo "SF41_COMMIT=${SF41_COMMIT:-unknown}"
        echo "DATE_UTC=$(date -u +%FT%TZ)"
        echo "HOST=$(hostname)"
        echo "PART=$part"
        echo "BIN_DIR=$BIN_DIR"
        if [ -d "$BIN_DIR" ]; then
            for b in interpolation_nonperiodic_x_tests sf41_cross_representation_tests dirichlet_controls_tests pollock_nonperiodic_tests \
                dirichlet_long_box_stage_tests lagrangian_ensemble lagrangian_medium_ab macroflow3d_pipeline inlet_slab; do
                [ -x "$BIN_DIR/$b" ] && echo "BIN_MD5_$b=$(md5sum "$BIN_DIR/$b" | awk '{print $1}')"
            done
        fi
        echo "BINARIES_CHECK=${BIN_CHECK:-not_applicable}"
        echo "DRYRUN=$DRY"
        echo "OUT_ROOT=$OUT_ROOT"
        echo "BASE_REFS=$BASE_REFS"
        echo "REPO=$REPO"
        tree_id | sed 's/^/TREE_/'
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        echo "GPU_NAME=$( (nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_ID" 2> /dev/null | head -1) || true)"
        echo "NVIDIA_DRIVER=$( (nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$GPU_ID" 2> /dev/null | head -1) || true)"
        echo "CUDA_VERSION_NVIDIA_SMI=$( (nvidia-smi 2> /dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1) || true)"
        echo "NVCC=$( (nvcc --version 2> /dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1) || true)"
        echo "GCC=$( (gcc --version 2> /dev/null | head -1) || true)"
        nvidia-smi 2> /dev/null | head -n 12 | sed 's/^/# nvidia-smi banner: /'
    } >> "$f"
    echo "  env: $(grep -v '^#' "$f" | tail -n 18 | tr '\n' ' ' | cut -c1-1100)"
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

check_binaries() { # check_binaries: BIN_CHECK = ok(<n> binaries, md5 file digest) | ABSENT | MISMATCH(<names>)
    local f=$OUT_ROOT/build/binaries.md5 n=0 bad="" md name
    if [ ! -s "$f" ]; then
        BIN_CHECK="ABSENT($f)"
        return 1
    fi
    while read -r md name; do
        [ -z "$name" ] && continue
        n=$((n + 1))
        if [ ! -x "$BIN_DIR/$name" ]; then
            bad="$bad $name:missing"
        elif [ "$(md5sum "$BIN_DIR/$name" | awk '{print $1}')" != "$md" ]; then
            bad="$bad $name:md5"
        fi
    done < "$f"
    if [ -n "$bad" ]; then
        BIN_CHECK="MISMATCH($bad)"
        return 1
    fi
    BIN_CHECK="ok($n binaries, md5file_sha256=$(sha256sum "$f" | cut -c1-16))"
    return 0
}

part_begin() { # part_begin <part>: sets PD/RES, bookkeeping; head parts (not build / baserefs) verify the binaries md5 first
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
    BIN_CHECK=""
    local refused=0
    case "$PART" in
    build) BIN_CHECK="not_applicable(build part)" ;;
    baserefs) BIN_CHECK="base binaries: md5 recorded only" ;;
    *)
        if check_binaries; then
            # results of another build must not be resumed or mixed into this part directory
            local dg
            dg=$(sha256sum "$OUT_ROOT/build/binaries.md5" | cut -c1-64)
            if [ -s "$PD/.build_digest" ] && [ "$(cat "$PD/.build_digest")" != "$dg" ]; then
                BIN_CHECK="STALE_PART_DIR(results of another build are in $PD: use a fresh SF41_OUT_ROOT or move them away)"
                refused=1
            else
                echo "$dg" > "$PD/.build_digest"
            fi
        else
            refused=1
        fi
        ;;
    esac
    write_env "$PD" "$PART"
    if [ "$refused" -eq 1 ]; then
        res "PART_REFUSED the binaries of $BIN_DIR do not match $OUT_ROOT/build/binaries.md5: $BIN_CHECK (run the build part, or rebuild cleanly; the part was not run)"
        BAD=$((BAD + 1))
        part_finish
        echo "run_sf41.sh: part=$PART REFUSED ($BIN_CHECK)" >&2
        exit 2
    fi
}

part_finish() { # part_finish: RECORDED part_seconds + PART_DONE + PART_EXIT
    local secs runs rc
    secs=$(( $(date +%s) - PART_T0 ))
    runs=$(awk -F'\t' '{ s += $2 } END { printf "%d", s }' "$PD/walls.tsv" 2> /dev/null)
    res "RECORDED part_seconds=$secs part_runs_seconds=${runs:-0} part=$PART (this invocation / sum of the recorded run walls)"
    if [ "$BAD" -eq 0 ]; then
        rc=0
    else
        rc=1
    fi
    echo "$rc" > "$PD/PART_EXIT"
    res "PART_DONE part=$PART rc=$rc"
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
# verdict-script plumbing
# ============================================================================================================
# run_verdict <script> <args...>: the python verdict script's lines through res; a non-zero exit is a blocking item
run_verdict() {
    local out rc
    out=$("$PY" "$SCRIPTS/$1" "${@:2}")
    rc=$?
    echo "$out" | res_block
    if [ "$rc" -ne 0 ]; then
        BAD=$((BAD + 1))
        res "VERDICT_SCRIPT_NONZERO script=$1 rc=$rc"
    fi
    return 0
}

sampler_peak_over() { # sampler_peak_over <run>: 0 iff the RR2 sampler is evaluable and the PID peak exceeds 30000 MiB
    "$PY" "$SCRIPTS/mem_verdict.py" peak "$PD" "$1" 2> /dev/null | grep -Eq '^LIMIT .* FAIL$'
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
# the 20-run R set (SF-40 r_items / r_args / r_expected = SF-37 RR13; RR9 of SF-38 and of SF-41)
# ============================================================================================================
r_items() { # lines `kind name arg1 arg2 arg3`
    local N e t c
    if [ "$DRY" = "1" ]; then
        echo "fwd smoke_g3d_n16 - - -"
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
    prod) [ "$DRY" = "1" ] && echo " 0 19 " || echo " 0 " ;;
    fwd) echo "$RC_FWD" ;;
    *) echo "$RC_POL" ;;
    esac
}

# G (RR9 (ii), (iii)) commands, identical for the base and the head binaries (only --out differs)
G2_ARGS=()
G3_ARGS=()
g_args() {
    if [ "$DRY" = "1" ]; then
        G2_ARGS=(--grid 64,16,16 --eps 2 --ell 0.25 --seed0 $SEED0 --nr 2 --window 0.1,0.9,0.1,0.9 --np-side 10 --planes auto --times auto --tol 1e-8
            --hmax-div 8 --fused-splines --single-launch --twin on --pollock-bridge on)
        G3_ARGS=(--grid 64,16,16 --ell 0.25 --eps 2 --seed0 $SEED0 --nr 2)
    else
        G2_ARGS=(--grid 512,64,64 --eps 2 --ell 0.125 --seed0 $SEED0 --nr 2 --window 0.1,0.9,0.1,0.9 --np-side 100 --planes auto --times auto
            --tol 1e-8 --hmax-div 8 --fused-splines --single-launch --twin on --pollock-bridge on)
        G3_ARGS=(--grid 512,64,64 --ell 0.125 --eps 2 --seed0 $SEED0 --nr 2)
    fi
}

# ============================================================================================================
# part build (HEAD): CLEAN build, tree echo, ctest -N, md5 of every binary
# ============================================================================================================
part_build() {
    if [ -d "$OUT_ROOT/build" ]; then # a re-invocation is a new clean build: the previous record is moved away, never resumed
        local old
        old=$OUT_ROOT/build.superseded.$(date -u +%Y%m%dT%H%M%SZ)
        mv "$OUT_ROOT/build" "$old" && echo "previous build record moved to $old"
    fi
    part_begin build
    local preset=${SF41_PRESET:-$(basename "$BIN_DIR")} b
    res "RECORDED build_tree $(tree_id | tr '\n' ' ')"
    res "RECORDED build_commit SF41_COMMIT=${SF41_COMMIT:-unknown} preset=$preset build_dir=$BIN_DIR"
    if [ "$DRY" = "1" ]; then
        res "RECORDED build=skipped (dry run: the local build directory is used as is, NOT a clean build)"
    else
        if [ "$BIN_DIR" != "$REPO/build/$preset" ]; then
            res "BUILD_REFUSED build_dir $BIN_DIR is not $REPO/build/$preset (the clean build is cmake --preset $preset)"
            BAD=$((BAD + 1))
            return 0
        fi
        # the stale-object hazard (rsync preserves mtimes): remove the build directory, configure, build with all cores of the V100 host
        case_run "$PD" clean_build "$PD/clean_build" " 0 " - - 0 -- bash -c 'cd "$1" && rm -rf "$2" && cmake --preset "$3" && cmake --build "$2" -j 2>&1 | tail -n 40; exit ${PIPESTATUS[0]}' _ \
            "$REPO" "$BIN_DIR" "$preset"
    fi
    if [ ! -d "$BIN_DIR" ]; then
        res "BUILD_VERDICT FAIL build directory $BIN_DIR does not exist after the build"
        BAD=$((BAD + 1))
        return 0
    fi
    case_run "$PD" ctest_N "$PD/ctest_N" " 0 " - - 0 -- ctest --test-dir "$BIN_DIR" -N
    res "RECORDED ctest_N $(grep -E 'Total Tests' "$PD/ctest_N.log" | tail -n 1)"
    grep -E '^ *Test +#[0-9]+: ' "$PD/ctest_N.log" | sed -E 's/^ *Test +#[0-9]+: +//' | LC_ALL=C sort > "$PD/ctest_names.txt"
    sha_file "$PD" ctest_names.txt
    ( cd "$BIN_DIR" && find . -maxdepth 1 -type f -perm -u+x ! -name '*.so*' ! -name '*.a' ! -name '*.cmake' | sed 's#^\./##' | LC_ALL=C sort | xargs -r md5sum ) > "$PD/binaries.md5.new"
    local missing=""
    for b in interpolation_nonperiodic_x_tests sf41_cross_representation_tests dirichlet_controls_tests pollock_nonperiodic_tests \
        dirichlet_long_box_stage_tests lagrangian_ensemble lagrangian_medium_ab macroflow3d_pipeline inlet_slab; do
        grep -q "  $b\$" "$PD/binaries.md5.new" || missing="$missing $b"
    done
    mv -f "$PD/binaries.md5.new" "$PD/binaries.md5"
    sha_file "$PD" binaries.md5
    res "BUILD_BINARIES n=$(wc -l < "$PD/binaries.md5" | tr -d ' ') md5file=$PD/binaries.md5 sha256=$(sha256sum "$PD/binaries.md5" | cut -c1-64)"
    while read -r md name; do
        case "$name" in
        interpolation_nonperiodic_x_tests | sf41_cross_representation_tests | dirichlet_controls_tests | pollock_nonperiodic_tests | dirichlet_long_box_stage_tests | \
            lagrangian_ensemble | lagrangian_medium_ab | macroflow3d_pipeline | inlet_slab) res "RECORDED binary_md5 $name $md" ;;
        esac
    done < "$PD/binaries.md5"
    if [ -z "$missing" ] && [ "$(tsv_rc "$PD/exit_codes.tsv" ctest_N)" = "0" ]; then
        res "BUILD_VERDICT PASS (binaries recorded; the head parts verify them)"
    else
        res "BUILD_VERDICT FAIL missing_binaries=[${missing}]"
        BAD=$((BAD + 1))
    fi
}

# ============================================================================================================
# part baserefs (BASE binaries): SF-27 trio, SF-38 20-run R set, G(ii), G(iii)
# ============================================================================================================
part_baserefs() {
    part_begin baserefs
    mkdir -p "$BASE_REFS/bin"
    local b
    for b in inlet_slab macroflow3d_pipeline lagrangian_ensemble lagrangian_medium_ab; do
        if [ "$BIN_DIR" != "$BASE_REFS/bin" ]; then
            cp -f "$BIN_DIR/$b" "$BASE_REFS/bin/"
        fi
    done
    for b in inlet_slab macroflow3d_pipeline lagrangian_ensemble lagrangian_medium_ab; do
        res "RECORDED base_binary_md5 $b $(md5sum "$BASE_REFS/bin/$b" | awk '{print $1}')"
    done
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
    local cfg ntrio=0
    if [ "$DRY" = "1" ]; then
        res "# DRY: the SF-27 trio is not run in the dry run (it exceeds the local limits); the live code is exercised on the R-set smoke only"
    else
        for cfg in $TRIO; do
            ntrio=$((ntrio + 1))
            case_run "$BASE_REFS" "trio_$cfg" "$BASE_REFS/trio/$cfg" " 0 " - "$BASE_REFS/trio/$cfg" 0 -- "$BASE_REFS/bin/macroflow3d_pipeline" "$REPO/apps/$cfg.yaml"
        done
    fi
    g_args
    echo "== G(ii) medium (a) lagrangian_ensemble run and G(iii) A/B run (base binaries)"
    res "RECORDED G2_INVOCATION ${G2_ARGS[*]}"
    res "RECORDED G3_INVOCATION ${G3_ARGS[*]}"
    case_run "$BASE_REFS" g_runner_a "$BASE_REFS/g_runner_a" " 0 " X - "$LONG_TMO" -- "$BASE_REFS/bin/lagrangian_ensemble" "${G2_ARGS[@]}" --out "$BASE_REFS/g_runner_a"
    case_run "$BASE_REFS" g_ab "$BASE_REFS/g_ab" " 0 " X - "$LONG_TMO" -- "$BASE_REFS/bin/lagrangian_medium_ab" "${G3_ARGS[@]}" --out "$BASE_REFS/g_ab"
    local nrec
    nrec=$(awk -F'\t' '{ n++ } END { print n + 0 }' "$PD/exit_codes.tsv")
    res "BASEREFS_RECORDED r_runs=$n trio_runs=$ntrio g_runs=2 recorded_in_exit_codes=$nrec unexpected_exit_codes=$BAD"
}

# ============================================================================================================
# part controls (HEAD): A, C1-C5, D, memory-model records
# ============================================================================================================
part_controls() {
    part_begin controls
    local AT="$(bin interpolation_nonperiodic_x_tests)" CT="$(bin dirichlet_controls_tests)" PT="$(bin pollock_nonperiodic_tests)" ST="$(bin dirichlet_long_box_stage_tests)"
    local afull="--full" cfull="--full" c4="--only C4" c5="--only C5" dd="--only D" dryflag=""
    if [ "$DRY" = "1" ]; then
        afull=""
        cfull=""
        c4=""
        c5=""
        dd=""
        dryflag="--dry"
        res "# DRY: fast tiers of every program (A, C, Pollock C4, C5 and D are all the default tiers); no compute-sanitizer; the verdict scripts run with --dry"
    fi
    # A (RR3): the full tier and the memcheck of the domain group (full tier)
    case_run "$PD" A_full "$PD/A_full" " 0 " X - "$LONG_TMO" -- "$AT" $afull
    local CS
    CS=$(command -v compute-sanitizer 2> /dev/null || echo /usr/local/cuda/bin/compute-sanitizer)
    if [ "$DRY" = "1" ]; then
        res "# DRY: A_memcheck skipped"
    elif [ -x "$CS" ]; then
        case_run "$PD" A_memcheck "$PD/A_memcheck" " 0 " X - "$LONG_TMO" -- "$CS" --tool memcheck "$AT" --full --only domain
    else
        res "A_MEMCHECK_UNAVAILABLE compute-sanitizer not found ($CS): A cannot be met"
        BAD=$((BAD + 1))
    fi
    run_verdict a_verdict.py "$PD" $dryflag
    # C1-C4 (RR5): tracker/Pollock program and the Pollock non-periodic core; C5 (solver, lean stage)
    case_run "$PD" C_full "$PD/C_full" " 0 " X - "$LONG_TMO" -- "$CT" $cfull
    case_run "$PD" C4_pollock "$PD/C4_pollock" " 0 " X - "$LONG_TMO" -- "$PT" $c4
    case_run "$PD" C5 "$PD/C5" " 0 " X - "$LONG_TMO" -- "$ST" $c5
    run_verdict c_verdict.py "$PD" $dryflag
    # D (RR6) with the RR2 sampler (the 3328x256x256 case is the memory-relevant one)
    case_run "$PD" D "$PD/D" " 0 " SX - "$LONG_TMO" -- "$ST" $dd
    run_verdict d_verdict.py "$PD" --run D $dryflag
    # memory model records (no gate): the E2/E3 and the E4 geometries, cuFFT work areas verified
    local mm
    for mm in a:3328,384,384 b:6656,256,256; do
        if [ "$DRY" = "1" ]; then
            case_run "$PD" "MEMMODEL_${mm%%:*}" "$PD/MEMMODEL_${mm%%:*}" any - - 0 -- "$ST" --memory-model 64,16,16 --verify-cufft
        else
            case_run "$PD" "MEMMODEL_${mm%%:*}" "$PD/MEMMODEL_${mm%%:*}" any - - 0 -- "$ST" --memory-model "${mm#*:}" --verify-cufft
        fi
        res "RECORDED MEMMODEL_${mm%%:*} grid=${mm#*:} exit=$(tsv_rc "$PD/exit_codes.tsv" "MEMMODEL_${mm%%:*}")"
        grep -E '^MEM_MODEL' "$PD/MEMMODEL_${mm%%:*}.log" 2> /dev/null | sed "s/^/RECORDED MEMMODEL_${mm%%:*} /" | res_block
    done
    mem_records_for D
}

# ============================================================================================================
# part cross (HEAD): B (RR4)
# ============================================================================================================
part_cross() {
    part_begin cross
    local bargs="--only B" dryflag=""
    if [ "$DRY" = "1" ]; then
        bargs=""
        dryflag="--dry"
    fi
    case_run "$PD" B "$PD/B" " 0 " X - "$LONG_TMO" -- "$(bin sf41_cross_representation_tests)" $bargs
    run_verdict b_verdict.py "$PD" --run B $dryflag
}

# ============================================================================================================
# part prodgrid (HEAD): E1-E4 (RR7) with the RR2 sampler and the pre-authorized mapped-host fallback
# ============================================================================================================
# e_table: lines `id grid eps ell np_side`
e_table() {
    if [ "$DRY" = "1" ]; then
        echo "E1 128,16,16 1 0.5 10"
        echo "E2 128,16,16 1.5 0.5 10"
        echo "E3 128,16,16 2 0.5 10"
        echo "E4 64,16,16 2 0.25 10"
    else
        echo "E1 2048,256,256 1 0.0390625 100"
        echo "E2 3328,384,384 1.5 0.041666666666666664 100"
        echo "E3 3328,384,384 2 0.08333333333333333 100"
        echo "E4 6656,256,256 2 0.25 100"
    fi
}

# e_cmd <grid> <eps> <ell> <np_side> <out> [extra args]: E_CMD = the RR7 runner command (X0 = 5 lambda is the runner's default for --medium b)
E_CMD=()
e_cmd() {
    E_CMD=("$(bin lagrangian_ensemble)" --medium b --grid "$1" --eps "$2" --ell "$3" --seed0 $SEED0 --nr 1 --window 0.1,0.9,0.1,0.9 --np-side "$4"
        --planes auto --times auto --tol 1e-8 --hmax-div 8 --fused-splines --single-launch --twin on --pollock-bridge on --out "$5" "${@:6}")
}

part_prodgrid() {
    part_begin prodgrid
    local id grid eps ell np dryflag="" force=0
    [ "$DRY" = "1" ] && dryflag="--dry"
    while read -r id grid eps ell np; do
        e_cmd "$grid" "$eps" "$ell" "$np" "$PD/$id"
        res "RECORDED E_INVOCATION $id: ${E_CMD[*]}"
        case_run "$PD" "$id" "$PD/$id" " 0 " SX - "$LONG_TMO" -- "${E_CMD[@]}"
        [ "$CASE_STATE" = "busy" ] && continue
        force=0
        [ "$DRY" = "1" ] && [ "${SF41_DRY_FORCE_MAPPED:-0}" = "1" ] && [ "$id" = "E1" ] && force=1
        if sampler_peak_over "$id" || [ "$force" = "1" ]; then
            # RR7: the PID peak exceeds 30000 MiB -> this grid is re-run once with the mapped-host head (E1 fallback); both runs are reported
            res "RECORDED E_FALLBACK $id: PID peak above ${MEM_LIMIT_MIB} MiB (or dry-run forcing): re-running once with --head-mapped-host as ${id}_mapped"
            e_cmd "$grid" "$eps" "$ell" "$np" "$PD/${id}_mapped" --head-mapped-host
            res "RECORDED E_INVOCATION ${id}_mapped: ${E_CMD[*]}"
            case_run "$PD" "${id}_mapped" "$PD/${id}_mapped" " 0 " SX - "$LONG_TMO" -- "${E_CMD[@]}"
        fi
    done < <(e_table)
    run_verdict e_verdict.py "$PD" $dryflag
}

# ============================================================================================================
# part evladder (HEAD): F (RR8), 24 runs
# ============================================================================================================
part_evladder() {
    part_begin evladder
    local eps seed ny nx run planes ell=0.5 x0m=2.5 np=100 L1=12 nplanes=14 eps_l="1 2" seeds="6101 6102 6103" grids="32 64 128 256" fargs=()
    if [ "$DRY" = "1" ]; then
        ell=0.25
        np=10
        L1=3
        nplanes=3
        eps_l="1 2"
        seeds="6101 6102"
        grids="8 16"
        fargs=(--eps 1,2 --seeds 6101,6102 --grids 8,16 --x0 1.25 --lambda 0.25 --nplanes 3)
        planes=$(awk -v l="$ell" -v n="$nplanes" 'BEGIN { for (m = 1; m <= n; m++) printf "%s%.10g", (m > 1 ? "," : ""), 5 * l + m * l }')
        res "# DRY: ell=0.25, L1=3, two rungs (ny=8,16), two seeds, 3 planes, np-side 10; NOT evidence"
    else
        # X0 = 5 lambda = 2.5; planes X0 + m lambda, m = 1..14 (3, 3.5, ..., 9.5 = L1 - 5 lambda), the same physical planes at every rung
        planes=$(awk 'BEGIN { for (m = 1; m <= 14; m++) printf "%s%.10g", (m > 1 ? "," : ""), 2.5 + 0.5 * m }')
    fi
    res "RECORDED F_PLANES $planes"
    for eps in $eps_l; do
        for seed in $seeds; do
            for ny in $grids; do
                nx=$((L1 * ny))
                run=F_e${eps}_s${seed}_g${ny}
                case_run "$PD" "$run" "$PD/$run" " 0 " X - "$LONG_TMO" -- "$(bin lagrangian_ensemble)" --medium b --grid "$nx,$ny,$ny" --eps "$eps" --ell "$ell" \
                    --seed0 "$seed" --nr 1 --window 0.1,0.9,0.1,0.9 --np-side "$np" --planes "$planes" --times auto --tol 1e-8 --hmax-div 8 --fused-splines \
                    --single-launch --twin off --pollock-bridge on --out "$PD/$run"
            done
        done
    done
    run_verdict f_verdict.py "$PD" ${fargs[@]+"${fargs[@]}"}
}

# ============================================================================================================
# part abrerun (HEAD): H (RR10), the two SF-38 A/B commands with the head binary
# ============================================================================================================
AB_S1=(--grid 2048,256,256 --ell 0.0390625 --eps 1)
AB_S4=(--grid 3328,256,256 --ell 0.125 --eps 2)
part_abrerun() {
    part_begin abrerun
    local nr=10 hargs=()
    if [ "$DRY" = "1" ]; then
        AB_S1=(--grid 64,16,16 --ell 0.25 --eps 1)
        AB_S4=(--grid 64,16,16 --ell 0.125 --eps 2)
        nr=2
        hargs=(--nr 2)
        res "# DRY: 64x16x16, 2 realizations; the byte comparison with the committed SF-38 files is meaningless here (different grids): H is expected NOT_OK"
    fi
    case_run "$PD" ab_s1 "$PD/ab_s1" " 0 " SX - "$LONG_TMO" -- "$(bin lagrangian_medium_ab)" "${AB_S1[@]}" --seed0 $SEED0 --nr $nr --out "$PD/ab_s1"
    case_run "$PD" ab_s4 "$PD/ab_s4" " 0 " SX - "$LONG_TMO" -- "$(bin lagrangian_medium_ab)" "${AB_S4[@]}" --seed0 $SEED0 --nr $nr --out "$PD/ab_s4"
    local r
    for r in ab_s1 ab_s4; do
        mem_records_for $r
    done
    if [ "$DRY" = "1" ]; then
        local out
        out=$("$PY" "$SCRIPTS/h_verdict.py" "$PD" ${hargs[@]+"${hargs[@]}"})
        echo "$out" | res_block
    else
        run_verdict h_verdict.py "$PD" ${hargs[@]+"${hargs[@]}"}
    fi
}

# ============================================================================================================
# part final (HEAD): G (RR9)
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

# g_bytes <name> <head_dir> <ref_dir> [allow-extra glob]: the tree comparison through g_compare.py (timings.json excluded)
g_bytes() {
    local out rc extra=()
    [ -n "${4:-}" ] && extra=(--allow-extra "$4")
    out=$("$PY" "$SCRIPTS/g_compare.py" tree --name "$1" --ref "$3" --cand "$2" --exclude timings.json ${extra[@]+"${extra[@]}"})
    rc=$?
    echo "$out" | res_block
    [ "$rc" -eq 0 ] || BAD=$((BAD + 1))
}

part_final() {
    part_begin final
    local PIPE
    PIPE=$(bin macroflow3d_pipeline)
    res "RECORDED binaries md5: $(md5sum "$PIPE" "$(bin inlet_slab)" "$(bin lagrangian_ensemble)" "$(bin lagrangian_medium_ab)" 2> /dev/null | tr '\n' ' ')"
    # ---- G: full ctest (dry run: one fast case)
    local cargs=()
    if [ -n "${SF41_CTEST_ARGS:-}" ]; then
        read -r -a cargs <<< "$SF41_CTEST_ARGS"
    elif [ "$DRY" = "1" ]; then
        cargs=(-R '^interpolation_nonperiodic_x$')
    fi
    case_run "$PD" ctest "$PD/ctest" any X - 0 -- ctest --test-dir "$BIN_DIR" --output-on-failure ${cargs[@]+"${cargs[@]}"}
    local crc summ ntot nfail
    crc=$(tsv_rc "$PD/exit_codes.tsv" ctest)
    summ=$(grep -E '^[0-9]+% tests passed' "$PD/ctest.log" | tail -n 1)
    ntot=$(echo "$summ" | sed -n 's/.*out of \([0-9]*\).*/\1/p')
    nfail=$(echo "$summ" | sed -n 's/.*, \([0-9]*\) tests failed.*/\1/p')
    nfail=${nfail:-0}
    grep -E '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$PD/ctest.log" | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: +([^ ]+) +\.+ *(\*\*\*)?([A-Za-z ]+[a-z]) +([0-9.]+) sec.*/\1\t\3\t\4/' > "$PD/ctest_times.tsv"
    sha_file "$PD" ctest_times.tsv
    res "RECORDED ctest_per_test_times written ($PD/ctest_times.tsv, $(wc -l < "$PD/ctest_times.tsv" | tr -d ' ') tests: name<TAB>status<TAB>seconds)"
    if [ "$crc" = "0" ] && [ -n "$ntot" ] && [ "$ntot" -ge 1 ] && [ "$nfail" = "0" ]; then
        res "G_CTEST exit=0 tests=$ntot failed=0 wall_s=$(tsv_col "$PD/walls.tsv" ctest 2) PASS (summary: $summ)"
    else
        res "G_CTEST exit=$crc tests=${ntot:--} failed=$nfail wall_s=$(tsv_col "$PD/walls.tsv" ctest 2) FAIL (summary: ${summ:-none})"
        BAD=$((BAD + 1))
    fi
    # ---- G: SF-27 trio vs the base references
    local cfg
    BC_N=0
    if [ "$DRY" = "1" ]; then
        res "# DRY: SF-27 trio not run"
    else
        for cfg in $TRIO; do
            case_run "$PD" "trio_$cfg" "$PD/trio/$cfg" any - "$PD/trio/$cfg" 0 -- "$PIPE" "$REPO/apps/$cfg.yaml"
            bc_tree "trio:$cfg" "$PD/trio/$cfg" "$BASE_REFS/trio/$cfg"
            bc_rc_check "trio:$cfg" "$CASE_RC" "$BASE_REFS/trio/$cfg.time"
        done
    fi
    if [ "$BC_N" -eq 0 ]; then
        res "G_TRIO IDENTICAL (SF-27 trio byte-compare clean vs $BASE_REFS/trio; manifest keys ${MANIFEST_DROPS[*]} dropped)"
    else
        res "G_TRIO DIFF n=$BC_N"
        BAD=$((BAD + 1))
    fi
    # ---- G: the 20-run R set vs the base references
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
    elif [ "$DRY" = "1" ]; then
        bc_res IDENTICAL "binary:md5_equal (dry run: the head binary is used as the base)"
    else
        bc_res DIFF "binary:md5_differs (head and base inlet_slab are the SAME file content $hmd5)"
    fi
    if [ "$BC_N" -eq 0 ]; then
        res "G_RSET IDENTICAL (R set byte-identical to $BASE_REFS/r; SF-40 compare rules)"
    else
        res "G_RSET DIFF n=$BC_N"
        BAD=$((BAD + 1))
    fi
    # ---- G (ii): medium (a) lagrangian_ensemble run; G (iii): lagrangian_medium_ab run (RR9)
    g_args
    res "RECORDED G2_INVOCATION ${G2_ARGS[*]}"
    res "RECORDED G3_INVOCATION ${G3_ARGS[*]}"
    case_run "$PD" g_runner_a "$PD/g_runner_a" " 0 " X - "$LONG_TMO" -- "$(bin lagrangian_ensemble)" "${G2_ARGS[@]}" --out "$PD/g_runner_a"
    g_bytes runner_medium_a "$PD/g_runner_a" "$BASE_REFS/g_runner_a"
    bc_rc_check "g_runner_a" "$(tsv_rc "$PD/exit_codes.tsv" g_runner_a)" "$BASE_REFS/g_runner_a.time"
    case_run "$PD" g_ab "$PD/g_ab" " 0 " X - "$LONG_TMO" -- "$(bin lagrangian_medium_ab)" "${G3_ARGS[@]}" --out "$PD/g_ab"
    g_bytes medium_ab "$PD/g_ab" "$BASE_REFS/g_ab" 'realizations/r_*/exits_b.csv'
    bc_rc_check "g_ab" "$(tsv_rc "$PD/exit_codes.tsv" g_ab)" "$BASE_REFS/g_ab.time"
    # ---- G verdict from the lines above (+ the two periodic-spline ctest entries Passed)
    local gout grc
    gout=$("$PY" "$SCRIPTS/g_compare.py" verdict "$RES" --ctest-times "$PD/ctest_times.tsv")
    grc=$?
    echo "$gout" | res_block
    [ "$grc" -eq 0 ] || BAD=$((BAD + 1))
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
    local fn need f
    case "$ARG_PART" in
    build | baserefs | controls | cross | prodgrid | evladder | abrerun | final) fn=part_$ARG_PART ;;
    *)
        echo "run_sf41.sh: unknown part '$ARG_PART'" >&2
        usage
        exit 2
        ;;
    esac
    if [ "$ARG_PART" = "build" ]; then
        BIN_DIR=$(realpath -m "$ARG_BIN_DIR")
    else
        BIN_DIR=$(cd "$ARG_BIN_DIR" 2> /dev/null && pwd) || {
            echo "run_sf41.sh: build_dir '$ARG_BIN_DIR' not found" >&2
            exit 2
        }
    fi
    case "$ARG_PART" in
    build) need="" ;;
    baserefs) need="inlet_slab macroflow3d_pipeline lagrangian_ensemble lagrangian_medium_ab" ;;
    controls) need="interpolation_nonperiodic_x_tests dirichlet_controls_tests pollock_nonperiodic_tests dirichlet_long_box_stage_tests" ;;
    cross) need="sf41_cross_representation_tests" ;;
    prodgrid | evladder) need="lagrangian_ensemble" ;;
    abrerun) need="lagrangian_medium_ab" ;;
    final) need="macroflow3d_pipeline inlet_slab lagrangian_ensemble lagrangian_medium_ab" ;;
    esac
    for f in $need; do
        [ -x "$BIN_DIR/$f" ] || { echo "run_sf41.sh: $BIN_DIR/$f not found or not executable" >&2; exit 2; }
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
            echo "run_sf41.sh: /usr/bin/time not found (host max RSS is required in a campaign)" >&2
            exit 2
        fi
    fi
    if [ "$DRY" = "1" ]; then
        echo "run_sf41.sh: SF41_DRYRUN=1 (tiny settings; NOT campaign evidence)"
        TRACE_THREADS=4
        PROD_N=16
        PROD_EXTRA=(--threads 4)
        LONG_TMO=1800
    fi
    echo "run_sf41.sh: part=$ARG_PART build_dir=$BIN_DIR host=$(hostname) gpu_id=$GPU_ID out_root=$OUT_ROOT base_refs=$BASE_REFS repo=$REPO start=$(date -u +%FT%TZ)"
    "$fn"
    part_finish
    echo "run_sf41.sh: part=$ARG_PART end=$(date -u +%FT%TZ) blocking_failures=$BAD"
    [ "$BAD" -eq 0 ]
    exit $?
}

main
