#!/usr/bin/env bash
# SF-39 campaign runner (one part per detached V100 job; contract C-RUN of preregistration.md section 9, reading rules RR0-RR6).
# Run from the synced mirror root (the repository root is derived from this script's path); every part is restartable.  Modelled on
# docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/scripts/run_sf41.sh (case_run, exit_codes.tsv / walls.tsv,
# binaries md5 gate, results.txt / PART_DONE) WITHOUT its exclusive-GPU precheck and WITHOUT any GPU selection: the V100 is shared and
# `scripts/remote` (slot, REMOTE_GPU_WAIT, REMOTE_GPU_MIN_FREE_MIB) governs.
#
#   bash run_sf39.sh <build_dir> checks            unit tests, geometry check, RR2 regression, RR3 (ii) solve checks, (iii) byte check, (iv) smoke;
#                                                  the checks that depend on the scripts tree (unittest, geometry, smoke_reduce, signtable) carry
#                                                  the scripts revision in their id (`unittest:<rev>`), so a new tree re-runs them and old records stay
#   bash run_sf39.sh <build_dir> pilot [s025|s1|s225|s4|par2]     20-realization pilot (directory A, narrow rungs, twin; Par2 pilot + field stats)
#   bash run_sf39.sh <build_dir> checkpoint [s025]  python3 -m sf39.checkpoint (no CUDA); without selector the original case (s1 s225 s4, case id
#                                                  `checkpoint`); `s025` = the separate case `checkpoint_s025` (everything under checkpoint/s025/)
#   bash run_sf39.sh <build_dir> extend [s025|s1|s225|s4|par2]    raise every arm to the budget (configs/budget.json) + directory B; Par2 extension
#   bash run_sf39.sh <build_dir> width             s4_w24 and s4_w12 (30 realizations each), then s4_w12c (17: seeds 7215-7231, continuation of
#                                                  the halted s4_w12, owner decision 2026-10-28; s4_w12 itself is never released)
#   bash run_sf39.sh <build_dir> analysis          python3 -m sf39.analyze (no CUDA)
#
# <build_dir> holds lagrangian_ensemble, lagrangian_medium_ab, macroflow3d_pipeline, inlet_slab, par2_field_stats (the clean build of RR1).
# Without a selector, pilot/extend run s025, s1, s225, s4 and par2 in that order.
#
# Environment (all optional):
#   SF39_OUT_ROOT  output root, NOT in the mirror (a `sync --delete` would erase it)       (default $HOME/sf39_campaign)
#   SF39_DRYRUN    1 = print every command with its resolved flags, touch nothing, verify nothing   (default 0)
#   SF39_BUDGET    path of the budget file read by `extend`                                (default <A>/configs/budget.json; for dry-run tests)
#   SF39_TIME_BIN  GNU time (-v -o <file>) used for every run                              (default /usr/bin/time; override only in tests)
#   SF39_REGRESS_HOME  base of regress/refs/<rev>/out and .cache/mf3d-tmp               (default $HOME; override only in tests)
#   SF39_COMMIT    commit of the scripts recorded in run.env                               (default: the mirror's .mf3d_source_rev, else git HEAD)
#   PYTHON         interpreter                                                             (default python3)
# PYTHONPATH gets <A>/scripts prepended by this script.
#
# Outputs (all under $SF39_OUT_ROOT; never inside the mirror):
#   build/binaries.md5            md5sum format, bare names relative to <build_dir> (written by the orchestrator's RR1 build job)
#   runs/<arm>/                   runner output directory of production / ladder / width arms (shared by pilot, extend and width)
#   par2/raw/par2_s{c}[_ext]/     Par2 pipeline outputs (the pipeline's working directory is $SF39_OUT_ROOT/par2)
#   <part>/                       bookkeeping: run.env  exit_codes.tsv  walls.tsv  results.txt  sha256.txt  resumed.txt  logs/  PART_EXIT
#                                 (checks: runs/ and regress_head/ of the check runs, never data)
# Bookkeeping files: exit_codes.tsv = `id<TAB>rc` (no header; the LAST record of an id wins); walls.tsv = `id<TAB>wall_s<TAB>end_utc`;
# results.txt = the verdict / RECORDED lines of the part, recomputed on every invocation, ending with PART_DONE part=<p> rc=<0|1>;
# sha256.txt = sha256 of the small text outputs (logs, tsv, run.env, per-realization DONE/status/solver_b/flow_stats/field/timings json).
#
# Resume rules (RR0): a runner id whose recorded rc is final (0 ok, 2 usage, 3 seed-face, 4 unusable solve, 7 option mismatch: deterministic
# outcomes) is never re-run; any other recorded rc, or partial outputs with no recorded rc (infrastructure interruption), is re-run with
# IDENTICAL settings (the runner skips realizations with DONE) and listed in resumed.txt.  Nothing is ever retried with changed settings.
# A runner exit 4 (or 3) halts that arm: it is appended to $SF39_OUT_ROOT/halted_arms.txt, every later invocation refuses that arm
# (ARM_HALTED, part rc 1) until the owner decides and removes the line; other arms continue (RR6).  A Par2 id is final only at rc 0; an
# interrupted or failed Par2 run is re-run from scratch into a cleaned output directory.
#
# bash 4.2 (the V100 host): no empty-array expansion under `set -u` (use ${a[@]+"${a[@]}"}), no declare -n, no ${var@Q}, no mapfile -d.
set -uo pipefail

SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=$(dirname "$SCRIPTS")
REPO=$(cd "$ART/../../../.." && pwd)
ARG_BIN_DIR=${1:-}
ARG_PART=${2:-}
SEL=${3:-}
PY=${PYTHON:-python3}
DRY=${SF39_DRYRUN:-0}
OUT_ROOT=${SF39_OUT_ROOT:-$HOME/sf39_campaign}
BUDGET=${SF39_BUDGET:-$ART/configs/budget.json}
TIME_BIN=${SF39_TIME_BIN:-/usr/bin/time} # override only for local tests of this script
export PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}"

BAD=0
CHILD=""
PART=""
PD=""
RES=""
PART_T0=0
BIN_CHECK=""
BIN_DIR=""
RUN_DIRS=()
CASE_RC=""
CASE_STATE=""
CASE_WALL=""

# constants (preregistration sections 1, 6, 7) ----------------------------------------------------------------------------
SEED_PROD=5001
SEED_CHECK=4901
SEED_W24=7101
SEED_W12=7201
SEED_W12C=7215 # s4_w12 halted at r=13 (seed 7214 skipped); s4_w12c = seeds 7215-7231
N_W12C=17
PILOT_NR=20
BASE_REV=1e40cfe
REG_HOME=${SF39_REGRESS_HOME:-$HOME} # override only for local tests of this script
REFS_DIR=$REG_HOME/regress/refs/$BASE_REV/out
TMP_BUILD=$REG_HOME/.cache/mf3d-tmp
FINAL_RUNNER=" 0 2 3 4 7 "
BINARIES="lagrangian_ensemble lagrangian_medium_ab macroflow3d_pipeline inlet_slab par2_field_stats"
AMPS="s025 s1 s225 s4"

usage() {
    echo "usage: run_sf39.sh <build_dir> <part> [selector]   (parts: checks pilot checkpoint extend width analysis; selector s025|s1|s225|s4|par2 for pilot/extend, s025 for checkpoint)" >&2
}

# helpers ------------------------------------------------------------------------------------------------------------------
res() { # res <line...>: a result line (echoed; appended to results.txt unless dry)
    echo "$*"
    [ "$DRY" = "1" ] || echo "$*" >> "$RES"
}

mk() { # mk <dir...>: mkdir -p (no-op in a dry run)
    [ "$DRY" = "1" ] || mkdir -p "$@"
}

tsv_rc() { # tsv_rc <tsv> <id> -> the last recorded rc of <id> (empty if none)
    [ -e "$1" ] || return 0
    awk -F'\t' -v c="$2" '$1 == c { r = $2 } END { if (r != "") print r }' "$1"
}

tsv_col() { # tsv_col <tsv> <id> <col> -> that column of the last record of <id>
    [ -e "$1" ] || return 0
    awk -F'\t' -v c="$2" -v k="$3" '$1 == c { r = $k } END { if (r != "") print r }' "$1"
}

safe_rm() { # safe_rm <path>: rm -rf, refusing anything that is not a deep absolute path
    case "$1" in
    /?*/?*/?*) rm -rf "$1" ;;
    *)
        echo "run_sf39.sh: refusing to remove '$1'" >&2
        exit 2
        ;;
    esac
}

# shellcheck disable=SC2329  # reached through the signal trap
kill_tree() { # kill_tree <pid>: TERM the process and all its descendants
    local c
    for c in $(pgrep -P "$1" 2> /dev/null); do
        kill_tree "$c"
    done
    kill -TERM "$1" 2> /dev/null
}

# shellcheck disable=SC2329  # reached through the signal trap
on_signal() {
    echo "run_sf39.sh: interrupted (signal); stopping the running case (no rc is recorded; a re-invocation resumes it)"
    [ -n "$CHILD" ] && kill_tree "$CHILD"
    exit 143
}
trap on_signal INT TERM

run_in() { # run_in <cwd|-> <time_file> cmd...   (runs in a background subshell; /usr/bin/time -v is the instrument of host max RSS)
    local cwd=$1 tf=$2
    shift 2
    if [ "$cwd" != "-" ]; then
        cd "$cwd" || return 97
    fi
    "$TIME_BIN" -v -o "$tf" "$@"
}

arm_flags() { # arm_flags <arm> -> the runner arguments of the arm (sf39.geometry, C-GEOM); exit 2 if the arm is unknown
    "$PY" -m sf39.geometry flags "$1" || {
        echo "run_sf39.sh: sf39.geometry flags $1 failed" >&2
        exit 2
    }
}

# case_run <pd> <id> <final rc list> <cwd | -> <clean-dir | -> -- cmd...
# Sets CASE_RC / CASE_STATE (done | ran | dry) / CASE_WALL.  Artifacts: <pd>/logs/<id>.log and .time.  <clean-dir> (Par2 only) is removed
# before a (re-)run: it is the rerun-from-scratch rule of an interrupted Par2 run.
case_run() {
    local pd=$1 id=$2 finals=$3 cwd=$4 clean=$5
    shift 6
    CASE_RC=""
    CASE_STATE=""
    CASE_WALL=""
    local lb tsv rc0 f
    f=${id//:/_}
    lb=$pd/logs/$f
    tsv=$pd/exit_codes.tsv
    if [ "$DRY" = "1" ]; then
        echo "DRYRUN [$id] ${cwd/#-/.}: $*"
        CASE_STATE=dry
        return 0
    fi
    mkdir -p "$pd/logs"
    rc0=$(tsv_rc "$tsv" "$id")
    if [ -n "$rc0" ]; then
        case "$finals" in
        *" $rc0 "*)
            echo "SKIP_DONE run=$id rc=$rc0"
            CASE_RC=$rc0
            CASE_STATE="done"
            CASE_WALL=$(tsv_col "$pd/walls.tsv" "$id" 2)
            return 0
            ;;
        esac
        if [ "$finals" != "ALWAYS" ]; then
            echo "RESUMED run=$id (recorded rc=$rc0 is not final: re-run with identical settings)"
            printf '%s\t%s\t%s\n' "$id" "$(date -u +%FT%TZ)" "recorded rc=$rc0 not final" >> "$pd/resumed.txt"
        fi
    elif [ -e "$lb.log" ]; then
        echo "RESUMED run=$id (log without a recorded rc: interrupted; re-run with identical settings)"
        printf '%s\t%s\t%s\n' "$id" "$(date -u +%FT%TZ)" "interrupted, no rc" >> "$pd/resumed.txt"
    fi
    if [ "$clean" != "-" ]; then
        safe_rm "$clean"
    fi
    if [ "$cwd" != "-" ]; then
        mkdir -p "$cwd"
    fi
    echo "+ [$id] ${cwd/#-/.}: $* > $lb.log 2>&1"
    local t0 rc end
    t0=$(date +%s)
    run_in "$cwd" "$lb.time" "$@" > "$lb.log" 2>&1 &
    CHILD=$!
    wait "$CHILD"
    rc=$?
    CHILD=""
    CASE_WALL=$(($(date +%s) - t0))
    end=$(date -u +%FT%TZ)
    printf '%s\t%s\n' "$id" "$rc" >> "$tsv"
    printf '%s\t%s\t%s\n' "$id" "$CASE_WALL" "$end" >> "$pd/walls.tsv"
    CASE_RC=$rc
    CASE_STATE=ran
    echo "  CASE_RESULT run=$id rc=$rc wall=${CASE_WALL}s end=$end"
    grep -E '^(R=[0-9]+ )?(STATUS|RUN_DONE|B_SOLVER|STAGE_FAILED|SEED_FACE_FAILED)' "$lb.log" 2> /dev/null | cut -c1-200 | sed 's/^/    /'
    return 0
}

# binaries gate (RR1) -------------------------------------------------------------------------------------------------------
check_binaries() { # BIN_CHECK = ok(...) | ABSENT | MISMATCH(...) | INCOMPLETE(...)
    local f=$OUT_ROOT/build/binaries.md5 b miss="" out
    if [ ! -s "$f" ]; then
        BIN_CHECK="ABSENT($f)"
        return 1
    fi
    if ! out=$(cd "$BIN_DIR" && md5sum -c "$f" 2>&1); then
        BIN_CHECK="MISMATCH($(echo "$out" | grep -v ': OK$' | tr '\n' ' ' | cut -c1-300))"
        return 1
    fi
    for b in $BINARIES; do
        grep -Eq "^[0-9a-f]{32}  \*?$b\$" "$f" || miss="$miss $b"
    done
    if [ -n "$miss" ]; then
        BIN_CHECK="INCOMPLETE(not listed in binaries.md5:$miss)"
        return 1
    fi
    BIN_CHECK="ok($(wc -l < "$f" | tr -d ' ') binaries, md5file_sha256=$(sha256sum "$f" | cut -c1-16))"
    return 0
}

# src_rev: the scripts revision.  The repository fallback is used only when $REPO itself is the top level of its repository (an enclosing
# repository's HEAD says nothing about this tree); `rev-parse --verify` prints the hash and fails on an unborn HEAD.
src_rev() {
    if [ -n "${SF39_COMMIT:-}" ]; then
        echo "$SF39_COMMIT"
    elif [ -s "$REPO/.mf3d_source_rev" ]; then
        cat "$REPO/.mf3d_source_rev"
    elif [ "$(git -C "$REPO" rev-parse --show-toplevel 2> /dev/null)" = "$(cd "$REPO" && pwd -P)" ] \
        && git -C "$REPO" rev-parse --verify -q HEAD 2> /dev/null; then
        :
    else
        echo unknown
    fi
}

write_env() { # write_env <pd> <part>   (appended as a new block when the file exists)
    local pd=$1 f b
    f=$pd/run.env
    [ -e "$f" ] && echo "# --- resumed/repeated invocation ---" >> "$f"
    {
        echo "DATE_UTC=$(date -u +%FT%TZ)"
        echo "HOST=$(hostname)"
        echo "PART=$2"
        echo "SELECTOR=${SEL:-none}"
        echo "BIN_DIR=$BIN_DIR"
        for b in $BINARIES; do
            [ -x "$BIN_DIR/$b" ] && echo "BIN_MD5_$b=$(md5sum "$BIN_DIR/$b" | awk '{print $1}')"
        done
        echo "BINARIES_CHECK=$BIN_CHECK"
        echo "OUT_ROOT=$OUT_ROOT"
        echo "REPO=$REPO"
        echo "MF3D_SOURCE_REV=$(cat "$REPO/.mf3d_source_rev" 2> /dev/null || echo absent)"
        echo "SCRIPTS_COMMIT=$(src_rev)"
        echo "PYTHON=$($PY --version 2>&1)"
        echo "REMOTE_GPU_MIN_FREE_MIB=${REMOTE_GPU_MIN_FREE_MIB:-unset}"
        echo "REMOTE_GPU_WAIT=${REMOTE_GPU_WAIT:-unset}"
        echo "CUDA_VISIBLE_DEVICES_inherited=${CUDA_VISIBLE_DEVICES:-unset}"
        echo "GPU=$( (nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv,noheader 2> /dev/null | tr '\n' ';') || true)"
        echo "NVCC=$( (nvcc --version 2> /dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1) || true)"
        echo "GCC=$( (gcc --version 2> /dev/null | head -1) || true)"
        nvidia-smi 2> /dev/null | head -n 12 | sed 's/^/# nvidia-smi banner: /'
    } >> "$f"
}

part_begin() { # part_begin <part>
    PART=$1
    PD=$OUT_ROOT/$PART
    PART_T0=$(date +%s)
    RUN_DIRS=()
    if [ "$DRY" = "1" ]; then
        BIN_CHECK="dry run: not verified"
        RES=/dev/null
        echo "# DRYRUN part=$PART selector=${SEL:-none} bin_dir=$BIN_DIR out_root=$OUT_ROOT (commands only; nothing is run or written)"
        return 0
    fi
    if [ ! -x "$TIME_BIN" ]; then
        echo "run_sf39.sh: $TIME_BIN -v is required (host max RSS); refusing" >&2
        exit 2
    fi
    mkdir -p "$PD"
    rm -f "$PD/PART_EXIT"
    RES=$PD/results.txt
    : > "$RES"
    local refused=0 dg
    if check_binaries; then
        dg=$(sha256sum "$OUT_ROOT/build/binaries.md5" | cut -c1-64)
        if [ -s "$PD/.build_digest" ] && [ "$(cat "$PD/.build_digest")" != "$dg" ]; then
            BIN_CHECK="STALE_PART_DIR(results of another build are in $PD)"
            refused=1
        else
            echo "$dg" > "$PD/.build_digest"
        fi
    else
        refused=1
    fi
    write_env "$PD" "$PART"
    if [ "$refused" -eq 1 ]; then
        res "PART_REFUSED the binaries of $BIN_DIR do not match $OUT_ROOT/build/binaries.md5: $BIN_CHECK (the part was not run)"
        echo 1 > "$PD/PART_EXIT"
        res "PART_DONE part=$PART rc=1"
        echo "run_sf39.sh: part=$PART REFUSED ($BIN_CHECK)" >&2
        exit 2
    fi
    res "PART_BEGIN part=$PART selector=${SEL:-none} binaries=$BIN_CHECK"
}

sha_part() { # sha256.txt: small text outputs of the part and of the runs this invocation touched (paths relative to $OUT_ROOT)
    local d
    (
        cd "$OUT_ROOT" || exit 0
        {
            find "$PART" -maxdepth 2 -type f ! -name results.txt ! -name sha256.txt ! -name '.build_digest'
            # the checkpoint part: the sealed files of a separate case (checkpoint/<case>/sealed/*, depth 3) like checkpoint/sealed/*
            [ "$PART" = "checkpoint" ] && find "$PART" -mindepth 3 -maxdepth 3 -type f -path '*/sealed/*'
            for d in ${RUN_DIRS[@]+"${RUN_DIRS[@]}"}; do
                [ -d "$d/realizations" ] && find "${d#"$OUT_ROOT"/}/realizations" -type f \
                    \( -name DONE -o -name status.json -o -name solver_b.json -o -name flow_stats.json -o -name field.json -o -name timings.json \)
            done
        } | LC_ALL=C sort -u | xargs -r sha256sum
    ) > "$PD/sha256.txt"
}

part_finish() {
    local secs runs rc
    if [ "$DRY" = "1" ]; then
        echo "# DRYRUN part=$PART end"
        return 0
    fi
    secs=$(($(date +%s) - PART_T0))
    runs=$(awk -F'\t' '{ s += $2 } END { printf "%d", s }' "$PD/walls.tsv" 2> /dev/null)
    res "RECORDED part_seconds=$secs part_runs_seconds=${runs:-0} part=$PART (this invocation / sum of the recorded run walls: GPU-busy upper bound, the GPU is shared)"
    sha_part
    if [ "$BAD" -eq 0 ]; then
        rc=0
    else
        rc=1
    fi
    echo "$rc" > "$PD/PART_EXIT"
    res "PART_DONE part=$PART rc=$rc"
    return "$rc"
}

# runner arms -------------------------------------------------------------------------------------------------------------
halted() { # halted <arm>: 0 (true) iff the owner has not yet released the arm
    [ -s "$OUT_ROOT/halted_arms.txt" ] && awk -F'\t' -v a="$1" '$1 == a { f = 1 } END { exit !f }' "$OUT_ROOT/halted_arms.txt"
}

done_count() { # done_count <out>: realizations with a DONE file
    local n=0 f
    for f in "$1"/realizations/r_*/DONE; do
        [ -e "$f" ] && n=$((n + 1))
    done
    echo "$n"
}

# run_arm <pd> <arm> <seed0> <nr> <outdir> [id]: one runner invocation of the arm (flags from sf39.geometry); sets CASE_*; records ARM_RESULT
run_arm() {
    local pd=$1 arm=$2 seed0=$3 nr=$4 out=$5 id=${6:-} fl rc n
    [ -n "$id" ] || id="$arm:s$seed0:n$nr"
    fl=$(arm_flags "$arm") || exit 2
    RUN_DIRS+=("$out")
    if halted "$arm"; then
        res "ARM_HALTED arm=$arm (listed in $OUT_ROOT/halted_arms.txt after a runner exit 3/4; owner decision pending; not run)"
        BAD=$((BAD + 1))
        CASE_RC=halted
        CASE_STATE=halted
        return 0
    fi
    # shellcheck disable=SC2086  # $fl is a flag string without spaces inside values or glob characters
    case_run "$pd" "$id" "$FINAL_RUNNER" - - -- "$BIN_DIR/lagrangian_ensemble" $fl --seed0 "$seed0" --nr "$nr" --out "$out"
    [ "$CASE_STATE" = "dry" ] && return 0
    rc=$CASE_RC
    n=$(done_count "$out")
    res "RUN id=$id rc=$rc wall=${CASE_WALL:-na}s realizations_done=$n/$nr out=$out"
    case "$rc" in
    0)
        if [ "$n" -lt "$nr" ]; then
            res "INCOMPLETE id=$id: rc 0 but only $n of $nr realizations have DONE"
            BAD=$((BAD + 1))
        fi
        ;;
    3 | 4)
        printf '%s\t%s\t%s\t%s\n' "$arm" "$PART" "$rc" "$(date -u +%FT%TZ)" >> "$OUT_ROOT/halted_arms.txt"
        res "ARM_HALT arm=$arm rc=$rc (RR6: recorded, owner decides; other arms continue)"
        grep -E '^(R=[0-9]+ )?(B_SOLVER|STAGE_FAILED|SEED_FACE_FAILED)' "$pd/logs/${id//:/_}.log" 2> /dev/null | head -3 | cut -c1-300 | res_block
        BAD=$((BAD + 1))
        ;;
    *)
        res "UNEXPECTED_EXIT_CODE id=$id rc=$rc (recorded; a re-invocation re-runs it with identical settings unless rc is 2 or 7)"
        BAD=$((BAD + 1))
        ;;
    esac
}

res_block() {
    local ln
    while IFS= read -r ln; do
        res "$ln"
    done
}

# An unknown amplitude fails (rc 1); callers capture with `x=$(amp_arms "$s") || exit 2` (an exit inside $(...) would only end the subshell and
# leave an empty loop that finishes PART_DONE rc 0).
amp_arms() { # amp_arms <amp>: production A, the narrow rungs, the twin (pilot order)
    case "$1" in
    s025) echo "s025_A s025_n05 s025_n10 s025_n20 s025_tw" ;;
    s1) echo "s1_A s1_n05 s1_n10 s1_n20 s1_tw" ;;
    s225) echo "s225_A s225_n08 s225_n16 s225_n32 s225_tw" ;;
    s4) echo "s4_A s4_n16 s4_n32 s4_n64 s4_tw" ;;
    *)
        echo "run_sf39.sh: amp_arms: unknown amplitude '$1'" >&2
        return 1
        ;;
    esac
}

amp_narrow() {
    case "$1" in
    s025) echo "s025_n05 s025_n10 s025_n20" ;;
    s1) echo "s1_n05 s1_n10 s1_n20" ;;
    s225) echo "s225_n08 s225_n16 s225_n32" ;;
    s4) echo "s4_n16 s4_n32 s4_n64" ;;
    *)
        echo "run_sf39.sh: amp_narrow: unknown amplitude '$1'" >&2
        return 1
        ;;
    esac
}

sels() { # sels <selector|empty> -> the list of selectors to run
    if [ -z "$1" ]; then
        echo "s025 s1 s225 s4 par2"
    else
        echo "$1"
    fi
}

# Par2 ----------------------------------------------------------------------------------------------------------------------
# par2_run <pd> <id> <config> <raw-dir name> <expected realizations>: pipeline (cwd $OUT/par2) then par2_field_stats on the same config
par2_run() {
    local pd=$1 id=$2 cfg=$3 rawname=$4 nr=$5 rc n why fsf w=$OUT_ROOT/par2
    mk "$w/raw"
    case_run "$pd" "$id" " 0 " "$w" "$w/raw/$rawname" -- "$BIN_DIR/macroflow3d_pipeline" "$cfg"
    if [ "$CASE_STATE" != "dry" ]; then
        rc=$CASE_RC
        n=$(find "$w/raw/$rawname/stats" -maxdepth 1 -name 'realization_*_timeseries.csv' 2> /dev/null | wc -l | tr -d ' ')
        res "RUN id=$id rc=$rc wall=${CASE_WALL:-na}s par2_series=$n/$nr out=$w/raw/$rawname"
        if [ "$rc" != "0" ] || [ "$n" -ne "$nr" ]; then
            res "PAR2_FAILED id=$id (rc=$rc, series $n of $nr; a re-invocation re-runs it from scratch)"
            BAD=$((BAD + 1))
        fi
    fi
    case_run "$pd" "fs_$id" " 0 " "$w" - -- "$BIN_DIR/par2_field_stats" "$cfg" --nr "$nr"
    if [ "$CASE_STATE" != "dry" ]; then
        rc=$CASE_RC
        fsf=$w/field_stats_$id.txt
        # a skipped case (recorded rc 0) never re-derives the analysis input from its log
        [ "$CASE_STATE" = "ran" ] && grep '^FIELDSTAT' "$pd/logs/fs_${id//:/_}.log" > "$fsf" 2> /dev/null
        n=$(wc -l 2> /dev/null < "$fsf" | tr -d ' ')
        n=${n:-0}
        res "RUN id=fs_$id rc=$rc wall=${CASE_WALL:-na}s fieldstat_lines=$n/$nr saved=$fsf sha256=$(sha256sum "$fsf" 2> /dev/null | cut -c1-64)"
        if [ "$rc" != "0" ] || [ "$n" -ne "$nr" ]; then
            why="rc=$rc, $n of $nr lines"
            [ -e "$fsf" ] || why="$why; $fsf is missing, the recorded rc 0 is final and the file is not regenerated: remove the fs_$id record from exit_codes.tsv to regenerate it"
            res "FIELDSTAT_FAILED id=fs_$id ($why)"
            BAD=$((BAD + 1))
        fi
    fi
}

# parts ---------------------------------------------------------------------------------------------------------------------
chk_line() { # chk_line <name> <rc> <verdict> <ok: 1|0>
    res "CHECK_$1 rc=$2 verdict=$3"
    [ "$4" = "1" ] || BAD=$((BAD + 1))
}

solver_verdict() { # solver_verdict <out> -> verdict of realization 0 (solver_b.json), else the B_SOLVER stdout line of the log (exit 4)
    local out=$1 log=$2 v
    if [ -s "$out/realizations/r_0000/solver_b.json" ]; then
        "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/realizations/r_0000/solver_b.json" 2> /dev/null && return 0
    fi
    v=$(grep -E '^(R=[0-9]+ )?B_SOLVER' "$log" 2> /dev/null | head -1 | cut -c1-300)
    echo "no_solver_b_json ${v:-no_B_SOLVER_line}"
}

# tree_rev: the scripts revision that ties the ids of the tree-dependent checks (the revision source of run.env: SF39_COMMIT, else the
# mirror's .mf3d_source_rev, else git HEAD), reduced to file-name characters and 24 characters (a full hash keeps its first 24; the
# `<12 hex>-dirty` of .mf3d_source_rev is kept whole)
tree_rev() {
    printf '%s' "$(src_rev)" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-24
}

part_checks() {
    local arm out rc v ok refs_state head_out rev tid
    mk "$PD"
    rev=$(tree_rev) # a real run was vetted by checks_precheck (main)
    # (i) unit tests of the package (whole suite); the id carries the scripts revision and the log is read from the id-derived name
    tid=unittest:$rev
    case_run "$PD" "$tid" " 0 " - - -- "$PY" -m unittest discover -s "$SCRIPTS" -t "$SCRIPTS"
    if [ "$CASE_STATE" != "dry" ]; then
        rc=$CASE_RC
        v=FAILED
        ok=0
        [ "$rc" = "0" ] && v=ok && ok=1
        chk_line unittest "$rc" "$v $(grep -E '^(Ran |OK|FAILED)' "$PD/logs/${tid//:/_}.log" 2> /dev/null | tr '\n' ' ' | cut -c1-200)" "$ok"
    fi
    # (ii) plane lists
    tid=geometry:$rev
    case_run "$PD" "$tid" " 0 " - - -- "$PY" -m sf39.geometry check
    if [ "$CASE_STATE" != "dry" ]; then
        rc=$CASE_RC
        v=FAILED
        ok=0
        [ "$rc" = "0" ] && v=ok && ok=1
        chk_line geometry "$rc" "$v $(tail -n 1 "$PD/logs/${tid//:/_}.log" 2> /dev/null)" "$ok"
    fi
    # (iii) RR2 regression: reference (reused iff REGRESS_DONE), head, compare
    mk "$TMP_BUILD"
    if [ -e "$REFS_DIR/REGRESS_DONE" ]; then
        refs_state=reused
    elif [ -e "$REFS_DIR" ]; then
        refs_state=incomplete
    else
        refs_state=run
    fi
    if [ "$DRY" = "1" ]; then
        echo "DRYRUN refs_state=$refs_state ($REFS_DIR)"
        [ "$refs_state" = "incomplete" ] && echo "DRYRUN a real run STOPS the regression check: $REFS_DIR exists without REGRESS_DONE"
    fi
    if [ "$refs_state" = "incomplete" ]; then
        chk_line regress NA "REFS_DIR_WITHOUT_REGRESS_DONE ($REFS_DIR exists, interrupted? stop and report; nothing touched)" 0
    else
        if [ "$refs_state" = "run" ]; then
            case_run "$PD" regress_refs " 0 1 2 " - - -- env "REGRESS_SOURCE_TREE=$BASE_REV (SF-39 base build)" "TMPDIR=$TMP_BUILD" \
                bash "$REPO/scripts/regress/run.sh" "$BIN_DIR" "$REFS_DIR"
            if [ "$CASE_STATE" != "dry" ]; then
                [ -e "$REFS_DIR/REGRESS_DONE" ] && v=REGRESS_DONE || v=NO_REGRESS_DONE
                chk_line regress_refs "$CASE_RC" "$v" "$([ "$v" = REGRESS_DONE ] && [ "$CASE_RC" = 0 ] && echo 1 || echo 0)"
            fi
        elif [ "$DRY" != "1" ]; then
            res "CHECK_regress_refs rc=NA verdict=reused($REFS_DIR/REGRESS_DONE)"
        fi
        head_out=$PD/regress_head/out
        case_run "$PD" regress_head " 0 1 2 " - "$head_out" -- env "REGRESS_SOURCE_TREE=$BASE_REV build, scripts $(src_rev)" "TMPDIR=$TMP_BUILD" \
            bash "$REPO/scripts/regress/run.sh" "$BIN_DIR" "$head_out"
        if [ "$CASE_STATE" != "dry" ]; then
            [ -e "$head_out/REGRESS_DONE" ] && v=REGRESS_DONE || v=NO_REGRESS_DONE
            chk_line regress_head "$CASE_RC" "$v" "$([ "$v" = REGRESS_DONE ] && [ "$CASE_RC" = 0 ] && echo 1 || echo 0)"
        fi
        case_run "$PD" regress_compare " 0 " - - -- bash "$REPO/scripts/regress/compare.sh" "$REFS_DIR" "$head_out"
        if [ "$CASE_STATE" != "dry" ]; then
            v=$(tail -n 1 "$PD/logs/regress_compare.log" 2> /dev/null)
            chk_line regress "$CASE_RC" "${v:-no_output}" "$([ "$v" = "REGRESS IDENTICAL" ] && echo 1 || echo 0)"
        fi
    fi
    # (iv) RR3(ii) solve checks: seed 4901, one realization per narrow size, twin and width geometry (check arms, never data)
    for arm in $("$PY" -c 'from sf39 import geometry as g; print(" ".join(g.SOLVE_CHECK_ARMS))'); do
        out=$PD/runs/$arm
        mk "$PD/runs"
        run_arm "$PD" "$arm" "$SEED_CHECK" 1 "$out"
        if [ "$CASE_STATE" != "dry" ] && [ "$CASE_STATE" != "halted" ]; then
            v=$(solver_verdict "$out" "$PD/logs/${arm}_s${SEED_CHECK}_n1.log")
            ok=0
            case "$v" in converged | fallback) ok=1 ;; esac
            res "CHECK_solve_$arm rc=$CASE_RC verdict=$v"
            # run_arm already counted a non-zero rc; an unusable verdict with rc 0 cannot happen (the runner exits 4), still counted
            [ "$ok" = "1" ] || [ "$CASE_RC" != "0" ] || BAD=$((BAD + 1))
        fi
    done
    # (v) RR3(iii) byte check: the s4_A geometry at seed 4901, --hmax-div 8,16,32 vs 8
    mk "$PD/runs"
    run_arm "$PD" chk_s4_A "$SEED_CHECK" 1 "$PD/runs/byte_multi" "byte_multi"
    run_arm "$PD" chk_s4_A1 "$SEED_CHECK" 1 "$PD/runs/byte_single" "byte_single"
    case_run "$PD" bytecheck " 0 " - - -- "$PY" -m sf39.bytecheck "$PD/runs/byte_multi" "$PD/runs/byte_single"
    if [ "$CASE_STATE" != "dry" ]; then
        v=$(tail -n 1 "$PD/logs/bytecheck.log" 2> /dev/null)
        chk_line bytecheck "$CASE_RC" "${v:-no_output}" "$([ "$CASE_RC" = 0 ] && echo 1 || echo 0)"
    fi
    # (vi) RR3(iv) smoke of the reducer on the SF-41 outputs (not evidence); tied to the scripts revision like (i) and (ii)
    tid=smoke_reduce:$rev
    case_run "$PD" "$tid" " 0 " - - -- "$PY" -m sf39.reduce smoke "$HOME/sf41_campaign"
    if [ "$CASE_STATE" != "dry" ]; then
        chk_line smoke_reduce "$CASE_RC" "$(tail -n 1 "$PD/logs/${tid//:/_}.log" 2> /dev/null | cut -c1-200)" "$([ "$CASE_RC" = 0 ] && echo 1 || echo 0)"
    fi
    tid=signtable:$rev
    case_run "$PD" "$tid" " 0 " - - -- "$PY" -m sf39.reduce signtable "$HOME/sf41_campaign/evladder"
    if [ "$CASE_STATE" != "dry" ]; then
        chk_line signtable "$CASE_RC" "$(tail -n 1 "$PD/logs/${tid//:/_}.log" 2> /dev/null | cut -c1-200)" "$([ "$CASE_RC" = 0 ] && echo 1 || echo 0)"
    fi
}

part_pilot() {
    local s c arm arms
    mk "$OUT_ROOT/runs"
    for s in $(sels "$SEL"); do
        if [ "$s" = "par2" ]; then
            for c in $AMPS; do
                par2_run "$PD" "par2_$c" "$ART/configs/par2_$c.yaml" "par2_$c" "$PILOT_NR"
            done
        else
            arms=$(amp_arms "$s") || exit 2
            for arm in $arms; do
                run_arm "$PD" "$arm" "$SEED_PROD" "$PILOT_NR" "$OUT_ROOT/runs/$arm"
            done
        fi
    done
}

part_checkpoint() {
    local id=checkpoint
    mk "$OUT_ROOT/checkpoint"
    # the freezes are write-once: a recorded rc 0 is never re-run
    if [ -z "$SEL" ]; then
        # the original case: s1 s225 s4 (the default --amps of sf39.checkpoint; never the four amplitudes of $AMPS)
        case_run "$PD" checkpoint " 0 " - - -- "$PY" -m sf39.checkpoint --out-root "$OUT_ROOT" --artifact "$ART"
    else
        # the separate s025 case (addendum 2026-10-29): everything under checkpoint/s025/, its own write-once tau_b_star_s025.json
        id=checkpoint_$SEL
        case_run "$PD" "$id" " 0 " - - -- "$PY" -m sf39.checkpoint --out-root "$OUT_ROOT" --artifact "$ART" --amps "$SEL" --subdir "$SEL"
    fi
    if [ "$CASE_STATE" != "dry" ]; then
        res "RUN id=$id rc=$CASE_RC wall=${CASE_WALL:-na}s"
        [ "$CASE_RC" = "0" ] || BAD=$((BAD + 1))
    fi
}

part_analysis() {
    mk "$OUT_ROOT/analysis"
    # analysis is deterministic and idempotent: every invocation re-runs it (finals = ALWAYS)
    case_run "$PD" analysis ALWAYS - - -- "$PY" -m sf39.analyze --out-root "$OUT_ROOT" --artifact "$ART" --dest "$OUT_ROOT/analysis"
    if [ "$CASE_STATE" != "dry" ]; then
        res "RUN id=analysis rc=$CASE_RC wall=${CASE_WALL:-na}s dest=$OUT_ROOT/analysis"
        [ "$CASE_RC" = "0" ] || BAD=$((BAD + 1))
    fi
}

part_width() {
    mk "$OUT_ROOT/runs"
    run_arm "$PD" s4_w24 "$SEED_W24" 30 "$OUT_ROOT/runs/s4_w24"
    run_arm "$PD" s4_w12 "$SEED_W12" 30 "$OUT_ROOT/runs/s4_w12"
    run_arm "$PD" s4_w12c "$SEED_W12C" "$N_W12C" "$OUT_ROOT/runs/s4_w12c"
}

# budget.json (written after the owner's budget): {"s025": {...}, "s1": {...}, "s225": {...}, "s4": {...}}, each with the integer keys
# N_narrow N_twin N_hmax N_R N_A N_run par2_NR (README documents the schema).  Extra keys are ignored.
# budget_check <required amplitude...>: every amplitude of $AMPS present in the file must be valid; the listed ones must be present
# (the s025 entry exists only after the s025 checkpoint, so an `extend` of the other amplitudes does not need it).
budget_check() {
    "$PY" - "$BUDGET" "$AMPS" "$*" << 'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
amps, required = sys.argv[2].split(), sys.argv[3].split()
keys = ("N_narrow", "N_twin", "N_hmax", "N_R", "N_A", "N_run", "par2_NR")
bad = []
for a in amps:
    if a not in d:
        if a in required:
            bad.append("amplitude %s missing" % a)
        continue
    n0 = len(bad)
    b = d[a]
    for k in keys:
        if not isinstance(b.get(k), int) or isinstance(b.get(k), bool) or b[k] < 1:
            bad.append("%s.%s must be a positive integer" % (a, k))
    if len(bad) > n0:
        continue
    for k in ("N_narrow", "N_twin", "N_R", "N_A", "par2_NR"):
        if b[k] < 20:
            bad.append("%s.%s < 20 (the pilot count)" % (a, k))
    if b["N_A"] < b["N_hmax"]:
        bad.append("%s: N_A < N_hmax" % a)
    if b["N_run"] < max(b["N_A"], b["N_R"], b["N_twin"]):
        bad.append("%s: N_run < max(N_A, N_R, N_twin)" % a)
for m in bad:
    print(m)
sys.exit(1 if bad else 0)
EOF
}

bud() { # bud <amp> <key>
    "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' "$BUDGET" "$1" "$2"
}

# checks_precheck: a real `checks` run needs a revision that identifies the tree, verified BEFORE the part starts (a refusal leaves the part
# directory untouched).  Refused: empty or blank, `unknown`, or ending in `-dirty` (a dirty tree keeps its revision after an edit, so the
# tree-tied check ids would not re-run).
checks_precheck() {
    local raw rev
    [ "$DRY" = "1" ] && return 0
    raw=$(src_rev)
    rev=${raw//[[:space:]]/}
    case "$rev" in
    "" | unknown | *-dirty)
        echo "run_sf39.sh: checks: the scripts revision '$raw' does not identify the tree (empty, unknown or -dirty; sources: SF39_COMMIT, the mirror's .mf3d_source_rev, the HEAD of the repository whose top level is $REPO); the ids of the tree-dependent checks would not follow the tree; refusing" >&2
        exit 2
        ;;
    esac
}

# extend_precheck: everything `extend` needs, verified BEFORE the part starts (a refusal leaves the part directory untouched).
extend_precheck() {
    local s c nR cfg req
    if [ ! -s "$BUDGET" ]; then
        echo "run_sf39.sh: extend: $BUDGET is absent: it is created only after the owner's budget (README, budget.json schema); refusing" >&2
        if [ "$DRY" = "1" ]; then
            echo "# DRYRUN extend: no budget file, so no commands can be resolved (a real run refuses); set SF39_BUDGET=<file> for a test budget"
            exit 0
        fi
        exit 2
    fi
    # the amplitudes this invocation runs need a complete entry: the selected one, else (no selector, par2) every amplitude of $AMPS
    case " $AMPS " in *" $SEL "*) req=$SEL ;; *) req=$AMPS ;; esac
    # shellcheck disable=SC2086  # $req is a list of amplitude ids
    if ! budget_check $req; then
        echo "run_sf39.sh: extend: $BUDGET is invalid or lacks an amplitude this invocation runs (see above); refusing" >&2
        exit 2
    fi
    for s in $(sels "$SEL"); do
        [ "$s" = "par2" ] || continue
        for c in $AMPS; do
            nR=$(bud "$c" par2_NR)
            [ "$nR" -gt "$PILOT_NR" ] || continue
            cfg=$ART/configs/par2_${c}_ext.yaml
            if [ ! -s "$cfg" ]; then
                echo "run_sf39.sh: extend par2: $cfg is absent (create it with: $PY -m sf39.par2 mkext $ART/configs/par2_$c.yaml $nR)" >&2
            elif ! cmp -s <("$PY" -m sf39.par2 mkext "$ART/configs/par2_$c.yaml" "$nR" 2> /dev/null) "$cfg"; then
                echo "run_sf39.sh: extend par2: $cfg differs from the output of sf39.par2 mkext (par2_$c.yaml, $nR); refusing" >&2
            else
                continue
            fi
            if [ "$DRY" = "1" ]; then
                echo "# DRYRUN extend par2 $c: a real run refuses here"
                continue
            fi
            exit 2
        done
    done
}

part_extend() {
    local s c arm narrow nA nN nT nR nRun out
    mk "$OUT_ROOT/runs"
    for s in $(sels "$SEL"); do
        if [ "$s" = "par2" ]; then
            for c in $AMPS; do
                nR=$(bud "$c" par2_NR)
                if [ "$nR" -le "$PILOT_NR" ]; then
                    res "PAR2_EXT_NOT_NEEDED amplitude=$c par2_NR=$nR <= pilot $PILOT_NR"
                    continue
                fi
                par2_run "$PD" "par2_${c}_ext" "$ART/configs/par2_${c}_ext.yaml" "par2_${c}_ext" "$((nR - PILOT_NR))"
            done
            continue
        fi
        c=$s
        narrow=$(amp_narrow "$c") || exit 2
        nA=$(bud "$c" N_A)
        nN=$(bud "$c" N_narrow)
        nT=$(bud "$c" N_twin)
        nRun=$(bud "$c" N_run)
        run_arm "$PD" "${c}_A" "$SEED_PROD" "$nA" "$OUT_ROOT/runs/${c}_A"
        if [ "$nRun" -gt "$nA" ]; then
            run_arm "$PD" "${c}_B" "$((SEED_PROD + nA))" "$((nRun - nA))" "$OUT_ROOT/runs/${c}_B"
        fi
        for arm in $narrow; do
            out=$nN
            [ "$arm" = "s4_n16" ] && [ "$out" -lt 30 ] && out=30
            run_arm "$PD" "$arm" "$SEED_PROD" "$out" "$OUT_ROOT/runs/$arm"
        done
        out=$nT
        [ "$c" = "s4" ] && [ "$out" -lt 30 ] && out=30
        run_arm "$PD" "${c}_tw" "$SEED_PROD" "$out" "$OUT_ROOT/runs/${c}_tw"
    done
}

# main ----------------------------------------------------------------------------------------------------------------------
case "$ARG_PART" in
checks | width | analysis)
    if [ -n "$SEL" ]; then
        usage
        exit 2
    fi
    ;;
checkpoint)
    case "$SEL" in "" | s025) ;; *)
        usage
        exit 2
        ;;
    esac
    ;;
pilot | extend)
    case "$SEL" in "" | s025 | s1 | s225 | s4 | par2) ;; *)
        usage
        exit 2
        ;;
    esac
    ;;
*)
    usage
    exit 2
    ;;
esac
if [ -z "$ARG_BIN_DIR" ]; then
    usage
    exit 2
fi
cd "$REPO" || exit 2
if BIN_DIR=$(cd "$ARG_BIN_DIR" 2> /dev/null && pwd); then
    :
elif [ "$DRY" = "1" ]; then
    BIN_DIR=$ARG_BIN_DIR
else
    echo "run_sf39.sh: build_dir '$ARG_BIN_DIR' not found" >&2
    exit 2
fi

[ "$ARG_PART" = "extend" ] && extend_precheck
[ "$ARG_PART" = "checks" ] && checks_precheck
part_begin "$ARG_PART"
case "$ARG_PART" in
checks) part_checks ;;
pilot) part_pilot ;;
checkpoint) part_checkpoint ;;
extend) part_extend ;;
width) part_width ;;
analysis) part_analysis ;;
esac
part_finish
exit $?
