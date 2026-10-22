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

# ============================================================================================================
# N6b: forward / bytecmp / digest parts: v2, v3a, v1, v3b, bytecmp, digest   (appended; the lines above are frozen)
#
#   bash <A>/scripts/run_sf37.sh <bin_dir> v2       V2/lambda-h: eps {1,1.5,2} x seeds 3101..3103 x N {40,80,32,64,128,256,512}, primary
#                                                   setting (--tol 1e-8 --hmax-div 8), planes/times ladder; --save-vd for eps 1.5/2, N >= 128
#   bash <A>/scripts/run_sf37.sh <bin_dir> v3a      O3a/criterion C: eps 1, N {64,128,256}, planes 0.25,0.5,0.75,1, NO --times
#   bash <A>/scripts/run_sf37.sh <bin_dir> v1       V1/V5: eps x seeds x N {256,512} x tol {1e-6..1e-10} at --hmax-div 8, then
#                                                   --tol 1e-10 --hmax-div 32; all with --covectors --save-seed-velocity
#   bash <A>/scripts/run_sf37.sh <bin_dir> v3b      V3b: eps x seeds x N {64,128,256,512}, primary setting + --comparator pollock
#   bash <A>/scripts/run_sf37.sh <bin_dir> bytecmp  criterion R (RR13): head binary vs $SF37_BASE_REFS (job sf37-base-refs)
#   bash <A>/scripts/run_sf37.sh <bin_dir> digest   the four digests on $SF37_OUT_ROOT (cheap; no GPU)
#   (<bin_dir> for these parts = build/v100-release of the integrated head; for `digest` it is accepted but unused, although the
#   dispatcher above still requires <bin_dir>/inlet_slab to exist and records its md5 in run.env.)
#
# Common settings of the forward parts (all pre-registered; nothing is overridable, nothing is retried with other settings):
#   --track forward --sigma2 1 --ell 0.25 --seed S --n N --eps E --np-side 256 --max-arclength 1000
#   ladder planes --planes 0.25,0.5,1,2,4,8,16 and times --times 0.25,0.5,1,2,4,8,16 (v3a: planes 0.25,0.5,0.75,1 only);
#   primary setting --tol 1e-8 --hmax-div 8 (v1 varies it as above).  E codes 10/15/20 stand for eps 1/1.5/2 (the driver gets
#   --eps 1 | 1.5 | 2); S in 3101 3102 3103.  The forward tracker does not use --threads (none is passed).
#
# Layout (contract C-LAYOUT, RR11), part P in {v2, v3a, v1, v3b}, run name R (sf37lib.v2_name / v1_name; v3a_e{E}_s{S}_N{N},
# v3b_e{E}_s{S}_N{N}):
#   $OUT_ROOT/P/R/            --save-particles directory (with --save-vd: $OUT_ROOT/P/R/vd)
#   $OUT_ROOT/P/R.log         stdout + stderr         $OUT_ROOT/P/R.json   driver --summary
#   $OUT_ROOT/P/R.time        /usr/bin/time -v        $OUT_ROOT/P/R.mem.gz  nvidia-smi memory.used every 200 ms (N = 512 runs only)
#   $OUT_ROOT/P/exit_codes.tsv  EXACTLY two tab-separated columns `R<TAB>rc`, no header (the digests take the last column)
#   $OUT_ROOT/P/walls.tsv       `R<TAB>wall_s<TAB>end_utc`, no header      $OUT_ROOT/P/run.env  (COMMIT, BIN, BIN_MD5, host, GPU, ...)
#   $OUT_ROOT/P/sha256.txt      sha256 of every file of each finished run     $OUT_ROOT/P/resumed.txt  runs re-run after an interruption
# Every finished run prints `CASE_RESULT run= rc= wall= end=`; N = 512 runs also `MEM_PEAK run= mib=`.
# Expected exit codes: 0; 22 (track_fail) and 18 (darcy_failed) are recorded outcomes (not retried); v3b also 23
# (comparator_fail).  Any other code is a failure: recorded, the part continues, the final exit status is non-zero.
# Resume (RR12): a run with a recorded rc is finished (SKIP_DONE; a recorded rc 0/22/23 without summary JSON prints
# SKIP_DONE_NOJSON and is counted as a failure, it is not re-run); a run with partial outputs and NO recorded rc is removed and
# re-run with identical settings (RESUMED, listed in resumed.txt).  A SIGINT/SIGTERM records no rc (re-invocation resumes).
#
# bytecmp (criterion R): re-runs, with $BIN and IDENTICAL arguments, the commands of the job sf37-base-refs into
# $OUT_ROOT/bytecmp/{trace,fwd,neutral}/ and compares with $SF37_BASE_REFS/{trace,fwd} (+ bin/inlet_slab):
#   trace      N{32,64}_e{0.5,1.0}_{host,gpu}, N64_e2.0_gpu: psi1.npy psi2.npy (cmp); labels.json without transport.seconds_*;
#              lines CASE PLANE BACKFLOW GROWTH RESIDUAL TRANSPORT_STATUS STATUS (lines_cmp.py); exit code (.time of the reference)
#   production --production --n 32 --eps 0.25 ... --oracle-ladder: lines CASE ORACLE ORACLE_SUMMARY EXTRA vs
#              $SF37_BASE_REFS/trace/prod32.log (or $SF37_BASE_REFS/prod32.log); exit code vs the reference .time when present, else 0
#   fwd        the seven --track forward cases: every file of the reference directory (cmp; track.json without timings_s via
#              json_strip_cmp.py), same file set, lines SETUP TRACK BACKFLOW TPLANE TTIME TSTATUS STATUS, exit code
#   neutral    gauss_n64_e1_cov, gauss_n64_e2, gauss_n64_e2_tol re-run with --comparator pollock --save-vd <dir>/vd
#              --save-seed-velocity: every PRE-EXISTING file/line identical (new pollock_*, seed_v, vd files ignored); exit code
#              equal to the reference or 23 (reported)
#   binary     md5 of $BIN differs from $SF37_BASE_REFS/bin/inlet_slab; the usage text lists --comparator
# Never compared: summary command fields, TIMING/MEMORY lines, stage seconds, nvidia-smi peaks.  bytecmp is always re-run from
# scratch (cheap; no resume).  Output $OUT_ROOT/bytecmp/results.txt: one line per comparison `IDENTICAL|DIFF|MISSING <item>`, then
# `BYTECMP_ALL_IDENTICAL` or `BYTECMP_DIFF n=<k>`; exit non-zero on any DIFF/MISSING.  The base references are mandatory in a
# campaign (absent -> MISSING); in a dry run an absent $SF37_BASE_REFS gives `SKIP` lines and `BYTECMP_SKIPPED`, exit 0.
#
# digest: from the repository root, for NAME in digest_v1v2 (-> analysis/v1v2), digest_v4v5 (-> analysis/v4v5), digest_v3
# (-> analysis/v3), o3 (-> analysis/o3): `python3 <A>/scripts/NAME.py $OUT_ROOT $OUT_ROOT/analysis/<dir> --design full`;
# log $OUT_ROOT/analysis/<dir>.log, /usr/bin/time file <dir>.time, rc in $OUT_ROOT/analysis/exit_codes.tsv (two columns).  Non-zero
# exit if any digest exits non-zero.  Dry run: --design discovered.
#
# SF37_DRYRUN=1 (local smoke only; NOT campaign evidence): seeds 3001 only; --np-side 16; planes 0.25,0.5,1,2 and times 0.5,1,2
# (v3a planes 0.25,0.5,0.75,1); v2 N {16,32,64} (--save-vd at 32 and 64 for eps 1.5/2); v3a N {16,32,64}; v1 N {32,64};
# v3b N {16,32,64}; the GPU memory sampler runs for N >= 64 (instead of 512) to exercise its code path; the eps list, the tolerance
# ladder and the bytecmp commands are unchanged.  Unset, nothing changes.
# ============================================================================================================
FWD_RC=" 0 18 22 "
FWD_RC_V3B=" 0 18 22 23 "
FWD_EPS="10:1 15:1.5 20:2"
FWD_SEEDS="3101 3102 3103"
FWD_NPSIDE=256
FWD_PLANES="0.25,0.5,1,2,4,8,16"
FWD_TIMES="0.25,0.5,1,2,4,8,16"
FWD_MEMN=512
SAMPLER_PID=""

fwd_settings() { # dry-run overrides (the campaign values above are the defaults)
    if [ "$DRY" = "1" ]; then
        FWD_SEEDS="3001"
        FWD_NPSIDE=16
        FWD_PLANES="0.25,0.5,1,2"
        FWD_TIMES="0.5,1,2"
        FWD_MEMN=64
    fi
}

fwd_sampler_start() { # fwd_sampler_start <file>   (nvidia-smi memory.used every 200 ms on the physical GPU)
    SAMPLER_PID=""
    if ! command -v nvidia-smi > /dev/null 2>&1; then
        echo "WARNING: nvidia-smi not found: no device-peak sample (recorded as empty file $1)"
        : > "$1"
        return 0
    fi
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" -lms 200 > "$1" 2> /dev/null &
    SAMPLER_PID=$!
    sleep 1
}

fwd_sampler_stop() {
    sleep 1
    if [ -n "$SAMPLER_PID" ]; then
        kill "$SAMPLER_PID" 2> /dev/null
        wait "$SAMPLER_PID" 2> /dev/null
    fi
    SAMPLER_PID=""
}

fwd_peak_of() { # fwd_peak_of <file> -> max integer (MiB) or empty
    [ -s "$1" ] && sort -n "$1" | tail -n 1
}

fwd_on_signal() { # the sampler must not outlive an interrupted job
    [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2> /dev/null
    on_signal
}

fwd_part_begin() { # fwd_part_begin <part>
    local part=$1
    mkdir -p "$OUT_ROOT/$part"
    trap fwd_on_signal INT TERM
    write_env "$OUT_ROOT/$part/run.env" "$part"
}

# fwd_case <part> <expected_rc_list> <run> <eps> <seed> <N> <vd 0|1> <extra driver args...>
fwd_case() {
    local part=$1 expected=$2 run=$3 eps=$4 seed=$5 n=$6 vd=$7
    shift 7
    local P=$OUT_ROOT/$part
    local dir=$P/$run json=$P/$run.json log=$P/$run.log tf=$P/$run.time mem=$P/$run.mem
    local tsv=$P/exit_codes.tsv sha=$P/sha256.txt
    local rc0
    rc0=$(tsv_rc "$tsv" "$run")
    if [ -n "$rc0" ]; then
        if [ -e "$json" ]; then
            echo "SKIP_DONE run=$run rc=$rc0"
        else
            echo "SKIP_DONE_NOJSON run=$run rc=$rc0 (rc recorded, summary JSON absent: finished outcome, not re-run)"
            case "$rc0" in 0 | 22 | 23) BAD=$((BAD + 1)) ;; esac
        fi
        case "$expected" in *" $rc0 "*) ;; *) BAD=$((BAD + 1)) ;; esac
        return 0
    fi
    if [ -e "$dir" ] || [ -e "$json" ] || [ -e "$log" ] || [ -e "$tf" ] || [ -e "$mem" ] || [ -e "$mem.gz" ]; then
        echo "RESUMED run=$run (partial outputs and no recorded rc: removed, re-run with identical settings)"
        printf '%s\t%s\t%s\n' "$run" "$(date -u +%FT%TZ)" "partial outputs removed" >> "$P/resumed.txt"
        rm -rf "$dir" "$json" "$log" "$tf" "$mem" "$mem.gz"
    fi
    mkdir -p "$P"
    local args=(--track forward --sigma2 1 --ell 0.25 --seed "$seed" --n "$n" --eps "$eps" --np-side "$FWD_NPSIDE"
        --max-arclength 1000 "$@")
    [ "$vd" = "1" ] && args+=(--save-vd "$dir/vd")
    local usemem=0
    [ "$n" -ge "$FWD_MEMN" ] && usemem=1
    echo "+ /usr/bin/time -v -o $tf $BIN ${args[*]} --save-particles $dir --summary $json > $log 2>&1"
    [ "$usemem" = "1" ] && fwd_sampler_start "$mem"
    local t0 rc
    t0=$(date +%s)
    timed "$tf" "$BIN" "${args[@]}" --save-particles "$dir" --summary "$json" > "$log" 2>&1 &
    CHILD=$!
    wait "$CHILD"
    rc=$?
    CHILD=""
    local wall=$(( $(date +%s) - t0 )) end
    end=$(date -u +%FT%TZ)
    if [ "$usemem" = "1" ]; then
        fwd_sampler_stop
        echo "  MEM_PEAK run=$run mib=$(fwd_peak_of "$mem" || true)"
        gzip -f "$mem"
    fi
    printf '%s\t%s\n' "$run" "$rc" >> "$tsv"
    printf '%s\t%s\t%s\n' "$run" "$wall" "$end" >> "$P/walls.tsv"
    append_sha "$sha" "$P" "$dir" "$json" "$log" "$tf" "$mem.gz"
    echo "  CASE_RESULT run=$run rc=$rc wall=${wall}s end=$end"
    grep -E '^(STATUS|PSTATUS)' "$log" | cut -c1-200 | sed 's/^/    /'
    case "$expected" in
    *" $rc "*) ;;
    *)
        echo "  UNEXPECTED_EXIT_CODE run=$run rc=$rc (recorded, not retried)"
        BAD=$((BAD + 1))
        ;;
    esac
}

part_v2() {
    fwd_settings
    local ns="40 80 32 64 128 256 512" vdns="128 256 512" ed elab eps s n vd
    if [ "$DRY" = "1" ]; then
        ns="16 32 64"
        vdns="32 64"
    fi
    fwd_part_begin v2
    for ed in $FWD_EPS; do
        elab=${ed%%:*}
        eps=${ed#*:}
        for s in $FWD_SEEDS; do
            for n in $ns; do
                vd=0
                if [ "$elab" != "10" ]; then
                    case " $vdns " in *" $n "*) vd=1 ;; esac
                fi
                fwd_case v2 "$FWD_RC" "v2_e${elab}_s${s}_N${n}" "$eps" "$s" "$n" "$vd" \
                    --planes "$FWD_PLANES" --times "$FWD_TIMES" --tol 1e-8 --hmax-div 8
            done
        done
    done
}

part_v3a() {
    fwd_settings
    local ns="64 128 256" s n
    [ "$DRY" = "1" ] && ns="16 32 64"
    fwd_part_begin v3a
    for s in $FWD_SEEDS; do
        for n in $ns; do
            fwd_case v3a "$FWD_RC" "v3a_e10_s${s}_N${n}" 1 "$s" "$n" 0 \
                --planes 0.25,0.5,0.75,1 --tol 1e-8 --hmax-div 8
        done
    done
}

part_v1() {
    fwd_settings
    local ns="256 512" ed elab eps s n t
    [ "$DRY" = "1" ] && ns="32 64"
    fwd_part_begin v1
    for ed in $FWD_EPS; do
        elab=${ed%%:*}
        eps=${ed#*:}
        for s in $FWD_SEEDS; do
            for n in $ns; do
                for t in 1e-6 1e-7 1e-8 1e-9 1e-10; do
                    fwd_case v1 "$FWD_RC" "v1_e${elab}_s${s}_N${n}_tol${t}_h8" "$eps" "$s" "$n" 0 \
                        --planes "$FWD_PLANES" --times "$FWD_TIMES" --tol "$t" --hmax-div 8 --covectors --save-seed-velocity
                done
                fwd_case v1 "$FWD_RC" "v1_e${elab}_s${s}_N${n}_tol1e-10_h32" "$eps" "$s" "$n" 0 \
                    --planes "$FWD_PLANES" --times "$FWD_TIMES" --tol 1e-10 --hmax-div 32 --covectors --save-seed-velocity
            done
        done
    done
}

part_v3b() {
    fwd_settings
    local ns="64 128 256 512" ed elab eps s n
    [ "$DRY" = "1" ] && ns="16 32 64"
    fwd_part_begin v3b
    for ed in $FWD_EPS; do
        elab=${ed%%:*}
        eps=${ed#*:}
        for s in $FWD_SEEDS; do
            for n in $ns; do
                fwd_case v3b "$FWD_RC_V3B" "v3b_e${elab}_s${s}_N${n}" "$eps" "$s" "$n" 0 \
                    --planes "$FWD_PLANES" --times "$FWD_TIMES" --tol 1e-8 --hmax-div 8 --comparator pollock
            done
        done
    done
}

# ------------------------------------------------------------------------------------------------------------
# bytecmp (criterion R)
# ------------------------------------------------------------------------------------------------------------
BC=""
BC_N=0

bc_res() { # bc_res <IDENTICAL|DIFF|MISSING|SKIP> <item...>
    echo "$*"
    echo "$*" >> "$BC/results.txt"
    case "$1" in IDENTICAL | SKIP) ;; *) BC_N=$((BC_N + 1)) ;; esac
}

bc_cmp() { # bc_cmp <item> <head_file> <ref_file>   (cmp)
    if [ ! -e "$2" ] || [ ! -e "$3" ]; then
        bc_res MISSING "$1 (head: $2 ref: $3)"
    elif cmp -s "$2" "$3"; then
        bc_res IDENTICAL "$1"
    else
        bc_res DIFF "$1"
    fi
}

bc_json() { # bc_json <item> <head_json> <ref_json> <drop spec>...
    local item=$1 a=$2 b=$3 out rc
    shift 3
    if [ ! -e "$a" ] || [ ! -e "$b" ]; then
        bc_res MISSING "$item (head: $a ref: $b)"
        return 0
    fi
    local dargs=() d
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

bc_lines() { # bc_lines <item> <head_log> <ref_log> <token>...
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

bc_run() { # bc_run <name> <log_base> <args...>  (head run; sets BC_RC; records `name<TAB>rc`)
    local name=$1 base=$2 t0
    shift 2
    echo "+ $BIN $*"
    t0=$(date +%s)
    timed "$base.time" "$BIN" "$@" > "$base.log" 2>&1 &
    CHILD=$!
    wait "$CHILD"
    BC_RC=$?
    CHILD=""
    printf '%s\t%s\n' "$name" "$BC_RC" >> "$BC/exit_codes.tsv"
    echo "  RUN name=$name rc=$BC_RC wall=$(( $(date +%s) - t0 ))s"
}

# bc_rc_check <item> <head_rc> <ref_time_file> [allow23]
bc_rc_check() {
    local item=$1 hrc=$2 brc
    brc=$(bc_rc_of "$3")
    if [ -z "$brc" ]; then
        bc_res MISSING "$item:rc (no Exit status in $3; head rc=$hrc)"
    elif [ "$brc" = "$hrc" ]; then
        bc_res IDENTICAL "$item:rc base=$brc head=$hrc"
    elif [ "${4:-}" = "allow23" ] && [ "$hrc" = "23" ]; then
        bc_res IDENTICAL "$item:rc base=$brc head=$hrc (comparator_fail, allowed for the opt-in run; REPORT)"
    else
        bc_res DIFF "$item:rc base=$brc head=$hrc"
    fi
}

bc_fwd_args() { # bc_fwd_args <name> -> BC_ARGS (the arguments of the job sf37-base-refs, without --save-particles/--summary)
    case "$1" in
    smoke_g3d_n16) BC_ARGS=(--analytic generic3d --n 16 --eps 0.25 --np-side 16 --planes "0.25,0.5,1" --times "0.5,1" --covectors) ;;
    g3d_n32_cov) BC_ARGS=(--analytic generic3d --n 32 --eps 0.25 --np-side 32 --planes "0.5,1" --times 1 --covectors) ;;
    gauss_n64_e1_cov) BC_ARGS=(--sigma2 1 --ell 0.25 --seed 3001 --n 64 --eps 1 --np-side 64 --planes "0.25,0.5,1,2" --times "0.5,1,2" --covectors) ;;
    gauss_n64_e2) BC_ARGS=(--sigma2 1 --ell 0.25 --seed 3001 --n 64 --eps 2 --np-side 64 --planes "0.25,0.5,1,2,4" --times "0.5,1,2,4") ;;
    gauss_n64_e2_tol) BC_ARGS=(--sigma2 1 --ell 0.25 --seed 3001 --n 64 --eps 2 --np-side 64 --planes "0.5,1,2" --times "1,2" --tol 1e-10 --hmax-div 32 --max-arclength 1000) ;;
    gauss_n40_e1) BC_ARGS=(--sigma2 1 --ell 0.25 --seed 3001 --n 40 --eps 1 --np-side 32 --planes "0.5,1" --times 1) ;;
    gauss_n80_e2) BC_ARGS=(--sigma2 1 --ell 0.25 --seed 3001 --n 80 --eps 2 --np-side 32 --planes "0.5,1" --times 1) ;;
    *) return 1 ;;
    esac
}

# bc_dir_files <item> <head_dir> <ref_dir> <strict 0|1>: every file of the reference directory must exist and be identical
# (cmp; *.json without timings_s anywhere); strict = the head directory must not hold files the reference lacks.
bc_dir_files() {
    local item=$1 hd=$2 rd=$3 strict=$4 f
    if [ ! -d "$rd" ] || [ ! -d "$hd" ]; then
        bc_res MISSING "$item (head dir: $hd ref dir: $rd)"
        return 0
    fi
    while IFS= read -r f; do
        case "$f" in
        *.json) bc_json "$item:$f" "$hd/$f" "$rd/$f" '**.timings_s' ;;
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

bc_trace_case() { # bc_trace_case <N> <eps> <tracer> <name>
    local n=$1 eps=$2 tracer=$3 name=$4
    local R=$BASE_REFS/trace
    bc_run "trace_$name" "$BC/trace/$name" --construct trace --sigma2 1 --ell 0.25 --seed 3001 --threads 32 --n "$n" --eps "$eps" \
        --tracer "$tracer" --save-labels "$BC/trace/$name" --summary "$BC/trace/$name.json"
    local hrc=$BC_RC f
    for f in psi1.npy psi2.npy; do
        bc_cmp "trace:$name:$f" "$BC/trace/$name/$f" "$R/$name/$f"
    done
    bc_json "trace:$name:labels.json" "$BC/trace/$name/labels.json" "$R/$name/labels.json" 'transport.seconds_*'
    bc_lines "trace:$name:lines" "$BC/trace/$name.log" "$R/$name.log" CASE PLANE BACKFLOW GROWTH RESIDUAL TRANSPORT_STATUS STATUS
    bc_rc_check "trace:$name" "$hrc" "$R/$name.time"
}

bc_fwd_case() { # bc_fwd_case <name> <neutral 0|1>
    local name=$1 neutral=$2 dir sub=fwd R=$BASE_REFS/fwd
    [ "$neutral" = "1" ] && sub=neutral
    dir=$BC/$sub/$name
    if ! bc_fwd_args "$name"; then
        bc_res MISSING "$sub:$name (no argument set)"
        return 0
    fi
    local extra=()
    [ "$neutral" = "1" ] && extra=(--comparator pollock --save-vd "$dir/vd" --save-seed-velocity)
    bc_run "${sub}_$name" "$BC/$sub/$name" --track forward "${BC_ARGS[@]}" ${extra[@]+"${extra[@]}"} \
        --save-particles "$dir" --summary "$BC/$sub/$name.json"
    local hrc=$BC_RC
    if [ "$neutral" = "1" ]; then
        bc_dir_files "$sub:$name" "$dir" "$R/$name" 0
        bc_lines "$sub:$name:lines" "$BC/$sub/$name.log" "$R/$name.log" SETUP TRACK BACKFLOW TPLANE TTIME TSTATUS STATUS
        bc_rc_check "$sub:$name" "$hrc" "$R/$name.time" allow23
    else
        bc_dir_files "$sub:$name" "$dir" "$R/$name" 1
        bc_lines "$sub:$name:lines" "$BC/$sub/$name.log" "$R/$name.log" SETUP TRACK BACKFLOW TPLANE TTIME TSTATUS STATUS
        bc_rc_check "$sub:$name" "$hrc" "$R/$name.time"
    fi
}

part_bytecmp() {
    BC=$OUT_ROOT/bytecmp
    BC_N=0
    rm -rf "$BC"
    mkdir -p "$BC/trace" "$BC/fwd" "$BC/neutral"
    : > "$BC/exit_codes.tsv"
    : > "$BC/results.txt"
    trap fwd_on_signal INT TERM
    write_env "$BC/run.env" bytecmp
    if [ ! -d "$BASE_REFS/trace" ] || [ ! -d "$BASE_REFS/fwd" ] || [ ! -x "$BASE_REFS/bin/inlet_slab" ]; then
        if [ "$DRY" = "1" ]; then
            bc_res SKIP "base_refs $BASE_REFS absent or incomplete (dry run: nothing compared)"
            echo "BYTECMP_SKIPPED" | tee -a "$BC/results.txt"
            return 0
        fi
        bc_res MISSING "base_refs $BASE_REFS (need trace/, fwd/, bin/inlet_slab): criterion R cannot be evaluated"
        echo "BYTECMP_DIFF n=$BC_N" | tee -a "$BC/results.txt"
        BAD=$((BAD + 1))
        return 0
    fi
    local N e t
    for N in 32 64; do
        for e in 0.5 1.0; do
            for t in host gpu; do
                bc_trace_case "$N" "$e" "$t" "N${N}_e${e}_${t}"
            done
        done
    done
    bc_trace_case 64 2 gpu N64_e2.0_gpu
    # production regression (SF-33 N32 ladder log); the reference run used no --threads, so none here
    bc_run prod32 "$BC/prod32" --production --n 32 --eps 0.25 --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder
    local phrc=$BC_RC pref=$BASE_REFS/trace/prod32
    [ -e "$pref.log" ] || pref=$BASE_REFS/prod32
    bc_lines "production:prod32:lines" "$BC/prod32.log" "$pref.log" CASE ORACLE ORACLE_SUMMARY EXTRA
    if [ -e "$pref.time" ]; then
        bc_rc_check "production:prod32" "$phrc" "$pref.time"
    elif [ "$phrc" = "0" ]; then
        bc_res IDENTICAL "production:prod32:rc head=0 (no reference .time; 0 expected)"
    else
        bc_res DIFF "production:prod32:rc head=$phrc (no reference .time; 0 expected)"
    fi
    local c
    for c in smoke_g3d_n16 g3d_n32_cov gauss_n64_e1_cov gauss_n64_e2 gauss_n64_e2_tol gauss_n40_e1 gauss_n80_e2; do
        bc_fwd_case "$c" 0
    done
    for c in gauss_n64_e1_cov gauss_n64_e2 gauss_n64_e2_tol; do
        bc_fwd_case "$c" 1
    done
    # binary check
    local hmd5 bmd5 usage
    hmd5=$(md5sum "$BIN" | awk '{print $1}')
    bmd5=$(md5sum "$BASE_REFS/bin/inlet_slab" | awk '{print $1}')
    if [ "$hmd5" != "$bmd5" ]; then
        bc_res IDENTICAL "binary:md5_differs head=$hmd5 base=$bmd5"
    else
        bc_res DIFF "binary:md5_differs (head and base binaries are the SAME file content $hmd5)"
    fi
    usage=$("$BIN" --sf37-invalid-option 2>&1 || true)
    if echo "$usage" | grep -q -- '--comparator'; then
        bc_res IDENTICAL "binary:usage_lists_--comparator"
    else
        bc_res DIFF "binary:usage_lists_--comparator"
    fi
    append_sha "$BC/sha256.txt" "$BC" "$BC/trace" "$BC/fwd" "$BC/neutral" "$BC/prod32.log" "$BC/prod32.time"
    if [ "$BC_N" -eq 0 ]; then
        echo "BYTECMP_ALL_IDENTICAL" | tee -a "$BC/results.txt"
    else
        echo "BYTECMP_DIFF n=$BC_N" | tee -a "$BC/results.txt"
        BAD=$((BAD + 1))
    fi
}

# ------------------------------------------------------------------------------------------------------------
# digest
# ------------------------------------------------------------------------------------------------------------
part_digest() {
    local design=full AN=$OUT_ROOT/analysis REPO pair script dir rc
    [ "$DRY" = "1" ] && design=discovered
    REPO=$(cd "$ART/../../../.." && pwd)
    mkdir -p "$AN"
    trap fwd_on_signal INT TERM
    write_env "$AN/run.env" digest
    : > "$AN/exit_codes.tsv"
    for pair in digest_v1v2:v1v2 digest_v4v5:v4v5 digest_v3:v3 o3:o3; do
        script=${pair%%:*}
        dir=${pair#*:}
        echo "+ (cd $REPO && $PY $SCRIPTS/$script.py $OUT_ROOT $AN/$dir --design $design) > $AN/$dir.log 2>&1"
        (cd "$REPO" && timed "$AN/$dir.time" "$PY" "$SCRIPTS/$script.py" "$OUT_ROOT" "$AN/$dir" --design "$design") > "$AN/$dir.log" 2>&1 &
        CHILD=$!
        wait "$CHILD"
        rc=$?
        CHILD=""
        printf '%s\t%s\n' "$dir" "$rc" >> "$AN/exit_codes.tsv"
        echo "  DIGEST $dir rc=$rc (log $AN/$dir.log)"
        tail -n 3 "$AN/$dir.log" | cut -c1-240 | sed 's/^/    /'
        [ "$rc" -eq 0 ] || BAD=$((BAD + 1))
    done
}
