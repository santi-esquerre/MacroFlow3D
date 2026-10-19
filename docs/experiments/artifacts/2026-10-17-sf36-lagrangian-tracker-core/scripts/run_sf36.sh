#!/usr/bin/env bash
# SF-36 campaign runner (one part per detached V100 job).  Run from the repository root (the synced mirror):
#
#   bash <A>/scripts/run_sf36.sh <build_dir> controls   A, B, C, D, E1/E2 (+ H/I are ctest): the two test binaries, 64^3
#   bash <A>/scripts/run_sf36.sh <build_dir> bytecmp    F: head `inlet_slab` outputs vs the base-commit references
#   bash <A>/scripts/run_sf36.sh <build_dir> mem        G: device peak at 256^3 eps 2 --construct trace vs base / probe
#   bash <A>/scripts/run_sf36.sh <build_dir> cost       J: --track forward at 64/128/256^3 x eps 1,2 x {plain, covectors}
#   bash <A>/scripts/run_sf36.sh <build_dir> stress     N5 finding (informative): inlet-backflow stress field, 30 min cap
#   bash <A>/scripts/run_sf36.sh <build_dir> all        controls, bytecmp, mem, cost, stress in that order
#
# <A> = docs/experiments/artifacts/2026-10-17-sf36-lagrangian-tracker-core.
#
# Logs go to $OUT = ${SF36_OUT_ROOT:-$HOME/sf36_campaign}/<part>/ ON THE HOST (not into the repository tree); the
# base-commit references are read from ${SF36_BASE_REFS:-$HOME/sf36_base_refs} (job sf36-base-refs).  Every command
# is printed, every exit code is recorded in $OUT/exit_codes.tsv, and a failing run is NEVER retried with other
# settings.  Digests: python3 digest_sf36.py $OUTROOT <outdir> (also run, non-fatally, at the end of every part).
#
# SF36_DRYRUN=1 (local smoke only; NOT campaign evidence): every grid becomes 16^3, np-side 8, the stress timeout
# 60 s, the test binaries run without --full, and the comparisons against the base references are skipped with
# SKIP when SF36_BASE_REFS is absent (and always for the 32^3 --production comparison, whose grid is shrunk).
# Unset, the switch changes nothing.
#
# bash 4.2 (the V100 host): no empty-array expansion under `set -u` (use ${a[@]+"${a[@]}"}); no backslashes in
# `${var//pat/rep}` replacements (none are used).  `scripts/remote run` exports CUDA_VISIBLE_DEVICES=<physical gpu>.
# Exit status: 0 unless a blocking item failed (test binary non-zero, byte comparison DIFF, exit-code mismatch,
# G violated, digest crash); informative parts (cost, stress) only count unexpected statuses of their own.
set -uo pipefail

BUILD_DIR=${1:?usage: run_sf36.sh <build_dir> controls|bytecmp|mem|cost|stress|all}
PART=${2:?usage: run_sf36.sh <build_dir> controls|bytecmp|mem|cost|stress|all}
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=$(dirname "$SCRIPTS")
SF33=$(dirname "$ART")/2026-10-06-sf33-gpu-inlet-labels
BUILD=$(cd "$BUILD_DIR" && pwd)
PY=${PYTHON:-python3}
OUTROOT=${SF36_OUT_ROOT:-$HOME/sf36_campaign}
BASE=${SF36_BASE_REFS:-$HOME/sf36_base_refs}
DRY=${SF36_DRYRUN:-0}
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
FIELD="--sigma2 1 --ell 0.25 --seed 3001"
BAD=0
OUT=""

if [ "$DRY" = "1" ]; then
    echo "run_sf36.sh: SF36_DRYRUN=1 (local smoke: grids 16^3, np-side 8, stress timeout 60 s; NOT campaign evidence)"
    NS_TRACE="16"       # bytecmp F matrix grids
    NS_E2="16"          # bytecmp eps 2 grids
    N_PROD=16
    N_MEM=16
    NS_COST="16"
    NP_SIDE=8
    N_STRESS=16
    STRESS_TIMEOUT=60
    FULL_FLAG=""
else
    NS_TRACE="32 64"
    NS_E2="64 128"
    N_PROD=32
    N_MEM=256
    NS_COST="64 128 256"
    NP_SIDE=256
    N_STRESS=16
    STRESS_TIMEOUT=1800
    FULL_FLAG="--full"
fi

[ -x "$BUILD/inlet_slab" ] || { echo "run_sf36.sh: $BUILD/inlet_slab not found" >&2; exit 2; }
echo "run_sf36.sh: part=$PART build=$BUILD host=$(hostname) gpu_id=$GPU_ID start=$(date -u +%FT%TZ)"

# /usr/bin/time -v is the instrument for host max RSS; where it does not exist (a local WSL dry run) the wall time is
# measured with date and the RSS is recorded as n/a, with a loud warning.
if [ -x /usr/bin/time ]; then
    timed() { /usr/bin/time -v -o "$@"; }
else
    echo "WARNING: /usr/bin/time not found: host max RSS will be n/a (acceptable for a local dry run only)"
    timed() { # timed <time_file> cmd...
        local tf=$1 t0 rc
        shift
        t0=$(date +%s)
        "$@"
        rc=$?
        printf '\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:%02d\n\tMaximum resident set size (kbytes): n/a\n\tExit status: %d\n' \
            $(( $(date +%s) - t0 )) "$rc" > "$tf"
        return $rc
    }
fi

SAMPLER_PID=""
sampler_start() { # sampler_start <file>   (nvidia-smi memory.used every 200 ms on the physical GPU)
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
sampler_stop() {
    sleep 1
    if [ -n "$SAMPLER_PID" ]; then
        kill "$SAMPLER_PID" 2> /dev/null
        wait "$SAMPLER_PID" 2> /dev/null
    fi
    SAMPLER_PID=""
}
peak_of() { # peak_of <file> -> max integer (MiB) or empty
    [ -s "$1" ] && sort -n "$1" | tail -n 1
}

write_env() {
    mkdir -p "$OUT"
    {
        echo "COMMIT=$(cat "$ART/COMMIT" 2>/dev/null || echo unknown)"
        echo "HOST=$(hostname)"
        echo "DATE_UTC=$(date -u +%FT%TZ)"
        echo "BUILD=$BUILD"
        echo "PART=$PART"
        echo "DRYRUN=$DRY"
        echo "BASE_REFS=$BASE"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        echo "GPU=$( (nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) || true)"
    } > "$OUT/run.env"
    echo "  run.env: $(tr '\n' ' ' < "$OUT/run.env")"
}

record_rc() { # record_rc <name> <rc>
    printf '%s\t%s\n' "$1" "$2" >> "$OUT/exit_codes.tsv"
}

res() { # res <words...>: print a result line and append it to $OUT/results.txt (parsed by digest_sf36.py)
    echo "$*"
    echo "$*" >> "$OUT/results.txt"
}

LAST_RC=0
# run_logged <name> <cmd...>: log -> $OUT/<name>.log, time -> $OUT/<name>.time, rc recorded (never retried)
run_logged() {
    local name=$1
    shift
    echo "+ /usr/bin/time -v -o $OUT/$name.time $* > $OUT/$name.log 2>&1"
    local t0
    t0=$(date +%s)
    timed "$OUT/$name.time" "$@" > "$OUT/$name.log" 2>&1
    LAST_RC=$?
    record_rc "$name" "$LAST_RC"
    echo "  CASE_RESULT name=$name rc=$LAST_RC wall=$(( $(date +%s) - t0 ))s end=$(date -u +%FT%TZ)"
}

# status_from_time <time_file> -> "Exit status: N" value or "?"
rc_from_time() {
    local v
    v=$(grep 'Exit status' "$1" 2> /dev/null | awk '{print $NF}')
    echo "${v:-?}"
}

digest_all() { # non-fatal convenience digest of whatever exists under $OUTROOT
    mkdir -p "$OUTROOT/analysis"
    echo "+ $PY $SCRIPTS/digest_sf36.py $OUTROOT $OUTROOT/analysis"
    "$PY" "$SCRIPTS/digest_sf36.py" "$OUTROOT" "$OUTROOT/analysis"
    echo "  digest_sf36.py rc=$?"
}

# ------------------------------------------------------------------------------------------------------------
part_controls() {
    OUT=$OUTROOT/controls
    mkdir -p "$OUT"
    : > "$OUT/exit_codes.tsv"; : > "$OUT/results.txt"
    write_env
    local sel
    # shellcheck disable=SC2086
    run_logged controls_full "$BUILD/lagrangian_controls_tests" $FULL_FLAG
    [ "$LAST_RC" -eq 0 ] || BAD=$((BAD + 1))
    grep -E '^SUMMARY:' "$OUT/controls_full.log" | sed 's/^/    /'
    run_logged controls_diag "$BUILD/lagrangian_controls_tests" --diag
    [ "$LAST_RC" -eq 0 ] || BAD=$((BAD + 1))
    grep -E '^SUMMARY:' "$OUT/controls_diag.log" | sed 's/^/    /'
    # shellcheck disable=SC2086
    run_logged gpu_host_full "$BUILD/lagrangian_gpu_host_tests" $FULL_FLAG
    [ "$LAST_RC" -eq 0 ] || BAD=$((BAD + 1))
    grep -E '^SUMMARY:|ALL PASS|SOME FAILED' "$OUT/gpu_host_full.log" | sed 's/^/    /'
    for sel in controls_full controls_diag gpu_host_full; do
        echo "  $sel: PASS=$(grep -c '^\[PASS\]' "$OUT/$sel.log") FAIL=$(grep -c '^\[FAIL\]' "$OUT/$sel.log")" \
             "RECORD=$(grep -c '^\[RECORD\]' "$OUT/$sel.log")"
    done
}

# ------------------------------------------------------------------------------------------------------------
bytecmp_case() { # bytecmp_case <name> <inlet_slab args...>  (head run + comparison to $BASE/trace/<name>)
    local name=$1
    shift
    local ld=$OUT/labels/$name
    mkdir -p "$OUT/labels" "$OUT/raw"
    # shellcheck disable=SC2086
    run_logged "$name" "$BUILD/inlet_slab" "$@" --save-labels "$ld" --summary "$OUT/raw/$name.json"
    local hrc=$LAST_RC
    grep -E '^(CASE|BACKFLOW|GROWTH|RESIDUAL|TRANSPORT_STATUS|STATUS)' "$OUT/$name.log" | cut -c1-220 | sed 's/^/    /'
    if [ ! -d "$BASE/trace" ] || [ ! -e "$BASE/trace/$name.log" ]; then
        res "BYTECMP $name npy=SKIP lines=SKIP rc=SKIP/$hrc (base reference $BASE/trace/$name.* absent)"
        return 0
    fi
    local npy=IDENTICAL f
    for f in psi1.npy psi2.npy; do
        cmp -s "$ld/$f" "$BASE/trace/$name/$f" || npy=DIFF
    done
    local lines lrc
    lines=$("$PY" "$SCRIPTS/lines_cmp.py" "$OUT/$name.log" "$BASE/trace/$name.log" \
        CASE PLANE BACKFLOW GROWTH RESIDUAL TRANSPORT_STATUS STATUS)
    lrc=$?
    echo "$lines" | sed 's/^/    /' > "$OUT/$name.linecmp"
    local lstate=IDENTICAL
    [ "$lrc" -eq 0 ] || lstate=DIFF
    local brc
    brc=$(rc_from_time "$BASE/trace/$name.time")
    res "BYTECMP $name npy=$npy lines=$lstate rc=$brc/$hrc"
    [ "$lstate" = "DIFF" ] && head -n 6 "$OUT/$name.linecmp"
    if [ "$npy" != "IDENTICAL" ] || [ "$lstate" != "IDENTICAL" ] || [ "$brc" != "$hrc" ]; then
        BAD=$((BAD + 1))
    fi
}

part_bytecmp() {
    OUT=$OUTROOT/bytecmp
    mkdir -p "$OUT"
    : > "$OUT/exit_codes.tsv"; : > "$OUT/results.txt"
    write_env
    if [ ! -d "$BASE" ]; then
        echo "SKIP base references $BASE absent: head cases run, comparisons are skipped"
        [ "$DRY" = "1" ] || BAD=$((BAD + 1)) # a campaign without base references is not evidence of F
    fi
    local F="--construct trace $FIELD --threads 32"
    local N e t
    for N in $NS_TRACE; do
        for e in 0.5 1.0; do
            for t in host gpu; do
                # shellcheck disable=SC2086
                bytecmp_case "N${N}_e${e}_${t}" $F --n "$N" --eps "$e" --tracer "$t"
            done
        done
    done
    for N in $NS_E2; do
        # shellcheck disable=SC2086
        bytecmp_case "N${N}_e2.0_gpu" $F --n "$N" --eps 2 --tracer gpu
    done
    # --production regression (SF-33 N32 ladder log); the base run used no --threads, so none here either
    local pname=prod32
    echo "+ $BUILD/inlet_slab --production --n $N_PROD --eps 0.25 $FIELD --oracle-ladder > $OUT/$pname.log"
    run_logged "$pname" "$BUILD/inlet_slab" --production --n "$N_PROD" --eps 0.25 --sigma2 1 --ell 0.25 --seed 3001 \
        --oracle-ladder
    grep -E '^(ORACLE_SUMMARY|STATUS)' "$OUT/$pname.log" | cut -c1-220 | sed 's/^/    /'
    local ref33=$SF33/logs/ladder_0.25_c4/N32.log
    local r
    if [ "$DRY" = "1" ]; then
        res "BYTECMP prod32_vs_base npy=n/a lines=SKIP rc=SKIP/$LAST_RC (dry run: grid shrunk to $N_PROD)"
        res "BYTECMP prod32_vs_sf33 npy=n/a lines=SKIP rc=SKIP/$LAST_RC (dry run: grid shrunk to $N_PROD)"
    else
        local hrc=$LAST_RC
        if [ -e "$BASE/trace/prod32.log" ]; then
            r=$("$PY" "$SCRIPTS/lines_cmp.py" "$OUT/$pname.log" "$BASE/trace/prod32.log" CASE ORACLE ORACLE_SUMMARY EXTRA)
            local lrc=$?
            echo "$r" | sed 's/^/    /' > "$OUT/prod32_vs_base.linecmp"
            local st=IDENTICAL
            [ "$lrc" -eq 0 ] || { st=DIFF; BAD=$((BAD + 1)); head -n 6 "$OUT/prod32_vs_base.linecmp"; }
            res "BYTECMP prod32_vs_base npy=n/a lines=$st rc=?/$hrc"
        else
            res "BYTECMP prod32_vs_base npy=n/a lines=SKIP rc=SKIP/$hrc ($BASE/trace/prod32.log absent)"
            BAD=$((BAD + 1))
        fi
        if [ ! -e "$ref33" ]; then
            ref33=$(find "$(dirname "$ART")/2026-10-06-sf33-gpu-inlet-labels" -name 'N32.log' -path '*ladder_0.25*' | head -n 1)
        fi
        if [ -n "$ref33" ] && [ -e "$ref33" ]; then
            r=$("$PY" "$SCRIPTS/lines_cmp.py" "$OUT/$pname.log" "$ref33" CASE ORACLE ORACLE_SUMMARY EXTRA)
            local lrc2=$?
            echo "$r" | sed 's/^/    /' > "$OUT/prod32_vs_sf33.linecmp"
            local st2=IDENTICAL
            [ "$lrc2" -eq 0 ] || { st2=DIFF; BAD=$((BAD + 1)); head -n 6 "$OUT/prod32_vs_sf33.linecmp"; }
            res "BYTECMP prod32_vs_sf33 npy=n/a lines=$st2 rc=?/$hrc (ref $ref33)"
        else
            res "BYTECMP prod32_vs_sf33 npy=n/a lines=SKIP rc=SKIP/$hrc (SF-33 N32 ladder log not found)"
            BAD=$((BAD + 1))
        fi
    fi
}

# ------------------------------------------------------------------------------------------------------------
part_mem() {
    OUT=$OUTROOT/mem
    mkdir -p "$OUT"
    : > "$OUT/exit_codes.tsv"; : > "$OUT/results.txt"
    write_env
    sampler_start "$OUT/N${N_MEM}_e2.mem"
    # shellcheck disable=SC2086
    run_logged "N${N_MEM}_e2" "$BUILD/inlet_slab" --construct trace --n "$N_MEM" --eps 2 $FIELD --tracer gpu \
        --threads 32 --summary "$OUT/N${N_MEM}_e2.json"
    sampler_stop
    grep -E '^(STATUS|TIMING|MEMORY)' "$OUT/N${N_MEM}_e2.log" | cut -c1-220 | sed 's/^/    /'
    local head_peak base_peak probe=3769
    head_peak=$(peak_of "$OUT/N${N_MEM}_e2.mem")
    base_peak=$(peak_of "$BASE/mem/N256_e2.mem")
    res "HEAD_PEAK_MIB=${head_peak:-n/a}   (N=$N_MEM, nvidia-smi memory.used, 200 ms, physical GPU $GPU_ID)"
    if [ -n "$base_peak" ]; then
        res "BASE_PEAK_MIB=$base_peak   (max of $BASE/mem/N256_e2.mem; orchestrator recorded 6239)"
    else
        res "BASE_PEAK_MIB=6239   (recorded value of job sf36-base-refs; $BASE/mem/N256_e2.mem absent)"
        base_peak=6239
    fi
    res "PROBE_REF_MIB=$probe   (P2 probe; band +-2 % = $(awk -v p=$probe 'BEGIN{printf "%.1f..%.1f", p*0.98, p*1.02}'))"
    if [ -z "$head_peak" ]; then
        res "G_VERDICT n/a (no sample)"
    elif [ "$DRY" = "1" ]; then
        res "G_VERDICT SKIP (dry run: grid shrunk to $N_MEM; peak is not comparable)"
    else
        local le_base in_band
        le_base=$(awk -v h="$head_peak" -v b="$base_peak" 'BEGIN{print (h <= b) ? "yes" : "NO"}')
        in_band=$(awk -v h="$head_peak" -v p="$probe" 'BEGIN{d = (h - p) / p; if (d < 0) d = -d; print (d <= 0.02) ? "yes" : "NO"}')
        res "G_VERDICT head<=base: $le_base ; head within 2 % of probe $probe: $in_band"
        [ "$le_base" = "yes" ] || BAD=$((BAD + 1))
        [ "$in_band" = "yes" ] || BAD=$((BAD + 1))
    fi
    # 21 roundtrip_exceeded, 19 trace_fail, 17 inlet_backflow are recorded results of a construct run (the peak is the
    # measurement; the exit code is compared with the base in bytecmp); anything else is unexpected
    case "$LAST_RC" in
    0 | 17 | 18 | 19 | 21) ;;
    *) res "UNEXPECTED_EXIT_CODE rc=$LAST_RC (recorded, not retried)"; BAD=$((BAD + 1)) ;;
    esac
}

# ------------------------------------------------------------------------------------------------------------
part_cost() {
    OUT=$OUTROOT/cost
    mkdir -p "$OUT"
    : > "$OUT/exit_codes.tsv"; : > "$OUT/results.txt"
    write_env
    local PL=0.25,0.5,1,2,4 TM=0.25,0.5,1,2,4
    local N e cov name extra
    for N in $NS_COST; do
        for e in 1 2; do
            for cov in plain cov; do
                name=N${N}_e${e}_$cov
                extra=""
                [ "$cov" = "cov" ] && extra="--covectors"
                sampler_start "$OUT/$name.mem"
                # shellcheck disable=SC2086
                run_logged "$name" "$BUILD/inlet_slab" --track forward $FIELD --n "$N" --eps "$e" \
                    --np-side "$NP_SIDE" --planes "$PL" --times "$TM" $extra --summary "$OUT/$name.json"
                sampler_stop
                grep -E '^(TSTATUS|TIMING|MEMORY|STATUS)' "$OUT/$name.log" | cut -c1-240 | sed 's/^/    /'
                res "  COST $name rc=$LAST_RC device_peak_MiB=$(peak_of "$OUT/$name.mem")" \
                     "max_rss_kB=$(grep 'Maximum resident' "$OUT/$name.time" 2> /dev/null | awk '{print $NF}')" \
                     "wall=$(grep 'Elapsed' "$OUT/$name.time" 2> /dev/null | awk '{print $NF}')"
                # rc 0 ok, 22 track_fail and 18 darcy_failed are recorded results (informative part); others unexpected
                case "$LAST_RC" in
                0 | 18 | 22) ;;
                *) echo "  UNEXPECTED_EXIT_CODE name=$name rc=$LAST_RC (recorded, not retried)"; BAD=$((BAD + 1)) ;;
                esac
            done
        done
    done
}

# ------------------------------------------------------------------------------------------------------------
part_stress() {
    OUT=$OUTROOT/stress
    mkdir -p "$OUT"
    : > "$OUT/exit_codes.tsv"; : > "$OUT/results.txt"
    write_env
    local name=backflow_stress_n$N_STRESS
    echo "+ timeout -k 10 $STRESS_TIMEOUT /usr/bin/time -v ... inlet_slab --track forward --n $N_STRESS --ell 0.15 --seed 2" \
         "--eps 4 --sigma2 1 --np-side $([ "$DRY" = "1" ] && echo "$NP_SIDE" || echo 16) --planes 0.5,1 --times 0.5,1 (default tol/h_max/cap)"
    local t0 np=16
    [ "$DRY" = "1" ] && np=$NP_SIDE
    t0=$(date +%s)
    timeout -k 10 "$STRESS_TIMEOUT" "$BUILD/inlet_slab" --track forward --n "$N_STRESS" --ell 0.15 --seed 2 --eps 4 \
        --sigma2 1 --np-side "$np" --planes 0.5,1 --times 0.5,1 --summary "$OUT/$name.json" \
        > "$OUT/$name.log" 2>&1
    LAST_RC=$?
    record_rc "$name" "$LAST_RC"
    res "  STRESS_RESULT rc=$LAST_RC wall=$(( $(date +%s) - t0 ))s timeout=${STRESS_TIMEOUT}s" \
         "($([ "$LAST_RC" -eq 124 ] && echo 'DID NOT TERMINATE within the timeout' || echo terminated))"
    grep -E '^(TRACK|BACKFLOW|TSTATUS|STATUS)' "$OUT/$name.log" | cut -c1-240 | sed 's/^/    /'
}

case "$PART" in
controls) part_controls ;;
bytecmp) part_bytecmp ;;
mem) part_mem ;;
cost) part_cost ;;
stress) part_stress ;;
all)
    part_controls
    part_bytecmp
    part_mem
    part_cost
    part_stress
    ;;
*)
    echo "run_sf36.sh: unknown part '$PART'" >&2
    exit 2
    ;;
esac
digest_all
echo "run_sf36.sh: part=$PART end=$(date -u +%FT%TZ) blocking_failures=$BAD"
[ "$BAD" -eq 0 ]
