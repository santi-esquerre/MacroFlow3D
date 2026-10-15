#!/usr/bin/env bash
# SF-32 spurious-spreading experiment: ladder launcher (one label field per invocation).
#
#   bash apps/spurious_spreading/run_ladders.sh <spurious_spreading-binary> <labels-prefix> <out-root>
#        [--seeds 8192] [--seed 20261006]
#
# <labels-prefix> is the prefix of a saved label pair: <prefix>.json (route metadata) plus
# <prefix>_u1.bin, <prefix>_u2.bin. The field name is derived from <prefix>.json
# (route, field|pair, eps|amplitude, n), e.g. stack_lester2021_e0.25_n256, analytic_G_a0.05_n128.
#
# Runs sequentially, with the pre-registered SF-32 ladders (orchestration record 3.6; do not edit):
#   pollock            --delta-ratio 1 2 4 8
#   rk                 --tol 1e-04 .. 1e-10 (the paper's five 1e-4..1e-8 plus 1e-9, 1e-10)
#   pseudo_symplectic  --tol-psi 1e-08 1e-10 1e-12
# Each run is
#   <binary> return-map --labels <prefix> --tracker <t> <level option> <value> --seeds N --seed S --out <run_dir>
# with <run_dir> = <out-root>/<field_name>/<tracker>_<level_tag>; its stdout/stderr go to
# <run_dir>/run.log. --dt-max (absolute, default 0.25; D-2) and --ds-ratio (default 0.5) are NOT
# passed: instrument defaults.
#
# A failed run never aborts the launcher (the experiment is a measurement: a failed level is
# reported, not hidden). It is appended to <out-root>/<field_name>/failures.txt and the launcher
# exits 1 at the end if any run failed. Usage errors exit 2.

set -euo pipefail

usage() {
    sed -n '4,5p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

[ $# -ge 3 ] || usage
BIN="$1"
PREFIX="$2"
OUT_ROOT="$3"
shift 3

N_SEEDS=8192
SEED=20261006
while [ $# -gt 0 ]; do
    case "$1" in
        --seeds)
            [ $# -ge 2 ] || usage
            N_SEEDS="$2"
            shift
            ;;
        --seed)
            [ $# -ge 2 ] || usage
            SEED="$2"
            shift
            ;;
        *) usage ;;
    esac
    shift
done

# ---- pre-registered ladders (SF-32 orchestration record 3.6; do not edit) ---------------------
POLLOCK_RATIOS=(1 2 4 8)
RK_TOLS=(1e-04 1e-05 1e-06 1e-07 1e-08 1e-09 1e-10)
PS_TOLS=(1e-08 1e-10 1e-12)
# -----------------------------------------------------------------------------------------------

[ -x "$BIN" ] || { echo "run_ladders: binary not executable: $BIN" >&2; exit 2; }
[ -f "${PREFIX}.json" ] || { echo "run_ladders: missing ${PREFIX}.json" >&2; exit 2; }

FIELD_NAME="$(python3 - "${PREFIX}.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
route = m["route"]
field = m.get("field", m.get("pair"))
if field is None:
    sys.exit("labels json has neither 'field' nor 'pair'")
if "eps" in m:
    amp = "e%g" % float(m["eps"])
elif "amplitude" in m:
    amp = "a%g" % float(m["amplitude"])
else:
    sys.exit("labels json has neither 'eps' nor 'amplitude'")
print("%s_%s_%s_n%d" % (route, field, amp, int(m["n"])))
PY
)"

FIELD_DIR="${OUT_ROOT}/${FIELD_NAME}"
mkdir -p "$FIELD_DIR"
cp "${PREFIX}.json" "${FIELD_DIR}/labels.json"
FAILURES="${FIELD_DIR}/failures.txt"
echo "# launcher invocation $(date -u +%Y-%m-%dT%H:%M:%SZ) seeds=${N_SEEDS} seed=${SEED} labels=${PREFIX}" >>"$FAILURES"

echo "run_ladders: field ${FIELD_NAME} -> ${FIELD_DIR} (seeds ${N_SEEDS}, seed ${SEED})"

N_FAILED=0
run_one() {
    # run_one <tracker> <level_tag> <level option> <level value>
    local tracker="$1" tag="$2" opt="$3" val="$4"
    local run_dir="${FIELD_DIR}/${tracker}_${tag}"
    mkdir -p "$run_dir"
    local t0 t1 rc wall
    t0="$(date +%s.%N)"
    set +e
    "$BIN" return-map --labels "$PREFIX" --tracker "$tracker" "$opt" "$val" \
        --seeds "$N_SEEDS" --seed "$SEED" --out "$run_dir" >"${run_dir}/run.log" 2>&1
    rc=$?
    set -e
    t1="$(date +%s.%N)"
    wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f", b - a }')"
    printf '%-18s %-12s exit=%d wall=%ss\n' "$tracker" "$tag" "$rc" "$wall"
    if [ "$rc" -ne 0 ]; then
        printf '%s_%s\texit=%d\twall=%ss\t%s\n' "$tracker" "$tag" "$rc" "$wall" "${run_dir}/run.log" >>"$FAILURES"
        N_FAILED=$((N_FAILED + 1))
    fi
}

for m in "${POLLOCK_RATIOS[@]}"; do
    run_one pollock "m${m}" --delta-ratio "$m"
done
for t in "${RK_TOLS[@]}"; do
    run_one rk "tol${t}" --tol "$t"
done
for t in "${PS_TOLS[@]}"; do
    run_one pseudo_symplectic "tolpsi${t}" --tol-psi "$t"
done

if [ "$N_FAILED" -ne 0 ]; then
    echo "run_ladders: ${N_FAILED} run(s) failed; see ${FAILURES}" >&2
    exit 1
fi
echo "run_ladders: all runs exited 0"
