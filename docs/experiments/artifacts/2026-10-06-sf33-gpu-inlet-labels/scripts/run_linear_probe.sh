#!/usr/bin/env bash
# SF-33 N7c: discriminating linear-solve experiment (Galerkin coarse correction on the weak
# subspace + P-A vs P-A alone) on frozen hard-stage Jacobians.
#
# Usage (from the repository root):
#   bash docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/run_linear_probe.sh <build_dir> [<out_root>]
# Needs the prototype exports under <out_root>/exports (export_proto.py; see run_proto.sh).
# Each probe: `inlet_slab --linear-probe <case> --eps-from A --eps-stage E --newton-steps k`
# (driver defaults: Psi-tc on, forcing ew, GMRES(100), cap 6000) with --probe-tol 1e-8 and the four
# preconditioners pa, mult1, mult2, add1 at mu = mu_SER(k) and mu = 0.
# Logs: <out_root>/logs/n7c_local/probe/<name>.log; JSON: <out_root>/raw/n7c_local/probe/<name>.json.
# Env: PROBES overrides the list (space-separated name:case:from:stage:k[:ladder[:extra]] with
# extra driver options joined by '+', e.g. "+--psitc+off").
set -euo pipefail

BUILD_DIR=${1:?usage: run_linear_probe.sh <build_dir> [<out_root>]}
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=$(dirname "$SCRIPTS")
OUT=${2:-$ART}
OUT=$(cd "$OUT" && pwd)
BIN=$(cd "$BUILD_DIR" && pwd)/inlet_slab
EXPORTS=$OUT/exports
LOGS=$OUT/logs/n7c_local/probe
RAW=$OUT/raw/n7c_local/probe
mkdir -p "$LOGS" "$RAW"

# name:case:from:stage:k:ladder:extra   (ladder = the continuation ladder that reproduces the
# solver's warm-start path; '-' = prototype ladder 0.25,0.5,1)
DEFAULT_PROBES="
gauss_ch_0.5_24_k0:gauss_ch_0.5_24:0.375:0.5:0:-:
gauss_ch_0.5_24_k1:gauss_ch_0.5_24:0.375:0.5:1:-:
gauss_ch_0.5_24_k8:gauss_ch_0.5_24:0.375:0.5:8:-:
gauss_ch_0.5_24_k1_psitcoff:gauss_ch_0.5_24:0.375:0.5:1:-:+--psitc+off
gauss_0.5_24_k0:gauss_0.5_24:0.4375:0.5:0:0.25,0.375:
gauss_0.5_24_k1:gauss_0.5_24:0.4375:0.5:1:0.25,0.375:
gauss_0.5_24_k8:gauss_0.5_24:0.4375:0.5:8:0.25,0.375:
generic3d_1_16_k0:generic3d_1_16:0.625:0.75:0:-:
generic3d_1_16_k1:generic3d_1_16:0.625:0.75:1:-:
generic3d_1_16_k7:generic3d_1_16:0.625:0.75:7:-:
gauss_0.25_16_k1:gauss_0.25_16:0:0.25:1:-:
"
PROBES=${PROBES:-$DEFAULT_PROBES}

echo "run_linear_probe.sh: bin=$BIN out=$OUT host=$(hostname) start=$(date -u +%FT%TZ)"
for spec in $PROBES; do
    IFS=: read -r name case from stage k ladder extra <<< "$spec"
    args=(--linear-probe "$EXPORTS/$case" --eps-stage "$stage" --newton-steps "$k"
          --probe-tol 1e-8 --probe-precs pa,mult1,mult2,add1 --summary "$RAW/$name.json")
    if [ "$from" != "0" ]; then
        args+=(--eps-from "$from")
    fi
    if [ -n "$ladder" ] && [ "$ladder" != "-" ]; then
        args+=(--probe-ladder "$ladder")
    fi
    if [ -n "${extra:-}" ]; then
        IFS='+' read -r -a ex <<< "${extra#+}"
        args+=("${ex[@]}")
    fi
    t0=$(date +%s)
    set +e
    echo "+ $BIN ${args[*]} > $LOGS/$name.log 2>&1"
    "$BIN" "${args[@]}" > "$LOGS/$name.log" 2>&1
    rc=$?
    set -e
    echo "  PROBE_RESULT $name rc=$rc wall=$(( $(date +%s) - t0 ))s"
    grep '^PROBE ' "$LOGS/$name.log" | sed 's/^/    /' | cut -c1-220 || true
done
echo "run_linear_probe.sh: end=$(date -u +%FT%TZ)"
