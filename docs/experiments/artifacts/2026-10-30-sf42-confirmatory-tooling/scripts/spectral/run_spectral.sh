#!/usr/bin/env bash
# SF-42 N3: pseudo-spectral cross-check of the FV Darcy velocity (acceptance E).  Restart safe.
#
#   bash run_spectral.sh <build_dir> <out_dir> [fv|spectral|scan|all]
#
# Phase fv       (GPU): sf42_periodic_flow on every (sigma2, seed, N) -> <out>/fv/s<sigma2>_seed<seed>/N<N>/
#                       {Y,U,V,W}.npy + report.json (report.json is written last: its presence = level done).
# Phase spectral (CPU only): python3 -m spectral.compare run  (reference pair M, 2M per sigma2, comparison,
#                       <out>/cases/*.json) then summarize (<out>/summary.{json,md}).
# Phase scan     (CPU only, optional, before the study): reference self-convergence ladder SF42_SCAN_MS (default
#                       "64 128 256") for every sigma2 at the first seed, no FV data needed -> <out>/scan_*.json; use it to
#                       choose SF42_REF_M if the study's verdict reports an unconverged reference.
# Phase all     = fv then spectral.
#
# Designed for detached `scripts/remote run` jobs (the spectral phase needs no GPU; run it as its own job).
# Environment (defaults = the full study of SF-42 item 7):
#   SF42_SIGMA2S="0.25 1 2.25 4"   SF42_SEEDS="420001 420002"   SF42_NS="16 32 40 64 128 256"
#   SF42_REF_M="0.25:64,1:64,2.25:96,4:128"   (reference pair M, 2M per sigma2)
#   SF42_FV_RTOL=1e-12  SF42_FV_MAXIT=4000  SF42_REF_TOL=1e-12  SF42_WORKERS=<nproc>  SF42_GEN_N=64
# The FV driver exit code 3 (a corrector did not converge) is recorded in <level>/rc and in report.json and makes the
# summary FAIL for that ladder; any other non-zero exit code aborts.
set -euo pipefail
BUILD=${1:?usage: run_spectral.sh <build_dir> <out_dir> [fv|spectral|scan|all]}
OUT=${2:?usage: run_spectral.sh <build_dir> <out_dir> [fv|spectral|scan|all]}
PHASE=${3:-all}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPTS=$(cd "$HERE/.." && pwd)
SIGMA2S=${SF42_SIGMA2S:-"0.25 1 2.25 4"}
SEEDS=${SF42_SEEDS:-"420001 420002"}
NS=${SF42_NS:-"16 32 40 64 128 256"}
REF_M=${SF42_REF_M:-"0.25:64,1:64,2.25:96,4:128"}
FV_RTOL=${SF42_FV_RTOL:-1e-12}
FV_MAXIT=${SF42_FV_MAXIT:-4000}
REF_TOL=${SF42_REF_TOL:-1e-12}
GEN_N=${SF42_GEN_N:-64}
WORKERS=${SF42_WORKERS:-$(nproc)}
DRIVER="$BUILD/sf42_periodic_flow"
# the CPU phases use scipy.fft workers and BLAS (face averages): bound both by SF42_WORKERS unless the caller set them
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$WORKERS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$WORKERS} MKL_NUM_THREADS=${MKL_NUM_THREADS:-$WORKERS}

eps_of() { python3 -c "import math,sys; print(repr(math.sqrt(float(sys.argv[1]))))" "$1"; }
tag_of() { python3 -c "import sys; print(('s%g' % float(sys.argv[1])).replace('.', 'p'))" "$1"; }

phase_fv() {
  [ -x "$DRIVER" ] || { echo "missing driver $DRIVER" >&2; exit 2; }
  for s2 in $SIGMA2S; do
    eps=$(eps_of "$s2"); tag=$(tag_of "$s2")
    for seed in $SEEDS; do
      for n in $NS; do
        d="$OUT/fv/${tag}_seed${seed}/N${n}"
        if [ -f "$d/report.json" ]; then echo "fv ${tag} seed ${seed} N=${n}: done (skipped)"; continue; fi
        mkdir -p "$d"
        echo "fv ${tag} seed ${seed} N=${n}: $DRIVER --n $n --eps $eps --seed $seed --rtol $FV_RTOL --max-iter $FV_MAXIT"
        set +e
        "$DRIVER" --n "$n" --eps "$eps" --seed "$seed" --rtol "$FV_RTOL" --max-iter "$FV_MAXIT" --out "$d"
        rc=$?
        set -e
        echo "$rc" > "$d/rc"
        if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then echo "driver failed (rc=$rc) at $d" >&2; exit "$rc"; fi
      done
    done
  done
}

phase_spectral() {
  cd "$SCRIPTS"
  python3 -m spectral.compare run --fv-root "$OUT/fv" --out "$OUT" --sigma2 $SIGMA2S --seeds $SEEDS --ns $NS \
    --ref-m "$REF_M" --gen-n "$GEN_N" --tol "$REF_TOL" --workers "$WORKERS"
}

phase_scan() {
  cd "$SCRIPTS"
  for s2 in $SIGMA2S; do
    python3 -m spectral.compare scan --out "$OUT" --sigma2 "$s2" --seed "${SEEDS%% *}" --ms ${SF42_SCAN_MS:-64 128 256} \
      --ns $NS --gen-n "$GEN_N" --tol "$REF_TOL" --workers "$WORKERS"
  done
}

case "$PHASE" in
  fv) phase_fv ;;
  spectral) phase_spectral ;;
  scan) phase_scan ;;
  all) phase_fv; phase_spectral ;;
  *) echo "phase must be fv|spectral|scan|all" >&2; exit 2 ;;
esac
