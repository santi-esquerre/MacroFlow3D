#!/usr/bin/env bash
# SF-42 N2: full MMS order study of the medium (b) Dirichlet head solve.
#   run_mms.sh <build_dir> <out_dir>
# Runs mms_dirichlet for ny in MMS_NY_LIST (default "16 32 64 128 256") x eps {0.5 1 1.5 2} at the production
# pcg rtol 1e-10, then the algebraic-error control (rtol 1e-12) at the finest ny for every eps.
# Restart-safe: a run whose JSON already exists is skipped. Written for a detached `scripts/remote run` job
# (the V100 host runs it; ny = 256 is 6.7e7 cells, about 5 GB of device memory). Exit status 1 if any run failed
# or did not converge (the JSON of a non-converged run is still written and flagged by analyze.py).
set -u
if [ $# -ne 2 ]; then echo "usage: $0 <build_dir> <out_dir>" >&2; exit 2; fi
BUILD=$1
OUT=$2
EXE="$BUILD/mms_dirichlet"
[ -x "$EXE" ] || { echo "missing executable $EXE" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../../../.." && pwd) # scripts/mms -> scripts -> R -> artifacts -> experiments -> docs -> root
COMMIT=${MMS_COMMIT:-$( (cd "$ROOT" && git rev-parse HEAD) 2>/dev/null || echo unknown)}
NY_LIST=${MMS_NY_LIST:-"16 32 64 128 256"}
EPS_LIST=${MMS_EPS_LIST:-"0.5 1 1.5 2"}
mkdir -p "$OUT"

echo "== self-check =="
"$EXE" --self-check || { echo "SELF-CHECK FAILED" >&2; exit 1; }

FAIL=0
run_one() { # ny eps rtol
  local f="$OUT/mms_ny$(printf '%04d' "$1")_eps$2_rtol$3.json"
  if [ -s "$f" ]; then echo "skip $f"; return; fi
  echo "== run ny=$1 eps=$2 rtol=$3 =="
  "$EXE" --ny "$1" --eps "$2" --pcg-rtol "$3" --out-dir "$OUT" --commit "$COMMIT" || { echo "FAILED ny=$1 eps=$2 rtol=$3" >&2; FAIL=1; }
}
FINEST=$(for n in $NY_LIST; do echo "$n"; done | sort -n | tail -1)
for eps in $EPS_LIST; do
  for ny in $NY_LIST; do run_one "$ny" "$eps" 1e-10; done
done
for eps in $EPS_LIST; do run_one "$FINEST" "$eps" 1e-12; done
python3 "$HERE/analyze.py" "$OUT" || FAIL=1
exit $FAIL
