#!/usr/bin/env bash
# Coverage simulation of the SF-42 confirmatory tooling (decision G5.3), for a DETACHED host job:
#
#   REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-42 run sf42-coverage -- \
#       "bash docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/mcstat/run_coverage.sh \
#            ~/sf42-coverage-out --workers 72"
#   scripts/remote wait sf42-coverage
#
# usage: run_coverage.sh <out_dir> [--workers N] [--profile full|full-grid|n6b|studentized|smoke] [--ladder-ns 30,100,200]
#
# * CPU only (stdlib + numpy + scipy; host Python 3.11 / numpy 1.26 / scipy 1.11).  It does not touch a GPU.
# * Restart-safe: rerun the same command after an interruption; finished chunks are skipped, and the result does not depend on the
#   number of workers, so a rerun may use another --workers.  A plan different from the one in <out_dir> is refused.
# * Default workers: the number of CPUs capped at 72 (the host is shared; pass --workers to override).
# * Outputs in <out_dir>: plan.json, chunks/ (one JSON per finished chunk), summary.json, summary.md (qualification matrix,
#   failures, UCL/tail tables), env.txt, run.log.  Cost of the default `full` profile: see the module docstring of
#   mcstat/coverage.py (MEASURED on the host: 2995 core-hours, 41.7 h wall on 72 workers; the earlier projection of 5.4-7.3 h from 261
#   workstation core-hours was 6-8 times too low; `full-grid` and `n6b` were never run on the host).
set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
OUT="$1"
shift
WORKERS="$(nproc 2>/dev/null || echo 1)"
[ "$WORKERS" -gt 72 ] && WORKERS=72
PROFILE=full
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --ladder-ns) EXTRA+=(--ladder-ns "$2"); shift 2 ;;
        *) echo "run_coverage.sh: unknown option $1" >&2; exit 2 ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(dirname "$HERE")"  # the directory that contains the mcstat package
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1  # one core per worker
export PYTHONHASHSEED=0

mkdir -p "$OUT"
cd "$SCRIPTS"
{
    date -u +"started %Y-%m-%dT%H:%M:%SZ"
    "$PY" -c 'import platform, numpy, scipy; print("python", platform.python_version(), "numpy", numpy.__version__, "scipy", scipy.__version__)'
    # plain `nproc` honours OMP_NUM_THREADS (exported above: it printed 1); `--all` is the installed logical CPU count
    echo "profile $PROFILE workers $WORKERS host $(hostname) cpus $(nproc --all 2>/dev/null || echo ?)"
    echo "lscpu: $(lscpu 2>/dev/null | awk -F': *' '/^Model name|^Socket\(s\)|^Core\(s\) per socket|^Thread\(s\) per core/ {printf "%s=%s; ", $1, $2}')"
} | tee "$OUT/env.txt"

# 1. environment check: a tiny end-to-end run of every procedure with warnings as errors (aborts before the long run)
SMOKE="$OUT/smoke"
"$PY" -W error -m mcstat.coverage run --profile smoke --target 0.3 --out "$SMOKE" --workers 2
"$PY" -W error -m mcstat.coverage summarize --out "$SMOKE" > /dev/null
echo "smoke check passed"

# 2. the study (creates plan.json on the first call; later calls resume)
nice -n 10 "$PY" -u -m mcstat.coverage run --profile "$PROFILE" ${EXTRA[@]+"${EXTRA[@]}"} --out "$OUT" --workers "$WORKERS" 2>&1 | tee -a "$OUT/run.log"

# 3. summary
"$PY" -m mcstat.coverage summarize --out "$OUT" | tee -a "$OUT/run.log"
date -u +"finished %Y-%m-%dT%H:%M:%SZ" | tee -a "$OUT/env.txt"
