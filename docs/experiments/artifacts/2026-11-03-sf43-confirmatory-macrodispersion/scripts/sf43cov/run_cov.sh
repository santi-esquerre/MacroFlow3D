#!/usr/bin/env bash
# Restricted coverage run of the SF-43 Gate 5 procedures (decision H15), for a DETACHED host job (bash 4.2, CPU only, no GPU):
#
#   scripts/remote --increment SF-43 run sf43-coverage -- \
#       "bash docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts/sf43cov/run_cov.sh ~/sf43-coverage-out --workers 32"
#   scripts/remote wait sf43-coverage
#
# usage: run_cov.sh <out_dir> [--workers N] [sf43cov plan options: --target 175 | --R 49000, --ns 100,150,..., --B 33599, --family-m 8, ...]
#
# * Idempotent and resumable: rerun the SAME command after an interruption; finished chunks are skipped (the counts do not depend on the
#   number of workers, so a rerun may use another --workers).  A plan different from the one in <out_dir> is refused (plan hash).
# * Runs at `nice -n 19`; default 32 workers (the host is shared with colleagues).
# * Outputs in <out_dir>: plan.json, chunks/, summary.json (the file sf43ana.gate5 reads: coverage_summary), summary.md, env.txt, run.log;
#   smoke/ holds a tiny end-to-end check that runs first (warnings are errors) and aborts the job before the long run if it fails.
set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
OUT="$1"
shift
WORKERS=32
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --out) echo "run_cov.sh: --out is the first positional argument" >&2; exit 2 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(dirname "$HERE")"  # the directory that contains sf43cov/ and sf43ana/
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1  # one core per worker
export PYTHONHASHSEED=0

mkdir -p "$OUT"
cd "$SCRIPTS"
{
    date -u +"started %Y-%m-%dT%H:%M:%SZ"
    "$PY" -c 'import platform, numpy, scipy; print("python", platform.python_version(), "numpy", numpy.__version__, "scipy", scipy.__version__)'
    echo "workers $WORKERS host $(hostname) cpus $(nproc --all 2>/dev/null || echo ?) options: ${EXTRA[*]+${EXTRA[*]}}"
    echo "lscpu: $(lscpu 2>/dev/null | awk -F': *' '/^Model name|^Socket\(s\)|^Core\(s\) per socket|^Thread\(s\) per core/ {printf "%s=%s; ", $1, $2}')"
} | tee -a "$OUT/env.txt"

# 1. environment check: a tiny end-to-end run of every procedure (both workers and the bootstrap), warnings as errors
SMOKE="$OUT/smoke"
"$PY" -W error -m sf43cov run --out "$SMOKE" --ns 60 --dists stress --report-ns "" --B 199 --target 2 --batch 10 --family-m 4 --workers 2
"$PY" -W error -m sf43cov summarize --out "$SMOKE" > /dev/null
echo "smoke check passed"

# 2. the study (plan.json is created by the first call; later calls resume)
nice -n 19 "$PY" -u -m sf43cov run --out "$OUT" --workers "$WORKERS" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "$OUT/run.log"

# 3. summary (summary.json, summary.md)
"$PY" -m sf43cov summarize --out "$OUT" | tee -a "$OUT/run.log"
date -u +"finished %Y-%m-%dT%H:%M:%SZ" | tee -a "$OUT/env.txt"
