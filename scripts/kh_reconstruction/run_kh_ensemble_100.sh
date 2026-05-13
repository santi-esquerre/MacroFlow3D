#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

phase="${1:-full}"
case "$phase" in
  smoke) count=2 ;;
  mini) count=10 ;;
  full) count=100 ;;
  *)
    echo "usage: $0 [smoke|mini|full]" >&2
    exit 2
    ;;
esac

job="kh-ensemble-${phase}-$(date +%Y%m%d-%H%M%S)"

scripts/remote sync
scripts/remote run "$job" -- "bash scripts/kh_reconstruction/remote_kh_ensemble_driver.sh ${count} ${phase}"

cat <<EOF
Launched remote KH ensemble job:
  job: ${job}
  seeds: 0..$((count - 1))
  backends: FACE_TRILINEAR and KH_POTENTIAL_RECONSTRUCTION

Monitor with:
  scripts/remote status ${job}
  scripts/remote tail ${job}
  scripts/remote wait ${job}

Collect/summarize on the remote after completion:
  scripts/remote exec -- "bash scripts/kh_reconstruction/collect_kh_ensemble_results.sh artifacts/kh_reconstruction"
EOF
