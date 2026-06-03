#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/../.." rev-parse --show-toplevel 2>/dev/null || true)"

if [[ -z "$repo_root" ]]; then
  candidate_root="$(cd "$script_dir/../.." && pwd)"
  if [[ -f "$script_dir/analyze_kh_higher_order.py" ]]; then
    repo_root="$candidate_root"
  fi
fi

if [[ -z "$repo_root" ]]; then
  echo "ERROR: collect_kh_higher_order_results.sh must run inside the repository or a worktree." >&2
  exit 1
fi

root="${1:-$repo_root/artifacts/kh_higher_order}"
python3 "$script_dir/analyze_kh_higher_order.py" "$root"
