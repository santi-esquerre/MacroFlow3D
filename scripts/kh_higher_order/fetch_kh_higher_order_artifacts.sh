#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/../.." rev-parse --show-toplevel 2>/dev/null || true)"

if [[ -z "$repo_root" ]]; then
  echo "ERROR: fetch_kh_higher_order_artifacts.sh must run inside the repository or a worktree." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$repo_root/scripts/remote.env"

remote_path="${REMOTE_REPO_DIR}/artifacts/kh_higher_order/"
local_path="${repo_root}/artifacts/kh_higher_order/"

mkdir -p "$local_path"
rsync -az "${REMOTE_HOST}:${remote_path}" "$local_path"
echo "Fetched ${REMOTE_HOST}:${remote_path} -> ${local_path}"
