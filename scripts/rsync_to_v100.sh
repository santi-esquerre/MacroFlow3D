#!/usr/bin/env bash
set -euo pipefail

# Sync the local repo to the remote V100 server.
#
# Defaults:
#   local repo:   current git repo root
#   remote host:  v100
#   remote path:  ~/MacroFlow3D
#
# Override:
#   REMOTE_HOST=myhost REMOTE_DIR=~/other/path scripts/rsync_to_v100.sh

REMOTE_HOST="${REMOTE_HOST:-v100}"
REMOTE_DIR="${REMOTE_DIR:-~/MacroFlow3D}"

if ! git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  echo "ERROR: run this inside the repository or a worktree." >&2
  exit 1
fi

echo "Syncing:"
echo "  local : $git_root"
echo "  remote: ${REMOTE_HOST}:${REMOTE_DIR}"

rsync -az --delete \
  --exclude '.git' \
  --exclude '.codex' \
  --exclude '.idea' \
  --exclude '.vscode' \
  --exclude '.cache' \
  --exclude '.venv' \
  --exclude 'build' \
  --exclude 'build-*' \
  --exclude 'output' \
  --exclude 'output_*' \
  --exclude '__pycache__' \
  "${git_root}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Done."
