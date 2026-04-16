#!/usr/bin/env bash
# Helper: create a git worktree for MacroFlow3D development.
#
# Usage:
#   scripts/create-worktree.sh <branch-name>
#
# Example:
#   scripts/create-worktree.sh chore/add-sccache
#   scripts/create-worktree.sh science/pspta-refinement
#
# Creates:
#   ~/src/MacroFlow3D/.codex/worktrees/<short-name>/
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <branch-name>" >&2
    echo "Example: $0 chore/add-pre-commit" >&2
    exit 1
fi

BRANCH="$1"
# Extract the short name (part after the last /)
SHORT_NAME="${BRANCH##*/}"

WORKTREE_BASE="${MACROFLOW3D_WORKTREE_BASE:-$HOME/src/MacroFlow3D/.codex/worktrees}"
WORKTREE_DIR="$WORKTREE_BASE/$SHORT_NAME"

# Find repo root
if ! REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    echo "ERROR: not inside a git repository." >&2
    exit 1
fi

if [[ -d "$WORKTREE_DIR" ]]; then
    echo "Worktree already exists: $WORKTREE_DIR" >&2
    echo "To enter: cd $WORKTREE_DIR" >&2
    exit 1
fi

mkdir -p "$WORKTREE_BASE"

echo "Creating worktree:"
echo "  branch:    $BRANCH"
echo "  directory: $WORKTREE_DIR"
echo ""

cd "$REPO_ROOT"
git worktree add -b "$BRANCH" "$WORKTREE_DIR"

echo ""
echo "Done. To start working:"
echo "  cd $WORKTREE_DIR"
