#!/usr/bin/env bash
# Helper: push current branch and create a PR with gh.
#
# Usage:
#   scripts/create-pr.sh
#   scripts/create-pr.sh --draft
#
# Requires: gh CLI authenticated
set -euo pipefail

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [[ "$BRANCH" == "master" || "$BRANCH" == "main" ]]; then
    echo "ERROR: do not create PRs from $BRANCH. Use a worktree branch." >&2
    exit 1
fi

echo "Branch: $BRANCH"
echo ""

# Push if needed
if ! git rev-parse --verify "origin/$BRANCH" &>/dev/null; then
    echo "Pushing branch to origin..."
    git push -u origin "$BRANCH"
else
    echo "Branch already on origin. Pushing latest..."
    git push
fi

echo ""

# Create PR
EXTRA_ARGS=("$@")

echo "Creating PR..."
gh pr create --fill "${EXTRA_ARGS[@]}"
