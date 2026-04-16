#!/usr/bin/env bash
# Check that required documentation files exist.
# Used as a pre-commit hook to prevent accidental deletion of key docs.
set -euo pipefail

REQUIRED_DOCS=(
    "AGENTS.md"
    "ARCHITECTURE.md"
    "docs/validation/acceptance-gates.md"
    "docs/runbooks/local-wsl.md"
    "docs/runbooks/remote-v100.md"
    "docs/runbooks/petsc-slepc.md"
    "docs/AGENTS.md"
)

exit_code=0

for doc in "${REQUIRED_DOCS[@]}"; do
    if [[ ! -f "$doc" ]]; then
        echo "ERROR: Required doc missing: $doc" >&2
        exit_code=1
    fi
done

if [[ $exit_code -eq 0 ]]; then
    echo "Required docs: OK"
fi

exit $exit_code
