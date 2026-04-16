#!/usr/bin/env bash
# Guard: warn when scientific-core files are changed without
# validation/doc references in the commit.
#
# This hook does NOT block commits. It prints a reminder.
# Blocking is enforced at PR review, not at commit time.
set -euo pipefail

# Files passed by pre-commit (staged scientific-core files)
CHANGED_FILES=("$@")

if [[ ${#CHANGED_FILES[@]} -eq 0 ]]; then
    exit 0
fi

# Classify which areas are touched
NUMERICS=false
PHYSICS=false
MULTIGRID=false
PSPTA=false

for f in "${CHANGED_FILES[@]}"; do
    case "$f" in
        src/numerics/*)  NUMERICS=true ;;
        src/physics/particles/pspta/*) PSPTA=true; PHYSICS=true ;;
        src/physics/*)   PHYSICS=true ;;
        src/multigrid/*) MULTIGRID=true ;;
        apps/*.yaml)     PHYSICS=true ;;
    esac
done

echo "============================================"
echo "  SCIENTIFIC CORE CHANGE DETECTED"
echo "============================================"
echo ""
echo "Changed areas:"
$NUMERICS  && echo "  - src/numerics/ (operators, solvers, BLAS)"
$PHYSICS   && echo "  - src/physics/ (flow, stochastic, transport)"
$MULTIGRID && echo "  - src/multigrid/ (transfers, smoothers, V-cycle)"
$PSPTA     && echo "  - src/physics/particles/pspta/ (HIGH RISK)"
echo ""
echo "Reminders:"
echo "  1. Identify the required acceptance gate (docs/validation/acceptance-gates.md)"
echo "  2. Run the corresponding eval tier (docs/validation/eval-tiers.md)"
echo "  3. Include evidence in the PR description"
echo "  4. Scientific-core changes require HUMAN REVIEW before merge"
echo ""

if $PSPTA; then
    echo "  *** PSPTA CHANGE: read src/physics/particles/pspta/AGENTS.md ***"
    echo "  Minimum validation:"
    echo "    ctest --test-dir <build-dir> --output-on-failure -R operator_tests"
    echo "    ./<build-dir>/macroflow3d_pipeline apps/config_pspta_small.yaml"
    echo ""
fi

echo "============================================"

# Exit 0: this is a warning, not a blocker.
# Blocking happens at PR review.
exit 0
