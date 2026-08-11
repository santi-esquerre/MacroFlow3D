# SF-22 — Grid continuation

- State: `pending`
- Goal: `Implementar continuación de malla preservando gauge y realización aleatoria.`
- Depends on: `SF-21`
- Unlocks: `SF-23`
- Branch: `science/lester-sf22-grid-continuation`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf21-grid-continuation`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + Gate 4`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Use accepted coarse streamfunctions as fine-grid initial states while keeping
the random realization, affine part, periodicity, and zero-mean gauge fixed.

## Preconditions

- SF-21 converges fixed-seed Gaussian cases at individual grids.

## In scope

- Cross-run prolongation/restriction, grid-ladder orchestration, comparison
  metrics, and reuse analysis of existing MG transfers.

## Out of scope

- Changing MG's internal transfer scheme or implementing high-order transfer.

## Files and symbols

- Add grid-stage orchestration to `ContinuationController` or a separate
  benchmark controller.
- Reuse existing `prolong_3d_add` on a zero fine buffer and existing restriction
  for comparisons; project after transfer.

## Implementation specification

1. Use the same continuous periodic realization of `Y` at every grid.
2. Prolong only periodic fluctuations; copy affine gradients unchanged.
3. Project each fine initial field to zero mean.
4. Compare prolonged/restricted fields and solver metrics at the same physical
   locations; do not call changing `ell/h` cases a mesh-convergence study.

## Expected numerical effect

The `32^3 -> 64^3 -> 128^3` ladder needs fewer nonlinear iterations than zero
initialization without changing the converged fine solution.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_grid_continuation
scripts/remote exec -- "<fixed-seed-128-grid-ladder-command>"
```

## Acceptance thresholds

- Transferred means meet the SF-03 threshold.
- Constant and trigonometric transfer controls pass documented accuracy tests.
- Final fine solutions from prolonged and zero starts agree within nonlinear
  tolerance; prolonged start reduces iterations on the fixed suite.

## Regression surface

- MG transfer semantics, additive prolongation, field normalization, and random
  realization mapping.

## Failure and rollback policy

- If piecewise-constant prolongation hurts convergence, record evidence and
  propose trilinear periodic transfer as a new increment; do not silently alter
  MG transfers here.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Grid-stage controller uses the same realization and accepted coarse state.
- [ ] Transfer and mean-zero projection tests pass.
- [ ] Prolonged/zero-start equivalence and iteration comparison are recorded.
- [ ] `32^3 -> 64^3 -> 128^3` fixed ladder completes.
- [ ] Gate 3A/4 review and experiment note pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-22 complete and selects SF-23.
<!-- completion-checklist:end -->

## Advancement rule

SF-23 may optimize kernels using the accepted Picard/Anderson/continuation
outputs as correctness baselines.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-11T14:10Z | re-sequenced into slot SF-22 (was SF-21) | Owner decision (option (a), 2026-08-11): Anderson moved before heterogeneity; grid continuation follows heterogeneity unchanged. See `docs/decisions/2026-08-11-anderson-before-heterogeneity.md`. | Content otherwise untouched. | Activate only when named by `NEXT`. |
