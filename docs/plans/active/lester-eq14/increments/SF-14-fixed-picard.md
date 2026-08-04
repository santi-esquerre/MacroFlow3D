# SF-14 — Fixed-relaxation Picard

- State: `pending`
- Goal: `Implementar Picard secuencial con relajación fija.`
- Depends on: `SF-13`
- Unlocks: `SF-15`
- Branch: `science/lester-sf14-fixed-picard`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf14-fixed-picard`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Establish the simplest auditable nonlinear fixed-point map before adding
globalization or acceleration.

## Preconditions

- SF-13 validates the complete zero-source solver path.

## In scope

- Sequential source evaluation, two projected PCG/MG solves, paired relaxation,
  projection, convergence history, and fixed iteration/tolerance limits.

## Out of scope

- Step rejection, adaptive relaxation, continuation, and Anderson acceleration.

## Files and symbols

- Extend `StreamfunctionSolver.cu` with the Picard loop and report history.
- Add small smooth manufactured/controlled nonlinear cases.

## Implementation specification

1. Evaluate both sources from one immutable current pair.
2. Solve `A*u1hat=affine1-eta*q*S2` and the paired equation consecutively with
   one hierarchy and solver workspace.
3. Update both fields with fixed `omega=0.25`, then project both.
4. Re-evaluate `F` after the update and log linear and nonlinear histories.

## Expected numerical effect

For a small, smooth perturbation, `r_F` decreases and converges to a stable
fixed point without relying on hidden damping.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_picard_fixed
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- The fixed small case reaches `r_F<=1e-6` within 500 iterations.
- Every linear solve reaches relative residual `<=1e-10`.
- Final means meet the gauge threshold and all outputs remain finite.

## Regression surface

- Source pairing, stale/current state use, buffer aliasing, and hierarchy reuse.

## Failure and rollback policy

- Failure to converge remains a recorded research result; do not add adaptive
  behavior in this branch.
- Use a smaller manufactured amplitude only if its intended difficulty was
  specified before the run.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Picard map uses one immutable source state.
- [ ] Sequential solves reuse operator, hierarchy, and workspace.
- [ ] Fixed relaxation and projection are tested.
- [ ] Residual/linear histories and physical metrics are recorded.
- [ ] Target case and full regressions pass with human review.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-14 complete and selects SF-15.
<!-- completion-checklist:end -->

## Advancement rule

SF-15 may globalize this unchanged Picard map with adaptive relaxation.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
