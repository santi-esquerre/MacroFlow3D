# SF-20 — Heterogeneity continuation

- State: `pending`
- Goal: `Implementar continuación adaptativa en heterogeneidad para campos gaussianos físicos.`
- Depends on: `SF-19`
- Unlocks: `SF-21`
- Branch: `science/lester-sf20-heterogeneity-continuation`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf20-heterogeneity-continuation`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + Gate 4`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Reach the target lognormal conductivity without assuming strongly heterogeneous
Picard convergence from a homogeneous initial state.

## Preconditions

- SF-19 provides periodic `Y`, Darcy flow, and target mean-flux semantics.

## In scope

- `K_lambda=exp(lambda*Y)`, q/hierarchy rebuilding per accepted stage, adaptive
  lambda steps, eta rescue, and small physical Gaussian runs.

## Out of scope

- Grid transfer, Anderson, `256^3`, and exponential covariance.

## Files and symbols

- Extend `ContinuationController`, solver config/history, and benchmark configs.
- Reuse the affine-periodic Darcy solve at each lambda stage where the reference
  velocity is required.

## Implementation specification

1. Begin at lambda zero with exact zero fluctuations and permit eta one.
2. Use lambda step `0.1`, minimum `0.0125`, maximum `0.2`; halve on failure and
   grow by `1.5` after two easy stages.
3. On failed lambda, restore the accepted state, solve the current lambda at
   eta zero, then ramp eta to one before retrying epsilon reduction.
4. Rebuild `q` and one MG hierarchy per lambda and reuse it across all nonlinear
   iterations/stages at that lambda.

## Expected numerical effect

Small smooth Gaussian cases reach lambda one with controlled residual and
physical metrics instead of catastrophic first-step divergence.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_continuation
./build/wsl-debug/macroflow3d_pipeline <fixed-32-or-64-gaussian-config>
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Fixed-seed `32^3` smoke reaches lambda one for `sigma_Y^2=0.25` and one.
- The `64^3` suite reaches lambda one for `0.25,1,2.25,4` with `ell/h=8`.
- Every accepted stage reports `r_F`, velocity, invariance, divergence,
  degeneracy, eta, epsilon, and MG rebuild count.

## Regression surface

- Continuation rollback, conductivity overflow/underflow, Darcy recomputation,
  hierarchy lifetime, and total runtime.

## Failure and rollback policy

- Minimum lambda-step failure is a recorded physical/numerical failure; do not
  skip to a later lambda.
- `sigma_Y^2=6.25` is prohibited in this increment.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Lambda stepping and hierarchy lifecycle are implemented.
- [ ] Eta rescue follows the documented ordering.
- [ ] Fixed-seed 32/64 Gaussian suites meet the acceptance set.
- [ ] Full Gate 3A metrics and experiment notes are recorded.
- [ ] Gate 4 interpretation and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-20 complete and selects SF-21.
<!-- completion-checklist:end -->

## Advancement rule

SF-21 may prolong accepted lambda-one solutions to finer versions of the same
periodic realization.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
