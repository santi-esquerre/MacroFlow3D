# SF-27 — Globalized Newton-Krylov

- State: `pending`
- Goal: `Implementar Newton-Krylov globalizado y su fallback reproducible a Picard.`
- Depends on: `SF-26`
- Unlocks: `SF-28`
- Branch: `science/lester-sf27-newton-krylov`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf27-newton-krylov`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + Gate 4`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Accelerate locally converged difficult states without sacrificing the accepted
Picard continuation path as the authoritative robustness fallback.

## Preconditions

- SF-26 validates matrix-free GMRES and the block preconditioner.

## In scope

- Newton activation, inexact linear forcing, Armijo line search, state rollback,
  Picard rescue, continuation fallback, and comparative histories.

## Out of scope

- Mixed precision, replacing Picard defaults, and scientific benchmarks beyond
  the SF-24 parameter regime.

## Files and symbols

- Add `NewtonKrylovSolver.cuh/.cu` or a method implementation behind the stable
  streamfunction solver API.
- Extend config/report only when the method is functional.

## Implementation specification

1. Activate Newton after Picard reaches `r_F<1e-2`, or after documented
   stagnation below `1e-1` with acceptable nondegeneracy.
2. Solve `J delta=-F` with an inexact tolerance tied to current `r_F` and capped
   by configured minimum/maximum forcing terms.
3. Project corrections and use Armijo decrease of `0.5*r_F^2`, backtracking
   without overwriting the accepted state.
4. On Newton failure, restore state, run five accepted Picard steps, retry once,
   then reduce the current continuation step.
5. Log all activations, Jv/GMRES counts, line-search trials, fallbacks, and final
   physical metrics.

## Expected numerical effect

Near a valid solution Newton reduces nonlinear iterations while failed Newton
attempts return deterministically to the same safe Picard/continuation state.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_newton
scripts/remote run lester-newton -- "<fixed-picard-vs-newton-suite>"
scripts/remote wait lester-newton
```

## Acceptance thresholds

- Small cases reach the same final fields/metrics as Picard within nonlinear
  tolerance.
- Forced line-search, GMRES, and retry failures preserve accepted state.
- At least one fixed difficult case shows fewer nonlinear residual evaluations
  or lower wall time than Picard without reducing robustness across the suite.

## Regression surface

- Coupled scaling, line-search state ownership, continuation interaction,
  memory, and nonlinear stopping criteria.

## Failure and rollback policy

- Newton remains opt-in until it passes the full fixed suite.
- After the documented retry, return to Picard/reduced continuation; do not add
  silent fallback variants.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Activation, inexact solve, Armijo, rollback, and fallback are implemented.
- [ ] Forced failure paths preserve accepted state.
- [ ] Picard/Newton final-solution equivalence is demonstrated.
- [ ] Runtime/residual-evaluation comparison is recorded.
- [ ] Gate 3A/4 regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-27 complete and selects SF-28.
<!-- completion-checklist:end -->

## Advancement rule

SF-28 may study mixed precision only after the full double Newton path is
accepted and merged.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
