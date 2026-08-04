# SF-10 — Coupled residual

- State: `pending`
- Goal: `Implementar el residuo acoplado y las reducciones adimensionales del solver.`
- Depends on: `SF-09`
- Unlocks: `SF-11`
- Branch: `science/lester-sf10-coupled-residual`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf10-coupled-residual`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Create the authoritative nonlinear convergence measure using exactly the same
discrete equations that Picard and future Newton iterations solve.

## Preconditions

- SF-09 provides accepted source terms; SF-06 provides affine RHS terms.

## In scope

- `F1=A*u1-div(q*gbar1)+eta*q*S2`, its paired `F2`, reusable reductions,
  dimensionless normalization, and histogram percentiles.

## Out of scope

- Velocity reconstruction, Picard updates, and convergence control.

## Files and symbols

- Add `src/physics/streamfunctions/ResidualEvaluator.cuh/.cu` and diagnostic
  reduction workspace.
- Reuse the exact `A`, affine RHS, and nonlinear source modules.

## Implementation specification

1. Compute `F` from operator and RHS arrays rather than re-discretizing the PDE.
2. Normalize component one by `q_rms*v_rms/L_ref` and component two by
   `q_rms/L_ref`; combine as `sqrt((r1^2+r2^2)/2)`.
3. Compute RMS/Linf and a fixed 512-bin logarithmic histogram for `|c|` without
   sorting or allocating inside an iteration.
4. Expose raw as well as normalized values in a POD report.

## Expected numerical effect

Convergence decisions become grid-size independent and consistent with the
linear equations.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Direct `A*u-b` and residual evaluator agree to reduction roundoff.
- CPU/GPU RMS and Linf agree within `1e-12` relative on deterministic fixtures.
- Histogram percentile bin error is bounded and documented.

## Regression surface

- Operator/source sign consistency, units, reduction synchronization, and
  percentile behavior for zeros.

## Failure and rollback policy

- Never accept an independently re-discretized residual as a substitute.
- Retain exact host reductions in tests if the production histogram needs later
  tuning.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Both coupled residual components use shared discrete primitives.
- [ ] Dimensionless normalization is implemented and tested.
- [ ] RMS, Linf, and histogram reductions use persistent workspace.
- [ ] CPU/GPU accuracy thresholds pass.
- [ ] Full regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-10 complete and selects SF-11.
<!-- completion-checklist:end -->

## Advancement rule

SF-11 may add physical diagnostics without changing this convergence residual.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
