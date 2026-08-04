# SF-06 — Affine-periodic right-hand sides

- State: `pending`
- Goal: `Representar correctamente las partes afines y ensamblar sus lados derechos periódicos.`
- Depends on: `SF-05`
- Unlocks: `SF-07`
- Branch: `science/lester-sf06-affine-periodic-rhs`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf06-affine-periodic-rhs`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Represent multi-valued affine streamfunctions without breaking periodic storage
or omitting the variable-coefficient affine forcing.

## Preconditions

- SF-05 validates the periodic `A(q)` linear path.

## In scope

- `AffineGauge`, periodic fluctuation semantics, and assembly of
  `div(q*gbar_i)` with the exact face flux convention of `A`.

## Out of scope

- Gradient/Hessian kernels, nonlinear sources, and full solver orchestration.

## Files and symbols

- Add initial types under `src/physics/streamfunctions/`.
- Add an affine RHS kernel/helper using the operator coefficient policy.
- Add CPU/GPU manufactured tests to the streamfunction test target.

## Implementation specification

1. Store `u1`, `u2` only; store affine gradients separately as three-component
   constants.
2. Default benchmark gradients to `(0,vbar,0)` and `(0,0,1)`.
3. Assemble opposite face flux differences with harmonic `q_f` and periodic
   neighbors; then project and report the raw mean.
4. Do not evaluate an affine scalar through wrapped coordinates.

## Expected numerical effect

The fluctuation equations remain periodic and compatible while representing
the correct total streamfunction gradients.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Affine RHS is zero to roundoff for constant `q`.
- Raw compatibility defect is at roundoff for periodic smooth `q`.
- Smooth manufactured variable-`q` RHS has L2 order at least 1.8.

## Regression surface

- Coefficient face rules, periodic indexing, units of `vbar`, and gauge policy.

## Failure and rollback policy

- Do not approximate the affine part by storing a discontinuous sawtooth field.
- Any mismatch with `A` must be fixed by sharing the coefficient/flux primitive,
  not by projecting away a large compatibility defect.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Affine and periodic components have separate types/ownership.
- [ ] RHS uses the exact `A` face coefficient convention.
- [ ] Constant and smooth-q thresholds pass.
- [ ] Raw and projected compatibility defects are reported.
- [ ] Full regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-06 complete and selects SF-07.
<!-- completion-checklist:end -->

## Advancement rule

SF-07 may calculate total gradients from the accepted affine representation.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
