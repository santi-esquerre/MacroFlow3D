# SF-07 — Streamfunction gradients

- State: `active`
- Goal: `Implementar gradientes periódicos cell-centered con contribuciones afines.`
- Depends on: `SF-06`
- Unlocks: `SF-08`
- Branch: `science/lester-sf07-gradients`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf07-gradients`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A operator subset`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-05T18:35Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Provide one validated definition of total streamfunction gradient for nonlinear
sources, invariance metrics, and velocity reconstruction.

## Preconditions

- SF-06 defines periodic fluctuations and affine gradients.

## In scope

- Second-order centered, triply periodic cell-centered gradients for both
  streamfunctions, including affine constants.

## Out of scope

- Hessians, fused source kernels, face reconstruction, and higher order.

## Files and symbols

- Add `src/physics/streamfunctions/DifferentialOperators.cuh/.cu`.
- Add analytic and CPU/GPU gradient tests.

## Implementation specification

1. Use `dx`, `dy`, `dz` explicitly even though current production grids are
   isotropically spaced.
2. Wrap cell indices independently in all three directions.
3. Add the affine vector after differentiating the periodic fluctuation.
4. Provide an output-buffer API for tests; production fusion is deferred.

## Expected numerical effect

Affine fields are exact, periodic modes converge at second order, and all later
operators share the same total gradient convention.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Pure affine gradients agree to roundoff.
- Periodic trigonometric fields show L2 order at least 1.8.
- Linf errors decrease monotonically from `16^3` through `64^3`.

## Regression surface

- Grid spacing assumptions, periodic indexing, and future shared-memory stencil
  layout.

## Failure and rollback policy

- Retain the explicit-buffer reference kernel until convergence is demonstrated.
- Do not fuse or introduce fourth-order differences in this increment.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Total-gradient API and kernel are implemented.
- [ ] Affine exactness and periodic convergence tests pass.
- [ ] Spacing and indexing conventions are documented.
- [ ] Full regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-07 complete and selects SF-08.
<!-- completion-checklist:end -->

## Advancement rule

SF-08 may use the accepted total gradients in Hessian-vector products and `B`.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-05T18:35Z | active; master=`origin/master=031e1af` | Activated SF-07 documentation state. | Checker PASS (`next=SF-07`); dependency SF-06 done; persistent Goal `Implementar gradientes periódicos cell-centered con contribuciones afines.`; branch `science/lester-sf07-gradients`; worktree `~/src/MacroFlow3D/.agents/worktrees/lester-sf07-gradients`. | Build DAG. |
