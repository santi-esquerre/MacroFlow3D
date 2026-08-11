# SF-22 — Matrix-free Jacobian-vector product

- State: `pending`
- Goal: `Implementar el vector acoplado y el producto Jacobiano-vector matrix-free.`
- Depends on: `SF-21`
- Unlocks: `SF-23`
- Branch: `science/lester-sf-22-matrix-free-jvp`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf25-matrix-free-jvp`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Reuse the accepted nonlinear residual to expose Jacobian action without
assembling or storing a coupled Jacobian.

## Preconditions

- SF-21 (partial closure, owner option (a) 2026-08-11) establishes the accepted Picard/Anderson residual evaluator and the heterogeneity-continuation machinery whose eta=1 non-contractive plateau motivates this Newton phase.

## In scope

- Coupled `2N` vector views, perturbation workspaces, finite-difference Jv,
  projection, step-size policy, and directional derivative tests.

## Out of scope

- Krylov iteration, Newton steps, line search, and mixed precision.

## Files and symbols

- Add `CoupledVectorView` and `JacobianVectorProduct.cuh/.cu` under
  `src/physics/streamfunctions/`.
- Reuse `ResidualEvaluator` without a second PDE implementation.

## Implementation specification

1. Present persistent fields as two views while allocating Krylov storage as
   contiguous `2N` buffers.
2. Project each component of direction and perturbed state.
3. Use forward difference
   `delta=sqrt(eps)*(1+||Psi||_w)/||p||_w` with documented weighted norm and
   configurable safeguards against under/overflow.
4. Cache `F(Psi)` for all Jv calls at one Newton state.
5. Validate against a central difference used only by tests.

## Expected numerical effect

Jv converges to the directional derivative over an observable finite-difference
step range and preserves the two zero-mean subspaces.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_jvp
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Forward/central Jv discrepancy has the expected U-shaped step study and meets
  a predeclared `1e-5` relative target at the chosen delta on small cases.
- Both Jv component means meet the projector threshold.
- No Jacobian matrix or Hessian fields are allocated.

## Regression surface

- Residual determinism, weighted units of coupled components, cancellation,
  and extra residual workspace.

## Failure and rollback policy

- Do not tune delta on the final benchmark only; use manufactured and Picard
  states across several norms.
- If forward difference is unreliable, document the range before considering a
  more expensive central production option.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Coupled vector/view semantics are implemented.
- [ ] Jv reuses the exact residual evaluator and cached base residual.
- [ ] Delta policy and weighted norm are documented and tested.
- [ ] Central-difference comparison and gauge thresholds pass.
- [ ] Gate 3A regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-22 complete and selects SF-23.
<!-- completion-checklist:end -->

## Advancement rule

SF-23 may use this Jv as the matrix-free operator in restarted GMRES.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
