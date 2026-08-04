# SF-08 — Hessian-vector products and B

- State: `pending`
- Goal: `Implementar productos Hessiano-vector y la construcción de B sin almacenar Hessianos.`
- Depends on: `SF-07`
- Unlocks: `SF-09`
- Branch: `science/lester-sf08-hessian-vector-b`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf08-hessian-vector-b`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A operator subset`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Evaluate the directional curvature required by Lester equation (14) with a
small periodic stencil and without a nine-component Hessian memory cost.

## Preconditions

- SF-07 total gradients and derivative conventions are accepted.

## In scope

- Direct `H(psi2)*grad(psi1)`, `H(psi1)*grad(psi2)`, and their difference `B`.

## Out of scope

- `c`, `S1`, `S2`, denominator regularization, and source fusion.

## Files and symbols

- Extend `DifferentialOperators` with a reference CUDA kernel.
- Add CPU analytic Hessian-vector and `B` controls.

## Implementation specification

1. Differentiate only periodic fluctuations in the Hessian; affine parts have
   zero Hessian.
2. Use centered diagonal and mixed second derivatives.
3. Load the radius-one union stencil: center, six axial neighbors, and twelve
   edge-diagonal neighbors per field.
4. Form products and `B=H(psi2)g1-H(psi1)g2` in registers.

## Expected numerical effect

Directional curvature converges at second order without persistent Hessian
buffers.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Each Hessian-vector component has L2 order at least 1.8.
- `B` matches CPU reference within the measured discretization error.
- Analytic controls with parallel/constant gradients produce `B` at roundoff.

## Regression surface

- Mixed-derivative signs, periodic diagonal indexing, register pressure, and
  future source fusion.

## Failure and rollback policy

- Keep the unfused kernel and componentwise diagnostics if a fused register-only
  implementation obscures an error.
- Do not allocate full Hessian fields as a workaround.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Direct Hessian-vector products are implemented without Hessian storage.
- [ ] B construction and analytic controls pass.
- [ ] Component convergence order is at least 1.8.
- [ ] Temporary memory is measured and documented.
- [ ] Full regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-08 complete and selects SF-09.
<!-- completion-checklist:end -->

## Advancement rule

SF-09 may construct regularized nonlinear sources from the accepted `B` and
gradient definitions.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
