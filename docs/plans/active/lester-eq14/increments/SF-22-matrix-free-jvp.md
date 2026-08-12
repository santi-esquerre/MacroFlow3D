# SF-22 — Matrix-free Jacobian-vector product

- State: `active`
- Goal: `Implementar el vector acoplado y el producto Jacobiano-vector matrix-free.`
- Depends on: `SF-21`
- Unlocks: `SF-23`
- Branch: `science/lester-sf22-matrix-free-jvp`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-12T01:20Z`
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
| 2026-08-12T01:20Z | activation on `master=6ad39d0` (SF-21 partial closure + Newton-first restructuring merged via PR #34) | SF-22 activated after verifying `NEXT: SF-22`, SF-21 `done`, checker `OK (30 increments, next=SF-22)`. Interpretive decisions PRESPECIFIED before implementation: (D1) **weighted norm** for the coupled state/direction: ||(a,b)||_w = sqrt( (RMS(a)/g1)^2 + (RMS(b)/g2)^2 ) with g1 = source_config.v_rms and g2 = 1 — exactly the per-component scales of the accepted residual normalization (r1, r2), so delta is dimensionless-consistent across both components; documented in the header. (D2) **delta policy** (spec item 3): forward difference, delta = sqrt(machine_eps) * (1 + ||Psi||_w) / ||p||_w; ||p||_w = 0 or nonfinite -> std::invalid_argument; configurable clamp [delta_min=1e-12, delta_max=1e2] with clamp counters surfaced in the report struct; any nonfinite in the perturbed residual -> structured failure, never silent. (D3) **projection discipline**: direction components are mean-zero-projected before use; the perturbed state is projected after the axpy (idempotent defense, same rationale as the SF-15 trial projection); the base state is a caller contract (already projected by the solver); Jv output means are MEASURED against the projector threshold (F is discretely mean-zero by construction, so nonzero Jv means indicate a defect — no silent output re-projection). (D4) **cached base residual** (spec item 4): a prepare_base(state) call evaluates and stores F(Psi) once per Newton state; every apply(p) performs exactly ONE perturbed residual evaluation; an evaluation counter is exposed and contract-tested. (D5) **central difference is test-only and built from the public API**: Jv_central = (Jv_forward(p) - Jv_forward(-p))/2 (identical delta since ||-p||_w = ||p||_w) — no library test-only code path. (D6) **prespecified fixtures for the U-study** (spec threshold 1): trig manufactured states at 16^3 and 32^3 (domain length 1, dx=1/n — the SF-21 C01 lesson is baked in) plus a CONVERGED adaptive-Picard state on the trig conductivity (the accepted SF-15/SF-20 fixture family), each against >=3 direction types (fixed-seed random mean-zero, gradient-like, single-Fourier-mode); sweep delta over >=6 decades bracketing the policy delta; GATES: observable U-shape (discrepancy decreases then increases), forward-vs-central relative discrepancy <= 1e-5 at the policy delta on the small cases, Jv component means <= the projector threshold used by the mean-zero contract tests. (D7) **eta=0 linearity oracle**: at eta=0, J p = [A p1; A p2] exactly; Jv must match the direct operator application within relative 1e-6 (FD cancellation noise budget at delta~1e-8, prespecified). (D8) venue policy stands (workers: local compile gate only; ALL test execution on the remote V100 by the orchestrator; checksum-verified syncs + md5 spot checks). Scope note recorded: no Jacobian matrix, no Hessian fields, no Krylov/Newton code in this increment. Branch field normalized to the house pattern (science/lester-sf22-...). | Gate 1 + Gate 2 + Gate 3A apply; human review required, so the PR stops at `awaiting_review` with `NEXT` unchanged. | Build the intra-increment DAG; delegate T01 (library) then T02 (tests) to isolated workers. |
