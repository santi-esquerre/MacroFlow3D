# SF-14 — Fixed-relaxation Picard

- State: `active`
- Goal: `Implementar Picard secuencial con relajación fija.`
- Depends on: `SF-13`
- Unlocks: `SF-15`
- Branch: `science/lester-sf14-fixed-picard`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-08T23:08Z`
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
| 2026-08-08T23:08Z | activation on `master=13717ee` (SF-13 closure merged via PR #25) | SF-14 activated after verifying `NEXT: SF-14`, SF-13 `done`, and checker `OK (29 increments, next=SF-14)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) Picard initialization is the accepted SF-13 zero-source solve (the paper's homogeneous initial estimate); `solve_streamfunctions` is extended in place, no second entry point. (2) **RHS borrowing:** the combined, mean-zero-projected `G_i = rhs_affine_i - eta*q*S_pair` computed internally by the accepted SF-10 residual evaluator ARE the Picard block RHSs (verified in `ResidualEvaluator.cu` lines 376-388); SF-14 adds an additive, read-only device-view accessor on `StreamfunctionResidualWorkspace` (valid until the next enqueue) instead of new solver-owned gradient/B/S fields — zero new device memory, no numerical change to SF-10, consistent with the accepted borrow-don't-own philosophy of the SF-12 memory decision record. (3) New `FixedPicardConfig { max_iter=500, tolerance=1e-6, omega=0.25 }` composed into `StreamfunctionSolverConfig` and host-validated (`max_iter >= 0`, finite `tolerance > 0`, finite `omega` in `(0,1]`); defaults are the dashboard-locked values. (4) Status semantics extended per the SF-12/13 enum's stated intent: `converged` now means `r_F <= tolerance`; `not_converged` means the iteration limit was exhausted or a linear block failed; `invalid_problem` unchanged; the SF-13 homogeneous controls converge at 0 Picard iterations (`r_F = 0`) and must pass unchanged. (5) `v_rms` is measured once at solve start via SF-11 diagnostics (a state-independent Darcy property) and the diagnostics are re-run on the final state for the report. (6) Manufactured nonlinear cases prespecified BEFORE any run per the spec's rollback rule: gating cases = `16^3` and `32^3` unit torus, `Y = a*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)`, `K = exp(Y)`, `a = 0.25`, uniform reference Darcy `(1,0,0)`, defaults (`omega=0.25`, `tol=1e-6`, `<=500` iters, PCG rtol `1e-10`); recorded research case `a = 0.5` at `32^3` whose convergence outcome is reported either way (gating only on finiteness and recorded histories). For heterogeneous `K`, `e_v`/invariance against the uniform reference are diagnostic-only (no Darcy solve exists until SF-19) and are NOT acceptance metrics. (7) Relaxation `u <- (1-omega)*u + omega*u_hat` uses the existing allocation-free `blas::scal` + `blas::axpy`; the block solutions `u_hat` live in the workspace's `f1`/`f2` buffers (overwritten by the next residual evaluation anyway); both fields are projected after relaxation with the accepted projector. (8) The report gains a per-iteration Picard history (host-side vector reserved to `max_iter+1`; host logging, not device hot-loop allocation). | Base commit is this activation commit on `master=13717ee`. Gate 1 + Gate 2 + Gate 3A apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. Memory option (a) remains in effect; SF-14 adds zero device fields. | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
