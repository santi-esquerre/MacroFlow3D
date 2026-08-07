# SF-10 — Coupled residual

- State: `active`
- Goal: `Implementar el residuo acoplado y las reducciones adimensionales del solver.`
- Depends on: `SF-09`
- Unlocks: `SF-11`
- Branch: `science/lester-sf10-coupled-residual`
- Worktree: `Claude-managed per-node isolated worktrees (native isolation: worktree)`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-07T17:39Z`
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
| 2026-08-07T17:39Z | active; base `master=origin/master=e81cd47378849370272f8ed727659b025de16f44` | Activated SF-10 documentation state under the Claude Code orchestration harness. | Preflight verified on the default branch: SF-09 `done` via PR #19 (`75eafef`) and closure PR #20 (`e81cd47`); checker PASS (`29 increments, next=SF-10`); clean tree. Reuse surface inspected in code: `LesterPositiveDiffusionOperator` (SF-02), `MeanZeroProjector`/`MeanZeroWorkspace` (SF-03), `assemble_affine_periodic_rhs` + workspace/diagnostics (SF-06), `enqueue_total_streamfunction_gradients`/`enqueue_streamfunction_hessian_vector_b` (SF-07/08), `enqueue_streamfunction_nonlinear_sources` (SF-09), `blas` reductions with `ReductionWorkspace`. Interpretive decision recorded for human review: the authoritative convergence residual uses the projected combined right-hand side, `F_i = A u_i - P(div_h(q gbar_i) - eta q S_pair)`, because the locked decisions require projecting right-hand sides and `A u` is discretely mean-zero on the periodic domain, so the literal unprojected spec formula would stagnate at the raw compatibility defect; raw RHS means remain reported as diagnostics and tests verify the projected/literal relationship explicitly. sccache remains disabled locally (documented in SF-09 activation). Persistent Goal `Implementar el residuo acoplado y las reducciones adimensionales del solver.`; delivery branch `science/lester-sf10-coupled-residual`. | Build the SF-10 intra-increment DAG and delegate implementation to isolated workers. |
