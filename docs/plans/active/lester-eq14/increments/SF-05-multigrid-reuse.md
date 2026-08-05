# SF-05 — Multigrid reuse

- State: `active`
- Goal: `Validar y adaptar el multigrilla cell-centered como precondicionador de A(q).`
- Depends on: `SF-04`
- Unlocks: `SF-06`
- Branch: `science/lester-sf05-multigrid-reuse`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf05-multigrid-reuse`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-05T15:11Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Test the priority reuse hypothesis rather than assuming the current flow
multigrid remains a valid symmetric preconditioner for periodic `q=1/K`.

## Preconditions

- SF-04 projected PCG converges with a simple preconditioner.

## In scope

- Generic coefficient naming/coarsening, sign adapter, per-level zero-mean
  projection, and quantitative PCG/MG tests.

## Out of scope

- Replacing multigrid, changing transfer order, or optimizing kernels.

## Files and symbols

- Extend `src/multigrid/MGHierarchy*`, V-cycle, GSRB, residual, coefficient
  coarsening, restriction, and prolongation only where tests require it.
- Add a projected, sign-correct preconditioner adapter.

## Implementation specification

1. Build the hierarchy from cell-centered `q` once and reuse it.
2. Use geometric 2x2x2 coefficient coarsening initially; record that for
   `q=1/K` it equals the inverse geometric coarsening of `K`.
3. Project level RHS, residuals, and corrections without replacing the
   physical smoother.
4. Check preconditioner symmetry and compare PCG iteration counts against
   unpreconditioned projected PCG.

## Expected numerical effect

The preconditioner reduces iteration count while the outer PCG retains the
same converged zero-mean solution.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Relative residual `<=1e-10` on constant and smooth positive `q`.
- At most 100 PCG iterations on the fixed `32^3` and `64^3` suite.
- Iteration growth from `32^3` to `64^3` is no more than 50%.
- MG-preconditioned and reference solutions agree within solver tolerance.

## Regression surface

- Flow MG hierarchy construction, negative-operator preconditioning, transfer
  kernels, and memory ownership.

## Failure and rollback policy

- If reuse fails symmetry or convergence criteria, document the failed
  hypothesis and make the smallest local correction; do not replace MG in this
  increment.
- A need for a new multigrid design requires a new decision record and plan
  revision.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] q hierarchy and sign adapter are explicit.
- [ ] Level projection and nullspace behavior are tested.
- [ ] Symmetry, residual, iteration, and mesh-growth thresholds pass.
- [ ] Existing flow MG results remain unchanged.
- [ ] Human review and evidence are recorded.
- [ ] Dashboard marks SF-05 complete and selects SF-06.
<!-- completion-checklist:end -->

## Advancement rule

SF-06 may assemble affine-periodic right-hand sides using the validated
operator and preconditioner.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-05T15:11Z | `c4d77c3`, active | Verified SF-05 is the dashboard `NEXT`, created the exact persistent runtime Goal, and created the canonical SF-05 branch/worktree after completing the required scientific, numerical, validation, workflow, architecture, and code preflight. | `master=origin/master=c4d77c3`; increment checker passed with `next=SF-05`; SF-04 is `done`; Goal is `Validar y adaptar el multigrilla cell-centered como precondicionador de A(q).`; existing MG uses the legacy negative-sign operator, coefficient buffers named `K`, `PinSpec`, and no per-level mean-zero projection. | Build the explicit SF-05 task DAG around coefficient hierarchy, projected sign adapter, quantitative controls, and legacy-flow regression evidence. |
