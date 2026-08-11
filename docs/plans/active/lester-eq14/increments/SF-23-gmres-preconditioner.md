# SF-23 — Restarted GMRES and block preconditioner

- State: `pending`
- Goal: `Implementar GMRES reiniciado con precondicionador bloque diagonal.`
- Depends on: `SF-22`
- Unlocks: `SF-24`
- Branch: `science/lester-sf-23-gmres-preconditioner`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf26-gmres-preconditioner`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Solve the coupled Newton linearization with bounded GPU memory and the already
validated diffusion blocks as a physics-informed preconditioner.

## Preconditions

- SF-22 provides an accepted matrix-free Jv.

## In scope

- Restarted double-precision GMRES, per-component projection, block diagonal
  `P=diag(A,A)`, fixed V-cycle application, and memory/iteration reports.

## Out of scope

- Newton globalization, adaptive/mixed preconditioning, and FGMRES production.

## Files and symbols

- Add a coupled restarted GMRES module or a narrowly reusable numerical solver.
- Add a block preconditioner adapter applying two projected MG V-cycles.

## Implementation specification

1. Default restart to 10; allow 15 only through explicit config and measured
   memory.  Store `2*(m+1)` scalar fields for the basis plus work vectors.
2. Use modified Gram-Schmidt with reorthogonalization trigger and Givens updates.
3. Project each coupled basis vector and true residual component.
4. Keep the preconditioner fixed and linear in this increment, so standard
   GMRES is valid; introduce FGMRES only when SF-29 makes it necessary.
5. Recompute the true residual at each restart and termination.

## Expected numerical effect

The block preconditioner reduces matrix-free Krylov iterations without changing
the converged linearized correction.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_gmres
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Small explicitly assembled/reference linearizations match GMRES corrections
  within `1e-8` relative.
- Reported and true residuals agree at restart/termination.
- Block preconditioning reduces iterations on the fixed suite.
- Restart-10 basis cost at `256^3` is approximately 2.75 GiB and agrees with
  actual allocation accounting.

## Regression surface

- Coupled scaling, orthogonality loss, MG linearity, memory, and restart logic.

## Failure and rollback policy

- If a supposedly fixed V-cycle is observably nonlinear, stop and either make
  it fixed or revise this increment to FGMRES through an explicit decision.
- Do not raise restart beyond the V100 memory budget to force convergence.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Restarted GMRES and true-residual checks are implemented.
- [ ] Block diagonal projected preconditioner is implemented.
- [ ] Reference correction and iteration-reduction tests pass.
- [ ] Restart memory accounting is measured and documented.
- [ ] Gate 3A regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-23 complete and selects SF-24.
<!-- completion-checklist:end -->

## Advancement rule

SF-24 may use the accepted linear solver inside a globalized Newton iteration.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
