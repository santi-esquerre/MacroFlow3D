# SF-02 — Discrete operator contract

- State: `pending`
- Goal: `Fijar y demostrar el contrato discreto del operador periódico A=-div(q grad).`
- Depends on: `SF-01`
- Unlocks: `SF-03`
- Branch: `science/lester-sf02-operator-contract`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf02-operator-contract`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Resolve the current sign/comment mismatch and prove that the reused matrix-free
operator represents the positive semidefinite periodic diffusion required by
the Lester formulation.

## Preconditions

- SF-01 reference operators and manufactured fixtures are accepted.

## In scope

- Tests and the smallest wrapper/comment corrections needed for `A(q)`.
- Harmonic face interpolation of the cell-centered coefficient `q`.

## Out of scope

- Nullspace projection, iterative solves, multigrid, and nonlinear terms.

## Files and symbols

- Inspect/extend `src/numerics/operators/VarCoeffLaplacian.*` and its wrappers.
- Add operator-contract cases to `streamfunction_operator_tests`.
- Update `src/numerics/AGENTS.md` only if a durable sign convention changes.

## Implementation specification

1. Preserve existing flow callers by adding an explicitly named positive
   wrapper if the underlying kernel remains `div(q grad)`.
2. Compute every face coefficient as the harmonic mean of `q`, not as the
   inverse of the harmonic mean of `K`.
3. Exercise triply periodic boundaries and verify the constant null mode.
4. Test constant and smooth positive `q` against the independent CPU reference.

## Expected numerical effect

No flow behavior change.  New Lester callers gain an unambiguous positive
operator contract.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- `RMS(A*1) <= 1e-13` after scale normalization.
- Symmetry defect `|x.Ay-y.Ax|/(|x.Ay|+|y.Ax|) < 1e-12`.
- Discrete energy is nonnegative to roundoff.
- Manufactured L2 convergence order is at least 1.8.

## Regression surface

- Existing flow CG/PCG sign wrappers and multigrid residual conventions.

## Failure and rollback policy

- Do not alter flow signs to satisfy a Lester test.
- If harmonic `q` conflicts with a current generic operator assumption, add a
  named coefficient policy and record the decision rather than branching
  silently on caller identity.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Actual legacy sign is documented with a regression test.
- [ ] Positive Lester wrapper applies `A=-div(q grad)`.
- [ ] Harmonic-q face tests pass.
- [ ] Nullspace, symmetry, energy, and convergence thresholds pass.
- [ ] Existing flow tests and smoke pass unchanged.
- [ ] Human review and evidence are recorded.
- [ ] Dashboard marks SF-02 complete and selects SF-03.
<!-- completion-checklist:end -->

## Advancement rule

SF-03 may use this accepted operator contract to define its gauge projector.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
