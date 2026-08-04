# SF-13 — Homogeneous solver

- State: `pending`
- Goal: `Resolver de extremo a extremo el caso homogéneo exacto.`
- Depends on: `SF-12`
- Unlocks: `SF-14`
- Branch: `science/lester-sf13-homogeneous-solver`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf13-homogeneous-solver`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Validate the full linear, affine, gauge, residual, and diagnostic path on the
known exact solution before enabling nonlinear iteration.

## Preconditions

- SF-12 exposes the stable API and preallocated workspace.

## In scope

- A solver entry point restricted to the homogeneous/zero-source case and a
  dedicated exact-control executable or CTest case.

## Out of scope

- Picard, user-facing pipeline config, continuation, and heterogeneous fields.

## Files and symbols

- Implement the initial `solve_streamfunctions` path in
  `StreamfunctionSolver.cu`.
- Add homogeneous cases for `16^3`, `32^3`, and `64^3`.

## Implementation specification

1. Set `K=q=1`, `u1=u2=0`, benchmark affine gradients, and periodic BCs.
2. Assemble and project both zero RHSs, solve/project, then evaluate residual
   and every SF-11 diagnostic.
3. Exercise repeated calls with the same workspace to verify hierarchy reuse
   and stable gauge.

## Expected numerical effect

The exact fluctuations remain zero and the cross-gradient reconstructs uniform
Darcy velocity.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_homogeneous
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- `RMS(u1),RMS(u2),RMS(S1),RMS(S2) <= 1e-13` in normalized units.
- Gauge meets the SF-03 threshold at all three grids.
- Velocity reconstruction relative error `<=1e-13`.
- No metric degrades under repeated solves.

## Regression surface

- Solver orchestration, workspace reuse, affine sign/pairing, and exact-zero
  convergence handling.

## Failure and rollback policy

- Any nonzero systematic source or velocity error blocks nonlinear work.
- Do not relax tolerances to hide an affine or sign defect.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Homogeneous end-to-end path is implemented.
- [ ] `16^3`, `32^3`, and `64^3` exact controls pass.
- [ ] Gauge and repeated-workspace tests pass.
- [ ] Gate 3A report contains all applicable metrics.
- [ ] Full regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-13 complete and selects SF-14.
<!-- completion-checklist:end -->

## Advancement rule

SF-14 may add fixed-relaxation Picard after the exact control is merged.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
