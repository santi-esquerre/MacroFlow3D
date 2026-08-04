# SF-15 — Adaptive Picard

- State: `pending`
- Goal: `Añadir relajación adaptativa, rechazo y detección de estancamiento a Picard.`
- Depends on: `SF-14`
- Unlocks: `SF-16`
- Branch: `science/lester-sf15-adaptive-picard`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf15-adaptive-picard`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Globalize Picard without changing its fixed-point map and make every rejected
or stalled update observable and recoverable.

## Preconditions

- SF-14 fixed-relaxation map and histories are accepted.

## In scope

- Trial buffers, Armijo-like residual safeguard, adaptive `omega`, rollback,
  degeneracy rejection, maximum iterations, and stagnation status.

## Out of scope

- Continuation parameter changes, Anderson, and Newton.

## Files and symbols

- Extend nonlinear control/report code in `StreamfunctionSolver`.
- Add deterministic tests that force accept, reject, minimum-step, and
  stagnation branches.

## Implementation specification

1. Start `omega=0.25`; halve a rejected trial to a minimum `0.01`.
2. Grow by `1.2` after three easy accepted trials, capped at one.
3. Reject nonfinite trials and dashboard-defined unexplained degeneracy growth.
4. Keep the last accepted pair immutable during backtracking; recompute only
   trial residual/diagnostics, not the expensive Picard map.
5. Flag stagnation after less than 1% residual reduction in ten accepted steps.

## Expected numerical effect

Residual growth is globally controlled, failed trials do not corrupt state, and
failure reason is deterministic.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_picard_adaptive
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Forced bad trials leave accepted fields bitwise unchanged.
- `omega` stays in `[0.01,1]` and follows the specified transition sequence.
- All exit statuses include iteration, residual, omega, and reason.

## Regression surface

- Trial/current buffer aliasing, extra residual evaluations, and degeneracy
  classification.

## Failure and rollback policy

- Do not accept a residual increase by relabeling it stagnation.
- If `omega_min` fails, return a structured failure for continuation to handle
  later.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Adaptive omega and rollback are implemented.
- [ ] Accept/reject/minimum/stagnation branches have deterministic tests.
- [ ] Degeneracy and nonfinite policies match the dashboard.
- [ ] Histories identify every trial and accepted state.
- [ ] Gate 3A regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-15 complete and selects SF-16.
<!-- completion-checklist:end -->

## Advancement rule

SF-16 may expose this validated solver through configuration and pipeline I/O.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
