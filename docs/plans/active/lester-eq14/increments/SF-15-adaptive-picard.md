# SF-15 — Adaptive Picard

- State: `active`
- Goal: `Añadir relajación adaptativa, rechazo y detección de estancamiento a Picard.`
- Depends on: `SF-14`
- Unlocks: `SF-16`
- Branch: `science/lester-sf15-adaptive-picard`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-09T03:16Z`
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
| 2026-08-09T03:16Z | activation on `master=b7d0c86` (SF-14 closure merged via PR #26) | SF-15 activated after verifying `NEXT: SF-15`, SF-14 `done`, and checker `OK (29 increments, next=SF-15)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) globalization parameters live in a new `AdaptivePicardConfig` composed as `config.adaptive` with `enabled` defaulting to **true** (the plan's progression makes adaptive Picard the operative solver); `enabled=false` reproduces the SF-14 fixed path exactly, and the accepted `picard_fixed_*` test cases are pinned to `enabled=false` so they keep testing the fixed map they were written for. (2) The "Armijo-like residual safeguard" is operationalized as accept iff `r_F(trial) <= (1 - c*omega_try) * r_F(accepted)` with `c` default `1e-4` (configurable, validated); a residual increase can therefore never be accepted, and is never relabeled stagnation. (3) Trial state lives in **two new device fields** (`u_trial1/2`, +2 fine-grid-equivalent fields — memory option (a) tolerates; the SF-12 estimator, memory-report categories, and the api-workspace closed-form test are amended coherently); the block solutions `u_hat` stay immutable in `f1`/`f2` during backtracking (the expensive Picard map is NOT recomputed — only the relaxed candidate and its residual/diagnostics are), and trial residual outputs go to the otherwise-idle `rhs1`/`rhs2` buffers. (4) Dashboard degeneracy guards operationalized per trial: reject on nonfinite trial residual/sources; reject if the trial's unexplained degenerate fraction (SF-11 split at the FIRST configured diagnostics threshold) exceeds `max_unexplained_fraction` (default 0.01) OR exceeds `growth_factor*f_prev + growth_offset` (defaults 2, 1e-4) where `f_prev` is the last ACCEPTED state's fraction; reject if the |c| 0.1% percentile (SF-10 residual histogram, `residual_histogram_percentile`) collapses by more than one decade vs the last accepted state while the unexplained fraction did not stay <= `f_prev` (the "without matching Darcy low-speed population" reading); trial SF-11 diagnostics are evaluated only when diagnostics degeneracy thresholds are configured (guards vacuous otherwise, saving per-trial cost). (5) Omega policy locked to the dashboard: start `config.picard.omega` (0.25), halve on rejection clamped to `omega_min=0.01` with exactly one final trial at the floor (a rejected floor trial is the structured failure), grow ×1.2 after three consecutive zero-backtrack ("easy") acceptances, capped at `omega_max=1`; omega persists across iterations. (6) Stagnation: after at least `window=10` accepted steps, exit when `r_F_now > (1 - 0.01) * r_F(window ago)`; forced-branch tests may set extreme window/reduction values to reach the branch deterministically, never to hide a residual increase. (7) `StreamfunctionSolveStatus` stays 4-valued; the report gains `PicardExitReason { none, converged, budget_exhausted, linear_block_failure, stagnated, omega_floor_rejected }`, `final_omega`, and a per-trial `PicardTrialRecord` history (iteration, omega, trial r_F, outcome) alongside the unchanged SF-14 accepted-state history, so every exit carries iteration, residual, omega, and reason and every trial is identifiable. (8) "Forced bad trials leave accepted fields bitwise unchanged" is verified by re-evaluating the residual on the returned fields after an omega-floor failure and matching the last accepted r_F bitwise. | Base commit is this activation commit on `master=b7d0c86`. Gate 1 + Gate 2 + Gate 3A apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. Memory: +2 device fields (trial pair) recorded explicitly against option (a). | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
