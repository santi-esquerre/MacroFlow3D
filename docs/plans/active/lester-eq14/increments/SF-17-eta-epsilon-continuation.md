# SF-17 — Eta and epsilon continuation

- State: `active`
- Goal: `Implementar continuación adaptativa en eta y epsilon con rollback.`
- Depends on: `SF-16`
- Unlocks: `SF-18`
- Branch: `science/lester-sf17-eta-epsilon-continuation`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-09T16:00Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Reach the unregularized nonlinear problem progressively while preserving the
last accepted state and making every homotopy decision reproducible.

## Preconditions

- SF-16 exposes adaptive Picard, histories, config, and output.

## In scope

- Reusable continuation-stage controller for `eta` and epsilon, adaptive steps,
  rollback, stage histories, and configuration.

## Out of scope

- `lambda`, stochastic generation, and grid continuation.

## Files and symbols

- Add `ContinuationController.hpp/.cu` under streamfunctions.
- Extend config/output only with eta and epsilon continuation fields.

## Implementation specification

1. For eta, use initial step `0.1`, minimum `0.0125`, maximum `0.25`, halve on
   failure, and grow by `1.5` after two easy stages.
2. Keep an accepted-state buffer and never overwrite it until a stage passes.
3. After eta one converges, reduce epsilon by decades from `1e-2` to `1e-6`;
   make `1e-8` configurable but not required.
4. Record parameter, attempted step, iterations, omega, residual, degeneracy,
   acceptance, and failure reason for every stage.

## Expected numerical effect

Controlled problems reach `eta=1` and lower epsilon without unrecoverable state
corruption.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_continuation
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Deterministic control reaches `eta=1` and `epsilon=1e-6`.
- Injected failures halve the step and restore the accepted state exactly.
- Minimum-step exhaustion returns a structured, logged failure.

## Regression surface

- State-buffer memory, parameter ordering, config serialization, and histories.

## Failure and rollback policy

- A stage failure never mutates the accepted solution.
- Do not skip failed parameter intervals or claim the fixed-epsilon result as
  the original system solution.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Eta stepping, epsilon stepping, and rollback are implemented.
- [ ] Step growth/reduction and minimum-step tests pass.
- [ ] Stage history contains every required diagnostic.
- [ ] Target continuation control reaches final parameters.
- [ ] Gate 3A regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-17 complete and selects SF-18.
<!-- completion-checklist:end -->

## Advancement rule

SF-18 may create the smooth periodic random fields required by the first
physical continuation benchmarks.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-09T16:00Z | activation on `master=dd66507` (SF-16 closure merged via PR #28) | SF-17 activated after verifying `NEXT: SF-17`, SF-16 `done`, and checker `OK (29 increments, next=SF-17)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) **Warm-start solver entry (library extension required by this increment):** `solve_streamfunctions` currently zero-initializes `u1/u2` on every call (SF-13), but continuation is meaningless unless each stage starts from the last ACCEPTED state; a minimal, default-preserving option `PicardInitialState { zero_source (default), warm_start }` is added to `StreamfunctionSolverConfig`: `warm_start` skips the zero-init and the zero-source block solves, takes state 0 = caller-provided `fields` (mean-zero projected on entry), leaves entry-0 PCG records default-constructed (the documented SF-14 convention), and changes nothing else; `zero_source` remains bitwise-identical to SF-13..16. The spec sentence "Extend config/output only with eta and epsilon continuation fields" is read as governing the pipeline YAML/output surface; this library option is the minimum mechanism the In-scope controller requires. (2) **One reusable stage machine for both axes:** `ContinuationController` implements a single host-only, GPU-free stage stepper parameterized by {start, target, initial_step, min_step, max_step, backtrack_factor, growth_factor, easy_streak}, advancing monotonically with clamping `attempt = min(param + step, target)`. Eta runs it in linear space with the spec-locked values (initial 0.1, min 0.0125, max 0.25, halve on failure, grow 1.5 after two easy stages); epsilon runs the SAME machine in `-log10(epsilon)` space from `-log10(epsilon_start)` (default 2) to `-log10(epsilon_target)` (default 6; 8 configurable but not required) with initial step 1.0 (one decade — the no-failure schedule is exactly the spec's decades 1e-3..1e-6), min step 0.125, max step 1.0, halve on failure, grow 1.5 after two easy stages; the epsilon min/max step values are project choices recorded here (the spec fixes only "by decades" plus the shared adaptive-step/rollback requirement). (3) **Stage semantics:** a stage = one warm-started `solve_streamfunctions` call at the attempted parameter; accepted iff `status == converged`; `not_converged` (any exit reason) → restore fields bitwise from the accepted-state snapshot and halve the step; a rejected stage whose attempted step was already at the floor exits with a structured `step_floor_exhausted` failure after exactly that one floor attempt (mirrors the SF-15 omega-floor rule); `invalid_problem` → immediate structured failure (measured Darcy `v_rms` is state-independent, so step reduction cannot fix it); `std::invalid_argument` propagates (caller misuse). (4) **Baseline stage and leg ordering:** stage 0 solves at (eta_start=0 default, epsilon_start) from the SF-13 `zero_source` init and establishes the first accepted state (recorded with step 0; failure → structured `baseline_failed`, no accepted state); the full eta leg then runs at fixed epsilon_start; the epsilon leg runs only after eta reaches its target, at fixed final eta (spec item 3). The report carries the final (eta, epsilon) actually reached — a fixed-epsilon result is never claimed as the original-system solution. (5) **Easy stage / persistent step:** easy = accepted on its FIRST attempt (zero halvings for that target); after 2 consecutive easy stages the persistent step grows ×1.5 (capped) and the streak resets; any halving resets the streak; after an accepted stage the persistent step is the step that succeeded (mirrors SF-15 persistent-omega). (6) **Accepted-state buffer ownership:** the continuation driver owns the two snapshot device buffers (2·n reals) — NOT `StreamfunctionWorkspace` — so the SF-12 closed-form memory report and its tests stay untouched; snapshot bytes are reported explicitly in the continuation report. (7) **Stage record (spec item 4):** axis, parameter at stage start, attempted value, attempted step, accepted flag, stage failure reason, solver `exit_reason`, `picard_iterations`, `final_omega`, final `r_F`/`r1`/`r2`, degeneracy evidence (threshold-0 unexplained fraction and the |c| 0.1% percentile), and the final block-solve iteration counts; append-only, one record per attempt. (8) **Deterministic failure injection:** the continuation driver accepts an optional stage-solve functor (production default = the real solver) so tests can inject deterministic failures — including scribbling on `fields` before failing — to verify step halving and bitwise state restore; a test seam, not a fallback path. (9) **Pipeline surface:** new strict YAML subsection `streamfunction_solver.continuation{enabled(false), eta{...}, epsilon{...}}`; when enabled, the existing top-level `eta` is the eta TARGET and top-level `epsilon` is the STARTING epsilon (no field silently ignored); exports gain `stage_history.csv` and summary continuation fields under the existing export switches; the disabled path must remain byte-identical to SF-16 (base-build comparison); the SF-16 fixture `config_streamfunctions_homogeneous.yaml` stays untouched — a new `apps/config_streamfunctions_continuation.yaml` is added. (10) **Deterministic controls and PRESPECIFIED thresholds (before any run, never tuned after):** library-level control = smooth trigonometric log-K (amplitude a=0.5, the accepted SF-14 research regime) on 32^3 with locked defaults, must reach eta=1 and epsilon=1e-6 with every accepted stage at `r_F <= 1e-6`; pipeline-level enabled control = homogeneous K=1 (sigma2=0), must reach eta=1 and epsilon=1e-6 with every stage 0 iterations and `r_F = 0.0` exact; stochastic heterogeneous fields remain SF-18+ scope. | Base commit is this activation commit on `master=dd66507`. Gate 1 + Gate 2 + Gate 3A apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
