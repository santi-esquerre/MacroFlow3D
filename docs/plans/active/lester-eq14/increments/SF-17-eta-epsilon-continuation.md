# SF-17 — Eta and epsilon continuation

- State: `awaiting_review`
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
- PR: [#29 — SF-17: adaptive eta/epsilon continuation with bitwise rollback (warm-started homotopy over the accepted Picard solver)](https://github.com/santi-esquerre/MacroFlow3D/pull/29)
- Commit: `d3aa9886a00c0146c8f96a1424ccf95e31b96085 (frozen audited source head)`

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
- [x] Eta stepping, epsilon stepping, and rollback are implemented.
- [x] Step growth/reduction and minimum-step tests pass.
- [x] Stage history contains every required diagnostic.
- [x] Target continuation control reaches final parameters.
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
| 2026-08-09T16:55Z | `d3aa988`, integration validation | Four-node DAG (T01 library, corrective C01, T02 tests, T03 pipeline) completed and orchestrator-audited node by node. T01 `f99f351`: default-preserving `PicardInitialState{zero_source, warm_start}` solver entry (the zero_source path executes the identical statements; the moved enqueue order touches disjoint buffers on the single stream — bitwise-equivalent, suite-verified) + `ContinuationController.hpp/.cu`: one reusable host stage stepper for both axes (eta linear with the spec-locked 0.1/0.0125/0.25/halve/1.5-after-2-easy numbers; epsilon in `-log10` space, decades on the no-failure path), driver with driver-owned snapshot buffers, bitwise rollback BEFORE every retry/exit decision, one-floor-attempt exit, append-only stage records, structured statuses, and the documented test-only StageSolveFn seam. **Audit T01-F1 (MINOR, doc-only): the `PicardInitialState` comment misattributed the init PCG results to `picard_history[0]`; corrective C01 `367caab` fixed the sentence.** T02 `2719ec7`: 12-case `streamfunction_continuation` ctest entry — rule-exact host stepper oracles (independent reimplementation, not circular), warm-start semantics, injected-failure bitwise restore (verified at the halved-retry entry AND at exit), floor exhaustion (exact 0.1/0.05/0.025/0.0125 sequence, one floor attempt), invalid_problem propagation, and the two prespecified controls. T03 `d3aa988`: strict `streamfunction_solver.continuation` YAML (decision-9 coupling: top-level eta = target, epsilon = start), continuation branch in the runner (disabled path byte-identical), `stage_history.csv` + summary continuation object, new fixture `apps/config_streamfunctions_continuation.yaml`. Single integrator verified the linear chain (18 files +2658/−30; cherry-picked patches byte-identical to the approved originals); zero conflicts; no integration commit. | Acceptance evidence: **trig a=0.5 32^3 library control reached eta=1 and epsilon=1e-6** (baseline r_F=1.4e-14; eta stages 17–21 adaptive iterations each, all first-attempt, every accepted stage r_F<=1e-6; epsilon stages 8/0/0/0 iterations; snapshot 524288 B = 2·32^3·8) — the one-shot a=0.5 case at (eta=1, eps=1e-2) was a non-gating SF-14 research case, while the staged warm-started homotopy reaches the (1, 1e-6) target: exactly the increment's intent. Homogeneous pipeline control: 12 stages, all accepted, 0 iterations, r_F=0.0 exact, eta schedule and epsilon decades exact. Injected failures: step halved exactly and state restored bitwise; floor exhaustion returns `step_floor_exhausted` with the accepted state intact. Full ctest 6/6; config rejections path-qualified. Hardware: RTX 3050 4 GiB, Debug sm_86, sccache launchers disabled. | Orchestrator FINAL_AUDIT on the control checkout. |
| 2026-08-09T17:05Z | `d3aa988`, final audit PASS | Orchestrator personally re-audited the integrated head on the control checkout: fresh configure/build; full ctest 6/6 (696 s); disabled-path byte-identity re-verified against the orchestrator's OWN base build (exact 4abbf64 refs): pspta_small and the SF-16 homogeneous fixture have IDENTICAL stdout and byte-identical artifacts except the documented resolved-continuation effective-config section and manifest run identity; continuation fixture artifacts inspected row by row (12 stages, exact schedules, exact zeros); checker OK with `NEXT: SF-17` unchanged. Gate 1 + Gate 2 + Gate 3A PASS (staged, explicit, logged epsilon regularization; per-stage degeneracy evidence; gauge maintained; controls on 16^3/32^3). | Flagged for the human reviewer: (1) the ten activation interpretive decisions — esp. the warm-start solver entry (the only solver change; default proven inert), the epsilon-axis min/max log10 steps 0.125/1.0 (project choices beyond the spec's "by decades"), the easy-stage definition, and the decision-9 YAML coupling; (2) the StageSolveFn test-only seam in the driver signature; (3) mandatory-review paths src/physics/streamfunctions/, src/io/, src/runtime/; (4) local full-suite runtime now ~12 min. Frozen audited source head: `d3aa988`. | Publish PR as `awaiting_review`; do not advance `NEXT`; await explicit human approval. |
| 2026-08-09T17:20Z | `8f60789` published, PR #29 open | Delivery branch pushed and [PR #29](https://github.com/santi-esquerre/MacroFlow3D/pull/29) opened as `awaiting_review` with the frozen audited source head `d3aa988` (later commits are increment-state documentation only). | PR description carries the DAG, the C01 corrective cycle, the ten interpretive decisions, full acceptance evidence (trig and homogeneous controls, bitwise rollback, floor exhaustion, disabled-path byte-identity), and the reviewer flags. No agent merges; `NEXT` remains `SF-17`. | Await explicit human review/approval of PR #29; on approval, add only the closure metadata commit (`done`, checklist, `NEXT: SF-18`) on this same PR. |
