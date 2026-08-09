# SF-14 — Fixed-relaxation Picard

- State: `awaiting_review`
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
| 2026-08-09T00:55Z | `8accb99`, integration validation | Three-node DAG (T01 implementation, T02 tests, corrective C01) completed and orchestrator-audited node by node. T01 `8b034ec`: Picard loop in `solve_streamfunctions` (one residual enqueue per iteration = one immutable source state producing both projected block RHSs `G_i` in the SF-10 workspace, borrowed via a single additive read-only accessor — zero new device memory; sequential projected-PCG block solves sharing the MG stack; fixed `omega=0.25` pair relaxation + explicit mean-zero projection; `FixedPicardConfig{500,1e-6,0.25}` validated; per-iteration history; extended `converged` = `r_F <= tolerance`). T02 `ae81716`: five prespecified cases + CTest `streamfunction_picard_fixed` (worker-authored; committed verbatim by the orchestrator as administrative recovery after the worker session was interrupted mid-validation — runbook §13 — with validation re-run by the orchestrator). **The T02 gating run then FAILED honestly**, exposing T01-F2 (MAJOR): the warm-start block-solve initial guess makes PCG's RELATIVE convergence criterion unattainable near the fixed point (evidence: r_F contracted 1.04e-2→7.8e-6 over 32 healthy (10,10)-iteration solves, then update 33 stagnated at (1000,1000)). Corrective C01 `8accb99`: zero initial guess per block solve (fixed point unchanged; criterion well-scaled at every iteration), NaN sentinel for the linear-failure history record (T01-F1), sentinel-aware test finiteness. Single integrator verified the linear chain (merge-base == base, 10 files +1098/−53, `diff --check` clean, `src/**` limited to five streamfunctions files) and reran the full suite green; final commit `8accb99`, no integration commit. | Acceptance evidence (integrator + orchestrator reruns agreeing): prespecified gating case `16^3`, `Y=0.25·sin(2πx)sin(2πy)sin(2πz)`, `K=e^Y`: **converged in 40 Picard iterations**, r_F `1.036e-2 → 9.91e-7` (geometric ~0.775/iter), EVERY block solve converged at (10,10) PCG iterations with relative residual ≤ 1e-10; `32^3` same field: **41 iterations**, `9.19e-3 → 8.97e-7`; research case `a=0.5` at `32^3`: **CONVERGED in 51 iterations** (final r_F 8.89e-7, recorded as research data, never gated); homogeneous control preserved exactly (0 Picard iterations, r_F=0, bytes 17,762,195 and pointers stable); gauge means ~1e-21 (SF-03 bound); config error contract 11/11 incl. boundary positives; no amplitude/tolerance/omega changed after any result (rollback rule). Full suite: ctest 4/4, runner 98/98 PASS, `run_operator_tests` 8/8, smoke OK, checker OK. Hardware: RTX 3050 Laptop 4 GiB, Debug sm_86, sccache launchers disabled. | Orchestrator FINAL_AUDIT on the control checkout, then publish PR as `awaiting_review`. |
| 2026-08-09T01:05Z | `8accb99`, final audit PASS | Orchestrator personally re-audited the integrated head against the original spec on the control checkout: fresh reconfigure/build, ctest 4/4, 98/98 case verdicts, 8/8 operator tests, smoke, checker all green; all three spec acceptance thresholds have explicit evidence (r_F ≤ 1e-6 within budget on both prespecified gating grids; every linear solve ≤ 1e-10 relative; gauge + finiteness); the failure/rollback policy was honored — the mid-increment gating failure was root-caused (warm-start/relative-residual trap) and fixed at the defect (zero initial guess), with no adaptive behavior, no hidden damping, and no post-hoc tuning. Gate 1 + Gate 2 + Gate 3A PASS; Gate 4/5, V100 N/A. | Flagged for the human reviewer: (1) the eight activation interpretive decisions (esp. G-RHS borrowing, zero-init rationale, extended `converged` semantics); (2) the recorded corrective cycle T01-F2/T01-F1 and the T02 worker-interruption recovery; (3) the a=0.5 research outcome; (4) mandatory-review path `src/physics/streamfunctions/`. Frozen audited source head: `8accb99`. | Publish PR as `awaiting_review`; do not advance `NEXT`; await explicit human approval. |
| 2026-08-09T01:10Z | `7543d0f` published, PR #26 open | Delivery branch pushed and [PR #26](https://github.com/santi-esquerre/MacroFlow3D/pull/26) opened as `awaiting_review` with the frozen audited source head `8accb99` (later commits on the branch are increment-state documentation only). | PR description carries the DAG, the recorded corrective cycle (T01-F2 MAJOR warm-start/relative-residual defect caught by the honest gating failure and fixed by zero-init; T01-F1 NaN sentinel), the T02 worker-interruption recovery note, full Gate 3A acceptance evidence (40/41-iteration gating convergence with every linear solve ≤ 1e-10, a=0.5 research outcome), interpretive decisions, and remaining risks. No agent merges; `NEXT` remains `SF-14`. | Await explicit human review/approval of PR #26; on approval, add only the closure metadata commit (`done`, checklist, `NEXT: SF-15`) on this same PR. |
