# SF-04 — Projected PCG

- State: `active`
- Goal: `Resolver el operador periódico singular mediante PCG proyectado.`
- Depends on: `SF-03`
- Unlocks: `SF-05`
- Branch: `science/lester-sf04-projected-pcg`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf04-projected-pcg`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-05T12:20Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Solve the singular periodic diffusion equation in its valid zero-mean quotient
space while preserving the existing nonprojected PCG API.

## Preconditions

- SF-02 defines `A`; SF-03 provides the projector and workspace.

## In scope

- A projected PCG policy/overload and tests with identity preconditioning.

## Out of scope

- Multigrid preconditioning and nonlinear solves.

## Files and symbols

- Extend `src/numerics/solvers/pcg.cuh` through an optional projector policy.
- Reuse existing `r`, `z`, `p`, and `Ap` workspace ownership.

## Implementation specification

1. Project raw RHS and log its preprojection compatibility defect.
2. Project the initial guess, residual, preconditioned residual, and search
   direction at the mathematically required points.
3. Report true projected residual and final field mean.
4. Preserve old PCG behavior when no projector is supplied.

## Expected numerical effect

Compatible periodic systems converge to the unique zero-mean representative;
incompatible constant RHS components are explicitly diagnosed and removed.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Projected relative residual is at most `1e-10`.
- Final gauge meets the SF-03 mean threshold.
- Solution agrees with the CPU/Fourier manufactured result within discretization
  error.

## Regression surface

- Current flow PCG iteration counts, sign wrappers, convergence reports, and
  host-device synchronization.

## Failure and rollback policy

- Do not replace current PCG or change default semantics.
- If the projected recurrence loses conjugacy, retain the simplest correct
  projection frequency and log synchronization cost for SF-23.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Optional projected PCG path is implemented.
- [ ] Raw RHS compatibility and final gauge are reported.
- [ ] Manufactured periodic solves meet tolerance.
- [ ] Existing PCG callers and tests remain unchanged.
- [ ] Human review and evidence are recorded.
- [ ] Dashboard marks SF-04 complete and selects SF-05.
<!-- completion-checklist:end -->

## Advancement rule

SF-05 may add the multigrid preconditioner to this accepted projected solve.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-05T12:20Z | `c6a66e7`, active | Verified the SF-03 implementation and procedural closeout on the default branch, completed its runtime Goal, created the exact SF-04 Goal, and created the canonical SF-04 worktree. | `master=origin/master=c6a66e7`; increment checker passed with `next=SF-04`; SF-03 is `done` with 7/7 checklist after PR #8 and closeout PR #9; no other Lester Goal is unfinished. | Establish the existing PCG/default-caller contract, projected recurrence points, diagnostics, and manufactured-test DAG before implementation. |
| 2026-08-05T12:31Z | `76435cf`, design audit | SF-04-T01 established the existing seven-argument `pcg_solve` as a frozen compatibility path and identified its sole production caller, synchronization pattern, workspace ownership, and missing breakdown/shape guards. | The flow path still pairs the negative diffusion operator and negative MG preconditioner; its small smoke baseline remains 10 iterations and residual `1.02e+01 -> 1.77e-13`. The projected path will use a separate result and entry point so legacy semantics remain unchanged. | Freeze the projected API and implement it without modifying the legacy body. |
| 2026-08-05T12:31Z | `76435cf`, oracle design | SF-04-T03 fixed independent CPU/Fourier manufactured oracles and nine positive/negative controls for compatibility, gauge, true projected residual, RHS immutability, legacy opt-in behavior, and mutation sensitivity. | Proposed fixed thresholds are projected relative residual `<= 1e-10`, SF-03 gauge, solution RMS error `<= 5e-9`, and raw constant defect recovery within `1e-12`; RHS generation must not call the GPU operator under test. | Implement the frozen solver contract, then add the independent test registry and run each control. |
| 2026-08-05T12:31Z | `76435cf`, failed design attempt | The first SF-04-T02 agent did not return an API report after two explicit closure requests and was interrupted without edits. | No repository state changed. The task was narrowed to exact signatures, recurrence points, diagnostics, ownership, breakdowns, and synchronization, then reassigned as SF-04-T02-R1. | Accept or correct the replacement API report before starting implementation. |
| 2026-08-05T12:34Z | `8fa7a1b`, API frozen | SF-04-T02-R1 defined a separate `projected_pcg_solve`, configuration, status-rich result, and caller-prepared workspace while leaving the seven-argument legacy solver untouched. | `b` remains immutable in a dedicated projected copy; raw compatibility is reported as `mean(b)` and `|mean(b)|/RMS(b)`; `b_hat,x,r,z,p` are projected; convergence uses the recomputed true residual `||P(b_hat-Ax)||_2` relative to its initial value, with explicit zero handling. All allocations precede the solve loop, projector operations add no host synchronization, and SPD/nonfinite failures are explicit. | Implement SF-04-T04 on an isolated branch and validate source-level legacy preservation before adding tests. |
| 2026-08-05T12:46Z | `aaae27a`, implementation accepted | SF-04-T04 added the isolated `projected_pcg_solve` path, explicit projector policy, prepared workspace, immutable projected RHS copy, projected recurrence, true-residual checks, final gauge, and distinct failure states in `pcg.cuh` only. | Root review corrected iteration-zero tolerance handling, the valid `max_iter=0` path, the final-iteration stop before an unused preconditioner, and the explicit policy argument before commit. The first serial build was externally interrupted and discarded; a fresh configure, serial build, checker, and CTest 2/2 then passed. The extracted legacy solver block is identical to `174878a`. | Instantiate the template through independent CPU/Fourier manufactured and contract controls in SF-04-T05. |
| 2026-08-05T13:09Z | `5bdd45a` + `6a42b02`, manufactured controls accepted | SF-04-T05A registered constant- and smooth-coefficient `17^3` projected-PCG solves with identity preconditioning, a long-double periodic CPU stencil, and a discrete Fourier oracle for `q=1`. | Constant/smooth solves converged in 5/35 iterations with reported relative residuals `5.67e-13`/`2.27e-13`, CPU solution RMS errors `1.01e-14`/`2.18e-15`, and final means at `O(1e-17)` or below. RHS copies were bit-identical and CTest 2/2 passed. Root audit rejected an unsigned diagnostic comparison; C01 now compares reported and CPU means with sign and its two cases plus targeted CTest pass. Several initial build invocations overlapped, were discarded, and one clean serial build supplied the accepted evidence. | Add incompatible-RHS, nonzero-gauge, and legacy opt-in contracts serially on the accepted fixture. |
