# SF-13 — Homogeneous solver

- State: `active`
- Goal: `Resolver de extremo a extremo el caso homogéneo exacto.`
- Depends on: `SF-12`
- Unlocks: `SF-14`
- Branch: `science/lester-sf13-homogeneous-solver`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-08T21:23Z`
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
| 2026-08-08T21:23Z | activation on `master=1bc62f9` (SF-12 closure merged via PR #24) | SF-13 activated after verifying `NEXT: SF-13`, SF-12 `done`, and checker `OK (29 increments, next=SF-13)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) the SF-13 `solve_streamfunctions` implements the **zero-source/harmonic linear path only** (assemble+project affine RHSs, sequentially solve both blocks with projected PCG + the shared MG hierarchy, project gauge, then evaluate the SF-10 coupled residual and every SF-11 diagnostic); `status = converged` means both linear block solves converged — the report's `r_F` is the honest coupled nonlinear metric and Picard iteration remains SF-14's deliverable. (2) `v_rms` is measured solver state per the SF-12 config contract: the solve runs SF-11 physical diagnostics first and threads the measured `v_d_rms` into `NonlinearSourceConfig::v_rms` and the histogram reference; a non-finite or non-positive measured `v_rms` yields `status = invalid_problem` (the SF-09 source contract and the `r1` normalization require strictly positive `v_rms`). (3) The solver zero-initializes `u1`/`u2` on every call for a deterministic exact control; warm-start policy belongs to SF-14. (4) Host-detectable misuse keeps throwing `std::invalid_argument` through `validate_streamfunction_problem` (SF-12 error contract); the `invalid_problem` status covers runtime-measured conditions and defensive mapping of non-converged PCG statuses. (5) `q` is computed in-solver by a pointwise kernel from `K` or `Y = ln K` per `ConductivityRepresentation` (SF-12 deferred exactly this to SF-13); device-content finiteness/positivity of `K`/`Y` remains a kernel-side precondition per the accepted SF-06 wording. (6) The spec's "dedicated exact-control executable or CTest case" is satisfied by a new homogeneous case registry inside the existing `streamfunction_operator_tests` runner plus a new `add_test(NAME streamfunction_homogeneous ...)` CTest entry selecting those cases, so `ctest -R streamfunction_homogeneous` runs exactly the exact controls. (7) "Normalized units": the controls use the unit torus `[0,1]^3` (`dx=1/N`) with benchmark gauge `vbar=1`, so `v_rms=1` and `L_ref=1` and the raw RMS values coincide with their normalized forms; this is stated per grid in the evidence. | Base commit is this activation commit on `master=1bc62f9`. Gate 1 + Gate 2 + Gate 3A apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. Memory decision: option (a) is in effect (67.58-field footprint accepted for bring-up; 4 GiB local GPU caps full-workspace runs at `128^3`, so `16^3/32^3/64^3` device controls are comfortable). | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
