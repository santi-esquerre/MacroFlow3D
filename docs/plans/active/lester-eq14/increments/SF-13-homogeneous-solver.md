# SF-13 — Homogeneous solver

- State: `done`
- Goal: `Resolver de extremo a extremo el caso homogéneo exacto.`
- Depends on: `SF-12`
- Unlocks: `SF-14`
- Branch: `science/lester-sf13-homogeneous-solver`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-08T21:23Z`
- Completed: `2026-08-08 (explicit owner approval of PR #25; closure metadata commit on the same PR)`
- PR: [#25 — SF-13: homogeneous end-to-end streamfunction solve (zero-source linear path + exact controls)](https://github.com/santi-esquerre/MacroFlow3D/pull/25)
- Commit: `c946bcdcf84e96ebf416175737ccf74719890e39 (frozen audited source head; later branch commits are increment-state documentation only)`

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
- [x] Homogeneous end-to-end path is implemented.
- [x] `16^3`, `32^3`, and `64^3` exact controls pass.
- [x] Gauge and repeated-workspace tests pass.
- [x] Gate 3A report contains all applicable metrics.
- [x] Full regressions and human review pass.
- [x] Evidence, PR, and commit are recorded.
- [x] Dashboard marks SF-13 complete and selects SF-14.
<!-- completion-checklist:end -->

## Advancement rule

SF-14 may add fixed-relaxation Picard after the exact control is merged.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-08T21:23Z | activation on `master=1bc62f9` (SF-12 closure merged via PR #24) | SF-13 activated after verifying `NEXT: SF-13`, SF-12 `done`, and checker `OK (29 increments, next=SF-13)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) the SF-13 `solve_streamfunctions` implements the **zero-source/harmonic linear path only** (assemble+project affine RHSs, sequentially solve both blocks with projected PCG + the shared MG hierarchy, project gauge, then evaluate the SF-10 coupled residual and every SF-11 diagnostic); `status = converged` means both linear block solves converged — the report's `r_F` is the honest coupled nonlinear metric and Picard iteration remains SF-14's deliverable. (2) `v_rms` is measured solver state per the SF-12 config contract: the solve runs SF-11 physical diagnostics first and threads the measured `v_d_rms` into `NonlinearSourceConfig::v_rms` and the histogram reference; a non-finite or non-positive measured `v_rms` yields `status = invalid_problem` (the SF-09 source contract and the `r1` normalization require strictly positive `v_rms`). (3) The solver zero-initializes `u1`/`u2` on every call for a deterministic exact control; warm-start policy belongs to SF-14. (4) Host-detectable misuse keeps throwing `std::invalid_argument` through `validate_streamfunction_problem` (SF-12 error contract); the `invalid_problem` status covers runtime-measured conditions and defensive mapping of non-converged PCG statuses. (5) `q` is computed in-solver by a pointwise kernel from `K` or `Y = ln K` per `ConductivityRepresentation` (SF-12 deferred exactly this to SF-13); device-content finiteness/positivity of `K`/`Y` remains a kernel-side precondition per the accepted SF-06 wording. (6) The spec's "dedicated exact-control executable or CTest case" is satisfied by a new homogeneous case registry inside the existing `streamfunction_operator_tests` runner plus a new `add_test(NAME streamfunction_homogeneous ...)` CTest entry selecting those cases, so `ctest -R streamfunction_homogeneous` runs exactly the exact controls. (7) "Normalized units": the controls use the unit torus `[0,1]^3` (`dx=1/N`) with benchmark gauge `vbar=1`, so `v_rms=1` and `L_ref=1` and the raw RMS values coincide with their normalized forms; this is stated per grid in the evidence. | Base commit is this activation commit on `master=1bc62f9`. Gate 1 + Gate 2 + Gate 3A apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. Memory decision: option (a) is in effect (67.58-field footprint accepted for bring-up; 4 GiB local GPU caps full-workspace runs at `128^3`, so `16^3/32^3/64^3` device controls are comfortable). | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
| 2026-08-08T21:55Z | `c946bcd`, integration validation | Two-node DAG completed and orchestrator-audited node by node: T01 `3ede66c` (first `solve_streamfunctions` definition in `StreamfunctionSolver.cu` — v1 zero-source/harmonic linear path: validate/prepare, pointwise `q=1/K` or `exp(-Y)` fill with no floor, zero-init `u1/u2`, coefficient hierarchy, projected affine RHS pair, two sequential projected-PCG block solves sharing the single MG hierarchy, SF-11 diagnostics measuring `v_rms=v_d_rms` with `invalid_problem` on non-finite/non-positive measurement, SF-10 coupled residual with the measured `v_rms`, memory report, `converged` = both linear blocks; truthful header rewrite of the SF-12 scope-lock stating loudly that `r_F` is the honest coupled nonlinear metric and Picard is SF-14); T02 `c946bcd` (four GPU exact-control cases `homogeneous_solver_exact_{16,32,64}` + `homogeneous_solver_repeated_reuse` with 16 hard non-relaxed 1e-13 checks per grid, `S1/S2` evaluated test-side via the accepted SF-07/08/09 primitives, SF-03 gauge bound, and the new CTest entry `streamfunction_homogeneous`). Zero blocking/major/minor audit findings; no corrective cycle. Single integrator verified the linear chain (merge-base == base, exactly 6 files +654/−15, `diff --check` clean, only the two StreamfunctionSolver files under `src/**`, docs/apps/particles/multigrid/numerics untouched) and reran the full suite green; final integrated commit is `c946bcd` with no integration commit needed. | Acceptance evidence (worker + integrator + orchestrator reruns all agreeing): every exact-control metric is LITERAL `0.0` at all three grids — `r_F=r1=r2=0`, `RMS(u1)=RMS(u2)=0`, test-side `RMS(S1)=RMS(S2)=0`, `e_v=0`, `e_div=0`, `invariance_e_psi1/2=0`, gauge means `0.0` (SF-03 bound met), `|c|` min=max=mean=1, `v_rms(measured)=1` (`32^3`: `0.99999999999999989`, roundoff within 1e-13), `L_ref=1` so raw == normalized; both PCG blocks `converged` with 0 iterations (exactly zero projected RHS); repeated-reuse: 3 calls with stable metrics and exact byte/pointer/`cudaMemGetInfo` identity (`17762195` B at `32^3`). Full suite: ctest 3/3 (incl. new `streamfunction_homogeneous`), streamfunction runner 93/93 PASS, `run_operator_tests` 8/8, PSPTA smoke OK, checker OK. Hardware: RTX 3050 Laptop 4 GiB, Debug sm_86, sccache launchers disabled. | Orchestrator FINAL_AUDIT on the control checkout, then publish PR as `awaiting_review`. |
| 2026-08-08T22:05Z | `c946bcd`, final audit PASS | Orchestrator personally re-audited the integrated head against the original spec on the control checkout: fresh reconfigure/build, ctest 3/3, 93/93 case verdicts, 8/8 operator tests, smoke, checker all green; every spec acceptance threshold has explicit evidence (all observed as exact zeros, thresholds kept exactly as specified — neither tightened nor relaxed); the rollback policy ("any nonzero systematic source or velocity error blocks nonlinear work") is NOT triggered. Gate 1 + Gate 2 + Gate 3A PASS (r_F, e_v, invariances, e_div, |c| stats, explicit epsilon=1e-2 with the accepted SF-09 `(eps*v_rms)^2` denominator as the only floor, gauge evidence, 16/32/64 grid progression); Gate 4/5 and V100 N/A (homogeneous exact control, no transport claim). | Flagged for the human reviewer: (1) the seven activation interpretive decisions, especially the v1 zero-source `converged` semantics, measured-`v_rms` threading, and per-call zero-initialization; (2) exact-zero observations make the 1e-13 spec gates generous; (3) mandatory-review path `src/physics/streamfunctions/`. Frozen audited source head: `c946bcd`. | Publish PR as `awaiting_review`; do not advance `NEXT`; await explicit human approval. |
| 2026-08-08T22:12Z | `c2a4b0e` published, PR #25 open | Delivery branch pushed and [PR #25](https://github.com/santi-esquerre/MacroFlow3D/pull/25) opened as `awaiting_review` with the frozen audited source head `c946bcd` (later commits on the branch are increment-state documentation only). | PR description carries the DAG, audit summaries, full Gate 3A acceptance evidence (all exact-control metrics literal `0.0` at `16^3/32^3/64^3`, repeated-reuse storage identity), the seven interpretive decisions, and remaining risks. No agent merges; `NEXT` remains `SF-13`. | Await explicit human review/approval of PR #25; on approval, add only the closure metadata commit (`done`, checklist, `NEXT: SF-14`) on this same PR. |
| 2026-08-08T23:03Z | PR #25 head `8730173`, human approval | The repository owner explicitly approved PR #25 with the instruction "Apruebo la PR #25, hacé el cierre". No GitHub review object exists (`reviews=0`); the approval fact is this recorded instruction. Verified before closure: PR #25 `OPEN` at head `8730173` — exactly the published state; frozen audited source head `c946bcd` unchanged (later commits are increment-state documentation only), so the approval applies to the audited content. | The approval covers the reviewed items flagged at publication: the seven activation interpretive decisions (v1 zero-source `converged` semantics, measured-`v_rms` threading, per-call zero-initialization, error-contract split, in-solver `q` computation, CTest registration, normalized-unit convention), the exact-zero Gate 3A evidence, and the mandatory-review path `src/physics/streamfunctions/`. | Closure metadata commit on this PR: set `done`, complete checklist, advance `NEXT` to `SF-14`. |
| 2026-08-08T23:03Z | closure metadata commit | SF-13 set `done`; checklist completed 7/7; dashboard updated (`SF-13` checked, `Last completed increment: SF-13`, `NEXT: SF-14`, active goal `none`); checker rerun. The new `NEXT: SF-14` exists only on this PR branch until a human merges it and does not authorize work ahead of the default branch. | Metadata/documentation-only diff (increment spec + dashboard); frozen audited source head remains `c946bcd`. | Human merges PR #25; SF-14 (fixed-relaxation Picard) may activate only after this closure state is visible on `master`. |
