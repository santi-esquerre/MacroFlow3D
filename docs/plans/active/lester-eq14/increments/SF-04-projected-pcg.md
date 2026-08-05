# SF-04 — Projected PCG

- State: `done`
- Goal: `Resolver el operador periódico singular mediante PCG proyectado.`
- Depends on: `SF-03`
- Unlocks: `SF-05`
- Branch: `science/lester-sf04-projected-pcg`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf04-projected-pcg`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-05T12:20Z`
- Completed: `2026-08-05T14:28Z`
- PR: `#10`
- Commit: `ef324c6`

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
- [x] Optional projected PCG path is implemented.
- [x] Raw RHS compatibility and final gauge are reported.
- [x] Manufactured periodic solves meet tolerance.
- [x] Existing PCG callers and tests remain unchanged.
- [x] Human review and evidence are recorded.
- [x] Dashboard marks SF-04 complete and selects SF-05.
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
| 2026-08-05T13:21Z | `f1316e1`, functional contracts accepted | SF-04-T05B added incompatible-RHS, nonzero-initial-gauge, and exact legacy-seven-argument opt-in controls without changing production code. | The `0.375` raw offset was recovered exactly with normalized defect `1.0492e-02`, the projected and compatible solves both took 5 iterations and differed by `1.36e-16` RMS in the quotient, an initial mean `2.75` was reduced to `O(1e-18)` in 10 iterations, and legacy/projected compatible solutions differed by `1.73e-16` RMS. RHS immutability and independent CPU residual checks passed. An incomplete Ninja-log attempt was discarded; the recovered single serial target build, all five cases, full runner, and targeted CTest passed. | Add explicit status/breakdown controls and demonstrate that omitting RHS or initial-gauge projection is rejected. |
| 2026-08-05T13:32Z | `e92bf0b`, failure contracts accepted | SF-04-T05C covered invalid configuration, size mismatch, caller/workspace aliasing, `pAp` and `rz` breakdown, nonfinite input, zero-iteration exhaustion, and iteration-zero convergence with immutable sentinels and explicit statuses. It also added no-RHS-projection and no-`x0`-projection mutants. | All eight status subpaths matched their expected enum; `max_iter=0` and `rtol=1` made zero preconditioner calls. The unprojected incompatible legacy solve did not converge and had raw relative residual `1.91e16`, while the projected path reached `5.67e-13`. The unprojected-`x0` legacy solve converged in the quotient but retained mean `2.75`; the projected path reached mean `3.41e-18`. A single serial build was polled to completion after the wrapper returned early; eight cases, the full runner, and targeted CTest passed. | Complete mutation sensitivity for recursive-residual substitution, caller-RHS mutation, and pinned-gauge substitution. |
| 2026-08-05T13:44Z | `356c616`, mutation controls accepted | SF-04-T05D completed the test-local mutation sensitivity for false recursive residuals, in-place caller-RHS projection, and pinned-cell gauge substitution. | With `max_iter=0`, the solver and independent CPU oracle both reported true residual `2.505e+03` and relative residual `1`, rejecting a fake recursive report of zero. Deliberate in-place projection changed an incompatible caller RHS by RMS `0.375`; pinning cell zero by `6.20e-02` raised the relative CPU residual from `5.13e-20` to `4.64e-02`. Concurrent build wrappers interfered with the first target attempt; it was discarded, then one authorized fresh serial build, all ten SF-04 cases, the full runner, and targeted CTest passed. | Integrate the six implementation/test commits topologically on the canonical worktree and run independent Gate 1 + Gate 2 validation. |
| 2026-08-05T13:51Z | `3737f4a`, integration validation | SF-04-T06 cherry-picked the exact linear chain `aaae27a..356c616` as `b74f524,6ec7211,f6e948a,492ae71,bd2f3d2,3737f4a`; scope is only PCG, CMake/registry/runner, and `projected_pcg_gpu_cases.cu`. The frozen legacy `PCGConfig`/`PCGResult`/`PCGWorkspace`/`pcg_solve` block compares exactly with `174878a`; no duplicate legacy buffers/APIs or flow/MG/PSPTA behavior changes were found. | `diff --check` and increment checker passed; fresh Debug configure, one original serial `-j1` build (a second overlapping serial wrapper was cancelled/discarded), then incremental `-j` no-op succeeded. `--list` reported 36 cases; all 10 `projected_pcg_*`, full runner, targeted CTest `1/1`, full CTest `2/2`, and `run_operator_tests` `8/8` passed. Independent CPU/Fourier oracles confirm positive periodic `A`, compatibility/gauge, true residual, and solution: constant/smooth are 5/35 iter, relres `5.666e-13`/`2.266e-13`, CPU RMS solution error `1.011e-14`/`2.179e-15`; all status/sentinel controls and four mutation controls reject their mutants. PSPTA smoke reproduced baseline head `10`, `1.02e+01 -> 1.77e-13`, active/exited `387/113`, zero stalls/failures. Local GPU RTX 3050 Laptop (4 GiB), CUDA 13.3, config seed 42/grid `64x32x32`; projected fixtures are periodic `17^3`, `rtol=1e-12`. `/usr/bin/time` is unavailable; local sampled runner/pipeline maxima were `357832`/`3892` KiB with elapsed 2 s each. Gate 1 PASS; Gate 2 PASS; Gate 3A/Gate 4/V100 N/A because SF-04 does not implement coupled streamfunctions, alter transport/physics, or require remote validation. | Human review remains required; do not advance state/NEXT. Residual risks: host reductions/projector synchronization and allocation behavior need later SF-23 measurement; sampled RSS excludes GPU allocation. |
| 2026-08-05T13:55Z | `73b2877`, root audit REWORK | The independent integration passed, but the root audit found that the CPU/GPU true-residual comparison adds a roundoff floor scaled by `max(||b||,||Ax||,1)` rather than by the two residuals being compared. | For the constant manufactured solve, the accepted limit is `2.288e-09`, larger than the correct reported residual `1.419e-09`; a false reported residual of zero would therefore pass this oracle. Observed correct CPU/GPU residual differences are only `6.04e-16` (constant) and `4.91e-15` (smooth), so this is a mutation-sensitivity defect, not a solver failure. Classification: REWORK, Gate 2 not yet accepted by the root auditor. | SF-04-C01 must scale both tolerance terms with `max(|r_cpu|,|r_gpu|,1)`, prove the zero-report mutant is rejected at convergence, and rerun independent integration. |
| 2026-08-05T14:04Z | `69ff0d2`, corrective accepted | SF-04-C01 centralized the true-residual comparison as `(1e-11 + 4096 eps) max(|r_cpu|,|r_gpu|,1)` in every manufactured, incompatible, legacy, and recursive-residual contract, and added an explicit zero-report mutant at converged constant/smooth solves. | Correct CPU/GPU differences remain `6.04e-16`/`4.91e-15` against the tightened `1.091e-11` limit, while zero-report differences are `1.419e-09`/`7.180e-10` and are rejected. All ten cases and the full runner passed. The first full CTest lacked `run_operator_tests` because only the streamfunction target had been built; that missing target was built serially and CTest then passed 2/2. No solver, tolerance, state, or dashboard code changed. | Cherry-pick only `69ff0d2` through a new integrator, repeat Gate 1 + Gate 2, then repeat the root audit from the complete canonical diff. |
| 2026-08-05T14:09Z | `d64a09c`, reintegration validation | SF-04-T08 independently verified that `69ff0d2` is the single test-only correction from the canonical REWORK parent `485e213` (the intervening canonical `89110b7` changes only this bitácora), then cherry-picked it as `d64a09c`. The centralized helper is `(1e-11 + 4096 eps) max(|r_cpu|,|r_gpu|,1)` in the manufactured, incompatible-RHS, legacy-API, and recursive-residual contracts; no `b`/`Ax` residual scaling remains. | `diff --check` and checker passed; fresh `wsl-debug` configure, one complete serial build, and incremental `-j` no-op passed. `--list` found 36 cases; all ten `projected_pcg_*` cases, runner, `run_operator_tests` 8/8, focal CTest 1/1, full CTest 2/2, and PSPTA smoke passed. Constant/smooth: 5/35 iterations, CPU/GPU residual differences `6.035e-16`/`4.913e-15` within `1.091e-11`; zero-report mutants `1.419e-09`/`7.180e-10` were rejected and printed `true`. `pcg.cuh` is unchanged from pre-correction and its legacy block matches `174878a`; smoke preserved 10 head iterations, `1.02e+01 -> 1.77e-13`, active/exited `387/113`, zero stalls/failures. Gate 1 PASS; Gate 2 PASS; Gate 3A/Gate 4/V100 N/A. | Root audit and required human review; do not advance state, checklist, or NEXT. Residual risk remains later measurement of host reductions/projector synchronization and GPU memory. |
| 2026-08-05T14:12Z | `e2ca5aa`, root audit PASS / validating | The root auditor repeated the complete review from the final diff after C01: mathematical sign and quotient-space recurrence, periodic cell-centered gauge, raw compatibility, true residual, vector projection/ownership, alias rejection, loop allocation/synchronization surface, legacy preservation, positive and intentional-failure tests, scope, gates, and reproducibility all satisfy SF-04. | Personal rerun: incremental build no-op, full streamfunction runner `36/36`, `run_operator_tests` `8/8`, CTest `2/2`, and PSPTA smoke passed. Constant/smooth relres are `5.666e-13`/`2.266e-13`; final means are `5.79e-18`/`1.16e-17`; CPU solution errors are `1.01e-14`/`2.18e-15`; tightened residual limits are `1.091e-11`, and both converged zero-report mutants are rejected. Legacy head and transport baseline are unchanged. Classification: PASS; state moved to `validating` and code is frozen. | Publish the reviewed scope, move to `awaiting_review`, and request the required human review without merging or advancing NEXT. |
| 2026-08-05T14:16Z | `0335909`, PR #10 / awaiting review | Published the exact root-audited SF-04 branch and opened draft PR #10 with scope, validation commands, numerical evidence, remaining risks, and intentionally untouched files. | Remote branch `science/lester-sf04-projected-pcg` points at the frozen audit commit; GitHub PR #10 targets `master`. The code remains frozen, SF-04 remains the active Goal, and `NEXT` remains SF-04 until the required human review and merge are visible on the default branch. | Commit this state transition, push it to PR #10, mark the PR ready, and await explicit human review. |
| 2026-08-05T14:17Z | `b7cf82e`, PR #10 ready for review | Pushed the `awaiting_review` transition and converted PR #10 from draft to ready. | GitHub reports head `b7cf82e`, base `master`, state `OPEN`, and the GitGuardian check in progress. No human review decision is attached yet. | Await human review and the completed remote check; do not merge, complete the Goal, or advance `NEXT`. |
| 2026-08-05T14:28Z | `ef324c6`, done | Required human review was completed and PR #10 was merged to `master`; the user explicitly confirmed the approved merge. | GitHub records PR #10 as `MERGED` at `2026-08-05T14:28:26Z`, merge commit `ef324c62405e84d2b0465f9caa611efa6fcd582d`, with GitGuardian `SUCCESS`. Accepted Gate 1 + Gate 2 evidence remains: projected cases and full runner passed, `run_operator_tests` `8/8`, CTest `2/2`, and PSPTA smoke retained its 10-iteration baseline. Closeout checker is required to report `next=SF-05`; residual risk is limited to later SF-23 measurement of host reductions/projector synchronization and GPU memory, while sampled RSS excludes GPU allocation. | SF-04 completo; enable and preflight SF-05 only after this closeout is merged on the default branch. |
| 2026-08-05T14:40Z | `a46d002`, closeout integration PASS | SF-04 closeout was cherry-picked exactly from T10 after confirming direct parent `ef324c6`, two-doc scope, PR #10 `MERGED` at `2026-08-05T14:28:26Z`, merge `ef324c6`, GitGuardian `SUCCESS`, and no formal GitHub reviews (user approval is recorded above). | Checker passed with `next=SF-05`; fresh `wsl-debug` configure and one serial build completed (an earlier parallel wrapper returned before its build had finished, so that incomplete attempt was discarded), CTest passed `2/2`, and `config_pspta_small.yaml` passed on RTX 3050 Laptop 4 GiB/CUDA 13.3: seed `42`, grid `64x32x32`, head 10 iterations `1.02e+01 -> 1.77e-13`, active/exited `387/113`, zero stalls/failures. Gate 1 + Gate 2 remain retained; Gate 3A/Gate 4/V100 are N/A. `diff --check` is clean and only dashboard plus SF-04 differ from `master`; no new physical solve metrics apply beyond the reproduced smoke. | Root audit and publish closeout; residual risk remains later SF-23 measurement of host reductions/projector synchronization and GPU memory. |
