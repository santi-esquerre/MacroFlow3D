# SF-06 — Affine-periodic right-hand sides

- State: `done`
- Goal: `Representar correctamente las partes afines y ensamblar sus lados derechos periódicos.`
- Depends on: `SF-05`
- Unlocks: `SF-07`
- Branch: `science/lester-sf06-affine-periodic-rhs`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf06-affine-periodic-rhs`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-05T17:28Z`
- Completed: `2026-08-05T18:16Z`
- PR: `#14 https://github.com/santi-esquerre/MacroFlow3D/pull/14`
- Commit: `c3453ea`

## Scientific or engineering intent

Represent multi-valued affine streamfunctions without breaking periodic storage
or omitting the variable-coefficient affine forcing.

## Preconditions

- SF-05 validates the periodic `A(q)` linear path.

## In scope

- `AffineGauge`, periodic fluctuation semantics, and assembly of
  `div(q*gbar_i)` with the exact face flux convention of `A`.

## Out of scope

- Gradient/Hessian kernels, nonlinear sources, and full solver orchestration.

## Files and symbols

- Add initial types under `src/physics/streamfunctions/`.
- Add an affine RHS kernel/helper using the operator coefficient policy.
- Add CPU/GPU manufactured tests to the streamfunction test target.

## Implementation specification

1. Store `u1`, `u2` only; store affine gradients separately as three-component
   constants.
2. Default benchmark gradients to `(0,vbar,0)` and `(0,0,1)`.
3. Assemble opposite face flux differences with harmonic `q_f` and periodic
   neighbors; then project and report the raw mean.
4. Do not evaluate an affine scalar through wrapped coordinates.

## Expected numerical effect

The fluctuation equations remain periodic and compatible while representing
the correct total streamfunction gradients.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Affine RHS is zero to roundoff for constant `q`.
- Raw compatibility defect is at roundoff for periodic smooth `q`.
- Smooth manufactured variable-`q` RHS has L2 order at least 1.8.

## Regression surface

- Coefficient face rules, periodic indexing, units of `vbar`, and gauge policy.

## Failure and rollback policy

- Do not approximate the affine part by storing a discontinuous sawtooth field.
- Any mismatch with `A` must be fixed by sharing the coefficient/flux primitive,
  not by projecting away a large compatibility defect.

## Completion checklist

<!-- completion-checklist:start -->
- [x] Affine and periodic components have separate types/ownership.
- [x] RHS uses the exact `A` face coefficient convention.
- [x] Constant and smooth-q thresholds pass.
- [x] Raw and projected compatibility defects are reported.
- [x] Full regressions and human review pass.
- [x] Evidence, PR, and commit are recorded.
- [x] Dashboard marks SF-06 complete and selects SF-07.
<!-- completion-checklist:end -->

## Advancement rule

SF-07 may calculate total gradients from the accepted affine representation.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-05T17:28Z | active | Activated SF-06 after the required documentation preflight and created the exact persistent runtime Goal in the canonical branch/worktree. | Master preflight confirmed `master=origin/master=5c7f7217612012b22c12da28c6535cf9910ff2d0`; this branch is `science/lester-sf06-affine-periodic-rhs` at that base in `~/src/MacroFlow3D/.agents/worktrees/lester-sf06-affine-periodic-rhs`; the increment checker passed with `next=SF-06`; SF-05 is `done`; Goal is `Representar correctamente las partes afines y ensamblar sus lados derechos periódicos.` | Build and execute the SF-06 DAG for affine gauge ownership and periodic RHS assembly. |
| 2026-08-05T18:07Z | validation PASS; acceptance reserved | Independent T08 integrated the accepted linear chain by `git merge --ff-only science/lester-sf06-affine-tests`: `79803f2`, `c28d541`, `7b01381`, `ef823c9`, `6542944`, `76627b5`, producing `76627b590fe4b104759c5e7a5b1c782d445913cd`. The original T05 `ac289381` and replacement `7b01381` have identical `affine_gauge.cuh` blob `36b113b2e6f8d42eb0d021da8b2eb9c9e382102b`; no cherry-pick was needed. Audit found only SF-06 files/CMake/test registration changed; no PSPTA, MG, flow, config, or SF-07+ behavior changed. RHS uses the same shared harmonic face helper as all 18 `A` face uses, periodic xyz wrapping, isotropic `h`, separate affine gradients/fluctuation spans, raw means before `P`, projected means after `P`, preallocated workspace, alias/error checks, and one explicit host diagnostic synchronization. The CPU long-double oracle is test-local and boundary checked. Reportable test-quality risks retained: the `arithmetic` and `inverse_hk` mutants are numerically duplicate, and the error-contract helper accepts any `std::exception`; the affine oracle divides by `spacing.x`, which is valid only because the accepted fixture and production contract are explicitly isotropic. | Final serial evidence on local WSL: checker PASS; `cmake --preset wsl-debug` PASS; one `cmake --build build/wsl-debug -j` PASS (only pre-existing warnings in `prolong_3d.cu` and projected-PCG test); all eight SF-06 cases PASS individually; full `streamfunction_operator_tests` PASS; targeted CTest PASS (1/1); `run_operator_tests` PASS (8/8); full CTest PASS (2/2); PSPTA smoke PASS. SF-06 metrics: CPU and GPU smooth-q order `1.991189` (threshold `>=1.8`), GPU/CPU RMS `3.5380286e-15` and boundary `1.1768364e-14` (threshold `1e-12`), constant/raw/projected means `0`, compatibility raw/projected max `1.734723475976807e-18`, sawtooth boundary mutant `384.574654916`, hidden offset raw `0.01` and post-projection `6.39425682413e-18`. Hardware: RTX 3050 Laptop GPU (4 GiB, driver 610.43.03), i7-12650H, Linux 7.1.5. Smoke config `apps/config_pspta_small.yaml`: grid `64x32x32`, dx `5`, sigma2 `1`, lambda `50`, seed `42`, transport 500 particles/500 steps/dt 1; head residual `1.02e+01 -> 1.77e-13`, divergence min/max `-8.1175e-14/+8.2979e-14`, particle `active=387 exited=113`, nonzero/max failures `0/0`. Gate 1 and Gate 2 pass; Gate 3A/4/V100 N/A for this RHS-only increment (no coupled solver, physical reconstruction, tracking change, PETSc/SLEPc, or production benchmark). Historical invalid evidence is retained: T03 initial CTest/smoke occurred before final link and was discarded; T06 targeted CTest was Not Run after only `macroflow3d_lib`; T07 first lacked `<iostream>` and only adjusted its mutation threshold, while its target-only full CTest initially lacked `run_operator_tests`; its subsequent serial 2/2 recovery is superseded by this independent T08 validation. `git diff --check master...HEAD` PASS. | Human review and root-only acceptance decision; do not alter State/checklist/NEXT/PR fields. |
| 2026-08-05T18:11Z | validating; master audit PASS from `fe9f9c9` | Root inspected the diff and commits; rederived sign, units, and the affine RHS formula; checked periodicity, gauge, and compatibility; reviewed ownership/aliasing, allocations/synchronization, error paths, positive and mutant tests, and scope/gates. Root reran all 8/8 SF-06 cases, `run_operator_tests` 8/8, full CTest 2/2, the PSPTA smoke, and checker/diff checks. | Classification: PASS; code frozen. Exact metrics: smooth-q order `1.991189`; GPU/CPU RMS `3.5380286e-15`, boundary `1.1768364e-14`; raw/projected compatibility defects `1.734723475976807e-18`; constant `0`; sawtooth `384.574654916`; smoke head residual `1.02e1 -> 1.77e-13`, particles `active/exited=387/113`, failures `0`. Minor, non-blocking audit risks are retained: the exception helper is broad although source exception types were audited; the oracle is isotropic under the accepted contract; and `arithmetic q` equals inverse harmonic `K` algebraically, explaining the duplicate mutant. | Publish PR and await required human review; do not change `NEXT` or completion fields. |
| 2026-08-05T18:16Z | `c3453ea`, done | The user explicitly approved completion; PR [#14](https://github.com/santi-esquerre/MacroFlow3D/pull/14) merged at `2026-08-05T18:16:41Z` as `c3453ea8700dc642ed41fbd0548075adf1637104`, with remote head `43bda1e` and GitGuardian `SUCCESS` verified. | Gate 1+2 evidence is preserved: smooth-q order `1.991189`; GPU/CPU RMS `3.5380286e-15`, boundary `1.1768364e-14`; raw/projected compatibility defects `1.734723475976807e-18`; constant `0`; sawtooth `384.574654916`; and smoke head residual `1.02e1 -> 1.77e-13`, particles `active/exited=387/113`, failures `0`. Retained minor risks are the broad exception helper, the isotropic accepted-contract oracle, and the algebraically duplicate `arithmetic q`/inverse-harmonic-`K` mutant. Post-merge-only publication metadata commits `c153108`/`e216875` were pushed after merge and were not included by GitHub; this authoritative closeout from master supersedes them. | SF-06 is complete; `NEXT` is SF-07, which remains pending and unactivated. |
| 2026-08-05T18:29Z | `74b1854`, closeout revalidation `SF-06-T14` | T13 evidence is invalid and excluded: root observed overlapping `cmake --build build/wsl-debug -j` PGIDs `824760`, `825684`, `826544` and `ninja -C` PGID `827234`, interrupted T13, and terminated only those groups; T14 first confirmed no `cmake`/`ninja`/`nvcc` process referenced this worktree, then configured `build/wsl-debug`, ran `cmake --build build/wsl-debug --target clean` (56 files), and completed exactly one monitored `cmake --build build/wsl-debug -j` (100 steps, exit 0). PR #14 independently reports `MERGED` into `c3453ea8700dc642ed41fbd0548075adf1637104` at `2026-08-05T18:16:41Z`, head `43bda1e`, GitGuardian `SUCCESS`. | Checker passed before/after (`29 increments, next=SF-07`); all eight SF-06 cases passed individually; full `streamfunction_operator_tests` passed; `run_operator_tests` `8/8`; targeted CTest `1/1`; full CTest `2/2`; and `apps/config_pspta_small.yaml` smoke passed. Gate 1+2 metrics: CPU/GPU smooth-q order `1.991189 >=1.8`; GPU/CPU RMS `3.5380286e-15`, boundary `1.1768364e-14 <=1e-12`; constant/raw/projected means `0`; compatibility raw/projected maximum `1.734723475976807e-18`; sawtooth boundary `384.574654916`; hidden offset raw/post-projection `0.01`/`6.39425682413e-18`. Smoke config: `64x32x32`, `dx=5`, `sigma2=1`, `lambda=50`, seed `42`, `500` particles/steps, `dt=1`; head 10 iterations `1.02e+01 -> 1.77e-13`, divergence `-8.1175e-14/+8.2979e-14`, `active/exited=387/113`, `nonzero/max failures=0/0`. `git diff --check master...HEAD` passed. Hardware/build: RTX 3050 Laptop GPU 4096 MiB, driver `610.43.03`; CUDA `13.3.73`; GCC `16.1.1`; Linux `7.1.5-1-cachyos`; Debug `wsl-debug`. Gate 3A/4/V100 are N/A to this RHS-only increment. Retained risks: broad exception helper, isotropic accepted-contract oracle, duplicate arithmetic/inverse-harmonic-K mutant; known pre-existing debug warnings in third-party/unchanged files. | Root-only acceptance decision; do not alter state/checklist/dashboard/NEXT or start SF-07. |
