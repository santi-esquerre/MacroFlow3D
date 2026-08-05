# SF-06 — Affine-periodic right-hand sides

- State: `active`
- Goal: `Representar correctamente las partes afines y ensamblar sus lados derechos periódicos.`
- Depends on: `SF-05`
- Unlocks: `SF-07`
- Branch: `science/lester-sf06-affine-periodic-rhs`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf06-affine-periodic-rhs`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-05T17:28Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

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
- [ ] Affine and periodic components have separate types/ownership.
- [ ] RHS uses the exact `A` face coefficient convention.
- [ ] Constant and smooth-q thresholds pass.
- [ ] Raw and projected compatibility defects are reported.
- [ ] Full regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-06 complete and selects SF-07.
<!-- completion-checklist:end -->

## Advancement rule

SF-07 may calculate total gradients from the accepted affine representation.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-05T17:28Z | active | Activated SF-06 after the required documentation preflight and created the exact persistent runtime Goal in the canonical branch/worktree. | Master preflight confirmed `master=origin/master=5c7f7217612012b22c12da28c6535cf9910ff2d0`; this branch is `science/lester-sf06-affine-periodic-rhs` at that base in `~/src/MacroFlow3D/.agents/worktrees/lester-sf06-affine-periodic-rhs`; the increment checker passed with `next=SF-06`; SF-05 is `done`; Goal is `Representar correctamente las partes afines y ensamblar sus lados derechos periódicos.` | Build and execute the SF-06 DAG for affine gauge ownership and periodic RHS assembly. |
| 2026-08-05T18:07Z | validation PASS; acceptance reserved | Independent T08 integrated the accepted linear chain by `git merge --ff-only science/lester-sf06-affine-tests`: `79803f2`, `c28d541`, `7b01381`, `ef823c9`, `6542944`, `76627b5`, producing `76627b590fe4b104759c5e7a5b1c782d445913cd`. The original T05 `ac289381` and replacement `7b01381` have identical `affine_gauge.cuh` blob `36b113b2e6f8d42eb0d021da8b2eb9c9e382102b`; no cherry-pick was needed. Audit found only SF-06 files/CMake/test registration changed; no PSPTA, MG, flow, config, or SF-07+ behavior changed. RHS uses the same shared harmonic face helper as all 18 `A` face uses, periodic xyz wrapping, isotropic `h`, separate affine gradients/fluctuation spans, raw means before `P`, projected means after `P`, preallocated workspace, alias/error checks, and one explicit host diagnostic synchronization. The CPU long-double oracle is test-local and boundary checked. Reportable test-quality risks retained: the `arithmetic` and `inverse_hk` mutants are numerically duplicate, and the error-contract helper accepts any `std::exception`; the affine oracle divides by `spacing.x`, which is valid only because the accepted fixture and production contract are explicitly isotropic. | Final serial evidence on local WSL: checker PASS; `cmake --preset wsl-debug` PASS; one `cmake --build build/wsl-debug -j` PASS (only pre-existing warnings in `prolong_3d.cu` and projected-PCG test); all eight SF-06 cases PASS individually; full `streamfunction_operator_tests` PASS; targeted CTest PASS (1/1); `run_operator_tests` PASS (8/8); full CTest PASS (2/2); PSPTA smoke PASS. SF-06 metrics: CPU and GPU smooth-q order `1.991189` (threshold `>=1.8`), GPU/CPU RMS `3.5380286e-15` and boundary `1.1768364e-14` (threshold `1e-12`), constant/raw/projected means `0`, compatibility raw/projected max `1.734723475976807e-18`, sawtooth boundary mutant `384.574654916`, hidden offset raw `0.01` and post-projection `6.39425682413e-18`. Hardware: RTX 3050 Laptop GPU (4 GiB, driver 610.43.03), i7-12650H, Linux 7.1.5. Smoke config `apps/config_pspta_small.yaml`: grid `64x32x32`, dx `5`, sigma2 `1`, lambda `50`, seed `42`, transport 500 particles/500 steps/dt 1; head residual `1.02e+01 -> 1.77e-13`, divergence min/max `-8.1175e-14/+8.2979e-14`, particle `active=387 exited=113`, nonzero/max failures `0/0`. Gate 1 and Gate 2 pass; Gate 3A/4/V100 N/A for this RHS-only increment (no coupled solver, physical reconstruction, tracking change, PETSc/SLEPc, or production benchmark). Historical invalid evidence is retained: T03 initial CTest/smoke occurred before final link and was discarded; T06 targeted CTest was Not Run after only `macroflow3d_lib`; T07 first lacked `<iostream>` and only adjusted its mutation threshold, while its target-only full CTest initially lacked `run_operator_tests`; its subsequent serial 2/2 recovery is superseded by this independent T08 validation. `git diff --check master...HEAD` PASS. | Human review and root-only acceptance decision; do not alter State/checklist/NEXT/PR fields. |
