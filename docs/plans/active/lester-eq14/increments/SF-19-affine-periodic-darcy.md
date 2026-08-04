# SF-19 — Affine-periodic Darcy solve

- State: `pending`
- Goal: `Resolver el flujo Darcy affine-periodic necesario para el benchmark triplemente periódico.`
- Depends on: `SF-18`
- Unlocks: `SF-20`
- Branch: `science/lester-sf19-affine-periodic-darcy`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf19-affine-periodic-darcy`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A prerequisite + Gate 4`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Produce a mass-conservative reference Darcy field with triply periodic
fluctuation and a controlled mean flux, rather than applying incompatible
Dirichlet head boundaries.

## Preconditions

- SF-18 supplies accepted periodic scalar conductivity fields.

## In scope

- Periodic affine-head cell problems, effective conductivity tensor, prescribed
  mean-flux solve, and affine-aware CompactMAC velocity reconstruction.

## Out of scope

- Changing the existing Dirichlet flow path or invoking the streamfunction
  nonlinear solver.

## Files and symbols

- Add `src/physics/flow/AffinePeriodicFlowSolver.cuh/.cu` and an affine velocity
  reconstruction overload.
- Reuse projected `A(K)`, PCG, and MG through explicit sign/coefficient adapters.

## Implementation specification

1. Solve three zero-mean periodic head-corrector cell problems for unit mean
   pressure gradients.
2. Integrate their face fluxes to construct the 3x3 effective conductivity.
3. Solve the small host 3x3 system for a pressure gradient producing target
   mean Darcy flux `(1,0,0)`.
4. Recombine correctors and reconstruct final CompactMAC velocity with affine
   pressure contribution and harmonic `K` faces.

## Expected numerical effect

The reference velocity is periodic, discretely mass conservative, and has the
specified mean flux even when a finite realization has transverse effective
coupling.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R affine_periodic_flow
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- `K=1` gives exact identity effective conductivity and uniform target flux.
- Effective tensor symmetry defect is below `1e-10` on test fields and all
  eigenvalues are positive.
- Mean flux error is below `1e-10` relative; mass residual meets flow tolerance.

## Regression surface

- Flow operator signs, harmonic-K velocity faces, periodic gauges, and existing
  head-solver APIs.

## Failure and rollback policy

- Do not force transverse mean flux to zero by discarding tensor coupling.
- Do not alter the current boundary-driven flow solver; keep this a separate
  explicit entry point.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Three cell problems and effective tensor are implemented.
- [ ] Target mean-flux solve and affine velocity reconstruction are implemented.
- [ ] Homogeneous, symmetry/SPD, flux, and mass tests pass.
- [ ] Existing flow cases remain unchanged.
- [ ] Gate 4 interpretation and human review are recorded.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-19 complete and selects SF-20.
<!-- completion-checklist:end -->

## Advancement rule

SF-20 may combine accepted periodic fields and Darcy flow with lambda
continuation in the streamfunction solver.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
