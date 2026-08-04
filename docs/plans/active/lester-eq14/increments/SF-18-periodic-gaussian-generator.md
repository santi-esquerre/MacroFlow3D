# SF-18 — Periodic Gaussian generator

- State: `pending`
- Goal: `Generar campos gaussianos suaves verdaderamente periódicos y reproducibles.`
- Depends on: `SF-17`
- Unlocks: `SF-19`
- Branch: `science/lester-sf18-periodic-gaussian-generator`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf18-periodic-gaussian-generator`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A prerequisite`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Remove the current finite-box nonperiodicity and direct-summation scaling from
the physical validation input so boundary roughness cannot masquerade as a
streamfunction failure.

## Preconditions

- SF-17 completes the solver-side homotopy infrastructure.

## In scope

- A discrete spectral generator for periodic Gaussian-covariance log
  conductivity, reproducible seeds, variance/log-mean normalization, and cuFFT
  integration where justified.

## Out of scope

- Exponential covariance, Darcy flow, lambda continuation, and modifying the
  existing generator's behavior for current callers.

## Files and symbols

- Add `src/physics/stochastic/PeriodicGaussianField.cuh/.cu` and config/types
  scoped to the new generator.
- Extend CMake with cuFFT only for this implementation.

## Implementation specification

1. Sample reciprocal-lattice wavevectors and enforce Hermitian symmetry.
2. Use the periodicized Gaussian spectrum corresponding to the documented
   covariance convention; set/control the zero mode separately.
3. Normalize realized variance and log mean explicitly and record both.
4. Define a seed-to-mode mapping that represents the same continuous
   realization under grid refinement, through spectral truncation or generation
   at the finest requested grid followed by controlled restriction.
5. Retain the old stochastic generator API and results for existing configs.

## Expected numerical effect

Generated `Y` and its derivatives join smoothly at domain boundaries and their
statistics approach the requested Gaussian covariance.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R periodic_gaussian
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Same config/seed produces identical device output on repeated runs.
- Boundary wrap value/derivative discrepancies match interior discretization
  error, not an O(1) jump.
- Mean and variance meet documented finite-sample tolerances.
- Radially binned spectrum/covariance matches the requested Gaussian shape over
  resolved modes; exact tolerances are fixed in the test before implementation.

## Regression surface

- cuFFT linking, seed semantics, covariance convention, zero mode, and memory
  at `256^3`.

## Failure and rollback policy

- Do not relabel the current continuous-wavevector direct sum as periodic.
- A disputed spectral normalization blocks physical benchmarks but not the
  previously accepted synthetic solver tests.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Periodic spectral construction and Hermitian symmetry are implemented.
- [ ] Seed, mean, variance, and refinement semantics are documented.
- [ ] Reproducibility, wrap, covariance, and spectrum tests pass.
- [ ] Existing stochastic configs are unchanged.
- [ ] Memory/runtime and human review are recorded.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-18 complete and selects SF-19.
<!-- completion-checklist:end -->

## Advancement rule

SF-19 may compute affine-periodic Darcy flow on these accepted periodic fields.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
