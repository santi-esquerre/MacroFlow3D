# SF-28 — Periodic tricubic B-spline interpolation

- State: `pending`
- Goal: `Implementar interpolación tricúbica B-spline periódica C² de campos triplemente periódicos centrados en celda, con valor y gradiente en doble precisión, en GPU y con espejo CPU.`
- Depends on: `SF-27`
- Unlocks: `SF-30, SF-31`
- Branch: `feat/lester-sf28-periodic-tricubic-spline`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

C² interpolation of cell-centered triply periodic fields (value and gradient, double precision), shared by the closure gate (SF-30, potential spline) and the tracker (SF-31, labels).

Context: `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md` and `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`.

## Preconditions

- `SF-27` is `done` on the default branch.

## In scope

- New module `src/numerics/interpolation/` (`PeriodicTricubicBSpline.{cu,cuh}`).
- B-spline prefilter (coefficient deconvolution) via cuFFT following the pattern of `src/physics/stochastic/PeriodicGaussianField.cu` (plan/exec/destroy, `MF3D_CUFFT_CHECK`, explicit workspace with byte accounting).
- Device evaluation of value and gradient at arbitrary points with exact periodic wrapping; no allocations in evaluation.
- CPU mirror (same coefficients, same evaluation) for tests and the CPU prototype path.
- Fast ctest contract cases on trigonometric fields with grid ladders.

## Out of scope

- Non-periodic boundaries (added when the long domain is wired).
- Float precision.
- Any solver change.

## Files and symbols

- `src/numerics/interpolation/PeriodicTricubicBSpline.cu`, `.cuh` (new)
- `CMakeLists.txt` and a new test under `tests/` (fast contract cases registered as `interpolation*`)
- `src/numerics/AGENTS.md` is consulted, not modified

## Implementation specification

1. Prefilter by FFT: divide the spectrum by the cubic B-spline symbol, per axis.
2. Evaluate value and analytic gradient from the 4x4x4 coefficient stencil with exact integer wrapping of cell indices.
3. Expose an explicit workspace with exact byte accounting; evaluation kernels allocate nothing.
4. Provide the CPU mirror with the same coefficients and the same evaluation order.
5. Contract tests on a smooth trigonometric field over grids 16/32/64.

## Expected numerical effect

New capability only; no existing result changes.

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R interpolation
scripts/remote sync
scripts/remote run sf28-ladder -- "ctest --test-dir build/v100-release --output-on-failure -R interpolation"
scripts/remote wait sf28-ladder
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

- Observed order >= 3.8 for values and >= 2.8 for gradients on a smooth trigonometric field over >= 3 grids (16/32/64).
- Periodic wrap exact: evaluating at `x` and `x + L e_i` is bitwise identical.
- CPU mirror vs GPU agreement <= 1e-13 relative.
- Memory accounting exact: `cudaMemGetInfo` unchanged across evaluations.

## Regression surface

- Future consumers SF-30 and SF-31; the build (new CUDA sources; cuFFT link already present).

## Failure and rollback policy

- Missed orders or wrap inexactness leave the increment active; no consumer is enabled.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Implementation matches the scope and contains no unrelated changes.
- [ ] Targeted validation passes and its evidence is recorded.
- [ ] Required regression tests pass.
- [ ] Scientific or engineering findings are appended to the bitácora.
- [ ] Required human review is recorded.
- [ ] PR and commit identifiers are recorded.
- [ ] The master checklist entry is checked in this branch and `check-lester-increments.sh` passes.
<!-- completion-checklist:end -->

## Advancement rule

`SF-30` and `SF-31` become eligible after this increment is merged and marked `done` on the default branch.

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-02T00:00Z | not started | Specification created by the 2026-10-02 re-sequencing (decision record `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`). | Replaces the cancelled SF-27..SF-30 specifications (git history at `4670fb5`). | Activate only when the checker reports it READY. |
