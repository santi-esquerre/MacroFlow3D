# SF-28 — Periodic tricubic B-spline interpolation

- State: `active`
- Goal: `Implementar interpolación tricúbica B-spline periódica C² de campos triplemente periódicos centrados en celda, con valor y gradiente en doble precisión, en GPU y con espejo CPU.`
- Depends on: `SF-27`
- Unlocks: `SF-30, SF-31`
- Branch: `feat/lester-sf28-periodic-tricubic-spline`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `Claude Fable 5.1 orchestrator session (2026-10-03)`
- Started: `2026-10-03T01:51Z on master=afd9419`
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
| 2026-10-03T01:51Z | activation on `master=afd9419` (PR #44 merged; checker OK ready=SF-28 SF-29, nonterminal=none); delivery branch `feat/lester-sf28-periodic-tricubic-spline` | UNDERSTAND: SF-27 `done` on the default branch; SF-29 runs concurrently in its own session (own delivery branch/PR, mirror `--increment SF-29`), so activating SF-28 gives two nonterminal increments. Human-review increment (interpolation code under `src/numerics`, autonomy policy). Scientific-rigor skill invoked (numerical-method category: approximation-order, exact-periodicity, CPU/GPU-equivalence and memory contracts; no physical claim). Numerical contract fixed before any implementation (orchestration record `understanding.md` §2): cell centres `(i+1/2)h`, origin 0, period `L = N h`; prefilter = division of the cuFFT spectrum by the per-axis symbol `2/3 + (1/3) cos(2 pi m/N)` (min 1/3, well posed) and by `N1 N2 N3`, no half-cell twist; evaluation reduces each coordinate to `[0, L)` FIRST, then `t = x/h - 1/2`, `i0 = floor t`, standard uniform cubic B-spline weights and derivative weights, exact integer wrap, 64 loads, no allocation; CPU mirror = exact cyclic tridiagonal deconvolution per axis sharing the same `__host__ __device__` evaluation. DAG: N1 module (`src/numerics/interpolation/`, CMake lib entry) -> N2 contract tests (`tests/interpolation/`, `add_test interpolation_periodic_tricubic`) -> one integrator; orchestrator-owned detached V100 jobs on `--increment SF-28` (`sf28-base-build` on the base tree now; `sf28-build`, `sf28-ladder`, `sf28-ctest-full`, `sf28-smoke` after integration). | Pre-registered readings for the reviewer: (D-1) the spec's bitwise-wrap sentence cannot hold for every double `x` because `x + L` is itself rounded; the contract enforced is bitwise identity for every `x` such that `x + qL` is exactly representable (reduction-first guarantees it; test on dyadic points `k/2^16`, `L = 1`, `q` in {-2,-1,1,2,3}, grids 16/32/64 and 16x32x24). (D-2) order gate = min of the two consecutive two-level estimates (16->32, 32->64) in BOTH max-norm and RMS over a fixed deterministic 1e4-point off-node set; thresholds 3.8/2.8 verbatim. (D-3) CPU/GPU relative = max-norm difference over the point set divided by the max-norm of the CPU reference (value; gradient magnitude); 1e-13 verbatim. (D-5) test field `sin(2 pi x) cos(4 pi y) sin(2 pi z) + 0.5 cos(2 pi (x+y)) + 0.25 sin(6 pi z)` on `[0,1)^3` (max wavenumber 3, >= 5 cells per wavelength at 16). | Commit activation; start `sf28-base-build` on V100; launch N1 worker. |
