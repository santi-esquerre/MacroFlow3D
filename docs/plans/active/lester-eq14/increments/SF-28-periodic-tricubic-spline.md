# SF-28 — Periodic tricubic B-spline interpolation

- State: `awaiting_review`
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
- Commit: `cb99732` (audited source-bearing head: integration = N1 `300a1cb` + N2 `cb99732` on the activation head `aebfb6f`, base `afd9419`; later commits are docs/metadata only)

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
- [x] Implementation matches the scope and contains no unrelated changes.
- [x] Targeted validation passes and its evidence is recorded.
- [x] Required regression tests pass.
- [x] Scientific or engineering findings are appended to the bitácora.
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
| 2026-10-03T02:20Z | N1 `300a1cb` + N2 `cb99732` on the activation head `aebfb6f`; integration = `cb99732` (fast-forward, no conflict, no integration-only change) | EXECUTE/AUDIT: N1 (worker, isolated worktree) delivered `src/numerics/interpolation/PeriodicTricubicBSpline.{cuh,cu}` (namespace `macroflow3d::interpolation`: POD view + `__host__ __device__ evaluate_point` with reduction-first wrap and the standard uniform cubic B-spline weights; cuFFT D2Z -> symbol/`1/N` kernel -> Z2D prefilter with per-call plans, `cufftGetSize`, exact-byte report; SoA batched kernel without allocation or sync; CPU mirror = exact cyclic tridiagonal Thomas + Sherman-Morrison sharing the same evaluation; `N_d >= 4`, odd `N` allowed) and the `INTERPOLATION_SOURCES` CMake entry. N2 (worker) delivered `tests/interpolation/periodic_tricubic_bspline_tests.cu` (65 checks: 31 validation, node condition, partition of unity, order ladder T1, bitwise wrap T2, CPU/GPU T3, memory T4, timing) registered as ctest `interpolation_periodic_tricubic` (16 entries). Orchestrator audits (`.claude/orchestration/SF-28-periodic-tricubic-spline/audits/`): every diff read in full; weights, symbol and the cyclic solver re-derived by hand and matched against an independent numpy reference; the worker self-check and the contract binary re-executed by the orchestrator with identical numbers; `compute-sanitizer memcheck` on the full test executable: 0 errors, 0 leaks. No blocking/major/minor findings; informational only (`cufftGetSize` = 0 on every plan <= 64^3 on the local GPU, 4.3 MB on V100; `grid.Lx()` may differ from an intended `L` by one ulp for `N` such as 49 — the module always uses `grid.Lx()`; the auto-detected sccache launcher must be disabled for local nvcc builds). | Local (RTX 3050, sm_86, CUDA 13.4, wsl-debug): orders 16->32->64 value 4.28/4.06 (max) and 4.34/4.13 (RMS), gradient 3.22/3.08 (max) and 3.34/3.09 (RMS) — gates 3.8/2.8 PASS; wrap 0 mismatches / 510 pairs on 16^3, 32^3, 64^3, 16x32x24 (GPU and CPU); CPU vs GPU normwise <= 3.2e-15 (independent prefilters) and <= 1.1e-15 (same coefficients), coefficients 2.5e-15; `cudaMemGetInfo` delta 0 across 4 evaluations; node condition <= 6.6e-16; executable 3.5 s. Integrator: three-commit linear chain verified, local build + `-R interpolation` 65/65, `ctest -N` = 16, checker OK. | V100 evidence on the SF-28 mirror, then FINAL_AUDIT. |
| 2026-10-03T03:20Z | FINAL_AUDIT positive on `cb99732`; State -> `awaiting_review` (human-review increment: interpolation code under `src/numerics`) | V100 (`scripts/remote --increment SF-28`, mirror `~/MacroFlow3D-SF-28`, preset `v100-release`, detached jobs, GPU 0): `sf28-base-build` on the base tree (`ctest -N` = 15); `sf28-build` on the integrated tree (0 errors, `ctest -N` = 16); **`sf28-ladder`: 65/65 PASS in 4.15 s** with the same digits as locally (value orders 4.2756/4.0603 max, 4.3433/4.1337 RMS; gradient 3.2191/3.0794 max, 3.3400/3.0946 RMS; wrap 0/510 x 8; CPU/GPU <= 3.2e-15; `cufft_work_area_bytes` = 4 325 376, total 8 585 216, `cudaMemGetInfo` delta 0 over 4 evaluations); **`sf28-ctest-full`: 16/16 passed, `Total Test time (real) = 2709.48 s`** (SF-27 baseline 2 709.82 s for 15 entries); `sf28-smoke` (`config_pspta_small`) exit 0. Full diff from `afd9419`: 5 files (`CMakeLists.txt` +12, the two module files, the test, this spec); frozen stack byte-identical (`git diff afd9419 HEAD -- src/physics src/numerics/{operators,solvers,blas,constraints} src/multigrid apps` empty). Logs archived under `.claude/orchestration/SF-28-periodic-tricubic-spline/integration/`. | Acceptance thresholds: (1) order >= 3.8 / 2.8 over 16/32/64: PASS (min 4.06 / 3.08); (2) wrap bitwise exact: PASS under the pre-registered reading D-1 (exactly representable shifts); (3) CPU mirror vs GPU <= 1e-13: PASS (<= 3.2e-15); (4) memory accounting exact, `cudaMemGetInfo` unchanged across evaluations: PASS. Gate 1 + Gate 2 satisfied. Review items left open for the human: D-1 (wrap contract reading), odd-`N` support, per-call cuFFT plans in the prefilter (D-6, not a hot path), the `grid.Lx()` ulp note. | Push the delivery branch and open the PR; stop at AWAIT_HUMAN_REVIEW; do not set `done` or check the master entry. |
