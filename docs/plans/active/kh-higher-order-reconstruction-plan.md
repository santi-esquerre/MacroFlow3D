# KH Higher-Order Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> or execute inline with explicit review checkpoints. Steps use checkbox syntax
> for tracking.

**Goal:** Replace the previous single KH backend with a backend family that can
compare `KH_LINEAR`, `KH_CUBIC_POTENTIAL_RECONSTRUCTION`, and
`KH_LOGK_CUBIC_POTENTIAL_RECONSTRUCTION` against `FACE_TRILINEAR`, then
validate whether higher-order consistent KH improves manufactured accuracy and
the key field/transport diagnostics.

**Architecture:** Keep the Par2 hot loop and MacroFlow3D/Par2 binding model
unchanged. Concentrate the interpolation change inside
`potential_flow_accessor.cuh`, expand the backend enum/config plumbing, then
extend tests and diagnostics around that stable interface.

**Tech Stack:** CUDA C++17, MacroFlow3D `real=double`, Par2_Core submodule,
CMake/CTest, `scripts/remote` for V100 validation.

---

## Scope and Constraints

- Keep PSPTA / Strategy A / Strategy C frozen.
- Keep the flow solver unchanged unless a minimal exposure fix is required.
- Preserve `FACE_TRILINEAR` compatibility.
- Preserve the current pure-advection-only restriction for KH backends in this
  phase.
- Prefer one focused branch with small commits over a large mixed refactor.

## File Map

### Primary implementation

- Modify: `src/external/Par2_Core/include/par2_core/types.hpp`
- Modify: `src/external/Par2_Core/src/internal/fields/potential_flow_accessor.cuh`
- Modify: `src/external/Par2_Core/src/kernels/move_particles.cu`
- Modify: `src/physics/particles/par2_adapter/par2_views.hpp`
- Modify: `src/physics/particles/par2_adapter/Par2TransportAdapter.cu`
- Modify: `src/io/config/Config.hpp`
- Modify: `src/io/config/ConfigValidator.hpp`
- Modify: `src/runtime/ensemble/EnsembleRunner.cu`
- Modify: `src/physics/flow/velocity_diagnostics.cu`
- Modify: `src/runtime/io/KhDiagnosticsWriter.hpp`
- Modify: `apps/test_kh_potential_reconstruction.cu`
- Modify: `CMakeLists.txt`

### Documentation

- Add: `docs/experiments/kh_cubic_reconstruction_design.md`
- Add: `docs/plans/active/kh-higher-order-reconstruction-plan.md`
- Add later: `docs/experiments/kh_higher_order_reconstruction_analysis.md`

## Task 1: Red tests for backend vocabulary and manufactured cubic behavior

- [ ] Expand the KH test app to express the four backend modes explicitly.
- [ ] Add failing tests for:
  - linear `h`, constant `K`;
  - linear `h`, smooth positive `K = exp(Y)`;
  - manufactured polynomial-compatible `h`;
  - periodic continuity in y/z;
  - `K` positivity / negative-event reporting.
- [ ] Add convergence/error reporting hooks for `N=16,32,64`.
- [ ] Build and run the KH test target to verify the new tests fail for the
  expected missing cubic functionality.

## Task 2: Backend plumbing and enum compatibility

- [ ] Replace the single KH mode in adapter/core enums with:
  - `FaceTrilinear`
  - `KhLinear`
  - `KhCubicPotentialReconstruction`
  - `KhLogKCubicPotentialReconstruction`
- [ ] Preserve parsing support for the previous
  `KH_POTENTIAL_RECONSTRUCTION` spelling by mapping it to `KH_LINEAR`.
- [ ] Extend config validation and runtime labeling without breaking previous
  FACE/KH runs.

## Task 3: Cubic interpolation kernel implementation

- [ ] Introduce local 1D cubic basis helpers and their analytic derivatives.
- [ ] Implement tensor-product 3D evaluation on a 4x4x4 stencil.
- [ ] Support value-only evaluation for `K` and `logK`.
- [ ] Support value+gradient evaluation for `h`.
- [ ] Keep stencil arithmetic in double.
- [ ] Document and implement x-boundary policy plus y/z periodic wrapping.
- [ ] Add diagnostics counters/metadata for interpolated `K <= 0` in direct
  cubic mode.

## Task 4: Integrate cubic evaluators into Par2 transport

- [ ] Route each KH backend to the right sampling function from the Par2 hot
  loop.
- [ ] Keep `FACE_TRILINEAR` on the legacy path.
- [ ] Keep transport semantics, particle status handling, and RNG unchanged.
- [ ] Preserve the pure-advection guard for all KH modes.

## Task 5: Extend diagnostics and CSV output

- [ ] Add backend labels for `KH_LINEAR`, `KH_CUBIC`, and `KH_LOGK_CUBIC`.
- [ ] Extend field diagnostics with `K` interpolation statistics:
  - min/max/mean interpolated `K`
  - count `K_interp <= 0`
  - clamp count if clamp is enabled
  - min/max interpolated `logK` for log-cubic
- [ ] Extend summary writers so the higher-order artifact tree can be built
  without replacing the old KH experiment bundle.

## Task 6: Local validation

- [ ] Configure or reuse a WSL debug build in the new worktree.
- [ ] Build `kh_potential_reconstruction_tests`.
- [ ] Run the targeted KH test executable / CTest target.
- [ ] Run at least one local smoke pipeline case per backend if the local CUDA
  environment permits.

## Task 7: Remote validation and experiment staging

- [ ] Run remote build/test smoke on V100.
- [ ] Run the 1-seed or 2-seed smoke stage first.
- [ ] Run the 10-seed smooth Gaussian case before any 100-seed stage.
- [ ] Only continue to the heavier stages if the manufactured tests and smoke
  diagnostics show improvement over `KH_LINEAR`.

## Task 8: Final analysis package

- [ ] Generate the required artifact tree under `artifacts/kh_higher_order/`.
- [ ] Produce the required CSV summaries and plots.
- [ ] Write `docs/experiments/kh_higher_order_reconstruction_analysis.md`.
- [ ] State a recommendation from A-F based on the measured data.
