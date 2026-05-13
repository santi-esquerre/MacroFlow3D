# KH Potential Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Add an experimental Par2 velocity evaluation backend that samples
cell-centered `K(x)` and hydraulic head `h(x)` on device memory and reconstructs
particle velocity as `q(x) = -K(x) grad h(x)`.

**Architecture:** Preserve the existing Par2 face-velocity backend as the
control path. Add a second explicit velocity source inside Par2_Core, bind it
from MacroFlow3D through a zero-copy `PotentialFlowView`, and select the backend
from YAML through `transport.velocity_eval_mode`. Keep PSPTA / Strategy A /
Strategy C untouched.

**Tech Stack:** CUDA C++17, Par2_Core static library, MacroFlow3D config parser,
CMake/CTest, existing remote V100 workflow.

---

## Scientific Intent

The hypothesis is that reconstructing velocity from smoother scalar fields,

```math
q(x) = -K(x) \nabla h(x),
```

may preserve the helicity-free structure of continuous isotropic Darcy flow
better than direct interpolation of discrete face-centered velocities.

Expected numerical effect:

- `KH_POTENTIAL_RECONSTRUCTION` may reduce sampled numerical helicity
  `q · curl(q)` relative to the existing face-velocity backend.
- If velocity-reconstruction error is a major source of transverse leakage,
  `alpha_T1(t)` and/or `alpha_T2(t)` should decrease relative to the same seed
  run with `FACE_TRILINEAR`.
- KH is not PSPTA and does not preserve the two invariants exactly.

Validation path:

- Unit/microtests for manufactured `K,h` fields.
- Build and targeted CTest.
- Smoke runs for both velocity modes with identical seeds/configs.
- Remote 2-seed and 10-seed gates before attempting 100 seeds.
- Ensemble CSV comparison and final report.

Regression surface:

- Par2 hot loop source selection.
- Face backend reproducibility.
- Boundary handling in KH gradients.
- MacroFlow3D config parsing and manifest/effective config output.
- Transport diagnostics and output paths.

## Backend Semantics

### `FACE_TRILINEAR`

Control backend. Uses the currently bound Par2 face/corner velocity field and
the existing Par2 particle stepping path. This name is the experiment-level
label for the current velocity-backend control, even though legacy Par2's direct
face sampling is per-component face interpolation.

### `KH_POTENTIAL_RECONSTRUCTION`

Experimental backend. For each particle step:

1. sample cell-centered conductivity `K(x)`,
2. estimate `grad h(x)` from the cell-centered hydraulic head field,
3. return `q(x) = -K(x) grad h(x)`.

Initial implementation is Option A from the experiment request:

- trilinear interpolation of cell-centered `K`,
- local finite-difference gradients of `h` at nearby cell centers,
- trilinear interpolation of those gradient components.

Tricubic/B-spline reconstruction is deliberately deferred and documented as the
next extension if the baseline KH signal is promising.

## Data Interface Contract

Par2_Core receives a non-owning view:

```cpp
template <typename T>
struct PotentialFlowView {
    const T* K;
    const T* head;
    size_t size;
    PotentialBoundaryConfig<T> head_bc;
};
```

Layout:

- `K` and `head` are cell-centered arrays of length `nx * ny * nz`.
- Linear index is `i + nx * (j + ny * k)`.
- `i` is x-fastest, matching MacroFlow3D `ScalarField`.
- Values are in device memory.

Ownership:

- MacroFlow3D owns the buffers.
- Par2_Core stores raw non-owning device pointers.
- Buffers must outlive the transport engine binding for the realization.
- No CPU copies are introduced.

Precision:

- Uses the Par2 template scalar `T`.
- MacroFlow3D binds `real`, currently `double`.

Geometry:

- Par2 `GridDesc<T>` supplies `nx, ny, nz`, `dx, dy, dz`, and origin.
- MacroFlow3D currently maps origin to `(0,0,0)`.

Boundary conventions:

- `y,z` periodicity follows the flow `BCSpec` and wraps scalar samples.
- `x` Dirichlet head values are passed explicitly to KH for one-sided boundary
  gradients near inlet/outlet.
- Non-periodic non-Dirichlet scalar boundaries use nearest valid interior
  values for the first implementation and are reported as a limitation.

Unsupported first-version combination:

- KH with nonzero molecular diffusion or dispersivity. The first KH backend is a
  pure-advection reconstruction test; enabling KH with dispersion must fail
  loudly rather than silently reusing face/corner drift logic.

## Files

Modify Par2_Core:

- `src/external/Par2_Core/include/par2_core/types.hpp`
  - add `VelocityEvalMode`.
- `src/external/Par2_Core/include/par2_core/views.hpp`
  - add scalar potential-flow view and boundary metadata.
- `src/external/Par2_Core/include/par2_core/transport_engine.hpp`
  - add `bind_potential_flow`.
- `src/external/Par2_Core/src/transport_engine.cu`
  - store and validate KH view, adjust source readiness and kernel launch.
- `src/external/Par2_Core/src/kernels/move_particles.cuh`
  - pass KH source to launch wrapper.
- `src/external/Par2_Core/src/kernels/move_particles.cu`
  - select face or KH velocity source inside the hot loop.
- `src/external/Par2_Core/src/internal/fields/potential_flow_accessor.cuh`
  - implement KH scalar sampling and gradient reconstruction.

Modify MacroFlow3D:

- `src/physics/particles/par2_adapter/par2_views.hpp`
  - add public adapter enum/config for velocity eval mode.
- `src/physics/particles/par2_adapter/par2_mapping.cuh`
  - map `KField`, `HeadField`, and `BCSpec` to Par2 KH view.
- `src/physics/particles/par2_adapter/Par2TransportAdapter.hpp`
  - add `bind_potential_flow`.
- `src/physics/particles/par2_adapter/Par2TransportAdapter.cu`
  - configure Par2 velocity eval mode and bind KH data.
- `src/io/config/Config.hpp`
  - add YAML field `transport.velocity_eval_mode`.
- `src/io/config/ConfigYaml.cpp`
  - parse and validate velocity mode strings.
- `src/io/config/ConfigValidator.hpp`
  - reject KH with dispersion in first version.
- `src/runtime/ensemble/EnsembleRunner.cu`
  - bind face velocity or KH potential flow explicitly.
- `apps/config_kh_potential_reconstruction.yaml`
  - standard paired experiment config.
- `apps/test_kh_potential_reconstruction.cu`
  - targeted manufactured-field tests.
- `CMakeLists.txt`
  - add `kh_potential_reconstruction_tests` CTest target.

Add scripts/docs:

- `scripts/remote/run_kh_ensemble_100.sh`
- `scripts/remote/collect_kh_ensemble_results.sh`
- `docs/experiments/kh_potential_reconstruction_interface_2026-05-13.md`
- `docs/experiments/kh_potential_reconstruction_ensemble.md`

## Tasks

### Task 1: Add failing KH tests

- [ ] Add `apps/test_kh_potential_reconstruction.cu`.
- [ ] Cover constant `K` with linear `h`.
- [ ] Cover variable `K` with linear `h`.
- [ ] Cover a smooth manufactured periodic transverse field.
- [ ] Cover periodic `y,z` samples near boundaries.
- [ ] Cover finite velocities for representative samples.
- [ ] Add the CTest target.
- [ ] Build and confirm the target fails because KH API/functions are missing.

### Task 2: Add Par2 KH view and source selection

- [ ] Add `VelocityEvalMode`.
- [ ] Add `PotentialFlowView` and scalar boundary metadata.
- [ ] Add `TransportEngine::bind_potential_flow`.
- [ ] Add KH source readiness checks.
- [ ] Keep `FACE_TRILINEAR` default behavior unchanged.

### Task 3: Implement KH sampling

- [ ] Implement cell-centered scalar interpolation.
- [ ] Implement local gradient estimates for `h`.
- [ ] Implement periodic `y,z` index wrapping.
- [ ] Implement x Dirichlet one-sided gradient support.
- [ ] Return `q = -K grad h`.
- [ ] Reject unsupported KH+dispersion combinations.
- [ ] Run KH tests and check tolerances.

### Task 4: Bind KH from MacroFlow3D

- [ ] Parse `transport.velocity_eval_mode`.
- [ ] Map YAML strings:
  - `FACE_TRILINEAR`
  - `KH_POTENTIAL_RECONSTRUCTION`
- [ ] Add adapter `bind_potential_flow(K, head, bc)`.
- [ ] In the Par2 path, bind face velocity for FACE and KH scalar fields for KH.
- [ ] Keep PSPTA path untouched.

### Task 5: Add required diagnostics and artifacts

- [ ] Add per-realization field diagnostics CSV for each backend.
- [ ] Add transport/runtime diagnostics CSV.
- [ ] Preserve existing macrodispersion time-series.
- [ ] Ensure `config_used.yaml` and logs are copied into the KH artifact tree.
- [ ] Add collection script producing:
  - `ensemble_summary.csv`
  - `alphaT_comparison_face_vs_kh.csv`
  - `helicity_comparison_face_vs_kh.csv`
  - `runtime_comparison.csv`

### Task 6: Add standard run configs and remote scripts

- [ ] Add a standard KH config matching the current Par2 macrodispersion setup.
- [ ] Add remote runner script for seeds `0..99` and both velocity modes.
- [ ] Add smoke/mini/full staging:
  - 2 seeds,
  - 10 seeds,
  - 100 seeds when timing is acceptable.
- [ ] Use `scripts/remote` workflow only.

### Task 7: Validate and report

- [ ] Local build, if CUDA driver permits.
- [ ] Remote build and targeted tests.
- [ ] Remote smoke for FACE and KH.
- [ ] Run at least the feasible ensemble stage.
- [ ] Generate figures/tables from CSVs.
- [ ] Write `docs/experiments/kh_potential_reconstruction_ensemble.md`.
- [ ] Commit in small, single-purpose commits.

## Baseline Local Check

On 2026-05-13, before KH code edits:

- `cmake --preset wsl-debug`: passed.
- `cmake --build build/wsl-debug -j`: passed.
- `ctest --test-dir build/wsl-debug --output-on-failure`: failed before test
  logic due local CUDA environment:

```text
CUDA driver version is insufficient for CUDA runtime version
```

This is treated as a local validation limitation. GPU execution evidence must
come from the remote V100 workflow for this branch.
