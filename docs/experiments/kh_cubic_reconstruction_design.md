# KH Cubic Reconstruction Design

Date: 2026-05-15

Status: design note for `feature/kh-higher-order-reconstruction`.

Use this note before editing the higher-order KH backend. It records the
current code path, data contract, and the boundary/precision assumptions that
the cubic variants must preserve.

This phase is limited to the Par2 / MacroFlow3D KH line. PSPTA, Strategy A,
and Strategy C remain on standby and are intentionally excluded from the write
set.

## Scientific Question

The previous KH backend tested

```math
q(x) = -K(x) \nabla h(x)
```

with cell-centered `K` and `h`, but used a low-order, partially inconsistent
reconstruction:

- `K(x)` from local trilinear interpolation of cell-centered `K`;
- `grad(h)` from finite differences at nearby cell centers;
- component-wise trilinear interpolation of those discrete gradients.

That experiment reduced normalized helicity relative to the face-velocity
backend, but it did not reduce `alpha_T` and it increased the divergence
diagnostic.

The present phase asks a narrower question:

> Did the previous KH result fail because the KH idea is wrong, or because the
> reconstruction was too low-order and not mathematically aligned with a single
> smooth interpolant for `h`?

## Current Code Path

### MacroFlow3D -> Par2_Core binding

The current Par2 transport path is:

1. `src/runtime/ensemble/EnsembleRunner.cu`
2. `src/physics/particles/par2_adapter/Par2TransportAdapter.*`
3. `src/physics/particles/par2_adapter/par2_mapping.cuh`
4. `src/external/Par2_Core/include/par2_core/transport_engine.hpp`
5. `src/external/Par2_Core/src/transport_engine.cu`
6. `src/external/Par2_Core/src/kernels/move_particles.cu`
7. `src/external/Par2_Core/src/internal/fields/potential_flow_accessor.cuh`

`EnsembleRunner` currently selects between:

- `FACE_TRILINEAR`
- `KH_POTENTIAL_RECONSTRUCTION`

using `transport.velocity_eval_mode`.

### Existing KH implementation site

The existing KH backend is implemented in:

- `src/external/Par2_Core/src/internal/fields/potential_flow_accessor.cuh`

The hot loop calls:

```cpp
sample_velocity_kh_potential(potential_flow, grid, x, y, z, vx, vy, vz);
```

from `src/external/Par2_Core/src/kernels/move_particles.cu`.

That function currently:

1. samples `K(x)` with trilinear interpolation on cell-centered `K`;
2. samples `dhdx`, `dhdy`, `dhdz` by trilinear interpolation of per-cell finite
   differences;
3. returns `q = -K grad(h)`.

This is the code that will be generalized into:

- `KH_LINEAR`
- `KH_CUBIC_POTENTIAL_RECONSTRUCTION`
- `KH_LOGK_CUBIC_POTENTIAL_RECONSTRUCTION`

without changing the Par2 particle step semantics.

## Current Data Contract

### Scalar fields

`Par2_Core` receives a non-owning:

```cpp
template <typename T>
struct PotentialFlowView {
    const T* K;
    const T* head;
    size_t size;
    PotentialBoundaryConfig<T> head_bc;
};
```

Current producer:

- `make_potential_flow_view` in
  `src/physics/particles/par2_adapter/par2_mapping.cuh`

Current ownership:

- MacroFlow3D owns `KField` and `HeadField`;
- Par2_Core stores raw device pointers only;
- no host copies are introduced.

### Memory layout

`K` and `head` are cell-centered arrays with x-fastest indexing:

```text
idx = i + nx * (j + ny * k)
```

This layout is consistent across:

- `KField`
- `HeadField`
- the KH test app
- `PotentialFlowView`

### Precision

Repository scalar type is:

```cpp
using real = double;
```

from `src/core/Scalar.hpp`.

So the current WSL and V100 builds already evaluate KH in double precision at
the storage level. The higher-order phase should still make the evaluation path
explicitly double-centric, so that cubic weights, stencil accumulation,
analytic derivatives, and `q = -K grad(h)` remain in double even if storage is
later reduced.

## Boundary Conventions

### Flow boundary conditions

MacroFlow3D's standard transport setup uses:

- x: Dirichlet west/east
- y: periodic
- z: periodic

The scalar BC mapping currently goes through:

- `to_par2_scalar_bc`
- `make_par2_potential_bc`

in `src/physics/particles/par2_adapter/par2_mapping.cuh`.

### Existing KH boundary treatment

In `potential_flow_accessor.cuh`:

- periodic axes use wrapped indices;
- non-periodic axes clamp interpolation to the nearest valid cell;
- x Dirichlet boundaries use one-sided half-cell gradients against the face
  value;
- Neumann currently maps to zero normal gradient;
- the sampler returns zero velocity if `x < px` or `x >= x_max()`.

### Implication for the cubic phase

The cubic phase must preserve:

- periodic wrap in y/z;
- explicit treatment of x near west/east boundaries;
- no hidden change in particle validity rules near x edges.

The boundary policy for cubic stencils in x must be documented. Reasonable
choices include:

- one-sided cubic from the first interior 4 cells plus Dirichlet face data;
- clamped cubic stencil with reduced effective order near boundaries;
- ghost-cell reconstruction derived from boundary values.

The implementation must pick one and state it explicitly.

## Current Particle/Transport Behavior

The Par2 kernel still owns:

- particle injection,
- advection,
- periodic wrap counters,
- exit handling,
- any future dispersion/drift path.

For KH mode today:

- dispersion is rejected at config validation and transport-engine
  construction;
- only pure advection is allowed;
- the transport hot loop is otherwise unchanged.

This is important because the higher-order KH work should remain a velocity
evaluation experiment, not a rewrite of transport semantics.

## Existing Diagnostics and Outputs

Current KH comparison outputs already exist for the previous experiment:

- field diagnostics via `compute_velocity_eval_diagnostics`
- face-vs-kh comparison via `compute_velocity_backend_comparison`
- CSV writing via `src/runtime/io/KhDiagnosticsWriter.hpp`

The current summaries capture:

- speed
- divergence diagnostic
- curl magnitude
- absolute and normalized helicity
- relative velocity difference vs FACE
- runtime
- transport final variances / active-particle counts

The higher-order phase should extend these outputs rather than inventing a
parallel diagnostics path.

## Design Direction for the Higher-Order Phase

### New backend modes

Keep compatibility with the previous experiment while expanding the enum-level
backend vocabulary to:

- `FACE_TRILINEAR`
- `KH_LINEAR`
- `KH_CUBIC_POTENTIAL_RECONSTRUCTION`
- `KH_LOGK_CUBIC_POTENTIAL_RECONSTRUCTION`

`KH_LINEAR` is the current trilinear / finite-difference baseline, renamed so
that the new cubic variants can be compared without ambiguity.

### Mathematical consistency requirement

The cubic variants must not do:

1. cubic interpolation of `h`
2. separate finite differences for `grad(h)`

Instead they must do:

1. construct one local cubic interpolant for `h`;
2. obtain `grad(h)` by analytic differentiation of that same interpolant;
3. combine with `K(x)` or `exp(logK(x))` inside the same device-side sampling
   routine.

### Interpolation family

A local tensor-product 4x4x4 cubic interpolant is the right fit for this phase.
The implementation can be Catmull-Rom, cubic Hermite, or an equivalent local
tricubic basis, provided that:

- it is device-friendly;
- it uses a local 4x4x4 stencil;
- it exposes both value and derivative;
- the exact basis is documented in code and in the final report.

### `K` positivity

Direct cubic interpolation of `K` may overshoot and produce nonphysical
negative values. This phase therefore needs two explicit policies:

- `KH_CUBIC`: detect and report `K_interp <= 0`, optionally clamp with
  diagnostics if needed;
- `KH_LOGK_CUBIC`: interpolate `Y = log(K)` and reconstruct `K = exp(Y)` so
  positivity is guaranteed by construction.

## Expected Write Set

Primary implementation files:

- `src/external/Par2_Core/include/par2_core/types.hpp`
- `src/external/Par2_Core/src/internal/fields/potential_flow_accessor.cuh`
- `src/external/Par2_Core/src/kernels/move_particles.cu`
- `src/physics/particles/par2_adapter/par2_views.hpp`
- `src/physics/particles/par2_adapter/Par2TransportAdapter.cu`
- `src/io/config/Config.hpp`
- `src/io/config/ConfigValidator.hpp`
- `src/runtime/ensemble/EnsembleRunner.cu`
- `src/physics/flow/velocity_diagnostics.cu`
- `src/runtime/io/KhDiagnosticsWriter.hpp`
- `apps/test_kh_potential_reconstruction.cu`
- `CMakeLists.txt`

Primary docs/artifacts to add or update:

- `docs/experiments/kh_cubic_reconstruction_design.md`
- `docs/experiments/kh_higher_order_reconstruction_analysis.md`
- `artifacts/kh_higher_order/...`

PSPTA files are intentionally out of scope.

## Acceptance Notes for This Phase

Before large ensembles, the cubic variants must first prove themselves on
manufactured fields:

- exactness for linear `h`, constant `K`;
- correct scaling for linear `h`, smooth positive variable `K`;
- lower gradient / velocity error than `KH_LINEAR` on manufactured
  polynomial-compatible fields;
- continuity across periodic y/z boundaries;
- explicit `K` positivity diagnostics.

If the cubic variants do not improve the manufactured tests relative to
`KH_LINEAR`, the ensemble stage should stop there and the issue should be
understood before further remote cost is spent.
