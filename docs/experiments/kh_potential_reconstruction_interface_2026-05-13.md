# KH potential reconstruction interface

Date: 2026-05-13

Status: experimental contract for `feature/kh-potential-flow-interpolation`.

Use this note when wiring or reviewing the KH velocity backend between
MacroFlow3D and `Par2_Core`. Do not use it as a PSPTA resumption plan; PSPTA
is frozen under `docs/experiments/pspta_standby_2026-05-13.md`.

## Scientific intent

For smooth locally isotropic Darcy flow with scalar conductivity,

```math
q(x) = -K(x) \nabla h(x),
```

the continuous velocity field is helicity-free in the Lester 2023 regime. The
KH backend tests whether evaluating interpolated scalar fields `K` and `h`, and
then reconstructing `q`, reduces numerical helicity and transverse
macrodispersion relative to direct face-velocity interpolation.

This is not PSPTA. It does not construct or preserve the two invariants
`psi1, psi2`, and it must not be reported as solving the full invariant
tracking problem.

## Backend modes

`Par2_Core::VelocityEvalMode` currently has two modes:

| Mode | Meaning |
| --- | --- |
| `FACE_TRILINEAR` | MacroFlow3D's existing Par2 path. The name follows the experiment vocabulary; the current Par2 setting is the legacy face-field linear interpolation path. |
| `KH_POTENTIAL_RECONSTRUCTION` | Experimental pure-advection backend. It samples cell-centered `K` and `h`, reconstructs `grad(h)`, and returns `q=-K grad(h)`. |

MacroFlow3D selects the mode through:

```yaml
transport:
  method: par2
  velocity_eval_mode: FACE_TRILINEAR
```

or:

```yaml
transport:
  method: par2
  velocity_eval_mode: KH_POTENTIAL_RECONSTRUCTION
```

The KH mode is rejected unless `diffusion=0`, `alpha_l=0`, and `alpha_t=0`.
That restriction is intentional for the first scientific comparison.

## Device data contract

MacroFlow3D owns all field storage. `Par2_Core` receives non-owning device
pointers and must not free or resize them.

| Field | Producer | Consumer | Location | Layout | Precision |
| --- | --- | --- | --- | --- | --- |
| `K` | MacroFlow3D stochastic stage | KH backend | device | cell-centered `i + nx*(j + ny*k)` | `real` |
| `head` | MacroFlow3D flow solve | KH backend | device | cell-centered `i + nx*(j + ny*k)` | `real` |
| `U,V,W` | MacroFlow3D velocity reconstruction | FACE backend and diagnostics | device | padded face arrays `(nx+1)*(ny+1)*(nz+1)` | `real` |

`real` is the repository scalar type from `src/core/Scalar.hpp`; current WSL
and V100 builds use double precision unless the scalar type is changed at build
time.

The KH binding path is:

1. `EnsembleRunner` owns `K_field` and `head_field`.
2. `Par2TransportAdapter::bind_potential_flow(K, head, bc)` maps them with
   `make_potential_flow_view`.
3. `Par2_Core::TransportEngine::bind_potential_flow` stores the raw view.
4. Particle stepping samples that view directly from device memory.

There is no CPU copy in this path.

## Grid and coordinates

MacroFlow3D passes a uniform `Grid3D`:

| Quantity | Convention |
| --- | --- |
| `nx, ny, nz` | cell counts |
| `dx, dy, dz` | uniform spacings |
| origin | Par2 grid origin, currently `(0,0,0)` via `make_grid` |
| cell center | `(i+0.5) dx`, `(j+0.5) dy`, `(k+0.5) dz` |
| x domain | `[0, nx*dx)` for active particle evaluation |
| y,z domain | periodic when YAML BCs are periodic |

Cell-centered scalar memory uses x-fastest indexing:

```text
idx = i + nx * (j + ny * k)
```

Padded face velocity memory uses:

```text
idx = i + (nx + 1) * (j + (ny + 1) * k)
```

for each component.

## Boundary treatment

The target macrodispersion setup uses:

| Axis | Current standard KH setup |
| --- | --- |
| x | Dirichlet head on west/east |
| y | periodic |
| z | periodic |

For scalar `h`, Par2 receives:

- x Dirichlet face values from the flow BC config.
- y/z periodic pairing when both faces are periodic.
- homogeneous Neumann as zero normal gradient if configured.
- extrapolate/clamp only as a fallback for unsupported scalar boundary types.

For KH gradients:

- interior cells use centered finite differences of `h`;
- periodic axes wrap indices;
- x Dirichlet boundaries use a half-cell one-sided gradient against the face
  value;
- x Neumann boundaries return zero normal gradient.

For `K(x)`, the first backend uses cell-centered trilinear interpolation with
the same periodic/clamping axis logic as the scalar sampler. The face-centered
velocity backend remains unchanged.

## Numerical method in this branch

Implemented now:

- trilinear interpolation of `K`;
- cell-centered finite-difference gradients of `h`;
- trilinear interpolation of those gradient components;
- direct reconstruction `q=-K grad(h)`;
- no projection, no velocity matching, no artificial divergence repair.

Not implemented yet:

- tricubic or B-spline scalar interpolation;
- exact invariant preservation;
- KH with molecular diffusion or mechanical dispersion;
- production-grade ensemble diagnostics beyond the sampled comparison scaffold.

## Comparison rules

FACE and KH comparisons must use:

- the same `K` seed;
- the same flow solve and `h`;
- the same grid, BCs, particles, `dt`, `n_steps`, and estimator;
- only `transport.velocity_eval_mode` and output paths may differ.

Do not compare runs if more than one scientific control changes.

## Required outputs

Per backend/run, the experiment harness should preserve:

- `config_used.yaml` from the effective config;
- `log.txt`;
- `alpha_timeseries.csv`;
- `field_diagnostics.csv`;
- `transport_diagnostics.csv`;
- `runtime_diagnostics.csv`.

Per ensemble, the collection step should write:

- `ensemble_summary.csv`;
- `alphaT_comparison_face_vs_kh.csv`;
- `helicity_comparison_face_vs_kh.csv`;
- `runtime_comparison.csv`.

## Retention and PSPTA boundary

During this KH phase, do not edit:

- `src/physics/particles/pspta/`;
- PSPTA invariant refinement logic;
- SLEPc invariant recovery code;
- pseudo-symplectic tracker behavior.

If KH work reveals a PSPTA issue, record it in the KH experiment report and
leave the PSPTA code untouched until the standby branch is explicitly resumed.
