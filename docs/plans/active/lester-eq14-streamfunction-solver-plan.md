# Lester equation (14) streamfunction solver plan

## Purpose

This is the authoritative operational plan for new invariant-construction work in MacroFlow3D.

The project direction is to design and integrate a numerical solver for the coupled nonlinear Lester et al. equation (14) system that computes two streamfunctions / invariants `psi1`, `psi2` for steady, smooth, locally isotropic 3D Darcy flow.

The invariant consumer will be designed after accepted equation (14) invariants exist. Existing PSPTA marching, transport-operator, eigensolver, PETSc/SLEPc, refinement, and transport code are legacy context and possible migration infrastructure, not proof that the equation (14) solver or final transport consumer already exists.

## Current confirmed implementation state

Confirmed from the repository on 2026-07-13:

- `real` is `double` in `src/core/Scalar.hpp`.
- Structured grids use cell-centered scalar fields with x-fastest indexing `i + nx*(j + ny*k)`.
- `Grid3D` stores `dx`, `dy`, `dz`, but current multigrid and flow comments state the implementation assumes isotropic spacing.
- `K` and head are cell-centered in `src/physics/common/fields.cuh`.
- Darcy velocity is reconstructed on CompactMAC faces: `U(nx+1,ny,nz)`, `V(nx,ny+1,nz)`, `W(nx,ny,nz+1)`.
- Padded face-field velocity exists for Par2 compatibility.
- Flow solve targets `-div(K grad h) = rhs` using harmonic face means.
- Boundary types are Dirichlet, Neumann, Periodic. Periodic boundaries must be paired.
- Head solvers include CG, standalone MG, and PCG with MG preconditioner.
- The MG hierarchy uses 2:1 coarsening in all directions and geometric coarsening of `K`.
- `VarCoeffLaplacian` implements a matrix-free variable-coefficient operator with harmonic face means and legacy sign conventions.
- `Poisson3DOperator` exists but is homogeneous-Dirichlet `-Delta`, not the equation (14) operator.
- PSPTA currently has legacy x-marching invariant construction in `PsptaPsiField`, a transport near-nullspace operator `D = v.grad`, SLEPc eigensolver integration, a `RefinementAC` skeleton, and a legacy transport engine.
- `RefinementAC.cu` is explicitly not implemented.
- Current configs default to exponential covariance in major smoke/reference cases; Gaussian covariance support exists as `covariance_type: 1`.

## Target mathematical system

For scalar, locally isotropic conductivity `k(x)>0`, Lester et al. equation (14) is represented here as:

```math
Delta psi1 - grad(log k).grad(psi1) = S2
Delta psi2 - grad(log k).grad(psi2) = S1
```

with

```math
S_i =
((B x grad psi_i).(grad psi1 x grad psi2)) /
|grad psi1 x grad psi2|^2
```

and

```math
B = (grad psi1.grad) grad psi2 - (grad psi2.grad) grad psi1.
```

The ideal constraints are:

```math
v = grad psi1 x grad psi2
v.grad psi1 = 0
v.grad psi2 = 0
```

Use the equivalent divergent form:

```math
Delta psi - grad(log k).grad psi = k div((1/k) grad psi).
```

Define

```math
q = 1/k,
A psi = -div(q grad psi).
```

Then a decoupled nonlinear iteration solves:

```math
A psi1 = -q S2
A psi2 = -q S1
```

This form is the priority because it avoids explicitly differencing `grad(log k)`, preserves a variable-coefficient diffusion structure, and uses the same linear operator for both streamfunctions for fixed `k`.

## Scope assumptions

Initial work is limited to:

- steady Darcy flow;
- scalar locally isotropic conductivity;
- `k(x)>0`;
- sufficiently smooth fields;
- no stagnation points in the first validation problems;
- structured domains;
- mean flow primarily in `x1`;
- transverse periodic or triply periodic benchmark conditions.

Gaussian-covariance log-conductivity fields are the primary validation case for smooth invariants.

Exponential-covariance fields are not equivalent. They may be useful later after explicit smoothing, but every such run must record the smoothing scale, resolution dependence, whether the original or regularized problem is being solved, and whether invariants converge under refinement.

Tensorial or locally anisotropic conductivity is out of initial scope. Do not assume two global invariants exist there.

## Architecture hypothesis

Priority hypothesis to verify:

- reuse the existing variable-coefficient operator, PCG, and MG preconditioner for `A psi = -div(q grad psi)` with `q=1/k`.

This is not confirmed yet. Verification must check:

- operator sign convention;
- whether coefficients should be harmonic means of `q` rather than `K`;
- boundary treatment for periodic and mean-flow streamfunctions;
- gauge handling for the nullspace;
- symmetry / positive definiteness after gauge fixing;
- compatibility between linear operator residuals and nonlinear differential operators;
- whether geometric coarsening of `K` is appropriate for `q`;
- whether the current MG smoother/residual implement the exact same operator.

Do not document MG reuse as accepted until these tests pass.

## Discretization requirements

Before choosing a final discretization, an implementation task must determine:

- storage location for `psi1`, `psi2`;
- compatible gradient operators;
- compatible divergence for `A`;
- how to compute `H(psi2) grad psi1 - H(psi1) grad psi2`;
- whether Hessian-vector products can be computed directly in local GPU kernels without storing all nine Hessian components;
- the order of accuracy;
- boundary-condition behavior for transverse-periodic and triply periodic tests;
- consistency between the operator used in linear solves and the residual `F(Psi)`.

Avoid mixing unrelated finite-difference formulas for the linear operator, nonlinear source, and validation metrics.

## Incremental strategy

### Stage A: minimal benchmark

Start with small reproducible cases:

- `k=1`;
- exact expected solution;
- grids `16^3`, `32^3`, `64^3`;
- periodic conditions;
- decomposed fields:

```math
psi1 = vbar x2 + psi1_tilde
psi2 = x3 + psi2_tilde
```

with gauges:

```math
mean(psi1_tilde)=0
mean(psi2_tilde)=0
```

### Stage B: homogeneous linear problem

Solve first with `S1=S2=0`. Use this as the initialization for nonlinear solves.

### Stage C: damped Picard

For `(psi1^n, psi2^n)`:

1. compute gradients;
2. compute required Hessian-vector products;
3. build `B`;
4. compute `S1^n`, `S2^n`;
5. solve `A psi1_hat = -q S2^n`, `A psi2_hat = -q S1^n`;
6. relax `psi_i^{n+1} = (1-omega) psi_i^n + omega psi_i_hat`;
7. restore gauges;
8. evaluate algebraic and physical residuals.

Anderson acceleration is a later extension, not the first implementation.

### Stage D: continuation

Support continuation in:

- heterogeneity: `k_lambda = exp(lambda Y)`, `lambda: 0 -> 1`;
- nonlinearity: `A psi1 = -eta q S2`, `A psi2 = -eta q S1`, `eta: 0 -> 1`;
- grid: `16^3 -> 32^3 -> 64^3 -> 128^3 -> 256^3`.

Reuse the solution from each stage as the initial condition for the next.

### Stage E: matrix-free Newton-Krylov

Only after residuals, Picard, differential operators, and small cases are validated, move toward:

```math
F(Psi) = [A psi1 + q S2, A psi2 + q S1]^T
J(Psi) deltaPsi = -F(Psi)
```

Approximate products without assembling the Jacobian:

```math
J(Psi) p ~= (F(Psi + epsilon p) - F(Psi)) / epsilon.
```

Priority preconditioner:

```math
P = [[A,0],[0,A]].
```

The intended inverse approximation for each block is the validated multigrid or PCG/MG path.

## Singularity policy

The denominator `|grad psi1 x grad psi2|^2` can become small.

Do not hide this with an arbitrary constant. Initial controlled regularization may use:

```math
|c|^2 -> |c|^2 + epsilon^2 v_ref^2
```

where `epsilon` is configurable and logged. Required diagnostics:

- minimum `|c|`;
- percentiles 0.1%, 1%, 5%;
- continuation `epsilon -> 0`;
- distinction between numerical failure and possible physical stagnation.

A result is not converged only because the algebraic residual decreased.

## Validation metrics

Every accepted solver milestone must report more than linear/PDE residuals.

Required metrics:

- coupled residual:

```math
r_F = sqrt(||F1||_2^2 + ||F2||_2^2)
```

- velocity reconstruction:

```math
e_v = ||grad psi1 x grad psi2 - v_D||_2 / ||v_D||_2
```

- Darcy invariance:

```math
e_i = ||v_D.grad psi_i||_2 / (||v_D||_2 ||grad psi_i||_2)
```

- reconstructed-flow divergence:

```math
e_div = ||div(grad psi1 x grad psi2)||
```

- non-degeneracy percentiles of `|grad psi1 x grad psi2|`;
- grid convergence between successive resolutions;
- conservation of `psi_i(X(t)) - psi_i(X(0))` along trajectories in reconstructed and Darcy velocity fields.

Do not accept a solver on tolerance achievement alone.

## CPU/GPU split

CPU reference is appropriate for:

- small cases;
- finite-difference verification;
- Picard/Newton experiments;
- kernel validation;
- comparison with external libraries already present or explicitly justified.

GPU production should own:

- operator application;
- gradients;
- Hessian-vector products;
- `S1`, `S2` construction;
- reductions;
- Picard loops;
- multigrid / preconditioning;
- matrix-free Jacobian products;
- grids `128^3` and above.

Do not add a heavy dependency without checking the toolchain, CUDA version, target V100/local GPU hardware, deployment, maintenance, and whether existing solvers can be reused.

## Task plan

| # | Task | Probable files/modules | Dependencies | Done criterion | Test | Risk type |
|---|------|------------------------|--------------|----------------|------|-----------|
| 1 | `A` operator test with `k=1` | `src/numerics/operators/`, new Lester operator tests | none | `A` matches periodic `-Delta` on manufactured fields and sign is documented | targeted operator executable | engineering |
| 2 | `A` operator test with smooth variable `k` | numerics operator tests, CPU reference helper | task 1 | manufactured `-div(q grad psi)` converges with refinement | CPU/GPU comparison | research + engineering |
| 3 | gradient tests | new differential-operator module or PSPTA invariant utilities | task 1 | gradients match manufactured periodic fields | gradient unit test | engineering |
| 4 | Hessian-vector product tests | local GPU kernels, CPU reference | task 3 | `H(psi) g` matches finite-difference reference | Hessian-vector test | research |
| 5 | `B` construction test | Lester source-term module | tasks 3-4 | `B=0` for simple analytic controls where expected | `B` unit test | research |
| 6 | `S1,S2` tests | Lester source-term module | task 5 | controlled regularization and denominator diagnostics work | source-term test | research |
| 7 | exact uniform-flow case | apps/test config, solver harness | tasks 1-6 | recovers `psi1=vbar*x2`, `psi2=x3` up to gauge | uniform flow executable | engineering |
| 8 | homogeneous linear solver | PCG/MG wrapper for `A` | tasks 1-2, 7 | solves `S=0`, restores gauges | linear solve test | engineering |
| 9 | Picard without acceleration | nonlinear solver module | tasks 6, 8 | Picard reduces `r_F` on small cases | Picard smoke | research |
| 10 | adaptive relaxation | nonlinear solver module | task 9 | rejects worsening steps and logs omega | Picard relaxation test | engineering |
| 11 | continuation in `eta` | solver orchestration/config | task 10 | staged `eta` path reaches 1 on small cases | continuation smoke | research |
| 12 | continuation in `lambda` | stochastic/config/orchestration | task 11 | reuses previous solution and records stages | Gaussian small run | research |
| 13 | grid continuation | solver orchestration, interpolation/prolongation | task 12 | `16^3 -> 64^3` reproducible with metrics | grid ladder | research |
| 14 | CPU/GPU comparison | CPU reference + GPU kernels | tasks 1-13 | differences within tolerance on small grids | comparison test | engineering |
| 15 | Anderson acceleration | nonlinear solver module | stable Picard | improves or safely disables when unstable | Anderson benchmark | research |
| 16 | coupled matrix-free residual | residual module | tasks 6, 9 | `F(Psi)` is validated by finite differences | residual test | engineering |
| 17 | FGMRES/Newton-Krylov | solver module, possible PETSc or local Krylov | task 16 | Newton step works on small cases | Newton smoke | research |
| 18 | block preconditioner | PCG/MG adapter for two blocks | task 17 and MG verification | `P^{-1}` reduces Krylov iterations | preconditioner benchmark | engineering |
| 19 | invariant-consumer integration | `PsptaInvariantField`, possible `PsptaEngine` adapter or replacement, pipeline config | accepted invariants | a transport consumer uses equation (14) invariants with diagnostics | small transport smoke | engineering + science |
| 20 | scaling and memory analysis | apps/benchmarks, docs/experiments | GPU implementation | memory footprint and runtime recorded for `128^3+` | V100 benchmark | engineering |

## Recording requirements

For every equation (14) experiment, create a note in `docs/experiments/` or update a dedicated run log with:

- question and hypothesis;
- grid, covariance model, smoothing if any;
- exact config and command;
- build directory and commit;
- residuals and physical metrics above;
- denominator percentiles;
- convergence / failure interpretation;
- whether the result is confirmed, provisional, or an open question.

## Open questions

- Which gauge and boundary formulation is best for throughflow with transverse periodicity?
- Can the existing MG hierarchy be mathematically reused for `q=1/k` without changing coarsening?
- Should `psi1`, `psi2` be stored as double for construction and cast only for PSPTA consumption?
- What is the correct discrete compatibility between CompactMAC Darcy velocity and cell-centered streamfunction gradients?
- How should multi-valued / affine-periodic mean parts be represented in periodic benchmarks?
- What denominator regularization path is scientifically acceptable near stagnation or near-degeneracy?
