# Estimating Two Independent Streamline Invariants (ψ₁, ψ₂) on Large 3D Structured Grids Without Solving Coupled Nonlinear Elliptic PDEs

## Archive status - 2026-07-13

This report is archived historical research for the old transport-near-nullspace / Strategy A+C direction.

It is not an active implementation plan. New invariant construction must follow:

- `docs/plans/active/lester-eq14-streamfunction-solver-plan.md`
- `docs/decisions/2026-07-13-lester-eq14-streamfunction-solver.md`

Do not use this report to reject or bypass the Lester equation (14) streamfunction solver direction.

## Problem reformulation

You have a steady 3D Darcy velocity field **v(x)** sampled on a structured grid *(Nₓ×Nᵧ×N_z)* (cell-centered or face-centered). The target is to reconstruct **two scalar fields** *(ψ₁, ψ₂)* such that:

- **Streamline invariance:**
  \[
  \mathbf v\cdot \nabla \psi_i \approx 0,\quad i\in\{1,2\}
  \]
- **Euler/Clebsch (Euler potentials) representation:**
  \[
  \mathbf v \approx \nabla \psi_1 \times \nabla \psi_2
  \]
- **Non-degeneracy and independence:** gradients should not vanish and should not be parallel too often:
  \[
  \|\nabla\psi_i\|>0\ \text{a.e.},\qquad \|\nabla\psi_1\times \nabla\psi_2\|>0\ \text{a.e.}
  \]

The base article you provided states exactly these invariance relations and the cross-product representation, and also describes an alternative formulation in terms of **coupled nonlinear elliptic PDEs** for *(ψ₁, ψ₂)* (attributed to Zijl-type streamfunction methods) whose direct solution is expensive and algorithmically delicate; this is the formulation we want to avoid. fileciteturn0file0

### Geometry and dynamical-systems view

If a divergence-free vector field admits a **Clebsch/Euler potentials** representation
\[
\boldsymbol\omega(x)=\nabla\alpha(x)\times\nabla\beta(x),
\]
then \(\alpha,\beta\) are **constants of motion** along trajectories of \(\dot x=\boldsymbol\omega(x)\), i.e. \(d\alpha/dt=0\) and \(d\beta/dt=0\), and trajectories lie on intersections of the two level surfaces (integral surfaces). citeturn31view2
This is the exact template of your target constraints with \(\boldsymbol\omega\equiv \mathbf v\), \(\alpha\equiv \psi_1\), \(\beta\equiv \psi_2\).

However, this representation is **not globally available for arbitrary 3D fields**. Yoshida explicitly highlights **incompleteness** of Clebsch representations (and links this to helicity constraints and topology), and also notes the need for care with boundary conditions and (possibly) multi-valued potentials. citeturn25view0turn31view2
In practice, even if the “physics” suggests integrability (e.g., isotropic Darcy), **discretization and interpolation errors** can break the exact structure, so the computational target should be: **approximately invariant**, **smooth**, **numerically stable**, **GPU-parallel** invariants.

### What must be preserved vs. what can be relaxed

**Must preserve (core physics/geometry):**
- Near-invariance: \( \mathbf v\cdot\nabla \psi_i \approx 0 \) globally.
- Two degrees of freedom: at least two scalar labels that separate streamline families.
- Robustness to the discrete nature of **v** (tabulated, noisy, not analytic).

**Can be relaxed (numerically):**
- Exact equality \( \mathbf v = \nabla\psi_1\times\nabla\psi_2 \) can be *weakened* to directional alignment plus a calibration step (or used as a refinement constraint).
- Global smooth single-valued potentials may be replaced by: (i) potentials with branch cuts, (ii) patchwise potentials, or (iii) weakly regularized approximate invariants.
- Independence can be enforced “on average” (global criteria), not necessarily pointwise.

A key contextual anchor is that steady isotropic Darcy flow is argued (in the dynamical-systems sense) to be **fully integrable** and to admit **two analytic constants of motion** corresponding to a pair of streamfunctions; this precludes chaotic advection and implies a constrained streamline topology. citeturn32view0
That is the theoretical justification for expecting a **two-dimensional nullspace** (or near-nullspace) of the discrete transport operator \(D(\cdot)=\mathbf v\cdot\nabla(\cdot)\).

## Candidate method landscape

The table below focuses on methods that **avoid** directly solving the *coupled nonlinear elliptic* system for \((\psi_1,\psi_2)\) by replacing it with: (i) characteristic transport, (ii) linear(ized) subproblems, (iii) constrained least-squares / eigenproblems, or (iv) operator-splitting + projection.

**Score tuple** format (1–5):
**[Fidelity, Invariants, Implementation, GPU, Cost/Mem, Robustness, Reusable code, Large-grid suitability]**

| Method / family | Core idea | What it solves | Avoids coupled nonlinear elliptic PDE? | GPU parallelization reality | Key risks / limitations | Structured 3D grid fit | Score tuple |
|---|---|---|---|---|---|---|---|
| **Nullspace / eigenvectors of transport residual** (recommended) | Find two smooth fields that minimize \(\| \mathbf v\cdot\nabla\psi\|^2\) under constraints; compute two smallest eigenvectors / singular vectors | Linear algebra on sparse stencil operators: \(A\psi=\lambda\psi\) with \(A\approx (v\cdot\nabla)^T(v\cdot\nabla)+\mu L\) | Yes (replaced by SPD eigenproblems + optional Poisson solves) | Excellent: SpMV + multigrid preconditioners; GPU AMG available (hypre, AmgX, PETSc) citeturn21search1turn21search6turn21search0 | Needs careful constraint handling to avoid trivial constants; near-stagnation regions can create degeneracy; eigen solve tuning | Excellent | **[4, 5, 4, 5, 4, 4, 4, 5]** |
| **Fourier/SVD approximate first integrals** (Haller group) | Expand \(H\) in Fourier basis; build matrix \(C h \approx (\nabla H\cdot v)\) on grid; take smallest right-singular vectors | Homogeneous least-squares / SVD on Fourier coefficients (grid-sampled) citeturn27view0turn26view0 | Yes (linear LS/SVD) | Good if FFT + randomized/SVD; but naive dense \(C\) can be memory-heavy; still conceptually strong | Assumes periodic or interior region away from boundaries; Fourier truncation; challenging at very large grids without matrix-free tricks | Good (periodic best) | **[4, 4, 3, 4, 3, 3, 2, 3]** |
| **Characteristic backtracing / semi-Lagrangian invariants** | Solve \(v\cdot\nabla\psi=0\) via characteristics: map each voxel back to a reference surface; set ψ labels from boundary coordinates | ODE integration (streamline tracing) + interpolation; (optionally) semi-Lagrangian advection methodology citeturn34search6turn33search1 | Yes | High parallelism (one trajectory per voxel) but bandwidth + branching; needs fast trilinear interpolation (texture or custom) | Needs global cross-section condition (no recirculation across the chosen reference surface); error accumulation; handling periodicity | Good | **[4, 5, 4, 4, 3, 3, 4, 4]** |
| **Linear advection–diffusion regularization** | Replace \(v\cdot\nabla\psi=0\) with \(v\cdot\nabla\psi-\varepsilon\Delta\psi=0\) to get well-posed linear problems for two BCs | Linear nonsymmetric PDE (Krylov + AMG) | Yes | Very GPU-friendly if AMG/Krylov used; hypre/PETSc/AmgX support citeturn21search1turn21search25turn21search6 | Picks a diffusion scale ε; may blur invariants; not guaranteed to yield two independent invariants globally | Excellent | **[3, 4, 4, 5, 4, 4, 5, 5]** |
| **Alternating minimization with Poisson projection** (refinement step) | Alternate: locally fit gradients to satisfy \(v\approx\nabla\psi_1\times\nabla\psi_2\), then project to integrable gradients via Poisson solves | Sequence of *linear* Poisson problems + local 3×3 solves per voxel | Yes | Excellent: local kernels + FFT Poisson (periodic) or AMG Poisson (general BCs) + stencil ops; cuFFT/CuPy/AmgX citeturn21search15turn21search6turn21search3 | Nonconvex overall; needs good initialization to avoid collapse; can stagnate near degenerate gradients | Excellent | **[4, 4, 3, 5, 4, 3, 4, 5]** |
| **Clebsch map / Dirichlet-energy minimization** (graphics) | Compute “Clebsch maps” via nonlinear energy minimization (harmonic-map/Dirichlet-type) to encode a flow field | Nonlinear optimization on meshes/grids (gradient descent/Newton) | Partially | GPU possible (autodiff + PDE ops), but heavier than linear nullspace methods | Nonconvex; may require complex constraints; historically for visualization and vorticity, not Darcy invariants | Moderate | **[3, 3, 2, 4, 2, 2, 2, 2]** |
| **Scientific ML (PINNs / neural operators)** | Learn \(\psi_1,\psi_2\) minimizing physics losses: \(\|v\cdot\nabla\psi_i\|^2+\|v-\nabla\psi_1\times\nabla\psi_2\|^2\) | Nonlinear training on samples; differentiable stencils | Yes | GPU-native, but training cost high; tricky to guarantee constraints everywhere | Generalization and topology issues; hard to enforce non-degeneracy globally; needs careful regularization | Good | **[3, 3, 3, 5, 2, 2, 3, 3]** |

**Most serious candidates for your constraints and GPU needs** are the first five rows. The standout is the **transport-nullspace / eigenvector** formulation because it directly targets the defining invariant condition \(\nabla\psi\cdot v=0\) in a linear-algebraic way (the same condition used to define first integrals). citeturn26view0turn27view0

## Literature synthesis

This section prioritizes works that (i) treat invariants/first integrals in 3D flows, (ii) deal with Clebsch/Euler potentials as *constraints and topology*, and (iii) provide computationally plausible surrogates for large grids.

### Steady isotropic Darcy flow admits two invariants

A foundational claim for your project is that steady isotropic Darcy flow is **fully integrable** and admits **two analytic constants of motion** (streamfunctions) that constrain streamline topology and preclude chaotic advection. citeturn32view0
This provides the mathematical “license” to look for **two** independent numerical invariants as a low-dimensional structure. It also suggests that if a numerical method is “good enough,” the two-invariant structure should appear as a **near-nullspace** of the discrete advection operator.

### Clebsch/Euler potentials: integrability, but global obstacles

In Yoshida’s treatment of Clebsch parameterization, a key point is that when a divergence-free vector field is representable as a **Clebsch 2-form**
\[
\omega=\nabla\alpha\times\nabla\beta,
\]
then \(\alpha,\beta\) are constants along trajectories and define integral surfaces; this is precisely your invariant condition. citeturn31view2
But Yoshida also emphasizes **nontrivial global issues** (incompleteness and non-uniqueness) and links feasibility to helicity/topology constraints, motivating the practical expectation of singular sets or multi-valuedness in general. citeturn25view0turn31view2
For Darcy flows this is encouraging: the physics argues you are in the “integrable” regime; numerically, you should still design algorithms that tolerate near-singular behavior (stagnation points, near-parallel gradients).

### Approximate first integrals via linear least squares on grid data

A highly transferable approach comes from work on *approximate streamsurfaces* and the construction of approximate first integrals from discrete velocity data. They define a first integral \(H\) by the condition
\[
\nabla H\cdot v = 0,
\]
and then relax it by minimizing
\[
J[H] = \frac12\int_U |\nabla H\cdot v|^2\,dV.
\]
citeturn26view0
Crucially for your use case, they show that when \(H\) is expanded in a Fourier basis and the field is sampled on a 3D grid, the discretized condition becomes a **linear system** \(C h\) representing the pointwise inner products \((\nabla H\cdot v)\) over all grid points; the solution can be taken from the **right singular vectors** of \(C\) (smallest singular values), i.e., solving a homogeneous least-squares problem. citeturn27view0turn27view2
While their paper focuses on constructing *one* approximate integral for visualization (and especially in elliptic regions), their linear-algebra formulation naturally generalizes to extracting a **basis** of near-null vectors—exactly what you need for two invariants.

### Using weaker invariant-surface notions as robustness tools

A recurring theme in the vortex-surface field (VSF) literature is that globally smooth Clebsch potentials are often only available for highly symmetric flows, and that **weaker criteria** (constructing scalar fields whose isosurfaces are approximately invariant surfaces) can be more robust in general fields. This is explicitly stated in a JFM work on tracking vortex surfaces: global smooth Clebsch potentials (up to isolated singularities) are “only successful in very few highly symmetric flows,” whereas VSF-type constructions can be approximate with small deviation in general 3D flows. citeturn17view0
For your problem, this motivates designing your algorithms with (a) regularization, (b) tolerance for localized violations, and (c) detection/handling of singular sets.

### Characteristic transport and semi-Lagrangian methods as a direct “PDE-avoidance” route

Solving \(v\cdot\nabla\psi=0\) by characteristics is a classical way to avoid elliptic PDE solves: you propagate labels along streamlines. Semi-Lagrangian methods are a mature computational framework for advection problems and are widely discussed as accurate and efficient for multi-dimensional flows. citeturn34search6turn34search30
In groundwater modeling specifically, method-of-characteristics ideas are also widely used for transport; MOC3D, for example, is explicitly integrated with MODFLOW to simulate 3D transport using characteristic/particle-based ideas. citeturn34search35
These are directly adaptable to constructing invariants: “backtrace to a reference surface; assign coordinates as invariants,” provided the flow admits a global section (e.g., a consistent throughflow direction).

## Open-source building blocks

The practical requirement is: **large 3D grids + GPU throughput**, which usually means (i) stencil operators, (ii) FFT-based Poisson if periodic, otherwise AMG, and (iii) matrix-free iterative solvers.

### GPU-capable linear solvers and multigrid

- **hypre / BoomerAMG**: supports GPU backends and provides GPU-enabled configuration options; key guidance is that most options are GPU-enabled but non-GPU options may require unified memory or CPU fallback. citeturn21search1turn21search21turn21search5
- **NVIDIA AmgX**: positioned as a GPU-accelerated algebraic multigrid and Krylov solver library designed for large sparse systems on GPUs, with open-source availability via GitHub. citeturn21search2turn21search6
- **PETSc GPU support**: PETSc explicitly lists CUDA/HIP/Kokkos GPU features and documents a GPU roadmap. citeturn21search0turn21search16
  PETSc also documents running hypre BoomerAMG on GPUs through its hypre interface (PCHYPRE), using GPU vectors/matrices so hypre’s GPU solvers are used automatically. citeturn21search25

**Why these matter:** your recommended MVP method reduces to **SPD eigenproblems and Poisson-like sub-solves**, where AMG preconditioning is usually decisive for performance and scalability.

### FFT backends for periodic Poisson and spectral regularization

- **cuFFT** is NVIDIA’s FFT library designed for high performance on NVIDIA GPUs. citeturn21search15
- **CuPy** provides GPU arrays and FFT interfaces that generate cuFFT plans internally and exposes sparse linear algebra routines similar to SciPy. citeturn21search7turn21search3turn35search14

If your domain is triply periodic (common in benchmark porous-media / random conductivity setups), FFT-based Poisson solves (and even spectral filtering) can be extremely attractive.

### Streamline/trajectory utilities and baseline CPU implementations

- **VTK** provides canonical streamline tracing algorithms (e.g., vtkStreamTracer) with RK2/RK4/RK45-type integrators and many controls; this is valuable both for validation and for a CPU baseline implementation. citeturn33search1turn35search1
- For vector field topology extraction (critical points/separatrices), VTK also contains tools discussed in the “open source vector field topology” context. citeturn35search5

### GPU kernel development in Python (optional but useful)

- **NVIDIA Warp** is an open-source Python framework that JIT-compiles Python kernels to CPU/GPU code and is intended for high-performance simulation-style kernels. citeturn35search0turn35search4turn35search12

This can accelerate prototyping of custom stencil operators and projection steps without dropping immediately into CUDA C++.

## Proposed computational strategies

Below are **three concrete, iterative strategies**, all designed to avoid the coupled nonlinear elliptic system and to map cleanly to GPU execution.

### Strategy A (recommended): Two invariants as the smooth near-nullspace of the discrete transport operator

#### Variables and operators
Let \( \psi \in \mathbb R^{N} \) be a scalar field on the grid (N = NₓNᵧN_z). Define a discrete operator:
- Gradient: \(G\psi \approx \nabla\psi\) (3 components).
- Transport (directional derivative):
  \[
  D\psi \equiv \mathbf v\cdot \nabla \psi \approx v_x\,\partial_x\psi + v_y\,\partial_y\psi + v_z\,\partial_z\psi.
  \]

Then define a quadratic energy (discrete analogue of \(\int |\nabla\psi\cdot v|^2\)):
\[
E(\psi) = \frac12\|D\psi\|_W^2 + \frac{\mu}{2}\|\nabla\psi\|^2
\]
where:
- \(W\) is a diagonal weight (e.g., cell volumes, or \(W=I\)),
- \(\mu>0\) is a **smoothness regularization** to suppress noisy or checkerboard modes.

This is aligned with the continuous least-squares objective used to build approximate first integrals, \(\int |\nabla H\cdot v|^2\), but adapted to a large-grid, stencil/matrix-free setting. citeturn26view0turn27view0

#### Key computational reformulation
Minimizing \(E(\psi)\) under a normalization constraint gives an eigenproblem:
\[
A\psi = \lambda \psi,\qquad A = D^TWD + \mu L,
\]
where \(L\) is a discrete Laplacian (from the \(\|\nabla\psi\|^2\) term). \(A\) is symmetric positive semidefinite if built consistently.

To obtain **two linearly independent invariants**, compute the **two smallest nontrivial eigenvectors** \(\psi_1,\psi_2\) (excluding the constant/near-constant mode). This is the large-grid analogue of taking multiple smallest right-singular vectors in the Fourier/SVD approach. citeturn27view0turn27view2

#### Enforcing independence and avoiding degeneracy
- Remove trivial constant solutions by imposing:
  - mean-zero: \(\langle \psi\rangle = 0\),
  - and/or a constraint like \(\|\psi\|_2=1\).
- Enforce independence by:
  - orthogonality: \(\langle \psi_1,\psi_2\rangle = 0\),
  - and checking \(\|\nabla\psi_1\times\nabla\psi_2\|\) statistics.

#### How to check the physics constraints
Compute on-grid diagnostics:
- Invariance residuals:
  \[
  r_i(x)=\frac{|\,\mathbf v\cdot\nabla\psi_i\,|}{\|\mathbf v\|\,\|\nabla\psi_i\|+\epsilon}
  \]
- Cross-product mismatch:
  \[
  e_{\times}(x)=\frac{\|\mathbf v - \nabla\psi_1\times\nabla\psi_2\|}{\|\mathbf v\|+\epsilon}
  \]
- Degeneracy/independence:
  \[
  s(x)=\frac{\|\nabla\psi_1\times\nabla\psi_2\|}{\|\nabla\psi_1\|\,\|\nabla\psi_2\|+\epsilon}
  \]

Even if \(\mathbf v\neq \nabla\psi_1\times\nabla\psi_2\) exactly at first, invariance plus independence implies the cross product should be **directionally aligned** with v; mismatch can then be reduced with Strategy C.

#### GPU viability
This strategy is dominated by:
- stencil gradient/divergence kernels,
- Sparse matrix–vector products (or matrix-free operator application),
- eigen-iterations (LOBPCG/Lanczos-type) with AMG preconditioning.

GPU AMG and GPU linear algebra support are mature in hypre and AmgX, and PETSc provides GPU backends and can route to hypre BoomerAMG on GPUs. citeturn21search1turn21search6turn21search25turn21search0

---

### Strategy B (backup): Streamline-coordinate reconstruction via characteristic backtracing to a global section

This is the most literal implementation of \(v\cdot\nabla\psi=0\): **ψ is constant along streamlines**, so assign each streamline a pair of labels from where it pierces a reference surface.

#### Assumption
There exists a global “section” surface \(\Sigma\) intersecting each streamline once per traversal (e.g., if \(v_1>0\) everywhere, use the plane \(x_1=0\) in a periodic domain).

#### Algorithmic definition
For each grid point \(x\), integrate the backwards characteristic:
\[
\frac{dX}{d\tau} = -\mathbf v(X),\qquad X(0)=x
\]
until \(X(\tau^*)\in\Sigma\). Then set:
\[
\psi_1(x)=\ell_1(X(\tau^*)),\qquad \psi_2(x)=\ell_2(X(\tau^*)),
\]
where \(\ell_1,\ell_2\) are two independent label functions on the section (e.g., \(\ell_1=y,\ell_2=z\)). This inherits the semi-Lagrangian / characteristic viewpoint used broadly in advection problems. citeturn34search6turn34search30

#### GPU viability
Embarrassingly parallel: one trajectory per voxel. The main performance bottleneck is **trilinear interpolation** of v (and later of ψ gradients). In CUDA contexts, hardware-accelerated 3D texture interpolation is a known high-throughput option, and there are open implementations of trilinear interpolation kernels. citeturn34search0turn34search8

#### Limitations (why this is “backup”)
- Requires a suitable global section (fails in recirculating/closed streamline regions).
- Backtracing over long distances can accumulate error; needs careful step control.
- Produces invariants, but cross-product magnitude matching may drift (often acceptable if Strategy C is applied as refinement).

---

### Strategy C (refinement): Alternating linear “fit + projection” to enforce \(v\approx\nabla\psi_1\times\nabla\psi_2\)

Use Strategy A or B to get a good initialization \((\psi_1,\psi_2)\), then reduce cross-product mismatch iteratively while keeping invariance approximately true.

#### Objective (discrete)
\[
\min_{\psi_1,\psi_2}\ \sum_x \Big(
\|\mathbf v - \nabla\psi_1\times\nabla\psi_2\|^2
+ \alpha\sum_{i=1}^2 |\mathbf v\cdot\nabla\psi_i|^2
\Big)
+ \beta\sum_{i=1}^2 \|\nabla\psi_i\|^2
\]
with small \(\beta\) stabilizing gradients and \(\alpha\) reinforcing invariance.

#### Alternating step (concept)
For fixed \(\psi_2\), the term \(\nabla\psi_1\times\nabla\psi_2\) is **linear in \(\nabla\psi_1\)** at each voxel, so you can compute a *local* best-fit gradient \(g_1^\star(x)\) by solving a tiny 3×3 least-squares system per voxel, then **project** \(g_1^\star\) to an integrable gradient field by solving:
\[
\psi_1 = \arg\min_\psi \sum_x \|\nabla\psi - g_1^\star\|^2
\quad\Rightarrow\quad
\Delta\psi_1 = \nabla\cdot g_1^\star
\]
(and similarly swapping roles). This turns a nonlinear coupled problem into a sequence of:
- local voxelwise fits (**pure GPU kernels**),
- Poisson solves (**FFT or AMG**),
- and stencil gradient updates.

FFT/GPU support is strong in cuFFT and CuPy’s FFT interface. citeturn21search15turn21search3
For nonperiodic BCs, hypre/PETSc/AmgX are a practical route to GPU Poisson solves. citeturn21search1turn21search6turn21search25

---

### Pseudocode (iterative MVP: Strategy A + optional Strategy C)

```text
Inputs:
  v on structured grid (Nx,Ny,Nz)
  boundary type (periodic or physical BC)
  params mu (smoothness), k (number of invariants=2), tol

Step 0: Preprocess v (optional but recommended)
  - enforce divergence-free as well as feasible (discrete projection or recompute from potential)
  - optionally smooth v slightly to reduce grid noise (low-pass)

Step 1: Define discrete operators (matrix-free preferred)
  Grad(psi): centered or upwind gradient
  Adv(psi): Dpsi = v · Grad(psi)
  Lap(psi): discrete Laplacian

  Define A(psi) = Adv^T(Adv(psi)) + mu * Lap(psi)

Step 2: Compute k=2 smallest nontrivial eigenvectors of A
  - impose constraint mean(psi)=0 (remove constant mode)
  - use LOBPCG / Lanczos with AMG-preconditioned inner solves
  - obtain psi1, psi2, orthonormalize

Step 3: Diagnostics
  r_i = ||v·Grad(psii)|| / (||v|| ||Grad(psii)|| + eps)
  e_x = ||v - Grad(psi1)×Grad(psi2)|| / (||v|| + eps)
  s   = ||Grad(psi1)×Grad(psi2)|| / (||Grad(psi1)|| ||Grad(psi2)|| + eps)

If e_x is acceptable -> return (psi1, psi2)

Optional Step 4 (refinement, Strategy C):
  repeat until convergence:
    - fix psi2, compute best-fit local gradient g1*(x) to match v ≈ g1×Grad(psi2)
    - solve Poisson: Lap(psi1) = Div(g1*)
    - fix psi1, compute best-fit local gradient g2*(x) to match v ≈ Grad(psi1)×g2
    - solve Poisson: Lap(psi2) = Div(g2*)
    - recheck diagnostics, stop if decrease < tol

Return (psi1, psi2)
```

## Recommended MVP approach

### Primary recommendation

**Use Strategy A (transport nullspace / eigenvectors) as the MVP**, optionally followed by **Strategy C (alternating fit + Poisson projection)** as a refinement stage.

**Why this is the strongest practical choice:**
- It targets the defining invariant condition \(\nabla\psi\cdot v=0\) in the same least-squares sense used in the literature on approximate first integrals, but replaces Fourier/SVD with a scalable stencil/eigen approach. citeturn26view0turn27view0
- The existence of two invariants in steady isotropic Darcy flow provides strong justification that there should be a clean two-dimensional structure to recover numerically. citeturn32view0
- It avoids the coupled nonlinear elliptic system entirely, replacing it with **GPU-friendly** sparse linear algebra and AMG preconditioning. citeturn21search1turn21search6turn21search25
- It is naturally “actionable” on tabulated v (no need for analytic expressions).

### Backup recommendation

**Use Strategy B (characteristic backtracing from a global section)** as the backup, because it enforces invariance by construction and is embarrassingly parallel on GPU, leveraging mature semi-Lagrangian/characteristic ideas. citeturn34search6turn34search30
This is especially attractive if your flow has a strong mean-throughflow direction (common in Darcy setups with periodic heterogeneity and imposed mean gradient).

### Suggested GPU technology stack

A robust, HPC-oriented stack for large 3D grids:

- **Core sparse/AMG solvers:** hypre (GPU BoomerAMG) citeturn21search1turn21search21
- **Framework glue and scalability:** PETSc (CUDA/HIP/Kokkos backends; easy routing to hypre GPU) citeturn21search0turn21search25turn21search16
- **Optional NVIDIA-native AMG:** NVIDIA AmgX citeturn21search6turn21search2
- **Periodic Poisson / FFT acceleration:** cuFFT + CuPy FFT stack for rapid prototyping citeturn21search15turn21search3turn21search7
- **Custom GPU kernels (stencils / local fits):** NVIDIA Warp (fast Python-to-GPU kernel path) citeturn35search0turn35search4
- **Verification & visualization:** VTK streamline tools as baseline reference implementations citeturn33search1turn35search1

### Three priority papers to start with

1. Katsanoulis et al., *Approximate streamsurfaces for flow visualization* (constructs approximate first integrals by minimizing \(\int|\nabla H\cdot v|^2\) and solving a linear-algebra problem from grid data). citeturn26view0turn27view0
2. Yoshida, *Clebsch parameterization – theory and applications* (links Clebsch 2-form to integrability and constants of motion; discusses global obstacles and helicity constraints). citeturn31view2turn25view0
3. Lester et al., *The Lagrangian kinematics of three-dimensional Darcy flow* (states isotropic steady Darcy flow admits two analytic constants/streamfunctions and is fully integrable). citeturn32view0

### Three priority repositories to start with

1. **hypre-space/hypre** (GPU-capable AMG, core preconditioner for large sparse problems). citeturn21search1turn21search5
2. **petsc/petsc** (GPU support + interfaces to hypre; production-grade linear algebra for eigen/Poisson workflows). citeturn21search0turn21search25turn21search16
3. **NVIDIA/AMGX** (GPU AMG and Krylov solvers; alternative path for very fast Poisson-like solves). citeturn21search6turn21search2

## Implementation roadmap

### Small-scale prototype

Goal: validate mathematics and diagnostics on ~\(64^3\) to \(128^3\).

- Implement Strategy A with matrix-free stencils on CPU (NumPy/SciPy), verify:
  - invariance residual distributions \(r_i(x)\),
  - cross-product directional alignment,
  - sensitivity to μ and discretization choice (upwind vs. centered in D).
- Cross-check with CPU streamline tracing (VTK) for sanity: streamlines should lie on intersections of ψ-level surfaces. citeturn33search1turn35search1

Deliverables:
- unit tests on synthetic integrable fields (e.g., known integrable flows) + on your Darcy data,
- plots/histograms of residuals.

### Medium-scale scaling

Goal: \(256^3\) with accelerated solvers (single GPU or multi-GPU if needed).

- Move to PETSc + hypre setup; use PETSc GPU backends and route to hypre BoomerAMG where applicable. citeturn21search25turn21search0
- Implement LOBPCG/Lanczos-like eigen iteration with AMG preconditioning (or solve shifted systems for inverse iteration).
- Add optional Strategy C refinement:
  - use FFT Poisson if periodic (CuPy+cuFFT), else AMG Poisson.

Deliverables:
- timing breakdown: stencil ops vs AMG vs eigen iterations,
- memory footprint estimates for matrix-free vs explicit sparse storage.

### Large-grid GPU production version

Goal: \(512^3\)–\(1024^3\) depending on resources.

- Adopt matrix-free operator evaluation (avoid storing full sparse matrices).
- Use hypre GPU AMG and/or AmgX depending on environment constraints. citeturn21search1turn21search6
- Use mixed precision where feasible (AMG smoothers and SpMV) but keep dot products and orthogonalization stable.

Validation gates:
- Track convergence of eigenvalues (near-zero modes) and stability of \(\psi_1,\psi_2\) under grid refinement.
- Detect problematic regions (near-stagnation or near-parallel gradients) and mark them for masking or local regularization.

## Final required items

### Primary recommendation
**Transport-operator near-nullspace method (Strategy A) + optional Poisson-projection refinement (Strategy C)**, because it replaces the coupled nonlinear elliptic problem with scalable linear algebra and is naturally GPU-ready via AMG-preconditioned iterations. citeturn26view0turn21search1turn21search6

### Proposed iterative algorithm (pseudocode)
Included above under *“Pseudocode (iterative MVP: Strategy A + optional Strategy C)”*.

### Suggested GPU stack
- PETSc (CUDA/HIP/Kokkos) + hypre BoomerAMG GPU + (optional) NVIDIA AmgX
  citeturn21search0turn21search25turn21search1turn21search6
- cuFFT + CuPy FFT/sparse linalg for prototypes and periodic Poisson citeturn21search15turn21search3turn21search7
- NVIDIA Warp for custom stencil/local kernels citeturn35search0turn35search4

### Three priority papers
- Katsanoulis et al. (approximate first integrals via least squares) citeturn26view0turn27view0
- Yoshida (Clebsch 2-form ⇒ integrability; global obstacles) citeturn31view2turn25view0
- Lester et al. (isotropic Darcy ⇒ two invariants; integrable) citeturn32view0

### Three priority repositories
- hypre-space/hypre citeturn21search1turn21search5
- petsc/petsc citeturn21search0turn21search16turn21search25
- NVIDIA/AMGX citeturn21search6turn21search2

### Key mathematical and numerical risks

- **Global existence / topology:** Clebsch-type representations can fail globally for generic fields and may require multi-valued potentials or singular sets; this is a known structural issue in Clebsch theory. citeturn25view0turn31view2
- **Discrete v not exactly structure-preserving:** grid interpolation can introduce divergence or break integrability; the two-dimensional nullspace becomes “fuzzy,” and eigenvectors may mix or localize. (Motivation: practical difficulty of globally smooth Clebsch potentials outside symmetric cases.) citeturn17view0
- **Stagnation/near-stagnation regions:** if \(\|\mathbf v\|\approx 0\), invariants become ill-conditioned; gradients may degenerate, causing numerical instability.
- **Boundary conditions:** Fourier/FFT strategies are simplest for periodic domains; physical BCs require careful discretization and AMG robustness. citeturn27view0turn21search1
- **Eigen-solver convergence:** computing the smallest two eigenmodes at large scale needs good preconditioning and stable orthogonalization; otherwise iteration count can be prohibitive. GPU AMG support exists but requires parameter tuning. citeturn21search1turn21search25
- **Independence collapse:** without explicit constraints, ψ₁ and ψ₂ may become functionally dependent (nearly collinear gradients) in parts of the domain; requires monitoring \(\|\nabla\psi_1\times\nabla\psi_2\|\) and adding regularization/deflation.
- **Refinement nonconvexity (Strategy C):** alternating schemes can converge to poor local minima if initialization is weak; Strategy A is intended to provide that initialization.
