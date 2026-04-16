# Operational Plan for PSPTA Integration in MacroFlow3D

## Goal

We are turning MacroFlow3D into a **scientific-grade HPC application** capable of estimating asymptotic macrodispersion coefficients in 3D porous media using a **pseudo-symplectic particle transport method** that is consistent with the kinematic constraints of smooth, locally isotropic Darcy flow.

The target regime is the one emphasized by Lester: in smooth, locally isotropic, helicity-free Darcy flow, the velocity field admits **two streamline invariants** (\psi_1,\psi_2), trajectories are confined to streamsurfaces, and purely advective transverse macrodispersion should not appear. The implementation must therefore preserve these invariants well enough that any measured transverse spreading is not a numerical artifact.

Our chosen route is:

1. **recover two numerical invariants (\psi_1,\psi_2)** from the velocity field by solving a smooth near-nullspace problem for the transport operator;
2. **optionally refine** these invariants to reduce the mismatch between the velocity field and the cross-product representation (v \approx \nabla \psi_1 \times \nabla \psi_2);
3. **feed the resulting invariants into PSPTA** so that particle transport follows the intended streamsurface structure rather than crossing it spuriously.

---

## Guiding principle

We are **not** going to solve the coupled nonlinear elliptic streamfunction system directly. The report explicitly recommends replacing that path with a scalable discrete operator formulation centered on the invariant condition (v \cdot \nabla \psi \approx 0), and then using a refinement stage only if necessary.

This gives us a route that is:

* mathematically aligned with the invariant structure we need,
* compatible with large structured 3D grids,
* GPU-friendly,
* and realistic for integration into the existing HPC pipeline.

---

## What we are going to build

## 1. Recover two invariants from the discrete transport operator

We start from the computed Darcy velocity field (v) on the structured grid and define a scalar transport operator

[
D\psi \approx v \cdot \nabla \psi
]

We then build a symmetric regularized operator

[
A = D^\top W D + \mu L
]

where:

* (D^\top W D) penalizes violation of the invariant condition,
* (L) is a Laplacian-type smoothness regularizer,
* (\mu > 0) suppresses noisy or checkerboard modes,
* and the constant trivial mode is removed explicitly.

We then compute the **two smallest nontrivial eigenvectors** of (A). These become the initial candidates for (\psi_1) and (\psi_2). The report states this directly as the recommended MVP path and describes the exact sequence: build (A), deflate the constant mode, solve for the two smallest nontrivial eigenvectors, and treat those as the two invariants.

### Why this is the correct first step

This directly targets the defining condition of a first integral:

[
v \cdot \nabla \psi \approx 0
]

instead of trying to reconstruct streamfunctions by solving a more fragile nonlinear PDE system from scratch. In the report, this is the main recommended route because it scales as sparse linear algebra on stencil operators and is naturally suited to GPU execution with AMG-preconditioned eigensolvers.

---

## 2. Measure invariant quality immediately after the eigensolve

Once (\psi_1,\psi_2) are recovered, we do **not** trust them blindly. We evaluate them with three diagnostics:

[
r_i = \frac{|v \cdot \nabla \psi_i|}{|v| , |\nabla \psi_i| + \varepsilon}
]

[
e_x = \frac{|v - \nabla \psi_1 \times \nabla \psi_2|}{|v| + \varepsilon}
]

[
s = \frac{|\nabla \psi_1 \times \nabla \psi_2|}{|\nabla \psi_1| , |\nabla \psi_2| + \varepsilon}
]

These mean, respectively:

* **invariance residual**: how well each field behaves like a streamline invariant;
* **cross-product mismatch**: how close the pair is to reproducing the velocity field;
* **independence/non-degeneracy score**: whether the two gradients are meaningfully distinct and non-parallel.

These metrics are not optional. They are the first scientific gate. If the invariance is poor, or if the gradients are nearly parallel or degenerate over too much of the domain, we do not proceed to transport as if the problem were solved.

---

## 3. Refine the invariants only if necessary

If the eigensolver output is already good enough, we keep it.

If the invariance is acceptable but the cross-product mismatch is still too large, we run a **refinement stage**. This refinement is not a separate conceptual route; it is a follow-up step on top of the initial invariant recovery.

The refinement works by alternating between the two fields:

* fix (\psi_2), compute the best target gradient for (\psi_1) that improves the local match to (v);
* project that target gradient back to an integrable scalar field through a Poisson solve;
* then do the same with roles swapped.

In practice, the report describes this as a loop of:

1. local best-fit gradient computation,
2. Poisson projection,
3. update,
4. quality re-evaluation,
5. stop when improvements fall below tolerance.

The key point is that this turns the difficult nonlinear coupling into a sequence of:

* local voxelwise fits,
* Poisson solves,
* gradient updates,
* and line-search / quality checks.

### Important implementation note

The current codebase already contains the **skeleton** of this refinement stage, including the conceptual algorithm, data structures, quality reports, backtracking, and gauge reapplication hooks — but it is still marked as **not implemented** in the actual `.cu` path. That means this is already the intended direction of the project, and the work now is to finish and validate it properly rather than invent a new architecture.

---

## 4. Integrate the recovered invariants into PSPTA

Once (\psi_1,\psi_2) are accepted, they become the invariant field consumed by the pseudo-symplectic transport engine.

This is the whole reason for the construction: PSPTA must move particles in a way that preserves the invariant structure of the flow, instead of letting interpolation or time integration drift particles across streamsurfaces. Lester’s paper is explicit that conventional reconstruction/integration methods can generate spurious transverse macrodispersion by violating these kinematic constraints, and that a pseudo-symplectic approach is precisely about preserving the invariants along trajectories.

So the transport side must be treated as an **invariant-preserving consumer** of (\psi_1,\psi_2), not as an unrelated advection routine.

---

## 5. Build and validate the solver stack on the V100 server

The eigensolver route requires a serious sparse linear algebra backend. The report recommends a GPU-oriented stack based on AMG-preconditioned sparse solves and explicitly identifies PETSc/SLEPc-compatible workflows as practical for this formulation.

The current codebase is already wired for that:

* PETSc/SLEPc is optional but supported,
* the build expects static installations in `src/external/petsc` and `src/external/slepc`,
* there is a validation executable for the eigensolver,
* and the code already distinguishes a validation backend and a production backend for the same mathematical operator.

So the work plan includes:

1. install PETSc/SLEPc correctly on the V100 server;
2. build the project with PETSc enabled;
3. run the smoke and validation executables;
4. use the eigensolver route as the authoritative path for invariant recovery.

---

## Execution phases

## Phase 1 — Infrastructure and solver bring-up

Purpose: make the invariant-recovery path executable end to end.

Tasks:

* install and validate PETSc/SLEPc on the V100 environment;
* ensure CMake picks them up cleanly;
* build the project with the eigensolver backend enabled;
* pass operator tests, PETSc smoke tests, and the SLEPc validation executable;
* confirm the GPU path is actually exercised.

Deliverable:

* a reproducible remote build and validation workflow for the invariant solver stack.

---

## Phase 2 — Initial invariant recovery

Purpose: recover (\psi_1,\psi_2) from the current Darcy velocity field.

Tasks:

* construct the discrete transport operator (D);
* construct the regularized operator (A = D^\top W D + \mu L);
* remove the constant null mode;
* solve for the two smallest nontrivial eigenvectors;
* store them in the project invariant field structure.

Deliverable:

* an initial pair of invariants for PSPTA, plus quality metrics.

---

## Phase 3 — Quality diagnostics and acceptance

Purpose: determine whether the initial pair is already usable.

Tasks:

* compute (r_1, r_2, e_x, s);
* inspect invariance loss, degeneracy zones, and mismatch patterns;
* verify that the fields are smooth enough, independent enough, and physically plausible;
* decide whether refinement is needed.

Deliverable:

* a quantitative report saying whether the eigensolver output is good enough for transport.

---

## Phase 4 — Refinement

Purpose: reduce mismatch while preserving or improving invariant quality.

Tasks:

* implement the alternating fit + Poisson projection loop;
* run backtracking line search;
* reapply gauge fixing after accepted updates;
* monitor monotonic improvement in quality;
* stop when improvement stalls or tolerances are met.

Deliverable:

* refined invariants with improved velocity reconstruction quality and no collapse of independence.

---

## Phase 5 — PSPTA integration

Purpose: use the accepted invariants in transport.

Tasks:

* feed (\psi_1,\psi_2) into the PSPTA invariant field;
* ensure the engine uses them consistently in its step/update logic;
* keep the hot path allocation-free;
* expose particle status, invariant quality, and Newton-type failure statistics.

Deliverable:

* a working PSPTA transport path driven by numerically recovered invariants.

---

## Phase 6 — Scientific validation

Purpose: show that the method improves the physically relevant behavior.

Tasks:

* run the small PSPTA case first;
* then move to larger controlled runs;
* compare against the existing transport baseline;
* evaluate whether transverse spreading in the smooth isotropic advective regime is reduced;
* treat any nonzero transverse macrodispersion with skepticism until shown not to be numerical.

This phase matters because the scientific target is not merely “recover two scalar fields,” but to recover invariants well enough that PSPTA suppresses the spurious transverse spreading mechanism that conventional methods can introduce.

Deliverable:

* evidence that the new path is numerically better aligned with the intended helicity-free invariant structure.

---

## Practical implementation rules

### Build order

We always proceed in this order:

1. build infrastructure,
2. operator/eigensolver correctness,
3. invariant quality,
4. refinement,
5. PSPTA transport integration,
6. scientific validation.

No skipping.

### Acceptance rule

We do not call the invariant recovery “done” because the eigenproblem converged.
It is only acceptable when:

* (v \cdot \nabla \psi_i) is small enough,
* the cross-product mismatch is controlled,
* the two gradients remain independent,
* and PSPTA behavior improves in the expected physical regime.

### What we are not doing

We are not:

* solving the coupled nonlinear streamfunction PDEs directly;
* switching to a different primary approach midstream;
* treating transport and invariant construction as unrelated modules;
* claiming scientific success based only on build/test success.

---

## Short operational summary

The plan is simple:

* compute two numerical invariants from the velocity field by solving a regularized transport near-nullspace problem;
* measure their quality immediately;
* refine them only if the initial pair is not good enough;
* use the resulting (\psi_1,\psi_2) inside PSPTA so particle motion respects the streamsurface structure;
* validate the whole path on the V100-backed HPC stack until the method behaves like a scientifically credible invariant-preserving transport scheme.
