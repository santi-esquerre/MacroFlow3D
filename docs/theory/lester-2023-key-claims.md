# Lester 2023 — Scientific theory notes for MacroFlow3D

## Reference

Daniel R. Lester, Marco Dentz, Prajwal Singh, and Aditya Bandopadhyay.
**Under What Conditions Does Transverse Macrodispersion Exist in Groundwater Flow?**
*Water Resources Research*, 59, e2022WR033059, 2023.

---

## Why this paper matters for this repository

This paper is the main scientific basis for the current PSPTA / invariant-preserving direction of MacroFlow3D.

Its core claim is not merely that some numerical methods are inaccurate. It is stronger:

- for **steady 3D Darcy flow with smooth, locally isotropic scalar conductivity** and **no stagnation points**, the flow is **helicity-free**,
- such flows admit **two invariants / streamfunctions**,
- those invariants constrain trajectories to 2D streamsurfaces,
- therefore **purely advective transverse macrodispersion is zero** in that regime,
- and conventional particle-tracking methods can produce **spurious positive transverse macrodispersion** if they do not preserve those kinematic constraints.

This means that, for MacroFlow3D, positive transverse spreading in the purely advective smooth-isotropic Darcy regime must be treated as **suspicious by default**, not as automatically physical.

---

## Core scientific claims

## 1. Smooth, locally isotropic Darcy flow is helicity-free

For isotropic Darcy flow with scalar conductivity `k(x)`,

```math
\mathbf{v}(x) = -k(x)\nabla \phi(x),
```

the helicity density

```math
h(x) = \mathbf{v}(x) \cdot (\nabla \times \mathbf{v}(x))
```

vanishes identically in the smooth case.

Interpretation:
- the flow is geometrically constrained,
- streamline topology is not generic 3D “free wandering” topology,
- 3D intuition based on arbitrary divergence-free fields does **not** apply automatically.

For this project, this is a **hard scientific constraint**, not a cosmetic detail.

---

## 2. Helicity-free steady 3D flows admit two invariants

The paper states that steady 3D helicity-free flows admit two invariants / streamfunctions `ψ1(x), ψ2(x)` satisfying

```math
\mathbf{v}(x) \cdot \nabla \psi_1(x) = 0,
\qquad
\mathbf{v}(x) \cdot \nabla \psi_2(x) = 0.
```

These are constants of motion along streamlines.

Interpretation:
- trajectories are confined to intersections of the level sets of `ψ1` and `ψ2`,
- streamlines lie on 2D streamsurfaces,
- streamline motion is effectively constrained in the same essential sense that forbids unbounded transverse wandering.

This is the conceptual bridge between:
- helicity-free Darcy flow,
- integrability,
- and the absence of purely advective transverse macrodispersion.

---

## 3. Euler-potential / dual-streamfunction representation

The same paper gives the velocity representation

```math
\mathbf{v}(x) = \nabla \psi_1(x) \times \nabla \psi_2(x),
```

with non-vanishing gradients away from degeneracies / stagnation issues.

Interpretation for MacroFlow3D:
- a correct numerical method should preserve, or at least respect, this invariant geometry,
- diagnostics should not stop at divergence-free reconstruction,
- preserving `\nabla\cdot v = 0` is not enough if the method destroys the invariant structure.

This is one of the central reasons PSPTA matters.

---

## 4. Zero purely advective transverse macrodispersion in the target regime

The paper proves that when the conductivity field is:
- smooth,
- locally isotropic,
- finite,
- and the flow is stagnation-free,

the asymptotic transverse macrodispersion coefficients vanish in the purely advective limit.

Practical consequence:
- in this regime, the target result is

```math
D^m_{22} = D^m_{33} = 0
```

for pure advection.

So for this repository:

- a numerically positive `α_T` in a smooth-isotropic purely advective Darcy case is **not validation**,
- it is a sign that the numerical pipeline may be leaking across streamsurfaces.

---

## 5. Why conventional methods can fail

The paper identifies **two distinct numerical failure sources**:

### A. Velocity reconstruction errors
A reconstruction may preserve divergence but fail to preserve the helicity-free / invariant structure.

Example discussed in the paper:
- Pollock-like / linearly reconstructed cellwise velocity fields.

The message is:

> divergence-free interpolation is not enough.

### B. Streamline integration errors
Even if the velocity field is exact or structure-consistent, a time integrator can still drift off the invariant surfaces if it does not preserve the invariants.

Example discussed in the paper:
- Runge–Kutta tracking without explicit invariant preservation.

The paper shows that both kinds of error can mimic Brownian-like transverse spreading and thus be falsely interpreted as macrodispersion.

---

## 6. Pseudo-symplectic tracking is not optional “nice to have”

The paper proposes a pseudo-symplectic particle-tracking method that explicitly preserves the invariants `ψ1, ψ2`.

The core logic is:
- compute or represent the two streamfunctions,
- parameterize motion along the streamline while preserving the invariant labels,
- prevent artificial crossing between streamsurfaces.

For MacroFlow3D, this is the scientific rationale for:
- PSPTA,
- invariant-aware tracking,
- and any future algorithm that prioritizes geometric preservation over generic ODE integration.

---

## 7. Local dispersion changes the story, but only in the correct way

The paper also analyzes the case with molecular/local dispersion.

Main point:
- for helicity-free locally isotropic Darcy flow, transverse macrodispersion with local dispersion present scales with the local dispersion magnitude,
- it smoothly tends to zero as local dispersion tends to zero.

Interpretation:
- the limit is **regular**, not singular, in the non-chaotic isotropic Darcy case,
- so if your numerical method predicts non-zero transverse macrodispersion in the pure-advection limit, that is not an innocent artifact of taking `D0 -> 0`; it is likely numerical leakage.

---

## 8. Flows where the Lester result does NOT apply

The paper is very clear that the zero-transverse result is not universal.

The constraints can be broken by:
- **locally anisotropic conductivity**,
- **non-smooth conductivity fields**,
- **stagnation points / source-driven degeneracies**,
- more general 3D flows without the two-invariant structure.

Those cases may exhibit:
- non-zero helicity,
- braiding / knotted / unconstrained streamline motion,
- genuine transverse macrodispersion in pure advection.

This distinction is critical. The repository must never overgeneralize the Lester result beyond its regime of validity.

---

## Operational implications for MacroFlow3D

## A. Scientific interpretation rules

### Rule 1
For smooth, locally isotropic, purely advective Darcy cases, positive asymptotic transverse macrodispersion is **not** accepted as physical by default.

### Rule 2
A method is not validated just because it is:
- stable,
- divergence-free,
- high-order,
- or visually plausible.

It must preserve the relevant kinematic constraints well enough.

### Rule 3
When a method changes interpolation, reconstruction, or tracking, the burden of proof is on the method to show that it does **not** generate spurious transverse leakage.

---

## B. What must be measured

For the PSPTA / invariant path, the following are scientifically meaningful diagnostics:

- residuals of `v · ∇ψ1`,
- residuals of `v · ∇ψ2`,
- mismatch between `v` and `∇ψ1 × ∇ψ2`,
- independence / non-degeneracy of `ψ1, ψ2`,
- Newton / projection failure counts,
- confinement of trajectories to invariant surfaces,
- transverse spreading in controlled purely advective smooth-isotropic cases.

A change that improves runtime but weakens these diagnostics is not automatically an improvement.

---

## C. Acceptance-gate consequences

This paper justifies the current gate structure in the repo:

- **Tier / Gate B** must cover operator / invariant integrity.
- **Tier / Gate C** must cover physics / ensemble behavior.
- scientific-core changes must not merge on the basis of compilation alone.

In particular:
- changes to interpolation,
- changes to particle stepping,
- changes to invariant construction,
- and changes to macrodispersion estimation

must be reviewed against the Lester constraints.

---

## D. What this means for baseline methods

The historical 3D macrodispersion literature often treated positive transverse macrodispersion in 3D as expected.
This paper says that conclusion depends on the **class of flow model** and **the numerical method**.

So in MacroFlow3D:

- baseline RWPT methods remain useful,
- but they are not the scientific oracle in the smooth-isotropic pure-advection regime,
- the invariant-preserving path is the scientifically privileged path for that regime.

---

## Practical checklist for developers

Before accepting a result as physical, ask:

1. Is the conductivity field smooth?
2. Is it locally isotropic?
3. Is the run purely advective or nearly so?
4. Are stagnation points absent or irrelevant?
5. Is the velocity reconstruction structure-preserving enough?
6. Does the tracking preserve the invariants or only the ODE numerically?
7. Could the observed `α_T` be caused by streamsurface crossing error?

If these questions are not answered, the result is not scientifically secure.

---

## What this paper should change in engineering decisions

This paper justifies the following repository policies:

- structure-preserving transport is a first-class concern,
- invariant diagnostics are mandatory, not optional,
- operator/invariant evals must exist before autonomy increases,
- local-vs-remote validation split must preserve scientific checks, not only HPC throughput,
- a “working” numerical method is insufficient if it violates the kinematic structure.

---

## Use this paper when

Read this note before tasks involving:

- PSPTA / pseudo-symplectic transport
- invariant construction
- streamfunction approximations
- velocity reconstruction
- particle interpolation
- trajectory integration
- macrodispersion validation
- interpretation of transverse spreading

---

## Do not overclaim

This paper does **not** say:
- all 3D groundwater flows have zero transverse macrodispersion,
- all isotropic-looking numerical fields are safe,
- any two scalar labels found numerically are automatically valid invariants,
- divergence-free interpolation is enough,
- PSPTA is trivial to implement.

Its message is narrower and stronger:
- in the specific smooth, locally isotropic Darcy regime, the geometry of trajectories is constrained,
- and numerical methods must respect that geometry if we want physically trustworthy transverse-dispersion predictions.
