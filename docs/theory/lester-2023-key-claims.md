# Lester 2023 — Scientific theory notes for MacroFlow3D

## Reference

Daniel R. Lester, Marco Dentz, Prajwal Singh, and Aditya Bandopadhyay.
**Under What Conditions Does Transverse Macrodispersion Exist in Groundwater Flow?**
*Water Resources Research*, 59, e2022WR033059, 2023.

---

## Why this paper matters for this repository

This paper is the main scientific basis for the current Lester equation (14) streamfunction-solver direction and for any future invariant-preserving transport consumer in MacroFlow3D.

Its core claim is not merely that some numerical methods are inaccurate. It is stronger:

- for **steady 3D Darcy flow with smooth, locally isotropic scalar conductivity** and **no stagnation points**, the flow is **helicity-free**,
- such flows admit **two invariants / streamfunctions**,
- those invariants constrain trajectories to 2D streamsurfaces,
- therefore, **under the paper's assumptions (bounded label fluctuations, §4 of the paper)**, purely advective transverse macrodispersion is zero in that regime,
- and conventional particle-tracking methods can produce **spurious positive transverse macrodispersion** if they do not preserve those kinematic constraints.

This note labels each claim below as either **What the paper states** or **Verified / refuted in this project**. Since 2026-10-02 the project does not use the zero-transverse claim as a regime expectation or as an acceptance oracle (owner decision O3 of `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`).

For MacroFlow3D, positive transverse spreading in the purely advective smooth-isotropic Darcy regime is **not automatically physical**; zero transverse spreading is **not presupposed** either. Every observed growth is classified physical / numerical / unresolved with refinement evidence.

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

**What the paper states.** Steady 3D helicity-free flows admit two invariants / streamfunctions `ψ1(x), ψ2(x)` satisfying

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

In the paper this is the conceptual bridge between:
- helicity-free Darcy flow,
- integrability,
- and the absence of purely advective transverse macrodispersion.

**Verified / refuted in this project** (`docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md`; limits: CPU probes, smooth Gaussian-covariance field, amplitude <= 1, `L/ell = 4`, one realization; the production-stack repeat is pending in SF-30).
- Two invariants exist **locally**: Euler potentials exist locally for any divergence-free field, so §1 (helicity-free) and local existence stand.
- A **global, affine + triply periodic** nondegenerate pair does **not** exist for a generic smooth scalar periodic `k`. Such a pair forces every Darcy streamline to close on the torus; the return map of the face `x1 = 0` instead differs from the identity at second order in the amplitude, independently of resolution. A 2-D control and the mirror-symmetric Lester (2021) §3 field close to roundoff; breaking the symmetry breaks closure.
- The periodic solution of equation (14) exists, but it is the closed-streamline field nearest to Darcy in the `1/k` energy (`e_v ~ amplitude^2`, resolution independent). Equation (14) imposes only the two components of `curl(c/k) = 0` across `c`; the helicity component `B . c = 0` is not imposed.

---

## 3. Euler-potential / dual-streamfunction representation

**What the paper states.** The same paper gives the velocity representation

```math
\mathbf{v}(x) = \nabla \psi_1(x) \times \nabla \psi_2(x),
```

with non-vanishing gradients away from degeneracies / stagnation issues.

Interpretation for MacroFlow3D:
- a correct numerical method should preserve, or at least respect, this invariant geometry,
- diagnostics should not stop at divergence-free reconstruction,
- preserving `\nabla\cdot v = 0` is not enough if the method destroys the invariant structure.

This is one of the central reasons invariant construction and invariant-preserving transport matter.

**Verified / refuted in this project.** The representation holds locally. Globally, with affine + periodic labels, it holds only for flows with closed streamlines (see §2 above); on a generic periodic Gaussian field the pair solving (14) represents a different flow than Darcy.

## 3A. Lester equation (14) solver formulation

**What the paper states.** The paper's invariant-construction equation is the coupled nonlinear streamfunction system of Lester et al. equation (14), here in the form used by the project's frozen solver stack:

```math
Delta psi1 - grad(log k).grad(psi1) = S1
Delta psi2 - grad(log k).grad(psi2) = S2
```

**Index convention (corrected 2026-09-30).** The paper prints the right-hand
sides crossed (`... psi1 = S2`, `... psi2 = S1`). Deriving from `v = grad psi1 x
grad psi2` and `curl v = grad(ln k) x v` gives `grad(psi1) L2 - grad(psi2) L1 = B`
with `L_i = Delta psi_i - grad(ln k).grad(psi_i)`, hence `L_i = S_i` with the
paper's own definition of `S_i` below. The exact Darcy pair `k = k(x1)`,
`psi1 = x2 + Phi(x3)`, `psi2 = x3` (`v = e1`) satisfies the same-index form to
roundoff and violates the crossed form by `|Phi''|`. Full derivation, numerical
check, and consequences: `docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md`.

where:

```math
S_i =
((B x grad psi_i).(grad psi1 x grad psi2)) /
|grad psi1 x grad psi2|^2
```

and:

```math
B = (grad psi1.grad) grad psi2 - (grad psi2.grad) grad psi1.
```

Use the equivalent divergent form:

```math
Delta psi - grad(log k).grad psi = k div((1/k) grad psi).
```

With `q=1/k` and `A psi = -div(q grad psi)`, decoupled nonlinear iterations solve:

```math
A psi1 = -q S1
A psi2 = -q S2
```

This reformulation is operationally important because it avoids explicit finite-difference evaluation of `grad(log k)` and exposes a variable-coefficient diffusion operator that may be compatible with existing PCG/MG machinery after verification.

**Provenance of the index swap (verified against Lester 2021).** Eq. (2.16) of Lester et al. (2021) prints `+B`, where the identity `curl(a x b) = a div b - b div a + (b.grad)a - (a.grad)b` gives `-B`. Eq. (2.20) is correct: crossing it with `grad psi2` and `grad psi1` gives `L2 = a1`, `L1 = a2`, with `L_i = Delta psi_i - grad f . grad psi_i`. Since the 2021 `a1 = (B x grad psi2).v/|v|^2` equals the 2023 `S2` and `a2` equals `S1`, this is `L_i = S_i` (same index). Eqs. (2.22)-(2.23) nevertheless print `L1 = a1`, `L2 = a2`, and the 2023 eq. (14) copies that swap. Same-index was also established independently by SF-26.

**Contract of the frozen stack versus the project's new target.** The affine + periodic-fluctuation formulation (and its gauge) is the contract of the frozen stack (SF-02..SF-26); it is verified infrastructure and the producer of invariants on symmetric controls (the Lester 2021 field). The project's new target is the **Darcy labels anchored at the inlet face and non-periodic in `x1`** (decision O1 = B), prototyped on CPU in SF-29. SF-29 selected the formulation for that target (`docs/decisions/2026-10-06-eq14-inlet-label-formulation.md`).

Do not treat multigrid reuse as confirmed by theory. It must be checked against the repository's actual operator sign, coefficient placement, boundary handling, gauge, and residual.

---

## 4. Zero purely advective transverse macrodispersion in the target regime

**What the paper states.** The paper proves that when the conductivity field is:
- smooth,
- locally isotropic,
- finite,
- and the flow is stagnation-free,

the asymptotic transverse macrodispersion coefficients vanish in the purely advective limit, under the assumption of bounded label fluctuations (§4 of the paper):

```math
D^m_{22} = D^m_{33} = 0
```

**Verified / refuted in this project** (`docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md`; limits: CPU probes, smooth Gaussian-covariance field, amplitude <= 1, `L/ell = 4`, one realization; the production-stack repeat is pending in SF-30).
- In the periodic cell the bounded-fluctuation assumption fails: the transverse displacement of the Darcy streamlines grows without bound over many periods (finding 3 of the note), because the streamlines do not close.
- The paper's `D_T = 0` for the periodic surrogate is a property of the surrogate (closed streamlines by construction), not a measurement on the Darcy flow.
- The project does not use `D_T = 0` as an oracle (owner decision O3). The value of `alpha_T` in a random, non-periodic medium is not established.

So for this repository:

- a numerically positive `α_T` in a smooth-isotropic purely advective Darcy case is **not validation by itself and not automatically physical**; it must be classified physical / numerical / unresolved,
- a numerically zero `α_T` is not presupposed either, and is not by itself evidence of correctness.

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
- the Lester equation (14) streamfunction solver,
- invariant-aware tracking,
- and any future transport algorithm that prioritizes geometric preservation over generic ODE integration.

---

## 7. Local dispersion changes the story, but only in the correct way

The paper also analyzes the case with molecular/local dispersion.

Main point:
- for helicity-free locally isotropic Darcy flow, transverse macrodispersion with local dispersion present scales with the local dispersion magnitude,
- it smoothly tends to zero as local dispersion tends to zero.

Interpretation (**what the paper states**, under its bounded-label-fluctuation assumption; not verified by this project, and see §4 for the limits found in the periodic cell):
- the limit is **regular**, not singular, in the non-chaotic isotropic Darcy case,
- the project measures `D_T(Pe)` in a later phase instead of assuming this scaling.

---

## 8. Flows where the Lester result does NOT apply

**What the paper states:** the zero-transverse result is not universal.

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

## 9. Smooth Gaussian fields vs exponential fields

For MacroFlow3D validation, Gaussian-covariance log-conductivity fields are the main smooth benchmark for invariant existence and equation (14) solver development.

Exponential-covariance fields are not automatically equivalent. Their reduced smoothness can violate the classical hypotheses needed for a global two-streamfunction representation. If they are used after regularization, the run must state:

- smoothing scale;
- resolution dependence;
- whether the original or regularized problem is solved;
- whether invariants converge under grid refinement.

Project status (2026-10-07): the code generates Gaussian covariance only; exponential fields would require reintroducing a generator and a new decision.

## 10. Tensor conductivity and local anisotropy

Do not assume two global invariants exist for locally anisotropic tensor conductivity. That case can break the helicity-free structure and is outside the initial equation (14) solver scope.

---

## Operational implications for MacroFlow3D

## A. Scientific interpretation rules

### Rule 1
For smooth, locally isotropic, purely advective Darcy cases, positive asymptotic transverse macrodispersion is **not** accepted as physical by default, **nor is zero presupposed**. Classify every observed growth as physical / numerical / unresolved with grid and tolerance refinement evidence, and compare against the streamline-closure oracle (SF-30).

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

For the streamfunction / invariant path, the following are scientifically meaningful diagnostics:

- residuals of `v · ∇ψ1`,
- residuals of `v · ∇ψ2`,
- mismatch between `v` and `∇ψ1 × ∇ψ2`,
- independence / non-degeneracy of `ψ1, ψ2`,
- coupled equation (14) residual when using the new solver,
- denominator percentiles for `|∇ψ1 × ∇ψ2|`,
- Newton / projection failure counts,
- confinement of trajectories to invariant surfaces,
- transverse spreading in controlled purely advective smooth-isotropic cases,
- the return map of independently integrated Darcy streamlines (closure oracle, SF-30),
- `e_v` under grid refinement on the same continuum field, with observed order.

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
- the invariant-preserving path is the scientifically privileged path for that regime, with labels of the actual Darcy flow (not the periodic surrogate),
- baseline results are not accepted as physical by default, and zero transverse spreading is not presupposed either.

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
- a “working” numerical method is insufficient if it violates the kinematic structure,
- the existence of any new target object is checked independently of the solver before the solver is accepted.

---

## Use this paper when

Read this note before tasks involving:

- Lester equation (14) streamfunction solving
- invariant-preserving transport
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
- invariant-preserving transport is trivial to implement,
- that a periodic solution of equation (14) represents the Darcy flow of a generic periodic Gaussian field (refuted in this project, see §2).

Its message is narrower and stronger:
- in the specific smooth, locally isotropic Darcy regime, the geometry of trajectories is constrained under the paper's assumptions (bounded label fluctuations, which this project found to fail in the periodic cell),
- and numerical methods must respect that geometry if we want physically trustworthy transverse-dispersion predictions.
