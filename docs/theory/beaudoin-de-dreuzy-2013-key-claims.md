# Beaudoin & de Dreuzy 2013 — Scientific theory notes for MacroFlow3D

## Reference

A. Beaudoin and J.-R. de Dreuzy.
**Numerical assessment of 3-D macrodispersion in heterogeneous porous media**
*Water Resources Research*, 49, 2489–2496, 2013.

---

## Why this paper matters for this repository

This paper is one of the key numerical references behind the classical expectation that, in 3D heterogeneous porous media, both:

- longitudinal macrodispersion becomes very large with heterogeneity, and
- transverse macrodispersion is **non-zero** and increases with heterogeneity.

Historically, this kind of result motivated the widespread belief that 3D flow-line braiding and unconstrained wandering naturally generate asymptotic transverse macrodispersion in purely advective transport.

For MacroFlow3D, this paper matters in two ways:

1. it defines an important **classical baseline / comparison target**, and
2. after Lester 2023, it must be interpreted with more care, because some of the regimes treated as generic 3D behavior may conflict with the constrained kinematics of smooth locally isotropic Darcy flow.

So this paper is not obsolete.
It is a baseline that must be read together with the newer kinematic interpretation.

---

## Core modeling setup

The paper studies:
- 3D heterogeneous porous media,
- lognormal permeability fields,
- isotropic **Gaussian-correlated** log-permeability,
- steady Darcy flow,
- particle tracking for purely advective transport,
- Monte Carlo estimation of macrodispersivities.

The key stochastic model is:

- `Y(x) = ln K(x)` is Gaussian,
- with variance `σ_Y^2`,
- with isotropic Gaussian covariance,
- and correlation length `λ` / `l` depending on notation.

This is directly relevant to MacroFlow3D, which also studies large 3D lognormal conductivity fields and Monte Carlo macrodispersion.

---

## Main scientific claims

## 1. 3D transverse macrodispersion is non-zero in their numerical study

The paper reports that the transverse macrodispersivity in 3D is significantly non-zero, unlike the 2D case.

Interpretation in the paper:
- in 2D, streamlines are confined to finite-width tubes and cannot diverge indefinitely,
- in 3D, flow lines can braid / diverge in the transverse directions,
- therefore asymptotic transverse macrodispersion is possible.

This was one of the numerical pillars supporting the “3D implies positive transverse macrodispersion” viewpoint.

---

## 2. Transverse macrodispersion grows strongly with heterogeneity

A major numerical result is that the transverse macrodispersivity increases strongly with `σ_Y^2`.

The paper characterizes this increase as steep and approximately quadratic in the heterogeneity range they analyze at moderate/high heterogeneity.

Interpretation:
- stronger heterogeneity produces stronger streamline deflection and braiding,
- that stronger geometrical complexity appears as enhanced transverse spreading.

For MacroFlow3D, this is historically important because it is part of the lineage behind the expectation that:

```math
\alpha_T > 0
```

and may increase rapidly with heterogeneity.

---

## 3. Longitudinal macrodispersion grows even more strongly

The paper also finds that longitudinal macrodispersion is much larger than transverse macrodispersion, and in 3D it grows very strongly with heterogeneity.

A key observation is that:
- longitudinal macrodispersivity in 3D can be much larger than in 2D,
- the 3D growth is much stronger than simple low-order perturbative intuition would suggest.

This is fully relevant to MacroFlow3D because even if the transverse story is scientifically reinterpreted in light of Lester 2023, the paper still reinforces that:

- 3D heterogeneous transport can exhibit very large longitudinal spreading,
- and longitudinal behavior remains an important validation target.

---

## 4. Transverse convergence is faster than longitudinal convergence

The paper reports that:
- transverse effective dispersivity tends to reach its asymptotic regime faster,
- longitudinal effective dispersivity converges much more slowly, especially at high heterogeneity.

This matters operationally for MacroFlow3D because it implies:

- domain length matters a lot,
- naive short runs may misestimate longitudinal asymptotes,
- “appears converged” is not enough unless the convergence regime is checked carefully.

---

## 5. Numerical methodology assumptions matter

Their study relies on:
- finite-volume Darcy solve,
- periodic transverse boundary conditions,
- particle tracking with linearly interpolated velocities,
- Monte Carlo averaging over many realizations,
- large elongated domains.

These are not side details.
They shape the inferred macrodispersion.

For MacroFlow3D, this means any comparison to Beaudoin & de Dreuzy must respect:
- geometry,
- injection protocol,
- domain aspect ratio,
- transverse periodicity,
- realization count,
- tracking algorithm.

Otherwise the comparison is weak.

---

## What this paper contributed historically

Before the more recent kinematic reinterpretation, this paper supported the following picture:

- in 3D, streamlines can wander freely enough to generate non-zero transverse macrodispersion,
- increasing heterogeneity amplifies both transverse and longitudinal macrodispersion,
- 3D transport is qualitatively different from 2D in an essential way.

This picture strongly influenced the interpretation of 3D groundwater transport simulations, including the expectation that positive `α_T` at large times is normal.

---

## What must now be read with caution

After Lester 2023, the strongest claims of this paper must be interpreted more carefully in the specific regime of:

- smooth conductivity,
- locally isotropic Darcy flow,
- purely advective transport,
- structure-preserving trajectory constraints.

Why?
Because Lester argues that in that particular regime, the flow is helicity-free and admits two invariants that confine trajectories, so purely advective asymptotic transverse macrodispersion should vanish.

That means the Beaudoin & de Dreuzy result remains scientifically important, but not automatically universal.

For MacroFlow3D, this translates to:

- treat this paper as a **classical baseline**, not as the final word on every 3D Darcy regime.

---

## What remains highly valuable for MacroFlow3D

## 1. Domain design lessons

This paper strongly supports practical numerical lessons that still matter:

- the domain must be **long** in the mean-flow direction,
- transverse directions should use periodic boundaries when appropriate,
- Monte Carlo sampling must be robust,
- and spatial resolution must be sufficient relative to correlation length.

These are fully aligned with the repository’s HPC pipeline design and with your presentation strategy.

---

## 2. Monte Carlo / ensemble discipline

The paper emphasizes:
- many realizations,
- many particles,
- careful asymptotic estimation,
- and convergence analysis.

That remains directly useful.
Even if the physical interpretation of `α_T` changes under Lester’s framework, the statistical discipline does not.

---

## 3. Longitudinal validation target

The paper is still valuable as a longitudinal benchmark reference:
- strong heterogeneity increases `α_L`,
- 3D can produce substantially larger longitudinal macrodispersion than 2D,
- convergence to asymptotic longitudinal behavior can be slow.

So MacroFlow3D should continue to treat longitudinal macrodispersion as a serious validation observable.

---

## 4. Warning about naive asymptotics

The paper shows that low-order perturbation intuition can fail badly at higher heterogeneity.
That is a useful general warning:

- do not overtrust weak-heterogeneity asymptotics in strongly heterogeneous regimes,
- numerical validation must be done in the actual parameter ranges of interest.

---

## What this paper does NOT give you

This paper does not prove that every positive transverse macrodispersion observed numerically in 3D Darcy simulations is physical.

From the perspective of the current project, it does **not** settle:

- whether the tracking method preserves the kinematic constraints of smooth locally isotropic Darcy flow,
- whether the reconstructed velocity field respects the invariant geometry,
- whether a measured positive `α_T` comes from physics or from streamsurface leakage.

That is exactly where the later Lester framework becomes decisive.

---

## Operational implications for MacroFlow3D

## A. Use this paper as a baseline, not as an oracle

In the repo, this paper should be used to motivate:

- large 3D domains,
- serious Monte Carlo statistics,
- attention to asymptotic convergence,
- historically expected 3D transverse behavior.

But it should **not** be used as the sole justification for accepting positive `α_T` in smooth locally isotropic purely advective Darcy runs.

---

## B. Preserve the useful numerical lessons

The paper’s workflow lessons remain strong:

- long domains,
- periodic transverse boundaries when appropriate,
- many realizations,
- enough particles,
- enough spatial resolution,
- careful asymptotic estimation.

These should remain part of the acceptance and experiment design culture of the repo.

---

## C. Compare like with like

If MacroFlow3D compares itself to this paper, comparisons should state explicitly:

- covariance model,
- variance of log-conductivity,
- dimensionality,
- boundary conditions,
- injection protocol,
- transport regime,
- tracking method,
- asymptotic-estimation procedure.

Without that, “agreement” or “disagreement” is scientifically weak.

---

## D. Why this paper still matters after Lester

A bad reaction would be:
> “Lester supersedes this, so we can ignore it.”

That would be wrong.

The right reaction is:
- this paper tells us what the classical numerical literature reports,
- Lester tells us that some of those reports may include numerical / kinematic inconsistency in certain regimes,
- therefore MacroFlow3D must be able to **reproduce, reinterpret, and separate** those effects.

This is exactly why your project is scientifically interesting.

---

## Practical checklist for developers

When using this paper in the repo, ask:

1. Are we matching the field model closely enough?
2. Are our domains long enough for asymptotic claims?
3. Are our statistics robust enough?
4. Are our local numerical methods capable of generating artificial transverse spreading?
5. Are we comparing classical 3D expectations against Lester-constrained isotropic Darcy theory in a controlled way?

If not, the comparison should be labeled exploratory, not definitive.

---

## What this paper should change in engineering decisions

This paper supports the following repository-level decisions:

- keep large-domain / long-time capability as a priority,
- keep ensemble throughput as a first-class requirement,
- preserve high-quality macrodispersion estimators,
- document asymptotic-estimation assumptions clearly,
- require controlled comparisons when claiming agreement with historical 3D literature.

It also supports the idea that:
- profiling, scaling, and throughput matter,
- because weak statistics and too-short domains can invalidate the scientific interpretation.

---

## Use this paper when

Read this note before tasks involving:

- large-domain macrodispersion studies
- Monte Carlo design
- asymptotic convergence analysis
- historical literature comparisons
- longitudinal/transverse estimator interpretation
- field-generation choices
- domain and boundary-condition setup

---

## Final positioning inside MacroFlow3D

Use this paper as the repository’s **classical 3D macrodispersion baseline**.

Use Lester 2023 as the repository’s **modern kinematic constraint paper**.

The project should be engineered to resolve the tension between them scientifically, not to assume one of them wins automatically in every numerical setup.
