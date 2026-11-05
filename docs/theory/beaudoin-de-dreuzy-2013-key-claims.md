# Beaudoin & de Dreuzy 2013 — Scientific theory notes for MacroFlow3D

## Reference

A. Beaudoin and J.-R. de Dreuzy.
**Numerical assessment of 3-D macrodispersion in heterogeneous porous media**
*Water Resources Research*, 49, 2489–2496, 2013. doi:10.1002/wrcr.20206 (p. 2489).

Changelog: 2026-11-04: the SF-43 project comparison no longer says "close to `sigma^4` scaling at small amplitude"; it gives `alpha_TP/sigma^4` and the local exponents. 2026-11-02: corrected after the full reading of SF-43's pre-activation review (bibliography section 7); see docs/decisions/2026-11-02-sf43-pre-activation-decisions.md H19. Page references are journal pages of WRR 49 with paragraph numbers [n]. Figure values are read from page images (about ±10-20 % on log axes). "Inference" marks reasoning of this project, not a statement of the paper.

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

The paper reports that the transverse macrodispersivity in 3D is significantly non-zero, unlike the 2D case. The paper states this "significantly non-zero" for `sigma_Y^2 >= 1` (p. 2493, [12]), while the same paper calls braiding mechanisms "negligible at low levels of heterogeneities (`sigma_Y^2 <= 1`)" (p. 2494, [18]); the tabulated `alpha_TA` is `< 1e-3` at `sigma_Y^2 = 0.25` and 0.01 at `sigma_Y^2 = 1` (Table 2, p. 2493).

Interpretation in the paper:
- in 2D, streamlines are confined to finite-width tubes and cannot diverge indefinitely,
- in 3D, flow lines can braid / diverge in the transverse directions,
- therefore asymptotic transverse macrodispersion is possible.

This was one of the numerical pillars supporting the “3D implies positive transverse macrodispersion” viewpoint.

---

## 2. Transverse macrodispersion grows strongly with heterogeneity

A major numerical result is that the transverse macrodispersivity increases strongly with `σ_Y^2`.

The paper characterizes this increase as steep: `alpha_TAM ~ alpha_TAM(1) sigma_Y^4` (eq. 7, p. 2493), i.e. proportional to `sigma_Y^4` and quadratic in `sigma_Y^2` (not "approximately quadratic in `sigma_Y`"). The range of validity is stated inconsistently in the paper: "for `sigma_Y^2 >= 1`" (p. 2493, [12]) but "for `sigma_Y^2 >= 2.25`" (p. 2494, [18]). At `sigma_Y^2 = 4`, `alpha_TAM ~ 0.11` (read from Fig. 4b; approximate, the paper tabulates `alpha_TA` only up to `sigma_Y^2 = 2.25`); eq. 7 with `alpha_TAM(1) = 0.01` gives 0.16 there, about 50 % above the plotted value (inference: the power law is only approximate; the paper does not point this out).

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

Caution: the paper's claim that `alpha_T` has converged by `t ~ 1-10` is contradicted by its own Fig. 2b (see "Reading cautions" below).

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

**Tracking scheme (p. 2490, [6]).** "Velocity v at the grid center is derived from the hydraulic head h by Darcy's law" (cell-centre vectors; the paper does not say whether face fluxes, Pollock-style, or cell-centre vectors are interpolated, so face-flux interpolation is not established). Transport is "restricted to the sole advection process and is simulated by a first-order explicit particle tracking method", `x(t + dt) = x(t) + v[x(t)] dt`, with the velocity "obtained with a linear interpolation in each direction from its grid values [Pollock, 1988]", the time step adapted to about 10 steps per cell, and no local dispersion. The integrator is explicit Euler, not Pollock's semi-analytic scheme, so the Pollock analysis of Lester et al. 2023 (exact integration within cells, `docs/theory/lester-2023-key-claims.md` §5A) is not this paper's scheme. Inference: in Lester et al. 2023's terms, error source (B) (a first-order integrator with no invariant preservation) is present, and source (A) (a linearly interpolated velocity that is not helicity-free) is probably present but depends on what is interpolated. The paper never considers numerical artefacts as a cause of `alpha_T`.

For MacroFlow3D, this means any comparison to Beaudoin & de Dreuzy must respect:
- geometry,
- injection protocol,
- domain aspect ratio,
- transverse periodicity,
- realization count,
- tracking algorithm.

Otherwise the comparison is weak.

### Protocol and numbers (read from the paper, 2026-10-27)

Locators are journal pages of WRR 49 (2489-2496).

- Covariance (eq. 1, p. 2489): `<Y'(x) Y'(x')> = sigma_Y^2 exp(-(|x - x'|/lambda)^2)`.
- First-order predictions (eqs. 2-3, pp. 2489-2490): `D_LA = (sqrt(pi)/2) <v> lambda sigma_Y^2`; `D_TA = 0` (first order, no local diffusion).
- Variances: `sigma_Y^2` in {0.25, 1, 2.25, 4, 6.25, 7.56, 9}; `lambda/Delta = 10`.
- Domain width: `L_T = 25.6 lambda` is stated only in the `sigma_Y^2 = 6.25` captions (Fig. 1, Fig. 5); for the other variances it is only implied by the injection window "around (20 lambda)^2 = 400 lambda^2" with "25 particles per correlation square" (p. 2491).
- Domain length: `L_L = 102.4 lambda` "is enough" for `sigma_Y^2 <= 1` (p. 2492, [9]); the longest domain is `L_L = 1638.4 lambda` (p. 2492, [10]); `L_L` for `sigma_Y^2 = 2.25` and 4 is not stated (Fig. 2 data run to about `t ~ 60-100` and `t ~ 200`, so those domains were much longer than 102.4 lambda; read from the figure). Fixed heads on the `L` faces, periodic transverse faces; finite volume with harmonic-mean `K`; HYPRE AMG (p. 2490).
- Injection: flux-weighted over `0.8 L_T x 0.8 L_T`, at least 5 `lambda` from the inlet (p. 2490). `N_p = 1e4`, `N_R = 500` (p. 2491).
- Estimator (eq. 4, p. 2490): `alpha_k(t) = (1/(2 lambda |<v>|)) (1/N_R) sum_i d/dt [ (1/N_p) sum_j x_k^{j,i}(t)^2 - ((1/N_p) sum_j x_k^{j,i}(t))^2 ]`, centred per realization in fixed axes; time normalized by `lambda/|<v>|`, with `|<v>| = (K_eq/theta)|h_+ - h_-|/L_L` and "K_eq the equivalent permeability [Matheron, 1967]" (p. 2490). The paper does not say whether `K_eq` comes from Matheron's formula or is computed; any mismatch with the true mean velocity rescales both the time axis and `alpha`.
- Asymptotic value (eq. 5, p. 2492): the mean of `alpha_k(t)` over `[0.75 t_b, t_b]`, with `t_b` "the time at which the first particle steps out of the simulation domain" (p. 2492, [9]).
- Table 1 (p. 2492), longitudinal fit `alpha_LAF / t_LF / a_LF`: 0.25: 0.24 / 1.35 / 1; 1: 1.27 / 2.21 / 0.80; 2.25: 4.35 / 4.83 / 0.63; 4: 12.7 / 20.5 / 0.49; "6.52" (printed so; probably 6.25): 58.6 / 41.9 / 0.40; 7.56: 131 / 208 / 0.39.
- Table 2 (p. 2493), "This study": `alpha_LA` = 0.26 / 1.27 / 4.35 and `alpha_TA` = `< 1e-3` / 0.01 / 0.044 at `sigma_Y^2` = 0.25 / 1 / 2.25. Inconsistencies in the paper: the caption says "Late-Time Averaging Method of Equation (6)" but eq. 6 is the fit and the late-time average is eq. 5; `alpha_LA` equals Table 1's `alpha_LAF` at `sigma_Y^2` = 1 and 2.25 but not at 0.25 (0.26 vs 0.24). The Dentz (0.007, 0.028) and Schwarze (0.009) non-zero values came from first-order synthetic (linearized) velocity fields, not from a full Darcy solve.
- Eq. 7 (p. 2493): `alpha_TAM ~ alpha_TAM(1) sigma_Y^4`; validity "for `sigma_Y^2 >= 1`" (p. 2493, [12]) vs "for `sigma_Y^2 >= 2.25`" (p. 2494, [18]), an inconsistency in the paper. Fig. 4b points (approximate): `alpha_TAM` about 0.11 (`sigma_Y^2 = 4`), 0.35 (6.25), 0.5 (7.56), 0.7 (9). Eq. 7 with `alpha_TAM(1) = 0.01` gives 0.16 at `sigma_Y^2 = 4`, about 50 % above the plotted 0.11. Fig. 1b (`sigma_Y^2 = 6.25`, linear axis) shows an `alpha_T` plateau of about 0.27 from `t ~ 10` to 150.

Reading cautions:
- `alpha_T(t)` is not monotone (Fig. 2b, p. 2491), and its shape depends on `sigma_Y^2` (figure readings, approximate): for `sigma_Y^2 <= 4` it rises to a peak at `t ~ 1-5`; for `sigma_Y^2 >= 6.25` the curves start high and decline slightly (they do not rise); at `sigma_Y^2 = 1` there is no clear plateau before the data end (`t ~ 30-40`); at `sigma_Y^2 = 2.25` it levels off only around `t ~ 30-50`.
- The paper states that the transverse asymptotic regime is reached "around the advection time ... at all levels of heterogeneity" ([9], p. 2491) and that at "10 times the advection time" it has converged (pp. 2491-2492). Fig. 2b contradicts this: at `t = 10`, `alpha_T` is about 0.015 (`sigma_Y^2 = 1`) and about 0.08 (`sigma_Y^2 = 2.25`), against the tabulated 0.01 and 0.044 (Table 2). Treat the "converged at `t = 10`" statement as not supported by the paper's own figure.
- The only resolution / `N_p` / `N_R` check is at `sigma_Y^2 = 6.25` (Fig. 1 and [7], pp. 2490-2491): `alpha_LAM` and `alpha_TAM` change by at most 2 % (longitudinal) and 4 % (transverse) when `N_p` is multiplied by 10, `N_R` goes 500 to 250 and `lambda/Delta` goes 10 to 20. "Smaller `sigma_Y^2` cases lead to similar conclusions" is asserted, not shown. Consequence (inference): `lambda/Delta = 20` also halves `dt` (10 steps per cell), so an error scaling like Lester et al. 2023's Pollock variance (proportional to `Delta^2`) should have shrunk about 4x; a change of at most 4 % argues against a pure `Delta^2`-interpolation origin of this paper's `alpha_T` at `sigma_Y^2 = 6.25`.
- No error bars are reported.
- Fig. 2's caption ("Full ... `alpha_LAM`, dashed ... fit") looks swapped relative to the plot (inference from the page image); take care if the figure is digitized.
- Numerical artefacts are never considered as a cause of the non-zero `alpha_T`.

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

Project comparison (SF-43, `docs/experiments/2026-11-03-sf43-confirmatory-macrodispersion.md`; different estimands, order of magnitude only): the plane-based `alpha_TP` over `[25, 94] lambda` is 5.3e-4 / 7.2e-3 / 2.9e-2 / 6.8e-2 `lambda` at `sigma_Y^2` 0.25 / 1 / 2.25 / 4, about 0.6-0.7 of this paper's `alpha_TA` (Table 2; Fig. 4b at 4) while `alpha_TP / sigma^4` falls from 8.4e-3 (0.25) to 7.2e-3, 5.7e-3 and 4.3e-3 (4), with local exponents `d ln alpha_TP / d ln sigma^2` of 1.89 [1.86, 1.92], 1.70 [1.67, 1.74] and 1.51 [1.45, 1.57] (0.25 -> 1 -> 2.25 -> 4; 1.93 [1.77, 2.09] from 0.04 to 0.25), so there is no `sigma^4` law over this range (post hoc S of the SF-43 note, section 3.4); the consumer's time-window `alpha_LAM` is 0.96 / 0.86 / 0.67 / 0.47 of eq. 6 with Table 1 parameters averaged over the same window (the gap is recorded as open question G9b).

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

- covariance model, including the correlation-length convention: here `C(r)=σ² exp(-(r/λ)²)` (the paper's eq. 1); the legacy code used `exp(-π r²/(4λ²))` (a code fact, not in the paper),
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
