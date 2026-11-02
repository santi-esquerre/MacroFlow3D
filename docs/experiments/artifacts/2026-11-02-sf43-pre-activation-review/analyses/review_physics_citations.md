I attacked the physics and re-checked 23 of the citations the review relies on. Nothing is blocking. Two findings are MAJOR: the first-order benchmark is computed for an unbounded medium, not SF-43's actual boxes, and section 2.1 leaves out Lester's proof claim. Sixteen are MINOR. All 21 DOIs I queried on Crossref resolve to the cited papers.

## Findings

**F1 (MAJOR). The first-order benchmark does not describe SF-43's medium.**
- **Statements attacked:**
  - §2.1: "First order predicts a positive `alpha_T` at every finite distance" and "a positive sign alone does not discriminate…", with the d-dependence "3.6 times … 6.9 times".
  - §3.b: "first order predicts decay `~ d^-2` (3.57 x between halves)".
  - §4.b: "first order 6.9x".
  - M1: "rejected by linear theory itself".
  - H3: its "Reason", and "halves against the first-order ratio 3.57 … window ratio against 6.92".
  - H17: "first-order benchmarks as H3".
- **Evidence from the code:** the K field repeats exactly across the box in all three directions, so it also repeats along the flow every L1 = domain length.
  - `PeriodicGaussianField.cuh:5,27-38` ("Truly torus-periodic"; `k_m = 2π m/L_d`, `L_d = N_d h_d`).
  - `DirichletLongBoxStage.cu:402,429-436` (`L1 = nx*h`; the generator is called on the flow grid itself).
  - Production boxes are 104λ long (SF-43 increment, l.40-41); the distance arms are about 208λ (l.82).
- **My calculation:** quadrature over the generator's own discrete spectrum, using the first-order transverse velocity `u2 = -U k1 k2/k² Ŷ`. It matches the continuum `X22` to 0.1–1.5 % for d ≤ 8λ.

| case | half the OLS slope of `X22`, in units of σ²λ, on [25, 94] | first half / second half |
|---|---|---|
| unbounded (the review's script) | +7.99e-5 | +1.39e-4 / +3.88e-5 (ratio 3.57) |
| W 24, transverse periodicity only (L1 → ∞) | +1.1e-6 | +3.6e-6 / +8.5e-10 (ratio about 4000) |
| W 24, L1 104 (production) | **−3.5e-5** | +3.5e-6 / −1.17e-4 |
| W 12, L1 104 (σ² = 4) | −4.1e-6 | +1.1e-8 / −1.4e-5 |
| W 48, L1 104 (wide `E_W` arm) | −8.6e-5 | |
| W 24, L1 208, window [94, 194] | −7.1e-6 (unbounded: +1.15e-5) | |

- **Why the box changes the answer:**
  - With periodic transverse faces, only wavenumbers `k2 ≥ 2π/W` survive. That removes the small-wavenumber contribution that produces the d⁻² tail, so first-order α_T decays like `e^{-2πd/W}` instead (decay length 3.8λ at W 24, 1.9λ at W 12).
  - Because the field repeats along x1, first-order `X22(d) = X22(L1 − d)`. So `X22(94) = X22(10)`, and the slope turns negative over the far half of the window.
  - I did not model the correction from the Dirichlet x faces within 5–10λ of the outlet.
- **Consequences:**
  - H3's null stays conservative: every in-box value is below 7.99e-5. The decision itself is unaffected.
  - H3's stated reason and M1 are false for the simulated medium. In the box, linear theory predicts ≤ 0, so G5.4's original sign null would also have separated it from a positive nonlinear dispersion.
  - The ratios 3.57 and 6.92 are not the first-order prediction for any SF-43 arm.
  - The 104λ period along the flow (only 10λ past the last plane) is a model assumption that §3.b does not list and that no arm varies. G7's "generator" factor mixes this periodicity with the variance normalization.
  - At σ² = 0.25, the in-box first-order slope on the second half (−2.9e-5 λ) is larger in magnitude than the null value (2.0e-5 λ).
- **Correction:**
  - Keep H3's number, but reword it as "the unbounded-medium first-order value, conservative against the in-box value (which is ≤ 0)".
  - Report the in-box first-order value per arm (W, L1), computed from the generator's spectrum, next to 3.57 and 6.92.
  - Add the periodicity along x1 to the §3.b table.
  - Soften M1.

**F2 (MAJOR). §2.1 leaves out the main contrary claim.**
- **Statement attacked:** "Beyond first order in 3D no theorem forces zero".
- **Evidence:**
  - Lester et al. 2023 (10.1029/2022WR033059), p. 1: "We prove that the transverse macrodispersion does not exist…". The proof (§4, pp. 8-9) rests on p. 8: "we assume that the fluctuations ψ̃_i(x) are also bounded".
  - Lester et al. 2021, eq. 5.46 (p. A27-25), states the same zero.
  - The project's own note, `docs/theory/lester-2023-key-claims.md:21,176-186`, records both the proof and its assumption, and that the assumption fails in the periodic cell.
  - The nonzero simulations the review cites (Attinger 2004 §7; Dentz 2002; the Dentz and Schwarze columns of B&dD Table 2) use linearized velocity fields. These are not helicity-free: `h = −K_G² J ∇h'·(∇Y × e1)` is nonzero at order σ². So they say nothing about exact Darcy flow.
- **Correction:** add one sentence stating Lester's conditional proof, why the review treats it as conditional, and that the nonzero results come from linearized fields, closures or non-smooth inclusion media.

**F3 (MINOR). Lester 2025 is read as reporting data it does not contain.**
- **Statement attacked (§2.2):** "Lester et al. 2025 (… App. D) report spurious transverse dispersion from Cash-Karp RK at 1e-14…".
- **Evidence:** arXiv 2412.05419v1, App. D, says this method "has been shown (Lester et al. 2023) to introduce spurious transverse dispersion…". That is an assertion with a citation, not a measurement; the paper then switches to streamfunction tracking.
  - Lester 2023 tested RK4 at tolerances 1e-4 to 1e-8, on a different velocity, and found the error goes to zero as the tolerance tightens (p. 13).
  - Lester 2025's velocity is `v = −K∇φ` built from 5th-order splines, which "implicitly enforces the helicity-free constraint". That is the MacroFlow3D construction.
- **Correction:** say "assert, citing Lester 2023, without data".

**F4 (MINOR). Lester 2023's Runge-Kutta test field is misdescribed.**
- **Statement attacked:** "Tolerance-controlled Runge-Kutta on an exactly helicity-free field".
- **Evidence:** Lester 2023, p. 10: the velocity from the interpolated streamfunctions "is not exactly helicity-free (although it is exactly divergence-free)". p. 13: the velocity is "exact (i.e., is given in terms of the interpolated streamfunctions)".
- **Correction:** "a divergence-free velocity built exactly from interpolated streamfunctions, so the labels are exact invariants; it is not exactly helicity-free". The transfer to MacroFlow3D, whose field is helicity-free but not divergence-free, is only by analogy.

**F5 (MINOR). B&dD's tracking scheme and Lester's attribution are overstated.**
- **Statement attacked (§2.3):** "explicit Euler on face-linear (Pollock-type) interpolation … which is the scheme Lester et al. 2023 (p. 20) hold responsible".
- **Evidence:**
  - B&dD, p. 2490 [6]: "Velocity v at the grid center…", then "a linear interpolation in each direction from its grid values [Pollock, 1988]". Face-flux interpolation is not established; the review's own bibliography (§7, discrepancy 3) says so.
  - Lester 2023, p. 20, attributes B&dD's result generically to "the particle tracking methods … that do not explicitly adhere to the kinematic constraints". It analyses Pollock with exact in-cell integration and RK4, never Euler on an interpolated velocity.
- **Correction:** describe the scheme as stated in B&dD, and call Lester's attribution a general one.

**F6 (MINOR). Lester 2021 eq. 5.43 and the three-way contrast.**
- **Statements attacked:** §2.2 "Lester's own pre-asymptotic prediction"; §3.b and §4.b "first order d⁻² / Lester saturation / B&dD plateau".
- **Evidence:**
  - p. A27-24: "we may approximate … as a first-order process" (eq. 5.41). Neither ζ nor σ²_ψ is ever estimated.
  - Saturation follows from eq. 5.40 plus stationarity (eq. 5.45). The exponential form only sets the rate.
  - σ²_c is the unnormalized concentration moment of a steady plume (eqs. 5.26-5.27), not a flux-weighted first-passage variance.
  - p. A27-23 assumes "∂φ/∂x1 < 0 ∀x", i.e. no backflow. SF-39 measured u ≤ 0 on 6.7e-4 of x faces at σ² = 4.
  - First order also saturates (`X22 → σ²λ²/3`). In the box, its approach is exponential too (F1), the same shape as eq. 5.43.
- **Correction:** the testable contrast is "decays toward zero (first order, Lester)" against "plateau (B&dD, or numerical drift)". Call eq. 5.43 an illustrative approximation and list its assumptions.

**F7 (MINOR). The ergodicity citation supports a different quantity.**
- **Statement attacked (§2.1):** "the two should be close (Cvetkovic, Shapiro & Dagan 1992 … 'almost ergodic' for cumulative flux)".
- **Evidence:**
  - The CSD 1992 abstract (10.1029/91WR03085) concerns "the variance of the cumulative mass flux (or total discharge)", a zeroth-moment quantity, not dispersion.
  - Dentz 2000a: the two coefficients become equivalent only through "local transversal dispersion", which is absent here. Dentz 2000b: only for "infinitely extended sources".
  - Dagan 1991 (bibliography item 16): the longitudinal ergodic limit needs a source of about 10² integral scales.
- **Correction:** drop CSD as support. This does not affect the B&dD comparison, since both studies use the per-realization ("effective") definition.

**F8 (MINOR). Backflow and multiple crossings are omitted.**
- **Statement attacked (§3.a):** "the tracker … records first passage", with no consequence discussed.
- **Evidence:**
  - SF-39 note, l.165-167: u ≤ 0 on 0 / 0 / 8.5e-6 / 6.7e-4 of x faces at σ² 0.25 / 1 / 2.25 / 4.
  - The tracker counts `recrossings` and `backflow_encounter` (`LagrangianTrackerDevice.cuh:657-676`).
  - Janković, Fiori & Dagan 2003 (abstract): "negative velocities are present at high σY²".
  - Par2 also records first crossing (`PlaneCrossingRecorder.cuh:5,13`), so Gate 5 is consistent.
- **Correction:** state that the estimand is the first-passage distribution, not net flux. Report recrossing fractions, plus a sensitivity of α_LP at σ² = 4 without backflow-encounter particles, since the slow tail that drives it is where backflow happens.

**F9 (MINOR). Chiogna 2015 does not support the sentence it is cited for.**
- **Statement attacked:** "Helical/whirling flow requires anisotropy (… Chiogna et al. 2014/2015)".
- **Evidence:** the Chiogna 2015 abstract (10.1002/2014WR015330): local helicity is zero in the fully resolved, locally isotropic case, yet "streamlines are helical in all three cases at scales larger than the characteristic scale". Lester 2021 (A27-26) disputes this, saying those fields are not smooth.
- **Correction:** cite it as contrary evidence, together with the dispute.

**F10 (MINOR). Two citations exceed their reading level.**
- **Statements attacked:** the lowest-order zero cited to "Dagan 1984/1988, A"; "In 2D the zero is exact at all orders (… Dagan 1994, A)".
- **Evidence:**
  - The abstracts of Dagan 1984 and 1988 contain no transverse-asymptote statement.
  - Dagan 1994 shows that one exact nonlinear correction term goes to zero, for a normal velocity field.
- **Correction:** attribute "all orders" to Attinger 2004 alone, and cite Dagan only for what the abstracts say.

**F11 (MINOR). The negative claim about confidence intervals is unverified.**
- **Statement attacked:** "No published study … gives a confidence interval for a 3D Gaussian `alpha_T` at `sigma^2 >= 1`".
- **Evidence:** Chin & Wang 1992 (10.1029/92WR00666) report "confidence limits … for 0.1 ≤ σY ≤ 1.5" in 3D, including transverse dispersivity. Their covariance type was never verified.
- **Correction:** hedge the claim.

**F12 (MINOR). The shadowing citations do not support the sentence.**
- **Statement attacked:** "Ensemble statistics can survive sensitive individual trajectories (shadowing: … Sauer, Grebogi & Yorke 1997 …)".
- **Evidence:** Sauer 1997 gives scaling laws for how long numerical chaotic solutions remain valid, i.e. the limits of shadowing, and it applies to chaotic systems. For a non-chaotic flow the relevant result is polynomial error growth (Calvo & Hairer 1995, which is in the bibliography).
- **Correction:** replace or requalify the citation.

**F13 (MINOR). The W8 answer has a dangling pointer and an incomplete argument.**
- **Problem with the pointer:** "Answer for W8 (section 3.a below)" points to nothing; W8 appears only at review lines 103, 123 and 349.
- **Problem with the argument:**
  - The numerical mechanisms it cites (F3, F4) produce Brownian-like drift, not exponential separation.
  - The separation should be split into an along-streamline part (shear, `F12 ~ t^r` with r up to 2) and a transverse part (`F22` bounded; Lester 2022 eqs. 4.23-4.25 and 6.12). Only the transverse part feeds α_TP.
  - The d-dependence separates decay from a plateau, not physics from numerics.

**F14 (MINOR). Lester 2023 p. 22 is quoted without its qualifier.** The full sentence begins "Although this study primarily is focused on asymptotic dispersion…". Quote the whole sentence.

**F15 (MINOR). The α_LP comparability conclusion is right, but the stated reason is not the operative one.**
- **What survives:** the d/α_L figures are correct. Using B&dD's eq. 6 at t = d/U they are 2.3–7.5.
- **What is wrong:** `d >> α_L` is the condition for the first-order linearization, which also needs σ² ≪ 1. The 1D Fickian identity holds at every d. The operative reason is non-Fickian, pre-asymptotic travel times (B&dD's t_LF = 20.5 at σ² = 4).
- **Correction:** compare with B&dD's eq. 6 α_L(t = d/U), as G9b already does.

**F16 (MINOR). The flux-weight bias in the displacement basis is asserted, not derived.**
- **Statement attacked (§3.a, H2):** "to `O(1/n_eff)` for flux weights".
- **Problem:** the pilot check ("means agree within the position-basis SE") is weak where the position-basis SD is 37 times larger.
- **Correction:** report the uniform-weight displacement estimand as a control, since it is exactly unbiased. Count n_eff in correlation areas of the source (about 370 at 19.2λ, about 90 at 9.6λ), not in particles.

**F17 (MINOR). Several B&dD details need qualifying.**
- The §2.3 table gives 25.6λ unqualified; it is stated only at σ² = 6.25 (Figs 1 and 5).
- "Design mirror" fails at σ² = 4: width 12 vs 25.6λ, source 9.6 vs about 20λ, λ/h 16–32 vs 10.
- Fixed-time, fixed-axis α_TAM also picks up drift tilt (about θ²α_L) and longitudinal-transverse coupling, which plane-based α_TP excludes (Lester 2023 p. 3).
- B&dD's text claims convergence "around the advection time" and "at 10 times" (p. 2491 [9]); "10–30" is a figure read-off, not their claim.

**F18 (MINOR). The `numerical` outcome is over-read (§3.d).** The review says a `numerical` outcome would support "Lester et al.'s attribution". Only an E_h (velocity representation) or E_int (integration) artifact would; a width (E_W) artifact is not Lester's mechanism.

## What survived, and how I checked it

| claim or citation | how verified |
|---|---|
| First-order formula, α_T > 0 at every finite d in an unbounded medium, d⁻² tail, 7.992e-5, halves ratio 3.57, distance ratio 6.92 | Derived independently: α_T/(σ²λ) = (τ/2)∫₀¹μ²(1−μ²)e^{−τ²μ²}dμ, which is ≥ 0 everywhere; `X22(∞) = σ²λ²/3`; tail (√π/8)σ²λ³/d². Ran the script (`-B`) and reproduced every number |
| H3's null is conservative | All in-box values are below it (F1 table) |
| Displacement basis has the same expectation (uniform weights) | E cov(x0, Δ) = 0 by transverse homogeneity of the periodic field with periodic y/z flow; weights are fixed per particle |
| Attinger, Dentz & Kinzelbach 2004 | Crossref; full text from the ETH copy: §5 p. 11 eqs. 9-10 (hole effect); §7 pp. 12-13 (higher orders do not cancel; linearized fields; credits Neuman & Zhang) |
| Neuman & Zhang 1990; Janković 2009 (quote verbatim); Janković, Fiori & Dagan 2003 ("small but persistent"); Dentz 2000a,b; Gelhar & Axness 1983; Sposito 1994 and 2001 | Crossref plus OpenAlex abstracts. Neuman & Zhang also exists under the duplicate DOI 89WR03602 |
| Lester 2021 | Local PDF: A27-1, A27-7 (ℓ ~ t^r), A27-18 eqs. 4.12-4.14 ("from sublinear to ballistic"), A27-24 eqs. 5.41 and 5.43, A27-25 eq. 5.46 |
| Lester 2022 | Cambridge open-access PDF: eqs. 4.23-4.25 on p. A18-14 (F22 a bounded ratio; infinite-time λ = 0); eq. 6.12 on p. A18-23 (0 < r < 2) |
| Lester 2023 | Local PDF: §5.3 and eq. 36 on p. 13; Fig. 4b on p. 14 (variance ∝ √tol). Also confirms H19 and m2: eq. 20 has an overbar (p. 8) and Fig. 3c is on p. 12 |
| B&dD 2013 | Local PDF: eq. 1; p. 2490 [6] periodic transverse faces, Euler, linear interpolation, about 10 steps per cell, 0.8 W flux-weighted source at ≥ 5λ; p. 2491 N_p 1e4, N_R 500, ≤ 4 %; eq. 5; Tables 1 and 2 (<1e-3 / 0.01 / 0.044); eq. 7; Fig. 4b about 0.11 at σ² = 4; Fig. 2b above the table value at t ≈ 10 |
| Dentz et al. 2016 (Phys. Rev. Fluids) | arXiv 1608.02208v1: eqs. 9, 10, 16, 17; flux injection makes the s-Lagrangian statistics stationary from the start |
| Shapiro & Cvetkovic 1988 (corrected DOI); Dagan, Cvetkovic & Shapiro 1992 (91WR03086); Demmy 1999; Janković & Fiori 2010 (about 10× at 100 I, σ² = 8; the abstract says "dispersivity", not "arrival-time variance") | Crossref plus abstracts |
| Berger 1982 DOI correction | Crossref: …10487790 is Berger; …10487759 is a Lucas & Crosier CUSUM paper |
| Friede & Kieser 2003 (slight inflation for equivalence); Kieser & Friede 2003 (none for superiority t-tests) | Abstracts |
| Variance-normalized generator's conditional ensemble | Negligible: realized variance SD ≈ √(3.94λ³/V) = 0.8 % at W 24, 1.6 % at W 12; mean bias about 1e-4 relative |
| Width-12 periodic images | Handled as the review says, through unwrapped coordinates plus E_W |

Not checked: most statistics citations (Hall, Hauschke, Fieller, Glynn & Heidelberger, Jones & Nachtsheim), the numerics citations, the FDA 2019 guidance and Dartois 2018. My in-box first-order numbers leave out the Dirichlet-face correction near the outlet.

Main risk: if H3 and H17 are pre-registered as written, the descriptive decay comparisons will be made against benchmarks that do not hold in SF-43's boxes. Recomputing them per arm takes seconds on a CPU.