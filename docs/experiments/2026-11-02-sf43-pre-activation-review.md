# SF-43 pre-activation review: scientific feasibility, fidelity to the literature and a design that fits the deadline (literature, code reading, own calculations, adversarial reviews)

- Date: 2026-11-02
- Status: `complete — design review before activation; NOT SF-43 evidence; no GPU run, no build, no SF-43 data; calculations are numpy/scipy on CPU and readings of committed SF-39/SF-42 data`
- Increment: [`SF-43-confirmatory-macrodispersion-study.md`](../plans/active/lester-eq14/increments/SF-43-confirmatory-macrodispersion-study.md)
  (State `pending`; checker on `master` `ef87a1bc`: `OK (44 increments, ready=SF-34 SF-43, nonterminal=none)`)
- Decision fed: [`2026-11-02-sf43-pre-activation-decisions.md`](../decisions/2026-11-02-sf43-pre-activation-decisions.md) (H1-H20)
- Precedents: [`2026-10-27-sf39-pre-activation-review.md`](2026-10-27-sf39-pre-activation-review.md) (format);
  [`2026-10-29-sf39-remediation-sf42-sf43.md`](../decisions/2026-10-29-sf39-remediation-sf42-sf43.md) (G1-G11, W1-W13)
- Inputs: SF-39 results and post-review evaluation ([`2026-10-27-sf39-macrodispersion-study.md`](2026-10-27-sf39-macrodispersion-study.md));
  SF-42 tooling, verification, pilots and findings (a)-(j) ([`2026-10-30-sf42-confirmatory-tooling.md`](2026-10-30-sf42-confirmatory-tooling.md));
  the owner's brief of 2026-11-02 (deadline: analysed results by 2026-11-09 00:00 local, UTC-3; a two-GPU first stage of at most 30 h)
- Theory: [`lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md), [`beaudoin-de-dreuzy-2013-key-claims.md`](../theory/beaudoin-de-dreuzy-2013-key-claims.md)
- Artifact: [`artifacts/2026-11-02-sf43-pre-activation-review/`](artifacts/2026-11-02-sf43-pre-activation-review/): verified
  [`bibliography.md`](artifacts/2026-11-02-sf43-pre-activation-review/bibliography.md) and [`references.bib`](artifacts/2026-11-02-sf43-pre-activation-review/references.bib),
  [`scripts/`](artifacts/2026-11-02-sf43-pre-activation-review/scripts/README.md)
- Method: eleven independent read-only analyses (seven literature topics including the full reading of the three local PDFs; costs and
  schedule; Gate 4 power and budgets; Gate 5 and coverage; executability against the code), each re-checked by the orchestrator against
  the committed data before use; then five adversarial reviews of this draft (section 7).

## 0. Summary for the owner

- **The design of the specification does not fit and does not need to.** On the lattice SF-42 found admissible it costs 147 GPU-h
  anchored, 162 with the measured job overhead and 185 with every pessimistic factor (cost audit; section 4.a); with the measured
  overhead it would not finish before the deadline from any realistic launch time, and it exceeds G10. Almost all of that cost buys
  position-basis precision (production and width arms at 1100-1650 realizations at `sigma^2` 0.25 and 1).
- **The displacement basis measures the same quantity with 1.3 to 1370 times less variance.** Subtracting each particle's seed position
  removes a term (the slope of the seed-displacement covariance) whose ensemble mean is zero; the SF-42 noise pilot gives SD ratios
  37 / 9.8 / 3.7 / 1.14 at `sigma^2` 0.25 / 1 / 2.25 / 4. With it every precision target is met at `N` of 30 or less, Gate 4's power is
  1.00 at every amplitude (section 3.d), and `sigma^2 = 0.25` becomes resolvable for the first time.
- **Recommended plan (variant A):** about 54 GPU-h planning (74 with every pessimistic factor), 37-49 h of wall with the two-GPU first
  stage (17-22 h on two GPUs, then one), finishing between Thursday evening and Friday morning (2026-11-05/06) for a Wednesday-morning
  launch, with 50-80 GPU-h of slack before the
  pre-registered cutoff (Sunday 16:00). It keeps every arm of the specification, adds a cheap coarse rung at `sigma^2 = 4`, and
  loses only position-basis precision (reported, not confirmatory) and the finer `E_h` level at 2.25 (replaced by a better-conditioned
  coarser one).
- **Four design problems are fixed before data:** a sign null that linear theory already rejects (H3); a Gate 5 with no qualified
  procedure (H5, solved by the Fieller form with existing coverage evidence plus a 2-hour CPU run); `E_h` gaps against the verification
  standards and a 12-fold fallback at 2.25 (H6); undefined levels of G5.7's classes (H16).
- **Alternatives for the owner:** B (faster, 38 GPU-h, Thursday) and C (more budget precision and a decay test at `sigma^2 = 4`, 88
  GPU-h, Saturday, least slack).

## 1. Question, scope and date of knowledge

**Question.** Before SF-43 is activated and pre-registered: (a) does each planned experiment answer a well-posed question, with a
well-defined estimand, a bounded error and an informative result under every possible outcome; (b) is the design faithful to the
literature it will be compared with; and (c) which design delivers the most scientific information before the owner's hard deadline
(analysed results available by Monday 2026-11-09 00:00 local, UTC-3), with every loss against the full design declared and
quantified?

**Hypothesis tested.** None about `alpha`. The review hypothesizes only that some design items are ill-posed, uninformative or
unaffordable for derivable reasons (a null that cannot discriminate, an arm whose information per GPU-hour is low, a procedure that does
not cover), which is checked by reading, derivation and simulation on CPU.

**Scope.** No SF-43 data exist and none were generated: no SF-43 seed was run, nothing was submitted to the V100 for this review (one
short `scripts/remote exec` read the host's free disk, `df` and `du`, no CUDA). Every number is a design calculation, a literature value,
or a reading of SF-39's and SF-42's committed data. No `alpha` claim is made or implied (`docs/theory/lester-2023-key-claims.md`).

**Date of knowledge (integrity statement).** This review was written after SF-39's results at `sigma^2` 0.25, 1, 2.25 and 4 and SF-42's
pilots were known and committed (PR #83, `104cc254`; PR #85, `ef87a1bc`). As in decision 2026-10-29, those data are used only as pilots:
variances, pairing correlations, costs, memory, failure modes and the per-basis estimator noise (SF-42 section G, R3). Their means are
read only where a design quantity needs them (the planning means of G5.2; the first-order comparison of section 2.1, which uses the
SF-39 means as published). Three design changes proposed here were prompted by those readings and are disclosed as data-informed:
(i) the displacement basis as the primary transverse estimator (it rests on the SF-42 SD ratios, not on any mean); (ii) shifting the
robust-sign null from 0 to the first-order value `alpha_T^FO` (it rests on theory; it costs no power at the pilot's magnitudes, which
were known); (iii) admitting `sigma^2 = 0.25` to the primary family (made possible by (i)). (i) keeps the estimand and the validity of
every decision rule; (ii) makes each null harder to reject; (iii) adds a hypothesis to the Holm family, which lowers every other Holm
level slightly, and makes a confirmatory class reachable at 0.25 that G5.7 had excluded because the position basis could not resolve it:
it is the one change that enlarges what can be claimed, and the owner decides it (H4).

## 2. Literature: what is established, what is disputed, what is comparable

Every claim below is in the verified bibliography
[`artifacts/2026-11-02-sf43-pre-activation-review/bibliography.md`](artifacts/2026-11-02-sf43-pre-activation-review/bibliography.md)
with its DOI (checked against Crossref/OpenAlex), its locator and the level at which it was read: **FT** full text, **A** abstract,
**M** metadata only. Most WRR/JFM/Springer full texts were blocked to the agents (Cloudflare 403); where only the abstract was read the
claim is limited to what the abstract states. Equation and page numbers are given only where the full text was read.

### 2.1 Theory of 3D purely advective transverse macrodispersion (isotropic media)

- **First order predicts a positive `alpha_T` at every finite distance, not zero.** Own quadrature of the first-order Lagrangian result
  for `C(r) = sigma^2 exp(-(r/lambda)^2)` (the formula of the SF-39 pre-activation review, section 5.1, independently re-derived by the
  theory agent in wave-number form): `alpha_T(x) ~ (sqrt(pi)/8) sigma^2 lambda^3 / x^2` far from the source, `X_22(inf) = sigma^2 lambda^2
  / 3`. For SF-43's estimand (half the OLS slope of `X_22` over planes `d` in `[25, 94] lambda`):
  `alpha_T^FO = 7.99e-5 sigma^2 lambda` (halves `[25, 59.5]`: 1.39e-4; `[59.5, 94]`: 3.89e-5; distance window `[94, 194]`: 1.16e-5,
  all in units of `sigma^2 lambda`; script `firstorder_window.py`). This is the unbounded-domain, one-particle value; periodic boxes
  (no transverse wave numbers below `2 pi / W`) and per-realization centring make it smaller. **Consequence: a positive sign alone does
  not discriminate between linear theory and a nonlinear transverse dispersion.** What discriminates is the size relative to
  `alpha_T^FO` (SF-39 pilot: 75 / 155 / 205 times at `sigma^2` 1 / 2.25 / 4) and the dependence on `d` (first order: 3.6 times smaller in
  the second half of the window than in the first, 6.9 times smaller on `[94, 194]` than on `[25, 94]`).
- **Lowest-order asymptote is zero** (Attinger, Dentz & Kinzelbach 2004, SERRA 18, sect. 5, eqs. 9-10, p. 11, FT: hole effect of the
  transverse velocity correlation; Gelhar & Axness 1983, A; Dagan 1984/1988, A). **Beyond first order in 3D no theorem forces zero**:
  Attinger et al. 2004 (sect. 6-8, FT) argue higher orders do not cancel in 3D (finite asymptote; their simulations use linearized Darcy
  fields, not `k grad phi`); Neuman & Zhang 1990 (quasi-linear, exponential covariance, A): nonzero asymptote proportional to
  `sigma_Y^4` for small `sigma_Y^2`; Janković, Steward, Barnes & Dagan 2009 (inclusions, A): `alpha_T > 0` for randomly oriented
  ellipsoids, zero for dilute spheres, "the two-point covariance ... is not able to detect the structures that produce advective
  mixing". In 2D the zero is exact at all orders (Attinger et al. 2004, FT; Dagan 1994, A).
- **Effective vs ensemble.** SF-43 centres moments per realization (the effective definition); ensemble coefficients overestimate the
  effective ones at finite time and finite source (Dentz, Kinzelbach, Attinger & Kinzelbach 2000a,b, A); with a source of 19.2 `lambda`
  (about 22 integral scales) the two should be close (Cvetkovic, Shapiro & Dagan 1992, A: about 20 integral scales is "almost
  ergodic" for cumulative flux). Not checked for plane statistics.

### 2.2 Topological constraints, helicity and chaos (relevant to W8)

- **Smooth, locally isotropic, stagnation-free Darcy flow is helicity-free and integrable; persistent exponential stretching is
  excluded.** Sposito 1994 (A) and 2001 (A); Lester, Dentz, Bandopadhyay & Le Borgne 2021, JFM 918 A27 (local PDF, FT: pp. A27-1,
  A27-7 algebraic growth `l(t) ~ t^r`, A27-18 eqs. 4.12-4.14 two superposed 2D shear flows, "from sublinear to ballistic"); Lester et
  al. 2022, JFM 945 A18 (FT, open access: eqs. 4.23-4.25, transverse stretch a bounded ratio of local quantities, Lyapunov exponent
  exactly 0; eq. 6.12 shear `|F_12| ~ t^r`, `0 < r < 2`). Helical/whirling flow requires anisotropy (Bakker & Hemker 2004, A; Hemker &
  Bakker 2006, A; Chiogna et al. 2014/2015, A). Pore-scale flow is chaotic (Lester, Metcalfe & Trefry 2013, A; Lester, Dentz & Le
  Borgne 2016, A).
- **Finite-time growth and numerical drift are not excluded.** Ballistic shear (`~t^2`) and large bounded swings of `F_22` in slow
  zones can look exponential over a finite window (inference from Lester 2022, eqs. 4.23-4.25 and 6.12). Tolerance-controlled
  Runge-Kutta on an exactly helicity-free field produces zero-mean trajectory errors whose variance grows like `sqrt(tol)` and is
  misread as dispersion (Lester, Dentz, Singh & Bandopadhyay 2023, WRR 59, local PDF, FT: section 5.3, eq. 36, Fig. 4b, p. 13-14);
  Lester et al. 2025 (JFM 1018 A35, arXiv v1, FT: App. D) report spurious transverse dispersion from Cash-Karp RK at 1e-14 on a
  5th-order spline velocity in the isotropic case. Ensemble statistics can survive sensitive individual trajectories (shadowing:
  Hammel, Yorke & Grebogi 1988, A; Sauer, Grebogi & Yorke 1997, A; Haller 2015, A).
- **Lester's own pre-asymptotic prediction.** Lester 2021 eq. 5.43 (FT, p. A27-24/25): `sigma_c^2 ~ sigma_c,0^2 + sigma_psi^2 (1 -
  exp(-x_1/zeta))`, a transverse spreading that **saturates**, so the finite-distance `alpha_T` decays with distance (this rests on an
  assumed exponential autocorrelation, eq. 5.41). Lester 2023 p. 22 states its findings apply "to both the pre-asymptotic and asymptotic
  regimes".
- **Answer for W8 (section 3.a below):** exponential pass-to-pass separation is not by itself a contradiction (finite-time or
  numerical), the `h/8`-`h/16` agreement bounds only the step-cap error, and the discriminating evidence is the dependence of
  `alpha_T` on `d` and on the representation, not the sign.

### 2.3 Numerical 3D studies at moderate and high `sigma^2`

| study | dim, covariance | `sigma^2` | width, BC | estimator, window | `N` | transverse result | read |
|---|---|---|---|---|---|---|---|
| Beaudoin & de Dreuzy 2013 (WRR 49) | 3D, **Gaussian** eq. 1 | 0.25-9 | 25.6 `lambda`, **periodic** | resident spatial moments, centred per realization (eq. 4), time; `alpha_kAM` over the last quarter of `[0, t_b]` (eq. 5); `alpha_LAF` fit eq. 6 | 500 (`N_p` 1e4) | `< 1e-3` / 0.01 / 0.044 at 0.25 / 1 / 2.25 (Table 2); about 0.11 at 4 (Fig. 4b read-off); `~ sigma^4` (eq. 7) | FT |
| Dartois, Beaudoin & Huberson 2018 (C. R. Méc.) | 2D/3D, exponential (stated) | <= 2.25 | 25.6 `lambda`, not stated | velocity covariance and moments | 100 | figures only | FT |
| Janković, Fiori & Dagan 2003, 2006 (WRR) | 3D inclusions (not multi-Gaussian) | 2-8 | AEM | velocity covariance (2003); flux at control planes, travel-time variance (2006, `alpha_L` only) | 3 (2006) | "small but persistent" | A |
| Chin & Wang 1992 (WRR 28) | 3D | <= 2.25 | n/v | confidence limits | n/v | first order valid only for `sigma_Y < 1` | A |
| de Dreuzy, Beaudoin & Erhel 2007 (WRR 43) | 2D, exponential | 0.25-9 | no-flow | resident, effective | 100 | 0 on average | FT |

Corrections to the brief's candidate list (verified): Salandin & Fiorotto 1998 and Hassan et al. 1998 are 2D; Englert et al. 2006
reports velocity statistics only; Dentz et al. 2002 and Schwarze et al. 2001 do not solve Darcy's equation. **No published study
varies the transverse width in 3D, gives a confidence interval for a 3D Gaussian `alpha_T` at `sigma^2 >= 1`, or uses control-plane
estimators for `alpha_T` in 3D.** B&dD's only sampling statement is a +-4 % sensitivity at `sigma^2 = 6.25`. SF-43's 12-`lambda`
width at `sigma^2 = 4` has no precedent.

**B&dD is the design mirror of SF-43** (Gaussian covariance, periodic transverse faces, `W` about 24-25.6 `lambda`, flux-proportional
injection over `0.8 W` at least 5 `lambda` from the inlet, `lambda/Delta = 10`, effective moments) but **not its numerical mirror**:
explicit Euler on face-linear (Pollock-type) interpolation, about 10 steps per cell, which is the scheme Lester et al. 2023 (p. 20, FT)
hold responsible for their nonzero `alpha_T`.

### 2.4 Control-plane estimands against spatial moments

- **`alpha_LP = (U^2/2) d var(t)/dd` is exact for 1D Fickian first passage** (inverse Gaussian: `var tau = 2 D d / U^3`; own numerical
  check) and, at first order in a heterogeneous medium, `alpha_LP(d) = alpha_L(t = d/U)` when `d >> alpha_L` (own derivation; Shapiro &
  Cvetkovic 1988, A: arrival-time variance linear far from the source; Dagan, Cvetkovic & Shapiro 1992, A; Kreft & Zuber 1978, M).
- **The condition `d >> alpha_L` fails at `sigma^2 = 4`:** with B&dD's `alpha_LAF` (0.24 / 1.27 / 4.35 / 12.7 `lambda`) the window
  `[25, 94] lambda` is `d/alpha_L` = 100-390 / 20-74 / 6-22 / **2-7** at `sigma^2` 0.25 / 1 / 2.25 / 4. The SF-42 J3B pairing pilot
  is consistent: plane-based `alpha_LP` about 10 `lambda` at `sigma^2 = 4` against SF-39's spatial `alpha_LAM` of 4.65 in its time
  window. **`alpha_LP` at 4 (and partly at 2.25) is a different finite-distance quantity, weighted by the slow tail
  (`E_s[1/v^2]`), not comparable with B&dD's spatial `alpha_L`.**
- **Travel time samples velocities in space (s-Lagrangian), i.e. flux-weighted** (Dentz, Kang, Comolli, Le Borgne & Lester 2016, PRF,
  FT arXiv: eqs. 9, 10, 16, 17): with flux-proportional injection the s-Lagrangian statistics are stationary from the start, which
  supports flux weights and `U_r = Delta d / Delta <tau>_r` (flux-weighted mean travel time); Demmy, Berglund & Graham 1999 (A) and
  Janković & Fiori 2010 (A): resident injection inflates arrival-time variance by up to about 10 times at `sigma^2 = 8`.
- **Transverse at planes:** at first order `var eta(d) = X_22(d/U)` (own derivation); the transverse crossing position depends on
  streamline geometry only (Dagan, Cvetkovic & Shapiro 1992, A).
- **Comparability with B&dD:** only `alpha_TP` (fixed axes) against their `alpha_TAM` if both are Fickian over the window; `alpha_L` only
  at `sigma^2 <= 1`; nothing at 4 except a plane-based secondary of B&dD's window, which SF-43 records descriptively.

### 2.5 Verification of the discretization budget

- **G5.4's `E_h` is the standard fine-grid GCI** (`Fs |e| / (r^p - 1)`; Fs = 1.25 with three grids and an observed order, Fs = 3 with
  two; NASA NPARC page, FT; Roy 2005, FT, sect. 3.4.3; Celik, Ghia, Roache & Freitas 2008, FT, eqs. 1-7) **and the multilevel Monte
  Carlo bias estimator** `E[P - P_L] ~ E[P_L - P_{L-1}] / (2^alpha - 1)` with coupled (common-random-number) level differences (Giles
  2015, FT, sect. 3.1; Cliffe, Giles, Scheichl & Teckentrup 2011, FT, which also requires monotone asymptotic decay). The fallback
  `3 max|diff| / (r - 1)` equals Eça & Hoekstra's 2008 rule for oscillatory convergence (`3 Delta_M`, FT) and Roache's suggestion (Fs =
  3, `p = 1`).
- **Four gaps against the standards** (the pre-registration closes them, decision H6): no sign check for oscillation (Celik eq. 3c);
  Fs = 1.25 presupposes agreement with a known formal order, which is not established for a dispersivity ensemble mean (Roy 2005), so the
  order must be capped (Eça & Hoekstra 2014, FT: `0.5 <= p < 2.1`); three levels leave no residual to test the asymptotic range (Eça &
  Hoekstra 2014 require at least four); the finest-level branch uses one difference where MLMC takes the maximum over the last
  extrapolated differences (Giles 2015, FT). When the production level is the finest (`sigma^2 = 4`), the estimate rests on the
  asymptotic range: "far outside the asymptotic range no choice for the factor of safety is sure to be conservative" (Roy 2005, FT).
- Smooth (Gaussian-covariance) fields admit higher weak convergence rates than exponential ones (Teckentrup, Scheichl, Giles & Ullmann
  2013, FT preprint: analytic coefficient; Charrier et al. 2013, A); SF-42 E measured second-order FV flux error from `lambda/h = 4` at
  `sigma^2` up to 4.

### 2.6 Statistics

- Holm 1979 (JSTOR 4615733, no DOI); Berger 1982 (intersection-union; correct DOI `10.1080/00401706.1982.10487790`); Schuirmann 1987
  and Berger & Hsu 1996 (TOST; each one-sided test at the full level); Clopper & Pearson 1934.
- **Paired ratio-of-means equivalence:** the Fieller/Sasabuchi form tests, for each margin `theta`, the mean of `D_r(theta) = L_r -
  theta P_r` (Hauschke, Kieser, Diletti & Burke 1999, A; Locke 1984, A; Fieller 1954, A); it keeps the ratio of arithmetic means as the
  estimand and removes the delta-method linearization, but leaves a one-sample test on a possibly skewed difference.
- **Skewness and small tails:** bootstrap-t is second-order accurate for one-sided limits (Hall 1988, A); choose `B + 1` a multiple of
  `1/alpha` (Hall 1986, A); Johnson's modified t and Hall's 1992 transformation are alternatives (A); **no reference guarantees
  one-sided coverage at `alpha` about 1e-3 under heavy skewness** — qualification has to come from simulation (Morris, White &
  Crowther 2019, FT: Monte Carlo SE of coverage).
- **Blinded sample-size re-estimation:** no type I inflation for superiority t-tests (Kieser & Friede 2003, A), **slight inflation for
  equivalence/TOST** (Friede & Kieser 2003, A; FDA 2019 guidance, FT p. 10: "sponsors should evaluate the extent of inflation").
- **Wall-clock budgets:** with a time budget, the mean of completed runs is biased only at `o(1/t)` when runtime and outcome are
  correlated, and the unbiased estimator needs a pre-selected set of runs (Glynn & Heidelberger 1990, 1991, A); missing-not-at-random if
  completion depends on outcome-correlated runtime (Rubin 1976, M). Remedy adopted (decision H13): a pre-registered confirmatory seed
  list per arm, scheduled first, with completion near-certain.
- **Split-plot:** analysing a split-plot as completely randomized inflates the type I error of whole-plot factors (Jones & Nachtsheim
  2009, FT pp. 344-345); resolution IV aliases two-factor interactions in pairs (Montgomery; Box, Hunter & Hunter, M).

## 3. The owner's questions, answered with evidence

### 3.a Estimands

- **Well defined.** Per realization: the flux-weighted, per-realization-centred moments of the first crossing of each plane `x0 + d`
  (the tracker lands the planes in order, so it records first passage; `LagrangianTrackerDevice.cuh` plane pass), `d` = 25, 26, ..., 94
  `lambda`; `alpha_TP` = half the OLS slope of `(var x2 + var x3)/2`, `alpha_TAP` the same orthogonal to the realization's mean crossing
  displacement, `alpha_LP = (U_r^2/2) slope[var tau]` with `U_r = 1 / slope[mean tau]` (`sf42red/reduce.py:28`, :248-250), the
  flux-weighted Delta-d over Delta-mean-time that the travel-time literature supports (section 2.4); ensemble estimand = mean of the
  per-realization values. The estimand of `alpha_LP` differs from `(Ubar^2/2)` times the mean slope by about `3 CV(U_r)^2`; both are reported
  with `CV(U_r)`.
- **The displacement basis (H2)** is the same estimand: `var x(d) = var x0 + var Delta(d) + 2 cov(x0, Delta(d))`, `var x0` is constant in
  `d`, and `E cov(x0, Delta(d)) = 0` exactly for uniform weights by transverse stationarity of the periodic field (the displacement of a
  seed has the same law at every seed position), to `O(1/n_eff)` for flux weights. Pilot means agree within the position-basis SE at every
  amplitude (Gate 4 analysis, item 12). The position basis stays a reported secondary.
- **Comparable with the literature:** `alpha_TP` (fixed axes) with B&dD's `alpha_TAM` only as an order of magnitude (their estimator:
  time moments in `[0.75 t_b, t_b]`, `t_b` the first exit; their Fig. 2b shows `alpha_T(t)` still above the tabulated value at `t` about
  10, so time and distance windows differ in phase); `alpha_TAP` and `alpha_LP` have no counterpart; `alpha_LP` against spatial `alpha_L`
  only where `d >> alpha_L`: yes at 0.25 and 1 (`d/alpha_L` 100-390, 20-74), partly at 2.25 (6-22), **not at 4 (2-7)**: there
  `alpha_LP` (about 10 `lambda` in the J3B pilot) is a slow-tail travel-time quantity, not B&dD's spatial `alpha_L` (4.65 in SF-39's time
  window). W6's remaining gap is addressed only through the secondary eq. 6 fit on the distance arms (G9b).

### 3.b Domain assumptions

| assumption | effect on the conclusion | how SF-43 handles it |
|---|---|---|
| periodic transverse boundaries, width 24 `lambda` (12 at 4) | the field has no transverse wave numbers below `2 pi / W`; moments use unwrapped coordinates, so images do not fold the plume (SF-39 post-review); at `sigma^2 = 4` the transverse displacement SD at `d = 94` (about 3.5 `lambda`) is comparable to the half-width, so the width enters | measured: `E_W` (12 vs 24 at 4; 24 vs 48 elsewhere) and the width x `lambda/h` interaction; beyond the widest width: stated assumption. No published 3D study varied the width (section 2.3) |
| source 0.8 x width (19.2 / 19.08 / 9.6 `lambda`) | sets the ergodicity of per-realization moments (about 22 integral scales; 11 at 4) and the effective (centred) estimand; fixed across arms of an amplitude so width arms change only the box | declared; the 0.6 % change at 2.25 (H10) is negligible against the 23 % width effects SF-39 saw |
| medium (b): Dirichlet `x` faces | boundary layers of about 4 `lambda` (SF-38); the window starts 25 `lambda` after injection and ends 5 `lambda` before the outlet; Dirichlet faces converge at order 2 (SF-42 D) | declared model restriction (decision 2026-10-29); not varied |
| SF-18 generator (periodic, variance normalized) | conditional ensemble at realized variance `sigma^2`: slightly smaller ensemble variance; mean could differ from a non-normalized generator (B&dD, Par2) | G7's factor G measures the generator effect on Par2's `alpha_LP` at 0.25 and 1; otherwise declared |
| finite distance `[25, 94] lambda` | first order predicts decay `~ d^-2` (3.57 x between halves); Lester 2021 predicts saturation; B&dD claim a plateau after `t` 10-30 | no asymptotic claim; halves reported as an interval; distance arms `[94, 194]` against the first-order ratio 6.92 |

### 3.c The error budget G5.4

It is standard practice: the fine-grid GCI and the multilevel Monte Carlo bias estimator with coupled levels (section 2.5). The
analysis of the SF-39 data with `mcstat`'s own classes (Gate 4 analysis, sections 1 and 4) adds a decisive fact: on the production box
at `sigma^2 >= 1` the per-seed grid differences are at the particle-noise floor (variance ratio 0.72-1.07), so the mean differences are
not resolved, `p_L = -inf`, and `E_h` almost always takes the fallback `3 UCB(range) / (r_min - 1)` (order branch in 0 % of replicates at
2.25 and 4, 1-10 % at 1, 51-100 % only at 0.25). Consequences: (i) the "production finest" question at `sigma^2 = 4` is mostly moot: the
operative bound is a data-range bound that does not use the asymptotic-range assumption; when the order branch does fire, H6 caps the
order at 2, takes the smaller lower limit of two triples (8/16/32 and 4/8/16) and the Giles maximum; (ii) at 2.25 the specification's
alternative 8/16/20 is the worst choice (no constant ratio, fallback multiplied by `3/(1.25 - 1) = 12`; `E_h` 0.31 of the mean against
0.19-0.32 for 4/8/16, which is also seven times cheaper); (iii) `E_int` is at most 1e-6 relative at every amplitude (negligible, kept
for completeness); (iv) `E_W` is the dominant term at 0.25 and 1 in the displacement basis (0.08-0.16 of the mean) and `E_h` at 2.25
and 4 (0.11-0.31); (v) a width deficit counts twice (it biases the production mean and enters `E_W`), which is conservative by design.
The transfer assumptions that remain stated: `E_W` from `lambda/h` 10 (2.25) and 16 (4) to production; beyond the widest width; the
asymptotic range only if the order branch is used at 2.25 or 4.

### 3.d Power, informativeness and pre-mortem

Gate 4 (simulated with `mcstat`'s classes, R = 400, B = 4000; scenarios S1 order-2 biases from SF-39, S1w5/S1w10 production width 5 /
10 % short, S2 adverse non-monotone ladder, S3 order-1 bias; two noise models at 2.25 and 4):

| design | P(robust limit > 0) at the worst Holm level | MDE / planning mean |
|---|---|---|
| specification (position basis) | 0.25: 0.00; 1: 1.00; 2.25 (8/16/20): 0.81-1.00; 4 (N 30): 0.14-1.00 | 2.4 / 0.46 / 0.81 / 0.70-1.04 |
| displacement basis, `N` 100, arms 100 | 1.00 at every amplitude and scenario except S3 order-1 at 4 under the pessimistic noise model (0.69-0.90) | 0.28 / 0.23 / 0.33-0.48 / 0.38-0.56 |

The recommended variant A (production 150, ladders 100-150, width cells 150) is at least as strong. With the null at `alpha_T^FO`
(H3) nothing changes at these magnitudes (the pilot means are 25-200 times `alpha_T^FO`).

Gate 5 (Fieller form, `n` of H5): power at ratio 1 with the conservative pairing inputs 1.00 at 0.25, 1, 2.25 (`n` = 150) and about
0.9 at 4 (`n` = 250); the largest true ratio with power 0.8 to `agree` is about 1.08-1.09 at 0.25-2.25 and 1.03 at 4.

| outcome | what we learn |
|---|---|
| Gate 4 `physical (restricted)` at 0.25-4 (expected; P about 1) | `alpha_T` exceeds the linear-theory value at this distance, robustly to discretization, width and integration within the model, medium and window; with the halves and the distance arms, whether it decays as first order / Lester predict or persists as B&dD claim |
| `numerical` (only under an artifact truth) | the transverse growth is mostly a discretization or width artifact: Lester et al.'s attribution of published `alpha_T` to numerics would be supported in this model |
| `unresolved (bounded)` | an upper bound; the split of `B_k` says which of three causes (8->16 noise growing as `h^2`, a 12->24 width effect, an order-1 bias at 4) limits it |
| `unresolved` | estimator discordance, practically unreachable (`alpha_TAP` = `alpha_TP` within 4e-4) |
| Gate 5 `agree` (likely at 0.25-2.25: matched pilot ratios 1.001 / 1.007 / 1.015) | `alpha_L` validated against an independent code on the same field; SF-39's offsets then belong to the unmatched factors, which G7 apportions |
| Gate 5 `disagree` (plausible at 4: pilot ratio 1.14 on 8 pairs, one field drives it) | the travel-time `alpha_LP` at 4 depends on the velocity representation in slow zones; the consumer-twin / Pollock-twin / Par2 decomposition says whether it is the representation or the stepping; SF-43 stays active until attributed or decided by the owner (specification) |
| Gate 5 `inconclusive` | an offset near the margin or a wider CV than planned; reported with achieved power |

No arm is uninformative in the displacement basis. In the position basis every `sigma^2 = 0.25` arm was (P(RL > 0) = 0).

### 3.e Gate 5

SF-42 Finding (j) is resolved without the 42-hour re-run (H5, H15): the Fieller form tests one-sample means of `L - 1.10 P` and `L -
P/1.10`; at the margin their distribution is exactly the `stress` variable of the SF-42 `sampling` cells, independent of the pairing
correlation and the CVs, so existing evidence applies: bootstrap-t qualifies at the Holm levels from `n` = 200 and at half the Holm
level at `n` 100, 200 and 500; Student never does. A skewness-corrected test (Hall 1992 transformation; prototype rate/tail <= 1.08 at
`n >= 150`, exact on normal data) is analytic and is qualified by a 45-core-hour run (about 1.5-2 h on 32 workers) in parallel with the
smoke test. The ordered rule (Hall, then bootstrap-t, then half-level bootstrap-t) always ends in a qualified procedure at the chosen
`n`, so the specification's "no qualified procedure keeps SF-43 active" cannot be triggered by Gate 5. Alternatives considered and not
recommended: the geometric-mean ratio (it shifts the estimand by 24-37 % of the margin at `sigma^2 = 4`); removing Gate 5 from the
family (gives up W1's confirmatory remedy); serial gatekeeping behind Gate 4 (tests Gate 5 with probability about 0.5).

### 3.f SF-42's open decisions

| decision | options | recommendation | consequence |
|---|---|---|---|
| budget and design on the admissible lattice (147.4 anchored, 162.3 with overhead) | keep the design and the budget; cut arms; change the basis | change the basis (H2), keep every arm, G10 unchanged as the hard cap | 54 GPU-h planning, 74 pessimistic |
| seed lattice at 2.25 (no common `M` at 19.2 `lambda`) | 19.08 or 19.32 `lambda`; per-arm `M`; an `E_W` `lambda/h` without a factor 5 | 19.08 `lambda`, `M` = 100 (H10) | one common source discretization; 0.6 % smaller source |
| 3x PCG rule (1040x240x240 at 0.25, 2080x240x240 at 1) | other grids; accept | accept (H11) | converged solves, measured cost; no pilot invalidated |
| Par2 horizon | 1.2 (censored 0.2-3.8 %); 3.6 | 3.6 with a per-realization completeness flag (H12) | +8-17 % Par2 time |
| Gate 5 `n` floor | 30 (J3B scenario); coverage-driven | 150 / 150 / 150 / 250 (H5) | 4.7 GPU-h for Gate 5 |

### 3.g Operational feasibility of the single job

- GPU-h and wall: variant A 53.5 GPU-h planning (`design_cost.py`), 56 with the 5 % reserve; stage 1 (two GPUs, heavy queue 33 GPU-h)
  16.7-22.0 h, stage 2 20-27 h; 37-49 h of wall for the planning to pessimistic margins (overhead 1.10, model under-prediction 1.32).
  With one GPU only (fallback): 59-70 h. Both fit the window from a Wednesday launch to the Sunday 16:00 cutoff (106 h).
- Memory: near-empty-GPU grids 2080x480x480 (about 29.2 GB for multi-realization runs) and 3328x384x384 (29.9 GB) on lane A; half-class
  grids 1040x480x480 (about 15.4 GB) and 1664x384x384 (15.7 GB) on lane B, which therefore needs only 17 GB free (H14). A colleague
  taking memory after launch makes a unit fail with OOM; it is requeued with backoff, not counted as a failure.
- Disk: raw output about 0.5 TB for variant A without pruning (consumer 45-230 MB per realization; 3-pass `h_max` about 2.1x); 548 GB free
  on 2026-11-02 (`df`, `~/sf39_campaign` 169 GB and `~/sf42_campaign` 28 GB untouched). Reduction runs in the background and pruning at
  least hourly (backlog below about 60 GB); retained subset about 20-25 GB; disk guard at 100 GB free.
- Death and resume: append-only ledger, `all --resume` re-runs only incomplete units (into new segment directories, because the
  consumer's DONE signature includes `seed0`), stage-1 start preserved; every resubmission needs a new job name (the helper kills a live
  session with the same name and truncates its log, which is evidence); a watcher must treat `state=running` with `tmux_active=0` as dead
  (the helper writes no heartbeat).

## 4. Experiment audit and time reduction

### 4.a Critical path

| milestone | base | pessimistic |
|---|---|---|
| this consultation | Mon 11-02 about 20:00 | same |
| owner answer and merge (increment base) | Tue 11-03 10:00 | Tue 11-03 18:00 |
| integration of the prepared tooling, audits, smoke test (2-4 h, with the coverage run in parallel), pre-registration pushed | Tue 11-03 20:00-Wed 02:00 | Wed 11-04 04:00-10:00 |
| launch; GPU wait up to 2 h per helper attempt | Tue 22:00-Wed 06:00 | Wed 14:00 |
| stage 1 ends (two GPUs) | +17-22 h | +22 h |
| plan complete (variant A) | Thu 11-05 11:00-Fri 11-06 07:00 | Fri 11-06 15:00 |
| clock cutoff (pre-registered) | Sun 11-08 16:00 | same |
| frozen analysis inside the job (1.5-2 h) and the human-readable report | right after the plan completes | Sun 18:00 at the latest |
| deadline | Mon 11-09 00:00 | same |

GPU-h available between launch and cutoff (two GPUs for 30 h, then one): 144 / 136 / 128 / 118 for launches Tue 22:00 / Wed 06:00 /
Wed 14:00 / Thu 00:00. After the margins (overhead 1.10, under-prediction up to 1.32, one resume of 2 h) the practical ceiling is about
100 GPU-h planning, which replaces G10 (150) as the binding number. Variant A uses 54, B 38, C 88.

### 4.b Arm-by-arm audit (variant A; GPU-h planning from `design_cost.py`)

| arm (amplitudes) | GPU-h | feeds | if cut or reduced | rank by information per GPU-h |
|---|---|---|---|---|
| production (all) | 10.0 | Gate 4 sampling term; every estimate; Gate 5 consumer at 0.25, 1 | nothing can be claimed | 1 |
| Gate 5 (Par2 everywhere; consumer at 2.25; at 4 shared with the `lambda/h` 16 rung) | 4.7 | 4 primary TOSTs; `alpha_L` validation (W1) | W1 stays open | 2 |
| `E_h` ladders (all) | 11.5 (2080x480x480 rungs at 0.25 and 1: 7.8) | Gate 4 budget (dominant at 2.25, 4); W2 | `E_h` becomes a stated assumption: the robust sign is conditional on grid convergence (W2 unremedied) | 3 |
| `E_W` cells (all) | 15.4 | Gate 4 budget (dominant at 0.25, 1); W3 | `E_W` becomes a stated assumption (W3 unremedied); `physical (restricted)` then means "within 24 `lambda`" | 4 |
| distance arms (1, 2.25) | 4.3 | descriptive: decay with distance (first order 6.9x; Lester saturation; B&dD plateau); B&dD eq. 6 secondary | the one direct test of the `d`-dependence is lost | 5 |
| G7 factorial (0.25, 1) | 3.7 | descriptive: attribution of SF-39's unmatched offsets | SF-39's W1 offsets remain unattributed | 6 |
| `h_max` passes (all) | 3.9 | Gate 4 `E_int` (negligible, 1e-6 relative); pathwise data for W8 | `E_int` becomes a stated assumption backed by SF-39; W8 data lost | 7 |
| `lambda/h` 4 rung at 4 | 0.1 | `E_h` order constraint and diagnostic | fallback only | (cheap) |

### 4.c Levers evaluated

| lever | numbers | verdict |
|---|---|---|
| displacement basis | precision targets met at `N` <= 30; Gate 4 power 1.00; total 147 -> 54 GPU-h | **adopt (H2)** |
| relax precision targets | `N` scales with the inverse square of the half-width; irrelevant once the displacement basis is used | not needed |
| `sigma^2 = 0.25` arms | in the position basis no arm is informative (P(RL > 0) = 0; the 1040x480x480 width arm cost 16 GPU-h and left `E_W` at 1.4 times the mean); in the displacement basis they cost 9.3 GPU-h and make 0.25 a confirmatory amplitude | keep, in the displacement basis (H4) |
| Gate 5 at `sigma^2 = 4` | 250 pairs cost 3.1 GPU-h (consumer shared with the 16 rung); SF-39's `agree` was a different (time-window, spatial) quantity | keep; no cheaper paired design exists |
| distance arms and the B&dD fit | 4.3 GPU-h; secondary per W6, but the only direct test of the decay that first-order theory and Lester 2021 predict | keep |
| grids with more factors of two | would deepen the multigrid hierarchy and avoid the 3x rule, but change `lambda/h` or the width and invalidate the grid, memory, cost and lattice pilots; the production arms are no longer the cost driver | reject (H11) |
| particle count `M` | fixed by the lattice rule at 0.25 and 1 (128); at 2.25 and 4 a larger common `M` would lower the particle-noise floor of `E_h` but raise every arm's cost; not needed for power | keep 128 / 100 / 100 |
| execution order | priority classes, progress-fraction interleaving, common-random-number blocks adjacent | adopt (H13) |
| two stages | heavy queue 33 GPU-h on two lanes (lane B only needs 17 GB free), 17-22 h | adopt (H14) |

### 4.d Job design for the deadline

The design asked for in the brief, with three refinements from the analyses: (i) every arm has a pre-registered confirmatory `N` and the
analysis uses the longest complete prefix of its seed order (Glynn & Heidelberger: the completed-run estimator is biased only at `o(1/t)`
under runtime-outcome correlation; SF-39's `sigma^2 = 4` data show a weak runtime-`alpha_TP` correlation, Spearman +0.19, and the long
runtimes come from fallback solves without a consistent `alpha` direction); (ii) the family is fixed and an untestable hypothesis enters
Holm with `p = 1`; (iii) the blinded re-estimation is triggered by count, never applied to Gate 5 arms, and outputs standard deviations
only. Lane rules: lane A takes the near-empty-GPU units first; lane B takes what its GPU can hold (live free-memory check before each
claim), never starts a unit that would end after the 30-h cap, and exits when no heavy unit remains; with one GPU the same queue runs on
one lane. Assignment to lanes and stages depends only on the plan and on memory, never on results; every realization's lane, GPU and
times are recorded.

### 4.e Recommended plan, losses and alternatives

**Recommended: variant A** (H9). Against the specification's full design it loses: position-basis precision (reported, half-widths at
`N` = 150 of about 26 % / 8 % / 4 % of the means at 1 / 2.25 / 4, unresolved at 0.25); the finer `E_h` level at 2.25 (`lambda/h` 20,
replaced by 4, better conditioned); `h_max` 50 -> 30 realizations (`E_int` negligible). It gains: 0.25 as a confirmatory amplitude, Gate 4
power 1.00, a qualified Gate 5, sharpened budgets, results about three days before the deadline and 50-80 GPU-h of slack.
**B (conservative, 38 GPU-h, Thursday):** smaller `N` everywhere (production 100, `E_W` 100, ladders 80, G7 30 per cell, distance 60);
the budgets widen by about 20 %; Gate 4 power stays above 0.95 except the order-1 corner at 4.
**C (ambitious, 88 GPU-h, Saturday, least slack):** production 300, `E_W` 300, ladders 150, Gate 5 300 / 300 / 200 / 300, distance 150 and
a 208-`lambda` arm at `sigma^2 = 4` on `lambda/h` 16 (a scope extension: the decay test at the highest variance); stage 1 reaches its 30-h
cap and about 28 h of heavy work moves to stage 2.

## 5. Own calculations

Scripts in [`artifacts/2026-11-02-sf43-pre-activation-review/scripts/`](artifacts/2026-11-02-sf43-pre-activation-review/scripts/README.md);
local CPU, seconds each (Python 3.13, numpy 2.5, scipy 1.18), no GPU, no host.

- `firstorder_window.py`: the first-order estimand on SF-43's plane list (section 2.1; H3): `alpha_T^FO / (sigma^2 lambda)` = 7.992e-5
  on `[25, 94]`, 1.385e-4 / 3.882e-5 on the halves (ratio 3.57), 1.155e-5 on `[94, 194]` (ratio 6.92). The theory agent's independent
  wave-number quadrature gives 8.0e-5.
- `design_cost.py`: GPU-h, two-stage schedule and wall time of variants A, B, C from the measured per-realization costs (sections 3.g, 4).
- `lattice_check.py`: common admissible `M` per amplitude with SF-42's exact `seedlattice.rule` (H10): 0.25 and 1 at 19.2 `lambda`:
  {128, 256}; 2.25: none at 19.2, 121 values in [40, 300] at 19.08 (including 100); 4: 65 values including 100, unchanged by the
  `lambda/h = 4` rung.
- `gate4_power_sim.py`: the Gate 4 simulation of section 3.d with `mcstat`'s classes (budget components from SF-39's reduced tables,
  P(RL > 0), forced `p_L`, MDE, the `numerical` fraction).
- `gate5_options.py`: the Gate 5 analyses of sections 3.d-3.e (pilot skewness and kurtosis with bootstrap bounds; re-judging SF-42's
  coverage cells for the Fieller form; the skewness-corrected prototype; sizing; the G7 effect sizes; runtime-outcome correlations).
- Sampling-limit coverage at the reporting tails, read from SF-42's `coverage/summary.json`: bootstrap-t passes the 0.05 and 0.025 tails at
  `n` 100, 200 and 500 under `normal`, `stress` and `stress-` (largest UCL/tail 1.199); Student fails the skewed side at every `n`.

## 6. Findings

### Blocking (for activation as specified)

- **B1.** The specified design on the admissible lattice does not fit before the deadline and exceeds G10 once the measured overhead is
  applied (147-185 GPU-h). -> H2, H9.
- **B2.** No Gate 5 procedure is qualified at the planned `n` (SF-42 Finding (j)). -> H5, H15.
- **B3.** The owner's two-GPU first stage contradicts decision 2026-10-27 items 1 and 3, `AGENTS.md` and the runbook ("no multi-GPU
  exception") and needs a recorded decision and a helper option. -> H14; helper node prepared (`scripts/remote` `REMOTE_GPU_COUNT`).
- **B4.** At `sigma^2 = 2.25` no `M` is admissible on every arm with the 19.2-`lambda` source (SF-42 Finding (a)). -> H10.

### Major

- **M1.** The robust-sign null `alpha_T <= 0` is rejected by linear theory itself (first order gives `alpha_T > 0` at every finite
  distance), so `physical (restricted)` would not separate linear theory from a nonlinear transverse dispersion. -> H3.
- **M2.** `E_h` (G5.4): no cap on the observed order, three levels without a residual, a single difference in the finest branch; the
  8/16/20 ladder at 2.25 multiplies the fallback by 12; the order branch almost never fires at `sigma^2 >= 1`. -> H6.
- **M3.** At `sigma^2 = 0.25` no position-basis arm is informative, and G5.7 has no label for the robustly positive sign the
  displacement basis yields. -> H2, H4.
- **M4.** `alpha_LP` at `sigma^2 = 4` is not B&dD's `alpha_L`, and Gate 5 there may `disagree` for slow-tail reasons. -> H5, H17.
- **M5.** G5.7's `numerical` and `unresolved (bounded)` have no levels, and `numerical` is undefined when the order is unresolved. -> H16.
- **M6.** Blinded variance re-estimation can inflate the size of an equivalence test (Friede & Kieser 2003; FDA 2019). -> H5 (Gate 5 arms
  not re-sized).
- **M7.** A wall-clock cutoff selects realizations by runtime; under runtime-outcome correlation "whatever completed" is biased. -> H13
  (pre-registered confirmatory `N`, seed-order prefix, fixed family).
- **M8.** SF-42's 3x PCG rule is tripped by two SF-43 grids. -> H11.
- **M9.** Par2's horizon censors the plane-based `alpha_LP` at factor 1.2. -> H12.
- **M10.** Executability gaps (from the code): a non-converged Par2 head solve exits 0 with a warning; Par2 restarts per invocation only and
  writes about 60 MB of snapshots per invocation; `sf42red prune` enforces 5 retained realizations per directory; `recompute` and
  `reconcile` are SF-39-specific; two lanes would interleave a shared job log; the helper writes no heartbeat and re-using a job name
  truncates its log. -> the Phase-2 nodes (engine, executor, recomputation and reconciliation; `--min-retained`; per-segment logs).

### Minor

- **m1.** Bibliographic corrections (Berger 1982 DOI; Holm 1979 without DOI; several candidate studies are 2D or do not solve Darcy).
- **m2.** Imprecisions in `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` (L_T = 25.6 stated only at `sigma^2 = 6.25`; L_L = 102.4 at
  `sigma^2 <= 1`; velocity at cell centres; eq. 7's range; Fig. 2b against the "converged at t = 10" claim; the refinement test's 4 %
  result) and in `lester-2023-key-claims.md` (eq. 20 bounds by the mean gradient; "smooth" is not defined on the cited pages; page
  references). -> H19.
- **m3.** `E_int` is negligible but still takes one of the `K` tail splits (kept: removing it would need a new coverage cell).
- **m4.** Decisions should be taken at the Holm levels (`RL(level) > alpha_T^FO`), not on the p-value lattice, which over-states `p` by
  up to 1.16 %.
- **m5.** `mcstat.sizing.gate4_power` is optimistic in the fallback regime (0.875 against a simulated 0.69 in one corner): size by
  simulation.
- **m6.** Single-realization memory peaks under-state multi-realization peaks by about 1 GB.

## 7. Adversarial verification summary

(Filled after the five independent adversarial reviews of this draft: statistics and design; numerics and verification; physics and
fidelity to the literature, re-verifying the citations; executability against the code; deadline and cost.)

## 8. Caveats

- Most literature was read at abstract level (paywalls); every content claim used here is marked with its reading level in the
  bibliography, and B&dD, Lester 2021, Lester 2023, Lester 2022, Attinger et al. 2004 and Dentz et al. 2016 were read in full.
- Planning rests on pilots: Gate 5 pairings on 8 pairs per amplitude (the `sigma^2 = 4` consumer CV rests on one field); SD ratios on one
  SF-39 set per amplitude without their own uncertainty; displacement-basis ladder SDs assumed equal to the position-basis ones
  (conservative: the particle components are 4-9 times smaller in variance in the displacement basis).
- The Gate 4 simulations use normal per-seed data (pilot skewness 0.15-0.39, within the `stress` envelope); the coverage of the
  robust-sign pipeline is SF-42's (very conservative, 0 misses in 24 500 at every Holm level), not re-simulated at `N` = 150 (bracketed by
  100 and 300).
- Costs: measured on SF-42's pilots; a colleague sharing the chosen GPU can double wall times (SF-39 `s4_tw`: 42.2 s against 21.9 s); the
  pessimistic margins (1.32) and the cutoff cover this, the budget guard bounds GPU-h.
- The skewness-corrected Gate 5 test is a prototype until the restricted coverage run qualifies it; the pre-qualified fallback
  (bootstrap-t at half the Holm level) guarantees a qualified procedure.
- Nothing here is SF-43 evidence; no `alpha` statement is made.

## Outcome

The recommendations are decisions H1-H20 of
[`2026-11-02-sf43-pre-activation-decisions.md`](../decisions/2026-11-02-sf43-pre-activation-decisions.md); the owner decides H2-H5, H9,
H10 and H20 in a single consultation, and the record states the outcome before this PR is merged. Its merge commit is SF-43's increment
base. The SF-43 Goal and acceptance thresholds are unchanged.
