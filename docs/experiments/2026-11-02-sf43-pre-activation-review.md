# SF-43 pre-activation review: scientific feasibility, fidelity to the literature and a design that fits the deadline (literature, code reading, own calculations, adversarial reviews)

- Date: 2026-11-02
- Status: `complete — design review before activation; NOT SF-43 evidence; no GPU run, no build, no SF-43 data; calculations are numpy/scipy on CPU and readings of committed SF-39/SF-42 data`
- Increment: [`SF-43-confirmatory-macrodispersion-study.md`](../plans/active/lester-eq14/increments/SF-43-confirmatory-macrodispersion-study.md)
  (State `pending`; checker on `master` `ef87a1bc`: `OK (44 increments, ready=SF-34 SF-43, nonterminal=none)`)
- Decision fed: [`2026-11-02-sf43-pre-activation-decisions.md`](../decisions/2026-11-02-sf43-pre-activation-decisions.md) (H1-H21)
- Precedents: [`2026-10-27-sf39-pre-activation-review.md`](2026-10-27-sf39-pre-activation-review.md) (format);
  [`2026-10-29-sf39-remediation-sf42-sf43.md`](../decisions/2026-10-29-sf39-remediation-sf42-sf43.md) (G1-G11, W1-W13)
- Inputs: SF-39 ([`2026-10-27-sf39-macrodispersion-study.md`](2026-10-27-sf39-macrodispersion-study.md), incl. the post-review evaluation);
  SF-42 ([`2026-10-30-sf42-confirmatory-tooling.md`](2026-10-30-sf42-confirmatory-tooling.md), sections C-G, findings (a)-(j)); the
  owner's brief of 2026-11-02 (analysed results by 2026-11-09 00:00 local, UTC-3; a two-GPU first stage of at most 30 h)
- Theory: [`lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md), [`beaudoin-de-dreuzy-2013-key-claims.md`](../theory/beaudoin-de-dreuzy-2013-key-claims.md)
- Artifact: [`artifacts/2026-11-02-sf43-pre-activation-review/`](artifacts/2026-11-02-sf43-pre-activation-review/): the analyses and adversarial reviews ([`analyses/`](artifacts/2026-11-02-sf43-pre-activation-review/analyses/README.md)), verified
  [`bibliography.md`](artifacts/2026-11-02-sf43-pre-activation-review/bibliography.md) and [`references.bib`](artifacts/2026-11-02-sf43-pre-activation-review/references.bib),
  [`scripts/`](artifacts/2026-11-02-sf43-pre-activation-review/scripts/README.md)
- Method: eleven independent read-only analyses (seven literature topics, including the full reading of the three local PDFs; costs and
  schedule; Gate 4 power and budgets; Gate 5 and coverage; executability against the code), re-checked by the orchestrator against the
  committed data; five adversarial reviews of the draft (statistics, numerics, physics and citations, executability, deadline and cost),
  whose surviving findings are incorporated (section 7).

## 0. Summary for the owner

- **The specified design does not fit, and it does not need to.** On the lattice SF-42 found admissible it costs 147 GPU-h anchored,
  162 with the measured job overhead and 185 with every pessimistic factor; with the overhead it would not finish before the deadline from
  any realistic launch time, and it exceeds G10. Almost all of that cost buys position-basis precision.
- **The displacement basis measures the same quantity with 1.3 to 1370 times less variance** (SD ratios 37 / 9.8 / 3.7 / 1.14 at
  `sigma^2` 0.25 / 1 / 2.25 / 4; SF-42 R3). With it every precision target is met at `N` of 30 or less; the probability that the robust
  lower limit exceeds the null is 1.00 at every amplitude except one pessimistic corner at `sigma^2 = 4` (0.69-0.90); and `sigma^2 = 0.25`
  becomes resolvable.
- **Recommended plan (variant A):** about 60 GPU-h planning (79 with every pessimistic factor); two GPUs for 12-16 h only, then one;
  47-63 h of wall (60-79 h if only one GPU is ever available). From a Wednesday-morning launch it ends between Thursday night and Saturday
  morning, a day or more before the pre-registered cutoff (Sunday 16:00). It keeps every arm of the specification and adds two cheap
  items: a positive control (the box's first-order prediction at `sigma^2 = 0.04`, 0.2 GPU-h) and, as a scope extension for the owner,
  a 208-`lambda` arm at `sigma^2 = 4` (the decay test at the highest variance, 1.4 GPU-h). It loses position-basis precision (reported,
  not confirmatory) and the finer `E_h` level at 2.25.
- **Problems fixed before data:**
  - the null's rationale (the SF-18 field is periodic along the flow, so in the simulated boxes linear theory gives a non-positive window
    value; the recommended null is the unbounded-medium value, conservative in the box; H3);
  - a Gate 5 with no qualified procedure (H5: Fieller form, existing coverage evidence and a 2-4 h CPU run);
  - `E_h` resting on an untested asymptotic range at 2.25 (H6: gates on a fourth level);
  - coverage evidence that did not reach every component of the robust sign (H15: pipeline-level qualification, bootstrap-t sampling at
    `N` = 200, new cells);
  - undefined G5.7 levels (H16).
- **Alternatives:** B (42 GPU-h, 33-44 h, budgets about 20 % wider) and C (93 GPU-h, 70-93 h, needs two GPUs from launch and a launch by
  about Wednesday 15:00).

## 1. Question, scope and date of knowledge

**Question.** Before SF-43 is activated and pre-registered: (a) does each planned experiment answer a well-posed question, with a
well-defined estimand, a bounded error and an informative result under every outcome; (b) is the design faithful to the literature it
will be compared with; (c) which design delivers the most scientific information before the owner's hard deadline, with every loss against
the full design declared and quantified?

**Hypothesis tested.** None about `alpha`. Only that some design items are ill-posed, uninformative or unaffordable for derivable
reasons, which is checked by reading, derivation and simulation on CPU.

**Scope.** No SF-43 data exist and none were generated. Nothing was submitted to the V100 for this review (one short `scripts/remote
exec` read the host's free disk with `df` and `du`, no CUDA). Every number is a design calculation, a literature value or a reading of
SF-39's and SF-42's committed data. No `alpha` claim is made (`docs/theory/lester-2023-key-claims.md`).

**Date of knowledge (integrity statement).** Written after SF-39's results and SF-42's pilots were committed (`104cc254`, `ef87a1bc`).
Those data are used only as pilots: variances, pairing correlations, costs, memory, failure modes and per-basis noise. Data-informed
changes, disclosed: the displacement basis (from SF-42's SD ratios, not from a mean); the null's value (from theory; free at the pilot
magnitudes, which were known); the admission of `sigma^2 = 0.25` to the primary family, which rests on the displacement basis and on the
displacement-basis pilot mean at 0.25 (5.1e-4 `lambda`, about 21 standard errors from zero). The last one enlarges what can be claimed and
is the owner's decision (H4). All are fixed before SF-43 data exist.

## 2. Literature: what is established, what is disputed, what is comparable

Every claim is in the verified bibliography
([`bibliography.md`](artifacts/2026-11-02-sf43-pre-activation-review/bibliography.md)) with its DOI (Crossref/OpenAlex), locator and
reading level: **FT** full text, **A** abstract, **M** metadata. Most WRR/JFM/Springer full texts were blocked to the agents; abstract-level
claims are limited to what the abstract states.

### 2.1 Theory of 3D purely advective transverse macrodispersion (isotropic media)

- **First order, unbounded medium.** `alpha_T(x) ~ (sqrt(pi)/8) sigma^2 lambda^3 / x^2`, `X_22(inf) = sigma^2 lambda^2 / 3` for
  `C(r) = sigma^2 exp(-(r/lambda)^2)` (formula of the SF-39 pre-activation review, section 5.1; re-derived independently by two
  reviewers). For SF-43's estimand (half the OLS slope of `X_22` on the planes `d` = 25..94 `lambda`) `alpha_T^FO = 7.992e-5 sigma^2
  lambda`; halves 1.385e-4 / 3.882e-5 (ratio 3.57 on half-`lambda` planes, 3.64 on integer planes); `[94, 194]` 1.155e-5 (ratio 6.92)
  (`firstorder_window.py`). Positive at every finite distance.
- **First order, in SF-43's boxes.** The SF-18 field is torus-periodic on the flow grid, i.e. also along the mean flow with the box length
  (`PeriodicGaussianField.cuh`; `DirichletLongBoxStage.cu` calls it on the `nx x ny x nz` grid). The transverse velocity has no `k_1 = 0`
  mode, so at first order the transverse displacement variance is symmetric about half the box and returns toward its starting value;
  finite widths remove the small transverse wave numbers that produce the `d^-2` tail. Window values (`firstorder_box.py`, one-particle,
  Dirichlet faces not modelled; matched independently by two reviewers): **-3.5e-5** (width 24, length 104), **-4.1e-6** (width 12,
  length 104), -8.6e-5 (width 48), +1.1e-6 on `[25, 94]` and -7.1e-6 on `[94, 194]` in the 208-`lambda` boxes, in units of `sigma^2
  lambda`. **In the simulated geometry linear theory predicts a non-positive `alpha_TP` over the window**; the positive unbounded value is
  an upper bound of it (decision H3 uses it as a conservative, geometry-independent null). The pilot displacement-basis means are 25 / 90 /
  164 / 204 times the unbounded value at `sigma^2` 0.25 / 1 / 2.25 / 4.
- **Beyond first order.** The lowest-order asymptote is zero (Attinger, Dentz & Kinzelbach 2004, SERRA 18, sect. 5, eqs. 9-10, FT); in
  2D the zero holds at all orders (Attinger et al. 2004, FT). In 3D, Attinger et al. (sect. 6-8, FT) argue that higher orders do not
  cancel, from simulations on linearized (not helicity-free) velocity fields; Neuman & Zhang 1990 (quasi-linear, exponential covariance,
  A) give a nonzero asymptote proportional to `sigma_Y^4`; Janković et al. 2009 (inclusion media, A) give `alpha_T > 0` for randomly
  oriented ellipsoids and note that "the two-point covariance ... is not able to detect the structures that produce advective mixing".
  **Against them, Lester et al. 2023 (local PDF, p. 1, FT) "prove that the transverse macrodispersion does not exist" for smooth, locally
  isotropic conductivity**; the proof (sect. 4, pp. 8-9) assumes that the streamfunction fluctuations are bounded (p. 8), the assumption
  the project does not presuppose and found to fail in the periodic cell (`lester-2023-key-claims.md`). The published nonzero 3D results
  come from linearized fields, closures or non-smooth inclusion media.
- **Effective vs ensemble.** SF-43 centres moments per realization (effective). Ensemble coefficients overestimate effective ones at finite
  time and finite source, and become equivalent only through local dispersion or infinitely extended sources (Dentz et al. 2000a,b, A),
  neither present here; both SF-43 and B&dD use the effective definition, so the comparison is like for like.

### 2.2 Topological constraints, helicity and chaos (W8)

- **Persistent exponential stretching is excluded in the exact flow.** Smooth, locally isotropic, stagnation-free Darcy flow is
  helicity-free and integrable (Sposito 1994, 2001, A; Lester et al. 2021, JFM 918 A27, local PDF, FT: pp. A27-1, A27-7 algebraic growth
  `l(t) ~ t^r`, A27-18 two superposed 2D shear flows "from sublinear to ballistic"; Lester et al. 2022, JFM 945 A18, FT: eqs. 4.23-4.25,
  the transverse stretch is a bounded ratio of local quantities and the Lyapunov exponent is zero; eq. 6.12, shear `|F_12| ~ t^r`,
  `0 < r < 2`). Helical flow needs anisotropy (Bakker & Hemker 2004; Hemker & Bakker 2006, A); Chiogna et al. 2015 (A) report helical
  streamlines at large scales in locally isotropic fields, which Lester 2021 (A27-26) attributes to non-smooth fields.
- **Finite-time growth and numerical drift are not excluded.** Ballistic shear and large bounded swings of `F_22` in slow zones can look
  exponential over a window (inference from Lester 2022). Lester et al. 2023 (local PDF, sect. 5.3, eq. 36, Fig. 4b, pp. 13-14, FT)
  integrate with adaptive RK4 a divergence-free velocity built from interpolated streamfunctions (exact label invariants, not exactly
  helicity-free, p. 10): the trajectory errors have zero mean, a variance growing like `sqrt(tol)`, and vanish as the tolerance tightens.
  Lester et al. 2025 (JFM 1018, App. D, arXiv v1) **assert**, citing Lester 2023, that RK on a spline velocity introduces spurious
  transverse dispersion; they show no data for it. For a non-chaotic flow the relevant numerical result is polynomial error growth
  (Calvo & Hairer 1995, A).
- **Lester's pre-asymptotic picture.** Lester 2021 eq. 5.43 (FT) writes a saturating transverse spreading from an exponential label
  autocorrelation it "may approximate" (eq. 5.41); neither the decay length nor the variance is estimated, and the derivation assumes no
  backflow (A27-23). It is illustrative; first order also saturates (and, in the box, approaches its value exponentially).
- **W8 reading.** The `h/8`-`h/16` separation of SF-39 compares two integrations of the same spline field. The separation splits into an
  along-streamline part (shear, up to ballistic) and a transverse part (bounded in the exact flow); only the transverse part feeds
  `alpha_TP`, and the passes agreed on `alpha_TP` (integration error at most 1e-6 relative). The `d`-dependence of `alpha_T` separates
  decay from plateau, not physics from numerics. G9a (not scheduled) remains the place for a separation-versus-`lambda/h` measurement.

### 2.3 Numerical 3D studies at moderate and high `sigma^2`

| study | dim, covariance | `sigma^2` | width, transverse BC | estimator, window | `N` | transverse result | read |
|---|---|---|---|---|---|---|---|
| Beaudoin & de Dreuzy 2013 (WRR 49) | 3D, Gaussian (eq. 1) | 0.25-9 | 25.6 `lambda` stated at `sigma^2 = 6.25` (Figs. 1, 5); periodic | resident spatial moments centred per realization (eq. 4), fixed time; mean over `[0.75 t_b, t_b]` (eq. 5); `alpha_LAF` fit eq. 6 | 500, `N_p` 1e4 | `< 1e-3` / 0.01 / 0.044 at 0.25 / 1 / 2.25 (Table 2); about 0.11 at 4 (Fig. 4b) | FT |
| Dartois, Beaudoin & Huberson 2018 | 2D/3D, exponential (stated) | <= 2.25 | 25.6, not stated | velocity covariance and moments | 100 | figures only | FT |
| Janković, Fiori & Dagan 2003, 2006 | 3D inclusions | 2-8 | AEM | velocity covariance; travel-time variance at planes (2006, `alpha_L` only) | 3 (2006) | "small but persistent" | A |
| Chin & Wang 1992 | 3D | <= 2.25 | n/v | ensemble, with confidence limits | n/v | first order valid only for `sigma_Y < 1` | A |
| de Dreuzy, Beaudoin & Erhel 2007 | 2D, exponential | 0.25-9 | no-flow | resident, effective | 100 | zero on average | FT |

B&dD is the closest design to SF-43 (Gaussian covariance, periodic transverse faces, flux-proportional injection over 0.8 of the width
at least 5 `lambda` from the inlet, `lambda/Delta` 10, effective moments) but not at `sigma^2 = 4` (width 12 vs 25.6 `lambda`, source 9.6 vs
about 20, `lambda/h` 16-32 vs 10) and not numerically: velocity "at the grid center" linearly interpolated "in each direction" (p. 2490),
explicit Euler with about 10 steps per cell; Lester et al. 2023 (p. 20) attribute their result generically to tracking methods that do not
enforce the kinematic constraint. Their fixed-time, fixed-axis `alpha_TAM` also contains drift tilt and longitudinal-transverse coupling,
which plane estimators exclude (Lester 2023 p. 3). Their refinement test at `sigma^2 = 6.25` (`lambda/Delta` 10 -> 20) moved `alpha_T` by
at most 4 %, which argues against a pure `Delta^2` interpolation error (p. 2491). No study found varies the transverse width in 3D;
Chin & Wang 1992 report confidence limits for `sigma_Y <= 1.5` (A); SF-43's width of 12 `lambda` at `sigma^2 = 4` has no precedent.
Corrections to the candidate list: Salandin & Fiorotto 1998 and Hassan et al. 1998 are 2D; Englert et al. 2006 is velocity statistics;
Dentz et al. 2002 and Schwarze et al. 2001 do not solve Darcy's equation.

### 2.4 Control-plane estimands against spatial moments

- `alpha_LP = (U^2/2) d var(t)/dd` is exact for 1D Fickian first passage (inverse Gaussian, `var tau = 2 D d / U^3`) and equals the
  spatial `alpha_L(t = d/U)` at first order in a heterogeneous medium (own derivation; Shapiro & Cvetkovic 1988, A; Dagan, Cvetkovic &
  Shapiro 1992, A). At `sigma^2 = 4` (and partly 2.25) transport over the window is non-Fickian and pre-asymptotic (B&dD's `t_LF` = 20.5
  at 4; `d / alpha_L` = 2-7 over the window): the J3B pilot's plane-based `alpha_LP` is about 10 `lambda` against SF-39's spatial window
  value of 4.65. The comparison with B&dD is therefore through their eq. 6 at `t = d/U`, not through their asymptote.
- Travel time samples velocities in space (flux-weighted, s-Lagrangian; Dentz et al. 2016, PRF, FT, eqs. 9, 10, 16, 17): with
  flux-proportional injection these statistics are stationary from the start, which supports flux weights and `U_r = Delta d / Delta
  mean tau` (the `sf42red` definition, `U_r = 1 / slope[mean tau]`); resident injection inflates arrival-time variance (Demmy et al. 1999;
  Janković & Fiori 2010, A).
- The estimand is the first-passage distribution at each plane (the tracker lands the planes in order and counts recrossings; Par2
  records first crossings). At `sigma^2 = 4`, 6.7e-4 of the `x` faces had `u <= 0` in SF-39, and backflow concentrates in the slow tail that
  drives `alpha_LP`: recrossing fractions and an `alpha_LP` sensitivity without backflow-encounter particles are reported (H17).

### 2.5 Verification of the discretization budget

- G5.4's `E_h` is the fine-grid GCI (Roy 2005, FT, sect. 3.4.3; Celik et al. 2008, FT; NASA NPARC, FT) and the multilevel Monte Carlo
  bias estimator with coupled levels (Giles 2015, FT, sect. 3.1; Cliffe et al. 2011, FT). Its fallback `3 max|diff| / (r - 1)` is Eça &
  Hoekstra's 2008 rule for oscillatory convergence and Roache's suggestion; it is a recognised conservative heuristic, not a derived
  bound (it holds for a monotone error decaying at least like `h^0.21-0.24` with no component common to all levels).
- Gaps closed by H6: an order cap at the formal order (Eça & Hoekstra 2014, FT: `0.5 <= p < 2.1`); a check of the asymptotic range,
  which three levels cannot provide (Eça & Hoekstra require four); the Giles maximum in the finest branch. Rates depend on the quantity
  (Cliffe et al. 2011: about 1.75 for `k_eff`, 0.75 for a point flux), so SF-42 E's second-order flux convergence from `lambda/h = 4` is
  flux evidence, not evidence for `alpha_T`.

### 2.6 Statistics

- Holm 1979 (JSTOR 4615733); Berger 1982 (intersection-union; DOI `10.1080/00401706.1982.10487790`); Schuirmann 1987 and Berger & Hsu 1996
  (TOST, each one-sided test at the full level); Clopper & Pearson 1934.
- Ratio-of-means equivalence in Fieller form (Fieller 1954; Locke 1984; Hauschke et al. 1999, A); bootstrap-t second-order accurate for
  one-sided limits (Hall 1988, A), `B + 1` a multiple of `1/alpha` (Hall 1986, A); Hall's 1992 skewness transformation and Johnson's 1978
  modified t (A). No reference guarantees one-sided coverage near `alpha` 1e-3 under heavy skewness: qualification is by simulation
  (Morris, White & Crowther 2019, FT).
- Blinded re-estimation: no inflation for superiority t-tests (Kieser & Friede 2003, A), slight inflation for equivalence (Friede &
  Kieser 2003, A; FDA 2019, FT p. 10).
- Time budgets: the completed-run mean is biased only at `o(1/t)` under runtime-outcome correlation; unbiasedness needs a pre-selected set
  of runs (Glynn & Heidelberger 1990, 1991, A). Split-plot: whole-plot factors need the whole-plot error (Jones & Nachtsheim 2009, FT).

## 3. The owner's questions, answered with evidence

### 3.a Estimands

- **Definitions.** Per realization, from the first crossing of each plane `x0 + d`, `d` = 25..94 `lambda` (flux weights, moments centred
  per realization): `alpha_TP` = half the OLS slope of the transverse displacement variance `(var dx2 + var dx3)/2` (H2: displacement basis,
  `dx = x23(d) - x23(seed)`); `alpha_TAP` the same orthogonal to the realization's mean displacement; `alpha_LP = (U_r^2/2) slope[var tau]`,
  `U_r = 1 / slope[mean tau]` (`sf42red/reduce.py:28`, :248-250); the ensemble estimand is the mean of per-realization values. The
  `alpha_LP` estimand differs from `(Ubar^2/2)` times the mean slope by about `3 CV(U_r)^2` (at most 0.44 % in the pilots); both are reported.
- **Displacement against position basis.** `var x = var x0 + var dx + 2 cov(x0, dx)`; `var x0` is constant in `d` (`set_differs = 0`
  in every pilot row) and `E cov(x0, dx) = 0` exactly for uniform weights (the field is transversally stationary and periodic, the weights
  are fixed per particle). For flux weights the per-realization normalization breaks the exact symmetry near the source edges; the bias is
  not derived, so the uniform-weight displacement estimand is reported as an exact control and the `pos - disp` contrast is reported. The
  pilot agreement test (`|pos - disp| / SE_pos` = 0.78 / 0.64 / 0.59 / 0.48) has little power and is not used as evidence.
- **Comparability.** `alpha_TP` (fixed axes) with B&dD's `alpha_TAM` as an order of magnitude only; `alpha_TAP` and `alpha_LP` have no
  counterpart; `alpha_L` comparisons through B&dD's eq. 6 at `t = d/U` (G9b).

### 3.b Domain assumptions

| assumption | effect on the conclusion | handling |
|---|---|---|
| periodic transverse boundaries, width 24 `lambda` (12 at 4) | no transverse wave numbers below `2 pi / W`; images do not fold the plume (unwrapped coordinates, SF-39 post-review); at `sigma^2 = 4` the displacement SD at `d = 94` (about 3.5 `lambda`) is comparable to the half-width | measured (`E_W`, interaction); beyond the widest width a stated assumption |
| **the field is periodic along `x` with the box length** (104 `lambda`; 208 in the distance arms) | at first order the transverse displacement variance is symmetric about half the box, so the window value is <= 0 in the box (section 2.1); the field near the outlet is the image of the field upstream of injection; box length did not move the pilot means (104 against SF-39's longer boxes: displacement-basis ratios 1.025 / 1.036 / 1.019 at 0.25 / 1 / 2.25, SE 4-5 %) | declared model restriction; first-order references per box (H17); G7's generator factor mixes this periodicity with variance normalization |
| source 0.8 x width (19.2 / 9.6 `lambda`) | ergodicity of per-realization moments (about 22 integral scales; 11 at 4); fixed across arms of an amplitude | declared |
| medium (b): Dirichlet `x` faces | boundary layers of a few `lambda` (about `W / 2 pi` for transverse modes: 3.8 `lambda` at width 24, 7.6 at 48); the window ends 5 `lambda` before the outlet, so width contrasts partly include a change of outlet influence | declared; stated with `E_W` |
| SF-18 generator, variance normalized | conditional ensemble at realized variance `sigma^2`; realized-variance SD would be 0.8 % (width 24) / 1.6 % (width 12) without normalization: negligible | declared |
| finite distance `[25, 94]` | no asymptotic claim | halves as an interval; distance arms (H17) |

### 3.c The error budget G5.4

The structure is standard (section 2.5). What the SF-39/SF-42 data imply (Gate 4 analysis and the numerics review):

- **`E_h`.** In the position basis the per-seed grid differences are at the particle-noise floor, so the order branch almost never fires
  and `E_h` is the fallback (a resolution bound). In the displacement basis the particle variance is 4-9 times smaller, and the order
  branch becomes operative at 2.25 (100 % of replicates in the review's scenario) and often at 4 (12-48 %). At 2.25 the triple 4 / 8 / 16
  then carries the bound with an untested coarsest level: under a pre-asymptotic error law the ungated bound misses the production error in
  11 % of replicates against a nominal 7.8e-4. H6 therefore gates the order branch on a fourth level (`lambda/h` 12 at 2.25, the triple
  4 / 8 / 16 against 8 / 16 / 32 at 4) and falls back otherwise. With production finest at 2.25 and 4 the remaining assumption is the
  asymptotic range of the gated triple, stated. The specification's alternative 8 / 16 / 20 at 2.25 has no constant ratio and multiplies
  the fallback by 12.
- **`E_W`.** Dominant at 0.25 and 1 in the displacement basis (about 0.08-0.16 of the mean). At 2.25 a 2 x 2 at `lambda/h` 8 / 10 would leave
  the transfer 10 -> 16 under-covered for an order-1 width effect (1.5 times the measured interaction); the 2 x 2 at 8 / 12 (H7) covers the
  transfer 12 -> 16, reuses the Gate 5 consumer grid and keeps the 19.2-`lambda` source admissible. A width deficit enters twice (in the
  production mean and in `E_W`), which is conservative.
- **`E_int`.** Negligible: the bound `2 UCB` at `N` = 30 is about 6e-5 of the mean at 2.25 and 3e-6 at 4. It keeps its tail split.
- **Solves at `sigma^2 = 4`.** SF-39 recorded 4 of 196 solves at 2000 PCG iterations at 4 (three accepted by the 1e-8 fallback, one
  unusable); on 1664x192x192, 3 of 61. SF-43 should expect about 12 stagnated and about 4 unusable solves among the 500 consumer and Par2
  solves of that grid, above the specification's 1 % failure level (its worst-case imputation will be reported). Replacement conditions on
  solvability, plausibly correlated with the slow zones that drive `alpha_LP`: H5's pair-exclusion sensitivity and H13's reserves address
  it. (A possible cause, not tested: the head solve's multigrid preconditioner applies red-black Gauss-Seidel in the same order before and
  after the coarse correction, which is not symmetric, while PCG assumes a symmetric preconditioner. Out of scope; recorded.)

### 3.d Power, informativeness and pre-mortem

Gate 4 (`gate4_power_sim.py`; `mcstat`'s classes, R = 400, B = 4000; point planning inputs, normal per-seed data; scenarios S1 order-2
biases from SF-39, S1w5/S1w10 production width 5/10 % short, S2 adverse non-monotone ladder, S3 order-1 bias):

| design | P(RL > null) at the worst Holm level | MDE / planning mean |
|---|---|---|
| specification (position basis) | 0.25: 0.00; 1: 1.00; 2.25 (8/16/20): 0.81-1.00; 4 (`N` 30): 0.14-1.00 | 2.4 / 0.46 / 0.81 / 0.70-1.04 |
| displacement basis, production 100, arms 100 | 1.00 at every amplitude and scenario except S3 order-1 at 4 under the pessimistic noise model (0.69-0.90) | 0.28 / 0.23 / 0.33-0.48 / 0.38-0.56 |

Variant A (production 200, ladders 100-150, width cells 150) is at least as strong. The `numerical` class (H16, fraction 0.5, one-sided
95 % upper limit) is reachable in the displacement basis at every amplitude: P(numerical) = 1.00 when the true corrected value is zero,
1.00 at a corrected/production ratio of 0.25, and 0.00 when the transverse growth is physical.

Gate 5 (Fieller form; conservative pilot inputs): power at ratio 1 is 1.00 at 0.25, 1 and 2.25 (`n` = 200) and about 0.90 / 0.89 / 0.82 at
the levels 0.05/7, 0.05/8, 0.05/16 at 4 (`n` = 250); the largest true ratio with power 0.8 to `agree` is about 1.085 / 1.083 / 1.06 at
0.25 / 1 / 2.25 (at `n` = 150; slightly larger at 200) and 1.02 at 4 (1.0125 at the halved level).

| outcome | what we learn |
|---|---|
| Gate 4 `physical (restricted)` (expected at every amplitude) | `alpha_T` exceeds the linear-theory value even of an unbounded medium, robustly to discretization, width and integration, within this model, medium and window; with the halves and the distance arms, whether it decays toward zero or plateaus |
| `numerical` (only under an artifact truth) | most of the transverse growth is a discretization (`E_h`) or integration artifact: support for Lester et al.'s attribution in this model; a width artifact would not be that |
| `unresolved (bounded)` | an upper bound, with the dominant `B_k` (8 -> 16 noise growing as `h^2`, a 12 -> 24 width effect, an order-1 bias at 4) |
| `unresolved` | estimator discordance (practically unreachable: `alpha_TAP` = `alpha_TP` within 4e-4) |
| Gate 5 `agree` (likely at 0.25-2.25: matched pilot ratios 1.001 / 1.007 / 1.015) | `alpha_L` validated against an independent code on the same field; SF-39's offsets then belong to the unmatched factors G7 apportions |
| Gate 5 `disagree` (plausible at 4: pilot ratio 1.14 on 8 pairs, driven by one field) | the travel-time `alpha_LP` at 4 depends on the representation or the stepping in slow zones; the consumer-twin / Pollock-twin / Par2 decomposition says which; SF-43 stays active until attributed or decided by the owner |
| Gate 5 `inconclusive` | an offset near the margin or a larger CV than planned |

No arm is uninformative in the displacement basis; in the position basis every `sigma^2 = 0.25` arm was (P = 0).

### 3.e Gate 5

SF-42 Finding (j) is resolved without the 42-hour re-run (H5, H15). In Fieller form the size depends only on the standardized law of
`D_r = L_r - theta P_r` at the margin, the law SF-42's `sampling` cells simulate; the claim that it is "independent of the pairing
correlation and the CVs" holds for that law, not for the data, and the Gate 5 data at `sigma^2 = 4` are more skewed than the `stress`
family (J3B `e_r`: skewness 1.82, bootstrap upper bounds 2.43 / 2.73; excess kurtosis 2.72, 6.38 / 7.59), so a J3B-bound family is gating
at 4 (a reviewer's simulation at `n` = 250: bootstrap-t miss/tail 0.77-0.88, Student 2.7-3.5). From the existing cells
(`gate5_options.py fieller-coverage`): bootstrap-t at the Holm level qualifies from `n` = 200; at half the level from `n` = 100 when judged
against the Holm level (the halved test's own tail fails at 100: 1.999), with the `m = 8` tail 3.125e-3 not simulated. Hall's skewness
correction is the first procedure in the list; it fails at `n` = 150 in the harness check (1.18 x tail at 6.25e-3) and is about 1.03 at 200
and 250, hence `n` = 200 / 200 / 200 / 250. The restricted run (H15) qualifies the list at the realized `n`. Alternatives: the
geometric-mean ratio shifts the estimand by 24-37 % of the margin at 4 (rejected); removing Gate 5 from the family gives up W1's
confirmatory remedy; serial gatekeeping behind Gate 4 (tested with probability 0.7-1.0 in the displacement basis) would allow the
existing delta-method bootstrap-t TOST at tails >= 0.0125, which already qualifies at `n` = 170 — a valid alternative the owner can choose.

### 3.f SF-42's open decisions

| decision | options | recommendation | consequence |
|---|---|---|---|
| budget and design (147.4 anchored, 162.3 with overhead) | keep; cut arms; change the basis | change the basis (H2), keep every arm, G10 the hard cap | about 60 GPU-h planning, 79 pessimistic |
| seed lattice at 2.25 (no common `M` at 19.2 `lambda` with `E_W` at `lambda/h` 10) | another source; per-arm `M`; an `E_W` `lambda/h` without a factor 5 | `E_W` at `lambda/h` 8 / 12 (H7): `M` = 100 among 65 common values, source unchanged (H10) | one source discretization, better transfer coverage |
| 3x PCG rule | other grids; accept | accept, and extend to the shallow grids at 0.25 (H11) | measured cost, converged solves |
| Par2 horizon | 1.2 (0.2-3.8 % censored); 3.6 | 3.6, completeness flag, usability rule (H12) | +8-17 % Par2 time |
| Gate 5 `n` floor | 30; coverage-driven | 200 / 200 / 200 / 250 (H5) | 5.8 GPU-h for Gate 5 |

### 3.g Operational feasibility of the single job

- **GPU-h and wall** (`design_cost.py`): variant A 59.6 GPU-h planning (62.6 with the 5 % reserve); near-empty-GPU work 24.8 GPU-h, half-class
  12.1, light 22.7. Lane A carries the near-empty grids, lane B only half-class ones and leaves when none remains: two GPUs for 12.1 /
  13.4 / 16.0 h, wall 47.4 / 52.2 / 62.6 h at the planning / overhead (1.10) / pessimistic (1.32) margins; one GPU only: 59.6 / 65.6 / 78.6 h.
  Not in these numbers: about 2-4 GPU-h of 2000-iteration solves at 4, and a colleague time-sharing a lane (SF-39 saw a factor 2), which the
  1.32 margin covers only for about 30 % of lane time.
- **Memory** (own-process peaks, MiB): 3328x384x384 29 881 (six realizations), 2080x480x480 about 29 229, 1248x576x576 about 25 000
  (estimated; measured in the smoke test) on lane A; 1664x384x384 15 703 and 1040x480x480 about 15 391 on lane B (17 000 MiB free suffices).
- **Disk**: consumer raw output 95-231 MB per realization with the Pollock bridge (M 100-128), three passes about 2.1x; about 0.6 TB for
  variant A without pruning, against 548 GB free on 2026-11-02 (`df`; `~/sf39_campaign` 169 GB and `~/sf42_campaign` 28 GB untouched).
  Background reduction and pruning every 30 minutes; plane arrays of the primary arms kept (about 80 GB, H21); disk guard at 100 GB free;
  reduction and pruning of every arm type are a smoke-test gate (pruning never ran on a real campaign).
- **Death and resume**: append-only ledger; `all --resume` re-runs only incomplete units into new segment directories; every resubmission
  needs a new job name (the helper kills a same-name session and truncates its log); the watcher must treat `state=running` with
  `tmux_active=0` as dead (no heartbeat in the helper).

## 4. Experiment audit and time reduction

### 4.a Critical path

| milestone | base | pessimistic |
|---|---|---|
| this consultation | Mon 11-02 afternoon | Mon 11-02 20:00 |
| owner answer and merge (increment base) | Tue 11-03 10:00 | Tue 11-03 18:00 |
| integration of the prepared tooling with its corrective nodes, audits, the smoke test with the coverage run in its background (one `run` job), pre-registration pushed: 12-25 h | Tue 11-03 22:00 | Wed 11-04 19:00 |
| launch; GPU wait up to 2 h per helper attempt | Tue 22:00-Wed 06:00 | Wed 21:00 |
| plan complete (variant A, pessimistic, one resume) | Fri 11-06 | Sat 11-07 afternoon |
| clock cutoff (pre-registered), grace 1 h, frozen analysis | Sun 11-08 16:00-20:00 | same |
| deadline | Mon 11-09 00:00 | same |

**Go/no-go launch limits** (variant A complete by the cutoff at the pessimistic margin with one 2-h resume): two GPUs, Thursday 21:00;
one GPU, Thursday 05:00. Practical ceilings: with two GPUs the capacity depends on the mix (lane B runs only half-class units); with one
GPU the planning GPU-h that fit at the pessimistic margin are 83 / 77 / 71 / 64 for launches on Tuesday 22:00 / Wednesday 06:00 /
Wednesday 14:00 / Thursday 00:00. These, not G10 (150), are the binding numbers. If the analysis package slips, the launch slips with it: the analysis runs inside the job and its code is fixed at
launch.

### 4.b Arm-by-arm audit (variant A; GPU-h planning; near-empty GPU-h in brackets)

| arm (amplitudes) | GPU-h | feeds | if cut or reduced | information per GPU-h |
|---|---|---|---|---|
| production (all), first 30 seeds with three passes | 13.3 + 1.9 [7.3 + 0.7 at 4] | Gate 4 sampling term; every estimate; Gate 5 consumer at 0.25, 1; `E_int` | nothing can be claimed | highest |
| Gate 5 (Par2 everywhere; consumer at 2.25; at 4 the 16 rung) | 5.8 | four primary TOSTs; W1 | W1 stays open | high |
| `E_W` cells | 16.8 [6.1 at 2.25] | Gate 4 budget (dominant at 0.25, 1); W3 | `E_W` becomes a stated assumption; `physical (restricted)` then means "within 24 `lambda`" | high at 0.25-1, medium elsewhere |
| `E_h` ladders | 11.6 [7.8 at 0.25, 1] | Gate 4 budget (dominant at 2.25, 4); W2 | `E_h` becomes a stated assumption (W2 unremedied) | high at 2.25-4 (cheap rungs), medium at 0.25-1 (expensive rungs) |
| distance arms (1, 2.25; extension at 4) | 5.8 [2.8 at 2.25] | descriptive: decay with distance; B&dD eq. 6 | the only direct test of the `d`-dependence | high (the extension at 4: 1.4 GPU-h, light grid) |
| G7 factorial (0.25, 1) | 4.3 | attribution of SF-39's unmatched offsets | W1's history unattributed | medium |
| positive control (`sigma^2` 0.04) | 0.2 | end-to-end check of the estimand pipeline against the box first order | no exact check of the transport estimand | high per GPU-h |

### 4.c Levers evaluated

| lever | numbers | verdict |
|---|---|---|
| displacement basis | precision targets at `N` <= 30; Gate 4 power 1.00; 147 -> 60 GPU-h | **adopt (H2)** |
| relax precision targets | irrelevant once the displacement basis is used | not needed |
| `sigma^2 = 0.25` arms | position basis: uninformative (the 16-GPU-h width arm left `E_W` at 1.4 times the mean); displacement basis: about 12 GPU-h, a confirmatory amplitude | keep (H4) |
| Gate 5 at `sigma^2 = 4` | 250 pairs, 3.1 GPU-h (2.0 beyond the ladder rung it shares); SF-39's `agree` was a different quantity | keep |
| distance arms and B&dD fit | 5.8 GPU-h; the decay test | keep, add `sigma^2 = 4` (owner) |
| grids with more factors of two | would change `lambda/h` or width and invalidate the pilots | reject (H11) |
| particle count `M` | fixed by the lattice at 0.25, 1 (128); 100 at 2.25, 4 | keep |
| `h_max` as a separate arm | duplicates the h/8 pass | merge into production (-2 GPU-h) |
| `E_W` at 2.25 on `lambda/h` 8 / 12 instead of 8 / 10 | +1.3 GPU-h of lane-A time | adopt (H7) |
| two stages | lane B 12-16 h | adopt (H14) |

### 4.d Job design for the deadline

As the brief asks, with the reviews' corrections: priority classes ordered first (near-empty-GPU units first only within a priority);
progress-fraction interleaving computed on the confirmatory `N`; common-random-number blocks adjacent; a pre-registered confirmatory `N`
and floor per arm, analysis on the longest complete prefix of the seed order (SF-39's `sigma^2 = 4` data show a weak runtime-`alpha_TP`
correlation, Spearman +0.19, the long runtimes coming from fallback solves without a consistent `alpha` direction; a reviewer's simulation
with 20 % truncation and correlations up to 0.5 gives size/tail 0.95-1.03); a fixed Holm family with `p = 1` for untestable hypotheses; the
blinded re-estimation triggered by count, outputting SDs only and inert unless the SD is several times the planning value; lane B
released when nothing it can hold remains; an outcome-blind GPU-availability rule (H13). Lane, GPU and times of every realization are
recorded.

### 4.e Recommended plan, losses and alternatives

**Recommended: variant A** (H9). Against the specification it loses position-basis precision (reported; half-widths at `N` = 200 about
23 % / 7 % / 4 % of the means at 1 / 2.25 / 4, unresolved at 0.25) and the finer `E_h` level at 2.25. It gains 0.25 as a confirmatory
amplitude, power about 1, a qualified Gate 5, gated budgets, a positive control, and results two to three days before the deadline.
**B (42 GPU-h, 33-44 h):** production 200 (150 at 4), ladders 60, width 100, G7 30 per cell, distance 60, no extension; budgets about 20 %
wider. **C (93 GPU-h, 70-93 h):** production 300 (250 at 4), ladders 150, width 300, Gate 5 300, distance 150 incl. the extension; needs two
GPUs from launch, working pruning, and a launch by about Wednesday 15:00.

## 5. Own calculations

Scripts in [`artifacts/2026-11-02-sf43-pre-activation-review/scripts/`](artifacts/2026-11-02-sf43-pre-activation-review/scripts/README.md)
(CPU, no host, no GPU):

- `firstorder_window.py`: unbounded first-order values (section 2.1; H3).
- `firstorder_box.py`: first-order values in SF-43's periodic boxes (section 2.1; H3, H17; the positive control's target).
- `design_cost.py`: GPU-h, two-GPU stage, wall times of variants A, B, C (sections 3.g, 4).
- `lattice_check.py`: common admissible `M` per amplitude (H10).
- `gate4_power_sim.py`: Gate 4 simulations with `mcstat` (components from SF-39's reduced tables; P(RL > 0), MDE, forced `p_L`, the
  `numerical` fraction). The `sigma^2` 2.25 / 4 cells of `pipeline` reproduce the review's values within Monte Carlo error (seeds were
  re-labelled); the review quotes the script's values.
- `gate5_options.py`: Gate 5 pilot moments with bootstrap bounds, Fieller-form re-judging of SF-42's coverage cells, the
  skewness-corrected prototype, sizing, G7 effect sizes, runtime-outcome correlations.
- Sampling-limit coverage at the reporting tails, read from SF-42's `coverage/summary.json`: bootstrap-t passes the 0.05 and 0.025 tails at
  `n` 100, 200, 500 under the three distributions (largest UCL/tail 1.199); Student fails the skewed side at every `n`.

## 6. Findings

### Blocking (for activation as specified)

- **B1.** The specified design does not fit before the deadline and exceeds G10 with the measured overhead (147-185 GPU-h). -> H2, H9.
- **B2.** No Gate 5 procedure is qualified at the planned `n` (SF-42 Finding (j)). -> H5, H15.
- **B3.** The two-GPU first stage contradicts decision 2026-10-27 items 1 and 3, `AGENTS.md` and the runbook. -> H14; helper option prepared.
- **B4.** SF-42's coverage evidence does not reach every component of the robust sign at the proposed sizes and Holm levels (no qualified
  paired contrast for `E_int`; Student sampling never qualifies; `m = 8` tails not simulated). -> H15 (pipeline route, bootstrap-t sampling
  at 200, new cells).

### Major

- **M1.** The first-order benchmarks were those of an unbounded medium; in SF-43's periodic boxes the first-order window value is <= 0.
  -> H3 (conservative null, rationale corrected), H17 (box references), section 3.b.
- **M2.** In the displacement basis `E_h` rests on the asymptotic range of a triple with an untested `lambda/h = 4` level at 2.25. -> H6
  (gates).
- **M3.** At 2.25 the `E_W` interaction measured from 8 to 10 does not cover the transfer to 16 under an order-1 width effect, and no common
  `M` exists with that design. -> H7, H10.
- **M4.** `sigma^2 = 0.25` is uninformative in the position basis and has no G5.7 label in the displacement basis. -> H2, H4.
- **M5.** `alpha_LP` at `sigma^2 = 4` is non-Fickian and slow-tail dominated; Gate 5 there may `disagree`. -> H5, H17.
- **M6.** G5.7's `numerical` and `unresolved (bounded)` had no levels. -> H16.
- **M7.** Solver stagnation at `sigma^2 = 4` will exceed the specification's 1 % failure level and replacement conditions on solvability.
  -> H5, H13 (reserves, Par2 usability rule, sensitivity, `max_div`).
- **M8.** The wall-clock cutoff and the engine's original order (heavy before priority) could cut primary arms first. -> H13 (priority-first,
  floors, prefix rule).
- **M9.** Blinded re-estimation can inflate an equivalence test and was not costed. -> H5, H9 (inert definition; Gate 5 excluded).
- **M10.** Executability gaps found in the prepared tooling (executor constructor and blinded-SD contract, reconciliation of Par2-generator
  logs, unretried failed reductions, recomputation coverage of the displacement basis, cut order, G7's matched cell outside the fraction).
  -> H20, H21 and the Phase-2 corrective nodes.

### Minor

- **m1.** Bibliographic corrections (Berger 1982 DOI, Holm 1979 without DOI, 2D studies, studies that do not solve Darcy).
- **m2.** Imprecisions in `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` and `lester-2023-key-claims.md` (bibliography section 7). -> H19.
- **m3.** Decisions at the Holm levels rather than on the p-value lattice (it over-states `p` by up to 1.16 %). -> H4.
- **m4.** `mcstat.sizing.gate4_power` is optimistic in the fallback regime (0.875 against a simulated 0.61-0.69). Size by simulation.
- **m5.** Single-realization memory peaks under-state multi-realization peaks by about 1 GB; peaks are in MiB.
- **m6.** The host stack has matplotlib (SF-32 used it in host jobs); H18 checks it in the smoke test.

## 7. Adversarial verification summary

Five independent read-only reviews tried to refute the draft (statistics and design; numerics and verification; physics and fidelity to the
literature, re-verifying 23 citations, all 21 DOIs queried resolving; executability against the code and the prepared tooling; deadline and
cost). Corrected as a result:

- the first-order benchmark (two reviewers independently): the SF-18 field's periodicity along the flow makes the box value non-positive;
  the null is kept as a conservative, geometry-independent threshold and the decay references are recomputed per box;
- `E_h`: the order branch is operative in the displacement basis, so it is now gated on a fourth level; the fallback's own assumptions are
  stated;
- coverage: the robust sign is qualified at the pipeline level, production moves to 200 with a bootstrap-t sampling limit, and new cells
  cover the `m = 8` tails and the variant's arm sizes; Gate 5's J3B-bound family is gating at 4; Hall's test needs `n` >= 200;
- `E_W` at 2.25 moves to `lambda/h` 8 / 12, which also removes the source change;
- solver stagnation at 4, Par2 usability, backflow and first passage, the Dirichlet outlet's reach, the box periodicity in the assumption
  table;
- citations: Lester's conditional proof is stated; Lester 2025's App. D is an assertion without data; Lester 2023's RK test field is not
  exactly helicity-free; B&dD's scheme is described as printed; Cvetkovic et al. 1992 no longer supports the ergodicity claim; Chiogna
  2015 is cited as contrary evidence; Dagan's abstracts are cited only for what they state; Calvo & Hairer replaces the shadowing citations;
- schedule: lane classes modelled, lane B released early, `h_max` merged into production, G7 runs all eight cells, phase 2 restated as
  12-25 h, go/no-go limits added, the coverage run placed inside the smoke-test job;
- executability defects in the prepared tooling: listed as Phase-2 corrective nodes (H20).

Survived (re-derived or re-run by the reviewers): the unbounded first-order formula and values; H3's null is conservative and its coverage
shift-invariant; Holm with `p = 1` is valid; the intersection-union over `alpha_TP` and `alpha_TAP`; the Fieller form keeps the estimand and
the procedure-selection rule is data-independent; bootstrap-t at half the Holm level judged against the Holm level (1.000 / 0.633 / 0.641
at `n` 100 / 200 / 500); the prefix rule; `var x0` constant in `d`; H6's bounds larger or equal pointwise; the lattice sets; the SD ratios;
the memory thresholds; common random numbers across `lambda/h` (the generator draws each mode from a stateless hash of the seed and the
mode index); the runner options for every arm; the two-GPU helper commit; variant A's fit to the deadline from the stated launch window.

## 8. Caveats

- Most literature was read at abstract level; B&dD, Lester 2021, 2022, 2023, Attinger et al. 2004 and Dentz et al. 2016 were read in
  full.
- Planning rests on pilots: Gate 5 pairings on 8 pairs per amplitude (the `sigma^2 = 4` consumer CV on one field: 0.30, or 0.12 without
  seed 420836); SD ratios on one SF-39 set per amplitude; displacement-basis ladder SDs assumed equal to position-basis ones in the power
  runs (conservative for the noise, not for the order branch, hence H6's gate).
- The box first-order values ignore the Dirichlet faces; the `lambda/h = 4` level and the 1248x576x576 grid are unmeasured for the transport
  estimand and cost (smoke test).
- Costs are measured on SF-42's pilots; colleagues sharing a lane can double wall times; the pessimistic margin covers about 30 % of lane
  time shared; the budget guard bounds GPU-h.
- The Hall-corrected and halved Gate 5 procedures and the pipeline at the variant's sizes are qualified only by H15's run, after this
  review; the ordered list always ends in a procedure or in a descriptive Gate 5 with `p = 1`.
- Nothing here is SF-43 evidence; no `alpha` statement is made.

## Outcome

The recommendations are decisions H1-H21 of
[`2026-11-02-sf43-pre-activation-decisions.md`](../decisions/2026-11-02-sf43-pre-activation-decisions.md). The owner decides H2-H5, H9,
H14, H15's qualification route and H20 in a single consultation, and the record states the outcome before this PR is merged. Its merge
commit is SF-43's increment base. The SF-43 Goal and acceptance thresholds are unchanged.
