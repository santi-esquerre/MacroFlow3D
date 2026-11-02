# SF-43 pre-activation review: verified bibliography

- What this is: the literature evidence of the SF-43 pre-activation review
  ([`../../2026-11-02-sf43-pre-activation-review.md`](../../2026-11-02-sf43-pre-activation-review.md)), one claim per row, each with
  its reference, DOI and locator, what it supports for SF-43, its assumptions, and the level at which it was read.
- What it is not: a literature review of macrodispersion in general, or evidence about SF-43's `alpha` (none exists).
- How it was made (2026-11-02): seven independent read-only agents, one per topic (theory; numerical studies; topology, chaos and W8;
  control-plane estimands; verification; statistics; full reading of the three local PDFs `docs/references/Dreuzy-2013-WRR.pdf`,
  `Lester-2023-WRR.pdf`, `Lester_2021.pdf`), using Crossref/OpenAlex for DOIs and WebFetch for primary texts. The orchestrator checked
  the claims the review relies on against the reports and the local PDFs. Most Wiley/AGU, Springer and Cambridge full texts were
  blocked (Cloudflare 403); where only the abstract was read, the claim is limited to what the abstract states.
- Levels: **FT** full text read and passage found; **A** abstract only; **M** metadata only; **S** secondary source. Candidates that
  could not be verified are listed in each section under "NOT VERIFIED — do not cite"; the review does not cite them for content.
- BibTeX of every verified reference: [`references.bib`](references.bib) (144 entries, DOIs checked against Crossref/OpenAlex;
  de-duplicated by DOI).
- Corrections found while verifying (the review uses the corrected forms): Berger 1982 is `10.1080/00401706.1982.10487790` (not
  `...10487759`, which is another paper); Holm 1979 has no DOI (JSTOR 4615733); Shapiro & Cvetkovic 1988 is
  `10.1029/WR024i010p01711`; Salandin & Fiorotto 1998 and Hassan, Cushman & Delleur 1998 are 2D; Salandin & Fiorotto 2000 is
  `10.1029/2000WR900037`; Englert et al. 2006 report velocity statistics only; Dentz et al. 2002 and Schwarze et al. 2001 do not solve
  Darcy's equation; part 1 of the 2003 WRR series is Dagan, Fiori & Janković (not Fiori et al.).
- Inference marked as such in the reports ("my inference", "derived estimate") is the agents' reasoning, not the papers'; the review
  repeats only inferences the orchestrator re-derived (section 5 of the review, scripts in [`scripts/`](scripts/)).

## 1. Theory of 3D purely advective transverse macrodispersion

**Main finding:** first-order theory already predicts that α_T is **positive at every finite distance**. It only falls to zero as d → ∞, and for Gaussian covariance it decays like d⁻². So a positive sign on [25, 94] λ does not, by itself, separate the theories. What does separate them is the size of α_T and how it changes with d. The pilot's α_TP is roughly 75 to 200 times the first-order prediction (section 2).

**Access limits:** I read only one paper in full: Attinger, Dentz & Kinzelbach 2004, via the ETH Research Collection copy. All Wiley/AGU, Cambridge, Annual Reviews and Springer pages, plus the HAL, CORE and Internet Archive mirrors, returned 403 or a bot check. Everything else is verified at abstract or metadata level only. I give **no** equation or page numbers for Dagan 1984/1987/1988/1989.

#### 1. Claims table

| # | Claim (precise) | Reference | DOI verified (Crossref) | Exact location | What it supports for SF-43 | Assumptions / regime | Verification |
|---|---|---|---|---|---|---|---|
| 1 | Second-order (lowest-order) transverse coefficient, infinite Pe: D22 = (1/U)∫₀^∞ ũ2(x1′,0,0)ũ2(0,0,0) dx1′. "The integral scale of the transverse velocity correlations is zero (Hole-effect) and the transverse dispersion coefficients always vanish in second order approximation." | Attinger, Dentz, Kinzelbach 2004, "Exact transverse macro dispersion coefficients for transport in heterogeneous porous media", SERRA 18(1):9–15 | yes, 10.1007/s00477-003-0160-6 | Sect. 5, Eqs. (9)–(10), p. 11 | Lowest-order *asymptotic* α_T = 0 in 3D isotropic media | Pe → ∞ (Eq. 7, "purely advective"); asymptotic; source extended over many correlation lengths transversely (Sect. 2); ergodic | full text read |
| 2 | Linearized Darcy flow has zero rotation along e1, so "the transverse macro dispersion coefficients are zero in second order approximation" | same | yes | Sect. 5.2, Eqs. (13)–(16), p. 11 | Explains why the first-order asymptote is zero | linearized flow; 3D | full text read |
| 3 | 2D: transverse macrodispersion is **exactly** zero at all orders. 3D: "Higher order contributions do not cancel out yielding finite transverse macro dispersion coefficients". In their simulations α_T grows with σ_f² (Fig. 2, data from Dentz et al. 2002). They credit Neuman & Zhang (1990) as the first to show a nonzero 3D asymptote. | same | yes | Abstract; Sect. 6 (p. 12); Sect. 7 and Sect. 8 (p. 13); Fig. 2 | The main analytical support for a **nonzero** asymptotic α_T in 3D beyond first order | Pe → ∞ in theory; the 3D "proof" is a non-existence argument (App. B) and gives no value; simulations use linearized Darcy fields; Pe of the 3D runs not stated | full text read |
| 4 | Without local dispersion, linear theories give transverse dispersivities → 0. Quasi-linear theory: they rise to a peak, then decline to **nonzero Fickian asymptotes ∝ σ_Y⁴** (σ_Y² ≪ 1); ∝ σ_Y when σ_Y ≫ 1 | Neuman & Zhang 1990, WRR 26(5):887–902 | yes, 10.1029/WR026i005p00887 (Crossref also has a duplicate, 10.1029/89WR03602) | Abstract | A competing nonlinear prediction; σ⁴ scaling | **Exponential** covariance; 3D; quasi-linear (Corrsin-type) | abstract only |
| 5 | Isotropic covariance: asymptotic transverse macrodispersivity is proportional to local dispersivity, orders of magnitude below α_L | Gelhar & Axness 1983, WRR 19(1):161–180 | yes, 10.1029/WR019i001p00161 | Abstract | Asymptotic α_T → 0 as local dispersion → 0 | Eulerian, first order, asymptotic, with local dispersion; covariance form not verified | abstract only |
| 6 | First-order, Lagrangian approach; longitudinal coefficients reach constant values only after "a few tens of conductivity integral scales" in 2D and 3D | Dagan 1984, J. Fluid Mech. 145:151–177 | yes, 10.1017/S0022112084002858 | Abstract | Pre-asymptotic regime at d = 25–94 λ is plausible; **no transverse formula verified** | first order | abstract only |
| 7 | Time-dependent spatial variances at first order in σ_Y², axisymmetric anisotropy; first order "warranted, at present, only for σ_Y² ≪ 1" | Dagan 1988, WRR 24(9):1491–1500 | yes, 10.1029/WR024i009p01491 | Abstract | Validity caveat for σ² ≥ 1; X22 equations **not seen** | first order, high Pe | abstract only |
| 8 | 2D exact nonlinear (transport) correction to transverse dispersion tends to zero asymptotically; Corrsin's conjecture wrongly gives a finite value | Dagan 1994, WRR 30(10):2699–2705 | yes, 10.1029/94WR00904 | Abstract | 2D only; contrasts with 3D (claim 3) | 2D, normal velocity field | abstract only |
| 9 | Second-order flow correction: asymptotic transverse macrodispersivity → 0, like first order. Corrsin gives a nonzero asymptote. Authors cannot say which approximation is right for σ² ≥ 1. | Hsu, Zhang & Neuman 1996, WRR 32(3):571–582 | yes, 10.1029/95WR03492 | Abstract | Higher order in flow does not create an asymptote in **2D** | 2D, isotropic, exponential | abstract only |
| 10 | Second-order corrections to transverse velocity variances differ from first order by ~2σ_f² (isotropic and anisotropic) | Deng & Cushman 1995, WRR 31(7):1659–1672 | yes, 10.1029/94WR02974 | Abstract | First-order velocity statistics are biased at σ² ~ 1; no α_T statement | exponential covariance | abstract only |
| 11 | Effective coefficient = average of the per-realization **centred** second moments; ensemble coefficient = second moment of the ensemble-mean concentration. The ensemble one "considerably overestimates" a typical realization at finite times. They become equivalent only after dispersive spreading over a correlation length, through **local transverse dispersion**. | Dentz, Kinzelbach, Attinger, Kinzelbach 2000a, WRR 36(12):3591–3604 | yes, 10.1029/2000WR900162 | Abstract | The pilot's per-realization centring is the **effective** definition | second-order perturbation; point injection; with local dispersion | abstract only |
| 12 | Effective and ensemble coefficients are equivalent at infinite time **or infinitely extended sources**, and differ for finite time and finite source. Explicit second-order solutions for several isotropic and anisotropic initial conditions. | Dentz et al. 2000b, WRR 36(12):3605–3614 | yes, 10.1029/2000WR900211 | Abstract | Finite-source effect (Q3, Q4) | second order; extended sources | abstract only |
| 13 | Same two definitions. With a growing source, effective → ensemble faster; for a very large source this happens over the advective time for one correlation length. | Attinger, Dentz, H. Kinzelbach, W. Kinzelbach 1999, J. Fluid Mech. 386:77–104 | yes, 10.1017/S0022112099004334 | Abstract | With a source of 19.2 λ (9.6 λ at σ² = 4), effective ≈ ensemble after a few λ, even with zero local dispersion | model has heterogeneous **retardation**, not conductivity; analogy only | abstract only |
| 14 | 3D random-walk simulations in linearized Darcy flow with Gaussian-distributed log K, compared with second-order and Corrsin results; large deviations as heterogeneity grows | Dentz et al. 2002, WRR 38(7) | yes, 10.1029/2001WR000436 | Abstract | Source of the 3D numerical α_T in claim 3 | linearized flow; covariance shape not verified | abstract only |
| 15 | Effective dispersion coefficients defined from spatial moments "depend generally on the initial size of the solute body and on travel time"; ergodicity conditions | Dagan 1990, WRR 26(6):1281–1290 | yes, 10.1029/WR026i006p01281 | Abstract | Finite-source dependence (Q4) | first order; stratified example | abstract only |
| 16 | Finite input zone: effective D_L for a thin body transverse to the flow reaches the ergodic limit only for l/I ~ 10²; for smaller sources it is bounded | Dagan 1991, J. Fluid Mech. 233:197–210 | yes, 10.1017/S0022112091000459 | Abstract | Longitudinal only. Relevant to α_LP: source 19.2 λ ≈ 21.7 I < 10² I | first order; isotropic | abstract only |
| 17 | Ensemble second moment with a plume-scale low-wavenumber filter (two-scale relative dispersion) in 3D | Rajaram & Gelhar 1993, WRR 29(9):3261–3276 | yes, 10.1029/93WR01068 | Abstract | Plume or source size filters heterogeneity (Q4) | small perturbation, Eulerian | abstract only |
| 18 | Self-consistent (SC) and first-order (FO) approximations give 3D transverse macrodispersivity → 0. Numerical simulations give "a small but persistent transverse macrodispersivity for large travel times". FO overestimates transverse values at small times. | Janković, Fiori, Dagan 2003, WRR 39(9) (part 3) | yes, 10.1029/2002WR001721 | Abstract | Direct numerical precedent for persistent α_T > 0 in 3D isotropic media at σ² = 2–8 | spherical-inclusion (multi-indicator) medium, **not** multi-Gaussian | abstract only |
| 19 | Multi-indicator framework (part 1) and semi-analytical SC results for isotropic media (part 2). First-order α_L holds up to σ² ≈ 4 through error compensation. | Dagan, Fiori, Janković 2003 (part 1); Fiori, Janković, Dagan 2003 (part 2), WRR 39(9) | yes, 10.1029/2002WR001717 and 10.1029/2002WR001719 | Abstracts | Context; abstracts make no transverse claim | inclusion model | abstract only |
| 20 | First-order theory gives α_T = 0 in 3D isotropic media. Dilute spheres give α_T = 0 (axisymmetry). Randomly oriented ellipsoids (macroscopically isotropic) give α_T > 0, of order (κ−1)⁴. "The two-point covariance … is not able to detect the structures that produce advective mixing." | Janković, Steward, Barnes, Dagan 2009, WRR 45(8) | yes, 10.1029/2009WR007741 | Abstract | Whether the 3D asymptotic α_T is nonzero depends on structure beyond the covariance. Supports a 4th-order (σ⁴-like) onset. | inclusions with locally isotropic K | abstract only |
| 21 | 3D transverse macrodispersion "is known to vanish only in 2-D and not in 3-D". In Gaussian-correlated log-normal fields it "steeply increases with σ_Y²" (braiding), stays ≥ 2 orders of magnitude below longitudinal, and converges faster than longitudinal | Beaudoin & de Dreuzy 2013, WRR 49(5):2489–2496 | yes, 10.1002/wrcr.20206 | Abstract | Closest numerical analogue to SF-43 | "Gaussian correlated" is ambiguous about covariance shape; local-dispersion setting not verified | abstract only |
| 22 | Flux-proportional injection over a large area, control planes, macrodispersivity from travel-time variance, x ≤ 121 I, σ² = 2, 4, 8 | Janković, Fiori, Dagan 2006, WRR 42(6) (part 1) | yes, 10.1029/2005WR004734 | Abstract | Methodological precedent for α_LP | inclusion medium; longitudinal | abstract only |

#### 2. First-order formulas for Q1 (derived estimate, not taken from a primary source)

I could not read Dagan's equations. Below is my own derivation from the standard first-order Lagrangian setup:
- velocity spectrum u_ij(k) = U²(δ_i1 − k_ik1/k²)(δ_j1 − k_jk1/k²) S_Y(k);
- X22(t) = 2∫₀ᵗ(t−τ) u22(Uτ e1) dτ;
- consistent with Taylor's formula in Attinger et al. 2004, Eqs. (9)–(10).

Assumptions: unbounded domain, zero local dispersion, x = Ut, and one-particle (ensemble) variance.

**Covariance and spectrum.** C_Y(r) = σ² exp(−r²/λ²), so S(k) = σ²λ³/(8π^{3/2}) e^{−k²λ²/4} and I = (√π/2)λ ≈ 0.886 λ.

**Transverse variance.** With ξ = x/λ:
- X22(x) = 2∫ S(k)(k2²/k⁴)(1 − cos k1x) d³k
- = (σ²λ²/4√π) ∫₀^∞ e^{−q²/4} [4/3 − 4(sin qξ − qξ cos qξ)/(qξ)³] dq

**Asymptote.** In general X22(∞) = (2/3)∫₀^∞ r C_Y(r) dr.
- Gaussian: X22(∞) = **σ²λ²/3**.
- Exponential: X22(∞) = (2/3)σ²I².

**Transverse dispersivity.**
- α_T(x) = ½ dX22/dx = (σ²λ/8√π) ∫₀^∞ q e^{−q²/4} J(qξ) dq, with J(a) = 4[(3−a²)sin a − 3a cos a]/a⁴.
- α_T > 0 for every x checked (numerically, x = 0.05–100 λ).
- Peak ≈ 0.046 σ²λ at x ≈ 1.2 λ.

**Far field.** α_T ≈ ∫C_Y d³r / (8πx²), so X22 approaches its asymptote like −1/x.
- Gaussian: α_T ≈ (√π/8) σ²λ³/x² = σ²I³/(πx²).
- Exponential: α_T ≈ σ²I³/x².
- At equal I the two models differ by a factor of π.

**Numbers at σ² = 1** (computed by quadrature; the far-field form agrees within 0.3%):

| d (λ) | X22/λ² | α_T/λ |
|---|---|---|
| 25 | 0.3156 | 3.5 × 10⁻⁴ |
| 94 | 0.3286 | 2.5 × 10⁻⁵ |

**Pilot-like estimand.** ½ × the least-squares slope of X22 vs d over [25, 94] λ (dense uniform planes) ≈ **8.0 × 10⁻⁵ σ²λ**. Compared with the pilot:

| σ² | First-order estimate (λ) | Pilot α_TP (λ) | Pilot / first order |
|---|---|---|---|
| 0.25 | ~2 × 10⁻⁵ | ~0 | consistent |
| 1 | 8 × 10⁻⁵ | 0.006 | ~75 |
| 2.25 | 1.8 × 10⁻⁴ | 0.028 | ~155 |
| 4 | 3.2 × 10⁻⁴ | 0.066 | ~205 |

Two further derived caveats:
- The d⁻² tail comes from the smallest wavenumbers. Periodic transverse boundaries (W = 24 λ or 12 λ) remove k⊥ < 2π/W, so first-order α_T in the simulated box should be even smaller at d ≫ W.
- Per-realization centring with a large source makes the first-order effective value slightly lower again.

#### 3. Synthesis (≤ 300 words)

What a positive α_T at finite distance does **not** show by itself:
- **Linear theory also predicts a positive sign.** First-order theory gives α_T > 0 at every finite d, going to zero as d⁻² (section 2). The sign alone does not refute it.
- **Numerical artefacts push the same way.** Non-solenoidal velocity interpolation, particle-tracking error and finite width usually add spurious transverse spreading, which is positive-definite. The claim that the sign is "robust to biases" therefore needs measured bounds on bias size, not the sign.

What the pilot does show, and how it fits the theory:
- **Two observations go beyond linear theory.** (i) The magnitude is about 75–200 times first order. (ii) Linear theory predicts a 14-fold drop in α_T between 25 and 94 λ; a slope that does not decay with d contradicts that.
- **The scaling fits nonlinear predictions qualitatively.** The pilot grows faster than linearly in σ² (between σ³ and σ⁴). That is consistent with the σ_Y⁴ asymptote of Neuman & Zhang (1990) and the (κ−1)⁴ onset of Janković et al. (2009). It also matches the 3D numerical results of Janković et al. (2003), Beaudoin & de Dreuzy (2013) and Attinger et al. (2004, Fig. 2). Consistent does not mean confirmed.

What the theory does **not** settle:
- **No theorem fixes the 3D asymptote.** In 3D no result forces α_T to zero beyond first order. Attinger et al. (2004) argue it is finite. The self-consistent sphere-inclusion theory and the dilute-sphere result give zero.
- **The covariance cannot decide it.** Janković et al. (2009) show the outcome depends on structure that the two-point covariance cannot detect. A Gaussian covariance does not determine whether the asymptotic α_T in multi-Gaussian fields is zero.
- **SF-43 cannot claim a Fickian or asymptotic α_T**, a mechanism (braiding, rotation along e1), or field relevance.
- **The estimand is the effective (centred) dispersion.** Ensemble values would be larger (Dentz et al. 2000a,b). With a 19.2 λ source the difference should be small, but this has not been checked for planes.

#### 4. BibTeX (verified metadata only)

(BibTeX entries: `references.bib`.)

Dagan 1987, Dagan 1989 and Dentz et al. 2002 are metadata-verified only. Cite them for existence, not for specific content.

#### 5. Not verified: do not cite for specific content

- **Dagan 1984, 1987, 1988, 1989:** any equation or page number for X22(t), the α_T decay law, or the peak value. Full text was blocked (403). The only claims usable are those in their abstracts.
- **Dagan 1987:** content unverified. OpenAlex attaches an abstract from an unrelated machine-learning review. Crossref pages are 183–213, not 183–215.
- **Rubin 2003, *Applied Stochastic Hydrogeology*** (OUP, 10.1093/oso/9780195138047.001.0001): the book exists, but I could not check its contents or any Gaussian-covariance formulas.
- **Deng & Cushman** on transverse *macrodispersion*: no such statement found. The 1995 paper is about velocity covariance and is 31(7), not 31(9). The 1998 "revisited" paper (10.1029/97WR02610) is metadata-only.
- **Covariance shape (Gaussian vs exponential):** not verified for Dentz et al. 2000/2002, Gelhar & Axness 1983, or Beaudoin & de Dreuzy 2013. Whether Beaudoin & de Dreuzy used zero local dispersion is also unverified.
- **Fiori et al. 2017, AWR 110:349–359** (10.1016/j.advwatres.2017.10.025) and **Kitanidis 1988, J. Hydrol. 102:453–473** (10.1016/0022-1694(88)90111-4): metadata only.
- **Legacy WRR article numbers** (e.g. 1268, W08415): not in Crossref. Cite by DOI.
- **Attribution fix:** the candidate "Fiori, Janković & Dagan 2003 parts 1–2" is wrong for part 1, whose authors are Dagan, Fiori & Janković.
- **Reference-list error in Attinger et al. 2004:** their entry for Hsu et al. is wrong ("1988, 33: 1187–2299"). Crossref gives 1996, 32(3):571–582.

Sources: [Attinger et al. 2004, ETH copy](https://www.research-collection.ethz.ch/handle/20.500.11850/51880) (also on [doc.rero.ch](https://doc.rero.ch/record/314853/files/477_2004_Article_160.pdf)); [Janković et al. 2009 record](https://wrl.mnpals.net/islandora/object/WRLrepository%3A1282); [EarthRef](https://earthref.org/ERR/68204); [Springer chapter, Hsu et al. 1996](https://link.springer.com/chapter/10.1007/978-94-009-0205-3_4); [CORE](https://core.ac.uk/works/29854367). Metadata came from the Crossref, OpenAlex, Semantic Scholar and Unpaywall APIs, accessed 2026-11-02.

---
I did not read Dagan's or Dentz's full texts, so section 2 is my own derivation, checked only for internal consistency (asymptote and far-field tail). I did not model plane-based versus time-based moments or the periodic-box effects; both are flagged above but not quantified.

## 2. Numerical 3D (and key 2D) studies

#### 0. What I could read, and the main corrections

**Full text read (3 papers):**
- Beaudoin & de Dreuzy 2013 (local PDF, pp. 2489–2496).
- de Dreuzy, Beaudoin & Erhel 2007 (HAL author preprint insu-00193397; line numbers below refer to the preprint, not the published WRR pages).
- Dartois, Beaudoin & Huberson 2018 (open access).

**Everything else:** I checked the DOI against Crossref and read the abstract through OpenAlex. Some details come from secondary sources, and each one is labelled with where it came from. Wiley, Springer and the unipd repository all returned 403 or a Cloudflare challenge, so I could not read those full texts.

**Several candidates are wrong as framed in the brief:**
- **Salandin & Fiorotto 1998 is 2D**, not 3D (stated in the abstract).
- **Salandin & Fiorotto 2000** is a short theoretical note: first-order 3D, with a 2D nonlinear comparison and pore-scale dispersion. It is not a 3D simulation study. Its DOI is **10.1029/2000WR900037**. The DOI 10.1029/2000WR900060 resolves to an unrelated Glass et al. correction.
- **Hassan, Cushman & Delleur 1998 is 2D** (fBm and exponential covariance).
- **Janković 2003/2006/2009** use a medium made of spherical or ellipsoidal inclusions. These are non-smooth and not multi-Gaussian, so they are a different medium class from ours.
- **Englert 2006** reports velocity statistics only. It has no dispersivities.
- **Schwarze 2001** uses a synthetic random velocity field, not a Darcy solution (Lester et al. 2023, p. 4, §2). de Dreuzy 2007's Table 1 lists it as a **2D** first-order study. The dimension is therefore inconsistent between sources and is not resolved.
- **Dartois et al. 2018** is a missing 3D study that should be added. It comes from Beaudoin's group (the PARADIS / H2OLAB code), but its text says **exponential** covariance (§1).

---

#### 1. Per-study tables

##### 1.1 Beaudoin & de Dreuzy 2013, WRR 49, 2489–2496 — FULL TEXT

| Item | Finding (location) |
|---|---|
| Dimension / covariance | 3D. **Gaussian**: σ² exp(−(r/λ)²), eq. (1), p. 2489. Integral scale I = (√π/2)λ ≈ 0.886λ, which is consistent with eq. (2). |
| σ² values | 0.25, 1, 2.25, 4, 6.25, 7.56, 9 (Fig. 2). Table 1 lists "6.52", probably a typo for 6.25. |
| Domain | L_L = 102.4λ for σ² ≤ 1, up to 1638.4λ at high σ² (p. 2492 [9]–[10]). L_T/λ = 25.6 (Fig. 1 caption, σ² = 6.25). |
| Resolution | λ/Δ = 10. One test at λ/Δ = 20 (p. 2491 [7], Fig. 1). |
| Boundary conditions | Fixed heads on the faces normal to L. **Periodic on the transverse faces**, chosen "to minimize the perturbation of the flow field generally reported under no-flow" boundaries (p. 2490 [6]). |
| Flow solver | Finite volumes with harmonic inter-cell K; HYPRE multigrid; FFTW spectral field generation (p. 2490 [6]). |
| Transport | Pure advection. First-order explicit (Euler) tracking. Velocity is linearly interpolated per direction (Pollock 1988), with about 10 steps per cell (p. 2490 [6]). |
| Injection | Plane of **0.8 L_T1 × 0.8 L_T2**, at least **5λ** downstream of the inflow face, **flux-proportional** (p. 2490 [6]). A (20λ)² window was tested and judged "more than enough" (p. 2491 [7]). |
| Estimator | **Resident spatial moments in time.** Eq. (4) centres the moments per realization ("effective"), takes d/dt, then averages over N_R. Time is in units of λ/‖⟨v⟩‖, and α_T1 = α_T2 (p. 2490). |
| Fitting window | α_kAM is the average over the **last quarter of [0, t_b]**, where t_b is the time the first particle leaves the domain (eq. 5, p. 2492). α_LAF comes from an exponential fit, eq. (6), Table 1. |
| Ensemble | **N_R = 500, N_p = 10⁴** (p. 2491 [7]). |
| α_T results | Table 2 (p. 2493): **<10⁻³ at σ² = 0.25, 0.01 at 1, 0.044 at 2.25**. For σ² ≥ 1, α_TAM ≈ α_TAM(1)·σ⁴ (eq. 7). Fig. 4b reads roughly 0.11 at σ² = 4 and about 0.5 at 7.56–9 (my approximate read-off). α_T(t) **overshoots** near t ≈ 1–3 and then decays to its asymptote by t ≈ 10–30 (Fig. 2b). α_T "converges much faster" than α_L (p. 2491 [9]). |
| α_L results | Table 1, α_LAF: 0.24, 1.27, 4.35, 12.7, 58.6, 131. t_LF: 1.35, 2.21, 4.83, 20.5, 41.9, 208. Above σ² = 2.25, α_LAM ≈ exp(σ²/1.55) (eq. 9). α_L/α_T ≈ 100–200 (p. 2493 [16]). α_L at σ² = 9 is never converged (p. 2492). |
| Sampling / convergence | Changing to N_p = 10⁵, N_R = 250, or λ/Δ = 20 (at σ² = 6.25 only) moved the asymptotic values by ≤2% (L) and **≤4% (T)** (p. 2491 [7]). There are no confidence intervals. |
| Stated limits | α_LAM is a lower bound for σ² ≥ 6.25 (p. 2492). The transverse width L_T was **never varied** in what is reported. |

Companion paper: Beaudoin & de Dreuzy 2013, *Mech. Ind.* 14(6) 461–464 (abstract only). It reports the 3D/2D α_L ratio as "up to 7 at σ² = 7.56"; the WRR paper says 5 at σ² = 6.25.

##### 1.2 de Dreuzy, Beaudoin & Erhel 2007, WRR 43, W10439 — FULL TEXT (HAL preprint)

| Item | Finding |
|---|---|
| Setup | 2D, **exponential** covariance, σ² 0.25–9 (abstract; l. 482–484). |
| Domain / resolution | 819×819λ for σ² ≤ 4; 1638×819λ for σ² ≥ 6.25 (l. 263–265). λ/l_m = 10; 5 and 20 were also tested and show weak sensitivity (l. 408–415). |
| Boundary conditions | Fixed head on two faces, **no-flow laterally** (l. 150–154). Particles stay out of a 15% "exclusion zone" next to the lateral walls (l. 215–220). |
| Transport / injection | Particle tracker with bilinear velocity interpolation. Line source of 0.4 L_y (about 328λ), injected **uniformly, not flux-weighted** (l. 208–215). |
| Estimator | Resident moments, d/dt, centred per realization ("effective"); the paper defines both effective and ensemble versions (l. 225–277). Window: average over [0.5 t_fb, t_fb] (l. 241–253). |
| Ensemble | N_S = 100, N_p = 10⁴. For N_S ≥ 20 the mean α_T changes by <10%. The standard deviation of D_T is about 0.6, far larger than its mean (l. 316–323). |
| Results | Pure-advection α_T is **0 on average**, with no sign trend (l. 431–441). D_LA ≈ 0.7σ² + 0.2σ⁴ (l. 429–430). |
| Limitations | A small 102.4×51.2λ domain was "far from" asymptotic. The authors attribute this partly to the narrow injection window: 20λ versus 327λ (l. 361–367). |

The preprint's Table 1 (p. 32) summarises earlier **2D** studies (secondary source). Its column headers read Lx/lm and Ly/lm, but the caption defines these as lengths in correlation lengths:

| Study | σ² | cells/λ | Lx×Ly (λ) | MC | particles |
|---|---|---|---|---|---|
| Bellin 1992 | 0–1.6 | 8 | 36×18 | MC·PT = 1500 | — |
| Cvetkovic 1996 | 0–4 | 4 | 24×18 | 500–1000 | 1 |
| Salandin & Fiorotto 1998 | 0.05–4 | 2, 4, 8 | 64×64 | 500 | 40 |
| Trefry 2003 | 0.25–4 | 8 | 256–1024 × 64–256 | 1 | Eulerian scheme with local dispersion |

##### 1.3 Dartois, Beaudoin & Huberson 2018, C. R. Mécanique 346, 89–97 — FULL TEXT (open access)

| Item | Finding |
|---|---|
| Setup | 2D and 3D, **"exponentially correlated"** isotropic lognormal field (§1). This conflicts with the Gaussian field of BdD 2013 from the same group; unresolved. |
| Parameters | σ² = 0.25, 1, 2.25. Pe = 50, 100, ∞ (Fig. 4–6). |
| Domain | 3D: **51.2 × 25.6 × 25.6λ**. 2D: 204.8 × 102.4λ (§2, Table 1). Boundary conditions are **not stated**. |
| Estimators | Lagrangian velocity covariance, plus the time derivative of spatial moments (§4). |
| Ensemble | N_S = 100, N_P = 10⁴ (§2). |
| Results | D_T(t) shows "an early spire followed by a rapid drop" to its asymptote (§4). Fitted 3D Lagrangian velocity variances: σ_T² = 0.075σ² + 0.083σ⁴, σ_L² = 0.52σ⁴ (eqs. 17–18). The diffusion-induced increase ΔD_T is 2× larger in 2D than 3D at Pe = 10, σ² = 2.25 (§4). Asymptotic α_T values appear only in figures. |

##### 1.4 Janković, Fiori & Dagan 2003, WRR 39(9) 1270 — ABSTRACT plus secondary

| Item | Finding |
|---|---|
| Medium | Spherical inclusions (3D) with a lognormal K pdf. Explicitly "different from the multi-Gaussian one". 3D: n ≤ 0.7, σ² = 2, 4, 8 (the 2D/3D maximum is 10). |
| Method | Analytic-element method. Macrodispersivities from Lagrangian velocity covariances, as a function of travel time. |
| Result | In 3D, α_T is "small but persistent" at large travel time, while the approximate theories go to zero. BdD 2013 p. 2493 quotes α_L/α_T ≈ **250 at σ² = 4**. |
| Not seen | Domain, boundary conditions, N, particles. |

##### 1.5 Janković, Fiori & Dagan 2006 (Part 1) and Fiori, Janković & Dagan 2006 (Part 2), WRR 42 — ABSTRACTS

| Item | Finding |
|---|---|
| Medium / runs | Part 1: 100,000 spherical inclusions; **3 realizations**, one each at σ² = 2, 4, 8. |
| Injection | **Flux-proportional** over A₀ ≈ 2000 I²; 40,000 particles. |
| Estimator | **Mass flux at control planes**. The "equivalent macrodispersivity" is defined from the **travel-time variance**, for x ≤ 121 I. This is the same estimator family as our α_LP. |
| Results | Strong tail, cutoff-dependent (0.995 vs 0.999 of mass), non-ergodic tail, anomalous over the simulated range. |
| Transverse | **Not reported in the abstracts.** Part 2 is a semi-analytical self-consistent model compared against these runs. |

##### 1.6 Janković, Steward, Barnes & Dagan 2009, WRR 45, W08415 — ABSTRACT

Ellipsoidal inclusions with random orientation. In the dilute limit, α_T = 0 for spheres. In numerical runs, α_T > 0; for κ = 10 and n = 0.2, **α_T/α_L ≈ 0.15** ("advective mixing"). The medium is non-smooth, and Lester 2023 (p. 20, §7) attributes the nonzero α_T to that.

##### 1.7 Chin & Wang 1992, WRR 28(6) 1531–1542 — ABSTRACT ONLY

3D Monte Carlo, isotropic, σ_Y 0.1–1.5 (σ² ≤ 2.25). **Confidence limits** for the ensemble averages were established. First-order α_T theory holds only for σ_Y < 1. Covariance type, domain, boundary conditions and N were not seen.

##### 1.8 Tompson & Gelhar 1990, WRR 26(10) 2541–2562 — ABSTRACT ONLY

3D random-walk particle method. **Single realizations** at 3 heterogeneity levels, with several source locations. Transverse mixing "differs markedly" from stochastic theory, and non-ergodic effects are strong. σ² values, covariance and domain were not verified; a secondary source says exponential, but I did not confirm it.

##### 1.9 Englert, Vanderborght & Vereecken 2006, WRR 42, W03418 — ABSTRACT plus secondary

3D multi-Gaussian field, σ² 0.5–3, anisotropy 0.017–1. **Velocity statistics only.** Per Nowak et al. 2008 ([11], p. 2), it used ≲10 realizations of >10⁷ cells and spatial averaging over 1/8 of the domain. It reported locally reversed flow above σ² = 3 (Nowak [47]). No-flow boundaries affect velocity variances only within <1λ (Nowak, p. 8). Covariance type not verified.

##### 1.10 Dentz et al. 2002 (WRR 38(7)) and Schwarze, Jaekel & Vereecken 2001 (TiPM 43) — ABSTRACT / secondary

- **Dentz 2002:** 3D, **linearized (first-order) velocity field**, random walk, both effective and ensemble coefficients. BdD Table 2 quotes α_TA = 0.007 (σ² = 1) and 0.028 (σ² = 2.25).
- **Schwarze 2001:** synthetic Kraichnan-type velocity field (Lester p. 4), with α_TA = 0.009 at σ² = 1 (BdD Table 2). de Dreuzy Table 1 lists it as 2D, σ² 0.1–1, 3200 MC.
- Neither study solves the Darcy equation.

##### 1.11 Two-dimensional studies (abstracts, plus de Dreuzy 2007 Table 1)

- **Salandin & Fiorotto 1998:** 2D, finite elements plus Monte Carlo, σ² 0.05–4. Found longitudinal dispersion above linear theory and slow convergence to the Fickian regime.
- **Bellin, Salandin & Rinaldo 1992:** 2D, σ² ≤ 1.6. I did not read the abstract.
- **Trefry, Ruan & McLaughlin 2003:** 2D, Eulerian taut-spline scheme with local dispersion, travel over 100λ, one realization each. Moments "continue to vary after long travel distances".
- **Gotovac, Cvetkovic & Andricevic 2009:** 2D, **exponential**, σ² ≤ 8, adaptive Fup Monte Carlo method (AFMCM). **Travel time and transverse displacement at control planes**; transverse displacement is nearly normal beyond x/I = 20. Domain and ensemble per Dartois 2018 Table 1 (secondary): 64×128λ, N = 500, 4000 particles.
- **Cvetkovic, Cheng & Wen 1996:** 2D, travel-time and transverse-displacement moments at control planes; first-order theory robust up to σ² ≈ 1.
- **Hassan, Cushman & Delleur 1998:** 2D, fBm and exponential fields, random walk with local dispersion. Tests Eulerian theory; not dispersivity-oriented.

---

#### 2. Cross-study comparison

| Study | Dim | Cov | σ² | Width/λ | Transverse BC | Estimator | Window | N | α_T result |
|---|---|---|---|---|---|---|---|---|---|
| BdD 2013 | 3D | **Gauss** | 0.25–9 | **25.6** | **periodic** | resident, time, effective | last ¼ of [0, t_b] | 500 (10⁴ p) | 0.01 (1), 0.044 (2.25), ∝σ⁴; ~0.5 at 7.56–9 |
| Dartois 2018 | 2D/3D | exp (stated) | ≤2.25 | 25.6 | not stated | velocity covariance + moments | — | 100 (10⁴ p) | figures only; early peak then decay |
| de Dreuzy 2007 | 2D | exp | 0.25–9 | 819 | no-flow | resident, effective | [½t_fb, t_fb] | 100 | 0 ± large SD |
| Janković 2003 | 3D | inclusions | 2–8 | n/v | (AEM) | Lagrangian velocity covariance | travel time | n/v | small, persistent; α_L/α_T ≈ 250 at σ² = 4 |
| Janković 2006 | 3D | inclusions | 2, 4, 8 | ~45 I source (A₀ ≈ 2000 I², if square) | n/v | **flux, travel-time variance** | x ≤ 121 I | 3 | not in abstract |
| Janković 2009 | 3D | ellipsoids | κ = 10 | n/v | — | — | — | — | α_T/α_L ≈ 0.15 |
| Chin & Wang 1992 | 3D | n/v | ≤2.25 | n/v | n/v | ensemble, with confidence limits | n/v | n/v | first-order valid only for σ_Y < 1 |
| Tompson & Gelhar 1990 | 3D | n/v | 3 levels | n/v | n/v | resident, single realization | — | 1 | differs from theory |
| Dentz 2002 | 3D | linearized velocity | ≤2.25 | — | — | random walk | — | n/v | 0.007 / 0.028 |
| Schwarze 2001 | 3D? | synthetic velocity | ≤1 | — | — | — | — | — | 0.009 |
| Gotovac 2009 | 2D | exp | ≤8 | 128? | n/v | **flux, control planes** | to ? | 500 | 2D (transverse displacement near-normal) |
| Cvetkovic 1996 | 2D | n/v | ≤4 | 18 | n/v | **flux, control planes** | — | 500–1000 | 2D |

(n/v = not verified.)

---

#### 3. Answers to questions 1–4

**Q1. Who reports 3D isotropic purely advective α_T at σ² ≥ 1, and with what estimator?**
- **Smooth multi-Gaussian Darcy fields:** only BdD 2013 (Gaussian) and Dartois 2018 (stated exponential, σ² ≤ 2.25). BdD reports positive α_T growing as σ⁴: 0.01 → 0.044 → ~0.5 for σ² from 1 to about 8. α_T converges fast, by about 10–30 advective times, after an early overshoot.
- **Inclusion media:** Janković 2003 and 2009 give α_T > 0, but the medium is non-smooth.
- **Not Darcy solutions:** Dentz 2002 and Schwarze 2001 (linearized or synthetic velocity).
- **Control-plane estimators:** none of the 3D studies applies one to α_T. Janković 2006 (inclusions) uses travel-time variance for α_L only. Gotovac 2009 and Cvetkovic 1996 use control planes for both α_T and α_L, but only in 2D.
- Lester 2023 (p. 20, §7) attributes BdD's nonzero α_T to particle-tracking error.

**Q2. Periodic transverse boundaries and width effects.**
- Only BdD 2013 among the transport studies is confirmed to use periodic transverse boundaries, with W = 25.6λ. Nowak 2008 also used periodic boundaries, but it is a flow-only study.
- de Dreuzy 2007 used no-flow walls plus an exclusion zone.
- **No study quantifies how the transverse width affects α_T.** BdD varied only the injection window. de Dreuzy found that a narrow domain or window delays longitudinal convergence (2D).
- Your **12λ case at σ² = 4 has no precedent.** My own estimate, not from the literature: with α_T ≈ 0.11–0.16λ, the transverse displacement standard deviation at d = 94λ is about 4.5–5.5λ, close to the 6λ half-width. Periodic wrap-around and self-correlation through periodic images need checking.

**Q3. Ensemble sizes and sampling error.**
- Typical N: 100–500 (BdD 500, de Dreuzy 100, Dartois 100). Janković 2006 used only 3.
- For α_T, the only sampling-type statement is BdD's ±4% sensitivity at σ² = 6.25. de Dreuzy reports N_S convergence and the standard deviation (2D).
- Chin & Wang give confidence limits (abstract). Naff et al. 1998 describe standard errors for macrodispersivities (abstract; the σ² range was not verified).
- No study gives a confidence interval for 3D Gaussian α_T at σ² ≥ 1.

**Q4. What is directly comparable to our [25, 94]λ design.**
- **Directly comparable (design mirror):** BdD 2013 — Gaussian field, periodic boundaries, W ≈ 24–25.6λ, flux-weighted 0.8W source at 5λ, λ/h = 10. If α_T is Fickian over d = 25–94λ, our α_TP should match BdD's α_TAM. Caveats: time domain versus space domain, resident versus flux weighting, and our λ/h = 16/32 at high σ².
- **Asymptotic only:** α_L for σ² ≥ 2.25. BdD's t_LF is 4.8 (σ² = 2.25) and 20.5 (σ² = 4), and α_L at σ² ≥ 6.25 is unconverged. Travel-time variance also weights slow tails differently (Janković 2006). Exponential-covariance studies are asymptotic-only too; when comparing, convert lengths through I = 0.886λ.
- **Not comparable:** the inclusion media and the linearized or synthetic velocity models.

---

#### 4. BibTeX (DOIs checked against Crossref)

(BibTeX entries: `references.bib`.)

#### 5. NOT VERIFIED — do not cite for content

- **Zhang & Seo 2004** (SERRA 18:205–215, doi 10.1007/s00477-004-0178-4): the citation exists, but I saw no content.
- **Schwarze 2001:** dimension (2D vs 3D) and setup.
- **Chin & Wang 1992 and Tompson & Gelhar 1990:** covariance, domain, boundary conditions, N, numerical values.
- **Englert 2006:** covariance type and boundary conditions.
- **Janković 2003:** domain, N, particle count.
- **Janković 2006:** whether any transverse result appears in the full text.
- **Gotovac 2009:** domain and N (secondary source only).
- **Dartois 2018:** transverse boundary conditions; whether its field is really exponential.
- **Naff 1998:** σ² range and isotropy.
- **Not examined in depth:** Janković et al. 2017 (AWR 100, 10.1016/j.advwatres.2016.10.024); Fiori et al. 2015 (10.1002/2015WR017118; 2D and 3D, control-plane breakthrough curves); Fiori et al. 2017 (AWR 110); Comolli et al. 2019; Hakoun et al. 2019; Moslehi / de Barros.
- **Dartois 2016 PhD thesis** (HAL tel-04316909): available, but the PDF text did not extract as searchable text.

**Sources:** local Dreuzy-2013-WRR.pdf and Lester-2023-WRR.pdf; [de Dreuzy 2007 (HAL)](https://insu.hal.science/insu-00193397); [Dartois 2018](https://comptes-rendus.academie-sciences.fr/mecanique/item/10.1016/j.crme.2017.12.012.pdf); [Nowak 2008](https://www.iws.uni-stuttgart.de/publikationen/hydrosys/paper/Nowak_al_2008_velocity_statistics.pdf); Crossref and OpenAlex APIs; [CROSBI record for Gotovac 2009](https://www.bib.irb.hr/408711); [EarthRef](https://earthref.org/ERR/79972); [CORE](https://core.ac.uk/works/29854367).

---
Not done: full texts of the Wiley and Springer 3D papers (blocked), and any check of BdD's unwrapping under periodic boundaries. Main risk: apart from BdD 2013, all the 3D numbers above are second-hand. Also, the σ = 4 / 12λ-width case has no literature precedent.

## 3. Topological constraints, helicity, chaos (W8)

**Bottom line:** the exponential pass-to-pass separation in W8 does not by itself contradict the helicity-free picture. The literature rules out *persistent* exponential separation in the exact flow, not finite-time separation or separation caused by numerical error. The two passes agreeing on α_T bounds only the error from the integration step cap. It does not check the error from the velocity representation, and that is the error the Lester group says creates spurious α_T.

#### 1. Claim table

| # | Claim | Reference | DOI verified? | Location | What it supports for SF-43/W8 | Assumptions | Verification level |
|---|---|---|---|---|---|---|---|
| 1 | Darcy flow in isotropic, heterogeneous K has zero helicity, so it cannot have tangled vortex lines or become chaotic. A function ℋ(K,H) exists on whose level surfaces all flow paths lie. | Sposito 1994, WRR 30(8) | Yes, 10.1029/94WR01328 | Abstract | Original non-chaos argument | Scalar K, steady flow | Abstract (OpenAlex) |
| 2 | (i) Zero helicity density is equivalent to Darcy's law. (ii) Flow with scalar K(x,t) is confined to surfaces on which streamlines are geodesics. (iii) Those surfaces are flat or tori. (iv) Chaotic streamlines are impossible, so mixing is poor. | Sposito 2001, AWR 24(7) 793–801 | Yes, 10.1016/S0309-1708(00)00077-4 | Abstract | Non-chaos; streamlines confined to 2D surfaces | Scalar K | Abstract from a repository mirror (geologyscience.ru) |
| 3 | Sposito's Lamb surfaces exist only for flows that are inherently 2D (axisymmetric, stratified), not for 3D isotropic Darcy flow in general. Non-chaos still holds because the flow is integrable. | Lester et al. 2019, TiPM 130 (cited in Lester 2021, Lester 2023) | Yes, 10.1007/s11242-019-01346-3 | Lester 2021 p. A27-6; Lester 2023 p. 5 | Corrects claims 1–2: the confining surfaces are streamfunction level sets, not Lamb surfaces | — | Secondary (as stated by the citing papers) |
| 4 | Zero-helicity steady 3D flows can be classified as integrable or non-integrable; numerical methods that preserve volume and helicity are developed. | Holm & Kimura 1991, Phys. Fluids A 3 | Yes, 10.1063/1.858083 | Abstract | Background | — | Abstract |
| 5 | Layered aquifers with horizontal anisotropy that differs between layers produce spiralling streamlines ("whirls"). With isotropic layers, streamlines are straight. | Bakker & Hemker 2004, AWR 27(11) 1075–1086 | Yes, 10.1016/j.advwatres.2004.08.009 | Abstract | Helical flow needs anisotropy | Dupuit / multilayer model | Abstract via VU repository (paraphrased there) |
| 6 | Whirls appear where the principal directions of the transmissivity tensor change between layers. | Hemker & Bakker 2006, WRR 42 | Yes, 10.1029/2006WR004901 | Abstract | Same as 5 | Semiconfined, layered aquifer | Abstract |
| 7 | Spiralling flow lines appear in layered aquifers whose adjacent layers have different anisotropy. | Hemker, van den Berg & Bakker 2004, Groundwater 42(2) | Yes, 10.1111/j.1745-6584.2004.tb02670.x | Abstract | Same as 5 | Numerical experiments | Abstract |
| 8 | Helical flow needs anisotropic K, in homogeneous or heterogeneous media. | Chiogna et al. 2014, AWR 73 | Yes, 10.1016/j.advwatres.2014.06.017 | Abstract | Isotropic K means no helicity | — | Abstract summary (FAU CRIS) |
| 9 | Fully resolved, locally isotropic K gives zero local helicity density. Ensemble-averaged and block-upscaled fields give non-zero helicity, so helicity is scale dependent. | Chiogna et al. 2015, WRR 51 | Yes, 10.1002/2014WR015330 | Abstract | Warns that upscaling can create spurious helicity | Non-stationary anisotropic covariance | Abstract. Note: the abstract also says streamlines look "helical in all three cases" at larger scales. Lester 2021 (p. A27-26) attributes this to non-smooth fields. |
| 10 | Chaotic advection is inherent to *pore-scale* flow, with saddle stagnation points acting like a 3D baker's map. | Lester, Metcalfe & Trefry 2013, PRL 111 | Yes, 10.1103/PhysRevLett.111.174101 | Abstract; Lester 2021 p. A27-8 | Pore scale is chaotic, Darcy-scale isotropic flow is not | Stokes flow | Abstract plus secondary |
| 11 | Pore-scale chaotic advection with no-slip walls gives anomalous longitudinal transport. | Lester et al. 2014, PRE 90, 063012 | Yes, 10.1103/PhysRevE.90.063012 | Abstract | Context only | Pore-network model | Abstract |
| 12 | In 3D pore-scale media, mixing grows exponentially with travel. In 2D media, topological constraints limit it to algebraic growth. | Lester, Dentz & Le Borgne 2016, JFM 803 | Yes, 10.1017/jfm.2016.486 | Abstract | Exponential stretching = chaos; algebraic = constrained flow | Pore scale | Abstract |
| 13 | In chaotic flows, deformation is set by the Lyapunov exponent. In non-chaotic steady flows, deformation is *algebraic* and comes from intermittent shear events, as in steady 2D flow. | Lester, Dentz, Le Borgne & de Barros 2018, JFM 855 | Yes, 10.1017/jfm.2018.654 (arXiv 1602.05273) | Abstract | The non-chaotic picture predicts algebraic, not exponential, deformation | Random steady flow | Abstract |
| 14 | Steady 2D heterogeneous flows stretch anywhere from sublinearly to superlinearly. Intermittent shear causes this subexponential stretching. | Dentz et al. 2016, PRE 94 | Yes, 10.1103/PhysRevE.94.061102 | Abstract | Algebraic stretching in 2D Darcy flow | 2D steady | Abstract |
| 15 | "A range of stretching regimes" depending on σ²_lnK; lamellar mixing model. | Le Borgne, Dentz & Villermaux 2013, PRL 110 | Yes, 10.1103/PhysRevLett.110.204501 | Abstract | Context only. Whether they describe exponential-looking regimes is **not verified**. | 2D | Abstract only |
| 16 | Lamella deformation is well modelled by a Langevin equation with multiplicative noise. | Le Borgne et al. 2015, JFM 770 | Yes, 10.1017/jfm.2015.117 | Abstract | Context only. Exact stretching law not verified. | 2D | Abstract only |
| 17 | For isotropic Darcy flow the principal strains are bounded ratios of local quantities, F₂₂ = v(0)m(t)/[v(t)m(0)]. Their ensemble mean is 1 and the Lyapunov exponent is exactly 0. | Lester et al. 2022, JFM 945 A18 | Yes, 10.1017/jfm.2022.556 | Eqs. 4.23–4.25, p. A18-14 | Transverse stretching is bounded but can be large over finite times | Smooth, stagnation-free, stationary flow | Full text read (Cambridge open-access PDF) |
| 18 | Shear deformation grows algebraically, \|F₁₂\| ~ t^r with 0<r<2. Longitudinal deformation Λ_l ~ t^r; transverse deformation Λ_t ~ 1. | Lester et al. 2022 | Same as 17 | Eq. 6.12, p. A18-23; Eq. 6.19, p. A18-24 | Up to ballistic (~t²) growth, which can look exponential over a short window (my inference) | CTRW model | Full text |
| 19 | 2D shadowing: there is a rigorous procedure showing a true orbit stays close to a noisy numerical orbit of a chaotic map for a long time. | Hammel, Yorke & Grebogi 1987 (J. Complexity 3) and 1988 (Bull. AMS 19) | Yes, 10.1016/0885-064X(87)90024-0 and 10.1090/S0273-0979-1988-15701-1 | 1988 abstract only (in French, OpenAlex); 1987 bibliographic only | Diverging numerical trajectories can still represent true trajectories, so ensemble statistics survive | Maps that are close to hyperbolic | 1988 abstract; 1987 bibliographic only |
| 20 | Scaling laws for how far and how long numerical chaotic solutions stay valid (shadowing distance and time). | Sauer, Grebogi & Yorke 1997, PRL 79 | Yes, 10.1103/PhysRevLett.79.59 | Abstract | Same as 19 | — | Abstract |
| 21 | Individual trajectories are sensitive to initial conditions, so single trajectories are unreliable. A robust skeleton of Lagrangian coherent structures (LCS) still shapes transport. | Haller 2015, Annu. Rev. Fluid Mech. 47 | Yes, 10.1146/annurev-fluid-010313-141322 | Abstract | Structural and statistical quantities are robust even when trajectories are not | — | Abstract |
| 22 | In integrable or periodic systems, global integration error grows linearly for symplectic or symmetric methods and quadratically for general methods. | Calvo & Hairer 1995, Appl. Numer. Math. 18 | Yes, 10.1016/0168-9274(95)00046-W | Abstract | In a non-chaotic flow, error growth should be polynomial, not exponential | Hamiltonian/integrable systems; Darcy advection is only an analogue | Abstract summary (UNIGE archive) |
| 23 | Same linear-vs-general error growth result for periodic orbits, with variable step sizes. | Cano & Sanz-Serna 1997, SIAM J. Numer. Anal. 34 | Yes, 10.1137/S0036142995281152 | Abstract | Same as 22 | Same as 22 | Abstract |
| 24 | First-order Euler tracking on linearly interpolated velocity (~10 steps per cell) gives a finite asymptotic α_T ∝ σ⁴_Y; flow lines expand "within a widening cone". | Beaudoin & de Dreuzy 2013, WRR 49 (local PDF) | Yes, 10.1002/wrcr.20206 | p. 2490 (method); p. 2493 (α_T ∝ σ⁴_Y); p. 2494 ("widening cone") | Opposing result. Lester 2021 (p. A27-26) attributes it to numerics. | — | Full text (local PDF) |
| 25 | Finite-time FTLE ridges coincide with stable and unstable manifolds. | Shadden, Lekien & Marsden 2005, Physica D 212 | Yes, 10.1016/j.physd.2005.10.007 | — | Would support the "finite-time separation near saddles" argument | — | **Bibliographic only. Do not cite for content.** |
| 26 | Smooth isotropic K gives zero asymptotic α_T; conventional tracking produces spurious α_T. Plus the claims in section 2 below. | Lester et al. 2021 and 2023 (local PDFs) | Yes | See section 2 | Core claims | See section 2 | Full text |
| 27 | Anisotropic K gives braiding, chaos and finite α_T, linked to the Lyapunov exponent. Isotropic K gives λ ≈ 0. Even 1e-14 RK tracking on a spline field produces spurious α_T. | Lester, Metcalfe, Trefry & Dentz 2025, JFM 1018 A35 | Yes, 10.1017/jfm.2025.10551 (arXiv 2412.05419v1) | See section 2c | Most directly relevant recent paper | — | Full text of arXiv v1 (JFM pagination not checked) |

#### 2. The two local PDFs and related full texts

**(a) `Lester-2023-WRR.pdf`:** Lester, D. R., Dentz, M., Singh, P., & Bandopadhyay, A. (2023). Under what conditions does transverse macrodispersion exist in groundwater flow? *Water Resources Research*, 59(3), e2022WR033059. https://doi.org/10.1029/2022WR033059 (open access, CC BY-NC). Page numbers are "x of 24".

- **p. 1:** "We prove that the transverse macrodispersion does not exist when the hydraulic conductivity field is smooth and locally isotropic." Conventional numerical methods do not preserve this constraint.
- **p. 5:**
  - Eq. 10 gives h = k∇φ·(∇k×∇φ) = 0.
  - Eq. 11 is the stagnation-free condition v ≠ 0.
  - Together these mean there are no knots or braids.
  - Eqs. 12–13 introduce the dual streamfunctions, v = ∇ψ₁×∇ψ₂.
- **p. 7:** Streamline spacing "can only fluctuate about a mean value… but cannot continually diverge without bound." Flows that are *not* helicity-free: locally anisotropic K, non-smooth K, or flows with stagnation points.
- **pp. 8–9, Section 4:** Proof. D^m₂₂ reduces to a surface integral ~L²/L³, which tends to 0 (Eqs. 25–27).
- **p. 10, Table 1:** Test case uses σ²_lnK = 4, ℓ = 1/16, a 256³ grid, cubic splines and a periodic domain.
- **p. 11, Section 5.2:** Interpolating gridded k and φ and taking the gradient analytically keeps the velocity helicity-free, "even if… the resultant velocity is not strictly divergence-free." That is the MacroFlow3D construction. Divergence-free cubic interpolation of velocity components does *not* impose helicity-free.
- **p. 13, Section 5.3:**
  - Adaptive RK4 on an exact helicity-free field, tolerance 10⁻⁴ to 10⁻⁸.
  - Trajectory errors have zero mean and finite variance; the variance grows roughly as √tol (Fig. 4b, p. 14).
  - Eq. 36: D^m = σ²_x/(2⟨τ_Ω⟩), i.e. the error is misread as dispersion, assuming error variance grows linearly in time.
  - The error "decays to zero as the tolerance tol limits toward zero," so rigorous convergence studies are needed.
- **pp. 13–14, Section 5.4, Eqs. 37–38:** A "pseudo-symplectic" tracker that integrates in 1D along fixed (ψ₁, ψ₂) gives exactly zero α_T.
- **p. 18:** For anisotropic media the transverse deformation grows exponentially, but this "does not persist" because it comes from a first-order truncation.
- **p. 20:** The advection–diffusion equation is singular as diffusivity → 0 only for chaotic flows (persistent exponential stretching), not for helicity-free flows. Beaudoin & de Dreuzy's result is attributed to their tracking method.
- **p. 21:** Tracking errors "mimics Brownian motion." Pollock error scales with Δ², RK error with tol. Apparent α_T goes to 0 as accuracy increases.
- **p. 22:** The findings apply to "both the pre-asymptotic and asymptotic regimes."
- **Citation pitfalls:** this paper's reference list gives Lester 2021 the wrong title ("The kinematics…") and misprints the Sposito 2001 DOI.

**(b) `Lester_2021.pdf`:** Lester, D. R., Dentz, M., Bandopadhyay, A., & Le Borgne, T. (2021). The Lagrangian kinematics of three-dimensional Darcy flow. *J. Fluid Mech.*, 918, A27. https://doi.org/10.1017/jfm.2021.362

- **A27-1:** Helicity is identically zero, so the flow is "complex lamellar" and fully integrable, with two constants of motion that "preclude chaotic advection."
- **A27-3:** Assumptions: isotropic, finite, smooth K; constant porosity; no sources or sinks; away from boundaries.
- **A27-7:** Integrable flow means fluid-element length grows algebraically, ℓ(t) ~ t^r, versus exp(λt) for chaotic flow.
- **A27-8:** Helicity-free is necessary and sufficient for a streamfunction pair. Pore-scale Stokes flow and tensorial-K Darcy flow are non-integrable.
- **A27-15:** "There is no known numerical method to enforce particles remain on streamsurfaces in steady 3-D flows." Spurious numerical dispersion persists even with the potential-based velocity v_φ. Without enforcing the constraint, "the only contribution to the measured transverse macrodispersion is numerical error."
- **A27-18, Eqs. 4.12–4.14:** The transverse shear component ε₂₃ = 0, which is equivalent to helicity-free. Deformation becomes two superposed 2D shear flows, with algebraic growth "from sublinear to ballistic."
- **A27-22, Eq. 5.25:** Streamline spacing fluctuations are stationary; streamlines "neither diverge nor converge."
- **A27-24 to 25:**
  - Eq. 5.43 gives σ²_c ≈ σ²_c,0 + σ²_ψ(1 − e^(−x₁/ζ)), i.e. **finite, pre-asymptotic transverse spreading that saturates**.
  - This rests on an *assumed* first-order decay of the autocorrelation (Eq. 5.41).
  - Eq. 5.46: D_T,∞ = 0.
- **A27-26:** Beaudoin & de Dreuzy's "widening cone" is inconsistent with the constraint. The Chiogna fields are not smooth.

**(c) Related full texts I also read:**
- **Lester 2025 (arXiv v1):**
  - Section 2.2, Eqs. 2.5–2.6: isotropic flow is non-chaotic and integrable.
  - Section 3.3, Eq. 3.8: λ∞ = ⟨λ_σ⟩/Pe_T with ⟨λ_σ⟩ ≈ 0.8529 (1D braid model; Eq. 3.9 generalises it).
  - Section 4.1, Eqs. 4.7–4.8: D_T and λ∞ rise from zero for any anisotropy δ > 0.
  - Section 4.3: for isotropic flow λ is "effectively zero" (|⟨ε′ᵢᵢ⟩| < 10⁻⁴). Caveat: this was computed with streamfunction-constrained tracking, so it is enforced by construction.
  - App. D: 5th-order spline velocity tracked with Cash–Karp RK at 10⁻¹⁴ "has been shown… to introduce spurious transverse dispersion for the isotropic… flow."
  - App. F: stretching without folding gives spurious exponential dispersion; folding makes it sub-exponential.
- **Lester 2022:** claims 17–18 in the table.

#### 3. W8 answer

**(a) Is it a contradiction?** *Established:* smooth, stagnation-free isotropic flow has zero infinite-time Lyapunov exponent. Its transverse stretch F₂₂ is a bounded ratio of local quantities, and its shear grows algebraically, up to ~t² (Lester 2022, Eqs. 4.23–4.25 and 6.12; Lester 2021 p. A27-7). So *persistent* exponential separation in the exact flow is excluded. What is not excluded is numerical drift across streamsurfaces. Lester 2023 (Section 5.3) and Lester 2025 (App. D) show tolerance-controlled RK on helicity-free fields produces it. *Inference:* W8 is either (i) finite-time growth, from large bounded F₂₂ swings in low-velocity zones at σ² = 4 or superlinear shear that looks exponential on a semilog plot, or (ii) a sign that the numerical flow (spline plus integrator) is not exactly integrable. Note that Lester 2023 saw *Brownian-like* error growth, not exponential, so W8 is not explained by that work.

**(b) Does it threaten α_T?** *Established:*
- Error that acts as zero-mean random kicks adds variance that shrinks with tolerance (Lester 2023, Eq. 36 and Fig. 4b).
- Shadowing results (Hammel 1988; Sauer 1997) and the LCS literature (Haller 2015) show that sensitive individual trajectories do not by themselves invalidate ensemble statistics.
- Even genuinely chaotic flows give *linear* growth of plume variance (Lester 2025, Eq. 3.8).

*Inference:* pass-to-pass separation is harmless if it acts like a perturbation of the starting point. It biases α_T upward if it acts like cumulative drift across streamsurfaces. The h/8 vs h/16 agreement only bounds the step-cap error. It does not test the tolerance, the velocity representation (Lester 2023, Section 5.2), or a bias shared by both passes. Also, under Lester's theory the true finite-distance α_T should *decrease* with d (Lester 2021, Eq. 5.43). An α_T that stays flat over d = 25–94λ would be a warning sign.

**(c) Diagnostics:**
1. Sweep the tolerance over 3 or more decades with the step cap fixed. Measure *transverse* separation on control planes, reporting median and percentiles, and check whether δ ≈ tol^p·G(d) with G exponential or algebraic.
2. Integrate dF/dt = ∇v_s·F along paths using the analytic spline gradient. In the Protean frame, check that ⟨ε′₂₂⟩ → 0, ln F₂₂ stays bounded and F₁₂ grows algebraically. Also monitor v·∇×v_s and ∇·v_s along the paths.
3. Compare FTLE and separation across velocity representations (k∇Φ versus separately interpolated components).
4. Check whether growth bursts coincide with minima of |v|.
5. Plot α_T(d) against tolerance and against d. Spurious α_T scales with tolerance and does not decay with d. As a rough cross-check from Eq. 3.8: α_T/ℓ ~ λ_apparent/0.85.

#### 4. BibTeX (verified DOIs)

(BibTeX entries: `references.bib`.)

#### 5. NOT VERIFIED (do not cite for these points)

- **Le Borgne 2013 / 2015 and Dentz 2016:** any specific stretching exponents, or "exponential-looking" regimes at high σ². The full texts were blocked by HAL's bot check.
- **Shadden, Lekien & Marsden 2005:** content (FTLE ridges equal stable/unstable manifolds in steady flows). DOI only.
- **Hammel et al. 1987/1988:** specific iteration counts or tolerances.
- **Sposito 1997** (Int. J. Eng. Sci. 35:197) and **Sposito 1998** (JFM 363): DOIs not checked against Crossref; content not read.
- **Item 7** (a paper reporting finite-time exponential separation in steady helicity-free 3D Darcy flow and how it decays): **none found**. The "finite-time / near-stagnation" explanation in section 3 is my inference.
- **Lester & Dentz 2024**, "Equating topological entropy and Lyapunov exponent" (PRFluids, listed as submitted): not checked.
- **Lester 2025, published JFM version:** page and equation numbers. I used arXiv v1, and author order differs between arXiv v1 and Crossref; the BibTeX follows Crossref.

I did not read the Le Borgne/Dentz full texts, Shadden 2005, or the JFM pagination of Lester 2025. The "zero α_T" position comes from one group and is disputed by Beaudoin & de Dreuzy 2013, Attinger 2004 and Janković 2009. The prediction that finite-distance α_T decays with d (Eq. 5.43) depends on an assumed exponential autocorrelation.

Sources: [Sposito 2001 abstract (repository mirror)](https://repository.geologyscience.ru/handle/123456789/24808?show=full) · [Bakker & Hemker 2004 (VU repository)](https://research.vu.nl/en/publications/analytic-solutions-for-groundwater-whirls-in-box-shaped-layered-a/) · [Chiogna 2014 (FAU CRIS)](https://cris.fau.de/publications/321008779) · [Lester 2025, arXiv 2412.05419](https://arxiv.org/abs/2412.05419) · [Lester 2018, arXiv 1602.05273](https://arxiv.org/abs/1602.05273) · [Lester 2022 open-access PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/8BB773D9BF670EA7176A3F640E657A43/S0022112022005560a.pdf/div-class-title-fluid-deformation-in-isotropic-darcy-flow-div.pdf) · [Calvo & Hairer 1995 (UNIGE archive)](https://archive-ouverte.unige.ch/unige:12482) · Crossref and OpenAlex APIs for DOI metadata and abstracts.

## 4. Control-plane (travel-time) estimands against spatial moments

The main relation holds. alpha_LP = (U²/2)·d var(t)/dd is exact for the 1D Fickian first-passage case, and at first order it equals the spatial alpha_L at t = d/U. But I could not find the equation printed in any paper I could actually read. Wiley/AGU, Elsevier, Royal Society and the Roma Tre repository all blocked full-text access (Cloudflare 403). So for most classic papers, what I checked is the abstract, not equations or pages.

One finding affects the design directly. B&dD's fitted asymptote at sigma²=4 is alpha_LAF = 12.7 λ (taken from `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md`, not re-read by me). That means the plane window d ∈ [25, 94] λ is only 2 to 7 dispersivities long. The "large x/α" condition is not met at sigma²=4.

#### 1. Claim table

DOI check legend: Crossref = Crossref API; OpenAlex = OpenAlex API.

| # | Claim | Reference | DOI check | Location | Supports | Regime | What I verified |
|---|---|---|---|---|---|---|---|
| C1 | Arrival-time variance grows quadratically near the source and linearly far away ("similar to a diffusion process"). The position analysis gives resident concentration, the arrival-time analysis gives flux concentration, and the two agree at low σ² or at distances large relative to the correlation length. | Shapiro & Cvetkovic 1988 | Yes, Crossref | Abstract | Items 1, 2 | Stochastic, perturbative | Abstract (Crossref) |
| C2 | Var τ = 2Dd/U³ = 2α_L d/U² for first passage in 1D advection-dispersion | Kreft & Zuber 1978 (cited by S&C 1988 and by Dagan 2017) | Yes, OpenAlex | Not seen | Item 1 | 1D Fickian | Content not seen. I checked the maths myself numerically (§2). |
| C3 | Control-plane framework: a particle crosses the plane at (η, ζ) at travel time τ. For steady flow without pore-scale dispersion, η, ζ and τ become independent, so the pdf of (η, ζ) can be derived without τ. | Dagan, Cvetkovic & Shapiro 1992 | Yes, Crossref | Abstract | Items 1, 3 | Steady flow, advective | Abstract only |
| C4 | A source of about 20 integral scales is "almost ergodic" for cumulative mass flux | Cvetkovic, Shapiro & Dagan 1992 | Yes, Crossref | Abstract | Items 5, 6 | Small perturbation (2D/3D) | Abstract only |
| C5 | Travel-time pdf is proportional to the mass that has crossed the surface; the travel-time and concentration approaches are compared, including ergodicity | Dagan & Nguyen 1989 | Yes, OpenAlex | Abstract | Items 1, 2 | Fickian, inert | Abstract (TAU repository page, paraphrased) |
| C6 | Travel-time moments depend on Lagrangian velocity "as a function of space rather than time". First-order results hold up to σ² ≈ 1. Lagrangian integral scale grows with σ² (preferential flow). | Cvetkovic, Cheng & Wen 1996 | Yes, Crossref | Abstract | Item 2 | 2D | Abstract |
| C7 | In 3D, uniform-resident injection makes the mean and variance of arrival time grow nonlinearly with distance. Flux injection gives a linear mean and a lower variance that becomes linear sooner. | Demmy, Berglund & Graham 1999 | Yes, Crossref | Abstract | Item 4 | 3D, stochastic-advective | Abstract |
| C8 | Injection mode changes breakthrough curves and their moments over hundreds of integral scales. At σ²=8 and 100 I, resident-mode dispersivity is about 10× flux-mode. | Janković & Fiori 2010 | Yes, OpenAlex | Abstract | Item 4 | 3D, σ² up to 8 | Abstract (Buffalo repository, paraphrased plus a short quote) |
| C9 | With flux-proportional injection, an "equivalent macrodispersivity is defined in terms of the traveltime variance" at control planes (x ≤ 121 I, σ² = 2, 4, 8). Results depend strongly on 0.995 vs 0.999 mass cutoffs, and the tail is non-ergodic. | Janković, Fiori & Dagan 2006, part 1 | Yes, Crossref | Abstract | Items 1, 2, 5 | 3D inclusion model, not multi-Gaussian | Abstract. The defining equation was not seen. |
| C10 | Breakthrough curve equals the travel-time pdf. Dispersivity is "very large and independent of x". The plume becomes Gaussian only at extremely large x/I, and finite plumes need a cutoff. | Fiori, Janković & Dagan 2006, part 2 | Yes, Crossref | Abstract | Item 2 | Self-consistent model | Abstract |
| C11 | First-order alpha_L matches up to σ² ≈ 4 through cancellation of errors, but the time to reach the asymptote grows with σ² | Fiori, Janković & Dagan 2003, part 2 (also Dagan et al. 2003, part 1) | Yes, Crossref | Abstract | Item 2 | Spatial moments | Abstract |
| C12 | Travel-time framework in which Lagrangian variables depend on space. Eulerian and Lagrangian velocity pdfs diverge as σ² grows. Transverse displacement is non-normal near the source and close to normal after 20 I_Y. Travel-time pdf develops a power-law tail. Slowness is linked to travel-time variance and to longitudinal dispersivity. | Gotovac, Cvetkovic & Andričević 2009 | Yes, Crossref | Abstract | Items 2, 3 | 2D, σ² up to 8 | Abstract |
| C13 | s-Lagrangian (equidistant) velocity pdf = flux-weighted t-Lagrangian pdf (eq. 9) = v·p_e(v)/⟨v_e⟩ (eq. 10). Uniform injection gives p0 = p_e, so t-Lagrangian is stationary (eq. 16). Flux injection gives p0 = p_s, so s-Lagrangian is stationary (eq. 17). Ballistic D(t) is set by the initial velocity variance, which is larger under flux weighting (eq. 92). | Dentz, Kang, Comolli, Le Borgne & Lester 2016 | Yes, OpenAlex and arXiv 1608.02208 | Sec. II A–B; eqs. 9, 10, 16, 17, 92 | Items 2, 4 | Ergodic, steady, speed along streamlines | **Full text read (arXiv v1)** |
| C14 | Mean plume velocity: U_RR = U_FF = U for matching injection/detection modes. Mixed modes differ near the injection plane and tend to U far away. | Dagan 2017 | Yes, OpenAlex | Abstract | Items 2, 5 (choice of U) | Random K | Abstract (TAU repository, paraphrased plus a short quote) |
| C15 | Effective dispersion (per-realization centred moments) vs ensemble dispersion. The ensemble version "considerably overestimates" the effective one at finite times. | Dentz, Kinzelbach, Attinger & Kinzelbach 2000 | Yes, OpenAlex | Abstract | Item 6 | Second-order perturbation, spatial moments | Abstract |
| C16 | Relative (pair-based) dispersion for solute flux in time and transverse space approaches absolute dispersion for sources larger than about 10 heterogeneity scales | Andričević & Cvetkovic 1998 | Yes, OpenAlex | Abstract | Item 6 | Perturbative | Abstract |
| C17 | Mean residence time = volume / flow rate | Danckwerts 1953 | Yes, OpenAlex | Not seen | Item 2 (U = d/⟨τ⟩) | Closed flow system | Metadata only. Classical result, applied here by my inference. |
| C18 | Uniform-resident injection traps particles in low-K zones at the injection plane; flux-proportional injection reduces this. alpha_L(t) from velocity covariance matches the second-spatial-moment value. | Janković, Fiori & Dagan, IAHS Publ. 304 (2006), 174–179 | No DOI | pp. 176–178, Figs. 1–2 | Item 4 | 3D, σ² = 4 and 8 | **Full text read** |
| C19 | Continuous-time random walk (CTRW) upscaling of equidistant velocities predicts breakthrough curves and spatial moments; Eulerian velocity pdf, tortuosity and correlation length are the key inputs | Comolli, Hakoun & Dentz 2019 | Yes, Crossref | Abstract | Item 2 | Heterogeneous porous media, advective | Abstract |
| C20 | Bias of ratio estimators | Hartley & Ross 1954 | Yes, OpenAlex | Not seen | Item 6 | General statistics | Metadata only |

#### 2. Derivation for item 1

**(a) 1D Fickian model.** Take X(t) = x0 + Ut + √(2D)·W(t), with D = α_L·U.
- The first-passage time at distance d has the inverse Gaussian density f(τ) = d/√(4πDτ³)·exp[−(d−Uτ)²/(4Dτ)], i.e. IG(μ = d/U, λ = d²/2D).
- Its mean is d/U and its variance is μ³/λ = 2Dd/U³ = 2α_L·d/U².
- Therefore d Var τ/dd = 2α_L/U², so α_L = (U²/2)·d Var τ/dd. This is exact for every d > 0 when injection and detection are both flux (first passage).
- My numerical check (scipy, d=10, U=1, D=0.3): mean 10.000, variance 6.000 = 2Dd/U³.
- Mixed mode, for comparison: a flux concentration built from the infinite-medium resident solution gives mean t0(1 + 1/Pe) and variance t0²(2/Pe + 5/Pe²), with Pe = d/α_L. These coefficients are from my check, not taken from Kreft & Zuber. The extra terms (D/U² and 5D²/U⁴) do not depend on d, so they move the OLS intercept and leave the slope unchanged.

**(b) Heterogeneous medium, first order.** Write X1(t) = Ut + X1′(t).
- The plane is crossed when X1(τ) = d, so τ ≈ d/U − X1′(d/U)/U.
- Hence Var τ(d) ≈ X11(d/U)/U², and d Var τ/dd = U⁻³·dX11/dt at t = d/U.
- This gives **α_LP(d) = α_L(t = d/U)**, pre-asymptotically as well.
- It needs: small σ²; |X1′| ≪ Ut, which is the same as 2α_L/d ≪ 1, i.e. d ≫ α_L; u1 > 0 so each plane is crossed once; and an ergodic source.
- S&C 1988 (abstract) states the asymptotic linearity and that the two analyses are complementary. I did not see the explicit equation σ_τ² = X11/U² in any source, so treat it as derived, not cited.

With B&dD's alpha_LAF, the ratio d/α_L over the window is:

| σ² | d/α_L over [25, 94] λ |
|---|---|
| 0.25 | ~100–390 |
| 1 | ~20–74 |
| 2.25 | ~6–22 |
| 4 | ~2–7.4 |

At sigma²=4 the first-order condition is not met.

#### 3. Answers to items 2–6

**2. Do they agree, and which U?** Established: at low σ² or far from the source, flux and resident descriptions agree (C1). In highly heterogeneous 3D media, travel-time dispersivity is skewed, depends on the cutoff and is non-ergodic, and becomes Gaussian only very far away (C9, C10). For FF mode the velocity is U (C14).

My inferences:
- Travel time integrates slowness over distance, which is s-Lagrangian sampling, i.e. flux-weighted (C13). So E_s[1/v] = 1/⟨v_e⟩, which supports **U_r = Δd/Δ⟨τ⟩_r**: flux-weighted mean travel time, consistent with C14 and C17.
- Do not use the mean of d/τ_i or the harmonic mean of particle speeds; by Jensen's inequality both are biased.
- Var τ needs E_s[1/v²] = ⟨1/v⟩_e/⟨v_e⟩ to be finite. Under strong low-velocity tails, alpha_LP will run above alpha_L(t).

**3. Transverse.** At first order η(d) ≈ X2′(d/U), so Var η ≈ X22(d/U) and α_TP = α_T(d/U); this is my derivation. η does not depend on τ (C3), so it reflects streamline geometry only and is free of speed effects. X22(t) at a fixed time mixes particles sitting at different x, so the two can differ beyond first order. Gotovac 2009 (2D) finds η close to normal after about 20 I (C12). I found no literature source for alpha_TAP; it is your own estimator.

**4. Injection mode.** Flux injection makes the s-Lagrangian statistics stationary from the start (C13), gives a linear, smaller arrival-time variance (C7), and avoids resident-mode inflation of up to about 10× (C8, C18). Flux injection is the right choice for control-plane estimators.

**5. Comparing with B&dD.** The two coincide only if all of these hold:
- the first-order regime, or a Fickian, Gaussian regime in both windows;
- times mapped to distances by d = U·t with the same U (B&dD use the global |⟨v⟩|);
- matched injection (theirs is flux-weighted over 0.8 L_T², which matches);
- per-realization centring in both (B&dD eq. 4, which matches);
- transverse comparison only for alpha_TP in fixed axes.

Their window is [0.75 t_b, t_b], set by the first particle leaving. That is not the same distance range as 25–94 λ. Compare against their alpha_L(t) at t = d/U, not against their asymptote. Any particle that has not crossed a plane truncates Var τ, so report per-plane arrival completeness (cutoff sensitivity, C9).

**6. Mean of per-realization estimates vs ensemble.** I found no source that discusses U_r²-weighting for travel-time dispersivity. Related: effective vs ensemble (C15), relative vs absolute flux dispersion (C16), and the ergodic source size (C4). The source is about 20.5 λ ≈ 23 I wide, so close to ergodic.

My inference, assuming Var_r τ = 2α_r d/U_r² within each realization:
- Mean of (U_r²/2)·S_r recovers mean(α_r).
- (Ū²/2)·mean(S_r) = mean(α_r·Ū²/U_r²) ≈ ᾱ(1 + 3CV_U²) + covariance terms. This is Jensen bias.
- B&dD's time-domain normalisation gives about ᾱ + Cov(α_r, U_r)/Ū.
- Estimating U_r from the particles adds only O(1/N_p) ratio bias (C20).

Report CV(U_r) and both estimators.

#### 4. BibTeX (DOIs verified)

(BibTeX entries: `references.bib`.)

#### 5. NOT VERIFIED — do not cite for these specifics

- **Equations and pages not seen:** the explicit σ_τ² ≈ X11(x/U)/U² in S&C 1988, Dagan et al. 1992 or Cvetkovic et al. 1992; the moment table in Kreft & Zuber 1978; sections and pages in Dagan 1989 and Rubin 2003; the defining equation for travel-time-equivalent alpha_L in Janković 2006 or Fiori 2006. Cite those papers only at abstract level.
- **Comolli & Dentz 2017, Eur. Phys. J. B:** the DOI 10.1140/epjb/e2017-80370-9 did not resolve in OpenAlex.
- **Not identified or not checked:** "Selroos" (no specific paper found); Cox & Miller 1965 (seen only in S&C 1988's reference list); Cochran 1977; Kitanidis 1988 (the DOI I tried failed); Hakoun et al. 2019; the content of Fiori et al. 2017 (Adv. Water Resour. 110:349–359, DOI 10.1016/j.advwatres.2017.10.025 — metadata only; abstract elided by publisher); Dagan & Cvetkovic 1996, Proc. R. Soc. A (DOI 10.1098/rspa.1996.0016 exists, content blocked).
- **Wrong DOI to avoid:** one web-search summary gave S&C 1988 as "10.1029/88WR03157". That is wrong; the correct DOI is 10.1029/WR024i010p01711.

I skipped any full-text reading of the WRR/AWR papers because of the publisher blocks. The B&dD numbers came from the repo's notes file, not from re-reading the PDF. Everything marked "inference" above needs your own check before it goes into the protocol. I wrote one file outside the repo, the arXiv PDF of Dentz 2016, at `/tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/dl/dentz2016.pdf`. No project files were changed.

Sources:
- [arXiv 1608.02208 (Dentz et al. 2016)](https://arxiv.org/abs/1608.02208)
- [IAHS Publ. 304, Janković et al.](https://iahs.info/uploads/dms/13489.32-174-179-IAHS304-02-030.pdf)
- [Buffalo repository: Janković & Fiori 2010](https://researchconnect.buffalo.edu/en/publications/analysis-of-the-impact-of-injection-mode-in-transport-through-str/)
- [TAU repository: Dagan 2017](https://cris.tau.ac.il/en/publications/solute-plumes-mean-velocity-in-aquifer-transport-impact-of-inject/)
- [TAU repository: Dagan & Nguyen 1989](https://cris.tau.ac.il/en/publications/a-comparison-of-travel-time-and-concentration-approaches-to-model/)
- [Cirpka & Nowak 2004 PDF](https://www.iws.uni-stuttgart.de/publikationen/hydrosys/paper/Cirpka_Nowak_2004_Traveltime_Moments.pdf)
- Metadata and abstracts from the Crossref and OpenAlex APIs.

## 5. Numerical verification (GCI, MMS, MLMC)

**Bottom line:** the G5.4 E_h structure matches the standard GCI recipe and the MLMC bias estimator. There are four gaps against the standards:
- **No sign check.** Celik et al. 2008 Eq. 3c uses the sign of the ratio of differences to detect oscillation; G5.4's p formula drops it.
- **Fs = 1.25 needs a known formal order.** The standards allow 1.25 only when the observed order agrees with a known formal order. No such order is verified for the dispersivity ensemble mean.
- **Three levels give no asymptotic check.** With three levels, p fits the data exactly, so nothing is left over to test whether the grids are in the asymptotic range.
- **The finest-level branch uses only the last difference.** The MLMC algorithms instead take the maximum over several extrapolated differences.

Verification levels used below: **FT** = I read the full text; **Abs** = abstract only; **TOC** = table of contents only; **Sec** = secondary source; **Meta** = DOI and metadata only.

#### 1. Claim table

| Claim / prescription | Reference | DOI verified? | Location | Supports | Assumptions | Level |
|---|---|---|---|---|---|---|
| Fine-grid GCI = Fs·\|ε\|/(r^p−1). "Fs=3.0 for comparisons of two grids and Fs=1.25 for comparisons over three or more grids." The asymptotic check is GCI23 ≈ r^p·GCI12. | NASA NPARC "Examining Spatial (Grid) Convergence" | n/a (web page) | GCI section; formulas are images, the text is quoted | Both G5.4 branches; Fs values | Constant r; asymptotic range | FT (attributes the method to Roache) |
| Two solutions (order assumed): Roache recommends Fs=3. Three solutions with observed order matching formal order: Fs=1.25. "Far outside the asymptotic range … no choice for the factor of safety is sure to be conservative!" | Roy 2005, citing Roache 1998 | Yes, 10.1016/j.jcp.2004.10.036 | §3.4.3; also §3.5.6 ("unreliable … possibly random") | Conditions for 1.25 vs 3; failure mode | Agreement with formal order | FT |
| Observed order is degraded by coding mistakes, non-smooth solutions, and grids outside the asymptotic range. Round-off and iterative error should be below 0.01·DE. Observed order: Eq. 18 for constant r, Eq. 19 for non-constant r. | Roy 2005 | Yes | §2.3.2, §3.3.3 | Gates on the observed order | | FT |
| Bect et al.'s reading of Roache 1994: Fs=3 in general; Fs=1.25 only when p is known a priori **and** the asymptotic range has been checked. | Bect et al. 2021 (arXiv 2103.14559) | arXiv | §2.1, Eqs. 5–6 | Strict reading of the 1.25 condition | | Sec |
| Roache: "If, as I have recommended, we use Fs = 3 for cases of non-monotonic convergence …". He criticises Fs·ΔM because it implicitly assumes r^p≈2. If p is not trusted, he suggests assuming p=1 and keeping 1/(r^p−1). He calls applying 1/(r^p−1) to ΔM for oscillatory envelopes "intuitive" and supported by empirical tests. E1 = ε/(r^p−1) is cited to Roache 1998, p.112, Eqs. 5.4.2–5.4.3. | Roache, Lisbon III workshop comments (2008/09) | n/a (workshop note) | pp.3–5 | The fallback form; Fs=3 for non-monotone | Heuristic; Roache adds that "maybe even p=1 is too optimistic" | FT, not peer reviewed |
| p from Eq. 3a–3c: s = sign(ε32/ε21), q(p) handles non-uniform r, solved by fixed-point iteration. ε32/ε21<0 means oscillatory, and the percentage should be reported. If ε32 or ε21 ≈ 0 "the procedure does not work": add grids or report as such. Extrapolated value: Eq. 4. GCI_fine = 1.25·e_a/(r^p−1): Eq. 7. Use three grids with r>1.3. Agreement with formal order indicates asymptotic range; "the converse should not necessarily be taken as" a failure. For noisy data, consider the least-squares GCI. Profile error bars use a global p_ave. | Celik, Ghia, Roache, Freitas 2008 | Yes, 10.1115/1.2960953 | Steps 1–5, Eqs. 1–7, Table 1 | Finest-level branch; oscillation handling | Monotone convergence | FT (ASME-hosted policy text) |
| "The overall accuracy of a simulation is strongly affected by the implementation and order of the boundary conditions." | JFE Editorial Policy (in the same ASME PDF) | n/a | Item 7 | Boundary order caveat for MMS | | FT |
| Needs n_g ≥ 4 grids. 0.5 ≤ p ≤ 2 is acceptable for a 2nd-order method. If p > 2, δRE is "likely to cause too small error estimates" and is replaced by fits with p fixed at 1 or 2. If p < 0.5, six fits are tried (including δ12). Oscillatory data are treated as anomalous. Data range parameter Δφ = (φmax−φmin)/(n_g−1) (Eq. 19). Fs=1.25 if 0.5 ≤ p < 2.1 and σ < Δφ, else Fs=3. U = Fs·ε+σ+\|φi−φfit\| (Eq. 20); for σ ≥ Δφ, U = 3(σ/Δφ)(…) (Eq. 21). | Eça & Hoekstra 2014 | Yes, 10.1016/j.jcp.2014.01.006 | §2.4.1, §2.4.2, §3, App. A | Fallback logic; cap on p | ≥4 grids; 1st/2nd-order schemes | FT (MARIN-hosted copy) |
| ΔM = max\|φi−φj\| (Eq. 5). Oscillatory convergence: Uφ = 3ΔM. Other anomalous cases: min(3ΔM, 3(δ12+Us12)). "Small perturbations in the data may lead to significant changes in the estimated value of p." | Eça & Hoekstra 2008, Lisbon III workshop paper | n/a | §2, pp.4–5 | Closest precedent for 3·max\|diff\| | Superseded by the 2014 paper | FT |
| The GCI gives unreasonably small uncertainty when the observed order exceeds the theoretical order. | Xing & Stern 2010 | Yes, 10.1115/1.4001771 | Abstract | Failure mode: p > formal | | Abs |
| Contents: §5 Code verification, §6.3 MMS, §7.3 Statistical sampling error, §8.3 Richardson extrapolation, §8.4 Reliability of estimators, §8.6 Roache's GCI, §8.7 Mesh refinement issues. | Oberkampf & Roy 2010 | Yes, 10.1017/CBO9780511760396 | TOC, pp. 170–338 | Where to look | | TOC only |
| MMS: boundary conditions come from the manufactured solution. MMS "is predicated on having smooth solutions". It cannot detect mistakes that affect efficiency or robustness. It is code verification, kept separate from solution verification (§3). | Roy 2005 | Yes | §2.5 | Scope of MMS | | FT |
| MMS "can identify any coding mistake that affects the order-of-accuracy". Blind tests show which mistakes "can (and cannot)" be exposed. | Salari & Knupp 2000 | Yes, 10.2172/759450 | Abstract | Scope of MMS | | Abs |
| Bias test max(M⁻¹\|Ŷ_{L−1}\|, \|Ŷ_L\|) < (M−1)ε/√2 (Eq. 10, written for weak order 1), "designed to avoid possible problems due to a change in sign of the correction". Richardson extrapolation is offered as an alternative. "The weakness in the heuristic algorithm lies in the bias estimation." | Giles 2008 | Yes, 10.1287/opre.1070.0496 | §4.2 (Eq. 10), §5 | The MLMC bias estimator | | FT (author PDF) |
| E[P−P_L] = E[P_L−P_{L−1}]/(2^α−1) if level differences scale as 2^{−αℓ}. "For robustness" it takes the max over extrapolations from the last three differences. "This algorithm is heuristic; it is not guaranteed." | Giles 2015 | Yes, 10.1017/S096249291500001X | §3.1, Algorithm 1; Theorem 1 in §2.1 | Finest-level branch | Rate estimated on the fly from the same run | FT (author version; published page numbers not mapped) |
| The computable bias estimator needs the extra assumption that the decay of \|E[Q_M−Q]\| "is actually monotonic for M ≥ M̄". V[Q_ℓ] stops being constant at h ≈ λ; the coarsest level should have h₀ slightly smaller than λ. Exponential (1-norm) covariance, σ²=1: observed rate ≈1.75 in h for k_eff, ≈0.75 in h for the point flux. | Cliffe, Giles, Scheichl, Teckentrup 2011 | Yes, 10.1007/s00791-011-0160-x | §2.2, §3, §4.1–4.2 | The monotone-asymptotic assumption; rates depend on the quantity | FE discretisation, σ²=1 | FT (author PDF) |
| Weak rate α < 2t and variance rate β < 4t; t=1 gives α=2, β=4 (Prop. 3.4). Exponential covariance gives t < 1/2. Gaussian covariance gives a ∈ C^∞. Constants depend on L^q norms of a_max, 1/a_min and ‖a‖_{C^t}. Observed for a continuous Gaussian-covariance field: H¹ error linear, L² error quadratic. | Teckentrup, Scheichl, Giles, Ullmann 2013 | Yes, 10.1007/s00211-013-0546-4 | Prop. 3.4, §2, §4, §5 numerics (arXiv v1) | Smooth vs exponential covariance | FE; Fréchet-differentiable functionals | FT (preprint) |
| FE error bounds for log-normal fields with exponential covariance. | Charrier, Scheichl, Teckentrup 2013 | Yes, 10.1137/110853054 | Abstract | Exponential-covariance rates | | Abs |
| MLMC for two-phase flow and Buckley–Leverett transport. | Müller, Jenny, Meyer 2013 | Yes, 10.1016/j.jcp.2013.03.023 | n/a | Possibly relevant | | Meta |

#### 2. Answers

**(a) Is the structure standard?**
- *Standards:* yes.
  - The finer-level branch is the coarse-grid GCI, r^p·Fs·|ε|/(r^p−1) (NASA page).
  - The finest branch is Celik Eq. 7.
  - With Fs=1 and α=p, the finest branch is exactly the Giles 2015 §3.1 estimator.
- *My inference:* using a UCB and p_L is stricter than the standards, which use point values. Three problems remain:
  1. The |·| in the p formula loses the sign. A negative ratio means oscillation (Celik) and should force the fallback.
  2. Fs=1.25 assumes a known formal order (Roy §3.4.3; Bect's reading of Roache). No formal order is verified for FV flow plus particle-tracking dispersivities. Teckentrup's α=2 is for FE pressure functionals. The pre-registered range therefore needs an upper cap (Eça & Hoekstra use p < 2.1 for 2nd order).
  3. Three levels fit p exactly, so there is no residual to test with (Eça & Hoekstra require ≥4 grids).

**(b) Production at the finest level (σ²=4).**
- *Standards:* the GCI is normally reported for the finest grid. Its reliability rests on the asymptotic range, and outside it "no choice for the factor of safety is sure to be conservative" (Roy). Roache uses Fs=3 for non-monotone convergence. The MLMC algorithms are explicitly heuristic. Giles uses the max over the last 2–3 extrapolated differences because the correction can change sign. Cliffe needs monotone asymptotic decay.
- *Failure modes in the sources:*
  - Pre-asymptotic grids.
  - Oscillation.
  - Observed p above formal, which makes the estimate too small (Xing & Stern; Eça & Hoekstra).
  - A near-zero difference, where Celik's procedure does not work.
- *My inference:* σ²=4 likely makes the constants larger, because they depend on moments of a_max and 1/a_min. A Giles-style fix is E_h = 1.25·max(UCB|a32−a16|, 2^{−p}·UCB|a16−a8|)/(2^p−1), with p = min(p_L, cap).

**(c) Is 3·max|diff|/(r_min−1) a recognised fallback?**
- *Standards:* with r=2 it equals 3ΔM. That is exactly Eça & Hoekstra's 2008 workshop rule for oscillatory convergence. It is also what Roache suggested: Fs=3, assume p=1, keep 1/(r^p−1).
- *Caveats:* Roache calls the envelope scaling "intuitive" and empirical, and says p=1 may be too optimistic. Eça & Hoekstra 2014 replaced it with fixed-order fits plus a σ/Δφ inflation.
- *My view:* it is a recognised conservative heuristic, not a derived bound, and it does not cover oscillatory divergence.

**(d) E_int = 2·UCB|diff|.**
- *Arithmetic:* Richardson with r=2:
  - p=1: error of the h/8 solution = 2|diff|; error of the h/16 solution = |diff|.
  - p=3: 8/7·|diff| at h/8 and 1/7·|diff| at h/16.
  - Richardson p=3 with Fs=1.25 gives 1.43|diff| (h/8) or 0.18|diff| (h/16). So E_int is conservative by about 1.4× or 11×.
- *Standards:* if the ~3 was not observed on ≥3 step levels of the same quantity, it counts as an assumed order. Roache's two-grid rule then gives Fs=3: 3·2|diff| = 6|diff| (h/8) or 3|diff| (h/16) with p=1. E_int would fall below that.

#### 3. BibTeX
(BibTeX entries: `references.bib`.)

#### 4. NOT VERIFIED — do not cite as such
- **Oberkampf & Roy 2010, content inside sections.** This includes "Fs=1.25 if observed order is within 10% of formal, else 3", "p = min(max(0.5, p̂), p_f)", and anything about using the finest grid for production. The only source is a software README (pyGCS).
- **Roache 1994 and 1997 full texts:** page and equation locations of the Fs rules. They are confirmed only through Roy 2005, Bect et al. 2021 and the NASA page.
- **Roache 1998:** any location other than p.112, Eqs. 5.4.2–5.4.3, which comes from Roache's own 2009 note.
- **Celik 2008 author list:** Coleman and Raad as co-authors. The ASME-hosted copy lists four authors.
- **Roache 2002:** content. **Salari & Knupp:** the report number SAND2000-1444 (OSTI was unreachable) and the full text.
- **Müller et al. 2013:** content. **Charrier et al. 2013:** specific rates.
- **No source found for:**
  - a bootstrap confidence interval on the observed order;
  - transferring an order measured on a coarser or narrower configuration to production. The sources only frame rates as estimated from the same run, under a monotone-asymptotic assumption.
  - weak rates for particle-tracking dispersivities on FV grids.

The repo docs say the generator produces Gaussian covariance only; I used that to frame the rate discussion. I did not check the MacroFlow3D code itself.


Sources: [NASA NPARC](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html), [ASME JFE policy + Celik 2008](https://qa12.asme.org/wwwasmeorg/media/ResourceFiles/Shop/Journals/JFENumAccuracy.pdf), [Roache Lisbon III comments](https://maretec.tecnico.ulisboa.pt/~maretec.daemon/html_files/CFD_workshops/html_files_2008/Discussion/RoacheCommentsLisbonIII.pdf), [Eça & Hoekstra 2008 workshop paper](https://maretec.tecnico.ulisboa.pt/~maretec.daemon/html_files/CFD_workshops/html_files_2008/papers/paper_istmarin.pdf), [Eça & Hoekstra 2014 (MARIN copy)](https://www.marin.nl/api/marin/downloads/get-download/5675/0/860/block_files_utb8kwxda60a/VsLN4atolTpeBkST8w5DtHMRBwg=/2014-JCP-Vol262-Eca_Hoekstra-V&VProc.pdf), [Roy 2005 copy](https://eclass.upatras.gr/modules/document/file.php/MECH1137/Y%CE%A0%CE%9F%CE%9B%CE%9F%CE%93%CE%99%CE%A3%CE%A4%CE%99%CE%9A%CE%97_%CE%9C%CE%97%CE%A7_%CE%A1%CE%95%CE%A5%CE%A3%CE%A4/%CE%92%CE%BF%CE%B7%CE%B8%CE%B7%CE%BC%CE%B1%CF%84%CE%B1/Papers_4Verification_Validation/Roy_Verif_Valid.pdf), [Bect et al. arXiv 2103.14559](https://arxiv.org/pdf/2103.14559), [O&R TOC](https://assets.cambridge.org/97805211/13601/frontmatter/9780521113601_frontmatter.htm), [Giles 2015](https://people.maths.ox.ac.uk/gilesm/files/acta15.pdf), [Giles 2008](https://people.maths.ox.ac.uk/gilesm/files/OPRE_2008.pdf), [Cliffe et al. 2011](https://people.maths.ox.ac.uk/gilesm/files/cgst.pdf), [Teckentrup et al. arXiv](https://arxiv.org/pdf/1204.3476), [pyGCS README](https://github.com/tomrobin-teschner/pyGCS), [MARIN tools](https://www.marin.nl/en/research/free-resources/verification-and-validation/verification-tools).

Not checked: the full texts of Roache 1994, Roache 1997, Roache 1998 and Oberkampf & Roy 2010 (paywalled or book). The Fs=1.25 condition rests on what Roy and Bect say Roache wrote.

## 6. Statistics and design

Most references check out. Two citation errors need fixing:
- **Berger 1982:** the DOI you were given (`10.1080/00401706.1982.10487759`) points to a different paper (Lucas & Crosier, about CUSUM charts). The correct DOI is `10.1080/00401706.1982.10487790` (JSTOR alias `10.2307/1267823`).
- **Holm 1979:** it has no Crossref DOI. `10.2307/4615733` gives a 404 at Crossref. Cite JSTOR stable 4615733.

Verification levels used below:
- **T** – full text read and the passage found
- **A** – abstract read (Crossref, PubMed or the author's page)
- **M** – publisher or catalogue metadata only (Crossref, OpenAlex, Open Library)
- **S** – secondary source only

#### 1. Claim table

| Claim / prescription | Reference | DOI/ISBN verified? | Location | Supports | Assumptions | Level |
|---|---|---|---|---|---|---|
| Exact binomial confidence limits | Clopper & Pearson 1934, *Biometrika* 26(4):404–413 | 10.1093/biomet/26.4.404 ✓ | whole paper | failure-rate CIs | independent Bernoulli outcomes | M |
| Holm step-down controls the family-wise error rate | Holm 1979, *Scand. J. Statist.* 6(2):65–70 | JSTOR 4615733; no DOI | – | Holm over the 7 p-values | none on dependence | M + S |
| TOST gives the same decision as the shortest 1−2α CI; TOST beats the "power approach" | Schuirmann 1987, *J. Pharmacokin. Biopharm.* 15(6):657–680 | 10.1007/BF01068419 ✓ | abstract | the 4 TOST tests | – | A |
| An intersection-union test (IUT) built from level-α tests is itself level α | Berger 1982, *Technometrics* 24(4):295–300 | 10.1080/00401706.1982.10487790 ✓ | – | p = max(p_TP, p_TAP); TOST | conditions in the paper | M + S |
| IUT unifies bioequivalence testing. The ratio problem is "not harder" than the difference problem. Treating size-α tests as 1−2α confidence sets is a misconception; methods for proper 1−α sets are given | Berger & Hsu 1996, *Statist. Sci.* 11(4):283–319 | 10.1214/ss/1032280304 ✓ | abstract | ratio TOST, IUT | – | A |
| Sasabuchi test + IUT gives a size-α TOST for a **ratio of means on untransformed normal data**. Exact power and sample size for parallel and **2-period crossover** designs | Hauschke, Kieser, Diletti & Burke 1999, *Stat. Med.* 18(1):93–105 | 10.1002/(SICI)1097-0258(19990115)18:1<93::AID-SIM992>3.0.CO;2-8 ✓ | abstract | Fieller-type paired TOST and its sample size | normal (L, P) | A |
| Exact CI for a ratio of means from untransformed crossover data | Locke 1984, *J. Pharmacokin. Biopharm.* 12(6):649–655 | 10.1007/BF01059558 ✓ | abstract | paired ratio | model in the paper | A |
| Fieller interval: roots of a linear combination of normal estimates with a jointly estimated covariance | Fieller 1954, *JRSS-B* 16(2):175–185 | 10.1111/j.2517-6161.1954.tb00159.x ✓ | abstract | Fieller | normality | A |
| Delta-method intervals are forced to be symmetric; Fieller coverage is closer to nominal | Hirschberg & Lye 2010, *Am. Stat.* 64(3):234–241 | 10.1198/tast.2010.08130 ✓ | – | why the delta-method SE fails | – | M + S |
| BCa interval | Efron 1987, *JASA* 82:171–185 | 10.1080/01621459.1987.10478410 ✓ | – | BCa (reported only) | – | M |
| Percentile-t and BCa are "most promising"; other methods give "serious errors in coverage" | Hall 1988, *Ann. Statist.* 16(3):927–953 | 10.1214/aos/1176350933 ✓ | abstract | bootstrap-t | smooth function of means | A |
| Percentile-t coverage error drops from O(n^-1/2) to O(n^-1) for **any** B, provided the nominal level is a multiple of 1/(B+1) | Hall 1986, *Ann. Statist.* 14(4):1453–1462 | 10.1214/aos/1176350169 ✓ | abstract | choice of B at α = 1/280 | continuous distribution | A |
| Survey of BCa, bootstrap-t, ABC and calibration; aims for an order-of-magnitude accuracy gain | DiCiccio & Efron 1996, *Statist. Sci.* 11(3):189–228 | 10.1214/ss/1032280214 ✓ | abstract | interval qualification | – | A |
| Empirical **monotone, invertible** transformation removes most skewness; works with a normal approximation or the bootstrap | Hall 1992, *JRSS-B* 54(1):221–228 | 10.1111/j.2517-6161.1992.tb01876.x ✓ | abstract | skewness fix | – | A |
| Modified t based on a Cornish–Fisher expansion | Johnson 1978, *JASA* 73(363):536–544 | 10.1080/01621459.1978.10480051 ✓ | – | skewness fix | finite moments | M + S |
| One-sided t is inaccurate at small n under skewness; Johnson's test is preferred; bootstrap tests are robust | Sutton 1993, *JASA* 88(423):802–810 | 10.1080/01621459.1993.10476345 ✓ | abstract (third-party index) | skewness fix | – | S |
| Relative tail error P(S_n/V_n ≥ x) = (1−Φ(x))(1 + O(1)(1+x)^{2+δ}/d_{n,δ}^{2+δ}), assuming only a finite (2+δ)th moment; applies to the studentized bootstrap | Jing, Shao & Wang 2003, *Ann. Probab.* 31(4):2167–2215 | 10.1214/aop/1068646382 ✓ | abstract | why small tails need larger n | mean zero at the boundary | A |
| BCa "tends to severely under-cover the mean"; bootstrap-t covers well but gives long, variable intervals | Owen 2025, arXiv:2508.10083 (preprint, not peer-reviewed) | – | abstract | keeping BCa as "reported only" | small n | A |
| Bootstrap textbook and monograph | Efron & Tibshirani 1993; Hall 1992 (book) | 10.1007/978-1-4899-4541-9 ✓; 10.1007/978-1-4612-4384-7 ✓ | chapters not checked | general | – | M |
| Internal pilot study: simulation shows a negligible α effect for two normal means | Wittes & Brittain 1990, *Stat. Med.* 9:65–71 (+ discussion) | 10.1002/sim.4780090113 ✓ | abstract | blinded sample-size re-estimation | normal data | A |
| Sample-size re-estimation without unblinding, normal outcome with unknown variance | Gould & Shih 1992, *Commun. Stat. Theory Methods* 21(10):2833–2853 | 10.1080/03610929208830947 ✓ | – | blinded re-estimation | – | M |
| Simple blinded variance estimators **do not affect** the type I error of the t-test; a permutation-test version keeps α for **any** blinded strategy | Kieser & Friede 2003, *Stat. Med.* 22(23):3571–3581 | 10.1002/sim.1585 ✓ | abstract | blinded re-estimation (sign/superiority side) | superiority t-test | A |
| **For non-inferiority/equivalence (ratio and difference), actual type I error exceeds nominal.** Usually negligible, but depends on N, pilot size and standardized margin; must be judged "case by case" | Friede & Kieser 2003, *Stat. Med.* 22(6):995–1007 | 10.1002/sim.1456 ✓ | abstract | **re-estimation × TOST** | normal data | A |
| Review of internal-pilot sample-size methods | Friede & Kieser 2006, *Biom. J.* 48(4):537–555 | 10.1002/bimj.200510238 ✓ | abstract | blinded re-estimation | – | A |
| Pre-specified adaptations on non-comparative data have "no effect or a limited effect" on type I error, **but** "potential for limited Type I error probability inflation in trials incorporating hypothesis tests of non-inferiority or equivalence … Sponsors should evaluate the extent of inflation" | FDA, *Adaptive Designs for Clinical Trials of Drugs and Biologics*, Nov 2019 | – | §IV, printed p. 10 | blinded re-estimation with TOST | full pre-specification | T |
| Conditioning on ancillary statistics / likelihood principle | Cox 1958, *Ann. Math. Statist.* 29(2):357–372; Cox & Hinkley 1974; Berger & Wolpert 1988 | 10.1214/aoms/1177706618 ✓; ISBN 0412124203 ✓; ISBN 0940600137 ✓ | text not fetched | fixed-n inference given n | n independent of outcomes | M |
| Clinical-trial monitoring | Proschan, Lan & Wittes 2006 | 10.1007/978-0-387-44970-8 ✓ | – | administrative looks | – | M |
| MCAR/MAR/MNAR and ignorability | Rubin 1976, *Biometrika* 63(3):581–592; Little & Rubin 2019 (3rd ed.) | 10.1093/biomet/63.3.581 ✓; 10.1002/9781119482260 ✓ | text not fetched | runtime-related dropout is not ignorable | – | M |
| With time budget t, the mean of the N(t) completed runs has bias o(1/t); including the run in progress (N(t)+1) gives bias O(1/t); exactly unbiased estimators exist | Glynn & Heidelberger 1990, *Oper. Res.* 38(5):801–814 | 10.1287/opre.38.5.801 ✓ | abstract | **cutoff bias** | renewal model | A |
| For parallel replications under a completion-time limit: bias expansions given. The unbiased estimator needs a **pre-selected** set of processors and may need to run past the limit | Glynn & Heidelberger 1991, *ACM TOMACS* 1(1):3–23 | 10.1145/102810.102811 ✓ | abstract (author's page) | **cutoff bias, prefix remedy** | – | A |
| Statistical properties of parallel discrete-event simulation | Heidelberger 1988, *SIAM J. Sci. Stat. Comput.* 9(6):1114–1132 | 10.1137/0909077 ✓ | – | – | – | M |
| Monte Carlo SE of coverage = √(C(1−C)/n_sim); n_sim = C(1−C)/MCSE² (worked example: 1900 runs); report MCSE | Morris, White & Crowther 2019, *Stat. Med.* 38(11):2074–2102 | 10.1002/sim.8086 ✓ | §5.2, Table 6; §5.3, Eq. 1; §6.2 | coverage qualification | – | T |
| Monte Carlo error is often larger than assumed; methods to choose the number of runs | Koehler, Brown & Haneuse 2009, *Am. Stat.* 63(2):155–162 | 10.1198/tast.2009.0030 ✓ | abstract | coverage qualification | – | A |
| Analysing a split-plot as a completely randomized design **inflates type I error for whole-plot factors** and type II error for subplot factors and their interactions. An unreplicated design leaves **0 df for whole-plot error**. Reset the hard-to-change factor between whole plots even when its level does not change | Jones & Nachtsheim 2009, *J. Qual. Technol.* 41(4):340–361 | 10.1080/00224065.2009.11917790 ✓ | pp. 344–345 | split-plot error strata | whole-plot errors i.i.d. N(0, σ_w²) | T |
| 2^(k−p) × 2^(q−r) split-plot designs; split-plot tutorial | Bisgaard 2000, *JQT* 32(1):39–56; Kowalski, Parker & Vining 2007, *Qual. Eng.* 19(1):1–15 | 10.1080/00224065.2000.11979970 ✓; 10.1080/08982110601057179 ✓ | – | split-plot | – | M |
| Resolution IV. With I = ABCD: A=BCD, B=ACD, C=ABD, D=ABC; two-factor interactions alias in pairs (AB=CD, AC=BD, AD=BC) | Montgomery, 10th ed. (Wiley 2019); Box, Hunter & Hunter 2005 (2nd ed.) | ISBN 9781119492443 (edition number from a retailer only); ISBN 9780471718130 ✓ | chapters not checked | 2^(4−1) design | – | M (alias algebra is elementary) |
| Common random numbers are **provably** beneficial only under monotonicity and continuity conditions | Glasserman & Yao 1992, *Manage. Sci.* 38(6):884–908 | 10.1287/mnsc.38.6.884 ✓ | abstract | pairing by seed | – | A |
| Simulation and Monte Carlo textbooks | Glasserman (Crossref gives 2003, usually cited 2004); Law 2015 (5th ed., Law sole author) | 10.1007/978-0-387-21617-1 ✓; ISBN 9780073401324 ✓ | – | common random numbers | – | M |
| Bioequivalence textbooks | Hauschke, Steinijans & Pigeot 2007; Chow & Liu 2008 (3rd ed.) | 10.1002/9780470094778 ✓; ISBN 9781584886686 ✓ | – | log-scale TOST practice | – | M |

#### 2. Short answers

**(i) Fixing one-sided coverage of the paired ratio TOST.**
1. **Fieller/Sasabuchi form (best fit).** For each margin θ, test the mean of D_r(θ) = L_r − θ·P_r with a one-sample test (Hauschke 1999, Locke 1984, Fieller 1954, Berger & Hsu 1996). The estimand stays the ratio of arithmetic means. The linearization in e_r goes away, and the test is exact if (L, P) is bivariate normal. Each side is then a plain one-sample mean test, so the skewness fixes below apply to it directly.
2. **Skewness correction on D_r(θ).** Bootstrap-t is second-order accurate (Hall 1988). Since 0.05/7/2 = 1/280, choose B so that B+1 is a multiple of 280, e.g. B = 27,999 (Hall 1986). Johnson's modified t or Hall's 1992 transformation are alternatives. None of these has a proven guarantee at α ≈ 3.6e-3, so each still has to pass your coverage simulation.
3. **Log scale (ratio of geometric means).** This changes the estimand to exp(E log L − E log P) and needs L, P > 0. Use it only if the scientific claim can be restated in those terms.
4. **Larger n.** By Jing–Shao–Wang (δ = 1), the relative tail error scales as (1+x)³ρ/√n. Holding that bound fixed when moving from α = 0.025 to 3.6e-3 takes about (3.69/2.96)⁶ ≈ 3.7 times the n. This is my own scaling estimate; the constant is unknown. It gets worse if the third moment is infinite.
5. **Check the "/2".** Under IUT, each one-sided TOST test runs at the full Holm level (Berger 1982, Schuirmann 1987, Hauschke 1999). If the /2 is there because of the TOST, it is not required and doubles the tail level you have to hit.

For requalification, Morris Eq. 1 gives n_sim ≈ 0.00446 × 0.9955 / (2e-4)² ≈ 111,000 runs per scenario for an MCSE of 2e-4 at the 1.25× threshold.

**(ii) Wall-clock cutoff.** Fixed-n inference given n is valid only if n does not depend on the outcomes (n ancillary). If runtime is correlated with the outcome, the completed set is missing-not-at-random (Rubin 1976) and the estimates are biased (Glynn & Heidelberger 1990, 1991). Even the "longest complete prefix" is chosen by runtime, so it is only approximately unbiased. Draining in-flight runs is not exact either: that is the N(t)+1 estimator, with bias O(1/t).

The exact fix is to pre-register the seed list 1..n_conf. Analyse confirmatorily only if all n_conf seeds complete; otherwise report descriptively. By an elementary bound, P(reject and complete) ≤ α. Choose n_conf so that completion is near-certain. Using "full N if it completes, otherwise the floor" is two looks, so split α (Bonferroni, or group-sequential spending since the sets are nested). Log runtime per seed and report its correlation with the outcome.

**(iii) Blinded sample-size re-estimation.** For superiority t-type tests, type I error is unaffected (Kieser & Friede 2003). For **equivalence tests (your TOST), slight inflation is documented** (Friede & Kieser 2003), and FDA (2019, p. 10) says it must be quantified. The upward-only rule and the 2× cap do not replace that simulation. The algorithm also has to be fully pre-specified, with no access to comparative results.

#### 3. BibTeX

(BibTeX entries: `references.bib`.)

#### 4. NOT VERIFIED — do not cite for these specifics

- **DOI `10.1080/00401706.1982.10487759` for Berger 1982:** wrong paper (Lucas & Crosier). Use `...10487790`.
- **`10.2307/4615733` as a DOI for Holm:** not registered at Crossref. Use the JSTOR stable id.
- **Bootstrap replicate counts:** "B = 1000 (Efron & Tibshirani)" and "B = 2000 (DiCiccio & Efron)" come from secondary sources only. Rely on Hall 1986 instead.
- **Book locations:** no chapter or page numbers were checked for any book (Efron & Tibshirani, Hall 1992 book, Montgomery, Box Hunter & Hunter, Law, Glasserman, Chow & Liu, Hauschke 2007, Proschan–Lan–Wittes, Cox & Hinkley, Berger & Wolpert, Little & Rubin).
- **Montgomery edition number:** the "10th" for ISBN 9781119492443 rests on a retailer listing only.
- **Johnson 1978 formula:** primary text not read.
- **Content beyond metadata:** Heidelberger 1988, Bisgaard 2000, Kowalski et al. 2007, Cox 1958, Rubin 1976.
- **Hirschberg & Lye coverage claim:** secondary sources only.
- **Proschan–Lan–Wittes on "administrative looks":** not checked.
- **No reference found that guarantees one-sided coverage at α ≈ 1e-3 for any bootstrap method under heavy skewness.** The orders of accuracy in the papers are asymptotic at a fixed α.

**My own reasoning, not quoted from the papers:** the D_r(θ) reformulation, the ≈3.7× n estimate, the P(reject and complete) ≤ α bound, and the n_sim ≈ 111,000 figure.

Sources: [FDA 2019 guidance](https://www.fda.gov/media/78495/download) · [Glynn & Heidelberger 1991 abstract](https://web.stanford.edu/~glynn/papers/1991/GH91a.html) · [Morris et al. 2019 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/) · [Jones & Nachtsheim 2009 (JMP-hosted copy)](https://www.jmp.com/content/dam/jmp/documents/en/white-papers/split-plot-designs-what-why-and-how.pdf) · [Owen 2025, arXiv:2508.10083](https://arxiv.org/abs/2508.10083) · [Sutton 1993 index record](https://academicnewsletter.sufe.edu.cn/info/446393) · [Holm 1979 PDF copy](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf) · [Hirschberg & Lye secondary (vLex)](https://au.vlex.com/vid/ratios-of-parameters-some-855678541) · [Montgomery 10e retailer listing](https://campusstore.torontomu.ca/product/order-online-design-and-analysis-of-experiments-tenth-edition-150-day-rental-enhanced-e-text) · Crossref, OpenAlex, PubMed E-utilities and Open Library APIs for metadata.

Not done: no project files were opened, and full texts behind paywalls (Berger & Hsu, Hall, Efron, the split-plot papers other than Jones & Nachtsheim) were not read. Before any of these are cited for a specific statement rather than as a general reference, someone with access should check the passage.

## 7. Full reading of the three local PDFs (Beaudoin & de Dreuzy 2013; Lester et al. 2023; Lester et al. 2021)

I read all three PDFs on every page, including appendices and reference lists. I also checked the key quotes against the PDF text.

- **Page references:** B&dD by journal page and paragraph [n]; L23 by page "n of 24"; L21 by page A27-n.
- **Figure values** are read from page images and are approximate (about ±10–20 % on log axes).
- **Labels:** "inference" means my reasoning, not what the paper says.

---

#### 1. Beaudoin & de Dreuzy (2013)

##### Extraction

**1. Citation (p. 2489).** Beaudoin, A., and J.-R. de Dreuzy (2013), Numerical assessment of 3-D macrodispersion in heterogeneous porous media, *Water Resour. Res.*, 49, 2489–2496, doi:10.1002/wrcr.20206. Received 1 Aug 2012, revised 31 Jan 2013, accepted 20 Mar 2013, published 22 May 2013.

**2. Model** (eq. 1 p. 2489; p. 2490 [6]; p. 2491 [7])

- **Covariance (eq. 1).** ⟨Y′(x)Y′(x′)⟩ = σ_Y² exp(−(|x−x′|/λ)²), with λ called the "correlation length".
  - This is the same convention as SF-43.
  - It matches eq. 2, D_LA = (√π/2)⟨v⟩λσ², and Table 2's "Gelhar" column (0.886σ²).
- **Variances.** σ_Y² ∈ {0.25, 1, 2.25, 4, 6.25, 7.56, 9} (Fig. 2). Table 1 has no σ² = 9 row and prints "6.52" where 6.25 is meant.
- **Domain width.** L_T/λ = 25.6 is stated only in the σ² = 6.25 captions (Fig. 1, Fig. 5). For the other σ² it is implied: the injection window is "around (20λ)² = 400λ²" with "25 particles per correlation square" (p. 2491).
- **Domain length.**
  - L_L = 102.4λ "is enough" for σ² ≤ 1 (p. 2492 [9]).
  - The longest domain is L_L/λ = 1638.4 (p. 2492 [10]).
  - L_L for σ² = 2.25 and 4 is not stated. Fig. 2 data run to about t ≈ 60–100 and t ≈ 200, so those domains were much longer than 102.4λ.
- **Resolution.** λ/Δ = 10 for all runs; λ/Δ = 20 was tested only at σ² = 6.25. The K field is generated by FFT.
- **Boundary conditions.** Fixed heads h₊, h₋ on the two faces normal to the flow direction L. Periodic on the four transverse faces, which "minimize the perturbation of the flow field generally reported under no-flow boundary conditions".
- **Flow solver.** Finite volume with harmonic averaging of K between cells, solved with HYPRE parallel algebraic multigrid. **No solver convergence criterion is reported.**
- **Velocity.** "Velocity v at the grid center is derived from the hydraulic head h by Darcy's law", then "obtained with a linear interpolation in each direction from its grid values [Pollock, 1988]".
  - It is unclear whether face fluxes (Pollock-style) or cell-centre vectors are interpolated.
  - The mean velocity is ‖⟨v⟩‖ = (K_eq/θ)|h₊−h₋|/L_L, with "K_eq the equivalent permeability [Matheron, 1967]". The paper does not say whether K_eq comes from Matheron's formula or is computed.
- **Tracking.** "Restricted to the sole advection process … first-order explicit": x(t+dt) = x(t) + v[x(t)]dt.
  - This is explicit Euler, not Pollock's semi-analytic scheme.
  - dt is set so a particle takes "on average 10 steps within each grid cell".
  - There is no local dispersion.
- **Injection.** A plane of 0.8 L_T1 × 0.8 L_T2, normal to L, "at least at five correlation lengths downstream from the side of the system". Particles are "injected proportionally to the flow through the plane" (flux-weighted).
- **Sample size.** N_p = 10⁴ particles per realization and N_R = 500 realizations, for every σ² ("kept the set of the reference parameters for all presented simulations").

**3. Estimators**

- **Eq. 4 (p. 2490).** α_k(t) = 1/(2λ‖⟨v⟩‖) · (1/N_R)Σ_i d/dt[(1/N_p)Σ_j x_k² − ((1/N_p)Σ_j x_k)²].
  - The variance is centred on each realization's own plume, then averaged over realizations (an "effective" dispersivity).
  - Axes are fixed (L, T1, T2); nothing is aligned with the realization's drift.
  - α_T1 = α_T2 was "systematically checked", and only one transverse component is reported.
  - Time t is in units of λ/‖⟨v⟩‖, so t is roughly the mean travel distance in λ.
- **Eq. 5 (p. 2492).** The reported value is the average of α_k(t) over [0.75 t_b, t_b].
  - t_b is "the time at which the first particle steps out of the simulation domain".
  - Later times are excluded because "the evaluation of plume statistics would be biased".
  - The σ² = 6.25 convergence test instead averaged over the last quarter of its plotted span, t ∈ [112.5, 150].
- **Eq. 6 (p. 2492).** f_L(t) = α_LAF(1 − exp(−t/t_LF))^a_LF, described as "heuristic" and meant for t > 1.
- **Table 1.**

  | σ² | α_LAF | t_LF | a_LF |
  |---|---|---|---|
  | 0.25 | 0.24 | 1.35 | 1 |
  | 1 | 1.27 | 2.21 | 0.80 |
  | 2.25 | 4.35 | 4.83 | 0.63 |
  | 4 | 12.7 | 20.5 | 0.49 |
  | "6.52" | 58.6 | 41.9 | 0.40 |
  | 7.56 | 131 | 208 | 0.39 |

  - No fit uncertainties are given.
  - My evaluation of eq. 6 at t = 25 and t = 94: flat at 0.24 and 1.27 for σ² ≤ 1; 4.33 → 4.35 for σ² = 2.25; 10.7 → 12.6 for σ² = 4.
- **Table 2 (p. 2493), "This study" columns.**
  - α_LA = 0.26 / 1.27 / 4.35 and α_TA = <10⁻³ / 0.01 / 0.044 at σ² = 0.25 / 1 / 2.25.
  - The caption says "Late-Time Averaging Method of Equation (6)", but eq. 6 is the fit; late-time averaging is eq. 5.
  - The α_LA values equal Table 1's α_LAF at σ² = 1 and 2.25, but not at 0.25 (0.26 vs 0.24).
  - The Dentz (0.007, 0.028) and Schwarze (0.009) α_T values come from first-order synthetic velocity fields, not a Darcy solve.
- **Eq. 7.** α_TAM ≈ α_TAM(1) σ_Y⁴.
  - p. 2493 [12] says it holds "for σ_Y² ≥ 1"; p. 2494 [18] says "for σ_Y² ≥ 2.25".
  - Fig. 4b points: ≈0.11 (σ² = 4), ≈0.35 (6.25), ≈0.5 (7.56), ≈0.7 (9).
  - Eq. 7 with α_TAM(1) = 0.01 gives 0.16 at σ² = 4, about 50 % above the plotted ≈0.11.
  - Fig. 1b (σ² = 6.25, linear axis) shows an α_T plateau of ≈0.27 from t ≈ 10 to 150.
- **Other relations.**
  - Eq. 8: 2D α_LAM ≈ 0.84σ² + 0.17σ⁴. Eq. 9: α_LAM ≈ exp(σ²/1.55) for σ² ≥ 2.25.
  - Fig. 3: t_LF grows roughly exponentially with σ².
  - Eq. 10 / Fig. 5: a transverse "tortuosity" τ(t), a running time-average of the size of each particle's (uncentred) transverse displacement. It is shown only for σ² = 6.25: it keeps growing in 3D (≈75 at t = 150) and levels off in 2D (≈25).
- **There is no control-plane, breakthrough-curve or travel-time analysis.** t_b is used only to cut off the time series.

**4. What they conclude, and their limitations**

- **Transverse dispersion.**
  - It is "significantly nonzero" for σ² ≥ 1 ([12]) and positive; α_TA(0.25) < 10⁻³.
  - It "increases quadratically in σ_Y²"; α_LAF/α_TAM is between 100 and 200 ([16]).
  - Braiding mechanisms "are negligible at low levels of heterogeneities (σ_Y² ≤ 1)" (p. 2494 [18]).
- **Convergence claims vs figures.**
  - They say α_T converges "much faster" than α_L at moderate-to-high σ², "around the advection time … at all levels of heterogeneity" ([9]).
  - They also say "at 10 times the advection time … [it has] converged" (pp. 2491–2492).
  - **Fig. 2b contradicts this.** At t = 10, α_T ≈ 0.015 (σ² = 1) and ≈ 0.08 (σ² = 2.25), against the tabulated 0.01 and 0.044. At σ² = 1 there is no clear plateau before the data end (t ≈ 30–40).
  - The curves rise to a peak at t ≈ 1–5 for σ² ≤ 4; for σ² ≥ 6.25 they start high and decline slightly.
- **Mechanism.** Braiding, and flow lines expanding "within a widening cone". Transverse spreading is "activated by the ergodic nature of the injection conditions" ([19]).
- **Uncertainty.**
  - The only convergence test is at σ² = 6.25 (N_p ×10, N_R 500 → 250, λ/Δ 10 → 20). Results changed by ≤2 % longitudinally and ≤4 % transversally.
  - "Smaller σ_Y² cases lead to similar conclusions" is stated without evidence.
  - No error bars anywhere.
  - α_LAF and α_LAM "may be slightly modified by larger simulations", and α_LAM is "an underestimate … for σ_Y² ≥ 6.25" ([15]).
  - The longitudinal asymptote is "never approached for σ_Y² = 9".
- **Boundaries.** They justify periodic transverse BCs, and the ≥5λ inlet offset "to avoid border effects on the velocity statistics". **There is no domain-width test**, only an injection-window test ("more than enough"). Outlet effects are handled only by stopping at t_b.
- **Numerical artefacts are never considered** as a possible source of α_T.

**5. Comparability with SF-43**

| Item | B&dD | SF-43 |
|---|---|---|
| Covariance | exp(−(r/λ)²) | same |
| Width | 25.6λ (stated only for σ² = 6.25) | 24λ; 12λ at σ² = 4 |
| Length | 102.4λ for σ² ≤ 1; longer, unstated, for 2.25 and 4 | 104λ |
| Source | 0.8 × width (≈20.5λ), at ≥5λ, flux-weighted | 0.8 × width (19.2λ; 9.6λ at σ² = 4), at 5λ, flux-weighted |
| Estimator | fixed-time spatial moments, centred per realization, fixed axes, one component, t ≤ t_b | control planes at d ∈ [25, 94]λ |
| Reported window | last quarter of [0, t_b]; σ² = 4 at t ≈ 150–200 (read from Fig. 2) | d ∈ [25, 94]λ |

- Only α_TP has even a loose counterpart (Table 2 / Fig. 4b). α_TAP and α_LP have none.
- Time t and distance d correspond only on average, and only once transport is Fickian.

##### Discrepancies with `/home/sesquerre/Projects/MacroFlow3D/docs/theory/beaudoin-de-dreuzy-2013-key-claims.md`

1. **"L_T = 25.6λ (Fig. 1 caption)".** The caption ties this to σ² = 6.25 (as does Fig. 5). For other σ² it is only implied by the "(20λ)²" injection window (p. 2491).
2. **"L_L up to 1638.4λ".** Correct, but it leaves out the stated L_L = 102.4λ for σ² ≤ 1 (p. 2492 [9]), which is almost SF-43's 104λ. The paper gives no length for σ² = 2.25 or 4.
3. **The tracking paragraph omits "Velocity v at the grid center"** (p. 2490). So face-flux (Pollock-style) interpolation is not established. The integrator is Euler, so L23's Pollock analysis (exact integration within cells) is not B&dD's scheme.
4. **"time normalized by λ/|⟨v⟩|"** omits that ‖⟨v⟩‖ is defined through K_eq "[Matheron, 1967]" (p. 2490). Any mismatch with the true mean velocity rescales both the time axis and α.
5. **Table 2 is cited only for α_TA.** The summary leaves out the α_LA column (0.26 / 1.27 / 4.35), the caption's wrong equation number (6 instead of 5), and the 0.24 vs 0.26 mismatch at σ² = 0.25.
6. **Eq. 7 "for σ_Y² ≥ 1".** p. 2494 [18] says "≥ 2.25". The summary's eq. 7 and its "≈0.11 at σ² = 4" differ by about 50 % (0.16 vs 0.11), and the summary does not point this out.
7. **"Rises, peaks near t ~ 1–3, plateau at t ~ 10–30" is imprecise.**
   - The σ² ≥ 6.25 curves do not rise.
   - σ² = 1 shows no plateau before the data end.
   - σ² = 2.25 levels off only around t ≈ 30–50 (Fig. 2b).
   - The paper's explicit "converged at t = 10" claim, which Fig. 2b contradicts, is not cited.
8. **The σ² = 6.25 refinement test is reported without its consequence.** Going from λ/Δ = 10 to 20 (which also halves dt, given the 10-steps-per-cell rule) left α_TAM within 4 % (Fig. 1b). An error that scales like L23's Pollock variance (∝ Δ²) should have shrunk about 4×. This bears directly on the summary's paragraph about numerical error sources.
9. **"Significantly non-zero" is stated for σ² ≥ 1** ([12]). The same paper calls braiding "negligible" for σ² ≤ 1 (p. 2494 [18]).
10. **Minor.** The DOI is missing from the summary. Fig. 2's caption ("Full … α_LAM, dashed … fit") looks swapped relative to the plot, which matters if anyone digitizes it.

##### Implications for SF-43 (≤300 words)

**What the paper states.** Its values are fixed-time moments, centred per realization, in fixed axes, for one transverse component. Time uses a K_eq-based velocity, and statistics stop at the first particle exit t_b. The σ² ≤ 1 runs used 102.4λ. α_T(t) at σ² = 1–2.25 is still falling at t ≈ 10–30. At σ² = 6.25, refining λ/Δ from 10 to 20 changed α_TAM by ≤4 %.

**Inference.**
- **Only an order-of-magnitude comparison of α_TP is defensible.** α_TAP and α_LP have no counterpart. Fixed-axis time moments also pick up drift tilt (∝ θ²α_L) and Dagan-type coupling; control planes avoid both.
- **Cheap exact comparison:** compute B&dD's estimator (centred per realization, per component, fixed time, t < t_b) on SF-43's own particles.
- **Candidate causes of the SF-39 gaps (W6), each checkable in SF-39 data:**
  - time windows extending past t_b in a 104λ domain (B&dD exclude these as "biased"; worse at high σ²);
  - the 9.6λ source at σ² = 4, against B&dD's ≈20λ ("more than enough"; smaller sources untested);
  - B&dD's own 8 % spread at σ² = 0.25 (0.24 vs 0.26);
  - the K_eq normalization;
  - eq. 6 parameters given without uncertainties.
  
  If SF-39's windows are at t ≳ 25, eq. 6 is essentially flat for σ² ≤ 2.25, so fit timing cannot explain those gaps.
- **Control-plane analogue of t_b:** var(t) at a plane is biased low if slow particles have not crossed before the run ends. This is critical at σ² = 4.
- **The refinement result argues against a pure Δ²-interpolation origin** for B&dD's α_T. A grid-refinement ladder alone may not separate physical from numerical α_TP; label-drift diagnostics are needed.
- **First-order theory** (the project's formula) gives α_T ≈ 3.5·10⁻⁴ to 2.5·10⁻⁵ σ²λ over d = 25–94λ, one to three orders below B&dD. An SF-43 α_TP between the two does not settle anything on its own.

---

#### 2. Lester, Dentz, Singh & Bandopadhyay (2023)

##### Extraction (item 6)

- **Citation (p. 1).** "Under What Conditions Does Transverse Macrodispersion Exist in Groundwater Flow?" *Water Resources Research*, 59, e2022WR033059, doi:10.1029/2022WR033059. Research article, received 21 Jun 2022, accepted 3 Mar 2023, CC BY-NC. Data and code: Figshare doi:10.25439/rmt.21314352 (p. 23).
- **Isotropic vs anisotropic.**
  - Smooth, locally isotropic, finite k gives zero helicity: h = k∇φ·(∇k×∇φ) = 0 (eq. 10, p. 5).
  - Zero helicity together with no stagnation points (eq. 11) means streamlines have no "knots or braided structures" (p. 5).
  - Two invariants follow (eq. 12), with the Euler representation v = ∇ψ₁×∇ψ₂ (eq. 13). "All steady 3D helicity-free flows admit the dual streamfunctions" (p. 6).
  - Streamline spacing "can only fluctuate about a mean value" (p. 7).
  - Locally anisotropic K, non-smooth k and stagnation points (source-driven flows) break this and can give non-zero α_T (p. 7). Statistically anisotropic but locally isotropic media stay constrained (pp. 7–8).
  - Non-smooth k is "a singular limit of a smooth but rapidly varying conductivity field", and is described in terms of "a discontinuity in the spatial conductivity gradient" (p. 7).
  - That pressure-driven flow has no stagnation points is simply asserted (p. 6, citing Bear 1972).
  - The synthetic Kraichnan-type field of Schwarze 2001 is "not explicitly helicity free" (pp. 4–5).
- **Asymptotic zero claim (§4, pp. 8–9; App. A).**
  - Streamfunctions are split as ψ_i = ψ̄_i + ψ̃_i, with means ψ̄₁ = v̄x₂ and ψ̄₂ = x₃.
  - Eq. 20 bounds the fluctuation gradient: |∇ψ̃_i| < |∇ψ̄_i|. It is presented as following from eq. 11, and then "we assume that the fluctuations ψ̃_i(x) are also bounded".
  - Eqs. 22–27: D^m₂₂ = lim (1/V)∮ n·f d²x, which goes to 0 if f is finite.
  - Further assumptions: statistical stationarity and ergodicity (p. 4), the purely advective limit (eq. 7), and averaging over V → ∞ (eq. 3).
- **Finite time and distance.**
  - The work is "relevant to … the non-Fickian pre-asymptotic regime", but "we shall focus mainly on the Fickian regime" (p. 2); also "relevant to both … regimes" (p. 22).
  - The paper studies "a steady, continuously injected solute plume". It explicitly excludes the Dagan (1994) mechanism, which gives "ln t growth of transverse variance when longitudinal dispersion is non-zero" but "does not lead to broadening … at fixed longitudinal locations" (p. 3).
  - With local dispersion, the deformation components F₂₂ and F₃₃ "fluctuate without growth" (p. 17; eq. 48, citing Lester et al. 2022).
  - **There is no decay law for finite-time transverse spreading in pure advection.**
- **Numerical artefacts (§5).**
  - **Test setup (Table 1, p. 10).** Periodic unit cube; σ²_ln k = 4; correlation length ℓ = 1/16; grid spacing Δ₀ = 1/256 (16 points per ℓ); "mean velocity v = 1".
    - **The covariance function is not stated**, only "log-Gaussian isotropic random field with correlation length ℓ".
    - The spline-interpolated streamfunction velocity is "not exactly helicity-free" (Fig. 2b, |h| up to ≈0.004).
    - The paper states that "integrable steady 3D periodic flows (such as zero helicity flows) admit periodic streamlines" (p. 10), and "As streamlines are periodic within Ω (due to periodicity of the streamfunctions)" (p. 13).
    - That eq. 14 reproduces the Darcy flow is supported only by citing L21 (p. 10).
  - **(A) Pollock reconstruction with exact integration** (pp. 11–13).
    - Divergence-free, but helicity is not captured and is violated at cell faces.
    - Position errors after one pass through the cell are Gaussian with zero mean. Their variance scales as (Δ/Δ₀)² (Fig. 3c, p. 12): σ²_x ≈ 10⁻³ (domain units) at Δ/Δ₀ = 1, rising to ≈0.1 at 8.
    - "Even highly resolved grids (… 16 grid points per correlation length) can impart significant spatial errors."
    - Eq. 36 converts this to a spurious coefficient, D = σ²_x/(2⟨τ_Ω⟩), "under the assumption that the variances … grow linearly with time".
    - Suggested fix: interpolate φ and k separately and form v = −k∇φ, which is helicity-free "even if … not strictly divergence-free" (p. 11).
  - **(B) 4th-order adaptive Runge-Kutta on the exact streamfunction velocity** (p. 13, Fig. 4). Errors are Gaussian with zero mean. Variance scales as √tol, from ≈4·10⁻⁷ (tol = 10⁻⁸) to ≈6·10⁻⁵ (tol = 10⁻⁴) per pass.
  - **Pseudo-symplectic tracker.** "Not implement[ed] … in the purely advective case as the results are trivial" (p. 15).
- **With local dispersion (§6).** Flux-weighted injection gives D^m = D₀(⟨m⟩² − 1); uniform injection gives D₀(⟨1/v⟩⟨v⟩⟨m⟩² − 1). Differences between the two protocols "persist for arbitrarily long times" (pp. 17–18). Eq. 68 holds only for t < t_d = ℓ²/D₀ (pp. 18–19).
- **Exponential separation and chaos (relevant to W8).**
  - Chaotic periodic flows have aperiodic streamlines that wander (p. 10).
  - In anisotropic media the blob deformation |Δ⊥| "grows exponentially", but this "does not persist" beyond the first-order approximation (p. 18).
  - Steady isotropic Darcy flow is non-chaotic, so the zero-diffusion limit is not singular (p. 20). Tracking errors "mimic Brownian motion" (p. 21).
  - **There is no analysis of how errors grow; linear variance growth is assumed** (eqs. 36, 69).
- **On B&dD.** Their results "may be attributed to the particle tracking methods" (pp. 8, 20). L23 does not reproduce B&dD's setup and reports no α_T value for pure advection.

##### Discrepancies with `/home/sesquerre/Projects/MacroFlow3D/docs/theory/lester-2023-key-claims.md`

1. **§4 transcribes eq. 20 as "|grad psi~_i| < |grad psi_i|".** On the page image the right-hand side is the *mean* gradient |∇ψ̄_i|; the text layer drops overbars here and in eq. 21. L21's eq. 5.6 is likewise bounded by C_i² ≡ ‖∇ψ̄_i‖² (A27-19). A bound by a constant is strictly stronger than v ≠ 0 and does not follow from it.
2. **§2 records the "necessary vs sufficient" inconsistency as "not resolved". The papers themselves resolve it (inference).**
   - L23's own test flow (Fig. 2b) is built from streamfunctions yet has non-zero helicity.
   - L21's Fig. 3 places the helicity-free class inside the integrable class, and A27-10 says the results hold for integrable flows "not just helicity-free flows".
   - So zero helicity is not *necessary* for a streamfunction pair. Both A27-8 ("necessary and sufficient") and L23 p. 11 ("necessary but not sufficient") get that direction wrong.
   - A minimal counterexample: v = (0, −sin x, 1) has invariants ψ₁ = x and ψ₂ = y + z sin x, but helicity h = −cos x.
3. **§9 says L23 pp. 7 and 22 "define smooth as 'not discontinuous'".** Neither page defines it. p. 7 puts non-smoothness at "a discontinuity in the spatial conductivity gradient". Read literally, that covers continuous fields whose gradient jumps, which matters for exponential-covariance fields and for cell-wise constant discretized K.
4. **The closure findings in §2 and §4 leave out supporting evidence from L23 itself.**
   - p. 10 asserts periodic streamlines for integrable periodic flows.
   - Its error metric (eq. 34) presupposes that streamlines close.
   - The claim that eq. 14 reproduces Darcy flow for its random σ² = 4 field rests on L21, which checked only a deterministic symmetric field, and only by angle.
5. **§5A gives "Fig. 3c, p. 13"; the figure is on p. 12.** The summary also does not mention three problems in the paper:
   - It calls Pollock "second order in space", but a variance ∝ Δ² means a first-order error.
   - The text says x₃ errors are larger, but Fig. 3c shows x₂ (blue) above x₃ (red).
   - The x₃ variance stops growing at Δ/Δ₀ ≥ 4.
6. **Unmentioned internal inconsistencies.**
   - Table 1 gives "mean velocity v = 1", but §6.2 gives a mean speed of ≈0.1453 (p. 18). A mean speed cannot be smaller than the size of the mean velocity.
   - The p. 18 formula "⟨1/v⟩ = 1/(β(α−1))" has α and β swapped (its 8.116 matches 1/(α(β−1))).
   - The covariance type is unstated, so mapping ℓ onto λ is uncertain.
7. **Minor page references.**
   - t_d is defined on p. 19, not p. 18.
   - "§4 argument, p. 10": §4 is on pp. 8–9, and the pseudo-symplectic zero is on pp. 14–15.
   - The summary credits the no-knots/no-braids property to the stagnation-free condition alone; the paper requires zero helicity *and* no stagnation together (p. 5).
8. **Relevant but missing from the summary:** the p. 3 exclusion of Dagan ln t growth, and the explicit "focus mainly on the Fickian regime" (p. 2).
9. **Checked and consistent:**
   - the crossed indices printed in eq. 14 (p. 5), traced back to L21's eqs. 2.16, 2.20 and 2.22–2.23;
   - the quotes on pp. 4, 8, 11, 13, 15 and 21;
   - the √tol scaling (Fig. 4b, p. 14).
   
   Lester et al. 2025 is not in `docs/references` and could not be checked.

##### Implications for SF-43 (≤300 words)

**What the paper states.**
- Zero transverse macrodispersion is proved only as time and volume go to infinity. It rests on three assumptions: bounded label fluctuations (p. 8), stationarity and ergodicity (p. 4), and no stagnation points.
- Pure advection gets no finite-time prediction.
- Transverse spreading means broadening at a fixed downstream position; the Dagan ln t growth of time-based variance is excluded (p. 3).
- The error scalings come from a periodic test flow whose streamlines close. Per-pass errors are Gaussian: Pollock ∝ Δ² (still significant at 16 cells per ℓ), Runge-Kutta ∝ √tol. Error growth with distance is assumed linear, not measured.

**Inference.**
- **This supports control-plane estimands.** Time-based α_T in B&dD and SF-39 can include positive transients (Dagan-type coupling, drift tilt) that do not appear at planes.
- **L23's blame on tracking for B&dD's α_T is qualitative.** Using L23's own numbers (σ²_x ≈ 10⁻³ per 16ℓ pass at σ² = 4), eq. 36 gives a spurious α_T of ≈0.006–0.008ℓ, or ≈0.02ℓ at 10 cells per ℓ. That is about 10–20 % of B&dD's ≈0.1, if ℓ ≈ λ.
- **On W8:**
  - L23 reports errors that accumulate like a random walk, never exponentially.
  - L21 and L23 (eq. 48) say transverse deformation stays bounded in the exact flow, with only algebraic growth from shear.
  - So exponential separation between the h/8 and h/16 passes is not explained by these papers.
  - To diagnose it, split the separation into label drift (Δψ, L23 eq. 35) and offset along the streamline. Only label drift feeds α_TP.
- **The zero claim cannot serve as a test oracle.** From eqs. 19 and 22, Δx₂ = −Δψ̃₁/v̄, so "bounded ψ̃" is the conclusion assumed in advance.

---

#### 3. Lester, Dentz, Bandopadhyay & Le Borgne (2021)

##### Extraction (item 7)

- **Citation.** "The Lagrangian kinematics of three-dimensional Darcy flow", *J. Fluid Mech.* (2021), vol. 918, A27, doi:10.1017/jfm.2021.362. Received 6 Apr 2020, revised 4 Mar 2021, accepted 20 Apr 2021.
- **Scope.** Steady 3D Darcy flow with scalar conductivity: "isotropic, finite and smooth … without sources and sinks and away from boundaries" (A27-3). Whether these constraints are physically plausible is left open.
- **Kinematic claims.**
  - No stagnation points (eq. 2.3), argued from the absence of closed streamlines (A27-5), which does not actually follow.
  - Zero helicity (eq. 2.4), but non-zero for tensor K (eq. 2.5).
  - Integrability with two invariants, and travel time as a 1D integral along each streamline (eq. 2.8).
  - Euler representation (eq. 2.9) with non-zero streamfunction gradients (eq. 2.10), plus gauge freedom (eqs. 2.11–2.14).
  - Zero helicity is called "necessary and sufficient" for a streamfunction pair (A27-8). Yet Fig. 3 and A27-10 treat integrable, stagnation-free flows as the wider class.
  - Governing equations 2.16–2.23 (the index problems noted above are verified).
- **Deformation and stretching.**
  - "The length ℓ(t) of fluid elements can only grow algebraically … ℓ(t) ∼ t^r", compared with exponential growth set by the Lyapunov exponent in chaotic flows (A27-7).
  - In streamline coordinates (φ, ψ₁, ψ₂), eqs. 4.11 give the principal strain rates in terms of the speed and a quantity m. L23 uses the symbol m for something different.
  - The velocity gradient is upper triangular with no transverse shear (ε₂₃ = 0; eqs. 4.12–4.13), i.e. "two superposed shear flows" (eq. 4.14). Growth "is algebraic and range[s] from sublinear to ballistic" (A27-18).
  - **No Lyapunov exponent, finite-time Lyapunov exponent or stretching rate is computed anywhere.**
  - The constraints "preclude … chaotic mixing" for "any degree of heterogeneity", for fields that are "smooth (i.e. not discontinuous)" (A27-27).
- **The test conductivity field (§3).**
  - Eq. 3.1: f = ln(k/θ) = sin(2πx₁)cos(2πx₂)sin(2πx₃) + (2/5)sin(2πx₁)sin(8πx₃), on a periodic unit cube. It is "not presented as a realistic representation".
  - The paper never calls it symmetric. By inspection, f is mirror-symmetric in x₂ (also odd in x₁ and in x₃ separately).
  - Its log-variance is σ²_ln k = 0.165 (my calculation), a weakly heterogeneous field.
  - The mean streamfunction gradient is set by ⟨k/θ⟩, the arithmetic mean, rather than the effective conductivity. This affects velocity magnitude but not direction.
  - Grids of 40, 80 and 160 points per period. Verification uses the divergence error and **only the angle** between the potential-based and streamfunction-based velocities (eq. 3.7, Fig. 4b). Magnitudes are never compared.
- **Numerics (A27-15).**
  - Linear interpolation is "no longer divergence free" globally.
  - "There is no known numerical method to enforce particles remain on streamsurfaces in steady 3-D flows". Spurious dispersion "persists even when … [tracking] under the potential-based velocity field v_φ".
- **Random media (§5).**
  - Stationarity of the velocity is assumed (eq. 5.3).
  - Condition 5.6 as printed does not exclude ∇ψ_i = 0.
  - Stationarity of ψ̃ is inferred from stationarity of the velocity (A27-20). This does not follow: stationary increments are not stationarity.
  - Streamline spacing "neither diverge[s] nor converge[s]" (eq. 5.25).
  - **Transverse macrodispersion is defined on planes:** D_T,∞ = lim_{x₁→∞} ⟨dσ²_c/dx₁⟩ (eq. 5.28). Here σ²_c is the concentration-weighted transverse moment of a steady Gaussian line-source plume, unnormalized and without the factor ½.
  - **No backflow is assumed:** "As ∂φ/∂x₁ < 0 ∀x" (A27-23).
  - The change of variables in eq. 5.37 ignores the Jacobian.
  - "We may approximate" the label autocorrelation as exp(−x₁/ζ) (eq. 5.41). This gives ⟨σ²⟩ ≈ σ²₀ + σ²_ψ(1 − e^(−x₁/ζ)) (eq. 5.43) and zero asymptotic dispersion (eq. 5.46). Neither ζ nor σ²_ψ is ever estimated.
  - B&dD's "widening cone" is called "inconsistent with the kinematic constraint" (A27-26).

##### Discrepancies with the project summaries

- **All L21 statements quoted in the L23 summary check out:** A27-8, A27-15, A27-20, A27-27, the index analysis of eqs. 2.16/2.20/2.22–2.23, and the mirror-symmetric §3 field.
- **Missing from the summaries:**
  - the angle-only verification;
  - the ⟨k/θ⟩ velocity scale;
  - the no-backflow assumption;
  - the assumed exponential autocorrelation;
  - the plane-based, concentration-weighted definition of transverse dispersion;
  - the A27-10 statement that the results cover integrable flows, not only helicity-free ones.

##### Implications for SF-43 (≤300 words)

**What the paper states.**
- Transverse macrodispersion is defined on control planes.
- Its approach with distance is exponential, but only because an exponential label autocorrelation is assumed; the decay length ζ is unknown.
- The paper assumes no backflow anywhere.
- Deformation is algebraic, and no rates are computed.
- Its only test field is deterministic, symmetric and weak (σ² ≈ 0.17), verified by angle only.
- Tracking on a potential-based velocity still leaks off streamsurfaces.

**Inference.**
- **SF-43 can tell three behaviours apart:**
  - α_TP(d) ∝ e^(−d/ζ) (L21; negligible over [25, 94]λ if ζ ~ λ);
  - ∝ d⁻² (first-order theory);
  - roughly constant (B&dD).
  
  A single slope fitted over [25, 94]λ throws this away. Slopes on sub-windows such as [25, 50] and [50, 94], or the full d-profile with uncertainties, keep it. For Gaussian covariance, first-order algebraic decay already contradicts an exponential label autocorrelation.
- **Backflow at σ² = 4.** Local backflow can make particles cross a plane more than once. SF-43 needs an explicit rule (first passage or signed net flux); L21 does not cover this.
- **Weighting differs.** SF-43's flux-weighted crossing variance is not L21's concentration-weighted moment at finite distance. Both go to zero asymptotically only if transverse displacements stay bounded.
- **No evidence for random fields.** L21 shows nothing about eq. 14 reproducing Darcy flow for random or σ² = 4 fields. Results on the symmetric field do not carry over, which fits the project's streamline-closure finding.
- **On W8.** Algebraic deformation with no transverse shear means transverse separation in the exact flow stays bounded. Persistent exponential growth of the h/8–h/16 difference *in label space* would therefore point to reconstruction or integration error, not Darcy kinematics. Even exponential growth only along the streamline is not predicted by these papers, so its source should be measured rather than labelled chaos.

---

Not checked: I read figure values off page images without zooming, so log-axis readings could be off by 10–20 %. I did not look at SF-39 or SF-43 code or data, so the W6 and W8 explanations above are hypotheses. Lester 2022 and 2025 are not in `docs/references`.
