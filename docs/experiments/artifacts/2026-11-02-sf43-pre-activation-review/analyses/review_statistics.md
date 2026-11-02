I found **1 blocking issue, 5 major and 10 minor**. The blocking one: as recommended, the design has no coverage-qualified procedure for some parts of the robust-sign test. Under the draft's own H4 rule ("no qualified procedure, p = 1"), all four robust-sign hypotheses would enter Holm with p = 1. Nothing was modified. All computations were inline Python 3.13 / numpy / scipy on CPU and wrote no files.

Notation: `R` = the draft review (`docs/experiments/2026-11-02-sf43-pre-activation-review.md`), `D` = the draft decisions (`docs/decisions/2026-11-02-sf43-pre-activation-decisions.md`), `S42` = `docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/`.

## Findings

**S1 — BLOCKING. The coverage evidence does not cover the recommended design. H15's "not re-simulated" list and H6's method statement are wrong.**
- **Statements attacked**
  - D:104-105: "Component method: Student (the only method that qualified the unpaired and interaction contrasts; it also qualified the ladders)".
  - D:170-172 (H15): the robust-sign pipeline is not re-simulated.
  - R:467-469: pipeline coverage "not re-simulated at N = 150 (bracketed by 100 and 300)".
  - R:448 (m3): E_int is kept.
  - D:80-81: option (iii) is "already qualified".
  - D:67-69: untestable hypotheses enter Holm with p = 1.
- **Evidence (`S42/coverage/summary.json` verdicts, which I recomputed)**
  - **(a) E_int (paired h_max contrast) has no qualified method at any n.** `contrast:paired`, `/student` and `/boott` are all `qualified=False` at n = 20, 30, 50, 100 and 200. For Student under stress, UCL/tail is 16.8 at n = 30 and 7.05 at n = 100. The SF-42 note says so explicitly (SF-42 note lines 975-978 and 1706-1710): "No method qualifies (a) the paired contrast … If SF-43 needs the paired contrast … it needs a different procedure or a larger n … does not extend to other n, tails or distributions".
  - **(b) The sampling lower limit L_e(a/2) is not qualified at N = 150.**
    - `sampling:student` lower is never qualified (stress− UCL/tail 1.73 at n = 500, 1.48 at n = 1650).
    - `sampling:boott` lower is qualified only at n = 200, 1000 and 1650. At n = 100 it fails (stress− 2.0); n = 500 is a marginal fail.
    - Production N = 150 (variant A) and 100 (variant B) therefore have no qualified sampling procedure.
    - The Gate 4 power runs used `StudentMean` (gate4_power.md:16), a procedure that never qualifies.
  - **(c) m = 8 adds tails that were never simulated.** `covproc.py:64` sets `HOLM_M = 7`. The new tails are the sampling tail 0.05/16 = 3.125e-3 (smallest simulated: 3.571e-3), the K=4 component tail 7.81e-4 (smallest simulated: 8.93e-4) and the E_h input tail 3.91e-4 (smallest simulated: 4.46e-4). H15's R = 49 000 is also below G5.3's 175/tail rule at 3.125e-3, which needs 56 000.
  - **(d) "Already qualified" for (iii) holds only under a reinterpreted criterion.** It requires judging the halved test against 1.25·a_j. Judged against "1.25 × the tail actually used" (G5.3's wording), n = 100 fails: UCL/tail at a_j/2 is 1.999 (stress− lower).
  - **What validity actually rests on.** The 189 pipeline cells have 0 misses in 24 500 each. But they are at m = 7, at N ∈ {100, 300, 1000} for K=3 and {30, 100, 300} for K=4, with biases of at most 2 SE. The draft relies on them without naming pipeline-level qualification as its route.
- **Correction (pick one route)**
  - **Route 1.** Have the owner record that Gate 4 is qualified at the pipeline level (an interpretation of G5.3). Add pipeline cells to H15 at the variant's arm sizes (production 150, ladders 100, E_W 150, h_max 30), at the m = 8 levels, under normal, stress and stress−. At m = 8 the pipeline is safe by monotonicity: UCL ≈ 1.2e-4, far below 1.25 × 6.25e-3.
  - **Route 2.** Raise production to N ≥ 200 with a bootstrap-t lower limit. Keep m = 7, or simulate 3.125e-3. Make E_int a stated assumption (SF-39: at most 1e-6 relative), with new K=2/3 pipeline cells.
  - **Either way.** Name the sampling method. Pre-register B (about 26 000 resamples for p_L at 3.9e-4). Use R ≥ 56 000 for the halved Gate 5 test at m = 8. State the halved-test criterion explicitly.

**S2 — MAJOR. α_T^FO is not the first-order value of SF-43's estimand, and the pre-registered first-order benchmarks are wrong.**
- **Statements attacked**
  - D:57-62: "value of the same estimand"; "halves against the first-order ratio 3.57 … window ratio against 6.92".
  - R:84-90: "75 / 155 / 205 times"; "3.6 times smaller"; "6.9 times smaller".
  - R:235.
  - D:219: first-order values "confirmed".
- **Evidence (my own first-order spectral quadrature, discrete transverse wavenumbers 2πm/W, uniform-weight centring with a square source)**
  - The continuous case reproduces 7.9922e-5.
  - Half-slope on [25, 94] in units of σ²λ, effective (centred) values:

    | W | source L | effective value | fraction of unbounded |
    |---|---|---|---|
    | 24 | 19.2 | 1.032e-6 (one-particle 1.089e-6) | 1/77 |
    | 12 | 9.6 | 3.0e-9 | 1/2.7e4 |
    | 48 | 19.2 | 7.9e-6 | 1/10 |

  - The local α_T decays exponentially, not as d⁻². At W=24 it is 4.97e-5 / 1.46e-6 / 1.3e-8 at d = 25 / 40 / 59.5; the unbounded values are 3.54e-4 / 1.38e-4 / 6.26e-5. The cause is the lowest nonzero transverse wavenumber 2π/W.
  - The first-half/second-half ratio is 4.2e3 at W=24 and 6e7 at W=12, not 3.57. On [94, 194] the first-order value is about 0.
- **Consequences**
  - H3 stays valid, and it is conservative, by 77× and 2.7e4×.
  - M1's premise is moot in this geometry. The linear-theory value is 0.026 SE at σ²=0.25 (2.6e-7 against an SE of 9.8e-6 at N=150) and 0.011 SE at σ²=1.
  - The 3.57 and 6.92 readings would label any measured decay "first-order-like" when first order predicts no growth after about 40λ.
  - At σ²=0.25 the pilot mean (displacement basis) is about 2000× the linear value for this geometry. That matters for how a "physical (restricted)" label at 0.25 (H4) is read, because the representation (Pollock) contrast is outside the budget.
- **Correction.** Keep the H3 threshold, or revert to 0 (equivalent in practice). Compute and pre-register the first-order benchmark for the simulated geometry: extend `firstorder_window.py` with the lattice sum and the source term. Relabel 3.57 and 6.92 as unbounded-domain values, and fix the "confirmed" classification.

**S3 — MAJOR. The Fieller-form qualification argument is overstated, and the stress+ check is non-gating at σ²=4.**
- **Statements attacked**
  - R:283-286 and D:77-79: "exactly the stress variable … independent of the pairing correlation and the CVs".
  - D:168-170: the J3B-bound block is "reported sensitivity".
- **Evidence**
  - In `covproc.py:173-182`, L = r(1+cv·z_L) and P = 1+cv·z_P share one cv, and z_L − z_P = 2B·u. So D = r·cv·2B·u by construction of SF-42's simulation, not of the data. J3B at σ²=4 has CV_L 0.301 and CV_P 0.145.
  - What does carry over: the test's size depends only on the standardized law of D_r at the boundary.
  - The only evidence specific to Gate 5 is J3B at σ²=4: e_r skewness G1 = 1.82 (upper bounds at 80%/95%: 2.43 / 2.73) and excess kurtosis G2 = 2.72 (6.38 / 7.59). The stress family is 1.94 / 4.71, built from SF-39 α_TP, α_TAM and α_LAM series, none of them paired or α_LP.
  - At σ²=4 the null is plausibly true and near the margin (pilot ratio 1.137, `j3b_pairing.json`). The binding side is H02, whose upper limit is the anti-conservative one under positive skew.
  - My simulation (n = 250, R = 8000, B = 1599, stress+ mixture p = 0.0628, v = 0.169): bootstrap-t miss/tail 0.77-0.88 at tails 3.1e-3 to 7.1e-3; Student 2.7-3.5. So candidates (ii) and (iii) probably pass; Student would not.
- **Correction.** Restate the claim. Make the J3B-bound family (2.73 / 7.59, plus its mirror) gating at σ²=4's n. That is cheap and likely to pass. Compute pilot moments of D_r(θ), not only e_r.

**S4 — MAJOR. H16 is empty, and G5.7's levels remain undefined.**
- **Statements attacked**
  - D:176: "Fixed … from the review's section 3.d (proposed value and reachability there)".
  - R:37-39: "fixed before data".
  - R:429.
- **Evidence.** In section 3.d of R, "numerical" appears only in the outcome table (R:272). There is no fraction, no UCL level, and no definition of the extrapolated estimate when p_L = −∞, which R:242-243 says is the usual case at σ² ≥ 1. The f = 0.5 proposal exists only in the uncommitted gate4_power.md section 7. Also undefined under H3:
  - whether the interval must exclude 0 or α^FO;
  - which class applies when 0 < RL ≤ α^FO;
  - the levels of U and B_k in "unresolved (bounded)";
  - the error rate of the non-Holm "numerical" label.
- **Correction.** Write H16 in full: the fraction f, the UCL level, the extrapolation in both E_h branches, the levels of U and B_k, thresholds consistent with H3, and a statement that the label is descriptive per amplitude.

**S5 — MAJOR. H13's floors and the analysis at a realized n are undefined.**
- **Statements attacked.** D:149-153; R:368-372.
- **Evidence**
  - Only Gate 5 has floors (D:82, set equal to N). There is no floor for the production, E_h, E_W or h_max arms.
  - There is no rule for which arm's shortfall makes a robust-sign hypothesis untestable.
  - There is no rule for which coverage grid point applies when an arm is analysed between its floor and N.
  - The one-hour grace turns the estimator into the type that includes in-progress runs, which has O(1/t) bias. The cited o(1/t) property is for completed runs only. The effect is negligible: at most two units are in flight.
- **Correction.** For each hypothesis, list the arms it needs and their floors (below floor: p = 1). Analyse with the procedure qualified at the largest simulated n not above the realized n, or require the full N (bib6_statistics.md section 2(ii)).

**S6 — MAJOR. The statistical claims cannot be reproduced from the artifact.**
- **Statements attacked.** R:402-405 and `scripts/README.md:11-12`.
- **Evidence.** `git ls-files` lists only `design_cost.py`, `firstorder_window.py` and `lattice_check.py`. `gate4_power_sim.py` and `gate5_options.py` are missing. They are the sources for P(RL>0), the MDEs, the "numerical" fraction, the Fieller re-judging, the Hall prototype and the Gate 5 sizing, which drive H4, H5, H9 and H16.
- **Correction.** Commit both scripts with their seeds, R and B.

**S7 — MINOR. Power is overstated and not computed at the G5.2 inputs.**
- **Statements attacked.** R:29-30 ("1.00 at every amplitude") and R:383.
- **Evidence**
  - The draft's own table at R:261 shows 0.69-0.90 in the S3 order-1 corner under the pessimistic noise model at σ²=4. Variant A has the same 100-seed ladders, so that corner is unchanged.
  - Point means and SDs were used, not the 80% bounds (gate4_power.md:130), with normal data, Student sampling and R = 400.
  - Width deficits were only simulated at 5-10%, while SF-39's width contrast is unresolved at 0.25 (SE = 2.05μ) and at 1 (−9.8% [−65, +45]).

**S8 — MINOR. The Gate 5 MDE and power numbers are off (R:266-267).**
- My Fieller-form simulation (bivariate normal, conservative pilot inputs):

  | σ² | n | power at ratio 1 (levels 0.05/7, 0.05/8, 0.05/16) | ratio with power 0.8 |
  |---|---|---|---|
  | 0.25 | 150 | 1.00 at all three levels | 1.085 |
  | 1 | 150 | 1.00 at all three levels | 1.083-1.085 |
  | 2.25 | 150 | 1.00 at all three levels | 1.06 (not 1.08-1.09) |
  | 4 | 250 | 0.899 / 0.892 / 0.82 | 1.02 full level, 1.0125 halved (not 1.03) |

**S9 — MINOR. H2's "same estimand" claims (R:27-28, R:216-219, D:46-47).**
- The exact zero holds only for uniform weights. With flux weights the per-realization normalization breaks the reflection symmetry, so the expected seed-displacement covariance is nonzero near the source edges. "O(1/n_eff)" is never defined or derived.
- The pilot agreement check (R:218-219) has no power: |pos−disp|/SE_pos = 0.78 / 0.64 / 0.59 / 0.48, while 2·SE_pos equals 3.5× / 0.43× / 0.13× / 0.07× the displacement mean.
- Disclosure gap: H4's admission of σ²=0.25 rests on the displacement-basis pilot mean (5.09e-4, t ≈ 21), not only on SD ratios.
- **Correction.** Define the confirmatory estimand as the displacement-basis one (spec item 1 allows this).

**S10 — MINOR. Serial gatekeeping is dismissed on a stale premise (R:291).** The "probability about 0.5" assumes Gate 4 power of 0.8 (position basis). With the displacement-basis power the draft claims, P ≈ 0.7-1.0. At tails ≥ 0.0125 the existing bootstrap-t TOST already qualifies at n = 170.

**S11 — MINOR. The E_h fallback is not free of assumptions (R:243-245).** The 3·range bound requires the unmeasured remainder to be at most 3× the observed range (roughly order ≥ 0.4). It should be listed among the stated assumptions.

**S12 — MINOR. H3 needs explicit code.** `budget.py:385-390` tests RL > 0. Shifting only the production data by α^FO would change E_W (`e_w` uses the production arm). Implement p = inf{a : RL(a) > c} with c explicit.

**S13 — MINOR. FWER wording.** G5.3 tolerates miss rates up to 1.25×, so the FWER is at most 6.25%. D:67 should say so.

**S14 — MINOR. The blinded re-estimation (D:125-126) is inert but untested.**
- A resize requires an SD 2.5× to 14× the planning SD (precision-target n of 0.8-23 against N = 150).
- It is not covered by the fixed-N pipeline cells.
- Excluding Gate 5 also removes the only safety net at σ²=4, where the planning CV rests on one field (0.30, or 0.12 without seed 420836).

**S15 — MINOR. Gaps in the Gate 5 and E_h rules.**
- The agree / disagree / inconclusive verdicts under the Fieller form need a ratio CI obtained by inverting the test over θ. It is not defined.
- "Achieved power" (acceptance C and D) is not defined.
- The λ/h = 4 rung at σ²=4 is priority 2 in `design_cost.py` but enters H6's E_h rule. The bound stays valid if the rung is missing, but this should be stated.

## Claims I tried to refute that survived
- The unbounded α_T^FO values (7.992e-5, halves 1.385e-4 / 3.882e-5, and 1.155e-5 on [94, 194]) reproduce with an independent spectral integral.
- H3 is valid and its coverage is shift-invariant, provided every arm shifts together. The direction of conservatism is confirmed (see S2).
- Holm with p = 1 for untestable hypotheses is valid; `multiplicity.py` counts NaN as 1 within m.
- The intersection-union max(p_TP, p_TAP) is valid and costs no power, since TAP ≈ TP.
- The Fieller form keeps the ratio-of-means estimand. The deterministic procedure-selection rule is independent of SF-43 data, so it is valid.
- Bootstrap-t at half the Holm level, judged against a_j: 1.000 / 0.633 / 0.641 at n = 100 / 200 / 500. At m = 8 the first level holds by monotonicity (7.14e-3 ≤ 7.81e-3).
- Bootstrap-t at the full Holm level at n = 200: maximum 1.196 over the simulated tails. Student never qualifies at the full level.
- The seed-order prefix rule: my simulation with about 20% truncation and correlation 0 / 0.2 / 0.5 gives size/tail 1.02 / 1.02 / 0.95-1.03. The Glynn-Heidelberger o(1/t) claim for completed runs agrees with a renewal argument.
- var x0 is constant in d: `set_differs = 0` in every J2/J3 pilot row.
- H6's changes give an E_h at least as large, pointwise (taking the min over triples needs no extra level split).
- The union bound makes G5.4's joint bootstrap unnecessary for validity (the deviation should still be recorded).
- SD ratios 37.13 / 9.84 / 3.72 / 1.14. The α_LP estimand difference is at most 0.44% in the pilots.
- Bootstrap-t passes the reporting tails (maximum 1.199).
- Excluding Gate 5 from the blinded re-estimation is conservative for type I error.

Not checked: the literature, cost and executability claims. I did not re-implement the Hall prototype, and I could not quantify the flux-weight pos−disp bias because the per-particle data are not in the repository. Risk: until S1 is resolved, the draft's own H4 rule would set every Gate 4 hypothesis to p = 1.

### Critical Files for Implementation
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/decisions/2026-11-02-sf43-pre-activation-decisions.md
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/2026-11-02-sf43-pre-activation-review.md
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/mcstat/covproc.py
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/coverage/summary.json
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/artifacts/2026-11-02-sf43-pre-activation-review/scripts/firstorder_window.py