No file was written anywhere. Every computation ran inline with `python3 -B -c` on the local stack (Python 3.13.13, numpy 2.5.0, scipy 1.18.0). Commands are labelled [C1]–[C20] at the end. Note: `scripts/mcstat/__pycache__` and `scripts/seedlattice/__pycache__` under ROOT appeared at 10:44–10:50 from another process. They contain modules I never imported (budget, multiplicity, seedlattice).

## Bottom line

1. **"stress" was not derived from Gate 5 data, but it fits σ² = 4.**
   - It is SF-39's `tp.alpha_TP` on the `s4_n32` ladder rung (n = 20), the maximum over 108 series.
   - The simulator applies it to the paired difference itself, so e_r carries the full shape at every ρ.
   - Direct e_r evidence (8 pairs per amplitude) is near-symmetric at 0.25, 1 and 2.25. At σ² = 4 it reaches the stress level (G1 1.82, G2 2.72), driven by one field.
   - That positive skew makes the H02 side (ratio ≥ 1.1) anti-conservative. H02 is the side that binds at σ² = 4, where the pilot ratio of means is 1.137.
2. **Rewriting the TOST in Fieller form keeps the estimand and is already qualified by R5.**
   - The ratio hypotheses become one-sample tests on `L − 1.1P` and `L − P/1.1`.
   - At the boundary this contrast is exactly the stress variable, so its size equals R5's existing `sampling` cells and does not depend on ρ or CV.
   - From those cells: bootstrap-t at each Holm level a_j qualifies for n ≥ 200. Bootstrap-t at a_j/2 qualifies **now** at n ∈ {100, 200, 500}. Student never qualifies.
3. **A skewness-corrected Fieller test (Hall 1992 / Johnson 1978) looks best.** In my local prototype under stress it reaches rate/tail ≤ 1.08 at n ≥ 150 and is exact on normal data. It is analytic, so qualifying it costs seconds of CPU.
4. **The geometric-mean-ratio TOST is ruled out.** At σ² = 4 it shifts the estimand by 24–37 % of the margin, and the per-seed log ratios are still skewed (G1 1.35).
5. **Removing Gate 5 from the family, or gating it behind Gate 4, needs no new Gate 4 coverage cells.** The m = 3 tails are a subset of the tails R5 already simulated.
6. **The restricted coverage run is about 28 core-h, plus 17.5 for a sensitivity block.** That is about 1.4 h on 32 workers at R5's measured per-cell speed.
7. **Running it inside the production job with a pre-registered selection rule is valid, with conditions.** The main trap is spec line 165: if no procedure qualifies, SF-43 stays active. Keep the halved bootstrap-t, which is already qualified, as the last entry before "descriptive".
8. **The σ² = 4 sizing rests on one seed.** The consumer CV of 0.30 drops to 0.12 without seed 420836.
   - Proposed floors: 150 pairs at 0.25, 1 and 2.25, and 205–250 at σ² = 4.
   - Estimated Gate 5 cost: 6.6 GPU-h if the production twin series is reused at 0.25 and 1. The same design with floor 30 costs 3.3 GPU-h on the same measured-time basis.

## 1. Is "stress" relevant to Gate 5?

**Where it comes from.**
- The envelope (skew 1.9357, excess kurtosis 4.7108) is at `stress_moments.json:65-68`.
- It equals the 95 % seed-bootstrap bounds of the SF-39 `s4_n32.csv` `tp.alpha_TP` series, n = 20, point G1 1.300 and G2 1.958 (`stress_moments.json:1153-1164`).
- The rule takes the maximum over 108 series and 4 amplitudes (`covdist.py:6-17`, `:223-232`). None of those series is `alpha_LP` or a paired contrast.
- The large SF-39 production series (n = 25–71) have G1 between −0.38 and 0.82. Their 95 % upper bound on |G1| is ≤ 0.84, except `alpha_LAM` at σ² 0.25 (1.41) [C1].

**How it enters the TOST cells.**
- `covproc.py:174-176` builds `zL − zP = 2B·u` with u drawn from the stress distribution. So the stress shape sits on e_r at every ρ and CV.
- The mixture is p = 0.0646 at +3.11 SD with within-variance 0.331 [C5].

**Per-realization moments** [C2, C4]. Eight pairs per amplitude. σ² 0.25 uses the J3 Par2 run at horizon 1.2. G1 and G2 are bias-corrected. Bounds are the 80/95 % quantiles of |G1\*| and G2\* over 20 000 seed resamples (covdist's rule).

| σ² | alpha_LP consumer G1 / G2 | alpha_LP Par2 G1 / G2 | e_r G1 (|G1| UCB80/95) | e_r G2 (UCB80/95) | log(L/P) G1 |
|---|---|---|---|---|---|
| 0.25 | 1.56 / 2.92 | 1.40 / 1.65 | −0.34 (0.89/1.42) | −0.86 (0.55/2.38) | −0.43 |
| 1 | 0.50 / 0.38 | 0.39 / −0.76 | −0.01 (0.88/1.38) | −1.45 (−0.18/2.00) | 0.00 |
| 2.25 | −1.08 / 0.72 | 0.45 / 1.48 | −0.93 (1.22/1.76) | 1.09 (1.86/3.61) | −0.67 |
| 4 | **2.24 / 5.58** | 0.91 / 0.26 | **+1.82 (2.43/2.73)** | **2.72 (6.38/7.59)** | +1.35 |

**alpha_TP G1, listed σ² 0.25 / 1 / 2.25 / 4** (|G1| ≤ 1 everywhere) [C1, C4]:
- J3 consumer, flux weights, position basis: 0.27 / −0.30 / 0.13 / −0.50.
- J3 consumer, flux weights, displacement basis: 0.47 / −0.44 / −0.05 / −0.97 (G2 = 3.08 at σ² 4).
- Par2, uniform weights, position: 0.31 / 0.00 / −0.56 / −0.21. Displacement: 0.52 / −0.76 / −0.20 / 0.09 (G2 = 2.66 at σ² 4).
- J2 (6 fields), position: −0.39 / −0.46 / −0.77 / 0.19. Displacement: 0.01 / 0.88 / 0.43 / 0.57.
- SF-39 `tp.alpha_TP` production, n = 25/50/62/71: 0.33 / 0.16 / −0.05 / 0.19, with 95 % bounds 0.82 / 0.54 / 0.52 / 0.50.

**How little 8 pairs constrain** [C5]:
- 58.6 % of n = 8 samples from the stress mixture contain no draw from its minority component.
- At n = 8 the G1 quantiles (5/50/95 %) are −1.03 / 0.67 / 2.39 under stress and −1.24 / 0 / 1.25 under normal. Values from −0.93 to −0.01 cannot discriminate (for example P(G1 ≤ −0.01) is 0.29 under stress and 0.49 under normal).
- Only σ² = 4 is informative: P(G1 ≥ 1.82) is 0.009 under normal and 0.21 under stress.
- That cell is one field, seed 420836 [C3]: L = 17.44 against a median near 9.3, P = 11.39, L/P = 1.53, e = +0.44 (about 2.1 SD). The other seven e lie between −0.15 and +0.18.
- Pooling the standardized e_r over amplitudes (n = 24, G1 0.25) assumes one common shape, which the σ² 4 cell contradicts.

**Verdict.** At σ² = 4 stress is plausible and possibly mild; the bootstrap bound of 2.73 / 7.59 is above it. Elsewhere it is unrefuted.

## 2. Options for a confirmatory Gate 5

**(i) The Fieller form is a "limit of a mean".**
- H02: μL − 1.1·μP ≥ 0 and H01: μL − μP/1.1 ≤ 0 are linear hypotheses. Test each as a one-sample test on d_r = L_r − c·P_r.
- The estimand (ratio of means) is unchanged and there is no linearization.
- At the boundary, d_r = 2B·r·cv·u exactly. So the size equals R5's `sampling` cells: the upper row is H02, the lower row is H01. The sampling tails are exactly {a_j/2} ∪ {0.05} (`covproc.py:75-76`).
- Re-judged from `summary.json` [C9]:
  - **Bootstrap-t at a_j:** max UCL/a_j is 1.607 at n = 100, 1.196 at 200, 1.186 at 500, 1.112 at 1000. Only the overlapping tails 8.33e-3, 0.0125, 0.025 and 0.05 exist. The missing 7.14e-3 is bracketed by 6.25e-3 (1.17) and 8.33e-3 (1.20) at n = 200.
  - **Bootstrap-t at a_j/2:** 2.598 at n = 50, **1.000 at n = 100**, 0.633 at 200, 0.641 at 500.
  - **Student at a_j/2:** 1.449 at n = 200, 1.027 at 500.
  - **Student at a_j:** never qualifies.

**(ii) Local prototype of skewness-corrected tests** [C10]. Rate/tail at the 7 Holm tails on the skewed side, 2×10⁵ replicates per n.

| n | Student | Johnson 1978 | Hall 1992 |
|---|---|---|---|
| 50 | 4.22–1.80 | 3.00–1.20 | 3.05–1.24 |
| 100 | 2.88–1.52 | 1.28–1.04 | 1.35–1.06 |
| 150 | 2.37–1.40 | 1.02–1.01 | 1.08–1.02 |
| 200 | 2.20–1.36 | 1.02–1.01 | 1.05–1.02 |
| 300 | 1.89–1.28 | 1.00–0.96 | 1.01–0.98 |

- On normal data all three methods give 0.96–1.03.
- Under a heavier mixture (2.73 / 7.59) the corrected tests are conservative at n ≥ 100 (0.55–0.95) and fail at n = 50 (5.06).
- My Student rows agree with the library's `run_sampling`, which gives 2.0–2.07 at 8.33e-3, n = 200.

**The options, one by one.**

- **(a) Bootstrap-t TOST on the log ratio at n ≥ 200–300.**
  - Basis: one-sided bootstrap-t coverage error is O(n⁻¹) against O(n^-½) for Student (Hall 1988, Ann. Statist. 16:927; Hall 1992, *The Bootstrap and Edgeworth Expansion*; Efron & Tibshirani 1993). TOST itself: Schuirmann 1987 (J. Pharmacokin. Biopharm. 15:657); Berger & Hsu 1996 (Stat. Sci. 11:283).
  - Estimand unchanged.
  - Coverage: 1.14–1.21 at n = 170 (`note:919-936`). Extrapolated to about 1.05–1.12 at n = 300. A procedure at 1.1× passes the G5.3 check with probability 0.83 at R = 49 000 [C11].
  - Needs a ρ grid. It is the most expensive option (about 10.7 GPU-h at 300 pairs everywhere [C20]).
- **(b) TOST on the mean of log(L_r/P_r).**
  - Basis: the bioequivalence TOST on log-transformed data (Schuirmann 1987).
  - Estimand shift relative to log 1.1 [C14]. With point CVs: +1.1 % / +0.3 % / −1.9 % / **+36.6 %**. Empirical: +0.8 % / +0.2 % / −1.2 % / **+23.8 %**. With the 80 % CV bounds at σ² 4: +84 %.
  - Skew persists at σ² 4 (G1 1.35). Rejected.
- **(c) Skewness-corrected t on the Fieller contrasts.**
  - Basis: Johnson 1978 (JASA 73:536); Hall 1992 (JRSS-B 54:221); Chen 1995 (JASA 90:767, verify).
  - Estimand unchanged.
  - Expected coverage: ≤ 1.08 at n ≥ 150; 1.28–1.35 at n = 100.
  - Use Hall's transform, because its cubic is monotone and can be inverted for the 90 % CI. Johnson's statistic is not monotone.
  - Needs about 30 lines of new code.
- **(d) Bonferroni-type halving (each side at a_j/2), applied to the Fieller bootstrap-t.**
  - Basis: a calibration safety factor (compare prepivoting, Beran 1987, Biometrika 74:457, verify).
  - The only option qualified at n ∈ {100, 200, 500} with no new simulation.
  - Power cost: n at σ² 4 rises from 205 to 232 [C19]. Very conservative at n ≥ 200 (0.63).
- **(e) Remove Gate 5 from the primary family (Holm m = 3).**
  - Basis: Holm 1979 (Scand. J. Stat. 6:65).
  - Gate 4's new tails (a/2 = 8.33e-3–0.025; a/6 = 2.78e-3–8.33e-3; a/8 = 2.08e-3–6.25e-3) are all among the tails R5 already simulated. No new Gate 4 cells.
  - Quantile ratios (z²) are 0.79 for the sampling term, 0.83 for K = 3 and 0.84 for K = 4: about 16–21 % fewer seeds for the same Gate 4 power [C18].
  - Gate 5 becomes descriptive. Its bootstrap-t 90 % CI is already qualified for n ≥ 100 (UCL/tail ≤ 1.11 [C8]).
  - Needs an owner decision (G5.6) and gives up W1's confirmatory remedy.
- **(f1) Fieller form**, as in (i).
- **(f2) Serial gatekeeping.**
  - Gate 4 (Holm m = 3) first, then Gate 5 (Holm m = 4, tails ≥ 0.0125) only if all three Gate 4 nulls are rejected. Basis: Bretz et al. 2009 (Stat. Med. 28:586); Dmitrienko & Tamhane 2007 (verify).
  - R5's log-ratio bootstrap-t already qualifies at n = 170 on those tails (max 1.23 [C8]).
  - Risk: with Gate 4 power 0.8 per amplitude, Gate 5 gets tested with probability of only about 0.5.

Other citations (verify): Fieller 1954 (JRSS-B 16:175); Hauschke et al. 1999 (Stat. Med. 18:93).

**Recommendation:** (f1) with (c) as the primary procedure, and (d) as the pre-qualified fallback.

## 3. Restricted coverage run

**Measured R5 host seconds per cell** [C6]. These were taken with 72 workers on 40 physical cores at `nice 10`, so the 11.5× host factor is already included.
- Student TOST: 0.4–1.2 s per cell.
- Bootstrap-t TOST (B 3000, R 24 500): 2054–3457 s at n = 170, about 16.4·n s. The family total was 15.8 core-h (`coverage.py:107`).
- `sampling` bootstrap-t (R 49 000): 1334 / 2622 / 11 121 s at n = 100 / 200 / 500.

**Replicates per cell.** Use R = 49 000 rather than the G5.3 minimum of 24 500 [C11]:
- Probability that an exactly nominal procedure fails one row: 0.0027 at 49 000, against 0.0815 at 24 500.
- Probability that a 1.1× procedure passes: 0.83 against 0.56. A 1.3× procedure passes with probability 0.007.

**Cells per option:**
- (a): 45 cells (n ∈ {200, 250, 300, 400, 500} × ρ ∈ {0, 0.5, 0.9} × 3 distributions), 7 tails, B 3000. 67 core-h at R 24 500, 134 core-h at R 49 000.
- (b) and (c): about 24 analytic cells, under 0.1 core-h.
- (d): no cells at n ∈ {100, 200, 500}. Twelve cells at n ∈ {150, 250, 300, 400}: 14.8 core-h.
- (e): none.
- (f2): (a) or (f1) at R 28 000, about 0.57 of their cost.

**Proposed plan (one new procedure, `tost:fieller`):**
- **Block A:** n ∈ {100, 150, 200, 250, 300, 400, 500} × {normal, stress, stress−}, R 49 000, B 3000. Each cell computes Student, Hall, Johnson and bootstrap-t, both sides, at all 14 tails (a_j and a_j/2), on the same draws. One grid serves all amplitudes. Cost 27.4 core-h.
- **Block S (reported, not gating):** the J3B σ² 4 bound as a "stress+" family (2.73, 7.59) and its mirror, n 150–400. Cost 17.5 core-h.
- **Block C:** the two-stage blinded re-estimation plus cutoff (section 7) for Hall. Negligible cost.
- **Total:** about 45 core-h, i.e. 1.4 h on 32 workers or 1.1 h on 40; ≤ 2.1 h with a 1.5× allowance for host load. The R5 log-ratio bootstrap-t backup (12 cells, 27 core-h) is redundant given (d).

**Running it inside the production job with "the first procedure in this ordered list that qualifies at n_grid".**
- **Ordered list:** Hall-Fieller; then bootstrap-t Fieller; then halved bootstrap-t Fieller; then descriptive.
- **Why it is valid:** the coverage output is synthetic, uses a fixed seed and a pre-registered plan hash, and n depends only on runtime under the prefix rule. Given n, the procedure is fixed and its size is what the simulation measured.
- **Pitfalls:**
  - n is unknown in advance. Analyze the largest grid point ≤ the reached n, using the seed prefix.
  - The coverage summary is not on G8's list of outputs readable during the blinded phase (decision `:176-177`). Either add it, or script the selection and seal it until the freeze.
  - Any cell unfinished at the freeze counts as not qualified.
  - The code must be frozen and tested before the job.
  - Run at `nice 19` with ≤ 32 workers, so roughly 8 physical cores stay free for the GPU job and colleagues.
  - Spec line 165 says a coverage requirement that no procedure meets keeps SF-43 active. Size Gate 5 at R5 grid points so the halved fallback always applies.

## 4. Gate 5 sizing (horizon 3.6; σ² 0.25 from J3) [C12]

Analytic Student-TOST approximation (`gate5.tost_power`) at a = 0.05/7. Cells show P(ratio = 1) / P(ratio = 0.95). MDE is the largest true ratio > 1 with power ≥ 0.8; "–" means power at ratio 1 is below 0.8. "LB" uses the 80 % conservative inputs (CV bounds and correlation lower bound).

| σ² | inputs | n₀.₈ | n = 30 | 50 | 100 | 200 | 300 | MDE at 30 / 50 / 100 / 200 / 300 |
|---|---|---|---|---|---|---|---|---|
| 0.25 | LB (0.148, 0.133, 0.955) | 7 | 1.00/1.00 | 1/1 | 1/1 | 1/1 | 1/1 | 1.069 / 1.077 / 1.084 / 1.089 / 1.091 |
| 0.25 | ρ = 0 | 64 | 0.12/0.05 | 0.60/0.17 | 0.98/0.40 | 1.00/0.75 | 1.00/0.92 | – / – / 1.029 / 1.050 / 1.059 |
| 1 | LB (0.103, 0.097, 0.875) | 7 | 1.00/0.98 | 1/1 | 1/1 | 1/1 | 1/1 | 1.066 / 1.074 / 1.082 / 1.087 / 1.090 |
| 1 | ρ = 0 | 34 | 0.71/0.20 | 0.97/0.38 | 1.00/0.73 | 1.00/0.97 | 1.00/1.00 | – / 1.028 / 1.049 / 1.064 / 1.071 |
| 2.25 | LB (0.149, 0.172, 0.725) | 26 | 0.90/0.29 | 1.00/0.52 | 1.00/0.87 | 1.00/1.00 | 1/1 | 1.018 / 1.038 / 1.056 / 1.069 / 1.075 |
| 2.25 | ρ = 0 | 83 | 0.03/0.01 | 0.33/0.11 | 0.91/0.29 | 1.00/0.61 | 1.00/0.81 | – / – / 1.019 / 1.043 / 1.053 |
| 4 | LB (0.451, 0.206, 0.617) | 205 | 0/0 | 0/0 | 0.13/0.05 | 0.79/0.23 | 0.96/0.36 | – / – / – / – / 1.026 |
| 4 | ρ = 0 | 380 | 0/0 | 0/0 | 0/0 | 0.20/0.07 | 0.61/0.18 | none |
| 4 | point (0.301, 0.145, 0.799) | 68 | 0.09/0.04 | 0.54/0.16 | 0.97/0.37 | 1.00/0.71 | 1.00/0.90 | – / – / 1.027 / 1.048 / 1.058 |

**Uncertainty of the σ² 4 consumer CV** [C13]:
- Leave-one-out CV_L is 0.119 without seed 420836 (n_point = 26) and 0.30–0.33 without any other seed (n_point 67–76).
- The bootstrap distribution is bimodal: quantiles at 20 / 50 / 95 % are 0.114 / 0.295 / 0.373.
- The normal-theory 90 % CI is [0.21, 0.54], but normal theory does not hold here.
- Taking the 80 % bound on sd(e) directly gives n = 121, against 205 from the separate-CV route that G5.5 prescribes.

**Proposed floors:**
- **150 pairs at 0.25, 1 and 2.25.** Coverage: Hall-Fieller (use 200 if the bootstrap-t is selected, 100 if the halved bootstrap-t). Stability: about 10 minority-component draws expected at n = 150, and at least 38 pairs at the 25 % re-estimation point.
- **205–250 at σ² 4.** The sizing and coverage requirements coincide there; at the halved level, n = 250 gives power 0.85.

**GPU cost** [C20]. Measured per-realization times (`note:1348-1358`), Par2 at 0.25 scaled ×1.09 (`note:1416`), and the consumer series at 0.25 and 1 reused from the production twin series (same 1040×240×240 grid, spec `:54` and `:76`; this needs the window, M and h_max to match):
- 150/150/150/250: **6.6 GPU-h**.
- 200/200/200/250: 7.75 GPU-h.
- 100/100/100/250: 5.45 GPU-h.
- 300 everywhere: 10.7 GPU-h.
- The floor-30 design on the same measured basis: 3.3 GPU-h with reuse, 3.7 without.

## 5. G7 factorial [C15]

Two-sided α = 0.05, power 0.8, on the log of cell means.
- **Within-seed main effects (I, D, R):** MDE = 0.258·CV_c at 60 seeds per cell, 0.368·CV_c at 30. Simple effects within one generator: 0.368 / 0.529. For example, CV_c = 0.03 gives 0.8 % / 1.1 %.
- **Generator G (Welch):** MDE = 0.516·CV_a at 60, 0.736·CV_a at 30.
  - With Par2 point CVs (0.095 at σ² 0.25, 0.070 at σ² 1): 4.9 % / 3.7 % at 60, 7.2 % / 5.3 % at 30.
  - With the 80 % CV bounds: up to 7.1 % at 60 and 10.3 % at 30.
  - This assumes Par2's random-spectral generator has the same CV as SF-18; that is not measured.

**Is it worth it?**
- G only explains SF-39's unmatched offset (W1: −4.8 % to −12.8 % at σ² 1, decision `:34`). That needs 60 per cell.
- For SF-43's matched Gate 5, both codes use SF-18. The free consumer-twin / Pollock-twin / Par2 decomposition already isolates the representation and the Euler-stepping-plus-injection effects, at all four amplitudes. That includes σ² 4, where the pilot signal is and where the factorial is not planned.
- Suggestion: run the factorial at σ² 1 with 60 per cell (1.75 GPU-h, `note:1481`). At 0.25, run only factor D on 30 seeds (about 0.12 GPU-h).
- If the Gate 5 time step equals the factorial's "dt/2" level, reuse the Gate 5 Par2 runs as the all-plus cell (saves about 0.23 GPU-h per amplitude).
- This changes G7 and needs an owner decision.

## 6. Wall-clock cutoff and runtime–outcome correlation

**Is a fixed-n test valid when n comes from a cutoff?**
- If n is independent of the outcomes, the test is valid conditional on n.
- With runtime–outcome correlation, a sequential prefix is a stopping-time sample: only the in-flight unit is length-biased (Wald's identity). The bias of the mean is O(W/n) for W concurrent streams.
- The dangers are "keep whatever completed" with many streams, timeouts, and SF-39-style arm-by-arm parts (`run_sf39.sh`), which truncate whole arms and break the joint allocation.

**What the data show** [C16, C17]. No reduced table has timing columns. Timings exist only in the pilot `RUN.json` files and in the SF-39 per-arm σ² 4 raw logs.
- **SF-39, σ² 4 arms:** 160 realization rows, about 120 distinct fields. Pooled Spearman correlations (confidence intervals too narrow, because the ladder rungs share fields):

  | quantity vs alpha | alpha_TP | alpha_LAM | alpha_TAM |
  |---|---|---|---|
  | runtime | +0.19 [0.03, 0.34] | −0.10 [−0.26, 0.06] | −0.05 [−0.21, 0.12] |
  | PCG iterations | +0.09 | −0.12 | −0.05 |
  | tracker work (`nfev`) | — | — | +0.21 [0.05, 0.36] |

- The runtime tail comes from fallback solves: 2000 PCG iterations, 234 s against about 20 s (seeds 5023 and 7205). Their alpha ranks out of 30 are 27/21/28 and 9/10/24, so there is no consistent direction.
- **SF-42 pilots:** runtime vs alpha_LP is −0.02 [−0.28, 0.23]. PCG iterations vs alpha_LP is −0.42 [−0.66, −0.10], from 9 groups that share fields, so it is overstated. The σ² 4 outlier field 420836 had an ordinary runtime (26.1 s, 50 PCG iterations).

**Pre-registered rules:**
- **Seed-order prefix rule.** Each arm analyzes its longest complete prefix of the pre-registered seed order. A replacement seed takes the failed seed's position. Completed units beyond the prefix are reported but not analyzed.
- **Interleaving.** Order all units by k/N_a (proportional interleaving). Keep paired units adjacent: the Gate 5 consumer and Par2 runs of a seed, ladder rungs, and h_max passes.
- **Timeouts** count as failures under the spec's failure rules (`spec:122-124`).

## 7. Blinded variance re-estimation under a cutoff

- **Trigger by count, not by time:** run it when the prefix first reaches ⌈0.25·N_a⌉.
- **Inputs for Gate 5:** only the variance of d_r = L − cP at c = 1.1 and 1/1.1. It is location-free and needs no planning means, which is stricter than `blinded_gate5`'s CV route (`blinded.py:74-91`).
- **Rule:** N′ = clamp(n_req(80 % upper bound of the SD), N, 2N). Realized n = min(N′, prefix reached at the cutoff).
- **Validity:**
  - Under normality the SD is ancillary for the mean. The small internal-pilot inflation is documented by Wittes & Brittain 1990 (Stat. Med. 9:65) and Kieser & Friede 2003 (Stat. Med. 22:3571). Friede & Kieser 2003 (Stat. Med. 22:995, verify) covers non-inferiority and equivalence.
  - Under skew, Cov(x̄, s²) = μ₃/n, so the "blinded" SD carries information about location. Block C simulates this two-stage rule.
- **Interaction with the priority order:** add increments after the planned N of the primary arms and before the secondary arms, as pre-registered. When the cutoff binds below N, report the output but it changes nothing.

## 8. Holm with possibly fewer than 7 hypotheses

- Keep the family fixed at m = 7.
- Any hypothesis that cannot be tested (arm below its floor, no qualified procedure, halted arm) enters Holm with p = 1. This is valid whatever the cause, because Holm needs only valid p-values (`multiplicity.py:12-33`).
- Dropping the hypothesis and using m′ < 7 is valid only if the drop event is independent of all p-values. Do not rely on that.
- If (f2) is adopted, the same p = 1 convention applies inside each family.
- State in the pre-registration that G5.3 allows true miss rates up to 1.25× the tail, so FWER ≤ 0.0625 in the worst case.

## Commands

All are inline `python3 -B -c`. Commands that import mcstat ran from `scripts/` with `PYTHONDONTWRITEBYTECODE=1`.

- **[C1]** `stress_moments.json`: envelope and `series[]`.
- **[C2]** Pilot rows filtered on `family=plane`, `weights=unif`, `basis=pos`, `set=perplane`, column `alpha_LP`.
  - Consumer: `J3/reduced/c_s*.csv`. Par2: `J3B/reduced/p_s*_h3.csv`, plus `J3/reduced/p_s025_1040x240.csv`. Paired by `r`.
  - `scipy.stats.skew` and `kurtosis` with `bias=False`. Bootstrap B = 20 000.
- **[C3]** Per-realization listing.
- **[C4]** The same filters with `alpha_TP` in the pos and disp bases; J2 uses M128 at σ² 0.25 and 1, M100 at 2.25 and 4.
- **[C5]** `covdist.get_dist("stress")`, 2×10⁵ samples at n = 8 and 24.
- **[C6]–[C9]** `coverage/summary.json`: `cells[].seconds` and `cells[].rows`; [C9] re-judges the halved tails against 2× the tail.
- **[C10]** Prototype:
  - `T = √n·x̄/s`
  - Johnson: `(x̄ + m3/(6 s² n) + m3/(3 s⁴)·x̄²)/(s/√n)`
  - Hall: `T + cT² + c²T³/3 + γ/(6√n)` with `c = γ/(3√n)` and `γ = m3/s³`
  - Thresholds ±`t_{a_j, n−1}`.
- **[C11]** `coverage.fail_threshold` and `nominal_fail_probability`.
- **[C12]** and **[C19]** `gate5.gate5_conservative_inputs`, `gate5_sample_size` and `tost_power` on `j3_pairing.json` / `j3b_pairing.json`.
- **[C13]** Bootstrap and leave-one-out of CV and sd(e).
- **[C14]** GMR shift.
- **[C15]** Noncentral t.
- **[C16]** SF-39 `raw/campaign/{pilot,extend,width}/logs/s4_*.log`: FIELD, B_SOLVER, TSTATUS `nfev`, TIMING lines, matched to `analysis/reduced/s4_*.csv` flux rows.
- **[C17]** J2 and J3 `RUN.json` `realizations[]`.
- **[C18]** `norm.isf`.
- **[C20]** GPU-h arithmetic.

Not checked: the citations marked "verify", Gate 4 pipeline cells at SF-43's final N, and the Pollock-bridge outputs of J3. The Hall/Johnson numbers come from my own prototype, not the library. If the production grid at σ² 0.25 changes (the 3× PCG rule tripped, Finding (f)), the twin-series reuse at 0.25 no longer holds.

### Critical Files for Implementation
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/mcstat/gate5.py
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/mcstat/covproc.py
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/coverage/summary.json
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/pilots/j3b_pairing.json
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/decisions/2026-10-29-sf39-remediation-sf42-sf43.md