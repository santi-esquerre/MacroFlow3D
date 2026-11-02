I ran every task. No files were written: the read-only rule for this session overrides the scratch-directory permission, so all code ran inline through `python3 - <<EOF` from the SF-39 `reduced/` directory or the SF-42 `scripts/` directory.

`ROOT=/tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check`

## Bottom line

1. **σ² 1 and 2.25 are safe; σ²=4 at N=30 depends on one number nobody has measured.** That number is the per-seed SD of the λ/h 8→16 difference on the production box. The evidence (point 2) says it is at the particle-noise floor, and then P(RL>0) = 1.00. If it instead grows as h², P(RL>0 at 0.05/7) is 0.69, and 0.14–0.40 if the production width is also 5–10 % short.
2. **G5.4's observed-order branch almost never fires.** It was used in 0 % of replicates at 2.25 and 4, 1–10 % at σ²=1, and 51–100 % only at 0.25. On the production box at σ² ≥ 1, particle-sampling noise explains the per-seed grid differences (variance ratio 0.72–1.07, [C5]). The mean differences are therefore unresolved, p_L = −∞, and E_h falls back to 3 × range, which is a noise floor.
3. **At 2.25, the 8/16/20 ladder in SF-42's projection is the worst choice.** It can never produce an order, because the ratio is not constant, and its fallback multiplies the range by 3/(1.25−1) = 12. The 4/8/16 ladder gives E_h 0.19–0.32 μ against 0.31 μ (S2: 0.12–0.25 against 0.46 μ), and is about 7× cheaper.
4. **The displacement basis makes Gate 4 cheap and strong.** P(RL>0) = 1.00 at every amplitude and scenario in my runs except one order-1 corner at σ²=4 (0.69–0.90), for 14–25 GPU-h of Gate-4 arms instead of about 84–87.
5. **At σ²=0.25 in the position basis, no arm is informative.** P(RL>0) = 0, and the class is unresolved (bounded) with α_T ≤ about 1.6e-3 λ. In the displacement basis RL > 0 with P ≈ 1, but G5.7 has no label for that result.
6. **Proposed "numerical" fraction: f = 0.5**, one-sided 95 % UCL. It is reachable everywhere except position basis at 0.25, and never fires when the truth is physical.

## 0. Method and traceability

All Gate 4 numbers come from mcstat's own classes on synthetic per-seed arms: `StudentMean`, `LadderBound(method='student')`, `e_w` / `e_int` / `e_interaction(method='student')`, `RobustLimit.lower` / `p_value` / `component_bounds`, and `multiplicity.holm`. The only re-implementation is a forced-p_L ladder for task 4. It matches `LadderBound.details()` (`order_branch_value` at p_min and `fallback_value`) with a maximum difference of 0.0. Each cell uses R = 400 replicates and B = 4000.

| id | computation |
|---|---|
| C1 | SF-43 estimand per realization: ½ × OLS slope of `sigma_T2/ell²` against d = (x1−x0)/ell, over d in [25, 94], from `reduced/sigma_series_<set>.csv` (ell and x0 from `<set>.csv`). It reproduces SF-42 R3 exactly (σ²=1: mean 0.006236936508, SD 0.01103110649; `verification/noise_summary.csv`). |
| C2 | Paired production − twin differences, narrow-ladder D_c and D_f, Student 90 % intervals. Skewness of production values is 0.15–0.39, excess kurtosis −0.6 to +0.3, so normal generation is adequate. |
| C3 | Width contrasts, Welch 90 % intervals. |
| C4 | h_max pass differences: SF-39 two-plane `tp.alpha_TP` (flux), `<amp>_A_hmax{8,16,32}.csv`. |
| C5 | var(production − twin of `tp.alpha_TP`) against the mean of (`split.alpha_TP.A` − `.B`)²/4 over the production and twin rows. |
| C6 | Main pipeline simulation (scenarios below). |
| C7 | Forced p_L, and order-1 / order-2 cases with large resolved bias. |
| C8 | Displacement-basis joint-allocation grid search. Planning inputs from `sizing.sd_upper` / `mean_lower` at 80 %. |
| C9 | "numerical"-class Monte Carlo (normal approximation). |
| C10 | `sizing.gate4_power` compared with C6. |

**Scenario parameters.** The level errors are order-2, anchored on the SF-39 production − twin means (D_tw).

| σ² | ladder (position of the production level) | level errors coarse → fine (S1) | per-seed noise |
|---|---|---|---|
| 0.25 | 5/10/20 (middle) | (−1.6e-5, −4.0e-6, −1.0e-6) | per seed C·h²: SD(5→10) 3.27e-5, SD(10→20) 8.2e-6, correlation 0.99985 |
| 1 | 5/10/20 (middle) | (−5.4e-5, −1.3e-5, −3.3e-6) | SD 3.55e-4 / 8.9e-5, correlation 0.990 |
| 2.25 | 4/8/16 (finest) or 8/16/20 (middle) | (1.45e-3, 3.6e-4, 9.1e-5) / (3.6e-4, 9.1e-5, 5.8e-5) | SD(8→16) 1.18e-3; two noise models, see below |
| 4 | 8/16/32 (finest) | (1.79e-3, 4.5e-4, 1.1e-4) | SD(16→32) 2.20e-3; two noise models |

- **Noise models at 2.25 and 4.**
  - **"nf" (noise floor):** independent per-level noise, so all consecutive differences have the same SD and correlation −0.5. The data support this model: narrow-ladder correlation at σ²=4 is −0.54, SD ratio 1.35, and C5 gives a ratio of 0.72.
  - **"o2" (pessimistic):** the coarser difference has 4× the SD, correlation 0.
- **Width:** each arm's SD equals the production SD; no interaction in S1.
- **S1w5 / S1w10:** the production width is 5 % / 10 % short (production biased low).
- **S2:** non-monotone level errors whose range equals the adverse SF-39 90 % limit of D_tw (2.5e-5 / 1.8e-4 / 6.7e-4 / 1.19e-3); the order branch cannot be used.
- **S3:** p_L forced, or a true order-1 bias at production.

## 1. Budget components from SF-39 data (SF-43 estimand, position basis; C1–C5)

Percentages are relative to the planning mean μ. At σ²=0.25 the position-basis mean is not resolved, so μ is the displacement mean.

| σ² | μ pos (n) / disp | production SD pos / disp | D_tw = production − coarse twin: pair, mean [90 % CI], per-seed SD | narrow ladder D_c / D_f: mean (SD) | per-seed corr(D_c, D_f); per-seed order (median); same-sign seeds | width contrast in SF-43 estimand | h/16 − h/8 mean (SD) | var(D_tw) / particle variance |
|---|---|---|---|---|---|---|---|---|
| 0.25 | −1.88e-4 (25) / 5.09e-4 | 4.48e-3 / 1.21e-4 | 5→10: 1.21e-5 [−5.7e-7, 2.47e-5] = +2.4 %; SD 3.27e-5 (6.4 %) | −4.5e-6 (4.39e-5) / −8.2e-7 (1.10e-5) | 0.99985; 2.01; 20/20 | 25.6 vs 12.8 λ: −8.1e-4 [−2.6e-3, 9.5e-4]; SE = 2.05 μ | 3.9e-13 (1.4e-12) | 21.9 |
| 1 | 6.237e-3 (50) / 7.217e-3 | 1.103e-2 / 1.121e-3 | 5→10: 4.0e-5 [−9.7e-5, 1.77e-4] = +0.6 %; SD 3.55e-4 (5.7 %) | 4.0e-6 (3.77e-4) / 9.1e-6 (9.5e-5) | 0.990; 1.88; 20/20 | 25.6 vs 12.8: −9.8 % [−65, +45] | −1.8e-11 (1.5e-10) | 1.07 |
| 2.25 | 2.833e-2 (62) / 2.946e-2 | 1.514e-2 / 4.07e-3 | 8→16: −2.72e-4 [−6.7e-4, 1.2e-4] = −1.0 %; SD 1.18e-3 (4.2 %) | 1.1e-4 (7.1e-4) / −1.9e-5 (1.9e-4); non-monotone | −0.08; 1.81; 12/20 | 24 vs 8 λ: +34 % [18, 50] (8 λ is narrower than the SF-43 source) | −1.9e-8 (1.3e-6) | 0.78 |
| 4 | 6.648e-2 (71) / 6.538e-2 | 1.918e-2 / 1.688e-2 | 16→32: −3.36e-4 [−1.19e-3, 5.1e-4] = −0.5 %; SD 2.20e-3 (3.3 %) | 1.9e-4 (4.6e-4) / −1.0e-4 (3.4e-4); non-monotone | −0.54; 0.26; 5/20 | 24 vs 12 λ (F10 sets): −0.5 % [−11.3, +10.3]; 12 vs 12 replication −4.5 % [−16, +7] | −3.5e-8 (9.7e-8) | 0.72 |

What this means for each component:

- **E_int** (C4) is at most 1e-6 relative at every amplitude. It is negligible, but it still costs one of the K tail splits.
- **D_tw** is unresolved everywhere: 0.5–2.4 % of μ, with per-seed SD 3–6 % of μ.
- **Disp-basis paired SDs** can't be computed from the repository, because the per-particle arrays stay on the host. I assumed they equal the position-basis values. That is conservative: the J2 particle components are 8.7× (2.25) and 4.4× (4) smaller in variance in the displacement basis (SF-42 note lines 1199/1202 and 1208/1211).
- **Narrow sections** are 12.8 / 12.8 / 8 / 4 λ wide. SF-39's boxes at σ² ≤ 2.25 are 205–208 λ long, so outlet effects at the SF-43 planes (5 λ from the outlet) are not represented.

## 2. Gate 4 simulation (C6, C7)

### Designs and GPU-h of the Gate-4 arms

Costs use the SF-42 per-100-realization table (note lines 1444–1483, `M_plan` basis). The admissible M is about 13 % more; the anchored values are about 25 % more.

| design | basis | σ²=0.25 | 1 | 2.25 | 4 | total |
|---|---|---|---|---|---|---|
| D-spec (SF-42 rows, lines 1407–1413) | pos | N 1100, ladder 300, width arm 1100, h_max 50: 30.5 | 1650 / 300 / 1650 / 50: 42.3 | 110, ladder 8/16/20 at 110, 2×2 at 110: 10.8 (4/8/16: 7.5) | 30 / 30 / 2×2 at 30 / 30: 3.5 | 87.1 (83.8 with 4/8/16) |
| D-disp | disp | 100 / 100 / 100 / 30: 5.1 | 5.3 | 4/8/16: 6.4 | 8.4 | 25.2 |
| D-lean | disp | N 100, arms 50: 2.9 | 3.0 | 4.5 | N 50, ladder 50, width 30: 4.3 | 14.7 |
| D-alloc (C8; P ≥ 0.95 under S1 and S1w10 at planning inputs) | disp | N50 / l30 / w30: 1.8 | N30 / l30 / w30: 1.8 | N75 / l50 / w30: 3.3 | N100 / l100 / w50: 7.0 | 13.9 |

### P(RL > 0) at a = 0.05/7 (worst Holm level)

Entries at 2.25 and 4 are "nf / o2". The last two columns are for S1.

| σ² | design | S1 | S1w5 | S1w10 | S2 | E[RL]/μ | dominant mean B_k/μ |
|---|---|---|---|---|---|---|---|
| 0.25 | D-spec | 0.000 | 0.000 | 0.000 | 0.000 | −1.16 | E_W 1.43, sampling 0.72 |
| 0.25 | D-disp | 1.000 | 1.000 | 1.000 | 1.000 | 0.76 | E_W 0.13, sampling 0.07 |
| 0.25 | D-lean | 1.000 | 1.000 | 1.000 | 1.000 | 0.66 | E_W 0.16, E_h 0.11 |
| 1 | D-spec | 1.000 | 1.000 | 1.000 | 1.000 | 0.58 | E_W 0.24, sampling 0.12, E_h 0.06 |
| 1 | D-disp | 1.000 | 1.000 | 1.000 | 1.000 | 0.79 | E_h 0.08, E_W 0.08 |
| 1 | D-lean (pos, arms 50) | 0.030 | 0.015 | 0.005 | 0.013 | −0.27 | E_W 1.01 |
| 2.25 | D-spec, 8/16/20 | 0.993 / 1.000 | 0.975 / 0.995 | 0.838 / 0.907 | 0.810 / 0.882 | 0.24 / 0.26 | E_h 0.31 (×12 fallback), E_W 0.29 |
| 2.25 | D-spec, 4/8/16 | 1.000 / 0.998 | 1.000 / 0.983 | 0.998 / 0.875 | 1.000 / 1.000 | 0.38 / 0.24 | E_W 0.28, E_h 0.19 / 0.32 |
| 2.25 | D-disp, 4/8/16 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.69 / 0.56 | E_h 0.18 / 0.31, E_W 0.08 |
| 2.25 | D-lean (pos, arms 50) | 0.48 / 0.56 | 0.29 / 0.40 | 0.11 / 0.19 | 0.04 / 0.11 | −0.01 | E_W 0.43, E_h 0.41 |
| 4 | D-spec, N 30 | **1.000 / 0.688** | 1.000 / **0.395** | 0.988 / **0.138** | 1.000 / 0.630 | 0.36 / 0.04 | E_W 0.30; E_h 0.15 / 0.37; E_W×h 0.04 / 0.14 |
| 4 | D-disp, N 100 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.66 / 0.48 | E_W 0.14; E_h 0.11 / 0.23 |
| 4 | D-lean, N 50 | 1.000 / 0.975 | 1.000 / 0.910 | 1.000 / 0.660 | 1.000 / 0.983 | 0.47 / 0.19 | E_W 0.27; E_h 0.13 / 0.30 |

S3 at σ²=4 (C7):
- **Forced p_L of 0.5 to 2:** P = 1.000 in every design and noise model.
- **True order-1 bias of −5 % / −10 % at production:** D-spec gives P = 0.35 / 0.77 (nf) and 0.01 / 0.10 (o2). D-disp gives 1.00 / 1.00 (nf) and 0.90 / 0.69 (o2).

### Holm

Sign p-values for σ² 1 / 2.25 / 4 are simulated; the four Gate 5 p-values are fixed at 1 or 0. The three Gate 5 configurations are:
- **none:** no Gate 5 test passes;
- **J3B-like:** 0.25, 1 and 2.25 pass and 4 fails (the J3B pilot ratio at σ²=4 was 1.137 on 8 pairs, note line 1286);
- **all:** all four pass.

| design / scenario | G5 none | J3B-like | G5 all |
|---|---|---|---|
| D-spec S1 nf | 1.00 / 0.998 / 1.00 | 1 / 1 / 1 | 1 / 1 / 1 |
| D-spec S1 o2 | 1 / 1 / **0.777** | 1 / 1 / 0.925 | 1 / 1 / 0.985 |
| D-spec S1w10 o2 | 1 / 0.922 / **0.210** | 1 / 0.963 / 0.492 | 1 / 0.980 / 0.690 |
| D-spec S2 nf / o2 | 1 / 0.855 / 1.00 and 1 / 0.915 / 0.723 | 1 / 0.955 / 1.00 and 1 / 0.973 / 0.895 | 1 / 0.988 / 1 and 1 / 0.993 / 0.973 |
| D-disp, every scenario | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 |
| D-lean S1w10 o2 | 1 / 1 / 0.733 | 1 / 1 / 0.887 | 1 / 1 / 0.950 |

## 3. Minimum detectable effect (C6)

RL shifts one-for-one with the true α while biases stay fixed, so MDE = μ − q₀.₂(RL). Values are MDE/μ at a = 0.05/7 / 0.05/3, in S1, as nf / o2 at 2.25 and 4.

| σ² | D-spec | D-disp | D-lean | D-alloc (planning inputs, at 0.05/7) |
|---|---|---|---|---|
| 0.25 | 2.38 / 2.20 (1.2e-3 λ) | 0.28 / 0.23 (1.4e-4 λ) | 0.42 / 0.37 | 0.72 |
| 1 | 0.46 / 0.42 (2.9e-3 λ) | 0.23 / 0.21 (1.7e-3 λ) | 0.29 / 0.27 | 0.50 |
| 2.25 | 8/16/20: 0.82 / 0.77 and 0.81 / 0.76; 4/8/16: 0.68 / 0.64 and 0.82 / 0.77 | 0.33 / 0.32 and 0.48 / 0.46 (9.7e-3 to 1.4e-2 λ) | 0.41 / 0.39 and 0.62 / 0.59 | 0.75 |
| 4 | 0.70 / 0.65 and **1.04 / 0.96** (4.7e-2 to 6.9e-2 λ) | 0.38 / 0.35 and 0.56 / 0.53 (2.5e-2 to 3.7e-2 λ) | 0.59 / 0.55 and 0.88 / 0.82 | 0.75 |

With the G5.2 inflation (SDs at their 80 % upper bound), the noise part grows by a factor of about 1.08–1.18.

## 4. E_h with production at the finest level (C7)

**What p_L to expect.**
- The FV flux error has order 2.00 at every σ² (SF-42 note lines 604 and 613–622), and the MMS shows order 2 at the Dirichlet faces (lines 1619–1621).
- The estimand's own differences at σ²=4 are noise: per-seed order 0.26 (IQR −0.52 to 1.67), same sign in 5/20 seeds.
- So G5.4's p_L is −∞ (fallback) with probability about 1 at 2.25 and 4: the order branch was used in 0 % of S1/S2 replicates.
- The order branch only appears when the bias at the coarse levels is large: true order 2 with production bias ≥ 2 % of μ gives p_L of 1.3–1.9. True order 1 with 5–10 % production bias gives p_L of 0.2–0.8, mostly below 0.5, so it falls back anyway.

E_h/μ, mean [10 %, 90 %] at a = 0.05/7:

| case | p_L 0.5 | 1 | 1.5 | 2 | fallback nf | fallback o2 | P(RL>0) at 0.05/7 / 0.05/3 |
|---|---|---|---|---|---|---|---|
| σ²=4, D-spec N 30 | 0.086 [0.067, 0.109] | 0.036 | 0.020 | 0.012 | 0.148 [0.125, 0.171] | 0.374 [0.29, 0.47] | 1.000 for all forced p_L and fallback nf; **0.640 / 0.860** for fallback o2 |
| σ²=4, D-disp N 100 | 0.050 | 0.021 | 0.011 | 0.007 | 0.113 | 0.227 | 1.000 everywhere |
| σ²=4, D-lean N 50 | 0.068 | 0.028 | 0.015 | 0.009 | 0.130 | 0.296 | fallback o2: 0.985 |
| σ²=2.25, 4/8/16, D-spec N 110 | 0.069 | 0.029 | 0.016 | 0.010 | 0.186 | 0.319 | ≥ 0.995 |
| σ²=2.25, 8/16/20 | no order possible | – | – | – | 0.31 (S2: 0.46) | 0.30 (S2: 0.45) | S2: 0.81 / 0.88 |

At α_TP ≈ 0.066 λ, RL > 0 is attainable for any p_L the order branch could return. The only threat is the fallback under the o2 noise model.

## 5. σ² = 0.25

**What each class means here.**
- **physical (restricted):** not available; 0.25 is outside the primary family.
- **numerical:** the production interval excludes 0 and the corrected estimate's UCL is below f × the production mean.
- **unresolved (bounded):** report α_T ≤ U + ΣB_k.
- **unresolved:** estimator discordance. This is practically unreachable, because α_TAP equals α_TP to within 4e-4 relative (SF-42 note line 1037).

**Expected class:** unresolved (bounded) with P about 1 in both bases. P(numerical | physical truth) = 0.00 (C9). The bound E[U + ΣB] at 0.05/7 is:
- position basis D-spec: 3.16 μ ≈ 1.6e-3 λ, about what SF-39 already gave (1.9e-3);
- displacement basis D-disp: 1.24 μ ≈ 6.3e-4 λ, with RL > 0 (P = 1.000, RL ≈ 3.9e-4 λ). That is a resolved positive sign that G5.7 cannot label.

**Which arms are informative.**
- In the position basis, none changes the class.
  - The 1040×480×480 width arm (1100 seeds, 16 GPU-h; note line 1458) still leaves E_W = 1.43 μ.
  - Bringing E_W below 0.2 μ would take about 5.7e4 seeds per arm.
  - With 50 seeds per arm, E_W is 5.0 μ. So the 16 GPU-h only tighten U.
  - E_h is 0.015 μ (the order branch works here: per-seed differences follow C·h² cleanly, p_L ≈ 1.95). E_int is about 0.
- In the displacement basis, the width arm at 30–100 seeds costs 0.4–1.5 GPU-h and is the dominant term (0.13–0.16 μ).
- Separately, SF-42 found that the 1040×240×240 grid at 0.25 trips its pre-stated 3× PCG-iteration rule (note lines 1658–1660), so the grid itself needs a pre-registration decision.

## 6. Pre-mortem

Probabilities are at the worst Holm level, D-spec / D-disp.

| σ² | physical (restricted) | numerical | unresolved (bounded) | unresolved | what each outcome would teach |
|---|---|---|---|---|---|
| 1 | 1.00 / 1.00 in S1–S2; only 0.03 with position-basis 50-seed arms | about 0 | about 0 | operational only | Physical: RL ≈ 0.6–0.8 μ (≈ 4–6e-3 λ). Unresolved (bounded): a design failure (E_W noise), not physics. |
| 2.25 | 8/16/20: 0.99–1.0 (S1), 0.84–0.91 (S1w10), 0.81–0.88 (S2); 4/8/16: 0.88–1.0; D-disp 1.00 | about 0 | 0–0.19 | operational | Unresolved (bounded) would point to the ×12 fallback or a 24→48 λ width effect of ≥ 10 %, which answers W3. |
| 4 | nf: 0.99–1.00; o2: 0.69 (S1), 0.40 (S1w5), 0.14 (S1w10), 0.63 (S2); S3 order-1: 0.01–0.77. D-disp: 1.00, except S3 o2 at 0.69–0.90 | about 0 | complement | operational | The B_k split identifies which of three causes: 8→16 noise that grows as h², a 12→24 λ width effect, or an order-1 bias. |
| 0.25 | not available | about 0 (physical truth); 1.00 if the signal were an artifact (disp) | about 1 | about 0 | The upper bound, plus a descriptive RL in the displacement basis. |

The "numerical" class is only reached under an artifact truth (true corrected value ≤ 0.25 × production): P ≥ 0.90 with f = 0.5, except position-basis 0.25.

## 7. The "numerical" fraction (C9)

Proposal: **f = 0.5, one-sided 95 % UCL, with the production 90 % interval required to exclude 0.** The meaning is "at least half of the production signal is a discretization or width artifact, at 95 %". Table entries are P(numerical) for true corrected/production ratio g = 0 / 0.25 / 1:

| design | SE of corrected estimate / μ | smallest f giving P ≥ 0.8 at g = 0 | f = 0.25 | f = 0.5 | f = 0.75 |
|---|---|---|---|---|---|
| D-spec 0.25 | 0.265 | 0.72 | 0.25 / 0.05 / 0 | **0.58** / 0.27 / 0 | 0.83 / 0.58 / 0.02 |
| D-spec 1 / 2.25 / 4 | 0.044 / 0.088 / 0.091 | 0.11 / 0.22 / 0.23 | 0.88–1.00 / ~0.04 / 0 | 1.00 / 0.90–1.00 / 0 | 1.00 / 1.00 / 0 |
| D-disp (all σ²) | 0.016–0.045 | 0.04–0.11 | 1.00 / ~0.05 / 0 | 1.00 / 1.00 / 0 | 1.00 / 1.00 / 0 |

Why 0.5:
- f = 0.25 is borderline at 2.25 and 4 in the position basis.
- f = 0.75 would label a signal that is 60 % physical as "numerical".
- f = 0.5 is reachable everywhere except position-basis 0.25.

## 8. Ill-defined points in G5.4 / G5.7, and where mcstat differs

1. **8/16/20 at 2.25.** `budget.py:272` requires a constant ratio for an order to exist. The fallback divides the full range by (r_min − 1) = 0.25 (`budget.py:302`), a factor of 12. G5.4's "range over all pairs ÷ (r_min − 1)" mixes the coarsest pair with the finest ratio. The coverage pipeline never simulated a ladder with mixed ratios (`covproc.py:67`, levels 5/10/20).
2. **The fallback is the operative rule, not the exception** (section 4). E_h ≈ 3·t_{t/3}·SE(range) + 3·|range|. Particle count M would lower that floor, but M is fixed across the arms of an amplitude.
3. **Order branch near p_min.** At p_L = 0.5 the factor is 4.27 (a finer level exists) or 3.02 (production finest), at least as large as the fallback. The rule is therefore not monotone across branches; mcstat only enforces monotonicity across tails (`budget.py:42–48`). p_min / p_max are mcstat defaults (0.5 / 4.0, `budget.py:217`) and must be pre-registered.
4. **Bootstrap wording.** G5.4 says the B_k come from a joint bootstrap over seeds. mcstat's qualified Student bounds are analytic, and each component draws its own resamples. Validity rests on the union bound, which is fine, but the pre-registration should say so. The studentized fallback also takes a Bonferroni t/3 over the range pairs (`budget.py:290–293`), which is stricter than G5.4's single UCB at t.
5. **A width deficit counts twice.** The production mean is biased by δ_W, and E_W ≥ |δ_W| on top of that, so RL loses about 2|δ_W| (compare S1 with S1w10).
6. **p-value lattice.** The p-value is reported on a lattice up to 1.16 % above the true value (`budget.py:50–55`). Pre-register `RL(Holm levels) > 0` instead, so threshold decisions don't flip at the boundary.
7. **`sizing.gate4_power` is optimistic for the fallback.** It ignores the folding of |Δ̂|, the t/3 split and the max over pairs (`sizing.py:98–110`): 0.875 against a simulated 0.69 for σ²=4 D-spec o2 (C10). Size the joint allocation by simulation.
8. **Coverage at small K=3 sizes.** The K=3 coverage cells exist only at N = 100 / 300 / 1000 (`coverage.py:652`). N = 30–50 at K=3, as in D-alloc and D-lean, is not qualified; re-run coverage at the final N.
9. **G5.7 "numerical" is undefined when the order is unresolved,** which is the typical case. The levels of "interval excludes 0" and of the UCL are not specified.
10. **G5.7 "unresolved (bounded)" has no entry condition** and no stated levels for U or B_k. "Estimator discordance" is undefined for SF-43, and the intersection-union over α_TP / α_TAP is effectively a single test.
11. **No label for a robustly positive sign at 0.25** (section 5).
12. **Basis.** G4 defines α_TP on crossing positions; the displacement variant drops the slope of cov(x0, Δ). The two means agree within the position-basis SE at every amplitude (differences −7e-4, −1e-3, −1.1e-3, +1.1e-3), but the primary basis must be fixed before any data.

### Critical Files for Implementation
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/mcstat/budget.py
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/mcstat/sizing.py
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/decisions/2026-10-29-sf39-remediation-sf42-sf43.md
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/analysis/reduced
- /tmp/claude-1000/-home-sesquerre-Projects-MacroFlow3D/0b813f1d-3625-4dbd-9e7d-1066997d5f71/scratchpad/master-check/docs/plans/active/lester-eq14/increments/SF-43-confirmatory-macrodispersion-study.md

Not checked: Gate 5 p-values are fixed configurations rather than simulated TOSTs; the power runs use normal data, not the stress family; displacement-basis ladder SDs are assumed equal to position-basis; costs are on the `M_plan` basis.
Main risk: σ²=4 at N=30 in the position basis depends on the unmeasured 8→16 per-seed SD on the production box.