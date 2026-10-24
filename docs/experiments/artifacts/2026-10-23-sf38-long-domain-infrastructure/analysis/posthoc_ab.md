# SF-38 post-hoc readings of the medium A/B evidence (N9)

**Post-hoc, NOT pre-registered.** Derived by `scripts/posthoc_ab.py` from `raw/sf38_campaign/ab/` (and `raw/sf38_campaign/prodgrid/B2.log`) after FINAL_AUDIT, at the owner's request. Exploratory (forking-paths risk): none of these readings changes any pre-registered verdict, threshold or analysis file, and the medium decision stands as the rule's outcome. No `alpha_L` / `alpha_T` statement; `S_T` and `S_tau` appear only as the rule's statistics.

## P1 Pairing efficiency (derived, post-hoc)

Per amplitude and primary statistic, over the ten paired realizations (`primary.json`): `mean_a`, `mean_b`, sample sd (n-1) `sd_a`, `sd_b`, Pearson `corr(a,b)`, `sd(D)` with `D_r = P_a,r - P_b,r`, and `eff = sd(D) / sqrt(sd_a^2 + sd_b^2)` (1 = no gain from pairing, i.e. as for independent samples with equal variances; 0 = perfect pairing). Raw scale of each statistic (S_T in box units). `eff` is on the raw difference, not on the relative scale of the rule's delta-method CI.

| amp | stat | mean_a | mean_b | sd_a | sd_b | corr(a,b) | sd(D) | eff = sd(D)/sqrt(sd_a^2+sd_b^2) |
|---|---|---|---|---|---|---|---|---|
| 1 | S_T | 3.198e-05 | 3.305e-05 | 0.0007163 | 0.0007275 | 1.000 | 2.259e-05 | 0.022 |
| 1 | S_tau | 0.0819 | 0.08089 | 0.009309 | 0.008318 | 0.932 | 0.003386 | 0.271 |
| 1 | V1 | 0.6151 | 0.6151 | 0.004526 | 0.004516 | 1.000 | 4.505e-05 | 0.007 |
| 1 | theta_2 | 1.144e-05 | 9.932e-05 | 0.00026 | 0.001286 | 0.324 | 0.001226 | 0.935 |
| 1 | theta_3 | 4.632e-05 | -2.102e-05 | 0.0001727 | 0.001742 | 0.646 | 0.001636 | 0.934 |
| 4 | S_T | 0.01631 | 0.01739 | 0.00559 | 0.0047 | 0.691 | 0.004128 | 0.565 |
| 4 | S_tau | 2.38 | 2.21 | 1.019 | 0.5843 | 0.815 | 0.6399 | 0.545 |
| 4 | V1 | 3.169 | 3.171 | 0.2137 | 0.2134 | 1.000 | 0.003156 | 0.010 |
| 4 | theta_2 | 0.000318 | 0.006189 | 0.00682 | 0.01655 | 0.752 | 0.01228 | 0.686 |
| 4 | theta_3 | -0.0004618 | -0.003709 | 0.005843 | 0.01552 | -0.309 | 0.01819 | 1.097 |

## P2 Point estimates against the margins (derived from the `C_STAT` lines, post-hoc)

`est` is `R = ln(Pbar_a/Pbar_b)` for `test=ratio` and the paired difference for `test=diff|floor`; `margin` on the same scale; `|est| > margin` flags a point estimate outside the equivalence window (the rule classifies by the 90 % CI, not by the point estimate).

| amp | stat | test | est | margin | |est| > margin | 90 % CI | class |
|---|---|---|---|---|---|---|---|
| 1 | S_T | floor | -1.08e-06 | 9.766e-05 | no | [-1.417e-05, 1.201e-05] | equivalent |
| 1 | S_tau | ratio | 0.01246 | 0.02469 | no | [-0.01144, 0.03636] | inconclusive |
| 1 | V1 | ratio | -0.0001057 | 0.04879 | no | [-0.0001482, -6.313e-05] | equivalent |
| 1 | theta_2 | diff | -8.787e-05 | 0.005 | no | [-0.0007988, 0.000623] | equivalent |
| 1 | theta_3 | diff | 6.734e-05 | 0.005 | no | [-0.0008808, 0.001015] | equivalent |
| 4 | S_T | ratio | -0.06383 | 0.04879 | yes | [-0.2087, 0.08109] | inconclusive |
| 4 | S_tau | ratio | 0.07412 | 0.02469 | yes | [-0.07783, 0.2261] | inconclusive |
| 4 | V1 | ratio | -0.0004837 | 0.04879 | no | [-0.001063, 9.555e-05] | equivalent |
| 4 | theta_2 | diff | -0.005872 | 0.005 | yes | [-0.01299, 0.001244] | inconclusive |
| 4 | theta_3 | diff | 0.003248 | 0.005 | no | [-0.007295, 0.01379] | inconclusive |

Plain statement (derived): 3 of the 5 inconclusive statistics have their point estimate outside the margin (3 of 4 at `sigma^2 = 4`; none at `sigma^2 = 1`: the single inconclusive statistic there has `|est| < margin`).

Rough sample size for the inconclusive statistics with `|est| < margin` (labelled rough: `N ~ 10 (h10 / (margin - |est|))^2` with `h10` the observed CI half-width at `N_R = 10`; assumes the point estimate and the sd stay constant and ignores the change of the t-quantile; it is the size at which the CI would just touch the margin, not a power calculation):

| amp | stat | est | margin | h10 | margin - |est| | N ~ (rough) |
|---|---|---|---|---|---|---|
| 1 | S_tau | 0.01246 | 0.02469 | 0.0239 | 0.01223 | 38 |
| 4 | theta_3 | 0.003248 | 0.005 | 0.01054 | 0.001752 | 362 |

## P3 Paired boundary-layer profile of `var_i(v1)/qbar^2` (derived, post-hoc)

Per realization `r` and face plane `i` (`kind=plane` rows of `e1_{a,b}.csv`, `u_var`): `D_i,r = u_var_b,i / qbar_b^2 - u_var_a,i / qbar_a^2`, with `qbar` the `qbar` of `e1.json` of each medium (= `u_mean` of plane 0; the maximum |qbar - u_mean(plane 0)| over all runs and media is printed below), as in the note. The comparison is (b) minus (a) on the same field; (a), which has no condition at the `x` faces, is the control. Ensemble mean and SE (sd over the ten realizations / sqrt(10)) of `D_i` against the distance from the west face (`x1`) and from the east face (`L1 - x1`) in units of `lambda` (nearest plane). `M_a` = mean over the interior planes `x1 in [5 lambda, L1 - 5 lambda]` of the (a) ensemble-mean profile. Thickness readings, defined here (post-hoc): `d_1%` = smallest distance `d` such that `|mean D_i| <= 0.01 M_a` for every plane between `d` and `20 lambda` from that face; `d_2SE` = smallest `d` such that `|mean D_i| <= 2 SE_i` for every plane between `d` and `20 lambda` (`n/a` = the plane at `20 lambda` already fails). `viol 5-20` counts the planes between `5 lambda` and `20 lambda` violating the `2 SE` condition (a pure-noise reference is ~5 % of the planes). Decay ratio = geometric mean of `D(d+1)/D(d)`, `d = 1, 2, 3` (nearest planes at 1, 2, 3, 4 `lambda`; `n/a` if a mean `D` is not positive).

### amp 1: `lambda = 0.0390625`, `L1 = 8.0`, planes 2049, interior planes 1949, `M_a = 0.6151`

west face:

| d/lambda (target) | d/lambda (plane) | mean D | SE | mean D / M_a |
|---|---|---|---|---|
| 0 | 0.000 | 1.9115e-01 | 1.984e-02 | 0.3108 |
| 0.25 | 0.200 | 1.8918e-01 | 1.770e-02 | 0.3076 |
| 0.5 | 0.500 | 1.6505e-01 | 1.425e-02 | 0.2683 |
| 1 | 1.000 | 1.0347e-01 | 9.453e-03 | 0.1682 |
| 1.5 | 1.500 | 5.9180e-02 | 6.760e-03 | 0.0962 |
| 2 | 2.000 | 3.3766e-02 | 3.467e-03 | 0.0549 |
| 3 | 3.000 | 1.2045e-02 | 1.746e-03 | 0.0196 |
| 4 | 4.000 | 5.0368e-03 | 1.377e-03 | 0.0082 |
| 5 | 5.000 | 3.7168e-03 | 9.262e-04 | 0.0060 |
| 6 | 6.000 | 2.5161e-03 | 8.604e-04 | 0.0041 |
| 8 | 8.000 | 1.2389e-03 | 3.157e-04 | 0.0020 |
| 10 | 10.000 | 2.6367e-04 | 9.921e-05 | 0.0004 |
| 20 | 20.000 | 2.2212e-05 | 2.944e-05 | 0.0000 |

east face:

| d/lambda (target) | d/lambda (plane) | mean D | SE | mean D / M_a |
|---|---|---|---|---|
| 0 | 0.000 | 2.3538e-01 | 2.472e-02 | 0.3827 |
| 0.25 | 0.300 | 2.2469e-01 | 2.136e-02 | 0.3653 |
| 0.5 | 0.500 | 2.0125e-01 | 1.867e-02 | 0.3272 |
| 1 | 1.000 | 1.2587e-01 | 1.226e-02 | 0.2046 |
| 1.5 | 1.500 | 7.4772e-02 | 8.207e-03 | 0.1216 |
| 2 | 2.000 | 4.4641e-02 | 7.275e-03 | 0.0726 |
| 3 | 3.000 | 1.6012e-02 | 3.544e-03 | 0.0260 |
| 4 | 4.000 | 4.7133e-03 | 1.925e-03 | 0.0077 |
| 5 | 5.000 | 1.4958e-03 | 9.972e-04 | 0.0024 |
| 6 | 6.000 | 7.1288e-04 | 7.306e-04 | 0.0012 |
| 8 | 8.000 | 1.0011e-04 | 4.307e-04 | 0.0002 |
| 10 | 10.000 | -2.4398e-05 | 1.435e-04 | -0.0000 |
| 20 | 20.000 | -1.9326e-05 | 2.503e-05 | -0.0000 |

### amp 4: `lambda = 0.125`, `L1 = 13.0`, planes 3329, interior planes 3009, `M_a = 3.1693`

west face:

| d/lambda (target) | d/lambda (plane) | mean D | SE | mean D / M_a |
|---|---|---|---|---|
| 0 | 0.000 | 1.9676e+00 | 5.717e-01 | 0.6208 |
| 0.25 | 0.250 | 1.9034e+00 | 5.166e-01 | 0.6006 |
| 0.5 | 0.500 | 1.5607e+00 | 4.247e-01 | 0.4924 |
| 1 | 1.000 | 8.6422e-01 | 2.602e-01 | 0.2727 |
| 1.5 | 1.500 | 5.0470e-01 | 1.850e-01 | 0.1592 |
| 2 | 2.000 | 2.5365e-01 | 9.738e-02 | 0.0800 |
| 3 | 3.000 | 1.1924e-01 | 4.687e-02 | 0.0376 |
| 4 | 4.000 | 6.2584e-02 | 3.428e-02 | 0.0197 |
| 5 | 5.000 | 6.8720e-03 | 1.097e-02 | 0.0022 |
| 6 | 6.000 | 8.1972e-03 | 1.431e-02 | 0.0026 |
| 8 | 8.000 | 1.1976e-02 | 1.860e-02 | 0.0038 |
| 10 | 10.000 | -4.2302e-03 | 4.808e-03 | -0.0013 |
| 20 | 20.000 | 2.5920e-04 | 6.032e-03 | 0.0001 |

east face:

| d/lambda (target) | d/lambda (plane) | mean D | SE | mean D / M_a |
|---|---|---|---|---|
| 0 | 0.000 | 9.1560e-01 | 2.548e-01 | 0.2889 |
| 0.25 | 0.250 | 7.7053e-01 | 2.294e-01 | 0.2431 |
| 0.5 | 0.500 | 5.4758e-01 | 1.918e-01 | 0.1728 |
| 1 | 1.000 | 3.6084e-01 | 1.432e-01 | 0.1139 |
| 1.5 | 1.500 | 3.6070e-01 | 9.608e-02 | 0.1138 |
| 2 | 2.000 | 2.5002e-01 | 9.644e-02 | 0.0789 |
| 3 | 3.000 | 6.9699e-02 | 4.802e-02 | 0.0220 |
| 4 | 4.000 | 2.9614e-02 | 2.117e-02 | 0.0093 |
| 5 | 5.000 | 2.3839e-02 | 1.552e-02 | 0.0075 |
| 6 | 6.000 | 5.7940e-03 | 3.173e-03 | 0.0018 |
| 8 | 8.000 | -3.0170e-03 | 9.157e-03 | -0.0010 |
| 10 | 10.000 | -5.5855e-03 | 2.979e-03 | -0.0018 |
| 20 | 20.000 | 7.2723e-03 | 6.080e-03 | 0.0023 |

### Derived thickness readings and decay (post-hoc)

| amp | face | d_1% / lambda | d_2SE / lambda | viol 5-20 (2SE) | decay ratio per lambda, 1-4 lambda |
|---|---|---|---|---|---|
| 1 | west | 3.800 | 15.100 | 64/151 | 0.365 |
| 1 | east | 3.800 | 12.900 | 12/151 | 0.335 |
| 4 | west | 4.281 | 3.906 | 0/481 | 0.417 |
| 4 | east | 3.969 | 10.000 | 13/481 | 0.435 |

max |qbar(e1.json) - u_mean(plane 0)| over all runs and media: 0.000e+00

## P4 Mean-flow forcing and tilt (derived, post-hoc)

Per realization `theta_2`, `theta_3` (interior, `primary.json`) and `qbar` of both media; sample sd over the ten realizations per medium and amplitude. For (b): the layer-mean `<v2>/qbar` and `<v3>/qbar` over the four contiguous quarters of the interior cell layers `[layer_lo, layer_hi]` of each realization (Q1 = nearest the west face). `G` is the mean gradient of the (a) solve (`STAGE ... G=` line of `ab_s*.log`).

### Imposed conditions (confirmed in code)

| medium | what is imposed | evidence |
|---|---|---|
| (a) SF-19 affine-periodic | the **mean flux**: target `qbar = (1, 0, 0)`; the mean gradient `G` is an output of the full 3x3 solve `K_eff G = qbar`, so it is free and may have transverse components | `src/physics/flow/AffinePeriodicFlowSolver.cuh:156-167` (`qbar` is the target mean Darcy flux, default `(1,0,0)`), `:212-213` (`G` = solution of `K_eff G = qbar`); `src/physics/flow/AffinePeriodicFlowSolver.cu:965` (lean path used by the stage: `solve_3x3_cramer(report.K_eff, qbar, report.G)`; non-lean `:837`); `src/physics/particles/lagrangian_labels/LongBoxStage.cu:251-253` (`fcfg.qbar = (1,0,0)`) and `:261` (`solve_affine_periodic_flow_lean`) |
| (b) `DirichletMedium` / `solve_head` | the **head drop along x**: `h(0) = G1 L1`, `h(L1) = 0` with `G1 = G[0]` of the (a) solve of the same realization, periodic `y`/`z`; hence the mean head gradient is along `x` and the mean transverse flux is free | `src/physics/particles/lagrangian_labels/DirichletMedium.cu:153-160` (`head_drop = spec.G1 * M.L1`; `xmin` Dirichlet `head_drop`, `xmax` Dirichlet `0`, `ymin/ymax/zmin/zmax` Periodic); `apps/lagrangian_medium_ab/lagrangian_medium_ab_main.cu:599` (`G1 = stage->G[0]`) and `:671` (`bs.G1 = G1`) |

Consequence (hypothesis consistent with the data, not tested here): the two media differ in the macroscopic forcing everywhere in the domain, not only near the `x` faces. In (a) the mean flux is `e1` by construction and the mean gradient carries the transverse components `G[1]`, `G[2]`; in (b) the mean gradient is `(G1, 0, 0)` and the mean flux of a finite realization is `K_eff (G1, 0, 0)` with whatever transverse part the realization's effective tensor gives (the per-realization `theta_2`, `theta_3` of (b) below).

### Per-medium statistics over the ten realizations

| amp | quantity | mean | sd | min | max |
|---|---|---|---|---|---|
| 1 | theta_2 (a) | 1.1444e-05 | 2.5997e-04 | -5.8426e-04 | 3.1398e-04 |
| 1 | theta_3 (a) | 4.6322e-05 | 1.7272e-04 | -1.7757e-04 | 3.4267e-04 |
| 1 | theta_2 (b) | 9.9318e-05 | 1.2857e-03 | -1.8598e-03 | 2.1902e-03 |
| 1 | theta_3 (b) | -2.1023e-05 | 1.7420e-03 | -4.1554e-03 | 1.9039e-03 |
| 1 | theta_2 (b) - (a) | 8.7874e-05 | 1.2264e-03 | -1.8512e-03 | 1.8762e-03 |
| 1 | theta_3 (b) - (a) | -6.7345e-05 | 1.6357e-03 | -3.9778e-03 | 1.6131e-03 |
| 1 | qbar (a) | 1.0000e+00 | 6.3379e-15 | 1.0000e+00 | 1.0000e+00 |
| 1 | qbar (b) | 1.0011e+00 | 1.1295e-04 | 1.0009e+00 | 1.0012e+00 |
| 1 | G2/G1 (a solve) | -8.5470e-05 | 1.2280e-03 | -1.8716e-03 | 1.8631e-03 |
| 1 | G3/G1 (a solve) | 6.6960e-05 | 1.6424e-03 | -1.6020e-03 | 3.9972e-03 |
| 4 | theta_2 (a) | 3.1796e-04 | 6.8199e-03 | -9.0615e-03 | 1.5850e-02 |
| 4 | theta_3 (a) | -4.6182e-04 | 5.8429e-03 | -8.8210e-03 | 8.4708e-03 |
| 4 | theta_2 (b) | 6.1895e-03 | 1.6551e-02 | -1.5054e-02 | 4.5847e-02 |
| 4 | theta_3 (b) | -3.7094e-03 | 1.5516e-02 | -2.7053e-02 | 2.2097e-02 |
| 4 | theta_2 (b) - (a) | 5.8715e-03 | 1.2276e-02 | -1.1706e-02 | 2.9997e-02 |
| 4 | theta_3 (b) - (a) | -3.2476e-03 | 1.8188e-02 | -3.5523e-02 | 2.4595e-02 |
| 4 | qbar (a) | 1.0000e+00 | 1.4013e-12 | 1.0000e+00 | 1.0000e+00 |
| 4 | qbar (b) | 1.0068e+00 | 3.5005e-03 | 1.0025e+00 | 1.0131e+00 |
| 4 | G2/G1 (a solve) | -5.2225e-03 | 1.1149e-02 | -2.6076e-02 | 1.1260e-02 |
| 4 | G3/G1 (a solve) | 2.8677e-03 | 1.7481e-02 | -2.4184e-02 | 3.5014e-02 |

### Tilt of (b) against the forcing

`corr` = Pearson correlation over the ten realizations of the (b) tilt with the transverse-to-longitudinal ratio of the mean gradient of the (a) solve (exploratory; n = 10). `rms |theta|` = sqrt(mean(theta_2^2 + theta_3^2)) of (b). Ratios only (no alpha statement): sd of each component of (b) over 0.005 (the D19 per-component margin) and `rms |theta|` over 0.0071 (the D19 vector bound).

| amp | corr(theta_2 b, G2/G1) | corr(theta_3 b, G3/G1) | rms abs theta (b) | sd(theta_2 b)/0.005 | sd(theta_3 b)/0.005 | rms abs theta (b)/0.0071 |
|---|---|---|---|---|---|---|
| 1 | -0.980 | -0.997 | 2.0565e-03 | 0.257 | 0.348 | 0.290 |
| 4 | -0.919 | -0.950 | 2.2700e-02 | 3.310 | 3.103 | 3.197 |

Geometric consequence (no alpha statement): in fixed axes, the transverse variance of a plume whose mean flow is tilted by `theta` picks up approximately `theta^2 sigma_L^2`, with `sigma_L^2` the longitudinal variance of the plume.

### Layer-mean tilt of (b) along `x1` (quarters of the interior layers)

The four quarters check whether the (b) tilt is one constant along `x1` (equal Q1..Q4) or varies.

| amp | r | theta_2 (b) | Q1 v2 | Q2 v2 | Q3 v2 | Q4 v2 | theta_3 (b) | Q1 v3 | Q2 v3 | Q3 v3 | Q4 v3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | -9.6106e-04 | 1.4944e-03 | -3.1749e-03 | 1.7226e-03 | -3.8864e-03 | 1.9039e-03 | 3.5384e-03 | -5.8036e-05 | 5.0101e-03 | -8.7497e-04 |
| 1 | 1 | -6.1246e-04 | -4.4697e-04 | -4.8532e-04 | -2.1404e-03 | 6.2283e-04 | 5.3613e-04 | 7.0631e-04 | -2.7758e-04 | 1.5386e-03 | 1.7723e-04 |
| 1 | 2 | 9.2233e-04 | 2.4813e-03 | 1.2713e-03 | -1.5898e-03 | 1.5266e-03 | 1.6316e-03 | 6.2411e-05 | 6.9786e-03 | -4.1399e-04 | -1.0048e-04 |
| 1 | 3 | 1.1403e-03 | -3.8000e-03 | 4.6824e-04 | 3.2918e-03 | 4.6010e-03 | -4.1554e-03 | -3.5752e-03 | 3.9813e-04 | -5.7481e-03 | -7.6964e-03 |
| 1 | 4 | -1.8598e-03 | -3.3809e-03 | -2.1305e-03 | -1.4739e-03 | -4.5393e-04 | -6.0606e-04 | -4.3884e-04 | -3.7651e-03 | 1.5804e-03 | 1.9929e-04 |
| 1 | 5 | -1.4300e-03 | -1.3065e-03 | 1.9803e-03 | 6.5675e-04 | -7.0507e-03 | -1.0072e-03 | 2.1717e-04 | -2.2772e-03 | -1.9356e-03 | -3.3219e-05 |
| 1 | 6 | 2.1902e-03 | 3.9855e-03 | -9.2852e-04 | 2.7856e-03 | 2.9182e-03 | -4.7440e-04 | 3.0835e-03 | -3.7694e-04 | -1.0512e-04 | -4.4991e-03 |
| 1 | 7 | 5.2530e-04 | -5.5993e-04 | -2.7147e-03 | 8.3002e-03 | -2.9244e-03 | 5.2500e-05 | 3.3167e-04 | -1.4553e-03 | -6.2289e-04 | 1.9565e-03 |
| 1 | 8 | 9.6872e-04 | 2.0456e-03 | 2.9551e-03 | -9.9847e-04 | -1.2735e-04 | 1.0981e-03 | 3.8507e-03 | 1.7824e-03 | -3.4657e-03 | 2.2249e-03 |
| 1 | 9 | 1.0974e-04 | 4.3318e-03 | -2.6552e-04 | 1.0259e-03 | -4.6532e-03 | 8.1062e-04 | 1.3828e-03 | -7.8012e-04 | 3.1686e-03 | -5.2871e-04 |
| 4 | 0 | -1.5054e-02 | 4.2348e-02 | -1.6527e-02 | -1.5362e-02 | -7.0677e-02 | 3.9816e-03 | 3.5100e-02 | 2.4703e-03 | -2.4523e-03 | -1.9192e-02 |
| 4 | 1 | -7.4378e-04 | -4.5659e-02 | 1.8458e-02 | 3.3420e-02 | -9.1942e-03 | -8.5265e-03 | -1.1725e-02 | 9.4316e-03 | 8.8931e-04 | -3.2702e-02 |
| 4 | 2 | -2.3226e-03 | 1.7307e-02 | 8.7663e-03 | -4.0931e-02 | 5.5675e-03 | 9.2575e-03 | 1.7965e-02 | 1.1797e-02 | 3.2888e-02 | -2.5620e-02 |
| 4 | 3 | 4.5847e-02 | -3.3575e-02 | 2.0141e-02 | 1.1472e-01 | 8.2102e-02 | -1.6868e-02 | 1.5908e-02 | -1.7447e-02 | -3.2861e-02 | -3.3071e-02 |
| 4 | 4 | 6.4190e-03 | 2.1067e-02 | -2.0810e-02 | 1.0705e-02 | 1.4715e-02 | 8.0291e-04 | 8.5260e-03 | -4.5929e-02 | 1.8999e-02 | 2.1615e-02 |
| 4 | 5 | -2.3082e-03 | -1.6461e-02 | 1.0208e-02 | 1.0174e-02 | -1.3153e-02 | -2.7053e-02 | -1.0001e-01 | 8.5480e-03 | -4.6121e-02 | 2.9368e-02 |
| 4 | 6 | 1.1055e-03 | -2.1370e-02 | 1.9179e-02 | -6.3499e-03 | 1.2963e-02 | -2.1787e-02 | -4.4429e-02 | 6.5114e-03 | -6.2154e-03 | -4.3016e-02 |
| 4 | 7 | 1.2772e-02 | 4.0352e-02 | -2.2053e-02 | 5.4249e-02 | -2.1461e-02 | 9.1473e-03 | 4.6954e-02 | -5.7056e-02 | 3.9305e-02 | 7.3865e-03 |
| 4 | 8 | 1.7372e-02 | 6.6272e-02 | -2.0785e-02 | 5.7130e-03 | 1.8287e-02 | -8.1459e-03 | -2.6242e-02 | 3.8844e-02 | -2.6506e-02 | -1.8680e-02 |
| 4 | 9 | -1.1918e-03 | 1.7792e-02 | 3.3418e-02 | -3.7459e-03 | -5.2232e-02 | 2.2097e-02 | -1.0760e-02 | 2.2085e-02 | 7.9332e-02 | -2.2686e-03 |

Summary of the quarter spread (derived): per realization `spread = max(Q) - min(Q)` over the four quarters, averaged over the ten realizations, against the mean `|overall|` of the same quantity. `(b)` = the table above; `(a)` = the same quarters of the affine-periodic medium (its window-mean fluctuation is the sampling control); `(b)-(a)` = quarter differences on the same field, which removes the field's own fluctuation (overall = `theta (b) - theta (a)`). A constant medium offset along `x1` would give `(b)-(a)` spread much smaller than its `|overall|`.

| amp | component | quantity | mean spread | mean abs overall | mean spread / mean abs overall |
|---|---|---|---|---|---|
| 1 | v2 | (b) | 6.1880e-03 | 1.0720e-03 | 5.772 |
| 1 | v2 | (a) | 6.1948e-03 | 1.8441e-04 | 33.592 |
| 1 | v2 | (b)-(a) | 5.2460e-05 | 1.0119e-03 | 0.052 |
| 1 | v3 | (b) | 5.3288e-03 | 1.2276e-03 | 4.341 |
| 1 | v3 | (a) | 5.3274e-03 | 1.3581e-04 | 39.228 |
| 1 | v3 | (b)-(a) | 6.5092e-05 | 1.1506e-03 | 0.057 |
| 4 | v2 | (b) | 7.5674e-02 | 1.0514e-02 | 7.198 |
| 4 | v2 | (a) | 7.6244e-02 | 4.8670e-03 | 15.665 |
| 4 | v2 | (b)-(a) | 2.9398e-03 | 1.0089e-02 | 0.291 |
| 4 | v3 | (b) | 7.1122e-02 | 1.2767e-02 | 5.571 |
| 4 | v3 | (a) | 7.0557e-02 | 4.6103e-03 | 15.304 |
| 4 | v3 | (b)-(a) | 4.0347e-03 | 1.4242e-02 | 0.283 |

## P5 Memory of medium (b) at the production grid (derived, post-hoc)

Measured peaks of the A/B runs (`RECORDED run=ab_s* pid_peak_mib` of `ab/results.txt`; the peak is the (b) `solve` phase, realization 0 `MEM_PHASE_B` of `ab_s*.log`). `B/cell (PID)` = PID peak MiB x 1048576 / cells, **including the CUDA context** and the unmodeled overhead; `B/cell (exact solve)` = `exact_bytes` of the `solve` phase / cells (allocations only). Extrapolation to 3328x384x384 (490733568 cells) is linear in cells (rough; the context does not scale, so it overstates slightly).

| run | grid | cells | PID peak MiB | B/cell (PID, with context) | B/cell (exact solve) | extrapolated PID peak at 3328x384x384, MiB (derived) |
|---|---|---|---|---|---|---|
| ab_s1 | 2048x256x256 | 134217728 | 12337 | 96.38 | 92.57 | 45107 |
| ab_s4 | 3328x256x256 | 218103808 | 19745 | 94.93 | 92.57 | 44426 |

Lean non-projected (b) flow solve: 68.57 B/cell (memory audit 2026-10-22 section 10, buffer count of `solve_head` with the section 8 tricks; class **estimate from code count, not measured**) x 490733568 cells = 32091 MiB (allocations only, no context) against the 30000 MiB ceiling. The non-periodic spline stage is not included in either figure (the extrapolated measured figure or the lean estimate).

Reference levels at the same box (measured, SF-38 B2): the lean periodic stage with the SF-19 flow and the spline reaches exact 60.655 B/cell (`MEM_PHASE phase=S2`, `prodgrid/B2.log`) and a PID peak of 28885 MiB = 61.72 B/cell with context; the ceiling 30000 MiB corresponds to 64.10 B/cell with context.

Conclusion (derived): medium (b) as measured (44426 MiB extrapolated from `ab_s4`, 45107 MiB from `ab_s1`) does not fit 3328x384x384 in 30000 MiB; even the lean non-projected estimate (32091 MiB, before the spline) exceeds it. A lean (b) flow stage together with the non-periodic spline would have to come in at or below the level of the SF-40 stage (~61 B/cell, 60.655 exact; PID peak 28885 MiB) to fit; otherwise stop rule R3 (domain reduction by owner decision) applies at `sigma^2 >= 2.25`.
