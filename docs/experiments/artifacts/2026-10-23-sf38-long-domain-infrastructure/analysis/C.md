`C_STAT` lines of `ab/results.txt` (= `ab/ab_decision.out`). `est` is `R = ln(Pbar_a/Pbar_b)` for `test=ratio` and the paired difference `Dbar` for `test=diff|floor`; `margin` is the half-width on the same scale (`ln 1.05` = 0.04879, `ln 1.025` = 0.02469; floor margin = `2.5e-3 lambda`, box units). Statistics in box units (S_T) and normalized (S_tau, V1, theta). Values printed with 4 significant digits; full precision in the raw lines.

| amp | stat | test | Pbar_a | Pbar_b | est | se | ci_lo | ci_hi | margin | CI width / (2 margin) (derived) | class |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | S_T | floor | 3.198e-05 | 3.305e-05 | -1.08e-06 | 7.142e-06 | -1.417e-05 | 1.201e-05 | 9.766e-05 | 0.134 | equivalent |
| 1 | S_tau | ratio | 0.0819 | 0.08089 | 0.01246 | 0.01304 | -0.01144 | 0.03636 | 0.02469 | 0.968 | inconclusive |
| 1 | V1 | ratio | 0.6151 | 0.6151 | -0.0001057 | 2.322e-05 | -0.0001482 | -6.313e-05 | 0.04879 | 0.001 | equivalent |
| 1 | theta_2 | diff | 1.144e-05 | 9.932e-05 | -8.787e-05 | 0.0003878 | -0.0007988 | 0.000623 | 0.005 | 0.142 | equivalent |
| 1 | theta_3 | diff | 4.632e-05 | -2.102e-05 | 6.734e-05 | 0.0005173 | -0.0008808 | 0.001015 | 0.005 | 0.190 | equivalent |
| 4 | S_T | ratio | 0.01631 | 0.01739 | -0.06383 | 0.07906 | -0.2087 | 0.08109 | 0.04879 | 2.970 | inconclusive |
| 4 | S_tau | ratio | 2.38 | 2.21 | 0.07412 | 0.0829 | -0.07783 | 0.2261 | 0.02469 | 6.154 | inconclusive |
| 4 | V1 | ratio | 3.169 | 3.171 | -0.0004837 | 0.000316 | -0.001063 | 9.555e-05 | 0.04879 | 0.012 | equivalent |
| 4 | theta_2 | diff | 0.000318 | 0.006189 | -0.005872 | 0.003882 | -0.01299 | 0.001244 | 0.005 | 1.423 | inconclusive |
| 4 | theta_3 | diff | -0.0004618 | -0.003709 | 0.003248 | 0.005752 | -0.007295 | 0.01379 | 0.005 | 2.109 | inconclusive |

`C_AMP amp=1 equivalent=4/5 non_equivalent=0 inconclusive=1`
`C_AMP amp=4 equivalent=1/5 non_equivalent=0 inconclusive=4`

`C_VERDICT selected=b equivalent=5/10 non_equivalent=0 inconclusive=5`
