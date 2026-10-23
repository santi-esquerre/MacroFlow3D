Recorded E1 quantities from `ab/ab_s*/realizations/r_*/e1.json` and `e1_{a,b}.csv` (means over the ten realizations; `qbar` of (a) is the plane mean of `U` at `i=0`, 1 to solver level).

| amp | medium | qbar min | qbar max | qbar mean | <var_i(v2)>/qbar^2 mean | <var_i(v3)>/qbar^2 mean |
|---|---|---|---|---|---|---|
| 1 | a | 1.000000 | 1.000000 | 1.000000 | 0.1044 | 0.1046 |
| 1 | b | 1.000923 | 1.001247 | 1.001075 | 0.1044 | 0.1046 |
| 4 | a | 1.000000 | 1.000000 | 1.000000 | 0.8447 | 0.8052 |
| 4 | b | 1.002470 | 1.013060 | 1.006767 | 0.8458 | 0.8077 |

Ensemble-mean plane profile of `var_i(v1)/qbar^2` (west face, nearest plane to the listed distance; SE = sd over the ten realizations / sqrt(10)).

| amp | medium | x1/lambda | mean | SE |
|---|---|---|---|---|
| 1 | a | 0.00 | 0.6052 | 0.0175 |
| 1 | a | 0.20 | 0.6046 | 0.0200 |
| 1 | a | 0.50 | 0.6058 | 0.0229 |
| 1 | a | 1.00 | 0.6129 | 0.0236 |
| 1 | a | 2.00 | 0.6064 | 0.0178 |
| 1 | a | 4.00 | 0.6067 | 0.0180 |
| 1 | a | 8.00 | 0.6196 | 0.0200 |
| 1 | b | 0.00 | 0.7963 | 0.0358 |
| 1 | b | 0.20 | 0.7937 | 0.0357 |
| 1 | b | 0.50 | 0.7709 | 0.0345 |
| 1 | b | 1.00 | 0.7163 | 0.0302 |
| 1 | b | 2.00 | 0.6402 | 0.0205 |
| 1 | b | 4.00 | 0.6117 | 0.0178 |
| 1 | b | 8.00 | 0.6209 | 0.0200 |
| 4 | a | 0.00 | 2.5639 | 0.2553 |
| 4 | a | 0.25 | 2.5461 | 0.2132 |
| 4 | a | 0.50 | 2.5573 | 0.1812 |
| 4 | a | 1.00 | 2.5027 | 0.1710 |
| 4 | a | 2.00 | 2.7118 | 0.2405 |
| 4 | a | 4.00 | 3.5182 | 0.6585 |
| 4 | a | 8.00 | 4.1019 | 0.6570 |
| 4 | b | 0.00 | 4.5315 | 0.6989 |
| 4 | b | 0.25 | 4.4495 | 0.6651 |
| 4 | b | 0.50 | 4.1180 | 0.5817 |
| 4 | b | 1.00 | 3.3669 | 0.4009 |
| 4 | b | 2.00 | 2.9654 | 0.2898 |
| 4 | b | 4.00 | 3.5808 | 0.6731 |
| 4 | b | 8.00 | 4.1139 | 0.6615 |

derived (literal reading of the specification's recorded boundary-layer quantity; not scripted by the campaign tooling): `M` = mean over the interior planes `x1 in [5 lambda, L1 - 5 lambda]` of the ensemble-mean profile; a plane is `outside` if `|mean_i - M| > 2 SE_i`; `west d` (`east d`) = the largest distance from the west (east) face of an `outside` plane in the west (east) half, in `lambda`. The same procedure is applied to medium (a), which has no `x`-face condition, as a control of the estimator.

| amp | medium | M | interior planes | interior planes outside | west d/lambda | east d/lambda |
|---|---|---|---|---|---|---|
| 1 | a | 0.6151 | 1949 | 115 | 99.60 | 79.30 |
| 1 | b | 0.6151 | 1949 | 116 | 99.60 | 79.30 |
| 4 | a | 3.1693 | 3009 | 365 | 42.38 | 47.22 |
| 4 | b | 3.1708 | 3009 | 365 | 42.34 | 47.22 |
