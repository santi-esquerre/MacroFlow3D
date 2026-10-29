# SF-39 decision points (preregistration section 10)

## primary

| amplitude | estimator | decision point | verdict | CI | n | note |
|---|---|---|---|---|---|---|
| s025 | alpha_TAM | Gate 4 label (D) | physical (bounded) | [-0.00180066, 0.00155315] | 25 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=-0.000123753; refinement trend detected: d16_8: -2.74e-06 % [-5.38e-06, -9.76e-08] |
| s025 | alpha_TP | Gate 4 label (D) | physical (bounded) | [-0.000547463, 0.00191332] | 25 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=0.000682929; refinement trend detected: D_tw/2^1.5: -0.56 % [-1.11, -0.0138] |
| s025 | alpha_TAM+alpha_TP | Gate 4 amplitude label (D) | physical (bounded) |  | 25 | alpha_TAM: physical (bounded); alpha_TP: physical (bounded) |
| s025 | alpha_LAM | Gate 5 (C) Lagrangian twin vs Par2 | inconclusive | [-0.0754664, -0.0265599] | 25/25 | diff=-0.0510132 margin=0.0270195; late Par2 beyond pilot 0/5 |
| s025 | alpha_LAM | Gate 5 sensitivity FV qbar | inconclusive | [-0.0749753, -0.0260681] | 25/25 | diff=-0.0505217 margin=0.0270195 |
| s025 | alpha_LAM | Gate 5 sensitivity ANCOVA | inconclusive | [-0.0765806, -0.0262464] | 25/25 | diff=-0.0514135 margin=0.0270595 adjusted Par2 mean=0.270595 |
| s1 | alpha_TAM | Gate 4 label (D) | physical (positive) | [0.00744747, 0.0115829] | 50 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=0.00951516 |
| s1 | alpha_TP | Gate 4 label (D) | physical (positive) | [0.0067542, 0.00972685] | 50 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=0.00824052 |
| s1 | alpha_TAM+alpha_TP | Gate 4 amplitude label (D) | physical (positive) |  | 50 | alpha_TAM: physical (positive); alpha_TP: physical (positive) |
| s1 | alpha_LAM | Gate 5 (C) Lagrangian twin vs Par2 | inconclusive | [-0.145973, -0.0542721] | 50/50 | diff=-0.100123 margin=0.113816; late Par2 beyond pilot 2/30 |
| s1 | alpha_LAM | Gate 5 sensitivity FV qbar | inconclusive | [-0.144256, -0.0522624] | 50/50 | diff=-0.098259 margin=0.113816 |
| s1 | alpha_LAM | Gate 5 sensitivity ANCOVA | inconclusive | [-0.14673, -0.0546057] | 50/50 | diff=-0.100668 margin=0.11387 adjusted Par2 mean=1.1387 |
| s225 | alpha_TAM | Gate 4 label (D) | physical (positive) | [0.0261837, 0.0312567] | 62 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=0.0287202 |
| s225 | alpha_TP | Gate 4 label (D) | physical (positive) | [0.0286589, 0.0329041] | 62 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=0.0307815 |
| s225 | alpha_TAM+alpha_TP | Gate 4 amplitude label (D) | physical (positive) |  | 62 | alpha_TAM: physical (positive); alpha_TP: physical (positive) |
| s225 | alpha_LAM | Gate 5 (C) Lagrangian twin vs Par2 | agree | [-0.126763, 0.0411191] | 62/62 | diff=-0.0428221 margin=0.320774; late Par2 beyond pilot 1/42 |
| s225 | alpha_LAM | Gate 5 sensitivity FV qbar | agree | [-0.121554, 0.0464275] | 62/62 | diff=-0.0375634 margin=0.320774 |
| s225 | alpha_LAM | Gate 5 sensitivity ANCOVA | agree | [-0.122403, 0.042396] | 62/62 | diff=-0.0400036 margin=0.320492 adjusted Par2 mean=3.20492 |
| s4 | alpha_TAM | Gate 4 label (D) | physical (positive) | [0.0649707, 0.0753498] | 71 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=0.0701602 |
| s4 | alpha_TP | Gate 4 label (D) | physical (positive) | [0.0621877, 0.0711073] | 71 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=0.0666475 |
| s4 | alpha_TAM+alpha_TP | Gate 4 amplitude label (D) | physical (positive) |  | 71 | alpha_TAM: physical (positive); alpha_TP: physical (positive) |
| s4 | alpha_LAM | Gate 5 (C) Lagrangian twin vs Par2 | agree | [-0.322621, 0.189846] | 71/71 | diff=-0.0663872 margin=0.568072; late Par2 beyond pilot 1/51 |
| s4 | alpha_LAM | Gate 5 sensitivity FV qbar | agree | [-0.288881, 0.223931] | 71/71 | diff=-0.0324749 margin=0.568072 |
| s4 | alpha_LAM | Gate 5 sensitivity ANCOVA | agree | [-0.323369, 0.188555] | 71/71 | diff=-0.067407 margin=0.568174 adjusted Par2 mean=5.68174 |

## key secondary

| amplitude | estimator | decision point | verdict | CI | n | note |
|---|---|---|---|---|---|---|
| s025 | alpha_TAM | B half-width (95 %) | met | [-0.00214667, 0.00189916] | 25 | halfwidth=0.00202291 target=0.0025 |
| s025 | alpha_TAM | A h narrow verdict (D_f) | converged | [-1.1887e-05, 3.34125e-06] | 20 | m=0.00125 mean D_f=-4.27286e-06 |
| s025 | alpha_TAM | A h production verdict (F4) | converged | [-8.82288e-05, 3.84826e-05] | 20 | CI of D_tw vs +-2^1.5 m=0.00353553 (inside=True) |
| s025 | alpha_TAM | A h order contrast | monotone order >= 1.5 not rejected | [-1.14125e-05, 9.06838e-06] | 20 | CI of s(2^1.5 D_f - D_c), s=-1 |
| s025 | alpha_TAM | A h_max verdict (h/16 - h/8) | converged | [1.20829e-13, 6.65552e-12] | 20 | m=0.00125 mean=3.38817e-12 |
| s025 | alpha_TP | B half-width (95 %) | met | [-0.000801337, 0.00216719] | 25 | halfwidth=0.00148427 target=0.0025 |
| s025 | alpha_TP | A h narrow verdict (D_f) | converged | [-4.02883e-06, 1.97855e-06] | 20 | m=0.00125 mean D_f=-1.02514e-06 |
| s025 | alpha_TP | A h production verdict (F4) | converged | [-2.13484e-05, -2.67101e-07] | 20 | CI of D_tw vs +-2^1.5 m=0.00353553 (inside=True) |
| s025 | alpha_TP | A h order contrast | monotone order >= 1.5 not rejected | [-5.70056e-06, 9.26329e-07] | 20 | CI of s(2^1.5 D_f - D_c), s=-1 |
| s025 | alpha_TP | A h_max verdict (h/16 - h/8) | converged | [-1.6905e-13, 9.45267e-13] | 20 | m=0.00125 mean=3.88109e-13 |
| s025 | alpha_LAM | B half-width (95 %) | not met | [0.206459, 0.233919] | 25 | halfwidth=0.0137301 target=0.0110095 |
| s025 | alpha_LAM | A h narrow verdict (D_f) | converged | [-0.000216888, 0.000300698] | 20 | m=0.00503004 mean D_f=4.19047e-05 |
| s025 | alpha_LAM | A h production verdict (F4) | converged | [-0.00117759, 3.3864e-05] | 20 | CI of D_tw vs +-2^1.5 m=0.0142271 (inside=True) |
| s025 | alpha_LAM | A h order contrast | monotone order >= 1.5 not rejected | [-0.000371473, 0.000218061] | 20 | CI of s(2^1.5 D_f - D_c), s=+1 |
| s025 | alpha_LAM | A h_max verdict (h/16 - h/8) | converged | [-4.12448e-11, 1.98941e-11] | 20 | m=0.00560406 mean=-1.06753e-11 |
| s1 | alpha_TAM | B half-width (95 %) | met | [0.00703675, 0.0119936] | 50 | halfwidth=0.00247841 target=0.0025 |
| s1 | alpha_TAM | A h narrow verdict (D_f) | converged | [-0.000167966, 3.92091e-05] | 20 | m=0.00125 mean D_f=-6.43787e-05 |
| s1 | alpha_TAM | A h production verdict (F4) | converged | [0.000102543, 0.00103975] | 20 | CI of D_tw vs +-2^1.5 m=0.00353553 (inside=True) |
| s1 | alpha_TAM | A h order contrast | monotone order >= 1.5 not rejected | [-0.000561402, 0.000183979] | 20 | CI of s(2^1.5 D_f - D_c), s=+1 |
| s1 | alpha_TAM | A h_max verdict (h/16 - h/8) | converged | [-1.56015e-10, 5.39868e-10] | 20 | m=0.00125 mean=1.91926e-10 |
| s1 | alpha_TP | B half-width (95 %) | met | [0.00645896, 0.0100221] | 50 | halfwidth=0.00178156 target=0.0025 |
| s1 | alpha_TP | A h narrow verdict (D_f) | converged | [-3.20374e-05, 4.02191e-05] | 20 | m=0.00125 mean D_f=4.09086e-06 |
| s1 | alpha_TP | A h production verdict (F4) | converged | [-0.000170128, 0.00013577] | 20 | CI of D_tw vs +-2^1.5 m=0.00353553 (inside=True) |
| s1 | alpha_TP | A h order contrast | monotone order >= 1.5 not rejected | [-0.000138618, 6.62284e-05] | 20 | CI of s(2^1.5 D_f - D_c), s=-1 |
| s1 | alpha_TP | A h_max verdict (h/16 - h/8) | converged | [-7.58195e-11, 4.05719e-11] | 20 | m=0.00125 mean=-1.76238e-11 |
| s1 | alpha_LAM | B half-width (95 %) | met | [1.00195, 1.07378] | 50 | halfwidth=0.0359121 target=0.0518932 |
| s1 | alpha_LAM | A h narrow verdict (D_f) | converged | [-0.00571214, 0.00395925] | 20 | m=0.0243512 mean D_f=-0.000876446 |
| s1 | alpha_LAM | A h production verdict (F4) | converged | [-0.0185043, 0.00836465] | 20 | CI of D_tw vs +-2^1.5 m=0.0688756 (inside=True) |
| s1 | alpha_LAM | A h order contrast | monotone order >= 1.5 not rejected | [-0.0297057, -0.00236421] | 20 | CI of s(2^1.5 D_f - D_c), s=-1 |
| s1 | alpha_LAM | A h_max verdict (h/16 - h/8) | converged | [-5.67193e-09, 4.75714e-09] | 20 | m=0.0263506 mean=-4.57394e-10 |
| s225 | alpha_TAM | B half-width (95 %) | not met | [0.0256835, 0.0317569] | 62 | halfwidth=0.00303673 target=0.00287202 |
| s225 | alpha_TAM | A h narrow verdict (D_f) | converged | [-0.000398307, 0.000736688] | 20 | m=0.00125 mean D_f=0.00016919 |
| s225 | alpha_TAM | A h production verdict (F4) | converged | [-0.00263635, 0.000591854] | 26 | CI of D_tw vs +-2^1.5 m=0.00353553 (inside=True) |
| s225 | alpha_TAM | A h order contrast | monotone order >= 1.5 not rejected | [-0.00311542, 0.00120373] | 20 | CI of s(2^1.5 D_f - D_c), s=-1 |
| s225 | alpha_TAM | A h_max verdict (h/16 - h/8) | converged | [-1.53901e-08, 8.20671e-09] | 20 | m=0.00142575 mean=-3.59169e-09 |
| s225 | alpha_TP | B half-width (95 %) | met | [0.0282403, 0.0333227] | 62 | halfwidth=0.00254121 target=0.00307815 |
| s225 | alpha_TP | A h narrow verdict (D_f) | converged | [-0.000331679, 0.000225646] | 20 | m=0.00129241 mean D_f=-5.30166e-05 |
| s225 | alpha_TP | A h production verdict (F4) | converged | [-0.000307273, 0.000932522] | 26 | CI of D_tw vs +-2^1.5 m=0.00365547 (inside=True) |
| s225 | alpha_TP | A h order contrast | monotone order >= 1.5 not rejected | [-0.00130572, 0.000708356] | 20 | CI of s(2^1.5 D_f - D_c), s=-1 |
| s225 | alpha_TP | A h_max verdict (h/16 - h/8) | converged | [-5.14142e-07, 4.75373e-07] | 20 | m=0.00147528 mean=-1.93845e-08 |
| s225 | alpha_LAM | B half-width (95 %) | met | [3.07129, 3.2099] | 62 | halfwidth=0.0693025 target=0.15703 |
| s225 | alpha_LAM | A h narrow verdict (D_f) | inconclusive | [-0.0821292, 0.0062874] | 20 | m=0.0739047 mean D_f=-0.0379209 |
| s225 | alpha_LAM | A h production verdict (F4) | inconclusive | [-0.107786, 0.0388941] | 26 | CI of D_tw vs +-2^1.5 m=0.209034 (inside=True) |
| s225 | alpha_LAM | A h order contrast | monotone order >= 1.5 not rejected | [-0.275268, 0.0346545] | 20 | CI of s(2^1.5 D_f - D_c), s=+1 |
| s225 | alpha_LAM | A h_max verdict (h/16 - h/8) | converged | [-3.5973e-07, 6.99308e-08] | 20 | m=0.0785595 mean=-1.449e-07 |
| s4 | alpha_TAM | B half-width (95 %) | met | [0.063951, 0.0763695] | 71 | halfwidth=0.00620926 target=0.00701602 |
| s4 | alpha_TAM | A h narrow verdict (D_f) | converged | [-0.000318004, 0.000118186] | 20 | m=0.00146704 mean D_f=-9.99087e-05 |
| s4 | alpha_TAM | A h production verdict (F4) | converged | [0.000476139, 0.00370095] | 20 | CI of D_tw vs +-2^1.5 m=0.00414942 (inside=True) |
| s4 | alpha_TAM | A h order contrast | monotone order >= 1.5 not rejected | [-0.00150686, -0.00017419] | 20 | CI of s(2^1.5 D_f - D_c), s=+1 |
| s4 | alpha_TAM | A h_max verdict (h/16 - h/8) | converged | [-3.38615e-10, 2.33162e-10] | 20 | m=0.00374365 mean=-5.27268e-11 |
| s4 | alpha_TP | B half-width (95 %) | met | [0.0613115, 0.0719836] | 71 | halfwidth=0.00533605 target=0.00666475 |
| s4 | alpha_TP | A h narrow verdict (D_f) | converged | [-0.00033262, 4.5937e-05] | 20 | m=0.00125 mean D_f=-0.000143342 |
| s4 | alpha_TP | A h production verdict (F4) | converged | [-0.00105794, 0.00178742] | 20 | CI of D_tw vs +-2^1.5 m=0.00353553 (inside=True) |
| s4 | alpha_TP | A h order contrast | monotone order >= 1.5 not rejected | [-0.00132743, 4.06242e-05] | 20 | CI of s(2^1.5 D_f - D_c), s=+1 |
| s4 | alpha_TP | A h_max verdict (h/16 - h/8) | converged | [-7.27348e-08, 2.65007e-09] | 20 | m=0.00320762 mean=-3.50424e-08 |
| s4 | alpha_LAM | B half-width (95 %) | met | [4.50578, 4.80307] | 71 | halfwidth=0.148649 target=0.232721 |
| s4 | alpha_LAM | A h narrow verdict (D_f) | converged | [-0.0301356, 0.000738439] | 20 | m=0.0960066 mean D_f=-0.0146986 |
| s4 | alpha_LAM | A h production verdict (F4) | converged | [-0.0199022, 0.0874097] | 20 | CI of D_tw vs +-2^1.5 m=0.271548 (inside=True) |
| s4 | alpha_LAM | A h order contrast | monotone order >= 1.5 not rejected | [-0.0145633, 0.0578378] | 20 | CI of s(2^1.5 D_f - D_c), s=-1 |
| s4 | alpha_LAM | A h_max verdict (h/16 - h/8) | converged | [1.82155e-09, 1.18161e-08] | 20 | m=0.120875 mean=6.81882e-09 |

## estimation

| amplitude | estimator | decision point | verdict | CI | n | note |
|---|---|---|---|---|---|---|
| s025 | alpha_TP | F7 Delta_half | not shown decreasing | [-0.00371841, 0.00104937] | 25 | mean Delta_half=-0.00133452 |
| s025 | rho(X_b), pair 8-16 | F8 readings | pathwise resolved |  | 20 | median=6.79164e-10 p95=6.93762e-10 (m=0.00125, S=0.360456) |
| s025 | alpha_TAM | F11 Delta_P production | Delta_P below 0 | [-7.01444e-05, -9.50143e-06] | 25 | mean Delta_P=-3.98229e-05 |
| s025 | alpha_TP | F11 Delta_P production | Delta_P below 0 | [-1.17339e-05, -6.90432e-07] | 25 | mean Delta_P=-6.21218e-06 |
| s025 | alpha_Tfixed | F11 Delta_P production | Delta_P below 0 | [-7.01486e-05, -9.52468e-06] | 25 | mean Delta_P=-3.98367e-05 |
| s025 | alpha_TAM | F11 Delta_P narrow_half | Delta_P includes 0 | [-7.76813e-05, 2.49804e-05] | 20 | mean Delta_P=-2.63504e-05 |
| s025 | alpha_TP | F11 Delta_P narrow_half | Delta_P includes 0 | [-2.8619e-05, 5.30864e-06] | 20 | mean Delta_P=-1.16552e-05 |
| s025 | alpha_Tfixed | F11 Delta_P narrow_half | Delta_P includes 0 | [-7.77135e-05, 2.51322e-05] | 20 | mean Delta_P=-2.62906e-05 |
| s025 | alpha_TAM | F11 Delta_P narrow_1x | Delta_P includes 0 | [-2.62843e-05, 1.31843e-05] | 20 | mean Delta_P=-6.55002e-06 |
| s025 | alpha_TP | F11 Delta_P narrow_1x | Delta_P includes 0 | [-8.56731e-06, 5.26695e-08] | 20 | mean Delta_P=-4.25732e-06 |
| s025 | alpha_Tfixed | F11 Delta_P narrow_1x | Delta_P includes 0 | [-2.62733e-05, 1.31759e-05] | 20 | mean Delta_P=-6.54873e-06 |
| s025 | alpha_TAM | F11 Delta_P narrow_2x | Delta_P includes 0 | [-3.27619e-06, 5.87662e-06] | 20 | mean Delta_P=1.30022e-06 |
| s025 | alpha_TP | F11 Delta_P narrow_2x | Delta_P includes 0 | [-2.27074e-06, 3.88399e-07] | 20 | mean Delta_P=-9.41172e-07 |
| s025 | alpha_Tfixed | F11 Delta_P narrow_2x | Delta_P includes 0 | [-3.2623e-06, 5.88597e-06] | 20 | mean Delta_P=1.31184e-06 |
| s025 | alpha_TAM | F11 Delta_P twin | Delta_P includes 0 | [-0.000145029, 5.55133e-05] | 20 | mean Delta_P=-4.47578e-05 |
| s025 | alpha_TP | F11 Delta_P twin | Delta_P below 0 | [-3.81225e-05, -1.53903e-05] | 20 | mean Delta_P=-2.67564e-05 |
| s025 | alpha_Tfixed | F11 Delta_P twin | Delta_P includes 0 | [-0.000144991, 5.54058e-05] | 20 | mean Delta_P=-4.47926e-05 |
| s025 | alpha_TAM | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-6.24796e-05, 2.28788e-05] | 20 | mean=-1.98004e-05 |
| s025 | alpha_TP | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-2.07533e-05, 5.95767e-06] | 20 | mean=-7.39784e-06 |
| s025 | alpha_Tfixed | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-6.25438e-05, 2.306e-05] | 20 | mean=-1.97419e-05 |
| s025 | alpha_TAM | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-2.47922e-05, 9.09177e-06] | 20 | mean=-7.85023e-06 |
| s025 | alpha_TP | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-6.64431e-06, 1.20168e-08] | 20 | mean=-3.31615e-06 |
| s025 | alpha_Tfixed | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-2.47951e-05, 9.07393e-06] | 20 | mean=-7.86057e-06 |
| s025 | alpha_TAM | F11 difference in differences twin_minus_production | difference includes 0 | [-8.49247e-05, 8.30462e-05] | 20 | mean=-9.39265e-07 |
| s025 | alpha_TP | F11 difference in differences twin_minus_production | difference below 0 | [-2.78738e-05, -1.15191e-05] | 20 | mean=-1.96964e-05 |
| s025 | alpha_Tfixed | F11 difference in differences twin_minus_production | difference includes 0 | [-8.49008e-05, 8.29729e-05] | 20 | mean=-9.63955e-07 |
| s1 | alpha_TP | F7 Delta_half | not shown decreasing | [-0.00643519, 0.00115292] | 50 | mean Delta_half=-0.00264114 |
| s1 | rho(X_b), pair 8-16 | F8 readings | pathwise resolved |  | 20 | median=5.37004e-08 p95=8.31056e-08 (m=0.00125, S=0.387342) |
| s1 | alpha_TAM | F11 Delta_P production | Delta_P includes 0 | [-0.000186375, 0.000102516] | 50 | mean Delta_P=-4.19296e-05 |
| s1 | alpha_TP | F11 Delta_P production | Delta_P includes 0 | [-7.83217e-05, 2.6016e-05] | 50 | mean Delta_P=-2.61528e-05 |
| s1 | alpha_Tfixed | F11 Delta_P production | Delta_P includes 0 | [-0.000187466, 0.000101399] | 50 | mean Delta_P=-4.30339e-05 |
| s1 | alpha_TAM | F11 Delta_P narrow_half | Delta_P includes 0 | [-0.000435164, 0.000157469] | 20 | mean Delta_P=-0.000138847 |
| s1 | alpha_TP | F11 Delta_P narrow_half | Delta_P below 0 | [-0.000426843, -3.18243e-05] | 20 | mean Delta_P=-0.000229334 |
| s1 | alpha_Tfixed | F11 Delta_P narrow_half | Delta_P includes 0 | [-0.000447737, 0.000152104] | 20 | mean Delta_P=-0.000147817 |
| s1 | alpha_TAM | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000185368, 6.27616e-05] | 20 | mean Delta_P=-6.1303e-05 |
| s1 | alpha_TP | F11 Delta_P narrow_1x | Delta_P below 0 | [-0.000127339, -1.88507e-05] | 20 | mean Delta_P=-7.3095e-05 |
| s1 | alpha_Tfixed | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.00018534, 6.20825e-05] | 20 | mean Delta_P=-6.16285e-05 |
| s1 | alpha_TAM | F11 Delta_P narrow_2x | Delta_P includes 0 | [-7.80215e-05, 8.67154e-05] | 20 | mean Delta_P=4.34694e-06 |
| s1 | alpha_TP | F11 Delta_P narrow_2x | Delta_P includes 0 | [-3.04902e-05, 2.60078e-06] | 20 | mean Delta_P=-1.39447e-05 |
| s1 | alpha_Tfixed | F11 Delta_P narrow_2x | Delta_P includes 0 | [-7.82868e-05, 8.68736e-05] | 20 | mean Delta_P=4.29338e-06 |
| s1 | alpha_TAM | F11 Delta_P twin | Delta_P includes 0 | [-0.00035312, 0.000504434] | 20 | mean Delta_P=7.56568e-05 |
| s1 | alpha_TP | F11 Delta_P twin | Delta_P below 0 | [-0.00039058, -9.42999e-08] | 20 | mean Delta_P=-0.000195337 |
| s1 | alpha_Tfixed | F11 Delta_P twin | Delta_P includes 0 | [-0.00035364, 0.000504917] | 20 | mean Delta_P=7.56389e-05 |
| s1 | alpha_TAM | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.000398082, 0.000242993] | 20 | mean=-7.75444e-05 |
| s1 | alpha_TP | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.000314335, 1.85778e-06] | 20 | mean=-0.000156239 |
| s1 | alpha_Tfixed | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.000409852, 0.000237476] | 20 | mean=-8.6188e-05 |
| s1 | alpha_TAM | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000202396, 7.10961e-05] | 20 | mean=-6.565e-05 |
| s1 | alpha_TP | F11 difference in differences narrow_1x_minus_2x | difference below 0 | [-0.000107962, -1.03384e-05] | 20 | mean=-5.91503e-05 |
| s1 | alpha_Tfixed | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000201836, 6.99926e-05] | 20 | mean=-6.59219e-05 |
| s1 | alpha_TAM | F11 difference in differences twin_minus_production | difference includes 0 | [-0.000273447, 0.000652481] | 20 | mean=0.000189517 |
| s1 | alpha_TP | F11 difference in differences twin_minus_production | difference below 0 | [-0.000385762, -5.22942e-05] | 20 | mean=-0.000219028 |
| s1 | alpha_Tfixed | F11 difference in differences twin_minus_production | difference includes 0 | [-0.000272033, 0.00065633] | 20 | mean=0.000192149 |
| s225 | alpha_TP | F7 Delta_half | not shown decreasing | [-0.00671303, 0.00336172] | 62 | mean Delta_half=-0.00167565 |
| s225 | rho(X_b), pair 8-16 | F8 readings | intermediate (none of the three readings) |  | 20 | median=0.000141543 p95=0.000725131 (m=0.00147528, S=0.428558) |
| s225 | alpha_TAM | F11 Delta_P production | Delta_P includes 0 | [-0.000424675, 0.000340994] | 62 | mean Delta_P=-4.18404e-05 |
| s225 | alpha_TP | F11 Delta_P production | Delta_P includes 0 | [-0.000332789, 9.53276e-05] | 62 | mean Delta_P=-0.000118731 |
| s225 | alpha_Tfixed | F11 Delta_P production | Delta_P includes 0 | [-0.000434931, 0.000335817] | 62 | mean Delta_P=-4.95568e-05 |
| s225 | alpha_TAM | F11 Delta_P narrow_half | Delta_P includes 0 | [-0.00128453, 0.000303805] | 20 | mean Delta_P=-0.000490363 |
| s225 | alpha_TP | F11 Delta_P narrow_half | Delta_P includes 0 | [-0.000646863, 8.91488e-05] | 20 | mean Delta_P=-0.000278857 |
| s225 | alpha_Tfixed | F11 Delta_P narrow_half | Delta_P includes 0 | [-0.0014023, 0.000200612] | 20 | mean Delta_P=-0.000600842 |
| s225 | alpha_TAM | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000436835, 0.000383136] | 20 | mean Delta_P=-2.68496e-05 |
| s225 | alpha_TP | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000134061, 0.000327665] | 20 | mean Delta_P=9.6802e-05 |
| s225 | alpha_Tfixed | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000417479, 0.000412666] | 20 | mean Delta_P=-2.40657e-06 |
| s225 | alpha_TAM | F11 Delta_P narrow_2x | Delta_P includes 0 | [-0.000277933, 0.000166297] | 20 | mean Delta_P=-5.58175e-05 |
| s225 | alpha_TP | F11 Delta_P narrow_2x | Delta_P includes 0 | [-0.000191414, 0.000145695] | 20 | mean Delta_P=-2.28595e-05 |
| s225 | alpha_Tfixed | F11 Delta_P narrow_2x | Delta_P includes 0 | [-0.000306044, 0.000143381] | 20 | mean Delta_P=-8.13314e-05 |
| s225 | alpha_TAM | F11 Delta_P twin | Delta_P includes 0 | [-0.0014632, 0.000645925] | 26 | mean Delta_P=-0.000408638 |
| s225 | alpha_TP | F11 Delta_P twin | Delta_P includes 0 | [-0.000652842, 0.000583532] | 26 | mean Delta_P=-3.4655e-05 |
| s225 | alpha_Tfixed | F11 Delta_P twin | Delta_P includes 0 | [-0.00146032, 0.000659679] | 26 | mean Delta_P=-0.000400322 |
| s225 | alpha_TAM | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.00127641, 0.000349387] | 20 | mean=-0.000463514 |
| s225 | alpha_TP | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.000802666, 5.13481e-05] | 20 | mean=-0.000375659 |
| s225 | alpha_Tfixed | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.00141559, 0.000218718] | 20 | mean=-0.000598435 |
| s225 | alpha_TAM | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000415263, 0.000473199] | 20 | mean=2.8968e-05 |
| s225 | alpha_TP | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000127858, 0.000367181] | 20 | mean=0.000119662 |
| s225 | alpha_Tfixed | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000363163, 0.000521012] | 20 | mean=7.89249e-05 |
| s225 | alpha_TAM | F11 difference in differences twin_minus_production | difference includes 0 | [-0.00175333, 0.000691443] | 26 | mean=-0.000530944 |
| s225 | alpha_TP | F11 difference in differences twin_minus_production | difference includes 0 | [-0.000790772, 0.000616907] | 26 | mean=-8.69328e-05 |
| s225 | alpha_Tfixed | F11 difference in differences twin_minus_production | difference includes 0 | [-0.00173026, 0.000726109] | 26 | mean=-0.000502073 |
| s4 | alpha_TP | F7 Delta_half | not shown decreasing | [-0.00858636, 0.00676019] | 71 | mean Delta_half=-0.000913087 |
| s4 | rho(X_b), pair 8-16 | F8 readings | pathwise resolved |  | 20 | median=1.3892e-05 p95=0.000143805 (m=0.00320762, S=0.435013) |
| s4 | alpha_TAM | F11 Delta_P production | Delta_P includes 0 | [-0.000470114, 6.87079e-05] | 71 | mean Delta_P=-0.000200703 |
| s4 | alpha_TP | F11 Delta_P production | Delta_P includes 0 | [-0.00015048, 0.00037593] | 71 | mean Delta_P=0.000112725 |
| s4 | alpha_Tfixed | F11 Delta_P production | Delta_P includes 0 | [-0.00050369, 4.33945e-05] | 71 | mean Delta_P=-0.000230148 |
| s4 | alpha_TAM | F11 Delta_P narrow_half | Delta_P above 0 | [8.90982e-05, 0.00104871] | 20 | mean Delta_P=0.000568906 |
| s4 | alpha_TP | F11 Delta_P narrow_half | Delta_P includes 0 | [-0.000396899, 7.72057e-05] | 20 | mean Delta_P=-0.000159846 |
| s4 | alpha_Tfixed | F11 Delta_P narrow_half | Delta_P above 0 | [0.000163591, 0.00108213] | 20 | mean Delta_P=0.00062286 |
| s4 | alpha_TAM | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000145173, 0.000178299] | 20 | mean Delta_P=1.6563e-05 |
| s4 | alpha_TP | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000287204, 3.85062e-05] | 20 | mean Delta_P=-0.000124349 |
| s4 | alpha_Tfixed | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000120184, 0.00016716] | 20 | mean Delta_P=2.3488e-05 |
| s4 | alpha_TAM | F11 Delta_P narrow_2x | Delta_P includes 0 | [-5.4731e-05, 0.000119642] | 20 | mean Delta_P=3.24555e-05 |
| s4 | alpha_TP | F11 Delta_P narrow_2x | Delta_P includes 0 | [-7.03452e-05, 0.000168325] | 20 | mean Delta_P=4.89898e-05 |
| s4 | alpha_Tfixed | F11 Delta_P narrow_2x | Delta_P includes 0 | [-7.18347e-05, 0.0001017] | 20 | mean Delta_P=1.49328e-05 |
| s4 | alpha_TAM | F11 Delta_P twin | Delta_P above 0 | [0.00042044, 0.00299174] | 20 | mean Delta_P=0.00170609 |
| s4 | alpha_TP | F11 Delta_P twin | Delta_P below 0 | [-0.0011444, -3.47944e-05] | 20 | mean Delta_P=-0.000589598 |
| s4 | alpha_Tfixed | F11 Delta_P twin | Delta_P above 0 | [0.000406437, 0.0030188] | 20 | mean Delta_P=0.00171262 |
| s4 | alpha_TAM | F11 difference in differences narrow_half_minus_1x | difference above 0 | [0.000163689, 0.000940997] | 20 | mean=0.000552343 |
| s4 | alpha_TP | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.000328952, 0.000257957] | 20 | mean=-3.54978e-05 |
| s4 | alpha_Tfixed | F11 difference in differences narrow_half_minus_1x | difference above 0 | [0.000220871, 0.000977873] | 20 | mean=0.000599372 |
| s4 | alpha_TAM | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000161679, 0.000129894] | 20 | mean=-1.58924e-05 |
| s4 | alpha_TP | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000376106, 2.94288e-05] | 20 | mean=-0.000173338 |
| s4 | alpha_Tfixed | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000125471, 0.000142581] | 20 | mean=8.55524e-06 |
| s4 | alpha_TAM | F11 difference in differences twin_minus_production | difference above 0 | [0.000415771, 0.00287489] | 20 | mean=0.00164533 |
| s4 | alpha_TP | F11 difference in differences twin_minus_production | difference includes 0 | [-0.000918347, 0.000428867] | 20 | mean=-0.00024474 |
| s4 | alpha_Tfixed | F11 difference in differences twin_minus_production | difference above 0 | [0.000423462, 0.00293935] | 20 | mean=0.00168141 |

## exploratory

| amplitude | estimator | decision point | verdict | CI | n | note |
|---|---|---|---|---|---|---|
| s4 | alpha_TAM | F10 width check w24_vs_tw | inconclusive | [-0.0190798, 0.00576286] | 30/30 | diff=-0.00665847 margin=0.0068662 |
| s4 | alpha_TP | F10 width check w24_vs_tw | inconclusive | [-0.00697705, 0.0139799] | 30/30 | diff=0.00350142 margin=0.0067672 |
| s4 | alpha_TAM | F10 width check w12_vs_n16 | disagree | [0.0215526, 0.0472036] | 30/30 | diff=0.0343781 margin=0.00635197; wide = s4_w12+s4_w12c; seed 7214 excluded (s4_w12 halted, owner decision 2026-10-28) |
| s4 | alpha_TP | F10 width check w12_vs_n16 | disagree | [0.0386602, 0.0541048] | 30/30 | diff=0.0463825 margin=0.00678013; wide = s4_w12+s4_w12c; seed 7214 excluded (s4_w12 halted, owner decision 2026-10-28) |

## process

| amplitude | estimator | decision point | verdict | CI | n | note |
|---|---|---|---|---|---|---|
| all |  | CHECK_unittest | ok |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_geometry | ok |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_regress_refs | reused(/home/sesquerre/regress/refs/1e40cfe/out/REGRESS_DONE) |  |  | rc=NA (from checks/results.txt) |
| all |  | CHECK_regress_head | REGRESS_DONE |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_regress | REGRESS |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s1_n05 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s1_n10 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s1_n20 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s1_tw | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s225_n08 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s225_n16 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s225_n32 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s225_tw | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s4_n16 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s4_n32 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s4_n64 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s4_tw | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s4_w24 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s4_w12 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s025_n05 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s025_n10 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s025_n20 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s025_tw | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_bytecheck | BYTECHECK |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_smoke_reduce | SMOKE |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_signtable | SIGNTABLE |  |  | rc=0 (from checks/results.txt) |
