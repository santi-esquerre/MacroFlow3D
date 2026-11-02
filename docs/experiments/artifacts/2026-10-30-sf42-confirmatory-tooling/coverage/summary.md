# Coverage simulation summary

Criterion (decision G5.3): one-sided miss rate with 95% one-sided Clopper-Pearson upper limit (UCL) at most 1.25 x the nominal tail, at least 175 replicates x tail per cell. Cells complete: 1134 of 1134. Core-seconds: 10782196.

Seed 20261030, profile full, plan hash `0b08b6be86ce6784`; environment: {"cpu_count": 80, "numpy": "1.26.4", "platform": "Linux-3.10.0-1160.119.1.el7.x86_64-x86_64-with-glibc2.17", "python": "3.11.7", "scipy": "1.11.4"}.

## Tail levels

- sampling limits (Student, bootstrap-t; lower and upper): 0.00357, 0.00417, 0.005, 0.00625, 0.00833, 0.0125, 0.025, 0.05
- budget bounds (ContrastBound, LadderBound; K = 3, 4): 0.000893, 0.00104, 0.00119, 0.00125, 0.00139, 0.00156, 0.00167, 0.00208, 0.00278, 0.00313, 0.00417, 0.00625, 0.00833
- Gate 5 TOST and whole robust-sign pipeline (Holm levels, m = 7): 0.00714, 0.00833, 0.01, 0.0125, 0.0167, 0.025, 0.05

## Qualification matrix

Entry: `ALL` if every tail passes in every simulated distribution, else `passing/total tails`; `rep` = reported, never judged (BCa); `n/a` = fewer than 175 expected misses at nominal, not judged; `inc` = a planned cell (distribution) of the group is not finished, so no verdict (`k/t inc`: already failing in the finished ones).

| procedure | row | scenario | n=20 | n=30 | n=50 | n=100 | n=170 | n=200 | n=300 | n=500 | n=1000 | n=1650 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bca:mean | lower | - | rep | rep | rep | rep |  |  |  |  |  |  |
| bca:mean | upper | - | rep | rep | rep | rep |  |  |  |  |  |  |
| contrast:interaction | bound | - | 0/13 | 0/13 | 0/13 | 2/13 |  | 9/13 |  |  |  |  |
| contrast:interaction/boott | bound | - | 0/13 | 0/13 | 0/13 | 0/13 |  | 3/13 |  |  |  |  |
| contrast:interaction/student | bound | - | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| contrast:paired | bound | - | 0/13 | 0/13 | 0/13 | 0/13 |  | 0/13 |  |  |  |  |
| contrast:paired/boott | bound | - | 0/13 | 0/13 | 0/13 | 0/13 |  | 3/13 |  |  |  |  |
| contrast:paired/student | bound | - | 0/13 | 0/13 | 0/13 | 0/13 |  | 0/13 |  |  |  |  |
| contrast:unpaired | bound | - | 0/13 | 0/13 | 0/13 | 1/13 |  | 10/13 |  |  |  |  |
| contrast:unpaired/boott | bound | - | 0/13 | 0/13 | 0/13 | 0/13 |  | 2/13 |  |  |  |  |
| contrast:unpaired/student | bound | - | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | nonmono,ip=1,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | nonmono,ip=1,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | nonmono,ip=1,kappa=50 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | nonmono,ip=2,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | nonmono,ip=2,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | nonmono,ip=2,kappa=50 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=1,ip=1,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=1,ip=1,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=1,ip=1,kappa=50 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=1,ip=2,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=1,ip=2,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=1,ip=2,kappa=50 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=1.5,ip=1,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=1.5,ip=1,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=1.5,ip=1,kappa=50 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=1.5,ip=2,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=1.5,ip=2,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=1.5,ip=2,kappa=50 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=2,ip=1,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=2,ip=1,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=2,ip=1,kappa=50 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=2,ip=2,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=2,ip=2,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h | bound | order=2,ip=2,kappa=50 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/boott | bound | nonmono,ip=1,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/boott | bound | nonmono,ip=2,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/boott | bound | order=1,ip=1,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/boott | bound | order=1,ip=1,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/boott | bound | order=1,ip=2,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/boott | bound | order=1,ip=2,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/boott | bound | order=2,ip=1,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/boott | bound | order=2,ip=1,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/boott | bound | order=2,ip=2,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/boott | bound | order=2,ip=2,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/student | bound | nonmono,ip=1,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/student | bound | nonmono,ip=2,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/student | bound | order=1,ip=1,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/student | bound | order=1,ip=1,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/student | bound | order=1,ip=2,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/student | bound | order=1,ip=2,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/student | bound | order=2,ip=1,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/student | bound | order=2,ip=1,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/student | bound | order=2,ip=2,kappa=12 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| ladder:E_h/student | bound | order=2,ip=2,kappa=25 | ALL | ALL | ALL | ALL |  | ALL |  |  |  |  |
| pipeline:K3,ip=1 | RL | bias,order=1 |  |  |  | ALL |  |  | ALL |  | ALL |  |
| pipeline:K3,ip=1 | RL | bias,order=2 |  |  |  | ALL |  |  | ALL |  | ALL |  |
| pipeline:K3,ip=1 | RL | boott,bias,order=1 |  |  |  | ALL |  |  |  |  |  |  |
| pipeline:K3,ip=1 | RL | zero |  |  |  | ALL |  |  | ALL |  | ALL |  |
| pipeline:K3,ip=1/boott | RL | bias,order=1 |  |  |  | ALL |  |  | ALL |  | ALL |  |
| pipeline:K3,ip=1/boott | RL | bias,order=2 |  |  |  | ALL |  |  | ALL |  | ALL |  |
| pipeline:K3,ip=1/boott | RL | boott,bias,order=1 |  |  |  | ALL |  |  |  |  |  |  |
| pipeline:K3,ip=1/boott | RL | zero |  |  |  | ALL |  |  | ALL |  | ALL |  |
| pipeline:K3,ip=1/student | RL | bias,order=1 |  |  |  | ALL |  |  | ALL |  | ALL |  |
| pipeline:K3,ip=1/student | RL | bias,order=2 |  |  |  | ALL |  |  | ALL |  | ALL |  |
| pipeline:K3,ip=1/student | RL | boott,bias,order=1 |  |  |  | ALL |  |  |  |  |  |  |
| pipeline:K3,ip=1/student | RL | zero |  |  |  | ALL |  |  | ALL |  | ALL |  |
| pipeline:K4,ip=2 | RL | bias,order=1 |  | ALL |  | ALL |  |  | ALL |  |  |  |
| pipeline:K4,ip=2 | RL | bias,order=2 |  | ALL |  | ALL |  |  | ALL |  |  |  |
| pipeline:K4,ip=2 | RL | boott,bias,order=1 |  | ALL |  | ALL |  |  |  |  |  |  |
| pipeline:K4,ip=2 | RL | zero |  | ALL |  | ALL |  |  | ALL |  |  |  |
| pipeline:K4,ip=2/boott | RL | bias,order=1 |  | ALL |  | ALL |  |  | ALL |  |  |  |
| pipeline:K4,ip=2/boott | RL | bias,order=2 |  | ALL |  | ALL |  |  | ALL |  |  |  |
| pipeline:K4,ip=2/boott | RL | boott,bias,order=1 |  | ALL |  | ALL |  |  |  |  |  |  |
| pipeline:K4,ip=2/boott | RL | zero |  | ALL |  | ALL |  |  | ALL |  |  |  |
| pipeline:K4,ip=2/student | RL | bias,order=1 |  | ALL |  | ALL |  |  | ALL |  |  |  |
| pipeline:K4,ip=2/student | RL | bias,order=2 |  | ALL |  | ALL |  |  | ALL |  |  |  |
| pipeline:K4,ip=2/student | RL | boott,bias,order=1 |  | ALL |  | ALL |  |  |  |  |  |  |
| pipeline:K4,ip=2/student | RL | zero |  | ALL |  | ALL |  |  | ALL |  |  |  |
| sampling:boott | lower | - | 0/8 | 0/8 | 0/8 | 2/8 |  | ALL |  | 6/8 | ALL | ALL |
| sampling:boott | upper | - | 0/8 | 0/8 | 0/8 | 2/8 |  | 6/8 |  | ALL | ALL | ALL |
| sampling:student | lower | - | 0/8 | 0/8 | 0/8 | 0/8 |  | 0/8 |  | 1/8 | 1/8 | 1/8 |
| sampling:student | upper | - | 0/8 | 0/8 | 0/8 | 0/8 |  | 0/8 |  | 1/8 | 1/8 | 2/8 |
| tost:gate5 | ratio_hi | rho=0 | ALL | ALL | 4/7 | 0/7 | 0/7 |  |  |  |  |  |
| tost:gate5 | ratio_hi | rho=0.5 | 0/7 | 0/7 | 0/7 | 0/7 | 0/7 |  |  |  |  |  |
| tost:gate5 | ratio_hi | rho=0.9 | 0/7 | 0/7 | 0/7 | 0/7 | 0/7 |  |  |  |  |  |
| tost:gate5 | ratio_lo | rho=0 | ALL | ALL | 4/7 | 0/7 | 0/7 |  |  |  |  |  |
| tost:gate5 | ratio_lo | rho=0.5 | 0/7 | 0/7 | 0/7 | 0/7 | 0/7 |  |  |  |  |  |
| tost:gate5 | ratio_lo | rho=0.9 | 0/7 | 0/7 | 0/7 | 0/7 | 0/7 |  |  |  |  |  |
| tost:gate5/boott | ratio_hi | rho=0 | ALL | ALL | ALL | 2/7 | 4/7 |  |  |  |  |  |
| tost:gate5/boott | ratio_hi | rho=0.5 | 0/7 | 0/7 | 0/7 | 2/7 | ALL |  |  |  |  |  |
| tost:gate5/boott | ratio_hi | rho=0.9 | 0/7 | 0/7 | 0/7 | 2/7 | 4/7 |  |  |  |  |  |
| tost:gate5/boott | ratio_lo | rho=0 | ALL | ALL | ALL | 4/7 | 5/7 |  |  |  |  |  |
| tost:gate5/boott | ratio_lo | rho=0.5 | 0/7 | 0/7 | 0/7 | 2/7 | 5/7 |  |  |  |  |  |
| tost:gate5/boott | ratio_lo | rho=0.9 | 0/7 | 0/7 | 0/7 | 2/7 | 4/7 |  |  |  |  |  |

## Verdict list

Qualified = every tail passes in every simulated distribution of that cell group.

- **contrast:interaction / bound**: qualified at n = none; not qualified at n = 20 (0 of 13 tails), 30 (0 of 13 tails), 50 (0 of 13 tails), 100 (2 of 13 tails), 200 (9 of 13 tails)
- **contrast:interaction/boott / bound**: qualified at n = none; not qualified at n = 20 (0 of 13 tails), 30 (0 of 13 tails), 50 (0 of 13 tails), 100 (0 of 13 tails), 200 (3 of 13 tails)
- **contrast:interaction/student / bound**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **contrast:paired / bound**: qualified at n = none; not qualified at n = 20 (0 of 13 tails), 30 (0 of 13 tails), 50 (0 of 13 tails), 100 (0 of 13 tails), 200 (0 of 13 tails)
- **contrast:paired/boott / bound**: qualified at n = none; not qualified at n = 20 (0 of 13 tails), 30 (0 of 13 tails), 50 (0 of 13 tails), 100 (0 of 13 tails), 200 (3 of 13 tails)
- **contrast:paired/student / bound**: qualified at n = none; not qualified at n = 20 (0 of 13 tails), 30 (0 of 13 tails), 50 (0 of 13 tails), 100 (0 of 13 tails), 200 (0 of 13 tails)
- **contrast:unpaired / bound**: qualified at n = none; not qualified at n = 20 (0 of 13 tails), 30 (0 of 13 tails), 50 (0 of 13 tails), 100 (1 of 13 tails), 200 (10 of 13 tails)
- **contrast:unpaired/boott / bound**: qualified at n = none; not qualified at n = 20 (0 of 13 tails), 30 (0 of 13 tails), 50 (0 of 13 tails), 100 (0 of 13 tails), 200 (2 of 13 tails)
- **contrast:unpaired/student / bound**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / nonmono,ip=1,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / nonmono,ip=1,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / nonmono,ip=1,kappa=50**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / nonmono,ip=2,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / nonmono,ip=2,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / nonmono,ip=2,kappa=50**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=1,ip=1,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=1,ip=1,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=1,ip=1,kappa=50**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=1,ip=2,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=1,ip=2,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=1,ip=2,kappa=50**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=1.5,ip=1,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=1.5,ip=1,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=1.5,ip=1,kappa=50**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=1.5,ip=2,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=1.5,ip=2,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=1.5,ip=2,kappa=50**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=2,ip=1,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=2,ip=1,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=2,ip=1,kappa=50**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=2,ip=2,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=2,ip=2,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h / bound / order=2,ip=2,kappa=50**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/boott / bound / nonmono,ip=1,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/boott / bound / nonmono,ip=2,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/boott / bound / order=1,ip=1,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/boott / bound / order=1,ip=1,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/boott / bound / order=1,ip=2,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/boott / bound / order=1,ip=2,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/boott / bound / order=2,ip=1,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/boott / bound / order=2,ip=1,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/boott / bound / order=2,ip=2,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/boott / bound / order=2,ip=2,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/student / bound / nonmono,ip=1,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/student / bound / nonmono,ip=2,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/student / bound / order=1,ip=1,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/student / bound / order=1,ip=1,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/student / bound / order=1,ip=2,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/student / bound / order=1,ip=2,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/student / bound / order=2,ip=1,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/student / bound / order=2,ip=1,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/student / bound / order=2,ip=2,kappa=12**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **ladder:E_h/student / bound / order=2,ip=2,kappa=25**: qualified at n = 20, 30, 50, 100, 200; not qualified at n = none
- **pipeline:K3,ip=1 / RL / bias,order=1**: qualified at n = 100, 300, 1000; not qualified at n = none
- **pipeline:K3,ip=1 / RL / bias,order=2**: qualified at n = 100, 300, 1000; not qualified at n = none
- **pipeline:K3,ip=1 / RL / boott,bias,order=1**: qualified at n = 100; not qualified at n = none
- **pipeline:K3,ip=1 / RL / zero**: qualified at n = 100, 300, 1000; not qualified at n = none
- **pipeline:K3,ip=1/boott / RL / bias,order=1**: qualified at n = 100, 300, 1000; not qualified at n = none
- **pipeline:K3,ip=1/boott / RL / bias,order=2**: qualified at n = 100, 300, 1000; not qualified at n = none
- **pipeline:K3,ip=1/boott / RL / boott,bias,order=1**: qualified at n = 100; not qualified at n = none
- **pipeline:K3,ip=1/boott / RL / zero**: qualified at n = 100, 300, 1000; not qualified at n = none
- **pipeline:K3,ip=1/student / RL / bias,order=1**: qualified at n = 100, 300, 1000; not qualified at n = none
- **pipeline:K3,ip=1/student / RL / bias,order=2**: qualified at n = 100, 300, 1000; not qualified at n = none
- **pipeline:K3,ip=1/student / RL / boott,bias,order=1**: qualified at n = 100; not qualified at n = none
- **pipeline:K3,ip=1/student / RL / zero**: qualified at n = 100, 300, 1000; not qualified at n = none
- **pipeline:K4,ip=2 / RL / bias,order=1**: qualified at n = 30, 100, 300; not qualified at n = none
- **pipeline:K4,ip=2 / RL / bias,order=2**: qualified at n = 30, 100, 300; not qualified at n = none
- **pipeline:K4,ip=2 / RL / boott,bias,order=1**: qualified at n = 30, 100; not qualified at n = none
- **pipeline:K4,ip=2 / RL / zero**: qualified at n = 30, 100, 300; not qualified at n = none
- **pipeline:K4,ip=2/boott / RL / bias,order=1**: qualified at n = 30, 100, 300; not qualified at n = none
- **pipeline:K4,ip=2/boott / RL / bias,order=2**: qualified at n = 30, 100, 300; not qualified at n = none
- **pipeline:K4,ip=2/boott / RL / boott,bias,order=1**: qualified at n = 30, 100; not qualified at n = none
- **pipeline:K4,ip=2/boott / RL / zero**: qualified at n = 30, 100, 300; not qualified at n = none
- **pipeline:K4,ip=2/student / RL / bias,order=1**: qualified at n = 30, 100, 300; not qualified at n = none
- **pipeline:K4,ip=2/student / RL / bias,order=2**: qualified at n = 30, 100, 300; not qualified at n = none
- **pipeline:K4,ip=2/student / RL / boott,bias,order=1**: qualified at n = 30, 100; not qualified at n = none
- **pipeline:K4,ip=2/student / RL / zero**: qualified at n = 30, 100, 300; not qualified at n = none
- **sampling:boott / lower**: qualified at n = 200, 1000, 1650; not qualified at n = 20 (0 of 8 tails), 30 (0 of 8 tails), 50 (0 of 8 tails), 100 (2 of 8 tails), 500 (6 of 8 tails)
- **sampling:boott / upper**: qualified at n = 500, 1000, 1650; not qualified at n = 20 (0 of 8 tails), 30 (0 of 8 tails), 50 (0 of 8 tails), 100 (2 of 8 tails), 200 (6 of 8 tails)
- **sampling:student / lower**: qualified at n = none; not qualified at n = 20 (0 of 8 tails), 30 (0 of 8 tails), 50 (0 of 8 tails), 100 (0 of 8 tails), 200 (0 of 8 tails), 500 (1 of 8 tails), 1000 (1 of 8 tails), 1650 (1 of 8 tails)
- **sampling:student / upper**: qualified at n = none; not qualified at n = 20 (0 of 8 tails), 30 (0 of 8 tails), 50 (0 of 8 tails), 100 (0 of 8 tails), 200 (0 of 8 tails), 500 (1 of 8 tails), 1000 (1 of 8 tails), 1650 (2 of 8 tails)
- **tost:gate5 / ratio_hi / rho=0**: qualified at n = 20, 30; not qualified at n = 50 (4 of 7 tails), 100 (0 of 7 tails), 170 (0 of 7 tails)
- **tost:gate5 / ratio_hi / rho=0.5**: qualified at n = none; not qualified at n = 20 (0 of 7 tails), 30 (0 of 7 tails), 50 (0 of 7 tails), 100 (0 of 7 tails), 170 (0 of 7 tails)
- **tost:gate5 / ratio_hi / rho=0.9**: qualified at n = none; not qualified at n = 20 (0 of 7 tails), 30 (0 of 7 tails), 50 (0 of 7 tails), 100 (0 of 7 tails), 170 (0 of 7 tails)
- **tost:gate5 / ratio_lo / rho=0**: qualified at n = 20, 30; not qualified at n = 50 (4 of 7 tails), 100 (0 of 7 tails), 170 (0 of 7 tails)
- **tost:gate5 / ratio_lo / rho=0.5**: qualified at n = none; not qualified at n = 20 (0 of 7 tails), 30 (0 of 7 tails), 50 (0 of 7 tails), 100 (0 of 7 tails), 170 (0 of 7 tails)
- **tost:gate5 / ratio_lo / rho=0.9**: qualified at n = none; not qualified at n = 20 (0 of 7 tails), 30 (0 of 7 tails), 50 (0 of 7 tails), 100 (0 of 7 tails), 170 (0 of 7 tails)
- **tost:gate5/boott / ratio_hi / rho=0**: qualified at n = 20, 30, 50; not qualified at n = 100 (2 of 7 tails), 170 (4 of 7 tails)
- **tost:gate5/boott / ratio_hi / rho=0.5**: qualified at n = 170; not qualified at n = 20 (0 of 7 tails), 30 (0 of 7 tails), 50 (0 of 7 tails), 100 (2 of 7 tails)
- **tost:gate5/boott / ratio_hi / rho=0.9**: qualified at n = none; not qualified at n = 20 (0 of 7 tails), 30 (0 of 7 tails), 50 (0 of 7 tails), 100 (2 of 7 tails), 170 (4 of 7 tails)
- **tost:gate5/boott / ratio_lo / rho=0**: qualified at n = 20, 30, 50; not qualified at n = 100 (4 of 7 tails), 170 (5 of 7 tails)
- **tost:gate5/boott / ratio_lo / rho=0.5**: qualified at n = none; not qualified at n = 20 (0 of 7 tails), 30 (0 of 7 tails), 50 (0 of 7 tails), 100 (2 of 7 tails), 170 (5 of 7 tails)
- **tost:gate5/boott / ratio_lo / rho=0.9**: qualified at n = none; not qualified at n = 20 (0 of 7 tails), 30 (0 of 7 tails), 50 (0 of 7 tails), 100 (2 of 7 tails), 170 (4 of 7 tails)

## False-fail probability of an exactly nominal procedure

A procedure whose true miss rate equals every tail exactly still fails the verdict by Monte Carlo noise alone: per tail `P = P(Bin(R, tail) >= k*)`, `k*` the smallest miss count with UCL > 1.25 x tail; at least one tail of a (cell, distribution, row) fails with `P(any)` (exact: the misses at nested tails are nested); `P(any of 3)` = at least one of three independent distributions (normal, stress, stress-). Design effect 1 assumed (shared bootstrap draws only raise these).

**R = 24500, 7 tails, 234 cells** (pipeline:K3,ip=1, pipeline:K3,ip=1/boott, pipeline:K3,ip=1/student, pipeline:K4,ip=2, pipeline:K4,ip=2/boott, pipeline:K4,ip=2/student, ...): P(any) = 0.0815, P(any of 3) = 0.2250

| tail | k* | k*/(R tail) | P(fail) |
|---|---|---|---|
| 0.00714 | 195 | 1.114 | 0.0714 |
| 0.00833 | 229 | 1.122 | 0.0456 |
| 0.01 | 278 | 1.135 | 0.02 |
| 0.0125 | 351 | 1.146 | 0.00627 |
| 0.0167 | 474 | 1.161 | 0.000736 |
| 0.025 | 721 | 1.177 | 8.25e-06 |
| 0.05 | 1469 | 1.199 | 2.03e-12 |

**R = 25000, 7 tails, 45 cells** (tost:gate5): P(any) = 0.0779, P(any of 3) = 0.2159

| tail | k* | k*/(R tail) | P(fail) |
|---|---|---|---|
| 0.00714 | 199 | 1.114 | 0.0691 |
| 0.00833 | 234 | 1.123 | 0.0419 |
| 0.01 | 284 | 1.136 | 0.0181 |
| 0.0125 | 359 | 1.149 | 0.00512 |
| 0.0167 | 484 | 1.162 | 0.000623 |
| 0.025 | 736 | 1.178 | 6.42e-06 |
| 0.05 | 1500 | 1.200 | 1.01e-12 |

**R = 49000, 8 tails, 48 cells** (sampling:boott, sampling:student): P(any) = 0.0819, P(any of 3) = 0.2261

| tail | k* | k*/(R tail) | P(fail) |
|---|---|---|---|
| 0.00357 | 195 | 1.114 | 0.0717 |
| 0.00417 | 229 | 1.122 | 0.0459 |
| 0.005 | 278 | 1.135 | 0.0203 |
| 0.00625 | 351 | 1.146 | 0.00642 |
| 0.00833 | 474 | 1.161 | 0.000772 |
| 0.0125 | 721 | 1.177 | 9.39e-06 |
| 0.025 | 1468 | 1.198 | 4.79e-12 |
| 0.05 | 2975 | 1.214 | 2.91e-26 |

**R = 196000, 13 tails, 795 cells** (contrast:interaction, contrast:interaction/boott, contrast:interaction/student, contrast:paired, contrast:paired/boott, contrast:paired/student, ...): P(any) = 0.0840, P(any of 3) = 0.2314

| tail | k* | k*/(R tail) | P(fail) |
|---|---|---|---|
| 0.000893 | 195 | 1.114 | 0.072 |
| 0.00104 | 229 | 1.122 | 0.0462 |
| 0.00119 | 264 | 1.131 | 0.0258 |
| 0.00125 | 278 | 1.135 | 0.0204 |
| 0.00139 | 310 | 1.139 | 0.0131 |
| 0.00156 | 351 | 1.146 | 0.00653 |
| 0.00167 | 375 | 1.148 | 0.0047 |
| 0.00208 | 474 | 1.161 | 0.0008 |
| 0.00278 | 638 | 1.172 | 4.92e-05 |
| 0.00313 | 720 | 1.176 | 1.22e-05 |
| 0.00417 | 969 | 1.187 | 1.13e-07 |
| 0.00625 | 1467 | 1.198 | 9.27e-12 |
| 0.00833 | 1968 | 1.205 | 4.21e-16 |


## Failures (1285 tail x distribution entries)

`marginal`: the point estimate of the miss rate is within 1.25 x tail and only the 95% limit exceeds it. This is compatible with an exactly nominal procedure (see the false-fail probabilities above: about 0.07 for the smallest tail, about 0.08 for some tail of a cell and distribution); the remedy is more replicates at SF-43's final `n`, not a different criterion. Without the mark, the point estimate itself exceeds 1.25 x tail: a clear failure.

| procedure | row | scenario | n | dist | tail | rate/tail | UCL/tail | |
|---|---|---|---|---|---|---|---|---|
| contrast:interaction | bound | - | 20 | normal | 0.000893 | 2.17 | 2.36 |  |
| contrast:interaction | bound | - | 20 | stress | 0.000893 | 3.42 | 3.66 |  |
| contrast:interaction | bound | - | 20 | stress- | 0.000893 | 3.43 | 3.67 |  |
| contrast:interaction | bound | - | 20 | normal | 0.00104 | 2.16 | 2.34 |  |
| contrast:interaction | bound | - | 20 | stress | 0.00104 | 3.27 | 3.48 |  |
| contrast:interaction | bound | - | 20 | stress- | 0.00104 | 3.34 | 3.55 |  |
| contrast:interaction | bound | - | 20 | normal | 0.00119 | 2.07 | 2.23 |  |
| contrast:interaction | bound | - | 20 | stress | 0.00119 | 3.22 | 3.42 |  |
| contrast:interaction | bound | - | 20 | stress- | 0.00119 | 3.15 | 3.35 |  |
| contrast:interaction | bound | - | 20 | normal | 0.00125 | 2.07 | 2.23 |  |
| contrast:interaction | bound | - | 20 | stress | 0.00125 | 3.25 | 3.45 |  |
| contrast:interaction | bound | - | 20 | stress- | 0.00125 | 3.13 | 3.32 |  |
| contrast:interaction | bound | - | 20 | normal | 0.00139 | 1.99 | 2.14 |  |
| contrast:interaction | bound | - | 20 | stress | 0.00139 | 3.04 | 3.22 |  |
| contrast:interaction | bound | - | 20 | stress- | 0.00139 | 2.95 | 3.12 |  |
| contrast:interaction | bound | - | 20 | normal | 0.00156 | 1.97 | 2.11 |  |
| contrast:interaction | bound | - | 20 | stress | 0.00156 | 2.93 | 3.10 |  |
| contrast:interaction | bound | - | 20 | stress- | 0.00156 | 2.87 | 3.03 |  |
| contrast:interaction | bound | - | 20 | normal | 0.00167 | 2.00 | 2.13 |  |
| contrast:interaction | bound | - | 20 | stress | 0.00167 | 2.98 | 3.14 |  |
| contrast:interaction | bound | - | 20 | stress- | 0.00167 | 2.92 | 3.08 |  |
| contrast:interaction | bound | - | 20 | normal | 0.00208 | 1.92 | 2.03 |  |
| contrast:interaction | bound | - | 20 | stress | 0.00208 | 2.85 | 2.99 |  |
| contrast:interaction | bound | - | 20 | stress- | 0.00208 | 2.71 | 2.84 |  |
| contrast:interaction | bound | - | 20 | normal | 0.00278 | 1.83 | 1.92 |  |
| contrast:interaction | bound | - | 20 | stress | 0.00278 | 2.61 | 2.73 |  |
| contrast:interaction | bound | - | 20 | stress- | 0.00278 | 2.53 | 2.65 |  |
| contrast:interaction | bound | - | 20 | normal | 0.00313 | 1.76 | 1.85 |  |
| contrast:interaction | bound | - | 20 | stress | 0.00313 | 2.52 | 2.63 |  |
| contrast:interaction | bound | - | 20 | stress- | 0.00313 | 2.48 | 2.58 |  |
| contrast:interaction | bound | - | 20 | normal | 0.00417 | 1.71 | 1.78 |  |
| contrast:interaction | bound | - | 20 | stress | 0.00417 | 2.37 | 2.46 |  |
| contrast:interaction | bound | - | 20 | stress- | 0.00417 | 2.35 | 2.44 |  |
| contrast:interaction | bound | - | 20 | normal | 0.00625 | 1.59 | 1.65 |  |
| contrast:interaction | bound | - | 20 | stress | 0.00625 | 2.12 | 2.19 |  |
| contrast:interaction | bound | - | 20 | stress- | 0.00625 | 2.09 | 2.16 |  |
| contrast:interaction | bound | - | 20 | normal | 0.00833 | 1.51 | 1.56 |  |
| contrast:interaction | bound | - | 20 | stress | 0.00833 | 1.96 | 2.02 |  |
| contrast:interaction | bound | - | 20 | stress- | 0.00833 | 1.95 | 2.01 |  |
| contrast:interaction | bound | - | 30 | normal | 0.000893 | 1.63 | 1.80 |  |
| contrast:interaction | bound | - | 30 | stress | 0.000893 | 2.68 | 2.89 |  |
| contrast:interaction | bound | - | 30 | stress- | 0.000893 | 2.75 | 2.96 |  |
| contrast:interaction | bound | - | 30 | normal | 0.00104 | 1.69 | 1.85 |  |
| contrast:interaction | bound | - | 30 | stress | 0.00104 | 2.57 | 2.76 |  |
| contrast:interaction | bound | - | 30 | stress- | 0.00104 | 2.72 | 2.92 |  |
| contrast:interaction | bound | - | 30 | normal | 0.00119 | 1.68 | 1.83 |  |
| contrast:interaction | bound | - | 30 | stress | 0.00119 | 2.52 | 2.70 |  |
| contrast:interaction | bound | - | 30 | stress- | 0.00119 | 2.69 | 2.88 |  |
| contrast:interaction | bound | - | 30 | normal | 0.00125 | 1.72 | 1.86 |  |
| contrast:interaction | bound | - | 30 | stress | 0.00125 | 2.50 | 2.67 |  |
| contrast:interaction | bound | - | 30 | stress- | 0.00125 | 2.72 | 2.90 |  |
| contrast:interaction | bound | - | 30 | normal | 0.00139 | 1.62 | 1.75 |  |
| contrast:interaction | bound | - | 30 | stress | 0.00139 | 2.34 | 2.50 |  |
| contrast:interaction | bound | - | 30 | stress- | 0.00139 | 2.54 | 2.70 |  |
| contrast:interaction | bound | - | 30 | normal | 0.00156 | 1.64 | 1.77 |  |
| contrast:interaction | bound | - | 30 | stress | 0.00156 | 2.27 | 2.42 |  |
| contrast:interaction | bound | - | 30 | stress- | 0.00156 | 2.46 | 2.61 |  |
| contrast:interaction | bound | - | 30 | normal | 0.00167 | 1.65 | 1.77 |  |
| contrast:interaction | bound | - | 30 | stress | 0.00167 | 2.33 | 2.47 |  |
| contrast:interaction | bound | - | 30 | stress- | 0.00167 | 2.46 | 2.61 |  |
| contrast:interaction | bound | - | 30 | normal | 0.00208 | 1.61 | 1.72 |  |
| contrast:interaction | bound | - | 30 | stress | 0.00208 | 2.16 | 2.29 |  |
| contrast:interaction | bound | - | 30 | stress- | 0.00208 | 2.35 | 2.48 |  |
| contrast:interaction | bound | - | 30 | normal | 0.00278 | 1.54 | 1.63 |  |
| contrast:interaction | bound | - | 30 | stress | 0.00278 | 2.02 | 2.13 |  |
| contrast:interaction | bound | - | 30 | stress- | 0.00278 | 2.17 | 2.28 |  |
| contrast:interaction | bound | - | 30 | normal | 0.00313 | 1.49 | 1.58 |  |
| contrast:interaction | bound | - | 30 | stress | 0.00313 | 1.98 | 2.07 |  |
| contrast:interaction | bound | - | 30 | stress- | 0.00313 | 2.11 | 2.21 |  |
| contrast:interaction | bound | - | 30 | normal | 0.00417 | 1.43 | 1.50 |  |
| contrast:interaction | bound | - | 30 | stress | 0.00417 | 1.89 | 1.97 |  |
| contrast:interaction | bound | - | 30 | stress- | 0.00417 | 1.97 | 2.05 |  |
| contrast:interaction | bound | - | 30 | normal | 0.00625 | 1.37 | 1.43 |  |
| contrast:interaction | bound | - | 30 | stress | 0.00625 | 1.72 | 1.79 |  |
| contrast:interaction | bound | - | 30 | stress- | 0.00625 | 1.79 | 1.85 |  |
| contrast:interaction | bound | - | 30 | normal | 0.00833 | 1.32 | 1.37 |  |
| contrast:interaction | bound | - | 30 | stress | 0.00833 | 1.63 | 1.68 |  |
| contrast:interaction | bound | - | 30 | stress- | 0.00833 | 1.69 | 1.74 |  |
| contrast:interaction | bound | - | 50 | normal | 0.000893 | 1.47 | 1.63 |  |
| contrast:interaction | bound | - | 50 | stress | 0.000893 | 1.97 | 2.15 |  |
| contrast:interaction | bound | - | 50 | stress- | 0.000893 | 2.02 | 2.21 |  |
| contrast:interaction | bound | - | 50 | normal | 0.00104 | 1.44 | 1.59 |  |
| contrast:interaction | bound | - | 50 | stress | 0.00104 | 1.92 | 2.09 |  |
| contrast:interaction | bound | - | 50 | stress- | 0.00104 | 2.05 | 2.22 |  |
| contrast:interaction | bound | - | 50 | normal | 0.00119 | 1.43 | 1.57 |  |
| contrast:interaction | bound | - | 50 | stress | 0.00119 | 1.88 | 2.03 |  |
| contrast:interaction | bound | - | 50 | stress- | 0.00119 | 2.03 | 2.19 |  |
| contrast:interaction | bound | - | 50 | normal | 0.00125 | 1.44 | 1.58 |  |
| contrast:interaction | bound | - | 50 | stress | 0.00125 | 1.90 | 2.05 |  |
| contrast:interaction | bound | - | 50 | stress- | 0.00125 | 2.03 | 2.19 |  |
| contrast:interaction | bound | - | 50 | normal | 0.00139 | 1.37 | 1.49 |  |
| contrast:interaction | bound | - | 50 | stress | 0.00139 | 1.81 | 1.95 |  |
| contrast:interaction | bound | - | 50 | stress- | 0.00139 | 1.90 | 2.04 |  |
| contrast:interaction | bound | - | 50 | normal | 0.00156 | 1.35 | 1.46 |  |
| contrast:interaction | bound | - | 50 | stress | 0.00156 | 1.77 | 1.90 |  |
| contrast:interaction | bound | - | 50 | stress- | 0.00156 | 1.83 | 1.96 |  |
| contrast:interaction | bound | - | 50 | normal | 0.00167 | 1.40 | 1.51 |  |
| contrast:interaction | bound | - | 50 | stress | 0.00167 | 1.77 | 1.90 |  |
| contrast:interaction | bound | - | 50 | stress- | 0.00167 | 1.85 | 1.98 |  |
| contrast:interaction | bound | - | 50 | normal | 0.00208 | 1.38 | 1.48 |  |
| contrast:interaction | bound | - | 50 | stress | 0.00208 | 1.71 | 1.82 |  |
| contrast:interaction | bound | - | 50 | stress- | 0.00208 | 1.80 | 1.91 |  |
| contrast:interaction | bound | - | 50 | normal | 0.00278 | 1.32 | 1.40 |  |
| contrast:interaction | bound | - | 50 | stress | 0.00278 | 1.66 | 1.76 |  |
| contrast:interaction | bound | - | 50 | stress- | 0.00278 | 1.68 | 1.78 |  |
| contrast:interaction | bound | - | 50 | normal | 0.00313 | 1.29 | 1.37 |  |
| contrast:interaction | bound | - | 50 | stress | 0.00313 | 1.61 | 1.70 |  |
| contrast:interaction | bound | - | 50 | stress- | 0.00313 | 1.68 | 1.77 |  |
| contrast:interaction | bound | - | 50 | normal | 0.00417 | 1.29 | 1.36 |  |
| contrast:interaction | bound | - | 50 | stress | 0.00417 | 1.53 | 1.61 |  |
| contrast:interaction | bound | - | 50 | stress- | 0.00417 | 1.58 | 1.65 |  |
| contrast:interaction | bound | - | 50 | stress | 0.00625 | 1.44 | 1.50 |  |
| contrast:interaction | bound | - | 50 | stress- | 0.00625 | 1.48 | 1.53 |  |
| contrast:interaction | bound | - | 50 | stress | 0.00833 | 1.39 | 1.43 |  |
| contrast:interaction | bound | - | 50 | stress- | 0.00833 | 1.44 | 1.49 |  |
| contrast:interaction | bound | - | 100 | normal | 0.000893 | 1.14 | 1.28 | marginal |
| contrast:interaction | bound | - | 100 | stress | 0.000893 | 1.22 | 1.37 | marginal |
| contrast:interaction | bound | - | 100 | stress- | 0.000893 | 1.19 | 1.34 | marginal |
| contrast:interaction | bound | - | 100 | normal | 0.00104 | 1.18 | 1.31 | marginal |
| contrast:interaction | bound | - | 100 | stress | 0.00104 | 1.23 | 1.37 | marginal |
| contrast:interaction | bound | - | 100 | stress- | 0.00104 | 1.26 | 1.40 |  |
| contrast:interaction | bound | - | 100 | normal | 0.00119 | 1.21 | 1.33 | marginal |
| contrast:interaction | bound | - | 100 | stress | 0.00119 | 1.24 | 1.37 | marginal |
| contrast:interaction | bound | - | 100 | stress- | 0.00119 | 1.25 | 1.38 |  |
| contrast:interaction | bound | - | 100 | normal | 0.00125 | 1.21 | 1.34 | marginal |
| contrast:interaction | bound | - | 100 | stress | 0.00125 | 1.27 | 1.39 |  |
| contrast:interaction | bound | - | 100 | stress- | 0.00125 | 1.27 | 1.40 |  |
| contrast:interaction | bound | - | 100 | normal | 0.00139 | 1.16 | 1.27 | marginal |
| contrast:interaction | bound | - | 100 | stress | 0.00139 | 1.18 | 1.30 | marginal |
| contrast:interaction | bound | - | 100 | stress- | 0.00139 | 1.22 | 1.34 | marginal |
| contrast:interaction | bound | - | 100 | stress | 0.00156 | 1.18 | 1.29 | marginal |
| contrast:interaction | bound | - | 100 | stress- | 0.00156 | 1.23 | 1.34 | marginal |
| contrast:interaction | bound | - | 100 | normal | 0.00167 | 1.16 | 1.26 | marginal |
| contrast:interaction | bound | - | 100 | stress | 0.00167 | 1.22 | 1.32 | marginal |
| contrast:interaction | bound | - | 100 | stress- | 0.00167 | 1.26 | 1.37 |  |
| contrast:interaction | bound | - | 100 | normal | 0.00208 | 1.18 | 1.28 | marginal |
| contrast:interaction | bound | - | 100 | stress | 0.00208 | 1.22 | 1.31 | marginal |
| contrast:interaction | bound | - | 100 | stress- | 0.00208 | 1.27 | 1.37 |  |
| contrast:interaction | bound | - | 100 | stress | 0.00278 | 1.20 | 1.28 | marginal |
| contrast:interaction | bound | - | 100 | stress- | 0.00278 | 1.25 | 1.33 |  |
| contrast:interaction | bound | - | 100 | stress | 0.00313 | 1.19 | 1.26 | marginal |
| contrast:interaction | bound | - | 100 | stress- | 0.00313 | 1.23 | 1.31 | marginal |
| contrast:interaction | bound | - | 100 | stress- | 0.00417 | 1.24 | 1.30 | marginal |
| contrast:interaction | bound | - | 200 | stress | 0.000893 | 1.14 | 1.28 | marginal |
| contrast:interaction | bound | - | 200 | stress- | 0.000893 | 1.16 | 1.30 | marginal |
| contrast:interaction | bound | - | 200 | stress- | 0.00104 | 1.16 | 1.29 | marginal |
| contrast:interaction | bound | - | 200 | stress- | 0.00119 | 1.14 | 1.26 | marginal |
| contrast:interaction | bound | - | 200 | stress- | 0.00125 | 1.18 | 1.30 | marginal |
| contrast:interaction/boott | bound | - | 20 | stress | 0.000893 | 2.19 | 2.38 |  |
| contrast:interaction/boott | bound | - | 20 | stress- | 0.000893 | 2.25 | 2.44 |  |
| contrast:interaction/boott | bound | - | 20 | stress | 0.00104 | 2.26 | 2.44 |  |
| contrast:interaction/boott | bound | - | 20 | stress- | 0.00104 | 2.26 | 2.44 |  |
| contrast:interaction/boott | bound | - | 20 | stress | 0.00119 | 2.30 | 2.47 |  |
| contrast:interaction/boott | bound | - | 20 | stress- | 0.00119 | 2.37 | 2.54 |  |
| contrast:interaction/boott | bound | - | 20 | stress | 0.00125 | 2.36 | 2.52 |  |
| contrast:interaction/boott | bound | - | 20 | stress- | 0.00125 | 2.40 | 2.56 |  |
| contrast:interaction/boott | bound | - | 20 | stress | 0.00139 | 2.28 | 2.43 |  |
| contrast:interaction/boott | bound | - | 20 | stress- | 0.00139 | 2.28 | 2.44 |  |
| contrast:interaction/boott | bound | - | 20 | stress | 0.00156 | 2.31 | 2.45 |  |
| contrast:interaction/boott | bound | - | 20 | stress- | 0.00156 | 2.27 | 2.41 |  |
| contrast:interaction/boott | bound | - | 20 | stress | 0.00167 | 2.45 | 2.60 |  |
| contrast:interaction/boott | bound | - | 20 | stress- | 0.00167 | 2.34 | 2.48 |  |
| contrast:interaction/boott | bound | - | 20 | stress | 0.00208 | 2.41 | 2.54 |  |
| contrast:interaction/boott | bound | - | 20 | stress- | 0.00208 | 2.34 | 2.47 |  |
| contrast:interaction/boott | bound | - | 20 | stress | 0.00278 | 2.33 | 2.44 |  |
| contrast:interaction/boott | bound | - | 20 | stress- | 0.00278 | 2.29 | 2.40 |  |
| contrast:interaction/boott | bound | - | 20 | stress | 0.00313 | 2.28 | 2.38 |  |
| contrast:interaction/boott | bound | - | 20 | stress- | 0.00313 | 2.26 | 2.36 |  |
| contrast:interaction/boott | bound | - | 20 | stress | 0.00417 | 2.24 | 2.33 |  |
| contrast:interaction/boott | bound | - | 20 | stress- | 0.00417 | 2.19 | 2.28 |  |
| contrast:interaction/boott | bound | - | 20 | stress | 0.00625 | 2.10 | 2.17 |  |
| contrast:interaction/boott | bound | - | 20 | stress- | 0.00625 | 2.09 | 2.16 |  |
| contrast:interaction/boott | bound | - | 20 | stress | 0.00833 | 2.01 | 2.07 |  |
| contrast:interaction/boott | bound | - | 20 | stress- | 0.00833 | 1.99 | 2.05 |  |
| contrast:interaction/boott | bound | - | 30 | stress | 0.000893 | 2.82 | 3.03 |  |
| contrast:interaction/boott | bound | - | 30 | stress- | 0.000893 | 2.62 | 2.83 |  |
| contrast:interaction/boott | bound | - | 30 | stress | 0.00104 | 2.81 | 3.01 |  |
| contrast:interaction/boott | bound | - | 30 | stress- | 0.00104 | 2.65 | 2.84 |  |
| contrast:interaction/boott | bound | - | 30 | stress | 0.00119 | 2.81 | 3.00 |  |
| contrast:interaction/boott | bound | - | 30 | stress- | 0.00119 | 2.67 | 2.85 |  |
| contrast:interaction/boott | bound | - | 30 | stress | 0.00125 | 2.87 | 3.05 |  |
| contrast:interaction/boott | bound | - | 30 | stress- | 0.00125 | 2.69 | 2.86 |  |
| contrast:interaction/boott | bound | - | 30 | stress | 0.00139 | 2.73 | 2.90 |  |
| contrast:interaction/boott | bound | - | 30 | stress- | 0.00139 | 2.54 | 2.71 |  |
| contrast:interaction/boott | bound | - | 30 | stress | 0.00156 | 2.65 | 2.81 |  |
| contrast:interaction/boott | bound | - | 30 | stress- | 0.00156 | 2.55 | 2.71 |  |
| contrast:interaction/boott | bound | - | 30 | stress | 0.00167 | 2.72 | 2.87 |  |
| contrast:interaction/boott | bound | - | 30 | stress- | 0.00167 | 2.60 | 2.75 |  |
| contrast:interaction/boott | bound | - | 30 | stress | 0.00208 | 2.61 | 2.74 |  |
| contrast:interaction/boott | bound | - | 30 | stress- | 0.00208 | 2.51 | 2.64 |  |
| contrast:interaction/boott | bound | - | 30 | stress | 0.00278 | 2.45 | 2.57 |  |
| contrast:interaction/boott | bound | - | 30 | stress- | 0.00278 | 2.34 | 2.45 |  |
| contrast:interaction/boott | bound | - | 30 | stress | 0.00313 | 2.38 | 2.49 |  |
| contrast:interaction/boott | bound | - | 30 | stress- | 0.00313 | 2.28 | 2.38 |  |
| contrast:interaction/boott | bound | - | 30 | stress | 0.00417 | 2.28 | 2.37 |  |
| contrast:interaction/boott | bound | - | 30 | stress- | 0.00417 | 2.16 | 2.25 |  |
| contrast:interaction/boott | bound | - | 30 | stress | 0.00625 | 2.09 | 2.16 |  |
| contrast:interaction/boott | bound | - | 30 | stress- | 0.00625 | 1.98 | 2.05 |  |
| contrast:interaction/boott | bound | - | 30 | stress | 0.00833 | 1.92 | 1.98 |  |
| contrast:interaction/boott | bound | - | 30 | stress- | 0.00833 | 1.89 | 1.94 |  |
| contrast:interaction/boott | bound | - | 50 | stress | 0.000893 | 2.56 | 2.77 |  |
| contrast:interaction/boott | bound | - | 50 | stress- | 0.000893 | 2.32 | 2.52 |  |
| contrast:interaction/boott | bound | - | 50 | stress | 0.00104 | 2.49 | 2.68 |  |
| contrast:interaction/boott | bound | - | 50 | stress- | 0.00104 | 2.32 | 2.51 |  |
| contrast:interaction/boott | bound | - | 50 | stress | 0.00119 | 2.44 | 2.61 |  |
| contrast:interaction/boott | bound | - | 50 | stress- | 0.00119 | 2.28 | 2.45 |  |
| contrast:interaction/boott | bound | - | 50 | stress | 0.00125 | 2.46 | 2.63 |  |
| contrast:interaction/boott | bound | - | 50 | stress- | 0.00125 | 2.27 | 2.43 |  |
| contrast:interaction/boott | bound | - | 50 | stress | 0.00139 | 2.31 | 2.47 |  |
| contrast:interaction/boott | bound | - | 50 | stress- | 0.00139 | 2.15 | 2.30 |  |
| contrast:interaction/boott | bound | - | 50 | stress | 0.00156 | 2.27 | 2.42 |  |
| contrast:interaction/boott | bound | - | 50 | stress- | 0.00156 | 2.08 | 2.22 |  |
| contrast:interaction/boott | bound | - | 50 | stress | 0.00167 | 2.32 | 2.47 |  |
| contrast:interaction/boott | bound | - | 50 | stress- | 0.00167 | 2.12 | 2.26 |  |
| contrast:interaction/boott | bound | - | 50 | stress | 0.00208 | 2.25 | 2.37 |  |
| contrast:interaction/boott | bound | - | 50 | stress- | 0.00208 | 2.02 | 2.14 |  |
| contrast:interaction/boott | bound | - | 50 | stress | 0.00278 | 2.06 | 2.16 |  |
| contrast:interaction/boott | bound | - | 50 | stress- | 0.00278 | 1.90 | 2.00 |  |
| contrast:interaction/boott | bound | - | 50 | stress | 0.00313 | 2.01 | 2.11 |  |
| contrast:interaction/boott | bound | - | 50 | stress- | 0.00313 | 1.87 | 1.96 |  |
| contrast:interaction/boott | bound | - | 50 | stress | 0.00417 | 1.94 | 2.02 |  |
| contrast:interaction/boott | bound | - | 50 | stress- | 0.00417 | 1.80 | 1.88 |  |
| contrast:interaction/boott | bound | - | 50 | stress | 0.00625 | 1.80 | 1.86 |  |
| contrast:interaction/boott | bound | - | 50 | stress- | 0.00625 | 1.65 | 1.71 |  |
| contrast:interaction/boott | bound | - | 50 | stress | 0.00833 | 1.71 | 1.76 |  |
| contrast:interaction/boott | bound | - | 50 | stress- | 0.00833 | 1.58 | 1.63 |  |
| contrast:interaction/boott | bound | - | 100 | stress | 0.000893 | 1.94 | 2.13 |  |
| contrast:interaction/boott | bound | - | 100 | stress- | 0.000893 | 1.96 | 2.14 |  |
| contrast:interaction/boott | bound | - | 100 | stress | 0.00104 | 1.99 | 2.16 |  |
| contrast:interaction/boott | bound | - | 100 | stress- | 0.00104 | 1.97 | 2.14 |  |
| contrast:interaction/boott | bound | - | 100 | stress | 0.00119 | 1.90 | 2.05 |  |
| contrast:interaction/boott | bound | - | 100 | stress- | 0.00119 | 1.96 | 2.12 |  |
| contrast:interaction/boott | bound | - | 100 | stress | 0.00125 | 1.93 | 2.09 |  |
| contrast:interaction/boott | bound | - | 100 | stress- | 0.00125 | 1.99 | 2.15 |  |
| contrast:interaction/boott | bound | - | 100 | stress | 0.00139 | 1.84 | 1.98 |  |
| contrast:interaction/boott | bound | - | 100 | stress- | 0.00139 | 1.87 | 2.02 |  |
| contrast:interaction/boott | bound | - | 100 | stress | 0.00156 | 1.78 | 1.91 |  |
| contrast:interaction/boott | bound | - | 100 | stress- | 0.00156 | 1.81 | 1.94 |  |
| contrast:interaction/boott | bound | - | 100 | stress | 0.00167 | 1.82 | 1.94 |  |
| contrast:interaction/boott | bound | - | 100 | stress- | 0.00167 | 1.84 | 1.97 |  |
| contrast:interaction/boott | bound | - | 100 | stress | 0.00208 | 1.68 | 1.78 |  |
| contrast:interaction/boott | bound | - | 100 | stress- | 0.00208 | 1.77 | 1.89 |  |
| contrast:interaction/boott | bound | - | 100 | stress | 0.00278 | 1.60 | 1.69 |  |
| contrast:interaction/boott | bound | - | 100 | stress- | 0.00278 | 1.64 | 1.73 |  |
| contrast:interaction/boott | bound | - | 100 | stress | 0.00313 | 1.58 | 1.66 |  |
| contrast:interaction/boott | bound | - | 100 | stress- | 0.00313 | 1.59 | 1.67 |  |
| contrast:interaction/boott | bound | - | 100 | stress | 0.00417 | 1.52 | 1.59 |  |
| contrast:interaction/boott | bound | - | 100 | stress- | 0.00417 | 1.52 | 1.59 |  |
| contrast:interaction/boott | bound | - | 100 | stress | 0.00625 | 1.44 | 1.49 |  |
| contrast:interaction/boott | bound | - | 100 | stress- | 0.00625 | 1.40 | 1.46 |  |
| contrast:interaction/boott | bound | - | 100 | stress | 0.00833 | 1.38 | 1.42 |  |
| contrast:interaction/boott | bound | - | 100 | stress- | 0.00833 | 1.36 | 1.41 |  |
| contrast:interaction/boott | bound | - | 200 | stress | 0.000893 | 1.30 | 1.45 |  |
| contrast:interaction/boott | bound | - | 200 | stress- | 0.000893 | 1.35 | 1.50 |  |
| contrast:interaction/boott | bound | - | 200 | stress | 0.00104 | 1.29 | 1.43 |  |
| contrast:interaction/boott | bound | - | 200 | stress- | 0.00104 | 1.40 | 1.55 |  |
| contrast:interaction/boott | bound | - | 200 | stress | 0.00119 | 1.30 | 1.43 |  |
| contrast:interaction/boott | bound | - | 200 | stress- | 0.00119 | 1.36 | 1.50 |  |
| contrast:interaction/boott | bound | - | 200 | stress | 0.00125 | 1.33 | 1.46 |  |
| contrast:interaction/boott | bound | - | 200 | stress- | 0.00125 | 1.42 | 1.55 |  |
| contrast:interaction/boott | bound | - | 200 | stress | 0.00139 | 1.26 | 1.38 |  |
| contrast:interaction/boott | bound | - | 200 | stress- | 0.00139 | 1.36 | 1.48 |  |
| contrast:interaction/boott | bound | - | 200 | stress | 0.00156 | 1.27 | 1.38 |  |
| contrast:interaction/boott | bound | - | 200 | stress- | 0.00156 | 1.33 | 1.44 |  |
| contrast:interaction/boott | bound | - | 200 | stress | 0.00167 | 1.31 | 1.42 |  |
| contrast:interaction/boott | bound | - | 200 | stress- | 0.00167 | 1.35 | 1.46 |  |
| contrast:interaction/boott | bound | - | 200 | stress | 0.00208 | 1.26 | 1.35 |  |
| contrast:interaction/boott | bound | - | 200 | stress- | 0.00208 | 1.31 | 1.40 |  |
| contrast:interaction/boott | bound | - | 200 | stress | 0.00278 | 1.24 | 1.32 | marginal |
| contrast:interaction/boott | bound | - | 200 | stress- | 0.00278 | 1.22 | 1.30 | marginal |
| contrast:interaction/boott | bound | - | 200 | stress | 0.00313 | 1.21 | 1.29 | marginal |
| contrast:interaction/boott | bound | - | 200 | stress- | 0.00313 | 1.20 | 1.27 | marginal |
| contrast:paired | bound | - | 20 | normal | 0.000893 | 3.88 | 4.13 |  |
| contrast:paired | bound | - | 20 | stress | 0.000893 | 34.42 | 35.14 |  |
| contrast:paired | bound | - | 20 | normal | 0.00104 | 3.69 | 3.92 |  |
| contrast:paired | bound | - | 20 | stress | 0.00104 | 31.55 | 32.20 |  |
| contrast:paired | bound | - | 20 | normal | 0.00119 | 3.60 | 3.81 |  |
| contrast:paired | bound | - | 20 | stress | 0.00119 | 29.12 | 29.70 |  |
| contrast:paired | bound | - | 20 | normal | 0.00125 | 3.62 | 3.83 |  |
| contrast:paired | bound | - | 20 | stress | 0.00125 | 28.54 | 29.10 |  |
| contrast:paired | bound | - | 20 | normal | 0.00139 | 3.41 | 3.60 |  |
| contrast:paired | bound | - | 20 | stress | 0.00139 | 26.31 | 26.82 |  |
| contrast:paired | bound | - | 20 | normal | 0.00156 | 3.25 | 3.42 |  |
| contrast:paired | bound | - | 20 | stress | 0.00156 | 24.54 | 25.00 |  |
| contrast:paired | bound | - | 20 | normal | 0.00167 | 3.22 | 3.39 |  |
| contrast:paired | bound | - | 20 | stress | 0.00167 | 23.99 | 24.43 |  |
| contrast:paired | bound | - | 20 | normal | 0.00208 | 3.02 | 3.17 |  |
| contrast:paired | bound | - | 20 | stress | 0.00208 | 20.79 | 21.15 |  |
| contrast:paired | bound | - | 20 | normal | 0.00278 | 2.69 | 2.81 |  |
| contrast:paired | bound | - | 20 | stress | 0.00278 | 17.28 | 17.57 |  |
| contrast:paired | bound | - | 20 | normal | 0.00313 | 2.56 | 2.67 |  |
| contrast:paired | bound | - | 20 | stress | 0.00313 | 16.07 | 16.34 |  |
| contrast:paired | bound | - | 20 | normal | 0.00417 | 2.37 | 2.46 |  |
| contrast:paired | bound | - | 20 | stress | 0.00417 | 13.52 | 13.73 |  |
| contrast:paired | bound | - | 20 | normal | 0.00625 | 2.06 | 2.13 |  |
| contrast:paired | bound | - | 20 | stress | 0.00625 | 10.51 | 10.66 |  |
| contrast:paired | bound | - | 20 | normal | 0.00833 | 1.90 | 1.96 |  |
| contrast:paired | bound | - | 20 | stress | 0.00833 | 8.79 | 8.90 |  |
| contrast:paired | bound | - | 30 | normal | 0.000893 | 2.67 | 2.89 |  |
| contrast:paired | bound | - | 30 | stress | 0.000893 | 26.93 | 27.57 |  |
| contrast:paired | bound | - | 30 | normal | 0.00104 | 2.65 | 2.85 |  |
| contrast:paired | bound | - | 30 | stress | 0.00104 | 24.70 | 25.27 |  |
| contrast:paired | bound | - | 30 | normal | 0.00119 | 2.55 | 2.73 |  |
| contrast:paired | bound | - | 30 | stress | 0.00119 | 22.75 | 23.27 |  |
| contrast:paired | bound | - | 30 | normal | 0.00125 | 2.55 | 2.72 |  |
| contrast:paired | bound | - | 30 | stress | 0.00125 | 22.21 | 22.71 |  |
| contrast:paired | bound | - | 30 | normal | 0.00139 | 2.40 | 2.56 |  |
| contrast:paired | bound | - | 30 | stress | 0.00139 | 20.46 | 20.92 |  |
| contrast:paired | bound | - | 30 | normal | 0.00156 | 2.33 | 2.48 |  |
| contrast:paired | bound | - | 30 | stress | 0.00156 | 18.95 | 19.36 |  |
| contrast:paired | bound | - | 30 | normal | 0.00167 | 2.29 | 2.44 |  |
| contrast:paired | bound | - | 30 | stress | 0.00167 | 18.44 | 18.83 |  |
| contrast:paired | bound | - | 30 | normal | 0.00208 | 2.18 | 2.30 |  |
| contrast:paired | bound | - | 30 | stress | 0.00208 | 16.04 | 16.37 |  |
| contrast:paired | bound | - | 30 | normal | 0.00278 | 2.03 | 2.13 |  |
| contrast:paired | bound | - | 30 | stress | 0.00278 | 13.31 | 13.57 |  |
| contrast:paired | bound | - | 30 | normal | 0.00313 | 1.96 | 2.06 |  |
| contrast:paired | bound | - | 30 | stress | 0.00313 | 12.37 | 12.60 |  |
| contrast:paired | bound | - | 30 | normal | 0.00417 | 1.88 | 1.96 |  |
| contrast:paired | bound | - | 30 | stress | 0.00417 | 10.33 | 10.52 |  |
| contrast:paired | bound | - | 30 | normal | 0.00625 | 1.70 | 1.76 |  |
| contrast:paired | bound | - | 30 | stress | 0.00625 | 7.99 | 8.12 |  |
| contrast:paired | bound | - | 30 | normal | 0.00833 | 1.60 | 1.65 |  |
| contrast:paired | bound | - | 30 | stress | 0.00833 | 6.65 | 6.75 |  |
| contrast:paired | bound | - | 50 | normal | 0.000893 | 1.71 | 1.88 |  |
| contrast:paired | bound | - | 50 | stress | 0.000893 | 14.00 | 14.47 |  |
| contrast:paired | bound | - | 50 | normal | 0.00104 | 1.75 | 1.91 |  |
| contrast:paired | bound | - | 50 | stress | 0.00104 | 12.84 | 13.26 |  |
| contrast:paired | bound | - | 50 | normal | 0.00119 | 1.73 | 1.88 |  |
| contrast:paired | bound | - | 50 | stress | 0.00119 | 12.01 | 12.39 |  |
| contrast:paired | bound | - | 50 | normal | 0.00125 | 1.71 | 1.86 |  |
| contrast:paired | bound | - | 50 | stress | 0.00125 | 11.71 | 12.07 |  |
| contrast:paired | bound | - | 50 | normal | 0.00139 | 1.63 | 1.77 |  |
| contrast:paired | bound | - | 50 | stress | 0.00139 | 10.82 | 11.15 |  |
| contrast:paired | bound | - | 50 | normal | 0.00156 | 1.61 | 1.73 |  |
| contrast:paired | bound | - | 50 | stress | 0.00156 | 10.04 | 10.34 |  |
| contrast:paired | bound | - | 50 | normal | 0.00167 | 1.68 | 1.80 |  |
| contrast:paired | bound | - | 50 | stress | 0.00167 | 9.84 | 10.12 |  |
| contrast:paired | bound | - | 50 | normal | 0.00208 | 1.62 | 1.73 |  |
| contrast:paired | bound | - | 50 | stress | 0.00208 | 8.55 | 8.79 |  |
| contrast:paired | bound | - | 50 | normal | 0.00278 | 1.53 | 1.62 |  |
| contrast:paired | bound | - | 50 | stress | 0.00278 | 7.09 | 7.28 |  |
| contrast:paired | bound | - | 50 | normal | 0.00313 | 1.48 | 1.56 |  |
| contrast:paired | bound | - | 50 | stress | 0.00313 | 6.57 | 6.74 |  |
| contrast:paired | bound | - | 50 | normal | 0.00417 | 1.41 | 1.48 |  |
| contrast:paired | bound | - | 50 | stress | 0.00417 | 5.61 | 5.74 |  |
| contrast:paired | bound | - | 50 | normal | 0.00625 | 1.37 | 1.43 |  |
| contrast:paired | bound | - | 50 | stress | 0.00625 | 4.47 | 4.57 |  |
| contrast:paired | bound | - | 50 | normal | 0.00833 | 1.35 | 1.39 |  |
| contrast:paired | bound | - | 50 | stress | 0.00833 | 3.84 | 3.92 |  |
| contrast:paired | bound | - | 100 | normal | 0.000893 | 1.29 | 1.44 |  |
| contrast:paired | bound | - | 100 | stress | 0.000893 | 5.62 | 5.93 |  |
| contrast:paired | bound | - | 100 | normal | 0.00104 | 1.30 | 1.44 |  |
| contrast:paired | bound | - | 100 | stress | 0.00104 | 5.37 | 5.64 |  |
| contrast:paired | bound | - | 100 | normal | 0.00119 | 1.29 | 1.42 |  |
| contrast:paired | bound | - | 100 | stress | 0.00119 | 5.13 | 5.38 |  |
| contrast:paired | bound | - | 100 | normal | 0.00125 | 1.30 | 1.42 |  |
| contrast:paired | bound | - | 100 | stress | 0.00125 | 5.05 | 5.29 |  |
| contrast:paired | bound | - | 100 | normal | 0.00139 | 1.22 | 1.34 | marginal |
| contrast:paired | bound | - | 100 | stress | 0.00139 | 4.75 | 4.97 |  |
| contrast:paired | bound | - | 100 | normal | 0.00156 | 1.18 | 1.29 | marginal |
| contrast:paired | bound | - | 100 | stress | 0.00156 | 4.50 | 4.71 |  |
| contrast:paired | bound | - | 100 | normal | 0.00167 | 1.21 | 1.31 | marginal |
| contrast:paired | bound | - | 100 | stress | 0.00167 | 4.45 | 4.65 |  |
| contrast:paired | bound | - | 100 | normal | 0.00208 | 1.19 | 1.29 | marginal |
| contrast:paired | bound | - | 100 | stress | 0.00208 | 4.12 | 4.29 |  |
| contrast:paired | bound | - | 100 | stress | 0.00278 | 3.62 | 3.76 |  |
| contrast:paired | bound | - | 100 | stress | 0.00313 | 3.45 | 3.58 |  |
| contrast:paired | bound | - | 100 | stress | 0.00417 | 3.07 | 3.18 |  |
| contrast:paired | bound | - | 100 | stress | 0.00625 | 2.65 | 2.73 |  |
| contrast:paired | bound | - | 100 | stress | 0.00833 | 2.38 | 2.45 |  |
| contrast:paired | bound | - | 200 | stress | 0.000893 | 2.99 | 3.21 |  |
| contrast:paired | bound | - | 200 | normal | 0.00104 | 1.20 | 1.33 | marginal |
| contrast:paired | bound | - | 200 | stress | 0.00104 | 2.88 | 3.09 |  |
| contrast:paired | bound | - | 200 | normal | 0.00119 | 1.16 | 1.28 | marginal |
| contrast:paired | bound | - | 200 | stress | 0.00119 | 2.79 | 2.97 |  |
| contrast:paired | bound | - | 200 | normal | 0.00125 | 1.19 | 1.31 | marginal |
| contrast:paired | bound | - | 200 | stress | 0.00125 | 2.83 | 3.01 |  |
| contrast:paired | bound | - | 200 | stress | 0.00139 | 2.67 | 2.84 |  |
| contrast:paired | bound | - | 200 | stress | 0.00156 | 2.60 | 2.76 |  |
| contrast:paired | bound | - | 200 | normal | 0.00167 | 1.18 | 1.28 | marginal |
| contrast:paired | bound | - | 200 | stress | 0.00167 | 2.58 | 2.73 |  |
| contrast:paired | bound | - | 200 | normal | 0.00208 | 1.16 | 1.25 | marginal |
| contrast:paired | bound | - | 200 | stress | 0.00208 | 2.35 | 2.47 |  |
| contrast:paired | bound | - | 200 | stress | 0.00278 | 2.14 | 2.25 |  |
| contrast:paired | bound | - | 200 | stress | 0.00313 | 2.08 | 2.18 |  |
| contrast:paired | bound | - | 200 | stress | 0.00417 | 1.98 | 2.06 |  |
| contrast:paired | bound | - | 200 | stress | 0.00625 | 1.81 | 1.87 |  |
| contrast:paired | bound | - | 200 | stress | 0.00833 | 1.69 | 1.75 |  |
| contrast:paired/boott | bound | - | 20 | normal | 0.000893 | 1.18 | 1.32 | marginal |
| contrast:paired/boott | bound | - | 20 | stress | 0.000893 | 13.97 | 14.44 |  |
| contrast:paired/boott | bound | - | 20 | normal | 0.00104 | 1.17 | 1.30 | marginal |
| contrast:paired/boott | bound | - | 20 | stress | 0.00104 | 13.40 | 13.83 |  |
| contrast:paired/boott | bound | - | 20 | normal | 0.00119 | 1.20 | 1.32 | marginal |
| contrast:paired/boott | bound | - | 20 | stress | 0.00119 | 12.84 | 13.23 |  |
| contrast:paired/boott | bound | - | 20 | normal | 0.00125 | 1.19 | 1.31 | marginal |
| contrast:paired/boott | bound | - | 20 | stress | 0.00125 | 12.84 | 13.22 |  |
| contrast:paired/boott | bound | - | 20 | normal | 0.00139 | 1.16 | 1.27 | marginal |
| contrast:paired/boott | bound | - | 20 | stress | 0.00139 | 12.06 | 12.41 |  |
| contrast:paired/boott | bound | - | 20 | stress | 0.00156 | 11.52 | 11.84 |  |
| ... 885 more in summary.json | | | | | | | | |

## UCL / tail by cell

Entries are UCL/tail (pass iff at most 1.25; `*` marks a failure); the header lists the tails.

### bca:mean / lower / normal

| n | R | maxD | 0.025 | 0.05 |
|---|---|---|---|---|
| 20 | 7000 | 1.000 | 1.64 | 1.42 |
| 30 | 7000 | 1.000 | 1.52 | 1.29 |
| 50 | 7000 | 1.000 | 1.31 | 1.21 |
| 100 | 7000 | 1.000 | 1.31 | 1.22 |

### bca:mean / lower / stress

| n | R | maxD | 0.025 | 0.05 |
|---|---|---|---|---|
| 20 | 7000 | 1.000 | 1.42 | 1.25 |
| 30 | 7000 | 1.000 | 1.39 | 1.17 |
| 50 | 7000 | 1.000 | 1.31 | 1.21 |
| 100 | 7000 | 1.000 | 1.22 | 1.10 |

### bca:mean / lower / stress-

| n | R | maxD | 0.025 | 0.05 |
|---|---|---|---|---|
| 20 | 7000 | 1.000 | 4.57 | 2.93 |
| 30 | 7000 | 1.000 | 3.30 | 2.10 |
| 50 | 7000 | 1.000 | 1.92 | 1.50 |
| 100 | 7000 | 1.000 | 1.34 | 1.15 |

### bca:mean / upper / normal

| n | R | maxD | 0.025 | 0.05 |
|---|---|---|---|---|
| 20 | 7000 | 1.000 | 1.64 | 1.42 |
| 30 | 7000 | 1.000 | 1.49 | 1.26 |
| 50 | 7000 | 1.000 | 1.28 | 1.20 |
| 100 | 7000 | 1.000 | 1.10 | 1.04 |

### bca:mean / upper / stress

| n | R | maxD | 0.025 | 0.05 |
|---|---|---|---|---|
| 20 | 7000 | 1.000 | 4.59 | 2.93 |
| 30 | 7000 | 1.000 | 3.19 | 2.02 |
| 50 | 7000 | 1.000 | 1.86 | 1.42 |
| 100 | 7000 | 1.000 | 1.34 | 1.21 |

### bca:mean / upper / stress-

| n | R | maxD | 0.025 | 0.05 |
|---|---|---|---|---|
| 20 | 7000 | 1.000 | 1.32 | 1.19 |
| 30 | 7000 | 1.000 | 1.29 | 1.16 |
| 50 | 7000 | 1.000 | 1.34 | 1.22 |
| 100 | 7000 | 1.000 | 1.17 | 1.14 |

### contrast:interaction / bound / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.009 | 2.36* | 2.34* | 2.23* | 2.23* | 2.14* | 2.11* | 2.13* | 2.03* | 1.92* | 1.85* | 1.78* | 1.65* | 1.56* |
| 30 | 196000 | 1.029 | 1.80* | 1.85* | 1.83* | 1.86* | 1.75* | 1.77* | 1.77* | 1.72* | 1.63* | 1.58* | 1.50* | 1.43* | 1.37* |
| 50 | 196000 | 1.012 | 1.63* | 1.59* | 1.57* | 1.58* | 1.49* | 1.46* | 1.51* | 1.48* | 1.40* | 1.37* | 1.36* | 1.24 | 1.25 |
| 100 | 196000 | 1.028 | 1.28* | 1.31* | 1.33* | 1.34* | 1.27* | 1.22 | 1.26* | 1.28* | 1.19 | 1.16 | 1.16 | 1.09 | 1.08 |
| 200 | 196000 | 1.000 | 1.18 | 1.17 | 1.15 | 1.18 | 1.13 | 1.09 | 1.11 | 1.11 | 1.10 | 1.12 | 1.13 | 1.11 | 1.08 |

### contrast:interaction / bound / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.011 | 3.66* | 3.48* | 3.42* | 3.45* | 3.22* | 3.10* | 3.14* | 2.99* | 2.73* | 2.63* | 2.46* | 2.19* | 2.02* |
| 30 | 196000 | 1.009 | 2.89* | 2.76* | 2.70* | 2.67* | 2.50* | 2.42* | 2.47* | 2.29* | 2.13* | 2.07* | 1.97* | 1.79* | 1.68* |
| 50 | 196000 | 1.022 | 2.15* | 2.09* | 2.03* | 2.05* | 1.95* | 1.90* | 1.90* | 1.82* | 1.76* | 1.70* | 1.61* | 1.50* | 1.43* |
| 100 | 196000 | 1.012 | 1.37* | 1.37* | 1.37* | 1.39* | 1.30* | 1.29* | 1.32* | 1.31* | 1.28* | 1.26* | 1.21 | 1.20 | 1.17 |
| 200 | 196000 | 1.007 | 1.28* | 1.25 | 1.21 | 1.23 | 1.19 | 1.15 | 1.19 | 1.21 | 1.18 | 1.18 | 1.15 | 1.12 | 1.10 |

### contrast:interaction / bound / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 3.67* | 3.55* | 3.35* | 3.32* | 3.12* | 3.03* | 3.08* | 2.84* | 2.65* | 2.58* | 2.44* | 2.16* | 2.01* |
| 30 | 196000 | 1.006 | 2.96* | 2.92* | 2.88* | 2.90* | 2.70* | 2.61* | 2.61* | 2.48* | 2.28* | 2.21* | 2.05* | 1.85* | 1.74* |
| 50 | 196000 | 1.012 | 2.21* | 2.22* | 2.19* | 2.19* | 2.04* | 1.96* | 1.98* | 1.91* | 1.78* | 1.77* | 1.65* | 1.53* | 1.49* |
| 100 | 196000 | 1.007 | 1.34* | 1.40* | 1.38* | 1.40* | 1.34* | 1.34* | 1.37* | 1.37* | 1.33* | 1.31* | 1.30* | 1.21 | 1.21 |
| 200 | 196000 | 1.010 | 1.30* | 1.29* | 1.26* | 1.30* | 1.23 | 1.22 | 1.24 | 1.20 | 1.18 | 1.17 | 1.15 | 1.13 | 1.13 |

### contrast:interaction/boott / bound / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.017 | 1.01 | 1.05 | 1.05 | 1.09 | 1.04 | 1.05 | 1.11 | 1.06 | 1.01 | 1.00 | 1.02 | 1.02 | 1.03 |
| 30 | 196000 | 1.021 | 1.16 | 1.20 | 1.17 | 1.17 | 1.14 | 1.11 | 1.13 | 1.12 | 1.07 | 1.06 | 1.05 | 1.05 | 1.05 |
| 50 | 196000 | 1.016 | 1.11 | 1.07 | 1.08 | 1.11 | 1.05 | 1.01 | 1.08 | 1.07 | 1.07 | 1.08 | 1.05 | 1.05 | 1.04 |
| 100 | 196000 | 1.025 | 1.15 | 1.15 | 1.15 | 1.18 | 1.12 | 1.11 | 1.15 | 1.09 | 1.08 | 1.06 | 1.07 | 1.07 | 1.04 |
| 200 | 196000 | 1.007 | 0.98 | 0.96 | 0.98 | 1.02 | 0.95 | 0.93 | 0.96 | 1.00 | 0.99 | 0.98 | 1.00 | 0.97 | 0.97 |

### contrast:interaction/boott / bound / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.028 | 2.38* | 2.44* | 2.47* | 2.52* | 2.43* | 2.45* | 2.60* | 2.54* | 2.44* | 2.38* | 2.33* | 2.17* | 2.07* |
| 30 | 196000 | 1.002 | 3.03* | 3.01* | 3.00* | 3.05* | 2.90* | 2.81* | 2.87* | 2.74* | 2.57* | 2.49* | 2.37* | 2.16* | 1.98* |
| 50 | 196000 | 1.002 | 2.77* | 2.68* | 2.61* | 2.63* | 2.47* | 2.42* | 2.47* | 2.37* | 2.16* | 2.11* | 2.02* | 1.86* | 1.76* |
| 100 | 196000 | 1.000 | 2.13* | 2.16* | 2.05* | 2.09* | 1.98* | 1.91* | 1.94* | 1.78* | 1.69* | 1.66* | 1.59* | 1.49* | 1.42* |
| 200 | 196000 | 1.007 | 1.45* | 1.43* | 1.43* | 1.46* | 1.38* | 1.38* | 1.42* | 1.35* | 1.32* | 1.29* | 1.25 | 1.21 | 1.20 |

### contrast:interaction/boott / bound / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.001 | 2.44* | 2.44* | 2.54* | 2.56* | 2.44* | 2.41* | 2.48* | 2.47* | 2.40* | 2.36* | 2.28* | 2.16* | 2.05* |
| 30 | 196000 | 1.004 | 2.83* | 2.84* | 2.85* | 2.86* | 2.71* | 2.71* | 2.75* | 2.64* | 2.45* | 2.38* | 2.25* | 2.05* | 1.94* |
| 50 | 196000 | 1.013 | 2.52* | 2.51* | 2.45* | 2.43* | 2.30* | 2.22* | 2.26* | 2.14* | 2.00* | 1.96* | 1.88* | 1.71* | 1.63* |
| 100 | 196000 | 1.013 | 2.14* | 2.14* | 2.12* | 2.15* | 2.02* | 1.94* | 1.97* | 1.89* | 1.73* | 1.67* | 1.59* | 1.46* | 1.41* |
| 200 | 196000 | 1.015 | 1.50* | 1.55* | 1.50* | 1.55* | 1.48* | 1.44* | 1.46* | 1.40* | 1.30* | 1.27* | 1.25 | 1.23 | 1.17 |

### contrast:interaction/student / bound / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 1.14 | 1.16 | 1.14 | 1.13 | 1.13 | 1.12 | 1.11 | 1.12 | 1.07 | 1.11 | 1.15 | 1.09 | 1.09 |
| 30 | 196000 | 1.000 | 1.21 | 1.20 | 1.16 | 1.17 | 1.17 | 1.19 | 1.18 | 1.20 | 1.14 | 1.14 | 1.11 | 1.07 | 1.05 |
| 50 | 196000 | 1.000 | 1.18 | 1.19 | 1.20 | 1.21 | 1.20 | 1.18 | 1.15 | 1.17 | 1.15 | 1.12 | 1.07 | 1.09 | 1.08 |
| 100 | 196000 | 1.000 | 0.97 | 1.01 | 1.07 | 1.06 | 1.06 | 1.08 | 1.08 | 1.08 | 1.08 | 1.09 | 1.10 | 1.06 | 1.06 |
| 200 | 196000 | 1.000 | 1.22 | 1.17 | 1.19 | 1.17 | 1.15 | 1.16 | 1.18 | 1.20 | 1.19 | 1.18 | 1.12 | 1.09 | 1.07 |

### contrast:interaction/student / bound / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.27 | 0.28 | 0.31 | 0.31 | 0.32 | 0.31 | 0.34 | 0.37 | 0.41 | 0.44 | 0.53 | 0.62 | 0.67 |
| 30 | 196000 | 1.000 | 0.50 | 0.55 | 0.57 | 0.58 | 0.56 | 0.58 | 0.58 | 0.62 | 0.64 | 0.66 | 0.69 | 0.75 | 0.80 |
| 50 | 196000 | 1.000 | 0.59 | 0.62 | 0.65 | 0.67 | 0.71 | 0.77 | 0.78 | 0.81 | 0.84 | 0.87 | 0.86 | 0.89 | 0.92 |
| 100 | 196000 | 1.000 | 0.95 | 0.90 | 0.88 | 0.90 | 0.93 | 0.92 | 0.91 | 0.90 | 0.91 | 0.90 | 0.95 | 0.96 | 0.98 |
| 200 | 196000 | 1.000 | 1.02 | 1.04 | 1.09 | 1.10 | 1.09 | 1.07 | 1.06 | 1.04 | 1.02 | 1.01 | 1.01 | 1.00 | 0.99 |

### contrast:interaction/student / bound / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.31 | 0.30 | 0.30 | 0.34 | 0.34 | 0.35 | 0.36 | 0.42 | 0.45 | 0.47 | 0.50 | 0.59 | 0.66 |
| 30 | 196000 | 1.000 | 0.45 | 0.45 | 0.48 | 0.51 | 0.52 | 0.53 | 0.55 | 0.61 | 0.69 | 0.69 | 0.68 | 0.77 | 0.84 |
| 50 | 196000 | 1.000 | 0.67 | 0.69 | 0.75 | 0.75 | 0.73 | 0.77 | 0.79 | 0.79 | 0.80 | 0.83 | 0.85 | 0.89 | 0.91 |
| 100 | 196000 | 1.000 | 0.83 | 0.89 | 0.92 | 0.94 | 0.95 | 0.95 | 0.96 | 1.01 | 0.96 | 0.99 | 0.98 | 0.98 | 0.98 |
| 200 | 196000 | 1.000 | 1.12 | 1.12 | 1.14 | 1.12 | 1.10 | 1.11 | 1.12 | 1.11 | 1.07 | 1.04 | 1.02 | 1.02 | 1.02 |

### contrast:paired / bound / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.020 | 4.13* | 3.92* | 3.81* | 3.83* | 3.60* | 3.42* | 3.39* | 3.17* | 2.81* | 2.67* | 2.46* | 2.13* | 1.96* |
| 30 | 196000 | 1.013 | 2.89* | 2.85* | 2.73* | 2.72* | 2.56* | 2.48* | 2.44* | 2.30* | 2.13* | 2.06* | 1.96* | 1.76* | 1.65* |
| 50 | 196000 | 1.019 | 1.88* | 1.91* | 1.88* | 1.86* | 1.77* | 1.73* | 1.80* | 1.73* | 1.62* | 1.56* | 1.48* | 1.43* | 1.39* |
| 100 | 196000 | 1.000 | 1.44* | 1.44* | 1.42* | 1.42* | 1.34* | 1.29* | 1.31* | 1.29* | 1.24 | 1.20 | 1.20 | 1.14 | 1.14 |
| 200 | 196000 | 1.004 | 1.23 | 1.33* | 1.28* | 1.31* | 1.24 | 1.25 | 1.28* | 1.25* | 1.22 | 1.18 | 1.18 | 1.13 | 1.10 |

### contrast:paired / bound / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 35.14* | 32.20* | 29.70* | 29.10* | 26.82* | 25.00* | 24.43* | 21.15* | 17.57* | 16.34* | 13.73* | 10.66* | 8.90* |
| 30 | 196000 | 1.011 | 27.57* | 25.27* | 23.27* | 22.71* | 20.92* | 19.36* | 18.83* | 16.37* | 13.57* | 12.60* | 10.52* | 8.12* | 6.75* |
| 50 | 196000 | 1.006 | 14.47* | 13.26* | 12.39* | 12.07* | 11.15* | 10.34* | 10.12* | 8.79* | 7.28* | 6.74* | 5.74* | 4.57* | 3.92* |
| 100 | 196000 | 1.017 | 5.93* | 5.64* | 5.38* | 5.29* | 4.97* | 4.71* | 4.65* | 4.29* | 3.76* | 3.58* | 3.18* | 2.73* | 2.45* |
| 200 | 196000 | 1.003 | 3.21* | 3.09* | 2.97* | 3.01* | 2.84* | 2.76* | 2.73* | 2.47* | 2.25* | 2.18* | 2.06* | 1.87* | 1.75* |

### contrast:paired / bound / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.016 | 0.42 | 0.47 | 0.48 | 0.50 | 0.47 | 0.45 | 0.49 | 0.51 | 0.49 | 0.50 | 0.52 | 0.54 | 0.55 |
| 30 | 196000 | 1.009 | 0.38 | 0.42 | 0.44 | 0.45 | 0.45 | 0.45 | 0.49 | 0.50 | 0.51 | 0.50 | 0.53 | 0.53 | 0.54 |
| 50 | 196000 | 1.000 | 0.42 | 0.43 | 0.45 | 0.46 | 0.44 | 0.45 | 0.47 | 0.47 | 0.49 | 0.50 | 0.51 | 0.53 | 0.57 |
| 100 | 196000 | 1.005 | 0.48 | 0.51 | 0.52 | 0.52 | 0.49 | 0.49 | 0.51 | 0.50 | 0.50 | 0.53 | 0.56 | 0.60 | 0.62 |
| 200 | 196000 | 1.040 | 0.53 | 0.54 | 0.58 | 0.61 | 0.57 | 0.61 | 0.61 | 0.61 | 0.62 | 0.64 | 0.64 | 0.70 | 0.72 |

### contrast:paired/boott / bound / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.008 | 1.32* | 1.30* | 1.32* | 1.31* | 1.27* | 1.24 | 1.30* | 1.24 | 1.18 | 1.15 | 1.14 | 1.11 | 1.08 |
| 30 | 196000 | 1.010 | 1.16 | 1.13 | 1.16 | 1.22 | 1.16 | 1.17 | 1.17 | 1.15 | 1.09 | 1.10 | 1.09 | 1.04 | 1.04 |
| 50 | 196000 | 1.010 | 1.12 | 1.13 | 1.13 | 1.16 | 1.12 | 1.14 | 1.18 | 1.20 | 1.14 | 1.10 | 1.12 | 1.11 | 1.10 |
| 100 | 196000 | 1.013 | 1.18 | 1.15 | 1.19 | 1.23 | 1.15 | 1.13 | 1.15 | 1.15 | 1.13 | 1.12 | 1.09 | 1.06 | 1.04 |
| 200 | 196000 | 1.018 | 1.04 | 1.09 | 1.05 | 1.09 | 1.08 | 1.11 | 1.15 | 1.13 | 1.07 | 1.06 | 1.05 | 1.06 | 1.05 |

### contrast:paired/boott / bound / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.017 | 14.44* | 13.83* | 13.23* | 13.22* | 12.41* | 11.84* | 11.95* | 10.88* | 9.57* | 9.17* | 8.21* | 6.91* | 6.08* |
| 30 | 196000 | 1.017 | 15.44* | 14.63* | 13.90* | 13.79* | 12.91* | 12.19* | 12.22* | 10.97* | 9.45* | 8.91* | 7.75* | 6.21* | 5.31* |
| 50 | 196000 | 1.006 | 10.76* | 9.98* | 9.40* | 9.23* | 8.51* | 7.92* | 7.71* | 6.74* | 5.65* | 5.29* | 4.47* | 3.50* | 2.98* |
| 100 | 196000 | 1.014 | 3.76* | 3.58* | 3.45* | 3.38* | 3.17* | 2.99* | 2.95* | 2.65* | 2.30* | 2.16* | 1.98* | 1.68* | 1.52* |
| 200 | 196000 | 1.005 | 1.48* | 1.49* | 1.50* | 1.56* | 1.48* | 1.47* | 1.51* | 1.42* | 1.33* | 1.31* | 1.24 | 1.17 | 1.14 |

### contrast:paired/boott / bound / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.001 | 0.09 | 0.08 | 0.08 | 0.07 | 0.07 | 0.08 | 0.09 | 0.09 | 0.10 | 0.12 | 0.18 | 0.26 | 0.37 |
| 30 | 196000 | 1.020 | 0.05 | 0.05 | 0.07 | 0.07 | 0.07 | 0.08 | 0.09 | 0.13 | 0.19 | 0.24 | 0.35 | 0.52 | 0.61 |
| 50 | 196000 | 1.007 | 0.23 | 0.30 | 0.31 | 0.34 | 0.37 | 0.40 | 0.44 | 0.53 | 0.63 | 0.66 | 0.72 | 0.77 | 0.79 |
| 100 | 196000 | 1.022 | 0.59 | 0.66 | 0.73 | 0.77 | 0.75 | 0.77 | 0.82 | 0.83 | 0.82 | 0.82 | 0.87 | 0.86 | 0.90 |
| 200 | 196000 | 1.007 | 0.98 | 0.97 | 0.98 | 1.00 | 0.97 | 0.96 | 0.98 | 1.01 | 0.97 | 0.98 | 0.99 | 1.00 | 1.00 |

### contrast:paired/student / bound / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 1.22 | 1.19 | 1.19 | 1.18 | 1.19 | 1.14 | 1.13 | 1.09 | 1.10 | 1.10 | 1.10 | 1.07 | 1.06 |
| 30 | 196000 | 1.000 | 1.09 | 1.11 | 1.06 | 1.08 | 1.07 | 1.07 | 1.07 | 1.08 | 1.04 | 1.07 | 1.04 | 1.04 | 1.03 |
| 50 | 196000 | 1.000 | 1.07 | 1.03 | 1.01 | 1.01 | 1.00 | 1.01 | 0.99 | 0.99 | 0.98 | 0.99 | 1.01 | 1.01 | 1.00 |
| 100 | 196000 | 1.000 | 1.01 | 1.06 | 1.05 | 1.06 | 1.04 | 1.02 | 1.00 | 1.01 | 1.01 | 1.02 | 1.02 | 1.00 | 1.01 |
| 200 | 196000 | 1.000 | 1.18 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.14 | 1.09 | 1.07 | 1.07 | 1.07 | 1.04 | 1.02 |

### contrast:paired/student / bound / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 14.58* | 14.10* | 13.36* | 13.15* | 12.76* | 12.21* | 11.89* | 11.01* | 9.90* | 9.44* | 8.50* | 7.15* | 6.34* |
| 30 | 196000 | 1.000 | 16.78* | 15.63* | 14.64* | 14.35* | 13.58* | 12.88* | 12.46* | 11.26* | 9.89* | 9.33* | 8.03* | 6.56* | 5.63* |
| 50 | 196000 | 1.000 | 12.60* | 11.61* | 10.77* | 10.52* | 9.99* | 9.35* | 9.05* | 8.10* | 6.95* | 6.51* | 5.59* | 4.56* | 3.94* |
| 100 | 196000 | 1.000 | 7.05* | 6.72* | 6.31* | 6.21* | 5.96* | 5.65* | 5.52* | 4.94* | 4.37* | 4.20* | 3.78* | 3.19* | 2.84* |
| 200 | 196000 | 1.000 | 4.60* | 4.38* | 4.11* | 4.01* | 3.90* | 3.70* | 3.64* | 3.39* | 3.04* | 2.91* | 2.62* | 2.31* | 2.11* |

### contrast:paired/student / bound / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 | 0.04 | 0.05 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.03 | 0.03 | 0.04 | 0.04 | 0.08 | 0.09 |
| 50 | 196000 | 1.000 | 0.06 | 0.06 | 0.06 | 0.05 | 0.05 | 0.06 | 0.07 | 0.08 | 0.09 | 0.09 | 0.11 | 0.14 | 0.18 |
| 100 | 196000 | 1.000 | 0.13 | 0.13 | 0.14 | 0.14 | 0.15 | 0.16 | 0.17 | 0.19 | 0.20 | 0.20 | 0.24 | 0.29 | 0.32 |
| 200 | 196000 | 1.000 | 0.23 | 0.26 | 0.26 | 0.26 | 0.28 | 0.30 | 0.31 | 0.33 | 0.34 | 0.35 | 0.39 | 0.43 | 0.47 |

### contrast:unpaired / bound / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.016 | 2.54* | 2.52* | 2.49* | 2.47* | 2.32* | 2.30* | 2.38* | 2.24* | 2.03* | 1.97* | 1.87* | 1.71* | 1.60* |
| 30 | 196000 | 1.011 | 1.90* | 1.93* | 1.98* | 1.93* | 1.83* | 1.78* | 1.83* | 1.74* | 1.64* | 1.61* | 1.53* | 1.48* | 1.43* |
| 50 | 196000 | 1.000 | 1.35* | 1.34* | 1.40* | 1.44* | 1.37* | 1.32* | 1.35* | 1.38* | 1.39* | 1.38* | 1.36* | 1.27* | 1.23 |
| 100 | 196000 | 1.009 | 1.37* | 1.35* | 1.35* | 1.36* | 1.29* | 1.28* | 1.32* | 1.27* | 1.25 | 1.27* | 1.21 | 1.18 | 1.14 |
| 200 | 196000 | 1.009 | 1.24 | 1.28* | 1.19 | 1.24 | 1.19 | 1.16 | 1.22 | 1.18 | 1.14 | 1.11 | 1.09 | 1.08 | 1.06 |

### contrast:unpaired / bound / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.011 | 3.66* | 3.54* | 3.45* | 3.43* | 3.25* | 3.18* | 3.18* | 2.93* | 2.73* | 2.63* | 2.43* | 2.17* | 2.00* |
| 30 | 196000 | 1.011 | 2.75* | 2.67* | 2.64* | 2.64* | 2.49* | 2.39* | 2.46* | 2.34* | 2.15* | 2.09* | 1.97* | 1.79* | 1.70* |
| 50 | 196000 | 1.007 | 1.84* | 1.83* | 1.79* | 1.80* | 1.72* | 1.67* | 1.73* | 1.69* | 1.61* | 1.57* | 1.54* | 1.45* | 1.39* |
| 100 | 196000 | 1.011 | 1.38* | 1.42* | 1.44* | 1.48* | 1.43* | 1.39* | 1.43* | 1.32* | 1.26* | 1.23 | 1.21 | 1.19 | 1.17 |
| 200 | 196000 | 1.011 | 1.13 | 1.13 | 1.15 | 1.18 | 1.11 | 1.10 | 1.11 | 1.09 | 1.10 | 1.08 | 1.10 | 1.13 | 1.11 |

### contrast:unpaired / bound / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.002 | 3.70* | 3.51* | 3.47* | 3.48* | 3.26* | 3.14* | 3.18* | 2.96* | 2.74* | 2.68* | 2.45* | 2.17* | 2.03* |
| 30 | 196000 | 1.000 | 2.86* | 2.82* | 2.74* | 2.76* | 2.62* | 2.51* | 2.52* | 2.40* | 2.17* | 2.08* | 1.96* | 1.81* | 1.72* |
| 50 | 196000 | 1.010 | 2.11* | 2.07* | 2.04* | 2.07* | 1.95* | 1.90* | 1.91* | 1.86* | 1.76* | 1.73* | 1.66* | 1.57* | 1.47* |
| 100 | 196000 | 1.000 | 1.44* | 1.42* | 1.41* | 1.43* | 1.37* | 1.30* | 1.37* | 1.35* | 1.35* | 1.35* | 1.32* | 1.25* | 1.21 |
| 200 | 196000 | 1.022 | 1.22 | 1.25 | 1.28* | 1.27* | 1.19 | 1.18 | 1.24 | 1.23 | 1.21 | 1.20 | 1.21 | 1.20 | 1.16 |

### contrast:unpaired/boott / bound / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 1.08 | 1.14 | 1.15 | 1.18 | 1.13 | 1.09 | 1.14 | 1.18 | 1.13 | 1.11 | 1.09 | 1.07 | 1.07 |
| 30 | 196000 | 1.000 | 1.02 | 1.08 | 1.05 | 1.09 | 1.04 | 1.05 | 1.09 | 1.11 | 1.08 | 1.07 | 1.05 | 1.04 | 1.03 |
| 50 | 196000 | 1.024 | 1.06 | 1.08 | 1.09 | 1.10 | 1.05 | 1.00 | 1.05 | 1.03 | 0.99 | 0.98 | 1.00 | 1.01 | 1.03 |
| 100 | 196000 | 1.008 | 1.09 | 1.13 | 1.11 | 1.12 | 1.07 | 1.06 | 1.11 | 1.09 | 1.08 | 1.07 | 1.08 | 1.05 | 1.06 |
| 200 | 196000 | 1.001 | 1.15 | 1.10 | 1.09 | 1.11 | 1.04 | 1.04 | 1.09 | 1.06 | 1.00 | 1.00 | 1.03 | 1.04 | 1.05 |

### contrast:unpaired/boott / bound / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.016 | 2.50* | 2.56* | 2.52* | 2.59* | 2.48* | 2.45* | 2.54* | 2.48* | 2.41* | 2.40* | 2.32* | 2.20* | 2.08* |
| 30 | 196000 | 1.017 | 2.83* | 2.81* | 2.86* | 2.91* | 2.76* | 2.69* | 2.69* | 2.60* | 2.40* | 2.35* | 2.25* | 2.04* | 1.95* |
| 50 | 196000 | 1.009 | 2.58* | 2.58* | 2.50* | 2.50* | 2.39* | 2.32* | 2.36* | 2.27* | 2.14* | 2.09* | 1.98* | 1.78* | 1.66* |
| 100 | 196000 | 1.015 | 2.06* | 2.01* | 2.00* | 1.99* | 1.89* | 1.82* | 1.84* | 1.81* | 1.65* | 1.62* | 1.55* | 1.44* | 1.37* |
| 200 | 196000 | 1.000 | 1.62* | 1.66* | 1.67* | 1.66* | 1.56* | 1.51* | 1.54* | 1.53* | 1.41* | 1.39* | 1.32* | 1.24 | 1.19 |

### contrast:unpaired/boott / bound / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.008 | 2.41* | 2.42* | 2.45* | 2.50* | 2.40* | 2.39* | 2.49* | 2.43* | 2.32* | 2.31* | 2.24* | 2.13* | 2.05* |
| 30 | 196000 | 1.009 | 2.80* | 2.82* | 2.83* | 2.86* | 2.74* | 2.67* | 2.71* | 2.64* | 2.45* | 2.38* | 2.25* | 2.07* | 1.96* |
| 50 | 196000 | 1.007 | 2.83* | 2.75* | 2.69* | 2.70* | 2.52* | 2.46* | 2.55* | 2.36* | 2.16* | 2.11* | 2.00* | 1.82* | 1.71* |
| 100 | 196000 | 1.000 | 1.99* | 1.97* | 1.92* | 1.94* | 1.85* | 1.79* | 1.78* | 1.73* | 1.61* | 1.57* | 1.52* | 1.45* | 1.40* |
| 200 | 196000 | 1.016 | 1.55* | 1.51* | 1.50* | 1.52* | 1.41* | 1.39* | 1.45* | 1.39* | 1.36* | 1.31* | 1.29* | 1.24 | 1.17 |

### contrast:unpaired/student / bound / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 1.06 | 1.09 | 1.11 | 1.12 | 1.12 | 1.07 | 1.05 | 1.05 | 1.03 | 1.01 | 1.05 | 1.05 | 1.05 |
| 30 | 196000 | 1.000 | 1.22 | 1.22 | 1.19 | 1.18 | 1.16 | 1.17 | 1.18 | 1.16 | 1.15 | 1.15 | 1.12 | 1.06 | 1.05 |
| 50 | 196000 | 1.000 | 1.07 | 1.04 | 1.00 | 1.00 | 0.99 | 1.01 | 1.02 | 1.03 | 1.03 | 1.04 | 1.05 | 1.05 | 1.02 |
| 100 | 196000 | 1.000 | 1.11 | 1.15 | 1.16 | 1.16 | 1.11 | 1.11 | 1.11 | 1.10 | 1.07 | 1.05 | 1.06 | 1.07 | 1.05 |
| 200 | 196000 | 1.000 | 1.11 | 1.08 | 1.08 | 1.06 | 1.08 | 1.05 | 1.05 | 1.10 | 1.07 | 1.05 | 1.05 | 1.02 | 1.02 |

### contrast:unpaired/student / bound / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.32 | 0.33 | 0.32 | 0.31 | 0.35 | 0.37 | 0.39 | 0.44 | 0.49 | 0.53 | 0.58 | 0.64 | 0.69 |
| 30 | 196000 | 1.000 | 0.46 | 0.50 | 0.51 | 0.51 | 0.53 | 0.56 | 0.57 | 0.61 | 0.66 | 0.68 | 0.71 | 0.77 | 0.82 |
| 50 | 196000 | 1.000 | 0.69 | 0.72 | 0.70 | 0.71 | 0.71 | 0.73 | 0.73 | 0.76 | 0.81 | 0.82 | 0.88 | 0.90 | 0.93 |
| 100 | 196000 | 1.000 | 0.91 | 0.96 | 0.97 | 0.97 | 1.00 | 1.01 | 1.01 | 1.00 | 0.98 | 0.98 | 1.00 | 1.02 | 1.01 |
| 200 | 196000 | 1.000 | 1.16 | 1.14 | 1.10 | 1.09 | 1.10 | 1.09 | 1.10 | 1.10 | 1.08 | 1.08 | 1.06 | 1.03 | 1.03 |

### contrast:unpaired/student / bound / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.32 | 0.32 | 0.35 | 0.37 | 0.38 | 0.42 | 0.42 | 0.49 | 0.51 | 0.52 | 0.55 | 0.62 | 0.68 |
| 30 | 196000 | 1.000 | 0.44 | 0.44 | 0.48 | 0.49 | 0.52 | 0.54 | 0.56 | 0.59 | 0.65 | 0.68 | 0.73 | 0.79 | 0.83 |
| 50 | 196000 | 1.000 | 0.79 | 0.76 | 0.78 | 0.80 | 0.82 | 0.83 | 0.82 | 0.81 | 0.85 | 0.86 | 0.90 | 0.92 | 0.95 |
| 100 | 196000 | 1.000 | 0.88 | 0.90 | 0.94 | 0.95 | 0.92 | 0.90 | 0.92 | 0.93 | 0.94 | 0.94 | 0.96 | 0.97 | 0.98 |
| 200 | 196000 | 1.000 | 1.15 | 1.14 | 1.09 | 1.11 | 1.10 | 1.10 | 1.11 | 1.11 | 1.06 | 1.06 | 1.05 | 1.05 | 1.04 |

### ladder:E_h / bound / nonmono,ip=1,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=1,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=1,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=1,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=1,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=1,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=1,kappa=50 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=1,kappa=50 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=1,kappa=50 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=2,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=2,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=2,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=2,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=2,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=2,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=2,kappa=50 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=2,kappa=50 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / nonmono,ip=2,kappa=50 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1,ip=1,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.07 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 | 0.05 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1,ip=1,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.08 | 0.08 | 0.09 | 0.09 | 0.08 | 0.08 | 0.08 | 0.07 | 0.06 | 0.07 | 0.07 | 0.08 | 0.09 |
| 30 | 196000 | 1.000 | 0.04 | 0.03 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 | 0.04 | 0.04 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1,ip=1,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1,ip=1,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.022 | 0.15 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.13 | 0.12 | 0.12 | 0.12 | 0.12 | 0.10 | 0.10 |
| 30 | 196000 | 1.000 | 0.10 | 0.10 | 0.09 | 0.10 | 0.09 | 0.08 | 0.08 | 0.07 | 0.06 | 0.06 | 0.05 | 0.05 | 0.05 |
| 50 | 196000 | 1.000 | 0.08 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.05 | 0.04 | 0.04 | 0.05 |
| 100 | 196000 | 1.000 | 0.04 | 0.04 | 0.03 | 0.04 | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| 200 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.02 | 0.03 | 0.03 |

### ladder:E_h / bound / order=1,ip=1,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.16 | 0.15 | 0.15 | 0.15 | 0.14 | 0.13 | 0.12 | 0.11 | 0.10 | 0.10 | 0.10 | 0.10 | 0.09 |
| 30 | 196000 | 1.026 | 0.07 | 0.06 | 0.06 | 0.07 | 0.07 | 0.07 | 0.08 | 0.08 | 0.07 | 0.07 | 0.07 | 0.07 | 0.06 |
| 50 | 196000 | 1.016 | 0.08 | 0.07 | 0.06 | 0.06 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 |
| 100 | 196000 | 1.000 | 0.04 | 0.04 | 0.03 | 0.04 | 0.05 | 0.04 | 0.05 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 |
| 200 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 |

### ladder:E_h / bound / order=1,ip=1,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| 30 | 196000 | 1.000 | 0.06 | 0.06 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 |
| 50 | 196000 | 1.000 | 0.04 | 0.04 | 0.05 | 0.05 | 0.04 | 0.04 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.04 |
| 100 | 196000 | 1.000 | 0.05 | 0.04 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 |
| 200 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |

### ladder:E_h / bound / order=1,ip=1,kappa=50 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 30 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| 100 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |

### ladder:E_h / bound / order=1,ip=1,kappa=50 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1,ip=1,kappa=50 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1,ip=2,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.002 | 0.56 | 0.51 | 0.52 | 0.51 | 0.49 | 0.47 | 0.47 | 0.44 | 0.39 | 0.37 | 0.34 | 0.31 | 0.28 |
| 30 | 196000 | 1.027 | 0.35 | 0.33 | 0.33 | 0.34 | 0.32 | 0.32 | 0.31 | 0.27 | 0.27 | 0.28 | 0.25 | 0.23 | 0.22 |
| 50 | 196000 | 1.001 | 0.28 | 0.26 | 0.23 | 0.22 | 0.22 | 0.22 | 0.22 | 0.20 | 0.19 | 0.19 | 0.18 | 0.18 | 0.16 |
| 100 | 196000 | 1.002 | 0.15 | 0.13 | 0.14 | 0.14 | 0.13 | 0.14 | 0.14 | 0.13 | 0.12 | 0.12 | 0.13 | 0.13 | 0.13 |
| 200 | 196000 | 1.079 | 0.09 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.09 | 0.09 | 0.09 | 0.08 | 0.09 | 0.10 |

### ladder:E_h / bound / order=1,ip=2,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.016 | 0.66 | 0.58 | 0.56 | 0.57 | 0.54 | 0.54 | 0.56 | 0.51 | 0.48 | 0.46 | 0.43 | 0.39 | 0.38 |
| 30 | 196000 | 1.002 | 0.49 | 0.47 | 0.50 | 0.50 | 0.47 | 0.46 | 0.48 | 0.42 | 0.41 | 0.39 | 0.36 | 0.32 | 0.29 |
| 50 | 196000 | 1.002 | 0.34 | 0.31 | 0.29 | 0.32 | 0.31 | 0.30 | 0.31 | 0.31 | 0.31 | 0.29 | 0.29 | 0.25 | 0.24 |
| 100 | 196000 | 1.013 | 0.21 | 0.18 | 0.17 | 0.19 | 0.18 | 0.18 | 0.18 | 0.17 | 0.18 | 0.18 | 0.18 | 0.18 | 0.17 |
| 200 | 196000 | 1.037 | 0.14 | 0.12 | 0.13 | 0.15 | 0.14 | 0.13 | 0.14 | 0.12 | 0.12 | 0.11 | 0.11 | 0.13 | 0.13 |

### ladder:E_h / bound / order=1,ip=2,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.008 | 0.21 | 0.19 | 0.19 | 0.18 | 0.18 | 0.18 | 0.18 | 0.17 | 0.17 | 0.16 | 0.17 | 0.16 | 0.15 |
| 30 | 196000 | 1.018 | 0.23 | 0.21 | 0.20 | 0.20 | 0.20 | 0.18 | 0.18 | 0.17 | 0.15 | 0.14 | 0.13 | 0.14 | 0.13 |
| 50 | 196000 | 1.000 | 0.14 | 0.12 | 0.13 | 0.13 | 0.12 | 0.12 | 0.13 | 0.12 | 0.12 | 0.12 | 0.12 | 0.11 | 0.11 |
| 100 | 196000 | 1.000 | 0.11 | 0.10 | 0.10 | 0.09 | 0.09 | 0.08 | 0.10 | 0.09 | 0.08 | 0.09 | 0.09 | 0.09 | 0.09 |
| 200 | 196000 | 1.000 | 0.12 | 0.11 | 0.12 | 0.12 | 0.12 | 0.11 | 0.12 | 0.11 | 0.11 | 0.12 | 0.10 | 0.10 | 0.10 |

### ladder:E_h / bound / order=1,ip=2,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.10 | 0.10 | 0.11 | 0.11 | 0.10 | 0.09 | 0.09 | 0.09 | 0.08 | 0.07 | 0.07 | 0.06 | 0.05 |
| 30 | 196000 | 1.000 | 0.08 | 0.07 | 0.07 | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| 50 | 196000 | 1.000 | 0.06 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.05 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 |

### ladder:E_h / bound / order=1,ip=2,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.04 | 0.04 | 0.06 | 0.06 | 0.05 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.05 | 0.04 |
| 30 | 196000 | 1.000 | 0.05 | 0.04 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 |
| 50 | 196000 | 1.000 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| 100 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.03 | 0.02 |
| 200 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 |

### ladder:E_h / bound / order=1,ip=2,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.05 | 0.06 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 |
| 30 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.01 | 0.02 | 0.02 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### ladder:E_h / bound / order=1,ip=2,kappa=50 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1,ip=2,kappa=50 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1,ip=2,kappa=50 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1.5,ip=1,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.13 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.12 | 0.12 | 0.10 | 0.11 | 0.10 | 0.11 | 0.11 |
| 30 | 196000 | 1.000 | 0.04 | 0.04 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 | 0.05 | 0.06 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.03 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1.5,ip=1,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.08 | 0.07 | 0.07 | 0.07 | 0.08 | 0.08 | 0.07 | 0.07 | 0.07 | 0.06 | 0.06 | 0.06 | 0.06 |
| 30 | 196000 | 1.000 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.05 | 0.05 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.04 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 |

### ladder:E_h / bound / order=1.5,ip=1,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.04 | 0.05 | 0.05 | 0.04 | 0.04 | 0.03 | 0.05 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.04 |
| 30 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.02 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1.5,ip=1,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.03 | 0.02 | 0.03 | 0.03 | 0.02 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 |
| 30 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 100 | 196000 | 1.000 | 0.03 | 0.02 | 0.03 | 0.03 | 0.02 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### ladder:E_h / bound / order=1.5,ip=1,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 30 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### ladder:E_h / bound / order=1.5,ip=1,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |

### ladder:E_h / bound / order=1.5,ip=1,kappa=50 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1.5,ip=1,kappa=50 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1.5,ip=1,kappa=50 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1.5,ip=2,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 |
| 30 | 196000 | 1.000 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 |
| 50 | 196000 | 1.000 | 0.06 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 100 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 |
| 200 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### ladder:E_h / bound / order=1.5,ip=2,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.03 | 0.02 | 0.03 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 50 | 196000 | 1.000 | 0.08 | 0.07 | 0.06 | 0.06 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.02 | 0.02 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |

### ladder:E_h / bound / order=1.5,ip=2,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.03 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 30 | 196000 | 1.000 | 0.03 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 |
| 50 | 196000 | 1.000 | 0.02 | 0.02 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### ladder:E_h / bound / order=1.5,ip=2,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1.5,ip=2,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1.5,ip=2,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1.5,ip=2,kappa=50 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1.5,ip=2,kappa=50 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=1.5,ip=2,kappa=50 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=1,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.07 | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.06 | 0.07 | 0.08 | 0.08 | 0.07 | 0.06 | 0.07 |
| 30 | 196000 | 1.000 | 0.06 | 0.05 | 0.06 | 0.05 | 0.05 | 0.05 | 0.04 | 0.06 | 0.04 | 0.04 | 0.05 | 0.05 | 0.05 |
| 50 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.02 | 0.02 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 |

### ladder:E_h / bound / order=2,ip=1,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 |
| 30 | 196000 | 1.000 | 0.04 | 0.04 | 0.05 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.02 | 0.03 | 0.03 | 0.03 | 0.02 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### ladder:E_h / bound / order=2,ip=1,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 |
| 30 | 196000 | 1.000 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.03 |
| 50 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 |

### ladder:E_h / bound / order=2,ip=1,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=1,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=1,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=1,kappa=50 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=1,kappa=50 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=1,kappa=50 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=2,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=2,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=2,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=2,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=2,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=2,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=2,kappa=50 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=2,kappa=50 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h / bound / order=2,ip=2,kappa=50 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / nonmono,ip=1,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / nonmono,ip=1,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / nonmono,ip=1,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / nonmono,ip=2,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / nonmono,ip=2,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / nonmono,ip=2,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / order=1,ip=1,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.03 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / order=1,ip=1,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.018 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 | 0.05 | 0.05 | 0.07 | 0.07 |
| 30 | 196000 | 1.038 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.04 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / order=1,ip=1,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.03 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / order=1,ip=1,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.07 | 0.06 | 0.07 | 0.07 | 0.06 | 0.06 | 0.05 | 0.05 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 |
| 30 | 196000 | 1.033 | 0.05 | 0.05 | 0.06 | 0.05 | 0.06 | 0.05 | 0.06 | 0.06 | 0.05 | 0.05 | 0.04 | 0.05 | 0.04 |
| 50 | 196000 | 1.000 | 0.04 | 0.03 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.03 |
| 100 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 |
| 200 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 |

### ladder:E_h/boott / bound / order=1,ip=1,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.012 | 0.12 | 0.10 | 0.11 | 0.11 | 0.10 | 0.09 | 0.09 | 0.09 | 0.08 | 0.09 | 0.09 | 0.08 | 0.08 |
| 30 | 196000 | 1.000 | 0.06 | 0.06 | 0.07 | 0.06 | 0.07 | 0.08 | 0.08 | 0.07 | 0.08 | 0.08 | 0.07 | 0.08 | 0.07 |
| 50 | 196000 | 1.000 | 0.12 | 0.11 | 0.10 | 0.09 | 0.09 | 0.09 | 0.09 | 0.08 | 0.07 | 0.07 | 0.07 | 0.07 | 0.06 |
| 100 | 196000 | 1.000 | 0.04 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| 200 | 196000 | 1.000 | 0.05 | 0.04 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 | 0.05 | 0.04 | 0.05 | 0.05 |

### ladder:E_h/boott / bound / order=1,ip=1,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| 30 | 196000 | 1.000 | 0.04 | 0.04 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 50 | 196000 | 1.000 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 | 0.05 | 0.04 | 0.04 |
| 100 | 196000 | 1.000 | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 |
| 200 | 196000 | 1.000 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 | 0.04 | 0.03 | 0.04 |

### ladder:E_h/boott / bound / order=1,ip=2,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.39 | 0.40 | 0.38 | 0.36 | 0.34 | 0.31 | 0.31 | 0.29 | 0.28 | 0.28 | 0.24 | 0.22 | 0.22 |
| 30 | 196000 | 1.028 | 0.26 | 0.25 | 0.24 | 0.24 | 0.22 | 0.22 | 0.23 | 0.20 | 0.19 | 0.18 | 0.17 | 0.17 | 0.16 |
| 50 | 196000 | 1.000 | 0.15 | 0.14 | 0.13 | 0.13 | 0.13 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| 100 | 196000 | 1.019 | 0.10 | 0.09 | 0.10 | 0.11 | 0.10 | 0.09 | 0.09 | 0.11 | 0.10 | 0.10 | 0.10 | 0.11 | 0.11 |
| 200 | 196000 | 1.000 | 0.10 | 0.09 | 0.08 | 0.08 | 0.08 | 0.08 | 0.09 | 0.10 | 0.10 | 0.10 | 0.10 | 0.12 | 0.12 |

### ladder:E_h/boott / bound / order=1,ip=2,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.024 | 0.57 | 0.55 | 0.53 | 0.52 | 0.51 | 0.48 | 0.47 | 0.45 | 0.42 | 0.40 | 0.37 | 0.35 | 0.33 |
| 30 | 196000 | 1.006 | 0.38 | 0.37 | 0.38 | 0.37 | 0.36 | 0.35 | 0.37 | 0.37 | 0.35 | 0.34 | 0.33 | 0.31 | 0.30 |
| 50 | 196000 | 1.002 | 0.39 | 0.35 | 0.33 | 0.34 | 0.33 | 0.31 | 0.31 | 0.30 | 0.27 | 0.27 | 0.26 | 0.24 | 0.25 |
| 100 | 196000 | 1.000 | 0.29 | 0.28 | 0.26 | 0.27 | 0.25 | 0.25 | 0.24 | 0.25 | 0.25 | 0.24 | 0.22 | 0.21 | 0.20 |
| 200 | 196000 | 1.007 | 0.15 | 0.15 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.15 | 0.15 | 0.15 | 0.14 | 0.14 | 0.14 |

### ladder:E_h/boott / bound / order=1,ip=2,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.16 | 0.14 | 0.17 | 0.18 | 0.16 | 0.17 | 0.16 | 0.16 | 0.16 | 0.16 | 0.17 | 0.16 | 0.16 |
| 30 | 196000 | 1.005 | 0.13 | 0.14 | 0.12 | 0.11 | 0.12 | 0.11 | 0.13 | 0.13 | 0.12 | 0.12 | 0.11 | 0.11 | 0.11 |
| 50 | 196000 | 1.004 | 0.14 | 0.15 | 0.15 | 0.15 | 0.15 | 0.13 | 0.13 | 0.11 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| 100 | 196000 | 1.000 | 0.12 | 0.12 | 0.12 | 0.13 | 0.13 | 0.14 | 0.14 | 0.13 | 0.13 | 0.13 | 0.12 | 0.11 | 0.11 |
| 200 | 196000 | 1.011 | 0.10 | 0.10 | 0.09 | 0.09 | 0.09 | 0.09 | 0.10 | 0.10 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 |

### ladder:E_h/boott / bound / order=1,ip=2,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.06 | 0.06 | 0.06 | 0.05 | 0.05 | 0.06 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| 30 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.02 | 0.02 |
| 50 | 196000 | 1.000 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.04 | 0.04 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 |
| 100 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 |

### ladder:E_h/boott / bound / order=1,ip=2,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.04 | 0.04 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 |
| 30 | 196000 | 1.000 | 0.06 | 0.06 | 0.06 | 0.05 | 0.06 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| 50 | 196000 | 1.000 | 0.05 | 0.04 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 100 | 196000 | 1.000 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 |
| 200 | 196000 | 1.000 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |

### ladder:E_h/boott / bound / order=1,ip=2,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 30 | 196000 | 1.000 | 0.03 | 0.02 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 |

### ladder:E_h/boott / bound / order=2,ip=1,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.04 | 0.04 | 0.03 | 0.04 | 0.04 | 0.05 | 0.05 | 0.06 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 |
| 30 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.03 | 0.03 | 0.02 | 0.03 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### ladder:E_h/boott / bound / order=2,ip=1,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.02 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 |
| 30 | 196000 | 1.000 | 0.03 | 0.02 | 0.03 | 0.03 | 0.02 | 0.03 | 0.02 | 0.03 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 |
| 100 | 196000 | 1.000 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 |

### ladder:E_h/boott / bound / order=2,ip=1,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 |
| 30 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 50 | 196000 | 1.000 | 0.03 | 0.02 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.02 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 |

### ladder:E_h/boott / bound / order=2,ip=1,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / order=2,ip=1,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / order=2,ip=1,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / order=2,ip=2,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / order=2,ip=2,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / order=2,ip=2,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / order=2,ip=2,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / order=2,ip=2,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/boott / bound / order=2,ip=2,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / nonmono,ip=1,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / nonmono,ip=1,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / nonmono,ip=1,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / nonmono,ip=2,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / nonmono,ip=2,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / nonmono,ip=2,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / order=1,ip=1,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / order=1,ip=1,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.03 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 30 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / order=1,ip=1,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / order=1,ip=1,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.022 | 0.07 | 0.07 | 0.06 | 0.06 | 0.05 | 0.05 | 0.05 | 0.04 | 0.05 | 0.05 | 0.06 | 0.06 | 0.06 |
| 30 | 196000 | 1.000 | 0.07 | 0.07 | 0.07 | 0.08 | 0.07 | 0.08 | 0.09 | 0.07 | 0.07 | 0.06 | 0.06 | 0.06 | 0.05 |
| 50 | 196000 | 1.000 | 0.03 | 0.02 | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 | 0.03 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| 100 | 196000 | 1.000 | 0.03 | 0.02 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 |
| 200 | 196000 | 1.000 | 0.08 | 0.08 | 0.07 | 0.07 | 0.06 | 0.06 | 0.06 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 |

### ladder:E_h/student / bound / order=1,ip=1,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 30 | 196000 | 1.000 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.03 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 |
| 50 | 196000 | 1.000 | 0.05 | 0.04 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.04 | 0.03 | 0.03 | 0.04 | 0.04 | 0.04 |
| 100 | 196000 | 1.000 | 0.04 | 0.04 | 0.03 | 0.04 | 0.03 | 0.04 | 0.04 | 0.03 | 0.03 | 0.04 | 0.03 | 0.03 | 0.04 |
| 200 | 196000 | 1.000 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 |

### ladder:E_h/student / bound / order=1,ip=1,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 100 | 196000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 |
| 200 | 196000 | 1.000 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.04 | 0.03 | 0.04 |

### ladder:E_h/student / bound / order=1,ip=2,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.006 | 0.30 | 0.30 | 0.29 | 0.29 | 0.27 | 0.26 | 0.26 | 0.25 | 0.25 | 0.25 | 0.24 | 0.22 | 0.21 |
| 30 | 196000 | 1.000 | 0.19 | 0.17 | 0.15 | 0.15 | 0.14 | 0.15 | 0.16 | 0.15 | 0.16 | 0.16 | 0.15 | 0.14 | 0.15 |
| 50 | 196000 | 1.019 | 0.15 | 0.15 | 0.16 | 0.16 | 0.17 | 0.16 | 0.16 | 0.15 | 0.14 | 0.15 | 0.15 | 0.14 | 0.14 |
| 100 | 196000 | 1.000 | 0.19 | 0.18 | 0.16 | 0.17 | 0.15 | 0.14 | 0.14 | 0.13 | 0.12 | 0.11 | 0.12 | 0.11 | 0.12 |
| 200 | 196000 | 1.005 | 0.14 | 0.12 | 0.11 | 0.12 | 0.11 | 0.10 | 0.11 | 0.10 | 0.10 | 0.10 | 0.10 | 0.11 | 0.11 |

### ladder:E_h/student / bound / order=1,ip=2,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.23 | 0.20 | 0.20 | 0.20 | 0.20 | 0.22 | 0.21 | 0.19 | 0.17 | 0.17 | 0.18 | 0.19 | 0.18 |
| 30 | 196000 | 1.007 | 0.21 | 0.22 | 0.22 | 0.23 | 0.22 | 0.22 | 0.22 | 0.20 | 0.20 | 0.20 | 0.20 | 0.21 | 0.20 |
| 50 | 196000 | 1.000 | 0.21 | 0.21 | 0.22 | 0.23 | 0.21 | 0.21 | 0.22 | 0.22 | 0.21 | 0.22 | 0.20 | 0.19 | 0.18 |
| 100 | 196000 | 1.041 | 0.17 | 0.15 | 0.14 | 0.14 | 0.14 | 0.13 | 0.13 | 0.14 | 0.14 | 0.14 | 0.13 | 0.14 | 0.14 |
| 200 | 196000 | 1.000 | 0.15 | 0.14 | 0.13 | 0.14 | 0.12 | 0.13 | 0.12 | 0.12 | 0.12 | 0.11 | 0.12 | 0.13 | 0.13 |

### ladder:E_h/student / bound / order=1,ip=2,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.07 | 0.07 | 0.08 | 0.09 | 0.08 | 0.09 |
| 30 | 196000 | 1.000 | 0.11 | 0.10 | 0.11 | 0.13 | 0.12 | 0.13 | 0.12 | 0.10 | 0.10 | 0.10 | 0.09 | 0.09 | 0.09 |
| 50 | 196000 | 1.000 | 0.06 | 0.06 | 0.07 | 0.08 | 0.08 | 0.09 | 0.09 | 0.08 | 0.08 | 0.07 | 0.07 | 0.08 | 0.08 |
| 100 | 196000 | 1.000 | 0.06 | 0.05 | 0.06 | 0.05 | 0.05 | 0.06 | 0.06 | 0.05 | 0.07 | 0.07 | 0.07 | 0.08 | 0.08 |
| 200 | 196000 | 1.000 | 0.06 | 0.06 | 0.06 | 0.06 | 0.07 | 0.08 | 0.08 | 0.08 | 0.07 | 0.07 | 0.08 | 0.08 | 0.08 |

### ladder:E_h/student / bound / order=1,ip=2,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.06 | 0.05 | 0.06 | 0.05 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| 30 | 196000 | 1.000 | 0.04 | 0.04 | 0.05 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 50 | 196000 | 1.000 | 0.07 | 0.06 | 0.06 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 |
| 100 | 196000 | 1.000 | 0.03 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### ladder:E_h/student / bound / order=1,ip=2,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 30 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 50 | 196000 | 1.000 | 0.06 | 0.05 | 0.05 | 0.04 | 0.04 | 0.03 | 0.04 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 100 | 196000 | 1.000 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 200 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 |

### ladder:E_h/student / bound / order=1,ip=2,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 |
| 30 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

### ladder:E_h/student / bound / order=2,ip=1,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.06 | 0.05 | 0.05 | 0.05 | 0.04 | 0.05 | 0.04 | 0.05 | 0.04 | 0.05 | 0.05 | 0.04 | 0.04 |
| 30 | 196000 | 1.000 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 |

### ladder:E_h/student / bound / order=2,ip=1,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 50 | 196000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 |

### ladder:E_h/student / bound / order=2,ip=1,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 |

### ladder:E_h/student / bound / order=2,ip=1,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / order=2,ip=1,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / order=2,ip=1,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / order=2,ip=2,kappa=12 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / order=2,ip=2,kappa=12 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / order=2,ip=2,kappa=12 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / order=2,ip=2,kappa=25 / normal

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / order=2,ip=2,kappa=25 / stress

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### ladder:E_h/student / bound / order=2,ip=2,kappa=25 / stress-

| n | R | maxD | 0.000893 | 0.00104 | 0.00119 | 0.00125 | 0.00139 | 0.00156 | 0.00167 | 0.00208 | 0.00278 | 0.00313 | 0.00417 | 0.00625 | 0.00833 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 50 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 196000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### pipeline:K3,ip=1 / RL / bias,order=1 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1 / RL / bias,order=1 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1 / RL / bias,order=1 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1 / RL / bias,order=2 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1 / RL / bias,order=2 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1 / RL / bias,order=2 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1 / RL / boott,bias,order=1 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1 / RL / boott,bias,order=1 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1 / RL / boott,bias,order=1 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1 / RL / zero / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1 / RL / zero / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1 / RL / zero / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/boott / RL / bias,order=1 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/boott / RL / bias,order=1 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/boott / RL / bias,order=1 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/boott / RL / bias,order=2 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/boott / RL / bias,order=2 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/boott / RL / bias,order=2 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/boott / RL / boott,bias,order=1 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/boott / RL / boott,bias,order=1 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/boott / RL / boott,bias,order=1 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/boott / RL / zero / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/boott / RL / zero / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/boott / RL / zero / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/student / RL / bias,order=1 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/student / RL / bias,order=1 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/student / RL / bias,order=1 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/student / RL / bias,order=2 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/student / RL / bias,order=2 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/student / RL / bias,order=2 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/student / RL / boott,bias,order=1 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/student / RL / boott,bias,order=1 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/student / RL / boott,bias,order=1 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/student / RL / zero / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/student / RL / zero / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K3,ip=1/student / RL / zero / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 1000 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2 / RL / bias,order=1 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2 / RL / bias,order=1 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2 / RL / bias,order=1 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2 / RL / bias,order=2 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2 / RL / bias,order=2 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2 / RL / bias,order=2 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2 / RL / boott,bias,order=1 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2 / RL / boott,bias,order=1 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2 / RL / boott,bias,order=1 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2 / RL / zero / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2 / RL / zero / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2 / RL / zero / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/boott / RL / bias,order=1 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/boott / RL / bias,order=1 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/boott / RL / bias,order=1 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/boott / RL / bias,order=2 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/boott / RL / bias,order=2 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/boott / RL / bias,order=2 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/boott / RL / boott,bias,order=1 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/boott / RL / boott,bias,order=1 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/boott / RL / boott,bias,order=1 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/boott / RL / zero / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/boott / RL / zero / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/boott / RL / zero / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/student / RL / bias,order=1 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/student / RL / bias,order=1 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/student / RL / bias,order=1 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/student / RL / bias,order=2 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/student / RL / bias,order=2 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/student / RL / bias,order=2 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/student / RL / boott,bias,order=1 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/student / RL / boott,bias,order=1 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/student / RL / boott,bias,order=1 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/student / RL / zero / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/student / RL / zero / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### pipeline:K4,ip=2/student / RL / zero / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 300 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |

### sampling:boott / lower / normal

| n | R | maxD | 0.00357 | 0.00417 | 0.005 | 0.00625 | 0.00833 | 0.0125 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 49000 | 1.000 | 1.12 | 1.09 | 1.12 | 1.08 | 1.11 | 1.09 | 1.05 | 1.03 |
| 30 | 49000 | 1.000 | 0.87 | 0.84 | 0.93 | 0.90 | 0.97 | 1.01 | 1.02 | 1.03 |
| 50 | 49000 | 1.000 | 1.07 | 1.14 | 1.18 | 1.07 | 1.10 | 1.07 | 1.00 | 1.01 |
| 100 | 49000 | 1.000 | 1.13 | 1.13 | 1.21 | 1.15 | 1.12 | 1.06 | 1.05 | 1.01 |
| 200 | 49000 | 1.000 | 1.06 | 1.05 | 1.18 | 1.13 | 1.12 | 1.06 | 1.05 | 1.03 |
| 500 | 49000 | 1.000 | 1.25 | 1.28* | 1.28* | 1.20 | 1.19 | 1.09 | 1.05 | 1.04 |
| 1000 | 49000 | 1.000 | 1.04 | 1.03 | 1.10 | 1.09 | 1.09 | 1.07 | 1.05 | 1.06 |
| 1650 | 49000 | 1.000 | 0.94 | 0.95 | 1.03 | 0.95 | 0.96 | 1.01 | 1.06 | 1.03 |

### sampling:boott / lower / stress

| n | R | maxD | 0.00357 | 0.00417 | 0.005 | 0.00625 | 0.00833 | 0.0125 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 49000 | 1.000 | 0.30 | 0.31 | 0.39 | 0.42 | 0.51 | 0.58 | 0.71 | 0.87 |
| 30 | 49000 | 1.000 | 0.50 | 0.50 | 0.57 | 0.55 | 0.60 | 0.69 | 0.82 | 0.92 |
| 50 | 49000 | 1.000 | 0.63 | 0.66 | 0.75 | 0.75 | 0.81 | 0.86 | 0.90 | 0.96 |
| 100 | 49000 | 1.000 | 0.80 | 0.84 | 0.87 | 0.87 | 0.93 | 0.91 | 0.98 | 1.00 |
| 200 | 49000 | 1.000 | 0.88 | 0.93 | 0.97 | 0.90 | 0.98 | 0.98 | 1.04 | 1.03 |
| 500 | 49000 | 1.000 | 0.97 | 1.01 | 1.03 | 1.01 | 1.06 | 1.03 | 1.04 | 1.02 |
| 1000 | 49000 | 1.000 | 0.94 | 0.92 | 0.97 | 0.96 | 1.01 | 1.01 | 1.00 | 1.00 |
| 1650 | 49000 | 1.000 | 1.21 | 1.21 | 1.24 | 1.14 | 1.13 | 1.08 | 1.10 | 1.05 |

### sampling:boott / lower / stress-

| n | R | maxD | 0.00357 | 0.00417 | 0.005 | 0.00625 | 0.00833 | 0.0125 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 49000 | 1.000 | 8.15* | 7.92* | 7.51* | 6.59* | 5.95* | 4.97* | 3.66* | 2.54* |
| 30 | 49000 | 1.000 | 8.16* | 7.67* | 7.00* | 6.12* | 5.38* | 4.26* | 2.82* | 1.89* |
| 50 | 49000 | 1.000 | 5.07* | 4.58* | 4.25* | 3.66* | 3.11* | 2.43* | 1.71* | 1.28* |
| 100 | 49000 | 1.000 | 2.00* | 1.93* | 1.87* | 1.68* | 1.61* | 1.39* | 1.20 | 1.10 |
| 200 | 49000 | 1.000 | 1.12 | 1.14 | 1.15 | 1.09 | 1.12 | 1.08 | 1.05 | 1.03 |
| 500 | 49000 | 1.000 | 1.07 | 1.04 | 1.08 | 0.98 | 1.05 | 1.05 | 1.07 | 1.06 |
| 1000 | 49000 | 1.000 | 1.19 | 1.16 | 1.15 | 1.07 | 1.11 | 1.09 | 1.08 | 1.04 |
| 1650 | 49000 | 1.000 | 1.07 | 1.09 | 1.17 | 1.09 | 1.13 | 1.07 | 1.08 | 1.04 |

### sampling:boott / upper / normal

| n | R | maxD | 0.00357 | 0.00417 | 0.005 | 0.00625 | 0.00833 | 0.0125 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 49000 | 1.000 | 1.13 | 1.17 | 1.19 | 1.12 | 1.11 | 1.03 | 1.03 | 1.02 |
| 30 | 49000 | 1.000 | 1.27* | 1.25 | 1.22 | 1.12 | 1.14 | 1.08 | 1.08 | 1.07 |
| 50 | 49000 | 1.000 | 0.94 | 0.98 | 1.00 | 1.00 | 1.08 | 1.06 | 1.07 | 1.02 |
| 100 | 49000 | 1.000 | 1.05 | 1.06 | 1.14 | 1.10 | 1.09 | 1.03 | 1.07 | 1.03 |
| 200 | 49000 | 1.000 | 1.12 | 1.14 | 1.12 | 1.06 | 1.02 | 0.97 | 0.98 | 0.99 |
| 500 | 49000 | 1.000 | 0.87 | 0.93 | 0.98 | 0.97 | 1.04 | 1.05 | 1.02 | 1.02 |
| 1000 | 49000 | 1.000 | 1.11 | 1.10 | 1.14 | 1.07 | 1.10 | 1.04 | 1.03 | 1.03 |
| 1650 | 49000 | 1.000 | 0.93 | 1.00 | 1.10 | 1.06 | 1.08 | 1.04 | 1.06 | 1.06 |

### sampling:boott / upper / stress

| n | R | maxD | 0.00357 | 0.00417 | 0.005 | 0.00625 | 0.00833 | 0.0125 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 49000 | 1.000 | 8.64* | 8.18* | 7.85* | 7.05* | 6.35* | 5.24* | 3.73* | 2.59* |
| 30 | 49000 | 1.000 | 8.21* | 7.78* | 7.28* | 6.33* | 5.49* | 4.32* | 2.88* | 1.90* |
| 50 | 49000 | 1.000 | 5.20* | 4.76* | 4.34* | 3.68* | 3.19* | 2.52* | 1.74* | 1.29* |
| 100 | 49000 | 1.000 | 1.86* | 1.77* | 1.73* | 1.56* | 1.48* | 1.30* | 1.12 | 1.03 |
| 200 | 49000 | 1.000 | 1.25* | 1.27* | 1.25 | 1.17 | 1.20 | 1.09 | 1.06 | 1.03 |
| 500 | 49000 | 1.000 | 1.21 | 1.22 | 1.23 | 1.16 | 1.13 | 1.09 | 1.05 | 1.02 |
| 1000 | 49000 | 1.000 | 1.02 | 1.05 | 1.11 | 1.04 | 1.09 | 1.02 | 0.98 | 1.01 |
| 1650 | 49000 | 1.000 | 1.10 | 1.09 | 1.14 | 1.06 | 1.09 | 1.04 | 1.07 | 1.05 |

### sampling:boott / upper / stress-

| n | R | maxD | 0.00357 | 0.00417 | 0.005 | 0.00625 | 0.00833 | 0.0125 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 49000 | 1.000 | 0.20 | 0.22 | 0.30 | 0.37 | 0.45 | 0.55 | 0.73 | 0.88 |
| 30 | 49000 | 1.000 | 0.47 | 0.50 | 0.52 | 0.52 | 0.61 | 0.68 | 0.85 | 0.94 |
| 50 | 49000 | 1.000 | 0.58 | 0.64 | 0.69 | 0.72 | 0.79 | 0.82 | 0.89 | 0.98 |
| 100 | 49000 | 1.000 | 0.79 | 0.83 | 0.90 | 0.89 | 0.91 | 0.93 | 0.97 | 0.98 |
| 200 | 49000 | 1.000 | 0.86 | 0.88 | 0.90 | 0.90 | 0.92 | 0.92 | 0.99 | 1.00 |
| 500 | 49000 | 1.000 | 1.14 | 1.09 | 1.15 | 1.07 | 1.05 | 0.98 | 1.00 | 1.02 |
| 1000 | 49000 | 1.000 | 1.06 | 1.13 | 1.12 | 1.07 | 1.11 | 1.07 | 1.06 | 1.06 |
| 1650 | 49000 | 1.000 | 1.00 | 1.02 | 1.08 | 1.05 | 1.06 | 1.06 | 1.04 | 1.01 |

### sampling:student / lower / normal

| n | R | maxD | 0.00357 | 0.00417 | 0.005 | 0.00625 | 0.00833 | 0.0125 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 49000 | 1.000 | 1.12 | 1.06 | 1.12 | 1.10 | 1.06 | 1.04 | 1.03 | 1.03 |
| 30 | 49000 | 1.000 | 1.18 | 1.13 | 1.09 | 1.14 | 1.12 | 1.12 | 1.08 | 1.04 |
| 50 | 49000 | 1.000 | 1.17 | 1.18 | 1.17 | 1.14 | 1.17 | 1.09 | 1.03 | 1.05 |
| 100 | 49000 | 1.000 | 1.21 | 1.21 | 1.22 | 1.21 | 1.16 | 1.12 | 1.06 | 1.04 |
| 200 | 49000 | 1.000 | 1.24 | 1.29* | 1.23 | 1.22 | 1.14 | 1.09 | 1.02 | 1.02 |
| 500 | 49000 | 1.000 | 1.17 | 1.17 | 1.14 | 1.11 | 1.07 | 1.06 | 1.04 | 1.02 |
| 1000 | 49000 | 1.000 | 1.07 | 1.01 | 1.04 | 1.08 | 1.11 | 1.07 | 1.03 | 1.04 |
| 1650 | 49000 | 1.000 | 1.01 | 0.98 | 0.97 | 0.93 | 0.92 | 0.95 | 0.97 | 1.01 |

### sampling:student / lower / stress

| n | R | maxD | 0.00357 | 0.00417 | 0.005 | 0.00625 | 0.00833 | 0.0125 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 49000 | 1.000 | 0.04 | 0.03 | 0.03 | 0.06 | 0.09 | 0.11 | 0.20 | 0.36 |
| 30 | 49000 | 1.000 | 0.06 | 0.06 | 0.09 | 0.10 | 0.13 | 0.16 | 0.26 | 0.42 |
| 50 | 49000 | 1.000 | 0.16 | 0.15 | 0.20 | 0.22 | 0.22 | 0.28 | 0.38 | 0.57 |
| 100 | 49000 | 1.000 | 0.29 | 0.29 | 0.31 | 0.30 | 0.33 | 0.39 | 0.52 | 0.64 |
| 200 | 49000 | 1.000 | 0.41 | 0.42 | 0.46 | 0.44 | 0.51 | 0.54 | 0.65 | 0.76 |
| 500 | 49000 | 1.000 | 0.54 | 0.58 | 0.61 | 0.69 | 0.71 | 0.72 | 0.78 | 0.86 |
| 1000 | 49000 | 1.000 | 0.66 | 0.67 | 0.71 | 0.74 | 0.74 | 0.78 | 0.83 | 0.88 |
| 1650 | 49000 | 1.000 | 0.72 | 0.74 | 0.74 | 0.72 | 0.74 | 0.80 | 0.87 | 0.92 |

### sampling:student / lower / stress-

| n | R | maxD | 0.00357 | 0.00417 | 0.005 | 0.00625 | 0.00833 | 0.0125 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 49000 | 1.000 | 9.32* | 8.81* | 8.14* | 7.32* | 6.51* | 5.42* | 3.94* | 2.71* |
| 30 | 49000 | 1.000 | 9.13* | 8.37* | 7.55* | 6.71* | 5.77* | 4.63* | 3.18* | 2.21* |
| 50 | 49000 | 1.000 | 6.54* | 6.04* | 5.38* | 4.82* | 4.19* | 3.41* | 2.48* | 1.85* |
| 100 | 49000 | 1.000 | 3.97* | 3.67* | 3.40* | 3.14* | 2.81* | 2.51* | 1.96* | 1.57* |
| 200 | 49000 | 1.000 | 2.90* | 2.75* | 2.67* | 2.47* | 2.19* | 1.95* | 1.59* | 1.35* |
| 500 | 49000 | 1.000 | 2.05* | 1.96* | 1.89* | 1.73* | 1.66* | 1.56* | 1.38* | 1.23 |
| 1000 | 49000 | 1.000 | 1.87* | 1.83* | 1.78* | 1.66* | 1.57* | 1.45* | 1.31* | 1.18 |
| 1650 | 49000 | 1.000 | 1.57* | 1.55* | 1.53* | 1.48* | 1.44* | 1.38* | 1.26* | 1.15 |

### sampling:student / upper / normal

| n | R | maxD | 0.00357 | 0.00417 | 0.005 | 0.00625 | 0.00833 | 0.0125 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 49000 | 1.000 | 1.08 | 1.06 | 1.06 | 1.05 | 1.07 | 1.07 | 1.05 | 1.02 |
| 30 | 49000 | 1.000 | 1.20 | 1.18 | 1.18 | 1.14 | 1.12 | 1.11 | 1.06 | 1.06 |
| 50 | 49000 | 1.000 | 1.06 | 1.01 | 1.01 | 1.01 | 0.98 | 0.99 | 0.99 | 0.99 |
| 100 | 49000 | 1.000 | 1.15 | 1.16 | 1.15 | 1.14 | 1.07 | 1.07 | 1.04 | 1.02 |
| 200 | 49000 | 1.000 | 1.10 | 1.07 | 1.06 | 1.10 | 1.08 | 1.07 | 1.05 | 1.02 |
| 500 | 49000 | 1.000 | 1.04 | 1.01 | 1.02 | 1.02 | 1.03 | 1.01 | 1.03 | 1.02 |
| 1000 | 49000 | 1.000 | 1.02 | 1.00 | 1.01 | 1.01 | 1.00 | 1.01 | 1.07 | 1.05 |
| 1650 | 49000 | 1.000 | 1.22 | 1.20 | 1.15 | 1.15 | 1.14 | 1.12 | 1.08 | 1.04 |

### sampling:student / upper / stress

| n | R | maxD | 0.00357 | 0.00417 | 0.005 | 0.00625 | 0.00833 | 0.0125 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 49000 | 1.000 | 9.31* | 8.65* | 8.09* | 7.40* | 6.50* | 5.38* | 3.80* | 2.66* |
| 30 | 49000 | 1.000 | 8.72* | 8.12* | 7.44* | 6.61* | 5.63* | 4.52* | 3.13* | 2.19* |
| 50 | 49000 | 1.000 | 6.26* | 5.80* | 5.23* | 4.67* | 4.00* | 3.28* | 2.41* | 1.79* |
| 100 | 49000 | 1.000 | 4.04* | 3.78* | 3.55* | 3.24* | 2.91* | 2.51* | 1.97* | 1.57* |
| 200 | 49000 | 1.000 | 2.52* | 2.46* | 2.36* | 2.22* | 2.03* | 1.80* | 1.54* | 1.36* |
| 500 | 49000 | 1.000 | 2.01* | 1.89* | 1.79* | 1.69* | 1.63* | 1.49* | 1.36* | 1.23 |
| 1000 | 49000 | 1.000 | 1.70* | 1.67* | 1.63* | 1.55* | 1.44* | 1.32* | 1.26* | 1.16 |
| 1650 | 49000 | 1.000 | 1.36* | 1.40* | 1.39* | 1.41* | 1.33* | 1.31* | 1.22 | 1.14 |

### sampling:student / upper / stress-

| n | R | maxD | 0.00357 | 0.00417 | 0.005 | 0.00625 | 0.00833 | 0.0125 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 49000 | 1.000 | 0.07 | 0.07 | 0.06 | 0.08 | 0.08 | 0.13 | 0.19 | 0.37 |
| 30 | 49000 | 1.000 | 0.07 | 0.07 | 0.08 | 0.13 | 0.13 | 0.18 | 0.27 | 0.45 |
| 50 | 49000 | 1.000 | 0.19 | 0.17 | 0.16 | 0.17 | 0.18 | 0.21 | 0.36 | 0.53 |
| 100 | 49000 | 1.000 | 0.19 | 0.22 | 0.30 | 0.31 | 0.36 | 0.43 | 0.52 | 0.68 |
| 200 | 49000 | 1.000 | 0.44 | 0.43 | 0.42 | 0.49 | 0.52 | 0.60 | 0.69 | 0.80 |
| 500 | 49000 | 1.000 | 0.59 | 0.61 | 0.65 | 0.67 | 0.70 | 0.68 | 0.74 | 0.85 |
| 1000 | 49000 | 1.000 | 0.70 | 0.71 | 0.69 | 0.71 | 0.73 | 0.76 | 0.81 | 0.87 |
| 1650 | 49000 | 1.000 | 0.83 | 0.85 | 0.82 | 0.83 | 0.84 | 0.85 | 0.89 | 0.94 |

### tost:gate5 / ratio_hi / rho=0 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |
| 30 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 50 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.15 |
| 100 | 25000 | 1.000 | 0.10 | 0.14 | 0.17 | 0.27 | 0.49 | 0.79 | 0.96 |
| 170 | 25000 | 1.000 | 1.09 | 1.11 | 1.08 | 1.11 | 1.06 | 1.04 | 1.05 |

### tost:gate5 / ratio_hi / rho=0 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.12 | 0.11 | 0.11 | 0.13 | 0.16 | 0.26 | 0.53 |
| 30 | 25000 | 1.000 | 0.25 | 0.28 | 0.33 | 0.41 | 0.51 | 0.68 | 0.95 |
| 50 | 25000 | 1.000 | 1.13 | 1.12 | 1.19 | 1.20 | 1.29* | 1.37* | 1.44* |
| 100 | 25000 | 1.000 | 2.91* | 2.81* | 2.70* | 2.51* | 2.25* | 1.99* | 1.58* |
| 170 | 25000 | 1.000 | 2.41* | 2.29* | 2.11* | 1.94* | 1.74* | 1.57* | 1.36* |

### tost:gate5 / ratio_hi / rho=0 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 30 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 50 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |
| 100 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.05 | 0.44 |
| 170 | 25000 | 1.000 | 0.12 | 0.20 | 0.26 | 0.37 | 0.50 | 0.58 | 0.73 |

### tost:gate5 / ratio_hi / rho=0.5 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.03 | 0.04 | 0.13 |
| 30 | 25000 | 1.000 | 0.04 | 0.06 | 0.06 | 0.09 | 0.10 | 0.18 | 0.44 |
| 50 | 25000 | 1.000 | 0.21 | 0.25 | 0.32 | 0.39 | 0.54 | 0.75 | 0.98 |
| 100 | 25000 | 1.000 | 1.02 | 0.99 | 1.02 | 1.04 | 1.07 | 1.09 | 1.07 |
| 170 | 25000 | 1.000 | 1.01 | 0.98 | 1.00 | 1.03 | 1.07 | 1.07 | 1.05 |

### tost:gate5 / ratio_hi / rho=0.5 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 2.06* | 2.12* | 2.21* | 2.30* | 2.47* | 2.57* | 2.25* |
| 30 | 25000 | 1.000 | 4.08* | 3.97* | 3.79* | 3.54* | 3.23* | 2.78* | 2.11* |
| 50 | 25000 | 1.000 | 4.31* | 4.01* | 3.65* | 3.31* | 2.93* | 2.47* | 1.84* |
| 100 | 25000 | 1.000 | 2.98* | 2.79* | 2.61* | 2.43* | 2.24* | 1.98* | 1.59* |
| 170 | 25000 | 1.000 | 2.49* | 2.34* | 2.18* | 2.08* | 1.88* | 1.67* | 1.44* |

### tost:gate5 / ratio_hi / rho=0.5 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 30 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.03 |
| 50 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.02 | 0.05 | 0.29 |
| 100 | 25000 | 1.000 | 0.17 | 0.20 | 0.26 | 0.31 | 0.38 | 0.53 | 0.69 |
| 170 | 25000 | 1.000 | 0.46 | 0.49 | 0.48 | 0.51 | 0.54 | 0.61 | 0.76 |

### tost:gate5 / ratio_hi / rho=0.9 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.94 | 0.95 | 0.91 | 0.92 | 0.93 | 0.97 | 1.02 |
| 30 | 25000 | 1.000 | 1.01 | 1.00 | 1.05 | 1.07 | 1.03 | 0.99 | 1.02 |
| 50 | 25000 | 1.000 | 1.08 | 1.07 | 1.06 | 1.09 | 1.07 | 1.08 | 1.04 |
| 100 | 25000 | 1.000 | 1.21 | 1.14 | 1.10 | 1.10 | 1.05 | 1.02 | 1.03 |
| 170 | 25000 | 1.000 | 1.13 | 1.09 | 1.09 | 1.10 | 1.11 | 1.08 | 1.03 |

### tost:gate5 / ratio_hi / rho=0.9 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 6.57* | 6.16* | 5.69* | 5.06* | 4.49* | 3.67* | 2.60* |
| 30 | 25000 | 1.000 | 6.28* | 5.82* | 5.32* | 4.68* | 3.94* | 3.23* | 2.24* |
| 50 | 25000 | 1.000 | 4.41* | 4.03* | 3.69* | 3.37* | 2.95* | 2.48* | 1.85* |
| 100 | 25000 | 1.000 | 3.04* | 2.86* | 2.67* | 2.45* | 2.19* | 1.91* | 1.54* |
| 170 | 25000 | 1.000 | 2.64* | 2.48* | 2.28* | 2.12* | 1.90* | 1.76* | 1.48* |

### tost:gate5 / ratio_hi / rho=0.9 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.03 | 0.02 | 0.03 | 0.04 | 0.05 | 0.10 | 0.32 |
| 30 | 25000 | 1.000 | 0.05 | 0.06 | 0.08 | 0.11 | 0.20 | 0.30 | 0.43 |
| 50 | 25000 | 1.000 | 0.15 | 0.17 | 0.23 | 0.26 | 0.29 | 0.39 | 0.56 |
| 100 | 25000 | 1.000 | 0.35 | 0.36 | 0.39 | 0.44 | 0.45 | 0.55 | 0.69 |
| 170 | 25000 | 1.000 | 0.40 | 0.47 | 0.50 | 0.54 | 0.60 | 0.65 | 0.74 |

### tost:gate5 / ratio_lo / rho=0 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |
| 30 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |
| 50 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 | 0.17 |
| 100 | 25000 | 1.000 | 0.09 | 0.13 | 0.22 | 0.33 | 0.51 | 0.79 | 1.02 |
| 170 | 25000 | 1.000 | 1.00 | 0.99 | 1.00 | 1.01 | 1.01 | 1.03 | 1.03 |

### tost:gate5 / ratio_lo / rho=0 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.01 | 0.01 |
| 30 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 50 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |
| 100 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.08 | 0.47 |
| 170 | 25000 | 1.000 | 0.14 | 0.19 | 0.29 | 0.37 | 0.49 | 0.60 | 0.75 |

### tost:gate5 / ratio_lo / rho=0 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.07 | 0.07 | 0.06 | 0.09 | 0.12 | 0.23 | 0.59 |
| 30 | 25000 | 1.000 | 0.18 | 0.24 | 0.30 | 0.44 | 0.56 | 0.76 | 0.98 |
| 50 | 25000 | 1.000 | 1.10 | 1.15 | 1.16 | 1.21 | 1.25* | 1.40* | 1.48* |
| 100 | 25000 | 1.000 | 2.61* | 2.52* | 2.46* | 2.33* | 2.22* | 1.95* | 1.55* |
| 170 | 25000 | 1.000 | 2.18* | 2.03* | 1.98* | 1.90* | 1.82* | 1.66* | 1.40* |

### tost:gate5 / ratio_lo / rho=0.5 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.02 | 0.03 | 0.03 | 0.02 | 0.03 | 0.05 | 0.13 |
| 30 | 25000 | 1.000 | 0.03 | 0.02 | 0.03 | 0.05 | 0.06 | 0.15 | 0.45 |
| 50 | 25000 | 1.000 | 0.17 | 0.23 | 0.29 | 0.41 | 0.58 | 0.78 | 1.00 |
| 100 | 25000 | 1.000 | 1.04 | 1.08 | 1.08 | 1.05 | 1.05 | 1.06 | 1.04 |
| 170 | 25000 | 1.000 | 1.12 | 1.12 | 1.06 | 1.04 | 1.08 | 1.03 | 1.06 |

### tost:gate5 / ratio_lo / rho=0.5 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.02 | 0.02 | 0.02 | 0.03 | 0.02 | 0.02 | 0.01 |
| 30 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.03 |
| 50 | 25000 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.03 | 0.25 |
| 100 | 25000 | 1.000 | 0.12 | 0.16 | 0.26 | 0.35 | 0.44 | 0.54 | 0.67 |
| 170 | 25000 | 1.000 | 0.50 | 0.51 | 0.50 | 0.52 | 0.60 | 0.67 | 0.78 |

### tost:gate5 / ratio_lo / rho=0.5 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 1.95* | 2.03* | 2.14* | 2.33* | 2.48* | 2.50* | 2.22* |
| 30 | 25000 | 1.000 | 4.18* | 4.07* | 3.85* | 3.69* | 3.33* | 2.82* | 2.13* |
| 50 | 25000 | 1.000 | 4.08* | 3.81* | 3.55* | 3.21* | 2.86* | 2.38* | 1.79* |
| 100 | 25000 | 1.000 | 3.02* | 2.87* | 2.72* | 2.56* | 2.23* | 1.93* | 1.59* |
| 170 | 25000 | 1.000 | 2.44* | 2.30* | 2.17* | 1.99* | 1.81* | 1.56* | 1.39* |

### tost:gate5 / ratio_lo / rho=0.9 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.93 | 0.90 | 0.86 | 0.97 | 1.01 | 1.02 | 1.07 |
| 30 | 25000 | 1.000 | 1.09 | 1.11 | 1.13 | 1.13 | 1.08 | 1.04 | 1.01 |
| 50 | 25000 | 1.000 | 1.08 | 1.05 | 1.01 | 0.98 | 0.98 | 0.97 | 1.01 |
| 100 | 25000 | 1.000 | 1.19 | 1.20 | 1.18 | 1.16 | 1.14 | 1.06 | 1.04 |
| 170 | 25000 | 1.000 | 1.11 | 1.15 | 1.11 | 1.10 | 1.09 | 1.10 | 1.07 |

### tost:gate5 / ratio_lo / rho=0.9 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 0.02 | 0.02 | 0.03 | 0.04 | 0.06 | 0.13 | 0.33 |
| 30 | 25000 | 1.000 | 0.07 | 0.11 | 0.13 | 0.16 | 0.20 | 0.29 | 0.45 |
| 50 | 25000 | 1.000 | 0.16 | 0.19 | 0.20 | 0.27 | 0.30 | 0.38 | 0.56 |
| 100 | 25000 | 1.000 | 0.31 | 0.34 | 0.37 | 0.40 | 0.44 | 0.52 | 0.67 |
| 170 | 25000 | 1.000 | 0.50 | 0.48 | 0.50 | 0.54 | 0.56 | 0.59 | 0.71 |

### tost:gate5 / ratio_lo / rho=0.9 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 25000 | 1.000 | 6.51* | 6.13* | 5.67* | 5.22* | 4.65* | 3.89* | 2.72* |
| 30 | 25000 | 1.000 | 5.82* | 5.40* | 4.95* | 4.38* | 3.76* | 3.05* | 2.19* |
| 50 | 25000 | 1.000 | 4.86* | 4.49* | 4.07* | 3.62* | 3.08* | 2.56* | 1.88* |
| 100 | 25000 | 1.000 | 2.79* | 2.61* | 2.47* | 2.31* | 2.11* | 1.88* | 1.52* |
| 170 | 25000 | 1.000 | 2.24* | 2.20* | 2.14* | 1.95* | 1.84* | 1.70* | 1.44* |

### tost:gate5/boott / ratio_hi / rho=0 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |
| 50 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.20 |
| 100 | 24500 | 1.000 | 0.12 | 0.16 | 0.24 | 0.36 | 0.58 | 0.83 | 1.04 |
| 170 | 24500 | 1.000 | 1.10 | 1.17 | 1.18 | 1.15 | 1.16 | 1.13 | 1.08 |

### tost:gate5/boott / ratio_hi / rho=0 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 0.07 | 0.08 | 0.09 | 0.11 | 0.15 | 0.23 | 0.52 |
| 30 | 24500 | 1.000 | 0.26 | 0.31 | 0.35 | 0.38 | 0.56 | 0.69 | 0.93 |
| 50 | 24500 | 1.000 | 0.99 | 1.03 | 1.12 | 1.14 | 1.11 | 1.09 | 1.05 |
| 100 | 24500 | 1.000 | 1.54* | 1.48* | 1.46* | 1.35* | 1.32* | 1.22 | 1.09 |
| 170 | 24500 | 1.000 | 1.28* | 1.31* | 1.25* | 1.17 | 1.12 | 1.09 | 1.05 |

### tost:gate5/boott / ratio_hi / rho=0 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 50 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.02 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.02 | 0.09 | 0.68 |
| 170 | 24500 | 1.000 | 0.19 | 0.34 | 0.50 | 0.63 | 0.80 | 0.90 | 0.99 |

### tost:gate5/boott / ratio_hi / rho=0.5 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.05 | 0.13 |
| 30 | 24500 | 1.000 | 0.04 | 0.06 | 0.06 | 0.06 | 0.09 | 0.13 | 0.44 |
| 50 | 24500 | 1.000 | 0.23 | 0.27 | 0.35 | 0.41 | 0.52 | 0.78 | 0.99 |
| 100 | 24500 | 1.000 | 1.02 | 1.02 | 1.07 | 1.07 | 1.10 | 1.09 | 1.02 |
| 170 | 24500 | 1.000 | 1.19 | 1.21 | 1.24 | 1.17 | 1.15 | 1.09 | 1.09 |

### tost:gate5/boott / ratio_hi / rho=0.5 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 1.75* | 2.00* | 2.09* | 2.14* | 2.36* | 2.45* | 2.24* |
| 30 | 24500 | 1.000 | 4.25* | 4.12* | 3.97* | 3.67* | 3.33* | 2.77* | 1.90* |
| 50 | 24500 | 1.000 | 3.66* | 3.38* | 3.07* | 2.70* | 2.31* | 1.84* | 1.36* |
| 100 | 24500 | 1.000 | 1.71* | 1.62* | 1.50* | 1.43* | 1.30* | 1.21 | 1.06 |
| 170 | 24500 | 1.000 | 1.19 | 1.22 | 1.22 | 1.19 | 1.14 | 1.12 | 1.05 |

### tost:gate5/boott / ratio_hi / rho=0.5 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.03 |
| 50 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.03 | 0.08 | 0.40 |
| 100 | 24500 | 1.000 | 0.36 | 0.50 | 0.58 | 0.68 | 0.86 | 0.94 | 1.02 |
| 170 | 24500 | 1.000 | 1.10 | 1.14 | 1.09 | 1.08 | 1.09 | 1.10 | 1.01 |

### tost:gate5/boott / ratio_hi / rho=0.9 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 1.08 | 1.14 | 1.18 | 1.15 | 1.12 | 1.12 | 1.03 |
| 30 | 24500 | 1.000 | 1.16 | 1.13 | 1.14 | 1.11 | 1.12 | 1.09 | 1.09 |
| 50 | 24500 | 1.000 | 1.18 | 1.14 | 1.12 | 1.10 | 1.11 | 1.07 | 1.03 |
| 100 | 24500 | 1.000 | 1.13 | 1.14 | 1.12 | 1.10 | 1.08 | 1.04 | 1.02 |
| 170 | 24500 | 1.000 | 1.13 | 1.13 | 1.13 | 1.07 | 1.05 | 1.04 | 1.03 |

### tost:gate5/boott / ratio_hi / rho=0.9 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 7.00* | 6.60* | 6.09* | 5.51* | 4.81* | 3.94* | 2.65* |
| 30 | 24500 | 1.000 | 6.39* | 5.95* | 5.30* | 4.55* | 3.85* | 3.01* | 1.98* |
| 50 | 24500 | 1.000 | 3.47* | 3.21* | 2.94* | 2.60* | 2.24* | 1.85* | 1.35* |
| 100 | 24500 | 1.000 | 1.65* | 1.60* | 1.52* | 1.42* | 1.30* | 1.24 | 1.11 |
| 170 | 24500 | 1.000 | 1.35* | 1.31* | 1.27* | 1.23 | 1.14 | 1.09 | 1.05 |

### tost:gate5/boott / ratio_hi / rho=0.9 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 0.06 | 0.08 | 0.08 | 0.09 | 0.14 | 0.24 | 0.69 |
| 30 | 24500 | 1.000 | 0.23 | 0.28 | 0.37 | 0.50 | 0.68 | 0.85 | 0.99 |
| 50 | 24500 | 1.000 | 0.88 | 0.86 | 0.88 | 0.89 | 0.93 | 0.97 | 1.00 |
| 100 | 24500 | 1.000 | 0.75 | 0.77 | 0.77 | 0.79 | 0.86 | 0.91 | 0.97 |
| 170 | 24500 | 1.000 | 1.05 | 1.07 | 1.08 | 1.07 | 1.05 | 1.05 | 1.03 |

### tost:gate5/boott / ratio_lo / rho=0 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |
| 50 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.03 | 0.20 |
| 100 | 24500 | 1.000 | 0.10 | 0.14 | 0.18 | 0.30 | 0.53 | 0.76 | 0.99 |
| 170 | 24500 | 1.000 | 1.15 | 1.19 | 1.15 | 1.10 | 1.11 | 1.06 | 1.01 |

### tost:gate5/boott / ratio_lo / rho=0 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 |
| 50 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 100 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.09 | 0.65 |
| 170 | 24500 | 1.000 | 0.28 | 0.39 | 0.48 | 0.67 | 0.80 | 0.92 | 0.99 |

### tost:gate5/boott / ratio_lo / rho=0 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 0.06 | 0.07 | 0.06 | 0.07 | 0.12 | 0.24 | 0.54 |
| 30 | 24500 | 1.000 | 0.32 | 0.35 | 0.41 | 0.47 | 0.59 | 0.75 | 0.94 |
| 50 | 24500 | 1.000 | 0.90 | 0.93 | 1.05 | 1.04 | 1.03 | 1.00 | 1.00 |
| 100 | 24500 | 1.000 | 1.47* | 1.40* | 1.32* | 1.25 | 1.21 | 1.15 | 1.06 |
| 170 | 24500 | 1.000 | 1.29* | 1.27* | 1.20 | 1.14 | 1.10 | 1.14 | 1.10 |

### tost:gate5/boott / ratio_lo / rho=0.5 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.13 |
| 30 | 24500 | 1.000 | 0.04 | 0.04 | 0.05 | 0.06 | 0.09 | 0.14 | 0.44 |
| 50 | 24500 | 1.000 | 0.19 | 0.20 | 0.33 | 0.47 | 0.66 | 0.86 | 1.07 |
| 100 | 24500 | 1.000 | 1.25* | 1.25* | 1.19 | 1.13 | 1.08 | 1.10 | 1.07 |
| 170 | 24500 | 1.000 | 1.01 | 1.03 | 1.04 | 1.03 | 1.06 | 1.07 | 1.02 |

### tost:gate5/boott / ratio_lo / rho=0.5 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| 30 | 24500 | 1.000 | 0.02 | 0.01 | 0.02 | 0.02 | 0.01 | 0.02 | 0.04 |
| 50 | 24500 | 1.000 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.05 | 0.43 |
| 100 | 24500 | 1.000 | 0.30 | 0.42 | 0.55 | 0.71 | 0.83 | 0.91 | 1.03 |
| 170 | 24500 | 1.000 | 1.08 | 1.05 | 1.06 | 1.05 | 1.01 | 0.99 | 1.02 |

### tost:gate5/boott / ratio_lo / rho=0.5 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 1.86* | 1.96* | 2.01* | 2.08* | 2.23* | 2.33* | 2.10* |
| 30 | 24500 | 1.000 | 3.83* | 3.80* | 3.73* | 3.48* | 3.19* | 2.62* | 1.81* |
| 50 | 24500 | 1.000 | 3.60* | 3.32* | 3.00* | 2.64* | 2.26* | 1.84* | 1.36* |
| 100 | 24500 | 1.000 | 1.74* | 1.64* | 1.55* | 1.44* | 1.32* | 1.21 | 1.09 |
| 170 | 24500 | 1.000 | 1.26* | 1.27* | 1.23 | 1.18 | 1.14 | 1.04 | 1.00 |

### tost:gate5/boott / ratio_lo / rho=0.9 / normal

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 1.07 | 1.10 | 1.13 | 1.15 | 1.18 | 1.11 | 1.07 |
| 30 | 24500 | 1.000 | 1.22 | 1.20 | 1.18 | 1.17 | 1.11 | 1.09 | 1.05 |
| 50 | 24500 | 1.000 | 1.21 | 1.21 | 1.15 | 1.10 | 1.11 | 1.10 | 1.08 |
| 100 | 24500 | 1.000 | 1.02 | 1.03 | 1.09 | 1.11 | 1.11 | 1.11 | 1.08 |
| 170 | 24500 | 1.000 | 1.04 | 1.06 | 1.03 | 1.03 | 1.04 | 1.04 | 1.07 |

### tost:gate5/boott / ratio_lo / rho=0.9 / stress

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 0.09 | 0.08 | 0.09 | 0.08 | 0.12 | 0.22 | 0.71 |
| 30 | 24500 | 1.000 | 0.23 | 0.30 | 0.37 | 0.49 | 0.69 | 0.81 | 0.94 |
| 50 | 24500 | 1.000 | 0.76 | 0.81 | 0.84 | 0.86 | 0.89 | 0.89 | 0.93 |
| 100 | 24500 | 1.000 | 0.92 | 0.95 | 0.97 | 0.94 | 0.92 | 0.97 | 1.00 |
| 170 | 24500 | 1.000 | 1.11 | 1.15 | 1.15 | 1.14 | 1.13 | 1.10 | 1.06 |

### tost:gate5/boott / ratio_lo / rho=0.9 / stress-

| n | R | maxD | 0.00714 | 0.00833 | 0.01 | 0.0125 | 0.0167 | 0.025 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 24500 | 1.000 | 7.67* | 7.19* | 6.61* | 5.84* | 5.01* | 4.03* | 2.75* |
| 30 | 24500 | 1.000 | 6.28* | 5.84* | 5.29* | 4.64* | 3.89* | 3.02* | 1.96* |
| 50 | 24500 | 1.000 | 3.75* | 3.50* | 3.11* | 2.71* | 2.33* | 1.91* | 1.36* |
| 100 | 24500 | 1.000 | 1.80* | 1.73* | 1.61* | 1.48* | 1.35* | 1.23 | 1.07 |
| 170 | 24500 | 1.000 | 1.25* | 1.30* | 1.26* | 1.19 | 1.17 | 1.11 | 1.04 |

## Unresolved bootstrap tails

none: every bootstrap tail was resolved by its B

