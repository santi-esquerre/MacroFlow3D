# SF-33 N6b — preconditioner gate (item 5) re-evaluated on V100 with the coarse-corrected solver

Run: job `sf33b-proto` (V100 GPU 0, 2026-10-07T00:36:36Z-00:52:18Z), mirror synced from `bfe4efb` (N0..N7c + C3),
driver defaults of C3 — P-A + Galerkin coarse correction (`--coarse mult --coarse-profiles 2 --coarse-assembly
colored --coarse-factor banded`, K = 2 P N^2 = 4 N^2), Eisenstat-Walker forcing (`--forcing ew`), pseudo-transient
continuation (`--psitc on`), GMRES `--restart 100 --max-inner 6000`, restart-stagnation factor 0.9
(`--gmres-stagnation-factor 0.9`, unchanged rule), `--max-newton 120 --bisect 4 --lin-tol 1e-12 --newton-tol 1e-13`.
No option was tuned. Sources: `logs/proto_b/*.log`, `raw/proto_b/*.json`, `raw/proto_b/digest_b.txt`
(`digest_newton.py`), `raw/proto_b/compare_proto_b.md`; per-stage coarse statistics parsed from the `COARSE build`
/ `COARSE apply` / `LINEAR` / `STAGE` / `STAGE_END` lines of the committed logs (recipe at the end).

## Gate verdict rule (verbatim, orchestrator)

PASS iff every 9a case (incl. the two new 32^3 eps 0.5 points) ends `STATUS converged` with `r_F <= 1e-10`, no
accepted Newton step has an unconverged linear solve (by construction), and no 9a case needed a bisection caused by
`linear_failure` where the prototype's PATH has none (a PATH that differs only by stage success at the same
amplitudes is noted, not a FAIL); metric agreement as in N6 (1e-6 vs full precision at 16/24, 4-digit bound at 32
where a reference exists). generic3d eps 1 results are reported as informative (outside 9a; known limitation from
N7c). Any FAIL: report with the curves (`digest_b.txt`); do not tune.

## Per-criterion evidence

| criterion | result | evidence |
|---|---|---|
| every 9a case `STATUS converged`, `r_F <= 1e-10` | **FAIL** (14/16) | the 14 N6 cases converge (r_F <= 9.6e-14); `gauss:0.5:32` and `gauss_ch:0.5:32` end `continuation_floor` (r_F 1.41e-2 / 9.66e-3 at stop; last accepted amplitude 0.421875 / 0.4375) |
| no accepted Newton step with an unconverged linear solve | PASS | 949 accepted Newton steps over all 21 logs, each preceded by a `LINEAR ... status=converged` solve; 0 exceptions |
| no `linear_failure` bisection where the prototype PATH has none | PASS for the 14 N6 cases (0 bisections, 0 linear failures); **FAIL** for the two 32^3 eps 0.5 points (4 bisections each, all caused by `linear_failure`; no prototype PATH exists there) | PATHs in `compare_proto_b.md`; per-stage table below |
| metric agreement (1e-6 full precision 16/24; 4-digit bound 32) | PASS for the 14 N6 cases (max rel diff 1.75e-12 full precision; 32^3 eps 0.25 `PASS(4dig)`); 32^3 eps 0.5: no prototype reference, ceiling sanity PASS (4.7e-15 / 2.8e-15) | `proto_comparison_b.md` |

## VERDICT: FAIL

The coarse-corrected solver passes the gate on the whole N6 matrix (16^3-24^3 at eps 0.25 and 0.5, 32^3 at eps
0.25: zero linear failures, zero bisections, GMRES its per accepted step max 500 / median <= 17), which N6 failed.
It **fails at the new point 32^3, eps 0.5** for both fields: from the 0.375 -> 0.5 continuation onward, the GMRES
solves with P-A + CC(mult, 2) stagnate (true relative residual 0.24-0.81 after 300-1600 iterations, against
eta <= 0.1; one solve ran to the 6000 cap at 1.6e-4 against eta 6.5e-5) at mid-stage Newton iterates (step
9-19, mu ~ 2e-4 .. 1e-2 for the stagnations, r_F ~ 1e-4 .. 6e-2 at stage stop); the continuation
bisects four times and stops at `continuation_floor`. Not tuned (no larger restart / cap, no other stagnation
factor), per the rule.

Observed growth with N at eps 0.5 (accepted-step GMRES its max / median, stage 0.5 or the hardest accepted
stage): 16^3 74 / 26, 24^3 400-500 / 55-74, 32^3 stalls (accepted intermediate stages need up to 2800 / 5800
iterations). Consistent with the N7c observation (generic3d eps 1 at 16^3: slow modes outside the x1-constant /
x1-linear column space of the coarse correction), now visible on the gauss fields at 32^3, eps 0.5.

## GMRES iterations per accepted Newton step (all stages), per case

| case | status | accepted-step its max | median | all solves: max / total its | linear failures |
|---|---|---|---|---|---|
| gauss_0.25_16 | converged | 30 | 6.5 | 30 / 107 | 0 |
| gauss_0.25_24 | converged | 49 | 8.5 | 49 / 210 | 0 |
| gauss_0.25_32 | converged | 77 | 8.5 | 77 / 370 | 0 |
| gauss_0.5_16 | converged | 74 | 10 | 74 / 496 | 0 |
| gauss_0.5_24 | converged | 400 | 16 | 400 / 1551 | 0 |
| gauss_0.5_32 | continuation_floor | 2800 | 51.5 | 2800 / 24667 | 6 |
| gauss_ch_0.25_16 | converged | 24 | 6 | 24 / 114 | 0 |
| gauss_ch_0.25_24 | converged | 45 | 7 | 45 / 209 | 0 |
| gauss_ch_0.25_32 | converged | 86 | 8.5 | 86 / 346 | 0 |
| gauss_ch_0.5_16 | converged | 74 | 11 | 74 / 528 | 0 |
| gauss_ch_0.5_24 | converged | 500 | 17 | 500 / 1941 | 0 |
| gauss_ch_0.5_32 | continuation_floor | 5800 | 40 | 6000 / 34547 | 7 |
| control2d_0.25_16 | converged | 9 | 3.5 | 9 / 39 | 0 |
| control2d_0.25_24 | converged | 7 | 3 | 7 / 44 | 0 |
| control2d_0.5_16 | converged | 11 | 4 | 11 / 93 | 0 |
| control2d_0.5_24 | converged | 10 | 3 | 10 / 102 | 0 |
| generic3d_0.25_16 (informative) | converged | 24 | 8 | 24 / 112 | 0 |
| generic3d_0.5_16 (informative) | converged | 77 | 10 | 77 / 442 | 0 |
| generic3d_1_16 (informative) | continuation_floor | 4700 | 47 | 4700 / 26679 | 4 |
| generic3d_1_20 (informative) | continuation_floor | 1800 | 30 | 1800 / 18894 | 6 |
| generic3d_1_24 (informative) | continuation_floor | 1700 | 31 | 1700 / 19634 | 6 |

N6 (P-A only, defaults) for comparison: linear failures 36 in 11/19 cases; accepted-step GMRES its max / median
5600 / 166 (16^3), 5900 / 300 (24^3), 1900 / 357 (32^3, eps 0.25 only). N6b on the same 19 cases: linear failures
16, all in the three generic3d eps 1 cases (informative); the 14 9a cases: 0.

## Per case and stage: linear failures, GMRES its per Newton step, coarse K, build / apply times

Restart 100, cap 6000, stagnation factor 0.9 for every stage (no per-case override). `builds` = coarse
factorizations (one per Newton step: the coarse operator is rebuilt with P-A at every factor). Times: host wall
time per build (`t_assembly`: colored Galerkin assembly, `t_lu`: banded host LU), mean over the stage; `t_apply`:
mean time per coarse-correction application (`t_apply_avg` of `COARSE apply`). `rcond_est`: minimum over the stage
of the 1-norm reciprocal condition estimate of the coarse matrix.

| case | stage (from) | stage status | Newton steps | linear failures (status) | GMRES its max / median per step | coarse K | builds | t_assembly mean [s] | t_lu mean [s] | t_apply mean [ms] | rcond_est min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gauss_0.25_16 | 0.25 (0) | converged (accepted) | 12 | 0 | 30 / 6.5 | 1024 | 12 | 0.015 | 0.033 | 0.69 | 9.8e-05 |
| gauss_0.25_24 | 0.25 (0) | converged (accepted) | 16 | 0 | 49 / 8.5 | 2304 | 16 | 0.010 | 0.181 | 2.49 | 3.3e-05 |
| gauss_0.25_32 | 0.25 (0) | converged (accepted) | 18 | 0 | 77 / 8.5 | 4096 | 18 | 0.027 | 0.619 | 7.14 | 1.6e-05 |
| gauss_0.5_16 | 0.25 (0) | converged (accepted) | 12 | 0 | 30 / 6.5 | 1024 | 12 | 0.015 | 0.033 | 0.70 | 9.8e-05 |
| gauss_0.5_16 | 0.5 (0.25) | converged (accepted) | 13 | 0 | 74 / 26 | 1024 | 13 | 0.014 | 0.034 | 0.66 | 2.9e-05 |
| gauss_0.5_24 | 0.25 (0) | converged (accepted) | 16 | 0 | 49 / 8.5 | 2304 | 16 | 0.010 | 0.191 | 2.95 | 3.3e-05 |
| gauss_0.5_24 | 0.5 (0.25) | converged (accepted) | 16 | 0 | 400 / 55 | 2304 | 16 | 0.010 | 0.188 | 2.43 | 7.4e-07 |
| gauss_0.5_32 | 0.25 (0) | converged (accepted) | 18 | 0 | 77 / 8.5 | 4096 | 18 | 0.026 | 0.557 | 6.04 | 1.6e-05 |
| gauss_0.5_32 | 0.5 (0.25) | linear_failure (failed) | 12 | 1 (stagnation) | 700 / 38.5 | 4096 | 12 | 0.026 | 0.557 | 6.04 | 3.4e-09 |
| gauss_0.5_32 | 0.375 (0.25) | converged (accepted) | 18 | 0 | 396 / 55 | 4096 | 18 | 0.026 | 0.570 | 6.03 | 6.0e-06 |
| gauss_0.5_32 | 0.5 (0.375) | linear_failure (failed) | 11 | 1 (stagnation) | 1100 / 51 | 4096 | 11 | 0.025 | 0.560 | 6.00 | 4.7e-07 |
| gauss_0.5_32 | 0.4375 (0.375) | linear_failure (failed) | 15 | 1 (stagnation) | 1000 / 91 | 4096 | 15 | 0.025 | 0.570 | 5.99 | 1.3e-07 |
| gauss_0.5_32 | 0.40625 (0.375) | converged (accepted) | 18 | 0 | 1100 / 97 | 4096 | 18 | 0.025 | 0.577 | 6.01 | 5.1e-07 |
| gauss_0.5_32 | 0.4375 (0.40625) | linear_failure (failed) | 15 | 1 (stagnation) | 1000 / 100 | 4096 | 15 | 0.026 | 0.571 | 6.01 | 4.0e-07 |
| gauss_0.5_32 | 0.421875 (0.40625) | converged (accepted) | 18 | 0 | 2800 / 100 | 4096 | 18 | 0.025 | 0.577 | 6.00 | 1.7e-07 |
| gauss_0.5_32 | 0.4375 (0.421875) | linear_failure (failed) | 14 | 1 (stagnation) | 800 / 93 | 4096 | 14 | 0.026 | 0.570 | 6.01 | 4.7e-07 |
| gauss_0.5_32 | 0.5 final (0.421875) | linear_failure (failed) | 9 | 1 (stagnation) | 400 / 26 | 4096 | 9 | 0.024 | 0.555 | 5.96 | 1.6e-06 |
| gauss_ch_0.25_16 | 0.25 (0) | converged (accepted) | 13 | 0 | 24 / 6 | 1024 | 13 | 0.015 | 0.034 | 0.68 | 9.2e-05 |
| gauss_ch_0.25_24 | 0.25 (0) | converged (accepted) | 16 | 0 | 45 / 7 | 2304 | 16 | 0.010 | 0.176 | 2.33 | 3.5e-05 |
| gauss_ch_0.25_32 | 0.25 (0) | converged (accepted) | 18 | 0 | 86 / 8.5 | 4096 | 18 | 0.027 | 0.594 | 6.30 | 1.9e-05 |
| gauss_ch_0.5_16 | 0.25 (0) | converged (accepted) | 13 | 0 | 24 / 6 | 1024 | 13 | 0.015 | 0.027 | 0.64 | 9.2e-05 |
| gauss_ch_0.5_16 | 0.5 (0.25) | converged (accepted) | 14 | 0 | 74 / 26 | 1024 | 14 | 0.015 | 0.035 | 0.70 | 1.4e-05 |
| gauss_ch_0.5_24 | 0.25 (0) | converged (accepted) | 16 | 0 | 45 / 7 | 2304 | 16 | 0.010 | 0.188 | 2.31 | 3.5e-05 |
| gauss_ch_0.5_24 | 0.5 (0.25) | converged (accepted) | 17 | 0 | 500 / 74 | 2304 | 17 | 0.009 | 0.189 | 2.12 | 6.0e-08 |
| gauss_ch_0.5_32 | 0.25 (0) | converged (accepted) | 18 | 0 | 86 / 8.5 | 4096 | 18 | 0.027 | 0.558 | 6.00 | 1.9e-05 |
| gauss_ch_0.5_32 | 0.5 (0.25) | linear_failure (failed) | 9 | 1 (stagnation) | 400 / 9 | 4096 | 9 | 0.026 | 0.544 | 5.99 | 5.0e-06 |
| gauss_ch_0.5_32 | 0.375 (0.25) | converged (accepted) | 19 | 0 | 400 / 47 | 4096 | 19 | 0.026 | 0.577 | 5.98 | 8.5e-07 |
| gauss_ch_0.5_32 | 0.5 (0.375) | linear_failure (failed) | 12 | 1 (stagnation) | 1600 / 67.5 | 4096 | 12 | 0.024 | 0.569 | 5.93 | 4.0e-07 |
| gauss_ch_0.5_32 | 0.4375 (0.375) | linear_failure (accepted) | 19 | 1 (max_iterations) | 6000 / 100 | 4096 | 19 | 0.025 | 0.590 | 5.95 | 2.5e-07 |
| gauss_ch_0.5_32 | 0.5 (0.4375) | linear_failure (failed) | 11 | 1 (stagnation) | 700 / 73 | 4096 | 11 | 0.026 | 0.575 | 6.00 | 4.3e-08 |
| gauss_ch_0.5_32 | 0.46875 (0.4375) | linear_failure (failed) | 14 | 1 (stagnation) | 1000 / 99.5 | 4096 | 14 | 0.026 | 0.586 | 6.00 | 5.6e-07 |
| gauss_ch_0.5_32 | 0.453125 (0.4375) | linear_failure (failed) | 15 | 1 (stagnation) | 1100 / 100 | 4096 | 15 | 0.026 | 0.586 | 5.99 | 5.8e-07 |
| gauss_ch_0.5_32 | 0.5 final (0.4375) | linear_failure (failed) | 11 | 1 (stagnation) | 700 / 73 | 4096 | 11 | 0.026 | 0.575 | 5.99 | 4.3e-08 |
| control2d_0.25_16 | 0.25 (0) | converged (accepted) | 10 | 0 | 9 / 3.5 | 1024 | 10 | 0.015 | 0.025 | 0.62 | 2.0e-04 |
| control2d_0.25_24 | 0.25 (0) | converged (accepted) | 13 | 0 | 7 / 3 | 2304 | 13 | 0.010 | 0.167 | 2.12 | 7.9e-05 |
| control2d_0.5_16 | 0.25 (0) | converged (accepted) | 10 | 0 | 9 / 3.5 | 1024 | 10 | 0.015 | 0.031 | 0.68 | 2.0e-04 |
| control2d_0.5_16 | 0.5 (0.25) | converged (accepted) | 11 | 0 | 11 / 5 | 1024 | 11 | 0.014 | 0.024 | 0.60 | 1.5e-04 |
| control2d_0.5_24 | 0.25 (0) | converged (accepted) | 13 | 0 | 7 / 3 | 2304 | 13 | 0.010 | 0.184 | 2.95 | 7.9e-05 |
| control2d_0.5_24 | 0.5 (0.25) | converged (accepted) | 13 | 0 | 10 / 4 | 2304 | 13 | 0.010 | 0.173 | 2.69 | 6.4e-05 |
| generic3d_0.25_16 | 0.25 (0) | converged (accepted) | 13 | 0 | 24 / 8 | 1024 | 13 | 0.015 | 0.032 | 0.71 | 1.1e-04 |
| generic3d_0.5_16 | 0.25 (0) | converged (accepted) | 13 | 0 | 24 / 8 | 1024 | 13 | 0.015 | 0.033 | 0.71 | 1.1e-04 |
| generic3d_0.5_16 | 0.5 (0.25) | converged (accepted) | 15 | 0 | 77 / 19 | 1024 | 15 | 0.015 | 0.028 | 0.64 | 3.9e-05 |
| generic3d_1_16 | 0.25 (0) | converged (accepted) | 13 | 0 | 24 / 8 | 1024 | 13 | 0.015 | 0.027 | 0.62 | 1.1e-04 |
| generic3d_1_16 | 0.5 (0.25) | converged (accepted) | 15 | 0 | 77 / 19 | 1024 | 15 | 0.014 | 0.028 | 0.62 | 3.9e-05 |
| generic3d_1_16 | 1 (0.5) | stagnation (failed) | 8 | 0 | 100 / 36.5 | 1024 | 8 | 0.014 | 0.034 | 0.63 | 1.7e-07 |
| generic3d_1_16 | 0.75 (0.5) | converged (accepted) | 17 | 0 | 378 / 55 | 1024 | 17 | 0.014 | 0.033 | 0.63 | 4.4e-06 |
| generic3d_1_16 | 1 (0.75) | stagnation (failed) | 8 | 0 | 400 / 74.5 | 1024 | 8 | 0.014 | 0.035 | 0.63 | 6.0e-07 |
| generic3d_1_16 | 0.875 (0.75) | converged (accepted) | 21 | 0 | 4700 / 200 | 1024 | 21 | 0.014 | 0.038 | 0.66 | 5.6e-07 |
| generic3d_1_16 | 1 (0.875) | linear_failure (failed) | 7 | 1 (stagnation) | 800 / 95 | 1024 | 7 | 0.014 | 0.037 | 0.63 | 4.6e-07 |
| generic3d_1_16 | 0.9375 (0.875) | linear_failure (failed) | 9 | 1 (stagnation) | 1400 / 100 | 1024 | 9 | 0.014 | 0.039 | 0.65 | 8.4e-07 |
| generic3d_1_16 | 0.90625 (0.875) | linear_failure (failed) | 12 | 1 (stagnation) | 1000 / 200 | 1024 | 12 | 0.014 | 0.039 | 0.65 | 1.0e-06 |
| generic3d_1_16 | 1 final (0.875) | linear_failure (failed) | 7 | 1 (stagnation) | 800 / 95 | 1024 | 7 | 0.014 | 0.039 | 0.65 | 4.6e-07 |
| generic3d_1_20 | 0.25 (0) | converged (accepted) | 15 | 0 | 31 / 7 | 1600 | 15 | 0.006 | 0.087 | 1.32 | 6.1e-05 |
| generic3d_1_20 | 0.5 (0.25) | converged (accepted) | 17 | 0 | 100 / 19 | 1600 | 17 | 0.006 | 0.092 | 1.35 | 2.1e-05 |
| generic3d_1_20 | 1 (0.5) | linear_failure (failed) | 10 | 1 (stagnation) | 800 / 48 | 1600 | 10 | 0.006 | 0.096 | 1.15 | 2.4e-08 |
| generic3d_1_20 | 0.75 (0.5) | converged (accepted) | 20 | 0 | 1800 / 89.5 | 1600 | 20 | 0.006 | 0.095 | 1.22 | 8.5e-07 |
| generic3d_1_20 | 1 (0.75) | linear_failure (failed) | 7 | 1 (stagnation) | 400 / 33 | 1600 | 7 | 0.006 | 0.099 | 1.16 | 4.0e-07 |
| generic3d_1_20 | 0.875 (0.75) | linear_failure (failed) | 9 | 1 (stagnation) | 700 / 58 | 1600 | 9 | 0.006 | 0.098 | 1.24 | 2.8e-07 |
| generic3d_1_20 | 0.8125 (0.75) | linear_failure (failed) | 13 | 1 (stagnation) | 1200 / 100 | 1600 | 13 | 0.006 | 0.097 | 1.18 | 1.1e-07 |
| generic3d_1_20 | 0.78125 (0.75) | linear_failure (failed) | 15 | 1 (stagnation) | 1000 / 100 | 1600 | 15 | 0.006 | 0.100 | 1.29 | 3.6e-07 |
| generic3d_1_20 | 1 final (0.75) | linear_failure (failed) | 7 | 1 (stagnation) | 400 / 33 | 1600 | 7 | 0.006 | 0.100 | 1.27 | 4.0e-07 |
| generic3d_1_24 | 0.25 (0) | converged (accepted) | 16 | 0 | 33 / 6.5 | 2304 | 16 | 0.010 | 0.165 | 2.04 | 3.9e-05 |
| generic3d_1_24 | 0.5 (0.25) | converged (accepted) | 18 | 0 | 163 / 18 | 2304 | 18 | 0.010 | 0.172 | 2.01 | 1.2e-05 |
| generic3d_1_24 | 1 (0.5) | linear_failure (failed) | 9 | 1 (stagnation) | 700 / 21 | 2304 | 9 | 0.009 | 0.192 | 1.98 | 1.7e-07 |
| generic3d_1_24 | 0.75 (0.5) | linear_failure (failed) | 14 | 1 (stagnation) | 600 / 45 | 2304 | 14 | 0.009 | 0.185 | 2.00 | 7.2e-07 |
| generic3d_1_24 | 0.625 (0.5) | converged (accepted) | 20 | 0 | 400 / 55 | 2304 | 20 | 0.009 | 0.181 | 1.99 | 6.0e-06 |
| generic3d_1_24 | 0.75 (0.625) | linear_failure (failed) | 13 | 1 (stagnation) | 1700 / 67 | 2304 | 13 | 0.009 | 0.188 | 1.99 | 2.9e-08 |
| generic3d_1_24 | 0.6875 (0.625) | linear_failure (failed) | 17 | 1 (stagnation) | 1200 / 90 | 2304 | 17 | 0.009 | 0.189 | 1.98 | 4.0e-06 |
| generic3d_1_24 | 0.65625 (0.625) | converged (accepted) | 20 | 0 | 1000 / 94.5 | 2304 | 20 | 0.009 | 0.187 | 1.98 | 4.9e-06 |
| generic3d_1_24 | 0.6875 (0.65625) | linear_failure (failed) | 17 | 1 (stagnation) | 1300 / 100 | 2304 | 17 | 0.009 | 0.191 | 1.98 | 4.0e-06 |
| generic3d_1_24 | 1 final (0.65625) | linear_failure (failed) | 7 | 1 (stagnation) | 400 / 12 | 2304 | 7 | 0.009 | 0.193 | 2.03 | 3.9e-08 |

linear failures per case: gauss_0.25_16 0, gauss_0.25_24 0, gauss_0.25_32 0, gauss_0.5_16 0, gauss_0.5_24 0, gauss_0.5_32 6, gauss_ch_0.25_16 0, gauss_ch_0.25_24 0, gauss_ch_0.25_32 0, gauss_ch_0.5_16 0, gauss_ch_0.5_24 0, gauss_ch_0.5_32 7, control2d_0.25_16 0, control2d_0.25_24 0, control2d_0.5_16 0, control2d_0.5_24 0, generic3d_0.25_16 0, generic3d_0.5_16 0, generic3d_1_16 4, generic3d_1_20 6, generic3d_1_24 6

Notes on the table: `gauss_ch_0.5_32` stage 0.4375 (from 0.375) ends `linear_failure` but is **accepted**: the last
accepted iterate has r_F = 7.73e-10 <= `stage_ok` = 1e-9 (the prototype's intermediate-stage rule; the failed
linear solve itself was not taken as a step). The `final` rows repeat the deterministic attempt at the target from
the last accepted amplitude (identical numbers to the earlier attempt from the same start).

## Curves of every failed linear solve at 32^3, eps 0.5 (`digest_b.txt`)

Per-restart true relative residual (restart 100); stop = `stagnation` when the residual at a restart exceeds 0.9 x
the one two restarts earlier, `max_iterations` at 6000.

gauss_0.5_32:

```
stage 0.5 from 0.25      step 12: stagnation its=300  eta=1.00e-01 mu=3.23e-03 : 6.03e-01 5.57e-01 5.52e-01
stage 0.5 from 0.375     step 11: stagnation its=400  eta=1.00e-01 mu=4.58e-03 : 6.70e-01 5.91e-01 5.62e-01 5.48e-01
stage 0.4375 from 0.375  step 15: stagnation its=400  eta=7.62e-02 mu=2.16e-04 : 7.86e-01 7.43e-01 7.06e-01 6.72e-01
stage 0.4375 from 0.40625 step 15: stagnation its=300 eta=6.56e-02 mu=1.97e-04 : 8.32e-01 8.10e-01 8.08e-01
stage 0.4375 from 0.421875 step 14: stagnation its=400 eta=1.00e-01 mu=7.41e-04 : 7.95e-01 7.08e-01 6.73e-01 6.47e-01
stage 0.5 (final) from 0.421875 step 9: stagnation its=400 eta=1.00e-01 mu=8.87e-03 : 3.19e-01 2.66e-01 2.53e-01 2.43e-01
```

gauss_ch_0.5_32:

```
stage 0.5 from 0.25       step 9:  stagnation its=400  eta=1.00e-01 mu=9.82e-03 : 4.51e-01 3.80e-01 3.58e-01 3.46e-01
stage 0.5 from 0.375      step 12: stagnation its=1600 eta=1.00e-01 mu=4.83e-03 : 4.18e-01 3.76e-01 3.49e-01 3.36e-01 3.11e-01 2.74e-01 2.51e-01 2.32e-01 2.07e-01 1.88e-01 1.60e-01 1.48e-01 1.40e-01 1.29e-01 1.22e-01 1.17e-01
stage 0.4375 from 0.375   step 19: max_iterations its=6000 eta=6.47e-05 mu=5.24e-10 : 7.16e-01 5.96e-01 5.42e-01 4.94e-01 ... 2.53e-04 1.98e-04 1.59e-04 (60 restarts, monotone; stage accepted at r_F 7.73e-10)
stage 0.5 from 0.4375     step 11: stagnation its=400  eta=1.00e-01 mu=6.55e-03 : 3.33e-01 3.02e-01 2.91e-01 2.83e-01
stage 0.46875 from 0.4375 step 14: stagnation its=400  eta=1.00e-01 mu=1.92e-03 : 6.83e-01 5.74e-01 5.63e-01 5.48e-01
stage 0.453125 from 0.4375 step 15: stagnation its=600 eta=1.00e-01 mu=8.29e-04 : 6.93e-01 6.01e-01 5.67e-01 5.24e-01 4.89e-01 4.83e-01
stage 0.5 (final) from 0.4375 step 11: stagnation its=400 eta=1.00e-01 mu=6.55e-03 : 3.33e-01 3.02e-01 2.91e-01 2.83e-01
```

The full Newton-level curves (r_F, eta, mu per step of every stage) are in `raw/proto_b/digest_b.txt`. Reading
(facts, no tuning): the stagnation stops occur with a decrease of 1-10 % per restart over the last two restarts,
from residuals of order 0.3-0.8; at that rate eta = 0.1 would need tens of further restarts (the one solve that ran
to the cap, `gauss_ch` 0.4375 step 19, decreased monotonically by ~4 orders over 6000 iterations). Whether a larger
stagnation factor / cap would let the 32^3 eps 0.5 cases converge was **not** tested (rule: do not tune); the
iteration counts already exceed the 24^3 values by an order of magnitude.

## Wall time (GPU `TIMING solve=`, V100) and memory

| case | solve [s] | job wall [s] | peak device [MB] |
|---|---|---|---|
| gauss_0.25_16 | 0.74 | 4 | 499 |
| gauss_0.25_24 | 3.93 | 8 | 529 |
| gauss_0.25_32 | 15.28 | 19 | 573 |
| gauss_0.5_16 | 1.85 | 5 | 499 |
| gauss_0.5_24 | 11.90 | 16 | 529 |
| gauss_0.5_32 | 270.26 | 274 | 581 |
| gauss_ch_0.25_16 | 0.82 | 4 | 499 |
| gauss_ch_0.25_24 | 3.81 | 7 | 529 |
| gauss_ch_0.25_32 | 14.41 | 18 | 573 |
| gauss_ch_0.5_16 | 1.94 | 6 | 499 |
| gauss_ch_0.5_24 | 12.76 | 16 | 529 |
| gauss_ch_0.5_32 | 329.82 | 333 | 581 |
| control2d_0.25_16 | 0.48 | 4 | 499 |
| control2d_0.25_24 | 2.61 | 7 | 529 |
| control2d_0.5_16 | 1.07 | 4 | 499 |
| control2d_0.5_24 | 5.67 | 10 | 529 |
| generic3d_0.25_16 | 0.79 | 4 | 499 |
| generic3d_0.5_16 | 1.88 | 6 | 499 |
| generic3d_1_16 | 45.96 | 49 | 499 |
| generic3d_1_20 | 52.60 | 56 | 515 |
| generic3d_1_24 | 87.55 | 91 | 531 |

(`peak device` = `MEMORY peak_device_bytes` / 2^20, device-wide.) N6 for comparison: `gauss_0.5_24` 111.9 s
(continuation_floor) -> 11.9 s (converged); `gauss_ch_0.5_24` 64.1 s -> 12.8 s; `gauss_0.25_24` 20.2 s -> 3.9 s.
Coarse cost per Newton step at 32^3 (K = 4096, banded host LU): assembly ~0.026 s + LU ~0.56-0.62 s per build;
~6-7 ms per application.

## Recipe of the per-stage table

Parsed from the committed `logs/proto_b/<case>.log`: a stage starts at a `STAGE field=... eps=E ... (from eps=A` /
`(final attempt from eps=A` line and ends at `STAGE_END ... status=S its=n ... -> accepted|FAILED`; inside it,
every `COARSE build` line gives `K`, `t_assembly`, `t_lu`, `rcond_est`; every `COARSE apply` line gives
`t_apply_avg`; every `LINEAR` line gives `its` and `status` (failure = status != converged). Accepted-step counts:
a `NEWTON ... lambda=` line (an accepted step) and the `LINEAR` line before it.
