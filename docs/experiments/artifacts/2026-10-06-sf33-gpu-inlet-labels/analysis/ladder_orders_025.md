# SF-33 N8' — production refinement ladder at eps = 0.25 (32 / 64 / 128), Gate 3A metrics, 128^3 cost

Deviation (recorded): the spec's step 9b amplitudes (eps 0.5 / 1.0) are blocked by the N6b preconditioner gate
(FAIL at 32^3, eps 0.5; spec failure policy: "the 64^3-128^3 runs do not start"). eps 0.25, the amplitude validated
by the gate at 16/24/32, is substituted as the demonstrable refinement study. No eps 0.5 / 1.0 run at 64 / 128 was
started.

## Command (job `sf33c-ladder-025`, V100 GPU 0, 2026-10-07T01:12:14Z-10:05:48Z, job exit 0)

`scripts/run_campaign_b.sh build/v100-release ladder` (same loop as `run_ladder.sh`, plus `/usr/bin/time -v` per
grid for the host RSS), per N in 32, 64, 128:
`inlet_slab --production --n N --eps 0.25 --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder --threads 32
--summary raw/ladder_0.25/N<N>.json > logs/ladder_0.25/N<N>.log` (`ell/h` = 8, 16, 32); then `ladder_orders.py`
-> `raw/ladder_0.25/ladder_orders.md`, `digest_newton.py` -> `raw/ladder_0.25/digest_newton.txt`; locally
`digest_coarse.py` -> `raw/digest_coarse.md`. Build: `build/v100-release` (CUDA 11.4, preset `v100-release`) of
commit `94c5d53` (chain head `3676bb0` + the N8' driver options, unused here).
Solver (SOLVER line of every log, the driver defaults after C3): P-A + coarse correction `mult`, 2 profiles,
colored assembly, banded host LU; Psi-tc on (SER, mu0 1, mu_max 100, D = q_v / h^2); EW forcing (eta0 = eta_max =
0.1, eta_min = 1e-12); GMRES restart 100, inner cap 6000, stagnation factor 0.9; max-newton 120, newton_tol 1e-13,
Newton stagnation window 5 / factor 0.5; ladder (0.25, 0.5, 1), bisect 4. Oracle: `h_max = h/8`, tol 1e-8, gate
1e-8 (primary), ladder `(h/16, 1e-8)`, `(h/16, 1e-10)`, 32 threads.
Per-case driver exit codes: N32 0 (converged), N64 0 (converged), N128 15 (continuation_floor).

## Per grid

| N | STATUS | r_F | PATH | Newton steps (all stages) | GMRES its total / max / median (all solves) | linear failures | wall [s] | host max RSS |
|---|---|---|---|---|---|---|---|---|
| 32 | converged | 1.073e-14 | `0.25` | 18 | 257 / 50 / 9.5 | 0 | 22 | 0.65 GiB |
| 64 | converged | 5.839e-14 | `0.25(fail)->0.125->0.25` | 60 | 4253 / 1000 / 3.5 | 1 (first 0.25 attempt, step 14, GMRES restart stagnation rel 0.33 vs eta 0.1) | 944 | 1.11 GiB |
| 128 | **continuation_floor** | 1.557e-01 (final attempt) | `0.25(fail)->0.125(fail)->0.0625(fail)->0.03125(fail)->0.015625(fail)->0.25(final)` | 106 | 1226 / 300 / 2 | 2 (both 0.25 attempts, step 13, rel 0.36 vs eta 0.1) | 31046 (8 h 37 min) | 4.79 GiB |

Stages at 128^3 (`digest_newton.txt`): 0.25 linear_failure (13 steps, r_F 0.156); 0.125 stagnation (20 steps,
r_F 1.08e-2); 0.0625 stagnation (20, 5.73e-3); 0.03125 stagnation (20, 2.94e-3); 0.015625 stagnation (20, 1.49e-3);
final attempt at 0.25 = the first attempt bitwise (linear_failure at step 13, r_F 1.557e-01). The 64^3 stages: 0.25
linear_failure (14 steps), 0.125 converged (23), 0.25 from 0.125 converged (23, GMRES max 1000, median 100).

## CASE values (cand=i1o4) and the ceiling (cand=oracle_fd4)

| N | e_v | e_psi | e_psi1 | e_psi2 | e_i1 | e_i2 | e_div | min_c | ceiling e_v | ceiling e_i1 / e_i2 | ceiling e_div |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1.711e-03 | 1.250e-02 | 1.250e-02 | 1.140e-02 | 8.425e-04 | 5.905e-04 | 7.440e-04 | 0.4950 | 2.247e-03 | 8.389e-05 / 7.095e-05 | 8.281e-04 |
| 64 | 3.904e-04 | 1.555e-03 | 1.483e-03 | 1.555e-03 | 1.449e-04 | 1.155e-04 | 5.528e-05 | 0.4946 | 5.684e-04 | 9.110e-06 / 7.396e-06 | 5.742e-05 |
| 128 | (5.379e-02) | (8.002e-01) | (7.095e-01) | (8.002e-01) | (4.122e-02) | (3.036e-02) | (3.900e-05) | (0.5061) | 1.427e-04 | 1.368e-06 / 1.198e-06 | 3.750e-06 |

The 128^3 `i1o4` values (parenthesized) are the metrics of the NON-CONVERGED final iterate (r_F = 0.156) printed by
the driver; they are not a solution and are excluded from the orders below. The ceiling row is the oracle's own
4th-order FD metrics and does not depend on the solver.

## Observed orders

| quantity | 32 -> 64 | 64 -> 128 |
|---|---|---|
| e_v | 2.13 | n/a (128 not converged; the script's -7.11 is meaningless) |
| e_psi | 3.01 | n/a (-9.01) |
| e_psi1 | 3.08 | n/a |
| e_psi2 | 2.87 | n/a |
| e_i1 | 2.54 | n/a |
| e_i2 | 2.35 | n/a |
| e_div | 3.75 | n/a |
| ceiling e_v (oracle_fd4) | 1.98 | 1.99 |

**Acceptance (b) threshold applied to eps 0.25 (reading for the reviewer): FAIL.** Orders of `e_v` (2.13) and
`e_psi` (3.01) are >= 1.8 on 32 -> 64 with r_F <= 1e-10 on both grids, but the 128^3 run ended
`continuation_floor` (r_F 0.156 > 1e-10), so the 64 -> 128 pair does not exist; the 64^3 run also needed one
bisection (PATH with a failed stage, no floor). Criterion 1 of N8' ("three grids converged, r_F <= 1e-10, no floor"):
FAIL at 128^3.

## Production oracle (acceptance (d): round trip <= 1e-8 on every plane at h/8)

| N | (h/8, 1e-8) max round trip | (h/16, 1e-8) | (h/16, 1e-10) | non_ok | FLAGGED planes | ORACLE_LABELDIFF max (h/16 vs primary) | t_trace primary / h16 [s] |
|---|---|---|---|---|---|---|---|
| 32 | 1.227e-09 | 1.053e-10 | 1.053e-10 | 0 | 0 | 7.1e-10 | 0.86 / 1.42 |
| 64 | 1.071e-10 | 1.065e-11 | 1.065e-11 | 0 | 0 | 6.3e-11 | 11.22 / 22.05 |
| 128 | 1.127e-11 | 9.331e-13 | 9.331e-13 | 0 | 0 | 5.8e-12 | 180.87 / 358.86 |

PASS at every grid and every plane (per-plane values: `ORACLE` lines of each log). The 128^3 oracle ran on the
target stage although the solver did not converge (the oracle depends only on the stage inputs).

SF-18 field identity: `applied_scale` = 1.602888711251e+00 and `raw_variance` = 3.892183071637e-01 at 32, 64 and
128 (max relative spread 0.0, assert <= 1e-10: PASS; active modes 15375 / 127007 / 1032255). SF-19: converged at
every grid (10 PCG its per corrector, rel <= 1.1e-12; div_max 2.5e-12 / 7.0e-12 / 1.8e-11); inlet vmin 0.5095 /
0.5078 / 0.5071.

## Cost (Gate 3A item (c)): TIMING per phase, memory, coarse correction

| N | field [s] | prepare [s] | stage builds [s] (per stage) | solve [s] (incl. builds) | of which coarse LU [s] | linear (GMRES) [s] | oracle [s] (3 runs) | metrics [s] | total [s] | peak device [GB] | workspaces [GB] | host max RSS [GiB] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.057 | 0.047 | 0.96 (1 stage) | 14.30 | 10.4 | 1.75 | 3.76 | 0.004 | 18.17 | 0.618 | 0.103 | 0.65 |
| 64 | 0.090 | 0.294 | 2.2 (1.09-1.10 each, 2 stages) | 884.9 | 623.5 | 231.3 | 55.5 | 0.011 | 940.8 | 1.291 | 0.728 | 1.11 |
| 128 | 0.599 | 2.226 | 8.7 (1.72-1.79 each, 5 stages) | 30140.4 | 29116.1 | 546.5 | 899.1 | 0.033 | 31042.4 | 6.506 | 5.960 | 4.79 |

Peak device = `cudaMemGetInfo` device-wide (GPU 0 was used by this job only; the concurrent N8' jobs ran on GPU 1).

| N | coarse K | kl = ku | builds | t_lu mean / max [s] | t_assembly mean [s] | t_cond mean [s] | rcond_est min | applications | t_apply mean [ms] (host banded solve) | GMRES s / it (approx.) |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 4096 | 639 | 18 | 0.579 / 0.638 | 0.027 | 0.035 | 1.6e-05 | 275 | 6.0 (5.1) | ~7 ms |
| 64 | 16384 | 1279 | 60 | 10.39 / 12.71 | 0.138 | 0.321 | 2.5e-06 | 4337 | 49.5 (47.4) | ~54 ms |
| 128 | 65536 | 2559 | 106 | 274.7 / 343.8 | 1.037 | 3.358 | 4.3e-06 | 1336 | 393.0 (384.6) | ~0.45 s |

The 128^3 cost is dominated by the host banded LU of the coarse correction: 274.7 s mean per Newton step (227 s
in the first stages, 315-344 s later in the run, variation not investigated; N7c extrapolated ~1 min), 97 % of the solve phase; the per-application
host banded solve (0.39 s) dominates each GMRES iteration (the N7c extrapolation was ~0.2 s). Host memory: 4.79 GiB
max RSS at 128^3 (N7c estimate ~4 GB for the band). The concurrent jobs (`sf33c-oracle32-vs-sf29`,
`sf33c-prod32-05`, 01:12-01:16Z) overlapped only with the 32^3 run and the first minutes of 64^3.

## The 128^3 failure (observations; the cause is a hypothesis, not established)

- Linear solves are not the limiting factor at small amplitude: in the 0.0625, 0.03125 and 0.015625 stages every
  GMRES solve converged in 1-3 iterations; the stages end on the NEWTON stagnation rule (merit after 4 steps >
  0.5 x merit 4 steps earlier) at exactly 20 steps, with r_F decreasing by a factor ~0.5 per step at first,
  ~0.75 in the middle and ~0.85 at the end (0.015625 stage: r_F 2.30, 1.10, 0.51, 0.26, ... 2.95e-3, 2.44e-3, 2.08e-3,
  1.78e-3, 1.49e-3; mu 1, 0.48, 0.22, ... 1.7e-3, 1.3e-3, 1.1e-3, 9.1e-4, 7.7e-4).
- The same slow phase exists at 64^3 (0.125 stage: per-step ratios 0.70-0.80 at steps 12-16 while mu ~ 3.8e-3 -
  1.4e-3) and ends when mu drops below ~1e-3 (steps 17-23: 0.63, 0.52, 0.36, 0.17, 0.035, 1.3e-3, 3.7e-6 -> converged
  in 23 steps).
- Hypothesis (to be tested by the owner/orchestrator, NOT tested here): the Psi-tc shift `mu D` with `D = q_v / h^2`
  grows 4x per refinement while the SER schedule `mu ~ merit` is grid-independent, so at 128^3 the shift dominates
  the smooth (low-wavenumber) part of the Jacobian for a residual range 4x wider than at 64^3; the Newton iteration is
  then a damped linear contraction and the 5-step stagnation rule fires before mu becomes small enough. A discriminating
  run would be the 128^3 eps 0.25 case (or a 0.0625 stage) with `--psitc off` or a smaller `--psitc-mu0`, which is a
  solver-policy change outside N8'.
- The first-attempt linear failure at eps 0.25 (step 13 at 128^3, step 14 at 64^3: GMRES restart stagnation at
  rel 0.33-0.36 with mu ~ 2-4e-3) is the N6b failure mode at a smaller amplitude on finer grids.
