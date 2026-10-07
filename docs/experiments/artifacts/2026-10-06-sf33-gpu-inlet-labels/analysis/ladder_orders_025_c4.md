# SF-33 N8'' — production refinement ladder at eps = 0.25 (32 / 64 / 128) with the C4 grid-scaled shift

Same case, same command and same driver defaults as N8' (`ladder_orders_025.md`), except the solver build: commit
`964c43a` (SF-33 C4: pseudo-transient shift `mu_0 (h_ref / h)^2`, `h_ref = 1/16`, i.e. effective `mu_0` = 1/4, 1/16,
1/64 at 32, 64, 128; printed as `psitc_h_ref=0.0625 psitc_mu0_eff=...` on the SOLVER line and on every STAGE line).
The deviation recorded in N8' (eps 0.5 / 1.0 ladders blocked by the N6b gate; eps 0.25 substituted) still applies.

## Command (job `sf33d-ladder-025`, V100 GPU 0, 2026-10-07T10:26:02Z-13:36:57Z, job exit 0)

`env LADDER_DIR=ladder_0.25_c4 bash scripts/run_campaign_b.sh build/v100-release ladder` (the only script change:
`LADDER_DIR` selects the output directory, default `ladder_0.25`), per N in 32, 64, 128:
`inlet_slab --production --n N --eps 0.25 --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder --threads 32
--summary raw/ladder_0.25_c4/N<N>.json > logs/ladder_0.25_c4/N<N>.log` under `/usr/bin/time -v`; then on the host
`ladder_orders.py` -> `raw/ladder_0.25_c4/ladder_orders.md` and `digest_newton.py` ->
`raw/ladder_0.25_c4/digest_newton.txt`. Locally (this node): `ladder_orders.py` re-run on the three logs (output
identical to the host file, `diff` empty), `digest_newton.py` re-run on the three JSON (identical up to file order),
`digest_coarse.py` -> `raw/ladder_0.25_c4/digest_coarse.md`.
Build: job `sf33d-build` (CUDA 11.4.152, preset `v100-release`, 6/6 `inlet_slab_` tests passed).
Solver: as N8' (P-A + coarse `mult`/2 colored banded; Psi-tc SER, `mu0` 1, `mu_max` 100, merit norm; EW forcing
eta0 = eta_max = 0.1; GMRES restart 100, cap 6000, stagnation factor 0.9; max-newton 120; Newton stagnation window
5 / factor 0.5; bisect 4) plus the C4 scaling. Oracle as N8'.
Per-case driver exit codes: N32 0 (converged), N64 0 (converged), N128 15 (continuation_floor).

## Per grid

| N | STATUS | r_F | PATH | Newton steps (all stages) | GMRES its total / max / median (all solves) | linear failures | wall [s] | host max RSS |
|---|---|---|---|---|---|---|---|---|
| 32 | converged | 1.097e-14 | `0.25` | 11 | 218 / 58 / 13 | 0 | 17 | 0.66 GiB |
| 64 | converged | 5.102e-14 | `0.25(fail)->0.125->0.25(fail)->0.1875->0.25` | 34 | 3914 / 1100 / 36 | 2 | 690 (11 min 30 s) | 1.11 GiB |
| 128 | **continuation_floor** | 9.158e-01 (final attempt) | `0.25(fail)->0.125->0.25(fail)->0.1875(fail)->0.15625(fail)->0.140625(fail)->0.25(final)` | 25 (incl. the 2 of the final attempt) | 4449 / 900 / 100 | 6 (incl. the final attempt) | 10747 (2 h 59 min) | 4.79 GiB |

Per stage (`digest_newton.txt`, `GMRES_STATS`):

| N | stage (from) | status | Newton steps | final r_F | GMRES total / max / median |
|---|---|---|---|---|---|
| 32 | 0.25 (0) | converged | 11 | 1.10e-14 | 218 / 58 / 13 |
| 64 | 0.25 (0) | linear_failure (step 4) | 4 | 5.54e-02 | 411 / 300 / 54.5 |
| 64 | 0.125 (0) | converged | 9 | 1.98e-14 | 172 / 58 / 12 |
| 64 | 0.25 (0.125) | linear_failure (step 4) | 4 | 2.14e-02 | 409 / 300 / 53.5 |
| 64 | 0.1875 (0.125) | converged | 8 | 6.41e-14 | 426 / 190 / 36 |
| 64 | 0.25 (0.1875) | converged | 9 | 5.10e-14 | 2496 / 1100 / 200 |
| 128 | 0.25 (0) | linear_failure (step 3) | 3 | 4.92e-01 | 390 / 300 / 88 |
| 128 | 0.125 (0) | converged | 9 | 7.94e-14 | 2118 / 900 / 100 |
| 128 | 0.25 (0.125) | linear_failure (step 2) | 2 | 9.16e-01 | 302 / 300 / 151 |
| 128 | 0.1875 (0.125) | linear_failure (step 3) | 3 | 3.57e-02 | 401 / 300 / 100 |
| 128 | 0.15625 (0.125) | linear_failure (step 3) | 3 | 1.64e-02 | 320 / 300 / 19 |
| 128 | 0.140625 (0.125) | linear_failure (step 3) | 3 | 8.12e-03 | 616 / 600 / 15 |
| 128 | 0.25 final (0.125) | linear_failure (step 2) | 2 | 9.16e-01 | 302 / 300 / 151 (bitwise = the 0.25-from-0.125 attempt) |

No Newton line-search retry (`retries=0` on every NEWTON line) and no stage ended by the Newton stagnation rule on
any grid.

## CASE values (cand=i1o4) and the ceiling (cand=oracle_fd4)

| N | e_v | e_psi | e_psi1 | e_psi2 | e_i1 | e_i2 | e_div | min_c | ceiling e_v | ceiling e_i1 / e_i2 | ceiling e_div |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1.711e-03 | 1.250e-02 | 1.250e-02 | 1.140e-02 | 8.425e-04 | 5.905e-04 | 7.440e-04 | 0.4950 | 2.247e-03 | 8.389e-05 / 7.095e-05 | 8.281e-04 |
| 64 | 3.904e-04 | 1.555e-03 | 1.483e-03 | 1.555e-03 | 1.449e-04 | 1.155e-04 | 5.528e-05 | 0.4946 | 5.684e-04 | 9.110e-06 / 7.396e-06 | 5.742e-05 |
| 128 | (8.219e-02) | (4.500e-01) | (4.429e-01) | (4.500e-01) | (4.818e-02) | (3.457e-02) | (5.993e-04) | (0.5810) | 1.427e-04 | 1.368e-06 / 1.198e-06 | 3.750e-06 |

The 32^3 and 64^3 values equal the N8' values to the printed digits (same converged discrete solution; the shift only
changes the path). The 128^3 `i1o4` values (parenthesized) are the metrics of the NON-CONVERGED final iterate
(r_F = 0.916, after one Newton step of the final attempt from the 0.125 solution); they are not a solution and are
excluded from the orders. The ceiling row is solver-independent and identical to N8'.

## Observed orders

| quantity | 32 -> 64 | 64 -> 128 |
|---|---|---|
| e_v | 2.13 | no converged solution at 128 (script value -7.72 is meaningless) |
| e_psi | 3.01 | no converged solution at 128 (-8.18) |
| e_psi1 | 3.08 | no converged solution at 128 (-8.22) |
| e_psi2 | 2.87 | no converged solution at 128 (-8.18) |
| e_i1 | 2.54 | no converged solution at 128 (-8.38) |
| e_i2 | 2.35 | no converged solution at 128 (-8.23) |
| e_div | 3.75 | no converged solution at 128 (-3.44) |
| ceiling e_v (oracle_fd4) | 1.98 | 1.99 |

**Acceptance (b) threshold applied to eps 0.25 (reading for the reviewer): FAIL**, unchanged from N8': orders of
`e_v` (2.13) and `e_psi` (3.01) are >= 1.8 on 32 -> 64 with r_F <= 1e-10 on both grids, but 128^3 ended
`continuation_floor` (r_F 0.916), so the 64 -> 128 pair does not exist; 64^3 needed two bisections (PATH with failed
stages, no floor). Criterion "three grids converged, r_F <= 1e-10, no floor": FAIL at 128^3. The script's own line:
`e_v orders >= 1.8 on every pair: no (2.13 -7.72); e_psi: no (3.01 -8.18); r_F <= 1e-10 on every grid: no;
continuation floor reached: yes; PATH with a failed stage: yes.`

## Production oracle (acceptance (d): round trip <= 1e-8 on every plane at h/8)

| N | (h/8, 1e-8) max round trip | (h/16, 1e-8) | (h/16, 1e-10) | non_ok | t_trace primary / h16 [s] |
|---|---|---|---|---|---|
| 32 | 1.227e-09 | 1.053e-10 | 1.053e-10 | 0 | 0.84 / 1.47 |
| 64 | 1.071e-10 | 1.065e-11 | 1.065e-11 | 0 | 11.39 / 22.39 |
| 128 | 1.127e-11 | 9.331e-13 | 9.331e-13 | 0 | 179.52 / 357.27 |

PASS at every grid (identical round trips to N8'; the oracle depends only on the stage inputs).
SF-18 field identity: `applied_scale` = 1.602888711251e+00 at 32, 64, 128 (relative spread 0.0 <= 1e-10: PASS;
active modes 15375 / 127007 / 1032255). SF-19 converged at every grid (10 PCG its per corrector; div_max 2.5e-12 /
7.0e-12 / 1.8e-11); inlet vmin 0.5095 / 0.5078 / 0.5071.

## Cost: TIMING per phase, memory, coarse correction

| N | field [s] | prepare [s] | stage builds [s] | solve [s] | of which coarse LU [s] | linear (GMRES) [s] | oracle [s] (3 runs) | total [s] | peak device [GB] | workspaces [GB] | host max RSS [GiB] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.056 | 0.047 | 0.95 | 9.61 | 6.5 | 1.50 | 3.82 | 13.53 | 0.618 | 0.103 | 0.66 |
| 64 | 0.087 | 0.287 | 3.24 | 629.27 | 394.4 | 214.8 | 56.24 | 685.90 | 1.324 | 0.746 | 1.11 |
| 128 | 0.598 | 2.211 | 8.76 | 9846.05 | 7818.3 | 1904.6 | 894.33 | 10743.27 | 6.506 | 5.960 | 4.79 |

| N | coarse K | kl = ku | builds | t_lu mean / max [s] | rcond_est min | applications | t_apply mean [ms] (host banded solve) |
|---|---|---|---|---|---|---|---|
| 32 | 4096 | 639 | 11 | 0.588 / 0.641 | 1.6e-05 | 229 | 6.0 (5.1) |
| 64 | 16384 | 1279 | 34 | 11.60 / 14.03 | 2.5e-06 | 3971 | 51.7 (49.6) |
| 128 | 65536 | 2559 | 25 | 312.7 / 372.8 | 3.5e-09 | 4503 | 397.6 (389.2) |

The 128^3 solve is again dominated by the host banded LU (79 % of the solve phase; 221-247 s for the first build of a
stage, 331-373 s for the later ones). Fewer Newton steps (25 vs 106) cut the 128^3 wall from 8 h 37 min (N8') to
2 h 59 min; 64^3 from 944 s to 690 s.

## The 128^3 failure: every failed stage

Every failed stage at 128^3 (and at 64^3) stopped by **linear_failure**: GMRES `status=stagnation` (the restart-to-
restart true-residual reduction exceeded the stagnation factor 0.9), the step was not taken. **None** stopped by the
Newton stagnation rule. Per failed stage (failing step k; eta_k = forcing target; mu_k = shift; per-restart true
relative residual; coarse `rcond_est` of the factor used in the failed solve):

| stage (from) | step | r_F history | eta at failure | mu at failure | GMRES its / cycles | per-restart true rel | rel / eta | coarse rcond_est |
|---|---|---|---|---|---|---|---|---|
| 0.25 (0) | 3 | 41.6, 1.87, 0.492 | 6.24e-02 | 1.85e-04 | 300 / 3 | 0.348, 0.347, 0.347 | 5.6 | 6.8e-09 |
| 0.25 (0.125) | 2 | 20.5, 0.916 | 1.80e-03 | 7.00e-04 | 300 / 3 | 4.85e-3, 4.76e-3, 4.74e-3 | 2.6 | 2.4e-06 |
| 0.1875 (0.125) | 3 | 9.84, 0.442, 3.57e-2 | 5.89e-03 | 5.68e-05 | 300 / 3 | 0.109, 0.108, 0.107 | 18 | 3.5e-09 |
| 0.15625 (0.125) | 3 | 4.84, 0.218, 1.64e-2 | 5.11e-03 | 5.31e-05 | 300 / 3 | 4.17e-2, 3.93e-2, 3.89e-2 | 7.6 | 1.1e-06 |
| 0.140625 (0.125) | 3 | 2.40, 0.109, 8.12e-3 | 5.02e-03 | 5.29e-05 | 600 / 6 | 1.40e-2, 1.14e-2, 9.72e-3, 8.83e-3, 8.57e-3, 8.37e-3 | 1.7 | 1.2e-06 |
| 0.25 final (0.125) | 2 | 20.5, 0.916 | 1.80e-03 | 7.00e-04 | 300 / 3 | 4.85e-3, 4.76e-3, 4.74e-3 | 2.6 | 2.4e-06 |

The final attempt is bitwise identical to the 0.25-from-0.125 attempt (same r_F, mu, GMRES curve, coarse norms).
At 64^3 the two failed 0.25 stages have the same signature: step 4, mu 2.32e-04 / 1.82e-04, eta 2.74e-02 / 2.25e-02,
per-restart rel 0.581, 0.569, 0.568 / 0.256, 0.245, 0.242, coarse rcond_est 4.0e-06 / 3.3e-06.

### mu sequences (C4 scaling: mu_k = mu0_eff r_F(k) / r_F(0), merit norm)

- 32^3 (mu0_eff = 1/4): 0.25: `2.50e-1, 7.14e-2, 2.04e-2, 4.87e-3, 1.78e-3, 1.49e-3, 5.03e-4, 2.01e-4, 2.26e-5, 4.76e-7,
  1.54e-10` (converged).
- 64^3 (mu0_eff = 1/16): 0.25: `6.25e-2, 7.38e-3, 1.33e-3, 2.32e-4`(fail); 0.125: `6.25e-2, 7.55e-3, 1.24e-3,
  1.96e-4, 1.50e-4, 1.54e-5, 3.73e-6, 1.94e-8, 4.81e-12`; 0.25 from 0.125: `6.25e-2, 7.40e-3, 1.15e-3, 1.82e-4`(fail);
  0.1875: `6.25e-2, 7.50e-3, 1.17e-3, 1.77e-4, 9.07e-5, 8.40e-6, 4.58e-7, 3.11e-10`; 0.25 from 0.1875: `6.25e-2,
  7.42e-3, 1.13e-3, 1.68e-4, 9.87e-5, 9.33e-6, 7.35e-7, 3.03e-9, 2.34e-13` (converged).
- 128^3 (mu0_eff = 1/64 = 1.5625e-2): 0.25: `1.56e-2, 7.03e-4, 1.85e-4`(fail); 0.125: `1.56e-2, 7.10e-4, 7.78e-5,
  1.86e-5, 6.67e-6, 2.02e-6, 1.34e-7, 2.30e-9, 8.22e-13` (converged); 0.25 from 0.125: `1.56e-2, 7.00e-4`(fail);
  0.1875: `1.56e-2, 7.01e-4, 5.68e-5`(fail); 0.15625: `1.56e-2, 7.05e-4, 5.31e-5`(fail); 0.140625: `1.56e-2, 7.08e-4,
  5.29e-5`(fail); final: as 0.25 from 0.125.

Every stage (also warm-started ones) restarts at mu0_eff, as in N8'.

## Comparison with N8' (same case, unscaled mu_0 = 1)

| | N8' (`ladder_orders_025.md`) | N8'' (C4) |
|---|---|---|
| 32^3 | converged, 18 Newton steps, 22 s | converged, 11 steps, 17 s |
| 64^3 PATH | `0.25(fail)->0.125->0.25` (60 steps, 944 s) | `0.25(fail)->0.125->0.25(fail)->0.1875->0.25` (34 steps, 690 s) |
| 64^3 failure mode | linear_failure step 14, rel 0.33 vs eta 0.1 | linear_failure step 4 (twice), rel 0.57 / 0.24 vs eta 2.7e-2 / 2.3e-2 |
| 128^3 stage 0.125 | Newton stagnation rule, 20 steps, r_F 1.08e-2 | **converged**, 9 steps, r_F 7.9e-14 |
| 128^3 other stages | 0.25 linear_failure (step 13); 0.0625 / 0.03125 / 0.015625 Newton stagnation (20 steps each) | 0.25 (twice + final), 0.1875, 0.15625, 0.140625 all linear_failure at step 2-3 |
| 128^3 wall | 31046 s (8 h 37 min), 106 Newton steps | 10747 s (2 h 59 min), 25 steps |
| mu at the failing linear solve | ~2-4e-3 | 5.3e-5 - 7.0e-4 |

What changed: the slow "damped linear contraction" phase of N8' is gone (no stage on any grid ends by the Newton
stagnation rule, the per-step r_F reduction is fast from step 1), so the N8' hypothesis (shift `mu D`, `D = q_v/h^2`,
dominating at 128^3) is consistent with these data for the Newton-stagnation failures; the 0.125 stage at 128^3 now
converges. What did not change: the eps 0.25 target and every stage above 0.125 at 128^3 fail in the linear solve
(GMRES restart stagnation, P-A + coarse `mult`) at the first or second step where mu has dropped to ~5e-5 - 7e-4; at
64^3 the same mode now costs one extra bisection. The bisection budget (4) is exhausted between 0.125 and 0.25 with
the smallest tried increment 0.015625 (0.140625) still failing (rel 8.4e-3 vs eta 5.0e-3, 6 restarts).

Observations, not tested here (hypotheses for the owner/orchestrator): (i) two of the failing solves (0.25 from 0,
0.1875) used a coarse factor with `rcond_est` ~ 3-7e-9, three orders below every other build of the run (1e-6 -
3e-5), while the other failing solves had ordinary rcond (1e-6 - 2e-6), so near-singularity of the coarse operator
is not a uniform explanation; (ii) the converged 0.125 stage passed through mu ~ 8e-5 - 2e-5 without linear failure
(GMRES max 900), so a small shift alone is not sufficient for failure; the failure depends on the amplitude
increment and the iterate. A discriminating study (restart / inner-cap / stagnation-factor or coarse-space variants
at 128^3, 0.1875 from 0.125) is a solver-policy change outside this evidence node.

## Files

`logs/ladder_0.25_c4/N{32,64,128}.{log,time}`, `raw/ladder_0.25_c4/N{32,64,128}.json`,
`raw/ladder_0.25_c4/ladder_orders.md` (host), `raw/ladder_0.25_c4/digest_newton.txt` (host),
`raw/ladder_0.25_c4/digest_coarse.md` (local), job logs `logs/jobs/sf33d-build.log`, `logs/jobs/sf33d-ladder-025.log`.
