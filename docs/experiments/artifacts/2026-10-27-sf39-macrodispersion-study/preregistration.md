# SF-39 pre-registration: fixed settings, executable estimator definitions, reading rules and script contracts (activation commit, before any run)

- Increment: `docs/plans/active/lester-eq14/increments/SF-39-macrodispersion-study.md` (State `active` in the same commit).
  Base: `1e40cfe9cb5132174bd167d7f21ba0c77910623e` (`master`; PR #82 merged on top of PR #81 `adec2080`). Authority: the SF-39
  specification; decisions `docs/decisions/2026-10-27-sf39-pre-activation-decisions.md` (F1-F14) and
  `docs/decisions/2026-10-27-remote-single-gpu.md`; through them D1, D3, D10, D11, D14-D20 (2026-10-23) and E4-E7 (2026-10-24).
- What this file is: the durable text that the activation bitácora row cites (sha256 there). It fixes, verbatim, the items of
  "Implementation specification" 1 and resolves, before any run, executability gaps found by the orchestrator's reconnaissance of the
  runner, the Par2 pipeline and the governance chain, after three adversarial reviews of a draft (governance fidelity, executability
  against the code, statistics). **It changes no Goal, threshold, margin, CI level or target, branch, geometry or estimator.** Where it
  fixes a convention the governance left open, the choice is derivable without data and listed in section 8 with its source; departures
  from a governance text forced by the code are listed there as such (R15, R27, R30) for the human review. The acceptance thresholds A-E
  are the specification text, copied verbatim into the activation row.
- Date of knowledge: no SF-39 realization exists (field seeds 5001+, 7101+, 7201+, 9001+ and the check seed 4901 are unused by every earlier
  run; Par2's injection seeds `123456789 + 1000 r` equal SF-38's, with different fields). The orchestrator writing this file has seen
  SF-38's A/B statistics (seeds 4001-4010) and post-hoc readings P1-P6, SF-41's F-ladder plane variances and the pre-activation review; no
  choice below depends on them.
- No `alpha_L` / `alpha_T` statement is made here or before the analysis part (Rule 1; Gate 4).

## 1. Fixed run settings (Implementation 1)

Runner `lagrangian_ensemble` (SF-41 medium (b)), one binary set built once at the base (RR1). Box unit = transverse period;
`ell = (lambda/h)/n_T` with `n_T = ny`; `L1 = nx/ny`; `x0 = 5 ell` (runner default for medium (b), not passed); `G1 = 1`;
`normalize_variance` on (hard-wired, E7); amplitude by `--eps` with `--sigma2 1`, so `sigma^2_lnK = eps^2` exactly per realization
(raw variance and applied scale in `field.json`).

Common flags of every runner arm: `--medium b --sigma2 1 --np-side 100 --times auto --tol 1e-8 --fused-splines --single-launch
--twin on --pollock-bridge on --planes <list of section 2>`; `--window 0.1,0.9,0.1,0.9` unless stated (`N_p = 1e4`, `M = 100`, flux
weights); `--pcg-rtol` default `1e-10`; no `--x0`, no `--max-arclength`, no `--head-mapped-host`.

| arm | role | `--grid` | `lambda/h` | `--ell` | `--eps` | `--seed0` | `--nr` | `--hmax-div` | planes |
|---|---|---|---|---|---|---|---|---|---|
| `s1_A` | production, directory A | `2048,256,256` | 10 | `0.0390625` | `1` | `5001` | pilot 20, then `N_A` | `8,16,32` | P1 |
| `s1_B` | production, directory B | `2048,256,256` | 10 | `0.0390625` | `1` | `5001+N_A` | `N_run - N_A` | `8` | P1 |
| `s1_n05` / `s1_n10` / `s1_n20` | narrow rungs 1/2, 1, 2 | `1024,64,64` / `2048,128,128` / `4096,256,256` | 5 / 10 / 20 | `0.078125` | `1` | `5001` | `N_narrow` | `8` | P4 |
| `s1_tw` | coarse twin | `1024,128,128` | 5 | `0.0390625` | `1` | `5001` | `N_twin` | `8` | P1 |
| `s225_A` / `s225_B` | production A / B | `3328,384,384` | 16 | `0.041666666666666664` | `1.5` | `5001` / `5001+N_A` | as `s1` | `8,16,32` / `8` | P2 |
| `s225_n08` / `s225_n16` / `s225_n32` | narrow rungs | `1664,64,64` / `3328,128,128` / `6656,256,256` | 8 / 16 / 32 | `0.125` | `1.5` | `5001` | `N_narrow` | `8` | P5 |
| `s225_tw` | coarse twin | `1664,192,192` | 8 | `0.041666666666666664` | `1.5` | `5001` | `N_twin` | `8` | P2 |
| `s4_A` / `s4_B` | production A / B | `3328,384,384` | 32 | `0.083333333333333329` | `2` | `5001` / `5001+N_A` | as `s1` | `8,16,32` / `8` | P3 |
| `s4_n16` / `s4_n32` / `s4_n64` | narrow rungs | `1664,64,64` / `3328,128,128` / `6656,256,256` | 16 / 32 / 64 | `0.25` | `2` | `5001` | `N_narrow` | `8` | P6 |
| `s4_tw` | coarse twin | `1664,192,192` | 16 | `0.083333333333333329` | `2` | `5001` | `max(N_twin, 30)` | `8` | P3 |
| `s4_w24` | width arm (i), F10 | `1664,384,384` | 16 | `0.041666666666666664` | `2` | `7101` | 30 | `8` | P7, `--window 0.3,0.7,0.3,0.7` |
| `s4_w12` | width arm (ii), F10 | `1664,192,192` | 16 | `0.083333333333333329` | `2` | `7201` | 30 | `8` | P3, `--window 0.36666666666666664,0.63333333333333336,0.36666666666666664,0.63333333333333336` |
| `chk_*` | checks only, never data | every narrow size, twin and width geometry of each amplitude, and the `s4_A` grid | as the arm | as the arm | as the arm | `4901` | 1 | `8` (`8,16,32` and `8` for the byte check) | as the arm |

(The `s4_n16` rung likewise runs `max(N_narrow, 30)` realizations: F10 compares against the first 30 of `s4_tw` and `s4_n16`; the ladder
analyses use their first `N_twin` / `N_narrow`.)

**Production counts** (F12 as written, plus the twin pairing): production estimate = realizations `g = 0 .. N_R - 1` (seed `5001 + g`);
`h_max` ladder = `g < N_hmax` (directory A); twin pairs = `g < N_twin`. Realizations run at the production geometry:
`N_run = max(N_R, N_A, N_twin)` with `N_A = max(20, N_hmax)`; directory A holds `g < N_A`, directory B holds `N_A <= g < N_run` (`--seed0
5001+N_A --nr N_run-N_A --hmax-div 8`; absent if `N_run = N_A`). Realizations with `g >= N_R` serve only the ladder they are run for and are
never in the production estimate (F12: "if `N_hmax >= N_R` ... production uses the first `N_R` realizations of A"). Pairs are matched by seed.
Each output directory is used with one `--seed0` only (runner signature; exit 7); raising `--nr` in the same directory extends it.

**Par2 (Gate 5 baseline, D10, F12).** `configs/par2_s{1,225,4}.yaml` = copies of `apps/config_pipeline_par2_sf38_s{1,225,4}.yaml` with
`stochastic.seed: 9001`, analysis `NR: 20` (pilot), `output.output_dir: raw/par2_s{1,225,4}` (relative to the working directory; the pipeline
runs with its working directory `$SF39_OUT_ROOT/par2/`, outside the remote mirror, R27) and an updated header; everything else unchanged
(`cg_rtol 1.0e-6`, `dt`, `n_steps`, `transport.seed 123456789`, `n_particles 10000`, `inject_x = 5 lambda`, `sample_every 10`, snapshots).
Extension, created only after the owner's budget: `configs/par2_s{c}_ext.yaml` = the pilot copy with `stochastic.seed: 9021`,
`transport.seed: 123476789` (`123456789 + 1000 x 20`), `NR: N_R - 20`, `output_dir: raw/par2_s{c}_ext`; global index `g = r` (pilot), `20 + r`
(extension), so field seed `9001 + g` and injection seed `123456789 + 1000 g` equal one uninterrupted launch of `N_R` (R15). Realized
`sigma^2` per realization: `par2_field_stats <config> --nr <NR>` (`FIELDSTAT ... var_lnK=`) on both configs.

Times `--times auto`: per realization `t_n = n ell/2`, `n = 1 .. max(2, floor((L1 - 10 ell)/qbar_r/(ell/2) + 1e-9))` (runner rule for
(b)); the analysis reads the times from `track.json` / `pollock.json` key `times`, never from stdout. `h_max` list `{h/8, h/16, h/32}` =
`--hmax-div 8,16,32` in directory A only. Interval method: Student `t` (primary); BCa bootstrap `B = 5000`, generator
`numpy.random.default_rng(20261027)` created afresh for each interval (sensitivity, section 4).

## 2. Plane lists (F3)

Normative algorithm (Python, doubles): `ell = lh/ny; L1 = nx/ny; x0 = 5.0*ell; L = L1 - 10.0*ell; Xa = x0 + L/2.0; Xb = L1 - 5.0*ell;
mmax = (nx - 10*lh)//lh` (integers); candidates `[x0 + m*ell for m in 1..mmax] + [x0 + L/4.0, Xa, Xb]`, sorted ascending; a value within
`1e-12*max(1,|x|)` of the previously kept one is merged, keeping `Xa` or `Xb` if either is involved; printed `"%.17g"` joined by `,`
(no spaces, no newline). sha256 of that ASCII string:

| list | arms (all members give the identical string) | P | sha256 |
|---|---|---|---|
| P1 | `s1_A`, `s1_B`, `s1_tw` | 197 | `0bf748196fff9c61acca5f9db5973eaf350f4364a8f3d313e66cf50b563340fa` |
| P2 | `s225_A`, `s225_B`, `s225_tw` | 199 | `87a4811e81e0cbf31c4a7d459084663f58e14ec9e40236845c1204a367c0f5a0` |
| P3 | `s4_A`, `s4_B`, `s4_tw`, `s4_w12` | 95 | `e7e54bb4e957ad293beb93bcb2bde8ab7162172e64c1e6f3e94e2b3125fed2eb` |
| P4 | `s1_n05`, `s1_n10`, `s1_n20` | 197 | `b03effae1b1c6172d6a13d12efca504a93a2ec2939c1a9dde7690615d953059f` |
| P5 | `s225_n08`, `s225_n16`, `s225_n32` | 199 | `592dee43813642071a4e92173b02be03c718afb66b8916e5c60a31dc7a90bcd1` |
| P6 | `s4_n16`, `s4_n32`, `s4_n64` | 95 | `2f0d0048c93883df69448f5d59a0ed981f384d589261c6091848865da2e696a3` |
| P7 | `s4_w24` | 95 | `fb48abd16a4fc39aa6217e99360357c869053d7b44acd645ba94c90b62b7ff83` |

`X_a`, `X_b` are kept exactly (canonical doubles) and `X_b` is the last plane of every list; at `sigma^2` 2.25 and 4 they coincide with
lattice planes; `x0 + L/4` is never a lattice plane. The analysis finds `X_a`, `X_b`, `x0 + L/4` by nearest plane within `1e-12 max(1,|x|)`.
Each list contains the runner's auto planes for its geometry to within `1e-12` relative (the auto `X_b` can differ by one ulp).

## 3. Estimators (F2, F3, E4; normative)

Paths: realization directory `R = <out>/realizations/r_%04d/`; pass root `P = R/hmax_8/` in directory A, `P = R` in every single-pass arm;
`R/pollock/`, `R/flow_stats.json`, `R/field.json` at the realization root; uniform-weight series `<out>/hmax_8/stats_twin/realization_%04d_timeseries.csv`
(directory A) or `<out>/stats_twin/...` (single-pass). Every weighted mean/variance is the **biased** weighted form with weights
renormalized over the selected particle set (as the runner). A record is present iff finite; records are prefix-complete (tracker and
Pollock).

- **Breakthrough.** `t_b,r = min over i with status_plane == 0 and a record at X_b of planes_tau[i, X_b]` (box time, h/8 pass; F2's "ok").
  Pilot freeze per amplitude (20 production realizations, directory A): `t_b,min = min_r t_b,r`; `U_prod = mean_r qbar_r` over those 20;
  `tau_b* = U_prod t_b,min / ell_prod`. Written to `tau_b_star.json` before any further run and never changed.
- **Velocity `U`.** A standalone value of an arm (production estimate, B, D's production CI, F6) uses `U` = mean of `qbar_r` over the
  realizations in that value (production: `g < N_R`, pooled over A and B). A **paired difference** uses, on each side, `U` = that side's mean
  `qbar_r` over the seeds of the pair set (narrow rungs, twin vs production, `h_max` passes); every pass of a realization and its Pollock
  bridge use the production-side `U` of that pair set, so passes share their window exactly. At the checkpoint all sets are the pilot's.
- **Window.** `W = [t_lo, t_hi] = [0.75, 1] x tau_b* ell / U` in the run's box time, same frozen `tau_b*` for every arm of the amplitude.
  `j_hi` = first snapshot with `t_j >= t_hi`; if none exists the reduction stops with an error (owner), never clips.
- **Particle set.** `S_r` = particles with a record at `j_hi` (consumer: `times_x`; Pollock: `pollock_times_x`). Both endpoints use `S_r`.
- **Window estimator.** For a unit direction `e`, `V_e(t_j)` = weighted variance of `x . e` over `S_r` at snapshot `j`; `V_e(t)` by linear
  interpolation of the variances between the two bracketing snapshots (error if `t < t_1`); `alpha_e,r = [V_e(t_hi) - V_e(t_lo)] /
  (2 ell U (t_hi - t_lo))` (units of `lambda`). Ensemble value = mean over realizations.
  - Fixed axes: `alpha_LAM = alpha_e1`; `alpha_T,fixed = (alpha_e2 + alpha_e3)/2` (components recorded).
  - Aligned (E4): `q_r = (mean_u, mean_v, mean_w)` of `flow_stats.json` `interior`; `e_L = q_r/|q_r|`; `e_T1`, `e_T2` = Gram-Schmidt of
    `e2`, `e3` against `e_L`; `alpha_TAM = (alpha_eT1 + alpha_eT2)/2` (order-independent); tilt `theta_r = atan2(|e_L x e1|, e_L . e1)`;
    fixed-minus-aligned difference recorded.
  - Plume-drift frame (recorded): `e_D` = unit vector of `mean(x; t_hi) - mean(x; t_lo)` over `S_r` (interpolated means), same Gram-Schmidt.
- **Plane estimator `alpha_TP`** (F3). `T_r` = particles with records at both `X_a` and `X_b` (`planes_x23`); `sigma_T^2(X) = (var x2 + var x3)/2`
  weighted over `T_r`; `alpha_TP,r = [sigma_T^2(X_b) - sigma_T^2(X_a)] / (2 ell (X_b - X_a))`; excluded weight fraction and non-crossing
  counts per plane recorded. Full `sigma_T^2(x1)` series: each plane over its own contributors.
- **Twin and Pollock arms** (F11, Gate 5). Twin = uniform weights over the same set rule. For the **consumer-vs-Pollock contrast** both arms
  use the intersection set (particles with a record in both arrays at `j_hi`, resp. at both planes), uniform weights, same `W`, `U`, `ell`
  (Pollock tracks zero-weight seeds the tracker never traces). Pollock flux values (`pollock_weights.npy`, own set) are recorded.
- **Exits and sensitivity.** Inlet/outlet exits before `t_{j_hi}` (`exits_time.csv` row `j_hi`), non-ok counts of both passes, excluded
  fraction recorded. Sensitivity "full population": every traced particle; an outlet exit frozen at `x1 = L1`, an inlet exit at `x1 = 0`,
  with its last recorded `(x2, x3)` (the exit point is not recorded). Realizations are never dropped.
- **Divergence weights.** Sensitivity `w_i exp(-I_i)` with `I_i` at the evaluated snapshot (`times_I`) or plane (`planes_I`), renormalized per
  evaluation. Lagrangian `e_v = 1 - exp(-I)` at `X_b`: unweighted percentiles 50, 90, 99, 99.9, 100 of `|e_v|` over contributors per
  realization; reported as the ensemble median and maximum of each.
- **Split-half** (F12). Lattice index `p = m2 M + m3`, `M = 100`; half A = `(m2 + m3)` even, half B = odd (checkerboard), each with its own
  renormalized weights and the same set rules; component = `mean_r[(alpha^A_r - alpha^B_r)^2]/4` as a fraction of the between-realization
  sample variance of `alpha_r`; above 25 % -> limitation, no re-run. An indicator (deterministic lattice, halves correlated), not an exact
  sampling variance.
- **Production values.** Consumer, flux weights, h/8 pass, production geometry, `g < N_R` pooled by seed.

## 4. Intervals, verdicts and classification (A-D, F4-F6)

- Student `t` 90 % two-sided CI of a mean over `n` realizations (`n - 1` d.o.f.) for A, C, D, F7, F11; 95 % two-sided for B; one-sided
  95 % upper bounds where stated (B&dD comparison). "Above 0" / "below 0" = the 90 % CI lies entirely above / below 0. BCa beside every
  Student interval (sensitivity): `scipy.stats.bootstrap(..., method='BCa', n_resamples=5000, confidence_level=<same>,
  alternative=<same sidedness>, random_state=numpy.random.default_rng(20261027))`, resampling unit = realization (a paired quantity: the
  per-realization difference as one sample `(d,)`; Welch: two independent samples, statistic reducing along `axis`; ANCOVA: an index sample
  resampling `(alpha_r, sigma^2_r)` jointly); a degenerate sample (constant differences) records `NaN` and the warning.
- **A (equivalence).** Margin per estimator and amplitude: transverse `max(0.05 |alpha|, 1.25e-3)`, `alpha_LAM` `0.025 alpha_LAM`, `alpha` = the
  ensemble mean of the finer setting. For `h`: `m = min(margin at the 2x-rung mean, margin at the production mean)` for the narrow test
  (`D_f`), the twin bound (`2^1.5 m` on `D_tw`) and `nr_rule`. For `h_max`: `m` at the `h/16` mean. Verdicts as the specification; F4
  production verdict as the specification, with `s = +1` if the mean `D_c` is exactly 0. "Trend recorded": the point estimate and 90 % CI of
  every paired difference. Richardson (order 2) of the production error is computed for reporting only under the specification's guard
  (observed narrow order in [1.5, 2.5], monotone), otherwise `|Delta Q|` is reported as a bound; never a correction.
- **D (rule D, F5/F6).** Exactly the specification's branches and order, evaluated **per ladder** (`h`: verdict = F4 production verdict,
  differences `D_f`, `D_tw`; `h_max`: verdict = the `h_max` verdict, difference `h/16 - h/8`): in branches 1-3 the non-`converged` verdict and
  the refinement difference belong to the same ladder, and "the other" is the other difference of that ladder (vacuous for `h_max`); a branch
  fires if it holds for either ladder. Branch 2's "finer-setting 90 % CI" is that of the qualifying difference (`D_f` -> 2x rung, `D_tw` ->
  production, `h/16 - h/8` -> `h/16` pass), and the branch fires if some qualifying difference meets it. Branch 3 is as written (no "other's
  point estimate" clause; recorded for the review, not changed). `physical` sub-labels from the production 90 % CI. Amplitude label: the common
  label of `alpha_TAM` and `alpha_TP`, otherwise `unresolved (estimator-discordant)` with both labels listed.
- **C (Gate 5).** `U_c(arm) = [mean_x(t_b,min,arm) - x_inj] / t_b,min,arm`, `mean_x` from the arm's per-realization uniform-weight series as
  written (Lagrangian: production `stats_twin`, all particles with a record at that time, `x_inj = x0`; Par2: `stats/realization_%04d_timeseries.csv`,
  `x_inj = inject_x`), interpolated linearly at `t_b,min` and averaged over the arm's realizations; `t_b,min` of the Lagrangian arm = the
  production pilot value; of Par2 = the minimum over its first 20 realizations of the last snapshot time (`step x dt`) with no particle at
  `x >= Lx - 5 lambda`. `tau_b*(twin) = U_c t_b,min / ell`, `tau_b*(Par2) = U_c t_b,min / corr_length`, frozen at the pilot with `U_c` over the
  20 pilot realizations; `tau_c` = the smaller. Final `W_c = [0.75, 1] x tau_c lambda_arm / U_c(arm)` with the final `U_c` (`lambda_arm` = `ell`
  for the Lagrangian arm, `corr_length` for Par2); per realization `alpha_LAM = [V_x(t_hi) - V_x(t_lo)] / (2 lambda_arm U_c (t_hi - t_lo))`
  (Lagrangian twin: section 3 set rule, uniform weights; Par2: its series). 90 % Welch interval, Welch-Satterthwaite d.o.f.; margin `0.10 x`
  the Par2 ensemble mean (a fixed number). Sensitivities: (i) FV `qbar` replaces `U_c` everywhere in the Lagrangian arm; (ii) ANCOVA: Par2
  `alpha_r = a + b c_r + e`, `c_r = sigma^2_r - sigma^2_nominal`, OLS, `SE_a = s_e sqrt(1/n_P + cbar^2/S_cc)`, against the Lagrangian mean with
  `nu = (SE_a^2 + s_L^2/n_L)^2 / [SE_a^4/(n_P - 2) + (s_L^2/n_L)^2/(n_L - 1)]`, margin `0.10 a`. Par2 realizations after the pilot whose last
  clean snapshot precedes the end of `W_c` are counted and reported, never dropped. Par2's in-pipeline `alpha` is not used. The Par2 `dt/2`
  sensitivity is the SF-38 E record, cited.

## 5. Estimation and exploratory (F7, F8, F10, F11, E)

- **F7.** Per realization `Delta_half,r = alpha_TP[x0 + L/4, X_a] - alpha_TP[X_a, X_b]`, each term over its own two-plane set; 90 % Student CI;
  "decreasing" iff above 0, and then the second-half value is also reported as an upper bound on the asymptote under monotone decay;
  otherwise no statement beyond "at the measured distance". Used only to downgrade wording.
- **F8** (verbatim definition). Per realization and plane, from directory A, `rho(X) = RMS |Delta x_T| / sigma_T(X)` over particles crossing
  `X` in both passes, `|Delta x_T|^2 = Delta x2^2 + Delta x3^2`, `sigma_T^2 = (var x2 + var x3)/2`, flux weights `w` (h/8) for both, `sigma_T`
  from the h/8 pass over the same particles; pairs (8,16) (gating reading of F8), (16,32), (8,32) recorded. `m` = the `alpha_TP` `h_max` margin
  (at the `h/16` mean); `Delta X = X_b - X_a`; `S = sigma_T^2(X_b) / (ell Delta X)` (ensemble mean over directory A, h/8). Readings, verbatim
  from F8: `pathwise resolved` if the 95th percentile over realizations of `rho(X_b) <= 0.1 m / S`; `pathwise-limited` if the median
  `rho(X_b) >= sqrt(0.2 m / S)`; `decorrelated` if the median `rho(X_b) >= 0.5`; the growth of the median `rho` with `x1` fitted (OLS of
  `log median rho` on `x1` and on `log(x1 - x0)`, planes `x1 >= x0 + ell`). Recorded beside: the centred per-component variant
  `sqrt(sum w [(d2 - d2bar)^2 + (d3 - d3bar)^2] / (2 sum w)) / sigma_T` and per-realization `rho_r S_r` (the reviews' bias relations:
  correlated `(rho_c + rho_c^2/2) S`, independent `rho_c^2 S / 2`). Recorded, non-gating.
- **F10.** Welch 90 % CI of `alpha(wide) - alpha(narrow)` per transverse estimator against `+-10 %` of the wider arm's mean (no floor);
  reference arms = the first 30 realizations of `s4_tw` and `s4_n16`; width arms are not scaled by the budget factor.
- **F11.** `Delta_P,r = alpha(Pollock) - alpha(consumer)` on the intersection set, uniform weights, per amplitude, arm (production, three narrow
  rungs, twin) and estimator (`alpha_TAM`, `alpha_TP`, fixed `alpha_T`); paired 90 % CIs; differences in differences across rungs (narrow
  1/2 - 1, 1 - 2; twin - production), paired by seed. Par2 fixed-axis `alpha_T` on `W_c` with its `U_c`; one-sided 95 % upper bounds of the
  consumer's fixed-axis `alpha_T` against B&dD's 0.01 / 0.044 (`sigma^2` 1 / 2.25) with B&dD's conditions; all descriptive.
- **E.** Every item of threshold E is produced by the analysis part; also the narrow production rung vs production domain difference (D's
  transverse-domain statement), the per-ladder distribution of per-realization paired differences and the realization with the largest one.

## 6. Pilot, `N`, checkpoint and budget (F9)

- Pilot (20 each): `s{c}_A` (directory A, `--nr 20`), every narrow rung, every twin, the Par2 pilot configs (+ `par2_field_stats`).
- `nr_rule.py`: per amplitude, pilot per-realization paired differences (narrow `D_f`; twin `D_tw`; `h/16 - h/8`) of `alpha_TAM` and
  `alpha_TP`, and pilot production per-realization values. `s_U = s sqrt((n-1)/chi2.ppf(0.20, n-1))` (upper one-sided 80 % limit; `n = 20`).
  Ladder `N` = smallest `N >= 20` with exact paired-TOST power at true difference 0, `P(|Dbar| + t_{0.95,N-1} S/sqrt(N) <= margin)`,
  `Dbar ~ N(0, s_U^2/N)`, `(N-1) S^2/s_U^2 ~ chi2_{N-1}` (one-dimensional quadrature over `S`, range clipped to the support where the inner
  probability is positive), `>= 0.80`; margin `m` (narrow, `h_max`) or `2^1.5 m` (twin); the larger over the two transverse estimators;
  search to 500, else `>500` printed and 500 used. Production `N_R` (B rule) = smallest `N >= 20` with `t_{0.975,N-1} s_U / sqrt(N) <=
  max(0.1 |alpha_pilot|, 2.5e-3)`, the larger over `alpha_TAM`, `alpha_TP`, cap 500. Then `N_narrow`, `N_twin`, `N_hmax`, `N_A`, `N_run`
  (section 1); Par2 `NR = N_R`. Margins use pilot means internally; `nr_rule.py` prints `N` only. Disclosed operating characteristics
  (statistics review): realized power 0.89-0.94 at the chosen `N`, the B rule meets its target with probability 0.91-0.997, internal-pilot TOST
  size about 0.055 at the margin; 0.80 is per test, not for the conjunction behind `physical`.
- **Checkpoint report** (blinded): per amplitude `tau_b*` (F2 pilot record; owner request) and per ladder/arm `N`; measured GPU-seconds per
  realization (runner `timings.json` `seconds.total`; Par2 and `par2_field_stats` wall per realization from file times); projected GPU-hours
  for the remaining realizations (width arm (i) estimated by cell-count scaling of the measured `s4` production rate, stated as an estimate);
  projected wall time on one shared GPU (= GPU-hours, with the pilot's measured wall/GPU-busy ratio stated); projected bytes on the V100 host
  (measured bytes per realization per arm x `N`). No means, sds, margins, verdicts or `alpha`. The other frozen times (`tau_b*(twin)`,
  `tau_b*(Par2)`, `tau_c`) are written to `tau_b_star.json`, not reported. Pilot per-realization estimates go to
  `$SF39_OUT_ROOT/checkpoint/sealed/`.
- **Budget rule.** If the projection exceeds the owner's budget: the `sigma^2 = 4` ladders (narrow, twin, `h_max`) go to 20; then one factor
  `f` (largest multiple of 0.01 in (0, 1]) scales every remaining `N` (`N_R`, ladders, Par2) as `max(20, ceil(f N))`; `N_A`, `N_run` recomputed;
  width arms and the 30 reference realizations unchanged; anything beyond -> owner. Every final `N` is recorded in the bitácora before any
  further run.
- **Anticipated outcome** (F9, pre-registered): within the budget the `sigma^2 = 4` verdicts are expected `inconclusive` (rule D:
  `unresolved`); reported as such, not re-run with tuned settings.

## 7. Reading rules (process)

- **RR0.** No threshold is tuned after any run. Every V100 computation is a detached orchestrator job `scripts/remote --increment SF-39
  run <job>` + `wait`, one job at a time on one GPU chosen by `scripts/remote` (never `--gpu`, never `CUDA_VISIBLE_DEVICES`, no GPU work
  under `exec`), `REMOTE_GPU_WAIT=7200`, `REMOTE_GPU_MIN_FREE_MIB=30000` for parts that launch CUDA work; no `sync` while an SF-39 job is
  `running` or `waiting_gpu`; workers never use the remote. Every part runs to completion whatever intermediate values show (no
  outcome-dependent stopping); infrastructure interruptions (exit 75/70, ssh drop, OOM caused by a colleague's process, host reboot) are
  resumed with identical settings (runner `DONE` files; an interrupted Par2 config is rerun from scratch into a cleaned directory) and
  disclosed; a scientific outcome is never re-run with changed settings. Dry runs are not evidence. GPU-hours per part are reported.
- **RR1 (build).** One clean `v100-release` build of a tree equal to the base `1e40cfe` (mirror `~/MacroFlow3D-SF-39`, synced from a clean
  detached worktree), before any SF-39 script reaches the mirror: `rm -rf build/v100-release && cmake --preset v100-release && mkdir -p
  $HOME/.cache/mf3d-tmp && TMPDIR=$HOME/.cache/mf3d-tmp cmake --build build/v100-release -j` as a detached `run` job; md5 of
  `lagrangian_ensemble`, `lagrangian_medium_ab`, `macroflow3d_pipeline`, `inlet_slab`, `par2_field_stats` in
  `$SF39_OUT_ROOT/build/binaries.md5`; every later part verifies them and refuses on mismatch. Later syncs carry scripts only (`git diff
  --quiet 1e40cfe <head> -- src apps tests cmake CMakeLists.txt CMakePresets.json scripts/regress` holds; `build/` is excluded from rsync); the
  build is never redone.
- **RR2 (regression).** `scripts/regress` set v2 with that build into `~/regress/refs/1e40cfe/out` (reused if its `REGRESS_DONE` exists),
  then into a fresh head directory and `compare.sh` -> last line `REGRESS IDENTICAL` (checklist item "Required regression tests pass", F12).
- **RR3 (checks part, before the pilot).** (i) unit tests of the analysis package on the host; (ii) solve check, seed 4901, one realization per
  narrow size and amplitude (9 runs), plus each twin and width geometry: `solver_b.json` `verdict == converged` (`rel <= 1e-10`, `balance <=
  1e-8`) or `fallback` (recorded); on exit 4 (no `solver_b.json`) the `B_SOLVER` line and exit code are recorded; an unusable narrow size
  stops that ladder (owner), an unusable twin/width geometry is reported (owner); (iii) byte check, seed 4901 at the `s4_A` geometry,
  `--hmax-div 8,16,32` vs `--hmax-div 8`, the F12 file list (`hmax_8/` vs single-pass realization root, `pollock/*` of the two roots,
  `<out>/hmax_8/stats*` vs `<out>/stats*` CSVs, `status.json` key by key with each single-pass key compared at its multi-pass location;
  `DONE`, `timings.json` excluded) -> any difference stops before production; (iv) the reducer on existing SF-41 outputs
  (`~/sf41_campaign/prodgrid/E1..E4`, `evladder/`) as a smoke (not evidence; missing planes/twin tolerated and reported) and the known-answer
  sign table of `docs/experiments/2026-10-27-sf39-pre-activation-review.md` §5.4 from `evladder/`; a failed smoke is a script defect
  (corrective). Check runs are never data.
- **RR4 (blinding).** Until the owner's budget is recorded no person (owner, orchestrator) reads pilot plume statistics: runner `analysis/`,
  `ensemble/`, `stats*/`, per-realization moment CSVs and arrays of pilot arms, Par2 pilot `stats/`, `analysis/`, `ensemble/`, the sealed
  directory. The checkpoint scripts read whatever they need. People read only `DONE`, `status.json`, `solver_b.json`, `flow_stats.json`,
  `timings.json`, `tau_b_star.json`, job logs, part bookkeeping and the checkpoint report. Nothing else from pilot arms is imported into the
  repository before the budget.
- **RR5 (outputs).** Runner and Par2 outputs under `$SF39_OUT_ROOT = $HOME/sf39_campaign` (outside the mirror, which `sync --delete` would
  erase); per-particle arrays stay on the host; reduced per-realization CSVs, logs and `sha256` lists are imported as text (base64 through
  `exec`, sha256 checked on both hosts).
- **RR6 (failure policy).** As the specification: unusable narrow solve -> stop that ladder; byte check fails -> stop before production
  (runner defect, corrective in the owning module); C `disagree` unexplained -> increment stays active; dependency defect -> recorded
  corrective, affected runs redone and superseded ones marked. In addition: a runner exit 4 (unusable solve) of any realization of any arm
  halts that arm (the runner stops; a restart repeats the seed) -> recorded, owner decides; no seed is skipped silently.

## 8. Resolutions (derivable unless marked; no threshold changed)

R1 amplitude by `--eps` with `--sigma2 1` (SF-37/38/41 convention). R2 `--ell` per arm. R3 narrow, twin and width arms use the production
flags (twin and bridge on everywhere: F11 needs both). R4 integer `mmax`. R5 `U` of standalone values over the realizations in the value
(F2 "pooled"), `tau_b*` frozen. R6 exits frozen at the face reached, last recorded `(x2, x3)`. R7 biased weighted variances, renormalized. R8
`U_c` from the series as written, anchors `x0` / `inject_x`. R9 `tau_b*(twin)` normalized by `U_c`. R10 Gate 5 normalized times frozen at the
pilot, `W_c` recomputed with final `U_c`. R11 late Par2 realizations counted, not dropped. R12 ANCOVA model; Par2 realized `sigma^2` by
`par2_field_stats` (SF-38 D17). R13 Welch-Satterthwaite, margin fixed at the Par2 mean. R14 Par2 `dt/2`: SF-38 E cited. **R15 (departure from
F12's letter, for review)** the Par2 extension advances `transport.seed` by `1000 x 20` with `stochastic.seed`: F12 asks for one ensemble of
`N_R` launched in parts; the pipeline derives the injection seed as `transport.seed + 1000 r` with `r` restarting at 0, so advancing only the
field seed would repeat the pilot's injection positions realization by realization. R16 `nr_rule` power formula, margins, `N_run`, rounding,
Par2 `NR = N_R`. R17 BCa details. R18 checkerboard split-half (indicator). R19 F8 verbatim; centred per-component variant recorded beside.
R20 `Delta_half` sets and CI. R21 plume-drift frame over `W`. R22 `exp(-I)` at the evaluated snapshot/plane. R23 rule D evaluated per
ladder (branch 1's "of that ladder"; F5 "applied symmetrically to the `h` and `h_max` ladders"); branch 2 existential over qualifying
differences (plain reading); branch 3 as written. R24 F10 margin `+-10 %` without floor. R25 F10 reference arms run to 30 (`s4_tw`, `s4_n16`)
whatever the ladder `N`; ladders use their first `N`. R26 byte check geometry/seed and `status.json` key mapping. **R27 (departure, for
review)** Par2 outputs under `$SF39_OUT_ROOT/par2/raw/par2_s{c}` (relative `output_dir` as F12 writes it, working directory outside the
mirror; reduced CSVs imported to the artifact's `raw/par2_s{c}`): a directory inside the mirror is erased by the next `sync --delete`. R28
`e_v` percentiles. R29 D11 ("same `t_b` definition") vs F2 (per-arm definitions): F2 governs (later, explicit). **R30 (departure, for
review)** builds run as a detached `run` job with the regress `TMPDIR` step, not the specification's `exec` line (longer than an `exec` step;
the V100 `/tmp` is 2 GB). R31 production counts `N_run` (twin pairs beyond `N_R` run extra production-geometry realizations outside the
estimate; F12's `N_hmax >= N_R` case kept). R32 paired differences use `U` over the pair's seeds; passes and bridge share the production
window. R33 consumer-vs-Pollock on the intersection set. R34 analysis reads times from `track.json`/`pollock.json`. R35 exit 4 policy.
R36 solve check extended to twin and width geometries (checks only).

## 9. Script contracts (scripts and configs only)

Artifact root `A = docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/`; package `A/scripts/sf39/` (Python >= 3.11, numpy
>= 1.26, scipy >= 1.11; must run on the V100 host's 3.11.7 / 1.26.4 / 1.11.4 and locally on newer versions; `random_state` accepted by both
scipy versions); tests `A/scripts/sf39/tests/` (`python3 -m unittest discover -s A/scripts -t A/scripts`), synthetic known answers, seconds.

- **C-GEOM** (`sf39/geometry.py`): the arm table of section 1 and the plane algorithm of section 2; `python3 -m sf39.geometry flags <arm>`
  prints the runner arguments; `planes <list>`; `check` verifies the seven sha256 values. Plane strings are generated at run time and
  checked, never hand-edited.
- **C-RUN** (`scripts/run_sf39.sh <build_dir> <part> [selector]`): parts `checks`, `pilot`, `checkpoint`, `extend`, `width`, `analysis`
  (optional selector `s1|s225|s4|par2` splits long parts into restart-safe sub-jobs); bash 4.2-safe; no GPU selection or GPU-exclusivity
  precheck (the slot and `REMOTE_GPU_MIN_FREE_MIB` govern); verifies binaries md5 (RR1); writes per-part `exit_codes.tsv`, `walls.tsv`,
  `results.txt` (`PART_DONE`), `sha256.txt`; resumes by runner `DONE` and recorded rc; `SF39_DRYRUN=1` prints commands only.
- **C-REDUCE** (`sf39/io.py`, `sf39/reduce.py`): section 3 from the runner files; per-realization reduced CSVs (one row per realization and
  method; columns documented in the module), `sigma_T^2(x1)` and `rho(X)` long-format CSVs.
- **C-PAR2** (`sf39/par2.py`): Par2 series and snapshot readers, Par2 `t_b`, `U_c`, `FIELDSTAT` parser, per-realization `alpha` on a window.
- **C-STATS** (`sf39/stats.py`, `sf39/decide.py`): pure functions on arrays: Student and BCa intervals, equivalence classification, F4, rule D
  and the amplitude label, Welch, ANCOVA, Richardson guard, F7, F8 readings, F10, F11, split-half fraction, decision-point table rows.
- **C-NR** (`scripts/nr_rule.py`): section 6; prints `N` only.
- **C-GLUE** (`sf39/checkpoint.py`, `sf39/analyze.py`): the `checkpoint` part (freeze, sealed pilot estimates, `nr_rule`, report) and the
  `analysis` part (all reductions, criteria A-E, F7, F8, F10, F11, the decision-point table, reduced CSVs for import).

## 10. Decision points (one table in the experiment note; inconclusive ones included)

Per amplitude `c` in {1, 2.25, 4}: primary — Gate 4 label per estimator (`alpha_TAM`, `alpha_TP`) and amplitude label (D); Gate 5 verdict
(C) with both sensitivities. Key secondary — A narrow verdict, F4 production verdict and order contrast, `h_max` verdict, for `alpha_TAM`,
`alpha_TP`, `alpha_LAM`; B achieved half-widths for the three estimators. Estimation — F7 `Delta_half`; F8 readings; F11 `Delta_P` per arm
and differences in differences. Exploratory — F10 (two arms, two estimators). Process — RR3 checks, regression (RR2), byte check.
