# SF-39 — Macrodispersion study at sigma_Y^2 in {1, 2.25, 4} with the Lagrangian consumer

- State: `active`
- Goal: `Medir alpha_L y alpha_T en el dominio largo a sigma^2 en {1, 2.25, 4} con el tracker lagrangiano, con convergencia en lambda/h, en tolerancia y en número de realizaciones evaluada por intervalos de confianza, alpha_L contrastado con Par2 RWPT en la misma configuración y cada crecimiento transversal clasificado como físico, numérico o no resuelto.`
- Depends on: `SF-37, SF-38, SF-41`
- Unlocks: `none`
- Branch: `science/lester-sf39-macrodispersion-study`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 4 + Gate 5`
- Human review: `required`
- Owner: `Claude Opus 5.5 orchestrator (session 2026-10-27)`
- Started: `2026-10-27T16:06Z on base 1e40cfe`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Decision records `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`, items 1, 4, 8,
`docs/decisions/2026-10-23-sf38-activation-decisions.md` (D1, D3, D11, D14, D15: production geometry, the `sigma^2 = 4`
estimator caveat, the Gate 5 protocol, the full-length ladders and the equivalence form of the criteria, frozen before
SF-38 produced any plume statistic or `alpha(t)` series at `sigma^2 = 4`), `docs/decisions/2026-10-24-sf41-sf39-planning-decisions.md`
(E4-E7: drift-aligned transverse moments, ladder `N_R` by an internal pilot rule, a checkpoint after the pilot,
`normalize_variance` on; medium (b), selected by SF-38, through SF-41), `docs/decisions/2026-10-27-remote-single-gpu.md` (one GPU,
the least-loaded, for all remote work; withdraws the two-GPU parts of E6 and F13) and `docs/decisions/2026-10-27-sf39-pre-activation-decisions.md`
(F1-F14, after the pre-activation review `docs/experiments/2026-10-27-sf39-pre-activation-review.md`: executable estimator definitions,
a plane-based conjunctive co-primary for `alpha_T`, a production `h` verdict without the ratio rule, rule D on measured direction,
bounded-value reporting, pathwise drift, power realism, a width check and the method contrast). The study the project exists for:
longitudinal and transverse macrodispersivities of purely advective transport in smooth, locally isotropic Gaussian log-conductivity
fields (Gaussian covariance, `C(r) = sigma^2 exp(-(r/lambda)^2)`, as B&dD eq. 1) **at the travel distances of the long domain, with
the approach to the asymptote characterized, not assumed** (F7), measured with the Lagrangian consumer verified in SF-36/SF-37 on
the long domain built in SF-38/SF-41. It is allowed to establish, per `sigma^2` in {1, 2.25, 4}:

- `alpha_L` and `alpha_T` (B&dD eq. 4-5 window estimator; for `alpha_T` also the plane-based estimator `alpha_TP`, F3) with confidence
  intervals and convergence evidence in `lambda/h`, `h_max` and `N_R`;
- the agreement or disagreement of `alpha_L` with Par2 RWPT in the same configuration (Gate 5);
- the classification of the transverse growth as physical / numerical / unresolved (Rule 1, Gate 4), without
  presupposing `alpha_T` — neither zero (the Lester 2023 claim rests on bounded label fluctuations, an explicit assumption that this
  project found to fail in the periodic cell) nor the B&dD values (Table 2: `alpha_T` 0.01 / 0.044 at `sigma^2` = 1 / 2.25; eq. 7:
  `alpha_TAM ~ sigma^4`), which are context only;
- as estimation (no verdict): the difference between the classical face-flux representation (Pollock bridge) and the consumer
  under refinement (F11), and the transverse-width sensitivity at `sigma^2 = 4` (F10, exploratory).

`sigma^2 = 6.25` is deferred out of SF-39 (F12).

## Preconditions

- `SF-37` and `SF-38` are `done` on the default branch: production `lambda/h` (SF-37 V2, as applied by decision 2026-10-23 D1:
  `sigma^2 = 1` at 10), V1/V2/V4 met at every amplitude, medium selected (SF-38 C: medium (b)), ensemble runner, Par2 configs.
- `SF-41` is `done` on the default branch: lean medium (b) stage, non-periodic-in-`x1` spline, `x`-exit handling (tracker and
  Pollock), `lagrangian_ensemble --medium b` with multi-pass `h_max` and `flow_stats.json`; its production-grid feasibility fixed the
  grids below (no R3 reduction).
- The pre-SF-39 items of the codebase audit (`docs/plans/active/audit-cleanup-2026-10.md`: AUD-00..AUD-06,
  AUD-10..AUD-14, AUD-20; decision 2026-10-26) are merged on the default branch: checked realization writers,
  ordered context stream, V100 build provenance, and the regression set v2 (`scripts/regress`, decision A4).
- The pre-activation governance PR (decision 2026-10-27, F1-F14) is merged on the default branch; its merge commit is the
  increment base.

## In scope

- **Production ensemble** (D1, F2, F3, F12). Medium (b) through the SF-41 stage (`lagrangian_ensemble --medium b`); grids:
  `sigma^2 = 1`: 2048x256x256 at `lambda/h = 10` (204.8 x 25.6 x 25.6 `lambda`); `sigma^2 = 2.25`: 3328x384x384 at 16 (208 x 24 x 24
  `lambda`); `sigma^2 = 4`: 3328x384x384 at 32 (104 x 12 x 12 `lambda`). Injection plane `x0 = 5 lambda`; window `[0.1, 0.9]^2`
  (0.8 `L_T`), `M = 100`, flux weights, `N_p = 1e4` fixed; SF-18 with `normalize_variance` on (E7), raw variance and applied scale
  recorded per realization; `--tol 1e-8 --fused-splines --single-launch --twin on --pollock-bridge on`; `--times auto`; `--planes` =
  the F3 list (union of `x0 + m lambda`, `x0 + L/4`, `X_a = x0 + L/2`, `X_b = L1 - 5 lambda`, `L = L1 - 10 lambda`; sorted,
  deduplicated within `1e-12`, printed `%.17g`). Seeds from 5001: realizations `r = 0 .. N_hmax - 1` in directory A with
  `--hmax-div 8,16,32` (their `hmax_8` pass is production data), `r = N_hmax .. N_R - 1` in directory B with `--hmax-div 8` and
  `--seed0 5001 + N_hmax` (same command line otherwise; pooled by seed). Statuses, exits, non-ok counts by type and backflow
  encounters recorded per plane and time; non-ok particles are reported, never silently dropped.
- **Estimators** (F2, F3, E4).
  - Time window: `t_b,r` = first arrival of an `ok` particle at `L1 - 5 lambda`; `tau_b* = U_prod min_r t_b,r / ell_prod` over the 20
    production-pilot realizations, frozen before any further run; each run uses `W = [0.75, 1] x tau_b* ell_run / U_run` (`ell` =
    `lambda` in box units of that run; `U_run` = mean `qbar_r` of `flow_stats.json` over the run's realizations). Per realization
    `alpha_k,AM,r = [Var_k,r(t_hi) - Var_k,r(t_lo)] / (2 ell U (t_hi - t_lo))`, `Var` interpolated linearly between snapshots, both
    endpoints over the particles `ok` at the first snapshot at or after `t_hi`; ensemble value = mean over realizations.
  - `alpha_TAM` (Gate 4): transverse components in the frame of the realization's interior mean Darcy flux (`flow_stats.json`):
    `e_L = q_r/|q_r|`, `e_T1`, `e_T2` by Gram-Schmidt from `e2`, `e3`; `alpha_TAM` = mean of the two aligned components; the fixed-axis
    components (B&dD eq. 4), the per-realization tilt and the fixed-minus-aligned difference are reported.
  - `alpha_TP` (Gate 4, conjunctive co-primary): `[sigma_T^2(X_b) - sigma_T^2(X_a)] / (2 ell (X_b - X_a))`, `sigma_T^2 = (var x2 + var
    x3)/2` flux-weighted over the particles crossing both planes (excluded fraction recorded).
  - `alpha_LAM`: fixed axes (A, B, C). At `sigma^2 = 4` it carries the pre-asymptotic caveat of D3 (104 `lambda` domain; B&dD: ~20.5
    advection times to the longitudinal asymptote at `sigma^2 = 4`; SF-37 D8 shadow: `lambda/h` 64 for that estimator).
  - Intervals: Student `t` on the realization mean (`N - 1` d.o.f.); realization BCa bootstrap (`B = 5000`, seed fixed in the
    activation row) reported beside it.
- **Pilot and `N_R`** (F9). Pilot: 20 realizations per amplitude (production, directory A), per narrow rung, per coarse twin, and the
  first 20 Par2 realizations per amplitude. Production `N_R` = the smallest count giving a 95 % CI half-width (the level of the original target, unchanged) `<= max(0.1 |alpha|, 2.5e-3)`
  for `alpha_TAM` and for `alpha_TP` (the larger), computed with the upper one-sided 80 % confidence limit of the pilot sd, capped at 500;
  `alpha_LAM` reports its achieved 95 % half-width against `<= 0.05 alpha_LAM` (classification intervals are 90 %, F6). Ladder `N` (E5 as refined by F9): `nr_rule.py` prints, from the
  pilot sd of the paired differences of `alpha_TAM` and `alpha_TP` only (upper 80 % confidence limit; no mean, no margin printed), the
  smallest `N` with power `>= 0.80` for the equivalence test of A when the true difference is 0; the coarse twin with margin `2^1.5 m`
  (F4); `N_hmax` likewise from the `h/8` vs `h/16` differences. The final analysis uses all realizations including the pilot.
- **Ladders at the full production length** (D14, D14b, D15, F4; paired: same seeds, same continuum; Pollock bridge on every rung):
  - (i) **narrow-section `lambda/h` ladder**, rungs {1/2, 1, 2} x production `lambda/h`: `sigma^2 = 1`: `lambda/h` 5 / 10 / 20 at
    204.8 x 12.8 x 12.8 `lambda` (1024x64x64, 2048x128x128, 4096x256x256); `sigma^2 = 2.25`: 8 / 16 / 32 at 208 x 8 x 8 `lambda`
    (1664x64x64, 3328x128x128, 6656x256x256); `sigma^2 = 4`: 16 / 32 / 64 at 104 x 4 x 4 `lambda` (1664x64x64, 3328x128x128,
    6656x256x256). Before the ladder, one realization per narrow size verifies the medium (b) solve: `||r|| / ||r_0|| <= 1e-10` and flux
    balance `<= 1e-8` (or the SF-38 C0 fallback `<= 1e-8` with balance met, recorded; otherwise stop, owner);
  - (ii) **coarse twin** at 1/2 x production `lambda/h` on the production domain: `sigma^2 = 1`: 1024x128x128 (`lambda/h = 5`, below
    SF-37's range); `sigma^2` 2.25 and 4: 1664x192x192 (`lambda/h` 8 and 16);
  - (iii) **`h_max` ladder** {`h/8`, `h/16`, `h/32`} from production directory A (same stage, same seeds); pathwise drift `rho(X)` (F8)
    from the per-pass `planes_x23.npy`;
  - Richardson extrapolation (order 2) of the production error only if the observed order of the ensemble mean over the narrow rungs is
    in [1.5, 2.5] and the sequence is monotone; otherwise `|Delta Q|` is reported as a bound; never applied as a correction to `alpha`.
- **Gate 5 baseline** (D10, D11, F2, F12). Par2 RWPT with copies of the SF-38 configs (same physical domain: 2048x256x256 at
  `lambda/h` 10; 2496x288x288 at 12 and 24) with `stochastic.seed: 9001`, `NR` = the amplitude's production `N_R`, output under the
  artifact, launched in parts (pilot 20, then the extension with the seed advanced); everything else unchanged (`cg_rtol 1e-6`,
  recorded). Comparison: the Lagrangian twin series (uniform weights over the same windowed particles) and Par2 (uniform over the full
  plane at `x = 5 lambda`) over the travel distance `L1 - 10 lambda`; normalized time `tau = U_c t / lambda` with `U_c` the ensemble
  centroid slope of each arm's uniform-weight series over `[0, t_b,min]` (FV `qbar` of the Lagrangian arm as a sensitivity);
  `tau_c = min(tau_b*(twin), tau_b*(Par2))` (Par2: last snapshot with no particle at `x >= Lx - 5 lambda`, minimum over its first 20
  realizations); common window `[0.75 tau_c, tau_c]`, `Var` interpolated in each arm's per-realization series (Par2:
  `stats/realization_%04d_timeseries.csv`, every 10 steps); Par2's realized `sigma^2` per realization recorded and used in an ANCOVA
  sensitivity; Par2's in-pipeline `alpha` (normalized by the nominal `J`) not used; the Pollock bridge series on the same realizations
  attributes any disagreement (representation vs generator, medium, time stepping). Stated differences: SF-18 periodic normalized
  generator vs Par2 random-spectral, non-periodic (a seam with the periodic `y/z` conditions), not normalized; DP5(4) vs Euler; window
  vs full-plane injection.
- **Method contrast** (F11, estimation, no verdict). `Delta_P = alpha(Pollock) - alpha(consumer)` per amplitude, rung and transverse
  estimator on the uniform-weight series of both arms, paired 90 % CIs, and its change across `lambda/h` (difference in differences);
  stated hypothesis (Lester et al. 2023 §5.2): `Delta_P > 0`, shrinking under refinement while the consumer's ladder converges. Par2
  `alpha_T` (fixed axes, normalized by `U_c`) descriptive. One-sided upper bounds of the consumer's fixed-axis `alpha_T` against B&dD's
  0.01 / 0.044 (`sigma^2` 1 / 2.25): hypothesis-generating, with B&dD's conditions stated.
- **Width check** (F10, exploratory, non-gating, `sigma^2 = 4`, `lambda/h = 16`, unpaired, 30 realizations per arm, `+-10 %`
  equivalence reading): (i) 24 `lambda` (1664x384x384, `--window 0.3,0.7,0.3,0.7` = 9.6 `lambda`, seeds 7101-7130) vs the coarse twin
  (12 `lambda`, 9.6 `lambda` source, first 30 realizations); (ii) 12 `lambda` (1664x192x192, `--window 0.36666666666666664,
  0.63333333333333336,0.36666666666666664,0.63333333333333336` = 3.2 `lambda`, seeds 7201-7230) vs the narrow 1/2 rung (4 `lambda`, 3.2
  `lambda` source, first 30 realizations). Both transverse estimators reported.
- **Compute** (E6 checkpoint; decision 2026-10-27 remote single GPU). One GPU, the least-loaded, chosen by `scripts/remote run` (single
  host-wide project slot): SF-39 jobs run one at a time from one synced build, never pin `--gpu`, set `REMOTE_GPU_WAIT=7200`, and parts
  that launch CUDA work set `REMOTE_GPU_MIN_FREE_MIB=30000` (SF-41 measured peak 28855 MiB at 3328x384^2 plus a margin), so they wait
  instead of starting on a GPU a colleague has partly filled; no `sync` while any SF-39 job is `running` or `waiting_gpu`. Long parts may
  be split into restart-safe sub-jobs (runner `DONE` files). After the pilot the checkpoint (F9) reports `N`, GPU-seconds per realization,
  projected GPU-hours, projected wall time on one shared GPU and bytes on the host; the owner fixes the budget before any further run
  (rough estimate, decision 2026-10-24: ~19 GPU-h typical, ~91 worst case, plus the Pollock bridge on the ladders and < 1 GPU-h for the
  width check; on one GPU the wall time is about the GPU-hours, longer when a colleague shares the GPU).
- Experiment note with the Gate 5 comparison statement (covariance model, `sigma^2`, boundary conditions, injection protocol, tracking
  method, estimation procedure) and the single table of every decision point (primary: the Gate 4 classification and the Gate 5
  verdict per amplitude; key secondary: the `alpha_LAM` ladder verdicts and B; estimation: F7, F8, F11; exploratory: F10).

## Out of scope

- Finite Péclet / local dispersion and `D_T(Pe)` (later phase); pseudo-symplectic cross-validation (deferred);
  any change to the tracker, the stage, the media, the runner or the Par2 code (a defect found here returns to the owning increment as
  a recorded corrective); `sigma^2 = 6.25` (deferred: needs its own resolution ladder, F12); covector statistics (the runner has no
  option; the divergence diagnostic is the Lagrangian `e_v`, F12).

## Files and symbols

- Scripts and configs only (runner from SF-38/SF-41), under `docs/experiments/artifacts/<date>-sf39-macrodispersion-study/`:
  `scripts/run_sf39.sh` (parts `checks`, `pilot`, `checkpoint`, `extend`, `width`, `analysis`), `scripts/nr_rule.py`, the
  analysis package (planes list, `t_b` and `tau_b*`, window estimator, aligned frame, `alpha_TP`, intervals, TOST, F4 production verdict,
  rule D/F6 classifier, F7 decay, F8 `rho`, F11 contrast, Gate 5 with `U_c` and ANCOVA, decision-point table), `configs/par2_s{1,225,4}.yaml`,
  `preregistration.md`; `logs/`, `raw/` (reduced per-realization CSVs; per-particle arrays stay on the V100 host), `analysis/`;
  `docs/experiments/<date>-sf39-macrodispersion-study.md`.

## Implementation specification

1. Fix in the activation row, before any run, verbatim: the base commit; medium, grids, `lambda/h`, `normalize_variance`, seeds (5001
   upward, width 7101/7201, Par2 9001), `N_p`, window, the planes list per geometry, times, `tol`, `h_max` list and runner flags; the
   estimator definitions (F2, F3, E4), the interval method and bootstrap seed (F6), the production `h` verdict (F4), rule D (F5, F6),
   F7, the F8 thresholds, the `N_R` rules, cap and checkpoint content (F9), F10, F11, the single-GPU rule and `REMOTE_GPU_MIN_FREE_MIB`, the anticipated
   outcome at `sigma^2 = 4` (F9) and thresholds A-E.
2. Checks on the V100 at the activation base, before the pilot: build; `scripts/regress` (set v2) reproduces its references; one
   realization per narrow size verifies the medium (b) solve; one realization run with `--hmax-div 8,16,32` and with `--hmax-div 8` is
   compared byte for byte (F12 file list; `DONE`, `timings.json` excluded); the analysis scripts reproduce known answers on synthetic
   inputs and run on existing SF-41 outputs.
3. Pilot (one GPU) -> `tau_b*` per amplitude and Par2 `tau_b*` frozen -> `nr_rule.py` and the checkpoint report (F9: `N`, GPU-s per
   realization, GPU-h, projected wall time, bytes; no means, margins or verdicts) -> owner budget -> every `N` recorded in the bitácora before further runs.
4. Extension (one GPU): production directories A and B, narrow ladders, coarse twins, Par2 extension, width arms.
5. Analysis -> classification -> experiment note -> PR `awaiting_review`.
6. Every run records statuses (including the SF-41 exit statuses), non-ok counts by type, backflow counters, Lagrangian `e_v`
   percentiles and `flow_stats.json`; moments over the stated particle sets with the counts.

## Expected numerical effect

None on existing outputs. Results not predicted, except the anticipated outcome of F9 (inconclusive `sigma^2 = 4` verdicts for power
reasons), stated so that it is not read as a failure.

## Validation commands

```bash
bash scripts/hooks/check-lester-increments.sh
A=docs/experiments/artifacts/<date>-sf39-macrodispersion-study/scripts
B=build/v100-release
scripts/remote --increment SF-39 sync
scripts/remote --increment SF-39 exec -- "cmake --preset v100-release && cmake --build $B -j"
G="REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000"   # one GPU, least-loaded, chosen by scripts/remote (never --gpu)
env $G scripts/remote --increment SF-39 run sf39-checks -- "bash $A/run_sf39.sh $B checks"
scripts/remote --increment SF-39 wait sf39-checks
env $G scripts/remote --increment SF-39 run sf39-pilot -- "bash $A/run_sf39.sh $B pilot"
scripts/remote --increment SF-39 wait sf39-pilot
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-39 run sf39-checkpoint -- "bash $A/run_sf39.sh $B checkpoint"
scripts/remote --increment SF-39 wait sf39-checkpoint
# STOP: owner budget (F9); every N recorded in the bitácora before any further run
env $G scripts/remote --increment SF-39 run sf39-extend -- "bash $A/run_sf39.sh $B extend"
scripts/remote --increment SF-39 wait sf39-extend
env $G scripts/remote --increment SF-39 run sf39-width -- "bash $A/run_sf39.sh $B width"
scripts/remote --increment SF-39 wait sf39-width
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-39 run sf39-analysis -- "bash $A/run_sf39.sh $B analysis"
scripts/remote --increment SF-39 wait sf39-analysis
```

## Acceptance thresholds

Pre-registered verbatim in the activation bitácora row before any run; no threshold tuned afterwards. **Date of knowledge (F1):** the
forms and margins frozen by decision 2026-10-23 (D15) were fixed before SF-38 produced any plume statistic at `sigma^2 = 4`; E4 and E5
(2026-10-24) and F1-F14 (2026-10-27) were fixed after SF-38's A/B statistics (seeds 4001-4010, Pollock plume, 8 `lambda` width at
`sigma^2 = 4`) and its post-hoc readings P1-P6, and (F1-F14) after SF-41's F-ladder plane variances, before any SF-39 realization.

- **A. Convergence at the production length** (D14, D15, E5, F4, F9). Equivalence test on paired realizations (90 % CI, Student `t`,
  `N - 1` d.o.f.), per amplitude, for `alpha_TAM` (drift-aligned, E4), `alpha_TP` (F3) and `alpha_LAM` (fixed axes); margins half of the
  CI targets: `alpha_TAM` and `alpha_TP` `+-max(0.05 |alpha|, 1.25e-3)` (units of `lambda`), `alpha_LAM` `+-0.025 alpha_LAM`, with `alpha`
  the ensemble mean of the finer setting. `converged` if the CI of the paired difference lies inside the margin, `not converged` if it
  lies entirely outside, `inconclusive` otherwise; trend recorded.
  - **`h`:** narrow ladder, production rung vs 2x production rung (margin at `min(m_narrow, m_prod)`). **Production verdict (F4):**
    `converged` iff the narrow verdict is `converged`, the 90 % CI of the production-domain paired difference production minus coarse
    twin lies within `+-2^1.5 m`, and the narrow ladder does not reject a monotone order `>= 1.5` (the 90 % CI of `s (2^1.5 D_f - D_c)`
    does not lie entirely above 0; `D_c` = production minus 1/2 rung, `D_f` = 2x minus production, `s` = sign of the mean `D_c`);
    `not converged` iff the narrow verdict is `not converged`; `inconclusive` otherwise. Stated assumption: the production-domain error
    is represented by the narrow ladder or by `|D_tw| / 2^1.5`.
  - **`h_max`:** `h/8` vs `h/16` at the production geometry, same rule (`h/32` recorded).
- **B. Ensemble size.** The production 95 % CI half-widths of `alpha_TAM` and `alpha_TP` meet `<= max(0.1 |alpha|, 2.5e-3)`, or the
  cap / budget of F9 is reported with the achieved half-widths; `alpha_LAM` reports its achieved half-width against `<= 0.05 alpha_LAM`.
- **C. Gate 5** (D11, D15b, D20, F2). `alpha_LAM` of the Lagrangian twin series vs `alpha_LAM` of Par2, over the common window and
  travel distance of "In scope", both normalized by their own `U_c`: `agree` if the 90 % CI of the difference (Welch, unpaired) lies within
  `+-0.10 alpha_LAM(Par2)`; `disagree` if it lies entirely outside, which must be explained with the Pollock bridge (representation vs
  generator, medium, time stepping) before the increment closes; `inconclusive` otherwise (reported with the achieved CI). Sensitivities:
  FV-`qbar` normalization; ANCOVA on Par2's realized `sigma^2`. At `sigma^2 = 4` the comparison carries the pre-asymptotic caveat (D3).
- **D. Gate 4 classification of `alpha_T`** (F5, F6). Per transverse estimator (`alpha_TAM`, `alpha_TP`), with directions taken from the
  signed paired difference finer minus coarser ("above / below 0" = its 90 % CI entirely above / below 0; refinement differences: `D_f` and
  `D_tw` for `h`, `h/16 - h/8` for `h_max`), applied symmetrically to both ladders, in this order:
  1. `unresolved (grid-limited, lower bound)` if a ladder verdict (production `h` of A, or `h_max`) is not `converged` and a refinement
     difference of that ladder lies above 0 while the other's point estimate is not below 0;
  2. otherwise `numerical` if a verdict is `not converged`, a refinement difference lies below 0 (the other's point estimate not above
     0) and the finer-setting 90 % CI contains 0 or has its upper limit `<= 2.5e-3 lambda`;
  3. otherwise `unresolved (grid-limited, upper bound)` if a verdict is not `converged` and a refinement difference lies below 0;
  4. otherwise `physical` if both verdicts are `converged`: `physical (positive)` if the production 90 % CI lower limit > 0;
     `physical (negative)` if its upper limit < 0; `physical (bounded)` if its upper limit `U <= 2.5e-3 lambda` (reported as `alpha_T <= U`);
     otherwise `unresolved (imprecise)`;
  5. otherwise `unresolved`.
  **Amplitude classification:** the common label when both estimators agree (`physical` with the same sub-label requires both);
  otherwise `unresolved (estimator-discordant)`, the per-estimator labels listed. Every value is qualified "within this model, medium and
  domain, at the measured distance"; no asymptotic statement beyond F7. The narrow-section (production rung) vs production-domain
  difference and the width check (F10) are recorded and stated with the classification (transverse-domain effect).
- **E. Recorded.** `alpha_L(t)`, `alpha_T(t)` series (flux-weighted, twin and Pollock bridge; aligned and fixed axes); full
  `sigma_T^2(x1)`; `t_b,r`, `tau_b*`; non-ok fractions by type, exit counts, excluded fractions, `active` counts; Lagrangian `e_v`
  percentiles; `alpha_T` with weights `w exp(-I)`; `alpha_T` in the plume-drift frame beside the `q_r` frame; per-realization tilt and the
  fixed-minus-aligned difference of `alpha_TAM`; `flow_stats.json` per realization (fraction of `x`-faces with `u <= 0`, cell-centre speed
  percentiles: the flow-reversal / stagnation context of Lester's stagnation-free hypothesis, `docs/theory/lester-2023-key-claims.md` §4);
  per ladder the distribution of the per-realization paired differences and the realization with the largest one (error tail,
  `docs/experiments/2026-10-23-resolution-vs-variance.md` §5 item 3); `Delta_half` (F7); `rho(X)` and its readings (F8); the split-half
  particle component of each `alpha`'s variance (F12); the method contrast (F11); the width check (F10); Par2 `dt/2` sensitivity (SF-38
  E) and realized `sigma^2`; B&dD Table 1-2 values beside the results as context, with the comparison conditions stated (fixed axes,
  `5 lambda` injection buffer, transverse width 25.6 `lambda` against 12 `lambda` at `sigma^2 = 4`, `N_R = 500`, Euler stepping on linear
  interpolation).

## Regression surface

- None in production code (scripts and configs only). The checklist item "Required regression tests pass" means: `scripts/regress`
  (set v2) reproduces its references with the activation-base V100 build used for every SF-39 run (F12).

## Failure and rollback policy

- A `not converged` or `inconclusive` at an amplitude -> that amplitude's `alpha_T` is classified by rule D; it is reported as such, not
  re-run with tuned settings. At `sigma^2 = 4` this is the anticipated outcome (F9).
- A narrow size whose medium (b) solve is not usable (`rel <= 1e-10`, or the C0 fallback `<= 1e-8` with balance `<= 1e-8`) -> stop
  before that ladder; owner.
- The `hmax_8` vs single-pass byte comparison fails -> stop before production; the cause is a runner defect (corrective in SF-41's
  module with its own evidence).
- C `disagree` unexplained -> the increment stays active; owner.
- A defect found in a dependency -> recorded corrective in the owning module with its own evidence; affected runs
  are redone and the superseded ones marked as such.
- Cost or disk beyond the budget fixed at the checkpoint -> the F9 allocation rule (`sigma^2 = 4` ladders to the floor of 20 first,
  then a common factor, floor 20); anything beyond -> owner decision before production.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Implementation matches the scope and contains no unrelated changes.
- [ ] Targeted validation passes and its evidence is recorded.
- [ ] Required regression tests pass.
- [ ] Scientific or engineering findings are appended to the bitácora.
- [ ] Required human review is recorded.
- [ ] PR and commit identifiers are recorded.
- [ ] The master checklist entry is checked in this branch and `check-lester-increments.sh` passes.
<!-- completion-checklist:end -->

## Advancement rule

No increment is unlocked directly; acceptance enables the specification of the finite-Péclet phase (dashboard
"Later phases").

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-17T19:30Z | not started | Specification created on the SF-35 closure PR (#53) after the owner decision of 2026-10-17 (`docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`). B&dD protocol values checked against the local copy (eq. 1, 4, 5, 7; `N_p = 1e4`, `N_R = 500`, `lambda/Delta = 10`, injection `0.8 L_T x 0.8 L_T` at `>= 5 lambda`; Table 2). | Depends on SF-37 and SF-38 `done` on `master`. | Activate only when the checker reports it READY; the activation row fixes the items of "Implementation specification" 1 before any run. |
| 2026-10-23T02:36Z | not started | Amended on the SF-38 activation governance PR (branch `docs/lester-sf38-activation-governance`) by the owner decisions of 2026-10-23 (`docs/decisions/2026-10-23-sf38-activation-decisions.md`, D1, D3, D11, D14, D14b, D15, D15a-c, D18, D20), frozen before SF-38 produces any plume statistic or `alpha(t)` series at `sigma^2 = 4`: production grids per amplitude; injection window `[0.1, 0.9]^2`, `M = 100`; `alpha_LAM` at `sigma^2 = 4` computed with the pre-asymptotic caveat; ladders at the full production length (narrow-section `lambda/h` ladder {1/2, 1, 2} x production with a representativeness transfer rule, coarse twin on the production domain, guarded Richardson); the tolerance ladder replaced by an `h_max` ladder; criteria A and C rewritten as equivalence tests (A: half of the CI targets; C: `+-10 %` of `alpha_LAM`) and rule D extended with the branch "`alpha_T` increases under refinement" (`unresolved (grid-limited)`, lower bound, never `physical`), with the branch precedence of D18; the Gate 5 margin relative to Par2's `alpha_LAM` (D20); Gate 5 protocol (twin series, common window and travel distance, `t_b` truncation, Pollock bridge). The Goal is unchanged; its "convergencia en tolerancia" is evaluated with the `h_max` ladder. The previous text is in git history at `ab78d7c`. | Docs only; no run. | Activate only when the checker reports it READY; the activation row fixes the items of "Implementation specification" 1 before any run. |
| 2026-10-24T06:00Z | not started | Amended on the docs-only governance PR (branch `docs/lester-sf41-sf39-planning`) by the owner decisions of 2026-10-24 (`docs/decisions/2026-10-24-sf41-sf39-planning-decisions.md`) after SF-38 selected medium (b): Depends on adds SF-41; medium (b) through the SF-41 stage, injection at `5 lambda`, measurement up to `L1 - 5 lambda`; transverse moments in the frame of each realization's mean Darcy flux for Gate 4, fixed axes reported (E4); ladder `N_R` by an internal-pilot rule on the sd of the paired differences, at least 20 (E5, refines D15a); two GPUs with an owner checkpoint after the pilot (E6, refines D16 for SF-39); `normalize_variance` on (E7); narrow-size solve check for medium (b); `h_max` ladder by SF-41 multi-pass; recorded flow-reversal statistics, exit counts, tilt, per-realization ladder tails and the B&dD comparison conditions. Margins, branch order and the Gate 5 rule frozen by D15 are unchanged. The previous text is in git history at `d8ffa7c`. | Docs only; no run. | Activate only when the checker reports it READY (SF-37, SF-38, SF-41 `done`); the activation row fixes the items of "Implementation specification" 1 before any run. |
| 2026-10-26T16:30Z | not started | Precondition added on the codebase-audit governance PR (branch `chore/aud-00-audit-governance`) by the owner decisions of 2026-10-26 (`docs/decisions/2026-10-26-audit-cleanup.md`): SF-39 activates after the pre-SF-39 audit items are merged (writers checked before `DONE`, blocking context stream, V100 build provenance, regression set v2 replacing the SF-27 trio, PETSc/PSPTA/`legacy/` retired). No change to Goal, scope, thresholds or the frozen rules. | Decision record A1-A10; plan `docs/plans/active/audit-cleanup-2026-10.md` | Activate SF-39 once AUD-00..06, AUD-10..14 and AUD-20 are merged. |
| 2026-10-27T15:00Z | not started | Amended on the docs-only pre-activation governance PR (branch `docs/lester-sf39-pre-activation-governance`) by decision 2026-10-27 (`docs/decisions/2026-10-27-sf39-pre-activation-decisions.md`, F1-F14; owner delegation, ratified by merge) after the pre-activation review `docs/experiments/2026-10-27-sf39-pre-activation-review.md` (five reader reports, four adversarial reviews, two red-team reviews of the decisions, own calculations): executable estimator definitions (per-realization window estimator, `tau_b*` frozen in normalized time from the pilot, normalization velocities, particle sets, Gate 5 window) (F2); plane-based `alpha_TP` as a conjunctive co-primary for Gate 4 (F3); the production `h` verdict without the ratio rule (F4: narrow verdict + twin bound `2^1.5 m` + order contrast); rule D on measured direction with symmetric branches and agreement of both estimators (F5); `physical (positive / negative / bounded)` with Student `t` intervals replacing "zero within resolution" (F6); no asymptotic claim, `Delta_half` (F7); pathwise drift `rho` from the `h_max` passes (F8); `N` from the transverse estimators with the upper 80 % sd limit, blinded checkpoint, budget rule, anticipated inconclusive outcome at `sigma^2 = 4` (F9); exploratory width check (F10); method contrast as estimation (F11); covectors dropped, `sigma^2 = 6.25` deferred, Par2 configs, `h_max` ladder from production directory A, byte check, disk (F12); two-GPU rule in the runbook (F13); theory notes and Gate 4 corrected (F14). Date of knowledge stated in the thresholds header (F1). Goal, D15 margins and CI targets, D1 geometry and medium unchanged. The previous text is in git history at `782c9d9`. | Docs only; no run; no `alpha` claimed. | Activate when the checker reports SF-39 READY with this PR merged; the activation row copies the thresholds verbatim and fixes the base commit (this PR's merge) before any run. |
| 2026-10-27T16:00Z | not started | Amended on the docs-only/tooling PR (branch `docs/remote-single-gpu`) by the owner decision of 2026-10-27 (`docs/decisions/2026-10-27-remote-single-gpu.md`): the V100 host is shared with colleagues, so the project uses one GPU at a time, the least-loaded, chosen by `scripts/remote run` (single host-wide slot, `nvidia-smi` sampling, `REMOTE_GPU_MIN_FREE_MIB`). Compute paragraph, `run_sf39.sh` parts (`pilot`/`extend` without a GPU argument), implementation items 1, 3, 4 and validation commands rewritten (one job at a time, never `--gpu`, `REMOTE_GPU_MIN_FREE_MIB=30000` for parts with CUDA work, `REMOTE_GPU_WAIT=7200`); the checkpoint adds projected wall time. Withdraws the two-GPU parts of E6 and F13. Goal, thresholds, estimators, margins and the scientific design unchanged. | Docs and `scripts/remote` arbitration only; no run. | Activate when the checker reports SF-39 READY with this PR merged; the activation row fixes the base commit (this PR's merge). |
| 2026-10-27T16:06Z | `active` on `master=1e40cfe` (base `1e40cfe9cb5132174bd167d7f21ba0c77910623e` = merge of PR #82 over PR #81 `adec2080`; checker OK ready=SF-34 SF-39, nonterminal=none; activating keeps nonterminal = 1); delivery branch `science/lester-sf39-macrodispersion-study`; remote `scripts/remote --increment SF-39` (one job at a time, one GPU chosen by `scripts/remote`, never `--gpu`/`CUDA_VISIBLE_DEVICES`, no GPU work under `exec`, `REMOTE_GPU_WAIT=7200`, `REMOTE_GPU_MIN_FREE_MIB=30000` for CUDA parts) | **Activation; pre-registration before any build, test or run of SF-39.** Orchestrator UNDERSTAND (runtime record `.claude/orchestration/SF-39/`; binding reading in dashboard order plus decisions F1-F14 and remote single GPU; code reconnaissance of `lagrangian_ensemble` (CLI, units, output layout, statuses 7/8 vs Pollock 8/9, restart signature), the Par2 pipeline (seeds `stochastic.seed + r`, `transport.seed + 1000 r`, output layout, `par2_field_stats`), `scripts/regress`, `scripts/remote`; three adversarial reviews of the draft: governance fidelity, executability against the code, statistics). Items of Implementation specification 1 fixed verbatim in `docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/preregistration.md` (this commit; sha256 `88fe1fa67eab28a209778cf5a240e270ab4ed1e5f44777c3d0f7cecc89ad7cb7`): base commit; medium (b); the arm table (grids, `lambda/h`, `--ell`, `--eps` with `--sigma2 1`, seeds 5001 upward for production/narrow/twin, 7101/7201 width, Par2 9001 with extension 9021, check seed 4901; production counts `N_R`, `N_A`, `N_run`), `normalize_variance` on, `N_p = 1e4`, window `[0.1, 0.9]^2`, the seven F3 plane lists with their sha256, `--times auto`, `tol 1e-8`, `--hmax-div 8,16,32` (directory A) / `8`, runner flags; the estimator definitions (F2, F3, E4) in executable form; Student `t` intervals and BCa (`B = 5000`, `numpy.random.default_rng(20261027)`); F4; rule D (F5, F6; evaluated per ladder); F7; F8 thresholds (verbatim); the `N_R` rules, cap 500 and checkpoint content (F9); F10; F11; the single-GPU rule and `REMOTE_GPU_MIN_FREE_MIB=30000`; the anticipated outcome at `sigma^2 = 4` (F9); reading rules RR0-RR6 (blinded checkpoint RR4); resolutions R1-R36 (derivable; departures from a governance letter forced by the code, listed for the human review: R15 Par2 extension advances `transport.seed` with `stochastic.seed` to reproduce one launch, R27 Par2 outputs outside the mirror, R30 builds as detached `run` jobs with `TMPDIR`); script contracts C-GEOM, C-RUN, C-REDUCE, C-PAR2, C-STATS, C-NR, C-GLUE. No threshold, margin, CI level, branch, Goal or design value changed; F5 branch 3 is applied as written (the statistics review's remark that it lacks the other branches' point-estimate guard is recorded for the review, not acted on). Date of knowledge: no SF-39 realization exists. (The preceding row's `16:00Z` stamp is later than this row; PR #82 merged at about 15:20Z.) Acceptance thresholds, verbatim from this file (line breaks joined; pipe characters escaped as `\|` for the table): Pre-registered verbatim in the activation bitácora row before any run; no threshold tuned afterwards. **Date of knowledge (F1):** the forms and margins frozen by decision 2026-10-23 (D15) were fixed before SF-38 produced any plume statistic at `sigma^2 = 4`; E4 and E5 (2026-10-24) and F1-F14 (2026-10-27) were fixed after SF-38's A/B statistics (seeds 4001-4010, Pollock plume, 8 `lambda` width at `sigma^2 = 4`) and its post-hoc readings P1-P6, and (F1-F14) after SF-41's F-ladder plane variances, before any SF-39 realization. - **A. Convergence at the production length** (D14, D15, E5, F4, F9). Equivalence test on paired realizations (90 % CI, Student `t`, `N - 1` d.o.f.), per amplitude, for `alpha_TAM` (drift-aligned, E4), `alpha_TP` (F3) and `alpha_LAM` (fixed axes); margins half of the CI targets: `alpha_TAM` and `alpha_TP` `+-max(0.05 \|alpha\|, 1.25e-3)` (units of `lambda`), `alpha_LAM` `+-0.025 alpha_LAM`, with `alpha` the ensemble mean of the finer setting. `converged` if the CI of the paired difference lies inside the margin, `not converged` if it lies entirely outside, `inconclusive` otherwise; trend recorded. - **`h`:** narrow ladder, production rung vs 2x production rung (margin at `min(m_narrow, m_prod)`). **Production verdict (F4):** `converged` iff the narrow verdict is `converged`, the 90 % CI of the production-domain paired difference production minus coarse twin lies within `+-2^1.5 m`, and the narrow ladder does not reject a monotone order `>= 1.5` (the 90 % CI of `s (2^1.5 D_f - D_c)` does not lie entirely above 0; `D_c` = production minus 1/2 rung, `D_f` = 2x minus production, `s` = sign of the mean `D_c`); `not converged` iff the narrow verdict is `not converged`; `inconclusive` otherwise. Stated assumption: the production-domain error is represented by the narrow ladder or by `\|D_tw\| / 2^1.5`. - **`h_max`:** `h/8` vs `h/16` at the production geometry, same rule (`h/32` recorded). - **B. Ensemble size.** The production 95 % CI half-widths of `alpha_TAM` and `alpha_TP` meet `<= max(0.1 \|alpha\|, 2.5e-3)`, or the cap / budget of F9 is reported with the achieved half-widths; `alpha_LAM` reports its achieved half-width against `<= 0.05 alpha_LAM`. - **C. Gate 5** (D11, D15b, D20, F2). `alpha_LAM` of the Lagrangian twin series vs `alpha_LAM` of Par2, over the common window and travel distance of "In scope", both normalized by their own `U_c`: `agree` if the 90 % CI of the difference (Welch, unpaired) lies within `+-0.10 alpha_LAM(Par2)`; `disagree` if it lies entirely outside, which must be explained with the Pollock bridge (representation vs generator, medium, time stepping) before the increment closes; `inconclusive` otherwise (reported with the achieved CI). Sensitivities: FV-`qbar` normalization; ANCOVA on Par2's realized `sigma^2`. At `sigma^2 = 4` the comparison carries the pre-asymptotic caveat (D3). - **D. Gate 4 classification of `alpha_T`** (F5, F6). Per transverse estimator (`alpha_TAM`, `alpha_TP`), with directions taken from the signed paired difference finer minus coarser ("above / below 0" = its 90 % CI entirely above / below 0; refinement differences: `D_f` and `D_tw` for `h`, `h/16 - h/8` for `h_max`), applied symmetrically to both ladders, in this order: 1. `unresolved (grid-limited, lower bound)` if a ladder verdict (production `h` of A, or `h_max`) is not `converged` and a refinement difference of that ladder lies above 0 while the other's point estimate is not below 0; 2. otherwise `numerical` if a verdict is `not converged`, a refinement difference lies below 0 (the other's point estimate not above 0) and the finer-setting 90 % CI contains 0 or has its upper limit `<= 2.5e-3 lambda`; 3. otherwise `unresolved (grid-limited, upper bound)` if a verdict is not `converged` and a refinement difference lies below 0; 4. otherwise `physical` if both verdicts are `converged`: `physical (positive)` if the production 90 % CI lower limit > 0; `physical (negative)` if its upper limit < 0; `physical (bounded)` if its upper limit `U <= 2.5e-3 lambda` (reported as `alpha_T <= U`); otherwise `unresolved (imprecise)`; 5. otherwise `unresolved`. **Amplitude classification:** the common label when both estimators agree (`physical` with the same sub-label requires both); otherwise `unresolved (estimator-discordant)`, the per-estimator labels listed. Every value is qualified "within this model, medium and domain, at the measured distance"; no asymptotic statement beyond F7. The narrow-section (production rung) vs production-domain difference and the width check (F10) are recorded and stated with the classification (transverse-domain effect). - **E. Recorded.** `alpha_L(t)`, `alpha_T(t)` series (flux-weighted, twin and Pollock bridge; aligned and fixed axes); full `sigma_T^2(x1)`; `t_b,r`, `tau_b*`; non-ok fractions by type, exit counts, excluded fractions, `active` counts; Lagrangian `e_v` percentiles; `alpha_T` with weights `w exp(-I)`; `alpha_T` in the plume-drift frame beside the `q_r` frame; per-realization tilt and the fixed-minus-aligned difference of `alpha_TAM`; `flow_stats.json` per realization (fraction of `x`-faces with `u <= 0`, cell-centre speed percentiles: the flow-reversal / stagnation context of Lester's stagnation-free hypothesis, `docs/theory/lester-2023-key-claims.md` §4); per ladder the distribution of the per-realization paired differences and the realization with the largest one (error tail, `docs/experiments/2026-10-23-resolution-vs-variance.md` §5 item 3); `Delta_half` (F7); `rho(X)` and its readings (F8); the split-half particle component of each `alpha`'s variance (F12); the method contrast (F11); the width check (F10); Par2 `dt/2` sensitivity (SF-38 E) and realized `sigma^2`; B&dD Table 1-2 values beside the results as context, with the comparison conditions stated (fixed axes, `5 lambda` injection buffer, transverse width 25.6 `lambda` against 12 `lambda` at `sigma^2 = 4`, `N_R = 500`, Euler stepping on linear interpolation). | Base commit and preregistration sha256 above; class human review (autonomy policy: code that computes and reports `alpha`). | Intra-increment DAG (`.claude/orchestration/SF-39/dag.json`): script nodes W1-W6 in isolated worktrees, orchestrator audit, one integration; V100 build at the base and regression references (R0) meanwhile; then `checks`, pilot, blinded checkpoint (F9) and **stop for the owner's budget** before any further run. |
| 2026-10-28T13:50Z | `active`; delivery branch at `07fe746e` (I1a `dd5799c1`, I1b `b0be775d` + `07fe746e`: W1-W6b-c integrated, orchestrator audits positive) + this commit | **Checks, pilot and blinded checkpoint (RR4: bookkeeping only, no plume statistic read by owner or orchestrator); final `N` recorded before any further run.** R0: base build job `sf39-build` on `1e40cfe` (5 binaries, md5 list sha256 `b1985f91d49d51f9...` gated in every part) and regression refs `sf39-baserefs`. R1 `sf39-checks` on `dd5799c1`: 173 unit tests OK; GEOMETRY_CHECK OK (seven plane sha256); regression set v2 head vs refs REGRESS IDENTICAL; solve checks seed 4901 14/14 converged; byte check `s4_A` geometry `--hmax-div 8,16,32` vs `8` BYTECHECK IDENTICAL (41 items); smoke on SF-41 outputs 28/28 runs, reducer vs runner moments max rel. diff 1.02e-13 (smoke, not evidence); sign table equal. R2 pilot on `b0be775d`, one GPU at a time chosen by `scripts/remote`: `sf39-pilot-s1` 4149 s, `-s225` 8661 s, `-s4` 9044 s, `-par2` 4690 s; all 15 Lagrangian arms rc 0 with 20/20 realizations, no halted arm; Par2 `par2_series` 20/20 and `par2_field_stats` 20/20 per amplitude. R3 `sf39-checkpoint` (89 s): report sha256 `2fc104ddbf563af704efd3a408f900a58b5a7c542570066d657f08c944389f71`, `tau_b_star.json` sha256 `d643047d0691af6d723875054d6afccf1adae800d6c5f64a63ccc9b3b6324355`; `tau_b*` s1 125.553719086, s225 93.3341223831, s4 23.2231267808; projected 6.83762 GPU-h for the remaining realizations, wall 6.84448 h on one shared GPU (pilot wall/GPU-busy 1.001), 133.47 GB final bytes (recomputed independently by the orchestrator from the reported per-realization values). | Owner budget (2026-10-28, to the orchestrator): "Tenés uso libre del servidor; hasta varios días de cómputo si se necesita; recordá que es una gpu a la vez." The projection is within the budget, so the budget rule is not applied (`f = 1`) and the final `N` are those printed by `nr_rule`: s1 `N_narrow` 20, `N_twin` 20, `N_hmax` 20, `N_R` 50, `N_A` 20, `N_run` 50, Par2 `NR` 50; s225 20, 26, 20, 62, 20, 62, 62; s4 20, 20, 20, 71, 20, 71, 71 (no value reached the cap). Width arms `s4_w24`, `s4_w12` 30 each; F10 reference arms `s4_tw`, `s4_n16` run to 30 (R25). `configs/budget.json` (sha256 `fc583081c2c8d6e0bf9bac23c0888c37391720e52650b713f1cd135c1ff05d17`) and `configs/par2_s{1,225,4}_ext.yaml` (`python3 -m sf39.par2 mkext`, NR 30/42/51, R15 seeds) in this commit. | Sync this commit to the mirror; `extend` (s1, s225, s4, par2) then `width`, one remote job at a time (`REMOTE_GPU_WAIT=7200`, `REMOTE_GPU_MIN_FREE_MIB=30000`); then `analysis`. |
