# SF-39 — Macrodispersion study at sigma_Y^2 in {1, 2.25, 4} with the Lagrangian consumer

- State: `pending`
- Goal: `Medir alpha_L y alpha_T en el dominio largo a sigma^2 en {1, 2.25, 4} con el tracker lagrangiano, con convergencia en lambda/h, en tolerancia y en número de realizaciones evaluada por intervalos de confianza, alpha_L contrastado con Par2 RWPT en la misma configuración y cada crecimiento transversal clasificado como físico, numérico o no resuelto.`
- Depends on: `SF-37, SF-38, SF-41`
- Unlocks: `none`
- Branch: `science/lester-sf39-macrodispersion-study`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 4 + Gate 5`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Decision records `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`, items 1, 4, 8, and
`docs/decisions/2026-10-23-sf38-activation-decisions.md` (D1, D3, D11, D14, D15: production geometry, the `sigma^2 = 4`
estimator caveat, the Gate 5 protocol, the full-length ladders and the equivalence form of the criteria, frozen before
SF-38 produced any plume statistic or `alpha(t)` series at `sigma^2 = 4`) and `docs/decisions/2026-10-24-sf41-sf39-planning-decisions.md`
(E4-E7: drift-aligned transverse moments, ladder `N_R` by an internal pilot rule, two GPUs with a checkpoint after the pilot,
`normalize_variance` on; medium (b), selected by SF-38, through SF-41). The study the project
exists for: asymptotic longitudinal and transverse macrodispersivities of purely advective transport in smooth,
locally isotropic Gaussian log-conductivity fields (Gaussian covariance, `C(r) = sigma^2 exp(-(r/lambda)^2)`, as
B&dD eq. 1), measured with the Lagrangian consumer verified in SF-36/SF-37 on the long domain built in SF-38. It
is allowed to establish, per `sigma^2` in {1, 2.25, 4}:

- `alpha_L` and `alpha_T` (B&dD eq. 4-5 estimator) with confidence intervals and convergence evidence in
  `lambda/h`, `tol` and `N_R`;
- the agreement or disagreement of `alpha_L` with Par2 RWPT in the same configuration (Gate 5);
- the classification of the transverse growth as physical / numerical / unresolved (Rule 1, Gate 4), without
  presupposing `alpha_T` — neither zero (the Lester 2023 claim rests on bounded label fluctuations, which this
  project found to fail in the periodic cell) nor the B&dD values (Table 2: `alpha_T` 0.01 / 0.044 at
  `sigma^2` = 1 / 2.25; eq. 7: `alpha_TAM ~ sigma^4`), which are context only.

`sigma^2 = 6.25` is exploratory (no classification required).

## Preconditions

- `SF-37` and `SF-38` are `done` on the default branch: production `lambda/h` (SF-37 V2, as applied by decision 2026-10-23 D1:
  `sigma^2 = 1` at 10), V1/V2/V4 met at every
  amplitude, V5's long-range regime, medium selected (SF-38 C: medium (b)), ensemble runner, Par2 configs.
- `SF-41` is `done` on the default branch: lean medium (b) stage, non-periodic-in-`x1` spline, `x`-exit handling (tracker and
  Pollock), `lagrangian_ensemble --medium b` with multi-pass `h_max` and `flow_stats.json`; its production-grid feasibility (or the
  owner-approved R3 reduction) fixes the grids below.

## In scope

- **Production ensemble.** Medium (b) (SF-38 C) through the SF-41 stage (`lagrangian_ensemble --medium b`); production grids of
  decision 2026-10-23 D1 (`sigma^2 = 1`: 2048x256x256 at `lambda/h = 10`, 204.8 x 25.6 x 25.6 `lambda`; `sigma^2 = 2.25`:
  3328x384x384 at 16, 208 x 24 x 24 `lambda`; `sigma^2 = 4`: 3328x384x384 at 32, 104 x 12 x 12 `lambda`; or the owner-approved
  reduction of SF-41 E / R3); injection plane `x1 = 5 lambda` (B&dD: at least `5 lambda` from the inlet), planes and the measurement
  window up to `L1 - 5 lambda`, window `[0.1, 0.9]^2` (0.8 `L_T`), `M = 100`, flux weights, `N_p = 1e4` (B&dD) unless the pilot shows
  `N_p` limits the CI; SF-18 with `normalize_variance` on (decision 2026-10-24 E7), raw variance and applied scale recorded per
  realization; seeds from 5001 upward; exited and other non-ok particles counted per plane and time (SF-41 statuses).
- **Estimator.** `alpha_k(t) = (1/(2 lambda |v|)) dVar_k/dt` averaged over realizations (B&dD eq. 4;
  `compute_macrodispersion`), time normalized by `lambda/|v|`; `alpha_kAM` = mean of `alpha_k(t)` over
  `[0.75 t_b, t_b]` (eq. 5), `t_b` = first breakthrough (first particle reaching the end of the measurement
  window). `alpha_T` = mean of the two transverse components (B&dD report `alpha_T1 = alpha_T2`); both are
  reported. Secondary: `sigma_T^2(x1)` per plane. CI by bootstrap over realizations (the unit). **Transverse frame (decision
  2026-10-24 E4):** in medium (b) the mean flow of each realization is tilted (SF-38 post-hoc P4); the transverse time moments used
  for Gate 4 are taken in the frame of the realization's interior mean Darcy flux `q_r` (from the flow solve, SF-41 `flow_stats.json`,
  not from the plume): `e_L = q_r / |q_r|`, `e_T1`, `e_T2` by Gram-Schmidt from `e2`, `e3`; `alpha_T` (aligned) is the mean of the two
  aligned components. The fixed-axis moments of B&dD eq. 4 are computed from the same particles and reported beside them (B&dD
  context; Gate 5 uses the fixed-axis `alpha_L`, as Par2); the per-realization tilt and the fixed-minus-aligned difference of
  `alpha_TAM` are recorded. At `sigma^2 = 4`
  `alpha_LAM` is computed with the same window and reported with the explicit caveat that the regime is pre-asymptotic
  (104 `lambda` domain; B&dD: ~20.5 advection times to the longitudinal asymptote at `sigma^2 = 4`; SF-37 D8 shadow:
  `lambda/h` 64 for that estimator) (decision 2026-10-23 D3).
- **Pilot and `N_R`.** Pilot `N_R = 20` per amplitude; `N_R` = the smallest count giving a 95 % CI half-width
  `<= max(0.1 |alpha_TAM|, 2.5e-3)` for `alpha_TAM` and `<= 0.05 alpha_LAM` for `alpha_LAM`, extrapolated from the
  pilot SE, capped at 500 (B&dD); if the cap binds, the achieved half-width is reported.
- **Ladders at the full production length** (decision 2026-10-23 D14, D14b, D15; paired: same seeds, same continuum;
  `N_R` per ladder and amplitude by the internal-pilot rule of decision 2026-10-24 E5: 20 realizations first; the pre-registered
  script `nr_rule.py` prints, from the sd of the paired differences of `alpha_TAM` and `alpha_LAM` only (the mean difference is not
  printed), the smallest `N` with power `>= 0.80` for the equivalence test of A when the true difference is 0 (two one-sided `t`
  tests, 90 % CI, margins of A evaluated at the ensemble mean of the finer setting), capped by the compute agreed at the pilot
  checkpoint; the realizations are then extended to that `N` and the final analysis uses all of them):
  - (i) **narrow-section `lambda/h` ladder** at the production length, rungs {1/2, 1, 2} x production `lambda/h`:
    `sigma^2 = 1`: `lambda/h` 5 / 10 / 20 at 204.8 x 12.8 x 12.8 `lambda` (1024x64x64, 2048x128x128, 4096x256x256);
    `sigma^2 = 2.25`: 8 / 16 / 32 at 208 x 8 x 8 `lambda` (1664x64x64, 3328x128x128, 6656x256x256);
    `sigma^2 = 4`: 16 / 32 / 64 at 104 x 4 x 4 `lambda` (1664x64x64, 3328x128x128, 6656x256x256). Before the ladder, one
    realization per narrow size verifies the medium (b) solve: `||r|| / ||r_0|| <= 1e-10` and flux balance `<= 1e-8` (the SF-38 C0
    fallback `<= 1e-8` with balance met is used and recorded; otherwise stop, owner);
  - (ii) **coarse twin** at 1/2 x production `lambda/h` on the production domain: `sigma^2 = 1`: 1024x128x128
    (`lambda/h = 5`, below SF-37's range, used under the guard below); `sigma^2` 2.25 and 4: 1664x192x192 (`lambda/h`
    8 and 16);
  - Richardson extrapolation (order 2) of the production error only if the observed order of the ensemble mean over the
    narrow rungs is in [1.5, 2.5] and the sequence is monotone; otherwise `|Delta Q|` is reported as a bound; never applied as
    a correction to `alpha`;
  - **integration-step ladder** (replaces the tolerance ladder, which is vacuous at the `h/8` cap, SF-37 RR15; the Goal's
    "convergencia en tolerancia" is evaluated with it): `h_max` in {`h/8`, `h/16`, `h/32`} at the production geometry,
    re-tracking the same stage (same flow) of each realization (SF-41 multi-pass `--hmax-div 8,16,32`); `N_R` by the E5 rule.
- **Gate 5 baseline** (decision 2026-10-23 D11). Par2 RWPT with the SF-38 configs (same physical domain:
  2048x256x256 at `lambda/h` 10; 2496x288x288 at 12 and 24) at production `N_R` (or the achieved CI), per amplitude.
  Comparison protocol: the Lagrangian twin series (uniform weights over the same windowed particles) and Par2 (uniform over
  the full plane at `x = 5 lambda`) over the same travel distance `L1 - 10 lambda` from the injection plane, with the same
  `t_b` definition (first particle reaching the end of that distance; Par2 from snapshots), in one common time window fixed
  in the activation row, truncated at `t_b`, time and dispersivities normalized by the measured mean velocity of each run;
  Par2's realized `sigma^2` per realization recorded; the Pollock bridge series (SF-38 runner) on the same realizations is
  used to attribute any disagreement (representation vs generator, medium, time stepping). Both arms now have fixed heads in `x` and
  periodic `y`/`z` (B&dD's boundary conditions); the generators still differ (SF-18 periodic, normalized; Par2 random-spectral,
  not normalized: realized `sigma^2` recorded).
- **Compute (decision 2026-10-24 E6).** Two GPUs: independent detached jobs on GPU 0 and GPU 1
  (`scripts/remote --increment SF-39 run <job> --gpu 0|1`, host-wide GPU locks) from one synced build, distinct output directories,
  no `sync` while any SF-39 job runs, a no-compute-process precheck per job. After the pilot (production pilot and the first 20
  realizations of every ladder) the projected GPU-hours of the remaining work are reported to the owner, who decides before
  production (rough estimate in the decision record: ~19 GPU-h typical, ~91 GPU-h worst case).
- `sigma^2 = 6.25` exploratory runs after 4 is classified. Experiment note with the Gate 5 comparison statement
  (covariance model, `sigma^2`, boundary conditions, injection protocol, tracking method, estimation procedure).

## Out of scope

- Finite Péclet / local dispersion and `D_T(Pe)` (later phase); pseudo-symplectic cross-validation (deferred);
  any change to the tracker, the stage or the media (a defect found here returns to the owning increment as a
  recorded corrective).

## Files and symbols

- Scripts and configs only (runner from SF-38): `docs/experiments/<date>-sf39-macrodispersion-study.md` and
  artifact (`scripts/`, `logs/`, `raw/`, `analysis/`), study configs under the artifact.

## Implementation specification

1. Fix in the activation row, before any run: medium, grid, `lambda/h`, `normalize_variance`, seeds, `N_p`,
   window, planes, the CI target, the bootstrap settings (realization bootstrap, `B = 5000`, generator seed fixed),
   the common Gate 5 time window, the transverse-frame rule (E4), the ladder `N_R` rule and its script (E5), the GPU
   assignment (E6) and the classification rule below.
2. Run pilot (production pilot and 20 realizations per ladder rung) -> fix the production `N_R` and the ladder `N_R`
   (recorded before further runs) -> owner checkpoint on the projected GPU-hours -> production -> ladders to their `N_R`
   (narrow, twin, `h_max`) -> Par2.
3. Every run records statuses (including the SF-41 exit statuses), non-ok fractions, backflow counters, Lagrangian `e_v`
   percentiles, covector statistics when enabled and `flow_stats.json`; non-ok particles are reported, never silently dropped
   (moments over `ok` particles with the counts).

## Expected numerical effect

None on existing outputs. Results not predicted.

## Validation commands

```bash
bash scripts/hooks/check-lester-increments.sh
A=docs/experiments/artifacts/<date>-sf39-macrodispersion-study/scripts
# two GPUs (decision 2026-10-24 E6): independent parts on --gpu 0 and --gpu 1, no sync while any job runs
scripts/remote --increment SF-39 run sf39-pilot-g0 --gpu 0 -- "bash $A/run_sf39.sh build/v100-release pilot 0"
scripts/remote --increment SF-39 run sf39-pilot-g1 --gpu 1 -- "bash $A/run_sf39.sh build/v100-release pilot 1"
scripts/remote --increment SF-39 wait sf39-pilot-g0; scripts/remote --increment SF-39 wait sf39-pilot-g1
for part in production ladder-h ladder-twin ladder-hmax par2; do scripts/remote --increment SF-39 run sf39-$part --gpu auto -- "bash $A/run_sf39.sh build/v100-release $part"; scripts/remote --increment SF-39 wait sf39-$part; done
```

## Acceptance thresholds

Pre-registered verbatim in the activation bitácora row before any run; no threshold tuned afterwards. The forms and
margins below were frozen by decision 2026-10-23 (D15), before SF-38 produced any plume statistic at `sigma^2 = 4`.

- **A. Convergence at the production length** (decision 2026-10-23 D14, D15; `N_R` by decision 2026-10-24 E5). Equivalence test
  on paired realizations (`N_R` of the E5 rule, at least 20; 90 % confidence interval, Student `t`, `N_R - 1` degrees of freedom), per
  amplitude, for `alpha_TAM` (drift-aligned, E4) and
  `alpha_LAM`; margins half of the CI targets: `alpha_TAM` `+-max(0.05 |alpha_TAM|, 1.25e-3)` (in units of `lambda`),
  `alpha_LAM` `+-0.025 alpha_LAM`, with `alpha` the ensemble mean of the finer setting.
  - **`h`:** the narrow-section ladder, production rung vs 2 x production rung: `converged` if the CI of the paired
    difference lies inside the margin, `not converged` if it lies entirely outside, `inconclusive` otherwise; trend
    recorded. **Transfer to the production domain:** the narrow verdict applies to the production domain iff the relative
    1/2 -> 1 differences of the ensemble means in the narrow ladder and in the production coarse twin have the same sign and
    a ratio in `[1/2, 2]`; otherwise the production `h` verdict is `inconclusive` and the twin bound `|Delta Q|` (or its
    Richardson estimate when its guard holds) is reported.
  - **`h_max`:** `h/8` vs `h/16` at the production geometry, same rule (`h/32` recorded).
- **B. Ensemble size.** The production CI half-widths meet the targets of "In scope" or the cap is reported.
- **C. Gate 5** (decision 2026-10-23 D11, D15b). `alpha_LAM` of the Lagrangian twin series vs `alpha_LAM` of Par2, over
  the common time window and travel distance of "In scope": `agree` if the 90 % CI of the difference (Welch, unpaired)
  lies within `+-0.10 alpha_LAM(Par2)` (the baseline is the reference, D20); `disagree` if it lies entirely outside, which must be explained with the Pollock
  bridge (representation vs generator, medium, time stepping) before the increment closes; `inconclusive` otherwise
  (reported with the achieved CI). At `sigma^2 = 4` the comparison carries the pre-asymptotic caveat of the estimator.
- **D. Gate 4 classification of `alpha_T`** (drift-aligned moments, decision 2026-10-24 E4), per amplitude, applying the branches
  in this order (decision 2026-10-23 D18; "increasing" / "decreasing" refer to the point estimate of the ensemble mean of `|alpha_TAM|` from the coarser to
  the finer setting):
  1. `unresolved (grid-limited)` if the `h` or `h_max` verdict is `not converged` or `inconclusive` with `|alpha_TAM|`
    increasing under refinement: the production value is reported as a lower bound and is never classified `physical`
    (coarse grids under-disperse `sigma_T^2`, `docs/experiments/2026-10-23-resolution-vs-variance.md` §3.7);
  2. otherwise `numerical` if the `h` or `h_max` verdict (A) is `not converged` with `|alpha_TAM|` decreasing under refinement;
  3. otherwise `physical` if both verdicts are `converged` (after transfer) and the production CI either excludes 0 (positive) or
    includes 0 with a half-width meeting its target (zero within resolution) — always qualified "within this model,
    medium and domain";
  4. otherwise `unresolved`.
  The narrow-section (production rung) vs production-domain difference of `alpha_TAM` is recorded (transverse-domain
  effect), and the classification states it.
- **E. Recorded.** `alpha_L(t)`, `alpha_T(t)` series (flux-weighted, twin and Pollock bridge; aligned and fixed axes); `t_b`;
  non-ok fractions and exit counts; Lagrangian `e_v`; covector statistics; per-realization tilt and the fixed-minus-aligned
  difference of `alpha_TAM`; `flow_stats.json` per realization (fraction of `x`-faces with `u <= 0`, cell-centre speed percentiles:
  the flow-reversal / stagnation context of Lester's stagnation-free hypothesis, theory note §4); per ladder the distribution of the
  per-realization paired differences and the realization with the largest one (error tail, `docs/experiments/2026-10-23-resolution-vs-variance.md`
  §5.3); Par2 `dt/2` sensitivity (SF-38 E) and realized `sigma^2`; B&dD Table 1-2 values beside the results as context, with the
  comparison conditions stated (fixed axes, `5 lambda` injection buffer, transverse width 25.6 `lambda` against 12 `lambda` at `sigma^2 = 4`,
  `N_R = 500`).

## Regression surface

- None in production code (scripts and configs only).

## Failure and rollback policy

- A `not converged` or `inconclusive` at an amplitude -> that amplitude's `alpha_T` is classified by rule D
  (`numerical`, `unresolved (grid-limited)` or `unresolved`); it is reported as such, not re-run with tuned settings.
- A narrow size whose SF-19 solve does not reach `rtol = 1e-10` -> stop before that ladder; owner.
- C `disagree` unexplained -> the increment stays active; owner.
- A defect found in a dependency -> recorded corrective in the owning module with its own evidence; affected runs
  are redone and the superseded ones marked as such.
- Cost overrun (projected `N_R` beyond the GPU budget agreed at activation) -> owner decision before production; the pilot
  checkpoint of decision 2026-10-24 E6 is mandatory.

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
