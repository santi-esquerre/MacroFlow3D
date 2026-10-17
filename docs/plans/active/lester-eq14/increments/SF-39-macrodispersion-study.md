# SF-39 — Macrodispersion study at sigma_Y^2 in {1, 2.25, 4} with the Lagrangian consumer

- State: `pending`
- Goal: `Medir alpha_L y alpha_T en el dominio largo a sigma^2 en {1, 2.25, 4} con el tracker lagrangiano, con convergencia en lambda/h, en tolerancia y en número de realizaciones evaluada por intervalos de confianza, alpha_L contrastado con Par2 RWPT en la misma configuración y cada crecimiento transversal clasificado como físico, numérico o no resuelto.`
- Depends on: `SF-37, SF-38`
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

Decision record `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`, items 1, 4, 8. The study the project
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

- `SF-37` and `SF-38` are `done` on the default branch: production `lambda/h` (SF-37 V2), V1/V2/V4 met at every
  amplitude, V5's long-range regime, medium selected (SF-38 C), ensemble runner, Par2 configs; if SF-38 selected
  medium (b), the non-periodic spline increment is `done`.

## In scope

- **Production ensemble.** Medium from SF-38; grid at the SF-37 production `lambda/h` (2048x256x256 at
  `lambda/h = 10`, or the owner-approved reduction); injection plane (medium (a): `x1 = 0`; medium (b):
  `x1 = 5 lambda`), window `0.8 L_T x 0.8 L_T`, flux weights, `N_p = 1e4` (B&dD) unless the pilot shows `N_p`
  limits the CI; SF-18 with `normalize_variance` chosen from SF-38 G and stated in the activation row; seeds from
  5001 upward.
- **Estimator.** `alpha_k(t) = (1/(2 lambda |v|)) dVar_k/dt` averaged over realizations (B&dD eq. 4;
  `compute_macrodispersion`), time normalized by `lambda/|v|`; `alpha_kAM` = mean of `alpha_k(t)` over
  `[0.75 t_b, t_b]` (eq. 5), `t_b` = first breakthrough (first particle reaching the end of the measurement
  window). `alpha_T` = mean of the two transverse components (B&dD report `alpha_T1 = alpha_T2`); both are
  reported. Secondary: `sigma_T^2(x1)` per plane. CI by bootstrap over realizations (the unit).
- **Pilot and `N_R`.** Pilot `N_R = 20` per amplitude; `N_R` = the smallest count giving a 95 % CI half-width
  `<= max(0.1 |alpha_TAM|, 2.5e-3)` for `alpha_TAM` and `<= 0.05 alpha_LAM` for `alpha_LAM`, extrapolated from the
  pilot SE, capped at 500 (B&dD); if the cap binds, the achieved half-width is reported.
- **Long-range ladders in reduced domains** (paired: same seeds, same continuum): `32 lambda x 6.4 lambda x
  6.4 lambda` at `lambda/h` in {10, 20, 40}; `tol` in {1e-7, 1e-8, 1e-9} at the production `lambda/h`;
  `N_R = 20` per amplitude per ladder.
- **Gate 5 baseline.** Par2 RWPT with the SF-38 configs at production `N_R` (or the achieved CI), per amplitude.
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
   window, planes, the CI target, the bootstrap settings (realization bootstrap, `B = 5000`, generator seed fixed)
   and the classification rule below.
2. Run pilot -> fix `N_R` (recorded before the production runs) -> production -> ladders -> Par2.
3. Every run records statuses, non-ok fractions, backflow counters, Lagrangian `e_v` percentiles and covector
   statistics when enabled; non-ok particles are reported, never silently dropped (moments over `ok` particles with
   the counts).

## Expected numerical effect

None on existing outputs. Results not predicted.

## Validation commands

```bash
bash scripts/hooks/check-lester-increments.sh
A=docs/experiments/artifacts/<date>-sf39-macrodispersion-study/scripts
for part in pilot production ladder-h ladder-tol par2; do scripts/remote --increment SF-39 run sf39-$part -- "bash $A/run_sf39.sh build/v100-release $part"; scripts/remote --increment SF-39 wait sf39-$part; done
```

## Acceptance thresholds

Pre-registered verbatim in the activation bitácora row before any run; no threshold tuned afterwards.

- **A. Convergence in the reduced domains.** For `alpha_LAM` and `alpha_TAM` per amplitude: `converged` in `h` if
  the paired mean difference between `lambda/h` 20 and 40 is within `2 SE`; `converged` in `tol` if 1e-8 vs 1e-9 is
  within `2 SE`; otherwise `not converged` with the trend recorded.
- **B. Ensemble size.** The production CI half-widths meet the targets of "In scope" or the cap is reported.
- **C. Gate 5.** `alpha_LAM` (O6) vs `alpha_LAM` (Par2): `agree` if the difference is within
  `2 sqrt(SE_O6^2 + SE_Par2^2)`; otherwise `disagree`, which must be explained (generator, representation,
  injection) before the increment closes.
- **D. Gate 4 classification of `alpha_T`**, per amplitude:
  - `numerical` if the `h` or `tol` ladder (A) is `not converged` with `|alpha_TAM|` decreasing under refinement;
  - `physical` if both ladders are `converged` and the production CI either excludes 0 (positive) or includes 0
    with a half-width meeting its target (zero within resolution) — always qualified "within this model, medium
    and domain";
  - `unresolved` otherwise.
  The reduced-domain vs long-domain difference of `alpha_TAM` is recorded (domain effect), and the classification
  states it.
- **E. Recorded.** `alpha_L(t)`, `alpha_T(t)` series; `t_b`; non-ok fractions; Lagrangian `e_v`; covector
  statistics; B&dD Table 1-2 values beside the results as context, with the comparison conditions stated.

## Regression surface

- None in production code (scripts and configs only).

## Failure and rollback policy

- A `not converged` at an amplitude -> that amplitude's `alpha_T` is `numerical` or `unresolved` by rule D; it is
  reported as such, not re-run with tuned settings.
- C `disagree` unexplained -> the increment stays active; owner.
- A defect found in a dependency -> recorded corrective in the owning module with its own evidence; affected runs
  are redone and the superseded ones marked as such.
- Cost overrun (projected `N_R` beyond the GPU budget agreed at activation) -> owner decision before production.

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
