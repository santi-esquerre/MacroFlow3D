# SF-38 — Long domain: non-cubic stage, medium A/B evaluation, ensemble runner and Par2 baseline

- State: `pending`
- Goal: `Construir la infraestructura del dominio largo (stage no cúbico SF-18/SF-19/SF-28 en 2048x256x256 con pico de memoria de dispositivo medido <= 30000 MiB, runner de ensemble con checkpoint por realización y baseline Par2 RWPT con la misma configuración) y elegir con un criterio pre-registrado entre el medio affine-periódico y el de Dirichlet en x.`
- Depends on: `SF-36, SF-40`
- Unlocks: `SF-39`
- Branch: `science/lester-sf38-long-domain-infrastructure`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 5 (baseline preparation)`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Decision record `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`, items 7-9, stop rules R3, R4. The study
(SF-39) needs a long, transversally periodic domain (Beaudoin & de Dreuzy 2013 §2 [6]-[7]: `L_T/lambda = 25.6`,
`lambda/Delta = 10`, fixed heads on the faces orthogonal to the mean flow, periodic transverse faces; local copy
`docs/references/Dreuzy-2013-WRR.pdf` p. 2490-2491) on one V100-32GB. SF-38 builds and measures that
infrastructure and is allowed to establish:

- (a) the non-cubic stage (field, flow, splines, injection) is correct on controls and fits the production grid
  (2048x256x256 = 1.34e8 cells; the probes measured 25925 MiB at the same cell count);
- (b) the choice between medium (a) SF-19 affine-periodic on the long box and medium (b) `solve_head` with
  Dirichlet in `x` and periodic `y`/`z`, by a criterion fixed before running;
- (c) a reproducible, restartable ensemble runner whose per-realization outputs feed `compute_macrodispersion`;
- (d) the Par2 RWPT baseline configuration of the study runs (pilot ensemble), for Gate 5 in SF-39.

No `alpha` estimate is reported as a result (SF-39).

## Preconditions

- `SF-36` is `done` on the default branch (forward tracker, P2, stage options). SF-37 may run in parallel; the
  production `lambda/h` it fixes is an input of SF-39, not of SF-38 (the infrastructure is parametric in
  `(n_x, n_T, lambda/h)`).
- `SF-40` is `done` on the default branch (lean non-cubic `Grid3D` stage at memory level S+,
  `docs/plans/active/lester-eq14/increments/SF-40-lean-lagrangian-stage.md`).

## In scope

- **Long-box stage.** Box `[0, L1] x [0, 1]^2`, `h = 1/n_T` isotropic (SF-19 requires `dx = dy = dz`),
  `L1 = n_x h`, transverse period 1 (so `InletLabels` on the `n_T x n_T` inlet still applies when `v1 > 0`).
  SF-18 on the box (even extents; `corr_length` in box units, `ell = (lambda/h) h`), SF-19 affine-periodic with
  `qbar = e1`, SF-28 splines on the box (non-cubic supported), flux-weighted injection on a plane `x1 = X0` over a
  transverse window, memory-lean lifetimes (P2 pattern). Production geometry 2048x256x256, `lambda/h = 10`
  (204.8 `lambda` x 25.6 `lambda` x 25.6 `lambda`).
  Note (2026-10-22): the lean non-cubic `Grid3D` stage (`LongBoxStage`, `--stage lean --grid`) and its exact controls
  (SF-40 F) are provided by SF-40 and consumed by SF-38; the production geometry and `lambda/h` are fixed at
  activation from SF-37's per-amplitude `lambda/h` and SF-40's measured capacity
  (`docs/decisions/2026-10-22-lean-lagrangian-stage.md`); the Goal is unchanged.
- **SF-19 at production size.** Convergence with the automatic MG hierarchy (coarsest level 32x4x4) at
  `sigma^2 = 4`; peak memory.
- **Medium A/B evaluation** on the same `K`: (a) as above; (b) `solve_head` (`src/physics/flow/solve_head.cuh`,
  Dirichlet in `x`, periodic `y`/`z`, harmonic face `K`) with head drop `G1^(a) L1` so that the mean gradients
  match, velocities by `compute_velocity_from_head` (`src/physics/flow/velocity_from_head.cuh`). Eulerian velocity statistics vs distance to the `x` faces and Pollock
  plume statistics on both media (SF-37 repack; for (b) the outlet face plane is dropped and particles are stopped
  at `L1 - 5 lambda`, so the periodic indexing of `PollockTracker` is never exercised across `x`).
- **Ensemble runner**: loop over realizations `r`: field(seed_r) -> flow -> splines -> injection -> forward tracking
  -> moments; time series written in the `io::OutputLayout` layout read by `AnalysisRunner` (`run_analysis`):
  `stats/realization_XXXX_timeseries.csv` (`CsvTimeSeriesWriter` format), plus per-realization
  `moments_planes.csv` and `track.json`; a completion marker per realization; restart skips
  completed realizations; aggregate through `compute_macrodispersion`.
- **Par2 RWPT baseline**: new configs (no existing config changed) derived from `apps/config_pipeline_par2.yaml` at
  `sigma^2` in {1, 2.25, 4}: 2048x256x256, `lambda/h = 10`, Dirichlet in `x`, periodic `y`/`z`, injection at
  `x = 5 lambda`, `N_p = 1e4`, purely advective (`diffusion = alpha_l = alpha_t = 0`), pilot `N_R = 10`; the baseline
  generator differs from SF-18 (recorded; the SF-39 comparison is statistical).
- Experiment note and artifact.

## Out of scope

- A spline non-periodic in `x1` (provisional SF-41, only if (b) is selected); the study and its `alpha` estimates
  (SF-39); changes to SF-18/SF-19/SF-28/SF-32 numerics; any change to existing configs or to the Par2 adapter.

## Files and symbols

- New: `src/physics/particles/lagrangian_labels/EnsembleRunner.{cuh,cu}` (or an `apps/` driver;
  decided in the DAG), `apps/lagrangian_ensemble/` (executable), `apps/config_pipeline_par2_sf38_s{1,225,4}.yaml`,
  `docs/experiments/<date>-sf38-long-domain-infrastructure.md` and artifact.
- Provided by SF-40 and extended (not created) here: `src/physics/particles/lagrangian_labels/LongBoxStage.{cuh,cu}`
  (`LongBoxSpec`, `LongBoxStage`, `build_long_box_stage`), `tests/lagrangian_labels/long_box_tests.cu`
  (`lagrangian_long_box`, `<= 60 s`).
- Read-only: SF-18 `generate_periodic_gaussian_field`, SF-19 `solve_affine_periodic_flow`, SF-28,
  `solve_head`/`compute_velocity_from_head`, `LagrangianTracker`, `PollockTracker` + SF-37 repack, `AnalysisRunner`
  (`src/runtime/analysis/AnalysisRunner.hpp`, `run_analysis`), `io::OutputLayout` (`src/io/output_layout.hpp`),
  `compute_macrodispersion`, `macroflow3d_pipeline`.

## Implementation specification

1. Long-box stage as in "In scope"; `require_inlet_labels = false` by default (labels are attributes only).
2. Device peak measured by nvidia-smi sampling (200 ms, effective period as in the 2026-10-17 note); host RSS by
   `/usr/bin/time -v`.
3. **Medium A/B design (pre-registered).** Seeds 4001-4005 (`N_R = 5`), `sigma^2` in {1, 4}, production grid (or
   the largest grid that passes B). Statistics: (E1) transverse-plane averages of mean and variance of each face
   velocity component vs `x1`; interior = `[5 lambda, L1 - 5 lambda]`; per realization the interior average of each
   profile; (E2) Pollock plume `sigma_T^2(x1)` and `var tau(x1)` at planes `X0 + m lambda` (`X0 = 5 lambda`,
   `m = 1, 2, 4, 8, ...` up to `L1 - 5 lambda`), window `0.8 L_T x 0.8 L_T`, `N_p = 1e4`. Paired difference per
   realization `D_r = Q_a - Q_b`; `SE = sd(D_r)/sqrt(5)`.
4. Ensemble determinism: realization `r` is a pure function of (config, seed_r); outputs written atomically
   (temporary name then rename) before the completion marker.

## Expected numerical effect

No existing output changes. Expected: medium (b) has boundary layers of a few `lambda` near the `x` faces; whether
its interior statistics differ from (a) beyond sampling is not predicted.

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R 'lagrangian|inlet_slab'
scripts/remote --increment SF-38 sync
scripts/remote --increment SF-38 exec -- "cmake --preset v100-release && cmake --build build/v100-release -j 2>&1 | tail -n 20; echo BUILD_EXIT=${PIPESTATUS[0]}"
A=docs/experiments/artifacts/<date>-sf38-long-domain-infrastructure/scripts
for part in controls prodgrid ab ensemble par2-pilot; do scripts/remote --increment SF-38 run sf38-$part -- "bash $A/run_sf38.sh build/v100-release $part"; scripts/remote --increment SF-38 wait sf38-$part; done
scripts/remote --increment SF-38 run sf38-ctest-full-bytecmp -- "ctest --test-dir build/v100-release --output-on-failure; <SF-27 trio byte-compare vs base refs>"
scripts/remote --increment SF-38 wait sf38-ctest-full-bytecmp
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

Pre-registered verbatim in the activation bitácora row before any run; refinements of the A/B statistics are
allowed only there and are recorded as such.

- **A. Controls on the long box** (64x16x16 and 128x16x16, `h = 1/16`): `k = 1` and layered `k(x1)` reproduce
  SF-36 A1/A2 (`<= 1e-12`; `tau`, `I` `<= 1e-9`); SF-19 on `k = 1` gives `K_eff = I` and mean flux `e1` to PCG level;
  a particle shifted by `+1` in `x2` or `x3` sees the same field (transverse periodicity) to `<= 1e-14`.
- **B. Production grid** (2048x256x256, `eps = 2`, seed 4001): the three SF-19 correctors converge to `rtol = 1e-10`
  with the automatic hierarchy; device peak `<= 30000 MiB` over field + stage + tracing of `1e4` particles over the
  full length; wall time and host RSS recorded.
- **C. Medium A/B (decision rule).** Medium (a) is selected iff `|mean_r D_r| <= 2 SE` for every interior statistic
  of E1 and every plane statistic of E2, at both amplitudes; otherwise medium (b) is selected (R4). The boundary-
  layer thickness of (b) (distance from the faces beyond which its profiles stay within `2 SE` of their interior
  mean) is recorded.
- **D. Ensemble runner.** A rerun of a realization is bitwise identical; a job killed mid-realization and
  restarted produces byte-identical final outputs; the CSVs read by `compute_macrodispersion` reproduce the
  in-process moments to `<= 1e-12` relative; per-realization wall time recorded.
- **E. Par2 baseline pilot.** The three configs run `N_R = 10` each to completion with outputs preserved; time per
  realization and `alpha_L(t)` series recorded (no threshold).
- **F. Regression.** SF-35/SF-36/SF-37 cube outputs byte-identical; full V100 `ctest` green; SF-27 trio byte-compare
  clean; no existing config changed.
- **G. Recorded.** SF-18 raw variance and applied scale on the long box for seeds 4001-4005 (input to SF-39's
  `normalize_variance` choice).

## Regression surface

- The cube stage and tracker (shared module), `ctest` wall time; Par2 path untouched (new configs only).

## Failure and rollback policy

- A or D not met -> defect (BLOCKING).
- B not met (no convergence or peak above 30000 MiB) -> R3: the domain is reduced (e.g. `L_T = 12.8 lambda`) by
  owner decision, recorded; C is then run on the reduced grid.
- C selects (b) -> R4: SF-38 closes with (b) selected; a non-periodic-in-`x1` spline increment (SF-41) is specified
  and must be `done` before SF-39 activates.
- E fails -> recorded; Gate 5 of SF-39 needs the baseline -> owner.

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

`SF-39` becomes eligible after this increment and `SF-37` are merged and marked `done` on the default branch (and,
if C selects medium (b), after the non-periodic spline increment).

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-17T19:30Z | not started | Specification created on the SF-35 closure PR (#53) after the owner decision of 2026-10-17 (`docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`, items 7-9). The memory ceiling is expressed in MiB from the probes' nvidia-smi measurement (25925 MiB at 1.34e8 cells). | Depends on SF-36 `done` on `master`; parallel with SF-37. | Activate only when the checker reports it READY; the activation row contains thresholds A-G verbatim and the base commit before any run. |
| 2026-10-22T15:20Z | not started | Re-wired on the governance PR (branch docs/lester-sf40-lean-stage-spec): Depends on SF-36, SF-40; lean non-cubic stage provided by SF-40; provisional non-periodic spline renumbered SF-41 (`docs/decisions/2026-10-22-lean-lagrangian-stage.md`). Memory audit: at 1.34e8 cells the current stage's exact-byte peak is ~213 B/cell and the measured nvidia-smi peaks range 25921-27981 MiB (`docs/experiments/2026-10-22-lagrangian-memory-audit.md` §4-§5), not only the 25925 MiB of the 2026-10-17 probes. Goal unchanged. | SF-38 is not READY until SF-40 is done on master. | Production geometry and lambda/h fixed in the activation row from SF-37 and SF-40's measured capacity. |
