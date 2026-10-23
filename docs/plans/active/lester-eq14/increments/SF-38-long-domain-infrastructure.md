# SF-38 — Long domain: non-cubic stage, medium A/B evaluation, ensemble runner and Par2 baseline

- State: `active`
- Goal: `Construir la infraestructura del dominio largo (stage no cúbico SF-18/SF-19/SF-28 en las geometrías de producción por amplitud, 2048x256x256 a lambda/h 10 para sigma^2 = 1 y 3328x384x384 a lambda/h 16 y 32 para sigma^2 = 2.25 y 4, con pico de memoria de dispositivo medido <= 30000 MiB; runner de ensemble con checkpoint por realización, serie gemela y puente Pollock para el Gate 5; baseline Par2 RWPT en el mismo dominio físico) y elegir entre el medio affine-periódico y el de Dirichlet en x con un test de equivalencia pre-registrado.`
- Depends on: `SF-36, SF-40`
- Unlocks: `SF-39`
- Branch: `science/lester-sf38-long-domain-infrastructure`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 5 (baseline preparation)`
- Human review: `required`
- Owner: `unassigned`
- Started: `2026-10-23`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Decision records `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md` (items 7-9, stop rules R3, R4) and
`docs/decisions/2026-10-23-sf38-activation-decisions.md` (owner decisions D1-D20 of 2026-10-23, which fix the production
geometry, the A/B rule, the Gate 5 protocol and the Par2 baseline). The study (SF-39) needs a long, transversally periodic
domain (Beaudoin & de Dreuzy 2013 §2 [6]-[7]: `L_T/lambda = 25.6`, `lambda/Delta = 10`, fixed heads on the faces orthogonal
to the mean flow, periodic transverse faces; local copy `docs/references/Dreuzy-2013-WRR.pdf` p. 2490-2491) on one
V100-32GB, at the `lambda/h` SF-37 validated. SF-38 builds and measures that infrastructure and is allowed to establish:

- (a) the non-cubic stage (field, flow, splines, injection) is correct on controls and fits the production grids
  (`sigma^2 = 1`: 2048x256x256 = 1.34e8 cells; `sigma^2` 2.25 and 4: 3328x384x384 = 4.91e8 cells, where SF-40 measured
  28885 MiB);
- (b) the choice between medium (a) SF-19 affine-periodic on the long box and medium (b) `solve_head` with Dirichlet in
  `x` and periodic `y`/`z`, by a pre-registered equivalence test;
- (c) a reproducible, restartable ensemble runner whose per-realization outputs feed `compute_macrodispersion`, with the
  Gate 5 twin series and the Pollock bridge;
- (d) the Par2 RWPT baseline configuration of the study runs in the same physical domain (pilot ensemble), for Gate 5 in
  SF-39.

No `alpha` estimate is reported as a result (SF-39).

## Preconditions

- `SF-36` is `done` on the default branch (forward tracker, P2, stage options). SF-37 is `done` (production `lambda/h`
  8 / 16 / 32 at `sigma^2` 1 / 2.25 / 4, as applied by decision 2026-10-23 D1: `sigma^2 = 1` at 10).
- `SF-40` is `done` on the default branch (lean non-cubic `Grid3D` stage at memory level S+,
  `docs/plans/active/lester-eq14/increments/SF-40-lean-lagrangian-stage.md`).

## In scope

- **Long-box stage.** Box `[0, L1] x [0, 1]^2`, `h = 1/n_T` isotropic (SF-19 requires `dx = dy = dz`; the lean stage
  requires `ny == nz`), `L1 = n_x h`, transverse period 1 (so `InletLabels` on the `n_T x n_T` inlet still applies when
  `v1 > 0`). SF-18 on the box (even extents; `corr_length` in box units, `ell = (lambda/h) h`), SF-19 affine-periodic with
  `qbar = e1`, SF-28 splines on the box (non-cubic supported), memory-lean lifetimes (SF-40 lean stage, `LongBoxStage`,
  `--stage lean --grid`, consumed here). **Production geometries (decision 2026-10-23, D1):**
  - `sigma^2 = 1`: 2048x256x256, `lambda/h = 10` (`ell = 10/256`, `eps = 1`): 204.8 x 25.6 x 25.6 `lambda`;
  - `sigma^2 = 2.25`: 3328x384x384, `lambda/h = 16` (`ell = 16/384`, `eps = 1.5`): 208 x 24 x 24 `lambda`;
  - `sigma^2 = 4`: 3328x384x384, `lambda/h = 32` (`ell = 32/384`, `eps = 2`): 104 x 12 x 12 `lambda`.
- **Injection (D9).** Flux-weighted on a plane `x1 = X0`, window `[0.1, 0.9] x [0.1, 0.9]` (0.8 `L_T`, centred, as B&dD),
  `M = 100` (`N_p = 1e4`). The runner asserts, before tracking, that no seed lies on a transverse cell face
  (`|n_T x - round(n_T x)| > 1e-6` for both transverse coordinates of every seed) and stops otherwise (with `M = 100` and
  this window the closest seed is `h/125` from a face for `n_T` 256, 288, 384; an `n_T` multiple of 5 would put seeds on
  faces). Pollock keeps its own RT0 weight (SF-37 RR7).
- **SF-19 at production size.** Convergence with the automatic MG hierarchy at each production grid; peak memory.
- **Medium A/B evaluation** on the same `K` (D4-D8): (a) as above, with the SF-19 face fluxes kept for Pollock; (b)
  `solve_head` (`src/physics/flow/solve_head.cuh`, Dirichlet in `x`, periodic `y`/`z`, harmonic face `K`) with head drop
  `G1^(a) L1` so that the mean gradients match, `mg_levels = 7` passed explicitly (the default is 4), PCG `rtol = 1e-10` on
  its own residual `||r|| / ||r_0||`, velocities by `compute_velocity_from_head`
  (`src/physics/flow/velocity_from_head.cuh`); the `FlowWorkspace` is released before the face repack (otherwise the peak is
  ~140.6 B/cell, ~29.7 GiB with the context at the A/B grid, within ~1 % of the 30000 MiB ceiling). **A/B grids:** `sigma^2 = 1` on its production box 2048x256x256;
  `sigma^2 = 4` on 3328x256x256 (`lambda/h = 32`, `ell = 32/256`: 104 x 8 x 8 `lambda`, the production length and resolution
  at two thirds of the production width; medium (b) does not fit 3328x384x384). Recorded assumption: the slowest transverse
  mode decays over `L_T/(2 pi)` (1.3 `lambda` at 8 `lambda`, 1.9 `lambda` at 12 `lambda`), so the A/B may slightly
  understate (b)'s boundary layer. Eulerian velocity statistics vs distance to the `x` faces and Pollock plume statistics on
  both media (SF-37 repack; for (b) the outlet face plane is dropped, the plane pass only is run, planes stop at
  `L1 - 5 lambda`, so the periodic indexing of `PollockTracker` is never exercised across `x`; a particle that leaves
  through `x = 0` by backflow is counted non-ok and recorded). A solver pilot of (b) (threshold C0) precedes the A/B.
- **Ensemble runner**: loop over realizations `r`: field(seed_r) -> flow -> splines -> injection -> forward tracking
  -> moments; time series written in the `io::OutputLayout` layout read by `AnalysisRunner` (`run_analysis`):
  `stats/realization_XXXX_timeseries.csv` (`CsvTimeSeriesWriter` format), plus per-realization
  `moments_planes.csv` and `track.json`; a completion marker per realization; restart skips
  completed realizations; aggregate through `compute_macrodispersion`. **Gate 5 outputs (D11):** per realization the runner
  also writes (i) a twin series with uniform weights `1/N_p` over the same windowed particles (recomputed with the public
  `finalize_track_result`; no tracker change) and (ii) a Pollock bridge: the Pollock comparator on the SF-19 faces of the same
  realization and seeds, flux-weighted and twin series; a plane `x1 = X_inj + (L1 - 10 lambda)` (the Par2 travel distance from the injection plane `X_inj`) is added to the
  production plane list and its per-particle first-arrival times are kept, so that SF-39 can define `t_b` of the twin
  series.
- **CSV reader fix (D13).** `read_timeseries_csv` (`src/analysis/macrodispersion/MacrodispersionAnalysis.hpp`) skips every
  line that starts with `#` and then one column-header line (today it skips one line, so the column header written by
  `CsvTimeSeriesWriter` becomes an all-zero row; audit `docs/experiments/2026-10-22-lagrangian-memory-audit.md` §7); the
  test that expected the artifact row (`tests/lagrangian_labels/lagrangian_gpu_host_tests.cu`) is updated.
- **Par2 RWPT baseline (D10-D12)**: new configs (no existing config changed) derived from `apps/config_pipeline_par2.yaml`,
  same physical domain as the Lagrangian study:
  - `s1`: 2048x256x256, `dx = 5`, `corr_length = 50` (`lambda/h = 10`), `sigma2: 1`, `mg_levels: 6`;
  - `s225`: 2496x288x288, `dx = 5`, `corr_length = 60` (`lambda/h = 12`; 208 x 24 x 24 `lambda`), `sigma2: 2.25`,
    `mg_levels: 5`;
  - `s4`: 2496x288x288, `dx = 5`, `corr_length = 120` (`lambda/h = 24`; 104 x 12 x 12 `lambda`), `sigma2: 4`,
    `mg_levels: 5` (six levels would leave a 78x9x9 coarsest grid whose red-black colouring races across the periodic seam).
  All: Dirichlet in `x` with the mean head gradient of the base config (`J = 100/10240`; the west head scales with `Lx`),
  periodic `y`/`z`, `inject_x = 5 lambda` (uniform over the full transverse plane, the only Par2 injection), `N_p = 1e4`,
  purely advective (`diffusion = alpha_l = alpha_t = 0`), `dt = 12.5` (the `dt/Delta` of the base config, ~35 steps per
  cell at the `sigma^2 = 1` mean velocity), `n_steps = ceil(1.2 (L1 - 10 lambda) / (J dt))` (nominal Darcy velocity
  `K_g J`, `K_g = 1`), snapshots every `floor(n_steps/100)` steps (for `t_b`;
  `n_steps` = 95749 / 116786 / 110887 for `s1` / `s225` / `s4`), `diagnostics.velocity_field: false`,
  `save_head: false`, `save_K: false`; the realized mean and variance of `ln K` per realization are measured by a new SF-38
  tool that regenerates each Par2 field with the read-only `generate_K_field` at the same seed and parameters, and that
  checks the `K` min/max/mean it obtains against the pipeline's `K stats` line (decision 2026-10-23 D17; nothing in the Par2
  path writes `K` to disk); pilot `N_R = 10` with K seeds `7001-7010`; a paired `dt/2` check (`dt = 6.25`, `n_steps` doubled) on the
  first 3 realizations of each config. The Par2 generator differs from SF-18 (random spectral, not periodic, `sigma^2` not
  normalized: recorded; the SF-39 comparison is statistical).
- Experiment note and artifact.

## Out of scope

- A spline non-periodic in `x1` (provisional SF-41, only if (b) is selected); the study and its `alpha` estimates
  (SF-39); changes to SF-18/SF-19/SF-28/SF-32 numerics; changes to `LagrangianTracker`, `PollockTracker`,
  `PollockComparator`, the SF-37 repack, the Par2 adapter or the Par2 path of `src/runtime/ensemble/EnsembleRunner.cu`; a
  lean `solve_head`; any change to existing configs.

## Files and symbols

- New: `src/physics/particles/lagrangian_labels/EnsembleRunner.{cuh,cu}` (or an `apps/` driver;
  decided in the DAG), `apps/lagrangian_ensemble/` (executable), `apps/config_pipeline_par2_sf38_s{1,225,4}.yaml`,
  the Par2 field-statistics tool of D17 (an `apps/` executable; location decided in the DAG),
  `docs/experiments/<date>-sf38-long-domain-infrastructure.md` and artifact.
- Changed (D13 only): `src/analysis/macrodispersion/MacrodispersionAnalysis.hpp` (`read_timeseries_csv`; nothing else in the
  header), `tests/lagrangian_labels/lagrangian_gpu_host_tests.cu` (the artifact-row expectation).
- Provided by SF-40 and extended (not created) here: `src/physics/particles/lagrangian_labels/LongBoxStage.{cuh,cu}`
  (`LongBoxSpec`, `LongBoxStage`, `build_long_box_stage`), `tests/lagrangian_labels/long_box_tests.cu`
  (`lagrangian_long_box`, `<= 60 s`).
- Read-only: SF-18 `generate_periodic_gaussian_field`, SF-19 `solve_affine_periodic_flow`, SF-28,
  `solve_head`/`compute_velocity_from_head`, `LagrangianTracker` (`finalize_track_result` used as is), `PollockTracker`,
  `PollockComparator` + SF-37 repack, `AnalysisRunner` (`src/runtime/analysis/AnalysisRunner.hpp`, `run_analysis`),
  `compute_macrodispersion`, `io::OutputLayout` (`src/io/output_layout.hpp`), `CsvTimeSeriesWriter`, `macroflow3d_pipeline`.

## Implementation specification

1. Long-box stage as in "In scope"; `require_inlet_labels = false` by default (labels are attributes only); the seed-face
   assertion of D9.
2. Device peak measured by nvidia-smi sampling per PID (`--query-compute-apps`, 200 ms, filtered by the run's PID, as in
   SF-40) plus the device-wide `memory.used` of the physical GPU (`-i`); host RSS by `/usr/bin/time -v`.
3. **Medium A/B design (pre-registered; decision 2026-10-23 D4-D7).** Seeds 4001-4010 (`N_R = 10`; one analysis after all
   ten, no interim look), `sigma^2` in {1, 4}, on the A/B grids of "In scope". For each realization `r` and medium:
   - `qbar` = mean of `v1` over an `x`-face plane (the plane flux, constant in `x1` up to the solver tolerance; 1 in (a));
     every statistic below is normalized by its own medium's `qbar`;
   - **E1 (Eulerian).** Per face plane `i` with `x1` in the interior `[5 lambda, L1 - 5 lambda]`: `var_i(v1)` over the
     `x`-faces of the plane; `mean_i(v2)`, `mean_i(v3)`, `var_i(v2)`, `var_i(v3)` over the `y`- and `z`-faces of cell layer
     `i`. Primary: `V1 = <var_i(v1)>_i / qbar^2`, `theta_2 = <mean_i(v2)>_i / qbar`, `theta_3 = <mean_i(v3)>_i / qbar`
     (`<.>_i` = average over interior planes). Recorded: `qbar`, `<var_i(v2)>_i / qbar^2`, `<var_i(v3)>_i / qbar^2`, every
     profile vs `x1`.
   - **E2 (Pollock plume, plane pass).** Planes `X_m = X0 + m lambda`, `X0 = 5 lambda`, `m = 1, 2, 4, ...` up to
     `L1 - 5 lambda` (`sigma^2 = 1`: `m <= 128`, 8 planes; `sigma^2 = 4`: `m <= 64`, 7 planes); injection of "In scope"
     (window `[0.1, 0.9]^2`, `M = 100`, Pollock's own flux weights). `sigma_T^2(X) = (var x2 + var x3)/2` over ok particles
     (flux-weighted), `var tau(X)`. Primary, over the last octave `[X_{m*/2}, X_{m*}]` (`m*` the largest `m`):
     `S_T = [sigma_T^2(X_{m*}) - sigma_T^2(X_{m*/2})] / ((m*/2) lambda)` and
     `S_tau = qbar^2 [var tau(X_{m*}) - var tau(X_{m*/2})] / ((m*/2) lambda)`. Recorded: every plane, mean `tau`, non-ok
     counts, the paired differences `D_r` of every E1/E2 statistic.
   - Primary statistics per amplitude: `S_T`, `S_tau`, `V1`, `theta_2`, `theta_3` (ten in total). The decision rule is C.
4. Ensemble determinism: realization `r` is a pure function of (config, seed_r); outputs written atomically
   (temporary name then rename) before the completion marker.
5. **Medium (b) solver pilot (D8)** before the A/B: seed 4001, `sigma^2 = 4`, 3328x256x256 (threshold C0).

## Expected numerical effect

No existing output changes, except `run_mode: analysis_only`, which now reads `CsvTimeSeriesWriter` files without the
spurious all-zero first row (the first two rows of its `analysis/macrodispersion.csv` change; no versioned result used that
path; the in-process path of `macroflow3d_pipeline` is unchanged). Expected: medium (b) has boundary layers of a few `lambda`
near the `x` faces; whether its interior statistics are equivalent to (a) within the margins is not predicted.

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R 'lagrangian|inlet_slab'
scripts/remote --increment SF-38 sync
scripts/remote --increment SF-38 exec -- "cmake --preset v100-release && cmake --build build/v100-release -j 2>&1 | tail -n 20; echo BUILD_EXIT=${PIPESTATUS[0]}"
A=docs/experiments/artifacts/<date>-sf38-long-domain-infrastructure/scripts
for part in controls prodgrid bpilot ab ensemble par2-pilot; do scripts/remote --increment SF-38 run sf38-$part -- "bash $A/run_sf38.sh build/v100-release $part"; scripts/remote --increment SF-38 wait sf38-$part; done
scripts/remote --increment SF-38 run sf38-ctest-full-bytecmp -- "ctest --test-dir build/v100-release --output-on-failure; <SF-27 trio byte-compare vs base refs>"
scripts/remote --increment SF-38 wait sf38-ctest-full-bytecmp
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

Pre-registered verbatim in the activation bitácora row before any run; refinements of the A/B statistics are allowed only
there and are recorded as such; no threshold is tuned afterwards.

- **A. Controls on the long box** (amended by decision 2026-10-23 D21 before activation). On 64x16x16 and 128x16x16
  (`h = 1/16`): `k = 1` reproduces SF-36 A1 (`<= 1e-12`; `tau`, `I` `<= 1e-9`); SF-19 on `k = 1` gives `K_eff = I` and
  mean flux `e1` to PCG level; a particle shifted by `+1` in `x2` or `x3` sees the same field (transverse periodicity) to
  `<= 1e-14`. On 64x32x32 and 128x32x32 (`h = 1/32`): layered `k(x1)` reproduces SF-36 A2 as amended by the owner on
  2026-10-17 (transverse positions and covectors `<= 1e-12`; `tau` `<= 1e-9` relative; `I` against the independent
  quadrature recorded, no threshold).
- **B. Production grids** (seed 4001): B1 2048x256x256, `eps = 1`, `ell = 10/256`; B2 3328x384x384, `eps = 1.5`,
  `ell = 16/384`; B3 3328x384x384, `eps = 2`, `ell = 32/384`. In each, the three SF-19 correctors converge to
  `rtol = 1e-10` with the automatic hierarchy; device peak `<= 30000 MiB` over field + stage + tracing of `1e4` particles over
  the full length with the SF-38 runner and every output it writes in production (plane and time passes, twin series,
  Pollock bridge); wall time and host RSS recorded.
- **C0. Medium (b) solver pilot** (seed 4001, `sigma^2 = 4`, 3328x256x256): `solve_head` with `mg_levels = 7` reaches
  `||r|| / ||r_0|| <= 1e-10`; independent checks: total-flux balance `|Q(x=0) - Q(x=L1)| / |Q(x=0)| <= 1e-8`, and the maximum
  cell divergence of the face velocities (recorded); device peak of medium (b) (solve, repack, Pollock) `<= 30000 MiB`.
  Fallback: if the PCG stalls above `1e-10` but reaches `<= 1e-8` with the flux-balance check met, that tolerance is used for
  the A/B and recorded; otherwise stop (owner).
- **C. Medium A/B (decision rule; equivalence test).** For each amplitude and each primary statistic `P` of Implementation 3,
  with `Pbar_a`, `Pbar_b` the means over the ten paired realizations:
  - ratio statistics `S_T`, `S_tau`, `V1`: `R = ln(Pbar_a / Pbar_b)` with the 90 % confidence interval of the delta method
    on the paired realizations, `R +- t_{0.95,9} SE_R`, `t_{0.95,9} = 1.833`,
    `SE_R = sd_r(P_{a,r}/Pbar_a - P_{b,r}/Pbar_b) / sqrt(10)`; margins `+-ln(1.05)` (`S_T`), `+-ln(1.025)` (`S_tau`),
    `+-ln(1.05)` (`V1`). Absolute floor for `S_T` (positions in units of `lambda`): if `Pbar_b < 0.05 lambda`
    (`= 2.5e-3 lambda / 0.05`) or `Pbar_a <= 0`, `S_T` is tested on the paired difference with margin `+-2.5e-3 lambda`.
    If `Pbar_a <= 0` or `Pbar_b <= 0` for `S_tau` or `V1`, the statistic is inconclusive;
  - paired differences (floor case and tilts): CI `Dbar +- 1.833 sd_r(D_r) / sqrt(10)`, `D_r = P_{a,r} - P_{b,r}`;
  - tilts `theta_2`, `theta_3`, each component separately: paired difference, 90 % CI, margin `+-0.005` (D19);
  - a statistic is **equivalent** if its CI lies inside the margin, **non-equivalent** if its CI lies entirely outside, and
    **inconclusive** otherwise.
  Medium (a) is selected iff all ten primary statistics (both amplitudes) are equivalent; otherwise (any non-equivalent or
  inconclusive statistic) medium (b) is selected (R4). Every E1/E2 statistic, `D_r` and CI is recorded. The boundary-layer
  thickness of (b) (distance from the faces beyond which its `var_i(v1)/qbar^2` profile stays within `2 SE` of its interior
  mean, SE = the standard deviation over the ten realizations of the (b) plane value / `sqrt(10)`) is recorded.
- **D. Ensemble runner.** A rerun of a realization is bitwise identical; a job killed mid-realization and
  restarted produces byte-identical final outputs (flux-weighted series, twin series and Pollock bridge included); the CSVs
  read by `compute_macrodispersion` through the fixed `read_timeseries_csv` reproduce the in-process moments to
  `<= 1e-12` relative; per-realization wall time recorded.
- **E. Par2 baseline pilot.** The three configs run `N_R = 10` each to completion with outputs preserved; time per
  realization, `alpha_L(t)` series, `t_b` (first snapshot with a particle at `x >= L1 - 5 lambda`) and the realized variance
  of `ln K` per realization recorded; the paired `dt/2` check recorded (no threshold).
- **F. Regression.** SF-35/SF-36/SF-37 cube outputs byte-identical; full V100 `ctest` green, with the single expected test
  change of D13 (`lagrangian_tracker_gpu_host`); SF-27 trio byte-compare clean; no existing config changed.
- **G. Recorded.** SF-18 raw variance and applied scale on the long box for seeds 4001-4010 (input to SF-39's
  `normalize_variance` choice).

**Compute (decision 2026-10-23 D16).** One V100; every remote job serialized on the SF-38 mirror; no GPU-hour ceiling;
the GPU-hours of each part are reported in the experiment note and in the closing bitácora row.

## Regression surface

- The cube stage and tracker (shared module), `ctest` wall time; `run_mode: analysis_only` (reader fix); Par2 path untouched
  (new configs only).

## Failure and rollback policy

- A or D not met -> defect (BLOCKING).
- B not met (no convergence or peak above 30000 MiB) -> R3: the domain is reduced by owner decision, recorded; C is then run on
  the reduced grid.
- C0 not met -> stop before the A/B; owner.
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
if C selects medium (b), after the non-periodic spline increment (SF-41)).

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-17T19:30Z | not started | Specification created on the SF-35 closure PR (#53) after the owner decision of 2026-10-17 (`docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`, items 7-9). The memory ceiling is expressed in MiB from the probes' nvidia-smi measurement (25925 MiB at 1.34e8 cells). | Depends on SF-36 `done` on `master`; parallel with SF-37. | Activate only when the checker reports it READY; the activation row contains thresholds A-G verbatim and the base commit before any run. |
| 2026-10-22T15:20Z | not started | Re-wired on the governance PR (branch docs/lester-sf40-lean-stage-spec): Depends on SF-36, SF-40; lean non-cubic stage provided by SF-40; provisional non-periodic spline renumbered SF-41 (`docs/decisions/2026-10-22-lean-lagrangian-stage.md`). Memory audit: at 1.34e8 cells the current stage's exact-byte peak is ~213 B/cell and the measured nvidia-smi peaks range 25921-27981 MiB (`docs/experiments/2026-10-22-lagrangian-memory-audit.md` §4-§5), not only the 25925 MiB of the 2026-10-17 probes. Goal unchanged. | SF-38 is not READY until SF-40 is done on master. | Production geometry and lambda/h fixed in the activation row from SF-37 and SF-40's measured capacity. |
| 2026-10-23T02:30Z | not started | Re-specified on the governance PR (branch `docs/lester-sf38-activation-governance`) after the owner decisions of 2026-10-23 (`docs/decisions/2026-10-23-sf38-activation-decisions.md`, D1-D20): Goal rewritten (production geometry per amplitude, Gate 5 twin series and Pollock bridge, equivalence test); production grids 2048x256x256 at `lambda/h` 10 and 3328x384x384 at 16 and 32; A/B on 2048x256x256 and 3328x256x256 with `N_R = 10` (seeds 4001-4010) and a two-one-sided-test rule on five primary statistics per amplitude (margins half of the SF-39 CI targets; inconclusive -> (b)); medium (b) solver pilot C0; injection window `[0.1, 0.9]^2`, `M = 100`, seed-face assertion; Par2 configs in the same physical domain (2048x256x256 at 10; 2496x288x288 at 12 and 24, `mg_levels: 5`), `dt = 12.5` with a paired `dt/2` check; CSV reader fix in scope (D13); Par2 realized `sigma^2` by a new field-statistics tool (D17); tilts tested per component (D19); one V100, serialized, no GPU-hour ceiling (D16); thresholds B, C, D, E, F rewritten and C0 added. The previous Goal and thresholds are in git history at `ab78d7c`. | Docs only; no run. SF-38 stays `pending` (READY on master once this PR is merged). | Activate after the merge: activation row with the base commit and thresholds A-G (with C0) verbatim before any run. |
| 2026-10-23T15:22Z | not started | Threshold A amended on a second docs-only governance PR (branch `docs/lester-sf38-criterion-a-amendment`) by owner decision D21 (addendum of `docs/decisions/2026-10-23-sf38-activation-decisions.md`), found by the orchestrator while preparing the activation and before any run: the layered `k(x1)` control moves to 64x32x32 and 128x32x32 (`h = 1/32`) with SF-36 A2 as amended by the owner on 2026-10-17 (transverse positions and covectors `<= 1e-12`, `tau <= 1e-9` relative, `I` recorded without threshold), because at `h = 1/16` SF-36 measured `tau` 4.6e-9 and the passive `I` carries an `O(h_max^~2)` spline error; the `k = 1`, `K_eff` and shift controls stay on 64x16x16 and 128x16x16. No other threshold changed. | Docs only; no run. | Activate after the merge, with thresholds A-G (with C0) verbatim. |
| 2026-10-23T16:17Z | activation on `master=d8ffa7c` (PR #61 merged after #60; checker OK ready=SF-34 SF-38, nonterminal=none; activating keeps nonterminal = 1); delivery branch `science/lester-sf38-long-domain-infrastructure`; remote `scripts/remote --increment SF-38` (one mirror, one job at a time, one GPU, decision D16) | UNDERSTAND (runtime record `.claude/orchestration/SF-38-long-domain-infrastructure/understanding.md`; code reconnaissance of `LongBoxStage`, `LagrangianTracker` (`finalize_track_result`, `write_track_outputs`), `PollockComparator`, `FaceFluxRepack`, `solve_head`/`compute_velocity_from_head`, `read_timeseries_csv`/`compute_macrodispersion`/`AnalysisRunner`, `io::OutputLayout`, the Par2 config and `EnsembleRunner` (Par2 path), `generate_K_field`; owner decisions D1-D21). **Pre-registration before any run.** Base commit `d8ffa7cad739cc8c61d870abbfb11cd748b314af`. Reading rules RR0-RR13 and implementation contracts C-RUNNER, C-DIRICHLET, C-AB, C-PROD, C-PAR2 in `docs/experiments/artifacts/2026-10-23-sf38-long-domain-infrastructure/preregistration.md` (this commit; sha256 `aca09f067e14d9461bae396bb25d40c8e2afceba90170785d0e88bd06337f0db`); they change no threshold. Acceptance thresholds, verbatim from this file (line breaks joined): Pre-registered verbatim in the activation bitácora row before any run; refinements of the A/B statistics are allowed only there and are recorded as such; no threshold is tuned afterwards. - **A. Controls on the long box** (amended by decision 2026-10-23 D21 before activation). On 64x16x16 and 128x16x16 (`h = 1/16`): `k = 1` reproduces SF-36 A1 (`<= 1e-12`; `tau`, `I` `<= 1e-9`); SF-19 on `k = 1` gives `K_eff = I` and mean flux `e1` to PCG level; a particle shifted by `+1` in `x2` or `x3` sees the same field (transverse periodicity) to `<= 1e-14`. On 64x32x32 and 128x32x32 (`h = 1/32`): layered `k(x1)` reproduces SF-36 A2 as amended by the owner on 2026-10-17 (transverse positions and covectors `<= 1e-12`; `tau` `<= 1e-9` relative; `I` against the independent quadrature recorded, no threshold). - **B. Production grids** (seed 4001): B1 2048x256x256, `eps = 1`, `ell = 10/256`; B2 3328x384x384, `eps = 1.5`, `ell = 16/384`; B3 3328x384x384, `eps = 2`, `ell = 32/384`. In each, the three SF-19 correctors converge to `rtol = 1e-10` with the automatic hierarchy; device peak `<= 30000 MiB` over field + stage + tracing of `1e4` particles over the full length with the SF-38 runner and every output it writes in production (plane and time passes, twin series, Pollock bridge); wall time and host RSS recorded. - **C0. Medium (b) solver pilot** (seed 4001, `sigma^2 = 4`, 3328x256x256): `solve_head` with `mg_levels = 7` reaches `\|\|r\|\| / \|\|r_0\|\| <= 1e-10`; independent checks: total-flux balance `\|Q(x=0) - Q(x=L1)\| / \|Q(x=0)\| <= 1e-8`, and the maximum cell divergence of the face velocities (recorded); device peak of medium (b) (solve, repack, Pollock) `<= 30000 MiB`. Fallback: if the PCG stalls above `1e-10` but reaches `<= 1e-8` with the flux-balance check met, that tolerance is used for the A/B and recorded; otherwise stop (owner). - **C. Medium A/B (decision rule; equivalence test).** For each amplitude and each primary statistic `P` of Implementation 3, with `Pbar_a`, `Pbar_b` the means over the ten paired realizations: - ratio statistics `S_T`, `S_tau`, `V1`: `R = ln(Pbar_a / Pbar_b)` with the 90 % confidence interval of the delta method on the paired realizations, `R +- t_{0.95,9} SE_R`, `t_{0.95,9} = 1.833`, `SE_R = sd_r(P_{a,r}/Pbar_a - P_{b,r}/Pbar_b) / sqrt(10)`; margins `+-ln(1.05)` (`S_T`), `+-ln(1.025)` (`S_tau`), `+-ln(1.05)` (`V1`). Absolute floor for `S_T` (positions in units of `lambda`): if `Pbar_b < 0.05 lambda` (`= 2.5e-3 lambda / 0.05`) or `Pbar_a <= 0`, `S_T` is tested on the paired difference with margin `+-2.5e-3 lambda`. If `Pbar_a <= 0` or `Pbar_b <= 0` for `S_tau` or `V1`, the statistic is inconclusive; - paired differences (floor case and tilts): CI `Dbar +- 1.833 sd_r(D_r) / sqrt(10)`, `D_r = P_{a,r} - P_{b,r}`; - tilts `theta_2`, `theta_3`, each component separately: paired difference, 90 % CI, margin `+-0.005` (D19); - a statistic is **equivalent** if its CI lies inside the margin, **non-equivalent** if its CI lies entirely outside, and **inconclusive** otherwise. Medium (a) is selected iff all ten primary statistics (both amplitudes) are equivalent; otherwise (any non-equivalent or inconclusive statistic) medium (b) is selected (R4). Every E1/E2 statistic, `D_r` and CI is recorded. The boundary-layer thickness of (b) (distance from the faces beyond which its `var_i(v1)/qbar^2` profile stays within `2 SE` of its interior mean, SE = the standard deviation over the ten realizations of the (b) plane value / `sqrt(10)`) is recorded. - **D. Ensemble runner.** A rerun of a realization is bitwise identical; a job killed mid-realization and restarted produces byte-identical final outputs (flux-weighted series, twin series and Pollock bridge included); the CSVs read by `compute_macrodispersion` through the fixed `read_timeseries_csv` reproduce the in-process moments to `<= 1e-12` relative; per-realization wall time recorded. - **E. Par2 baseline pilot.** The three configs run `N_R = 10` each to completion with outputs preserved; time per realization, `alpha_L(t)` series, `t_b` (first snapshot with a particle at `x >= L1 - 5 lambda`) and the realized variance of `ln K` per realization recorded; the paired `dt/2` check recorded (no threshold). - **F. Regression.** SF-35/SF-36/SF-37 cube outputs byte-identical; full V100 `ctest` green, with the single expected test change of D13 (`lagrangian_tracker_gpu_host`); SF-27 trio byte-compare clean; no existing config changed. - **G. Recorded.** SF-18 raw variance and applied scale on the long box for seeds 4001-4010 (input to SF-39's `normalize_variance` choice). **Compute (decision 2026-10-23 D16).** One V100; every remote job serialized on the SF-38 mirror; no GPU-hour ceiling; the GPU-hours of each part are reported in the experiment note and in the closing bitácora row. | Human-review increment (spec). DAG (runtime `.claude/orchestration/SF-38-long-domain-infrastructure/dag.json`): {N1 CSV reader fix ∥ N2 medium (b) library + E1 statistics ∥ N5 Par2 configs + field-statistics tool ∥ N6 A controls} -> N3 ensemble runner (on N1) -> N4 A/B application (on N2, N3) -> N7 campaign tooling -> I1 integration -> V100 campaign (orchestrator jobs: J0 base references, then controls, prodgrid, bpilot, ab, ensemble, field-only, par2-pilot, final) -> N8 experiment note -> I2 -> FINAL_AUDIT. | Launch N1, N2, N5, N6. |
| 2026-10-23T17:26Z | `active`; delivery branch at A0 `e4da02b`; J0a done; no evidence run yet | Accepted after orchestrator audit (records `.claude/orchestration/SF-38-long-domain-infrastructure/audits/`): N1 `6ff6d09` (`read_timeseries_csv` skips `#` lines then one header; test requires exactly T rows, rel 1.7e-16); N2 `36e43a3` (`DirichletMedium`: K = exp(eps Y) as the stage, Dirichlet `G1 L1`/0, periodic y/z, explicit `mg_levels`, FlowWorkspace freed before faces; `EulerianFaceStats` per RR5; layered control rel 4.7e-14); N5 `c134c16` (three Par2 configs per C-PAR2, `par2_field_stats` reproduces the pipeline `K stats` 6/6 on tiny configs); N6 `2cbeba9` + corrective `8c29cd5` (A block only on explicit `--only A`; round 1 ran it in the default ctest, MAJOR, corrected); N3 `c725c9e` + corrective `be499e3` (round 1 aborted the run on any non-ok particle and wrote an absolute `output_dir`, MAJOR; corrected: non-ok particles recorded in a deterministic `status.json` and the `STATUS` line, realization written, exit codes 5/6 removed, `analysis_config.yaml` with `output_dir: "."`). Remote: job `sf38-j0a-base-build` attempt 1 (16:20Z) ran after a failed sync (Par2_Core submodule not populated in the delivery worktree) on an empty mirror, `BUILD_EXIT=1`, cancelled, not evidence; attempt 2 (16:22-16:25Z, GPU 0, tree `e4da02b` == `d8ffa7c` in sources) `BUILD_EXIT=0`, base `inlet_slab` md5 `88526e87410416e2b5e91038e8406f76`, `macroflow3d_pipeline` md5 `f5f5204efff7c923f3e85e07e93cfda4` in `~/sf38_base_refs/bin`, `ctest -N` 43. **Reading clarifications (before any evidence run; no threshold changed):** (1) RR7 D-C (realization 4001 alone) is compared with D-A on the realization outputs (`realizations/r_0000/**`, `stats/` and `stats_twin/` CSVs of realization 0); run-level aggregates of a one-realization run (`analysis/`, `ensemble/`, `analysis_config.yaml`) differ from a three-realization run by construction; (2) the exit code 0 required by RR2/RR4/RR7 means the run completed; non-ok particles (tracker status >= 2, Pollock non-active) are recorded with their counts (`status.json`, `STATUS` lines) and never an exit condition (SF-39 "non-ok particles are reported, never silently dropped"). | N4 (A/B application) and N7 (campaign tooling) running. | Audit N4, N7; integration I1; campaign. |
| 2026-10-23T18:31Z | H1 `ddbfc6e` (source-bearing integrated head; delivery branch fast-forwarded) | Accepted: N4 `0e013e8` (`lagrangian_medium_ab`: (a) lean stage with faces and host `Y`, (b) `DirichletMedium` with `G1 = G[0]`, E1, Pollock plane passes, primary statistics, C0 fallback / exit 8, RR12; ctest 36 s incl. the `k = 1` control (media equal <= 1e-8)) and N7 `76c3473` (campaign tooling `run_sf38.sh` parts baserefs/controls/prodgrid/bpilot/ab/ensemble/fieldonly/par2pilot/final, `ab_decision.py` = threshold C, `c0_verdict.py`, `b_verdict.py`, `d_checks.py`, `par2_pilot.py`, `g_table.py`, SF-40 copies). I1 (single integrator): N1 `137c460`, N2 `758e859`, N3 `0cbf7de`+`533a7a9`, N5 `ac84476`, N6 `9c04d49`+`51be77f`, N4 `6d1fa97`, N7 `ddbfc6e`; patch-ids equal to the approved commits except `CMakeLists.txt` context in three (non-CMake patch-ids equal); no integration change. Local (light): ten targets built; ctests `lagrangian_tracker_gpu_host` 16.7 s, `lagrangian_dirichlet_medium` 1.8 s, `lagrangian_ensemble` 29.0 s, `lagrangian_medium_ab` 22.6 s, `lagrangian_long_box` 47.5 s; `long_box_tests --only A` `A_VERDICT all_pass=1` (30); dry runs (not evidence) of `bpilot`, `ab` (real `B_SOLVER` and `ab_primary.csv` parsed) and `ensemble` (kill/restart, 99/99 identical). **Reading clarifications ratified (before any evidence run; no threshold changed):** the `dt/2` copies also double `sample_every` and `output_every` (same physical sampling times); D-C excludes `timings.json` like D-B; the `TTIME` time column is compared at 1e-9 (printed `%.10g`), moments at 1e-12; "inside the margin" is the closed interval; `S_T`, its floor trigger `0.05 lambda` and margin `2.5e-3 lambda` are in box units (`lambda = ell`); the `analysis_only` reproduction runs on a copy of D-A reduced to `stats/` and `analysis_config.yaml`; the `git diff` item of F is checked by the orchestrator; the `ab` part refuses to start without a C0 PASS (failure policy). Remote: `sf38-j0b-baserefs` (base references) running since 18:21Z on GPU 0 from a clean tree at `76c3473` with the base binaries. | Campaign on H1. | J0b -> sync H1 -> head build -> parts in order. |
