# SF-37 — Verification of the Lagrangian consumer on the cube at sigma^2 <= 4, with blind O3

- State: `pending`
- Goal: `Verificar en el cubo periódico, con semillas nuevas y umbrales pre-registrados, que las estadísticas de la pluma del tracker lagrangiano convergen en tolerancia y en h para sigma^2 en {1, 2.25, 4}, contrastarlas con las etiquetas en grilla verificadas de SF-35 (sigma_Y <= 1) y con el comparador de Pollock sobre las caras de SF-19, fijar con evidencia el lambda/h de producción y aplicar a ciegas la métrica robusta O3 de las etiquetas en grilla.`
- Depends on: `SF-36`
- Unlocks: `SF-39`
- Branch: `science/lester-sf37-lagrangian-cube-verification`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 4 (tolerance and grid refinement, independent representation comparator) + Gate 3A diagnostics of the gridded labels (O3)`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Decision record `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`, items 2-4 and 8, stop rules R1, R2, R5.
The Lagrangian consumer (SF-36) has no structural guarantee against integration drift (source (B), theory note §5)
and its representation `v_s` is helicity-free with a divergence expected to be `O(h^2)` (source (A) reduced to
divergence; measured here by V4). SF-37
measures, on the cube where every comparison is affordable and on **fresh** realizations, whether the plume
statistics the study needs converge in `tol` and in `h` at `sigma^2` in {1, 2.25, 4}, and fixes the production
`lambda/h`. It is allowed to establish:

- (a) per amplitude, the classification of the tolerance-induced (V1) and grid-induced (V2) changes of the plume
  statistics against pre-registered thresholds, and the production `lambda/h`;
- (b) the agreement of the consumer with the gridded SF-35 labels where those are verified (V3a, `sigma_Y = 1`) and
  with an independent representation (Pollock on the SF-19 face fluxes, V3b), and the Lagrangian `e_v(h)` (V4);
- (c) the stretching and decorrelation behaviour that decides whether long-range criteria must be statistical (V5);
- (d) a blind verdict of the robust metric O3 of the gridded labels at `sigma^2` 2.25 and 4.

No claim about `alpha_L` or `alpha_T` (cube statistics are periodic-medium statistics; the study is SF-39), and
none presupposed (Rule 1).

## Preconditions

- `SF-36` is `done` on the default branch (forward tracker, `inlet_slab --track forward`, P2, covectors, controls).
- SF-35 constructor available (`--construct trace --save-labels`), SF-32 Pollock tracker (`PollockTracker.cuh`:
  `pollock_advance_to_x1`, `pollock_advance_to_time`; class `PollockTracker`, `step_to_x1`, `step`).

## In scope

- **Design (fixed here, pre-registered).** SF-18 Gaussian, `ell = 1/4`, `sigma2 = 1` with `normalize_variance` (the
  SF-35 convention; raw variance recorded), seeds **3101, 3102, 3103** (seed 3001 was used for exploration and is
  excluded), `eps` in {1, 1.5, 2} (`sigma^2` in {1, 2.25, 4}), `k = exp(eps Y)`. Grids `N` in {32, 64, 128, 256,
  512} (`lambda/h` = 8-128 at fixed `L`; SF-18 gives the same continuum at fixed `L`), plus the production-candidate
  points `N` in {40, 80} (`lambda/h` = 10, 20). Injection on the whole face `x1 = 0`, `M = 256` (65536 particles),
  flux weights. Planes `x1` in {0.25, 0.5, 1, 2, 4, 8, 16} (unwrapped; 1 = one period = `4 ell`); snapshot times
  `t` in {0.25, 0.5, 1, 2, 4, 8, 16}. Primary setting `(h/8, 1e-8)`.
- **V1 tolerance ladder** at `N` in {256, 512}: `(h/8, tol)` for `tol` in {1e-6, 1e-7, 1e-8, 1e-9, 1e-10} and the
  reference `(h/32, 1e-10)`.
- **V2 grid ladder** at the primary setting on all `N`.
- **V3a (O1 anchor)** at `eps = 1`: SF-35 gridded labels of each new seed at `N` in {64, 128, 256} with SF-35
  criterion C evaluated on them; `|psi_grid(x_p) - psi_p|` at the grid planes `x1` in {0.25, 0.5, 0.75, 1}.
- **V3b Pollock comparator**: stage option `keep_face_fluxes`; repack kernel CompactMAC (`U` `(nx+1) ny nz`, `V`,
  `W`; periodic duplicate plane dropped) -> `PeriodicFaceFluxView` (`i + nx (j + ny k)`); Pollock on the same seeds
  (weights from its own `v1`), planes and times, at `N` in {64, 128, 256, 512}, all `eps`.
- **V4** Lagrangian `e_v = 1 - exp(-I)` on the V2 runs. **V5** covector statistics and paired decorrelation on the
  V1 runs (covectors on).
- **O3 (blind)**: SF-35 gridded labels at `eps` in {1.5, 2}, `N` in {64, 128, 256, 512}, seeds 3101-3103; metric
  defined below before any run.
- Analysis scripts (bootstrap, orders, O3, V3a interpolation, V5 fits), V100 campaign, experiment note, artifact.
- Cost (from the probes, for planning; recorded, not a threshold): a 512^3 SF-35 construction costs ~3.9 h on the
  GPU tracer (O3 needs six: 2 amplitudes x 3 seeds), field + stage ~2 min per 512^3 forward run; label exports
  1.08 GB per field at 512^3 stay on the V100 host.

## Out of scope

- Any change to the SF-36 tracker numerics, SF-35, SF-30, SF-32 (read-only use; the repack is a new kernel), the
  frozen periodic stack, SF-18/SF-19.
- The long domain, ensemble runner, Par2 (SF-38); `alpha` estimation (SF-39); pseudo-symplectic tracking
  (deferred, decision item 5).

## Files and symbols

- New: `src/physics/particles/lagrangian_labels/FaceFluxRepack.{cuh,cu}` (`repack_compact_mac_to_periodic_faces`);
  `tests/lagrangian_labels/face_flux_repack_tests.cu` (`lagrangian_face_flux_repack`, `<= 60 s`: repacked faces
  equal the unique CompactMAC planes bitwise; Pollock on `k = 1` moves on straight lines);
  `docs/experiments/<date>-sf37-lagrangian-cube-verification.md` and artifact
  (`scripts/{run_sf37.sh, digest_v1v2.py, digest_v3.py, digest_v4v5.py, o3.py}`, `logs/`, `raw/`, `analysis/`).
- Extend: `SlabProductionSetup.{cuh,cu}` (`keep_face_fluxes`, default `false`, outputs unchanged otherwise),
  `apps/inlet_slab/inlet_slab_driver.cuh` (`--track forward --comparator pollock`), `CMakeLists.txt`.
- Read-only: `LagrangianTracker`, `PollockTracker`, `SlabLabelTransport`, `SlabMetrics`, SF-35
  `label_diagnostics.py` (as the reference implementation of the `c` self-convergence of `LD` section 3).

## Implementation specification

1. **Pairing.** Every ladder comparison uses the same field realization, the same stage build (deterministic) and
   the same fixed seed lattice; per-particle differences are available for every pair of settings.
2. **Quantities per plane `x1`** (weighted over `ok` particles): `sigma_T^2 = (var x2 + var x3)/2`,
   `var tau`, mean `tau`; per snapshot `t`: `var x1`, `var x2`, `var x3`. Statuses and non-ok fractions recorded.
3. **Standard error of a paired difference** `SE_D`: block bootstrap per realization, blocks = the `ell x ell`
   squares of the injection face (16 blocks of 64 x 64 seeds), `B = 2000` resamples with replacement, generator
   seed 20261017, the difference recomputed on each resample with the weights; `SE_D` = bootstrap standard
   deviation.
4. **Observed order** `p = log2(|Q_a - Q_b| / |Q_b - Q_c|)` on a refinement triple with ratio 2; `nan` when a
   difference changes sign or vanishes (then the triple cannot certify an order).
5. **V3a interpolation.** On grid plane `j` the label fluctuations `U_i = psi_i - x_i` (periodic in `x2`, `x3`) are
   interpolated at the particle's landed `(x2, x3)` by a periodic bicubic B-spline (SciPy
   `ndimage.map_coordinates(order=3, mode='grid-wrap')`, an implementation independent of SF-28; vertex lattice
   `m/N`); `psi_p` = the particle's D-1 label at its seed.
6. **O3 metric (blind; fixed here).** For a pair `(N/2, N)`: on the common vertices of interior planes
   `j = 1..N/2-1` compute `c` with the second-order central differences of `label_diagnostics.py` section 3,
   `d(x) = |c_N(x) - c_{N/2}(x)| / RMS|c_N|`; the marked set `M_q(N)` = the top `q = 1e-3` fraction of the common
   vertices ranked by `d`. **O3a**: `e_v^(q)(N)` = RMS over the unmarked common vertices of `|c_N - vD_N| / v_rms(N)`
   for `N` in {128, 256, 512} (marks from (64,128), (128,256), (256,512)). **O3b**: `f_half(N)` = fraction of
   common vertices carrying half of `sum d^2`, and the fraction with `d > 0.1`. **O3c**: weighted fraction of O6
   particles whose landing on a grid plane `x1 <= 1` lies within one coarse cell of a marked vertex. Disclosure:
   `q` and the ranking were chosen after the SF-35 post-hoc inspection of seed 3001 (`LD`, `LD20`); the application
   to seeds 3101-3103 is blind.
7. **V5 fits.** On the planes where `1e-12 < RMS(dx_T) < 0.1 sigma_T` (paired 1e-8 vs 1e-10 separations): least
   squares of `log RMS(dx_T)` against `x1` (exponential) and against `log x1` (algebraic); the residual sums of
   squares are reported and the better fit named. Covector norms: percentiles 50/90/99/max of `|a|`, `|b|` per
   plane and the finite-time rate `log(|a|/|a0|)/tau`.
8. **Failure behaviour.** No setting changes inside a job; non-ok particles are reported, never re-run.

## Expected numerical effect

No existing output changes. Expected (not presupposed): V1 deterministic agreement at the early planes and a
transition to statistical agreement after a few periods at `sigma^2 = 4` (SF-30: trajectories unresolved after 4-8
periods); V4 order ~2; V3b Pollock excess decreasing with `N`; O3 not predicted.

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R 'lagrangian|inlet_slab'
./build/wsl-debug/inlet_slab --track forward --comparator pollock --analytic generic3d --n 16 --eps 0.25 --np-side 16 --planes 0.5,1 --times 1
scripts/remote --increment SF-37 sync
scripts/remote --increment SF-37 exec -- "cmake --preset v100-release && cmake --build build/v100-release -j 2>&1 | tail -n 20; echo BUILD_EXIT=${PIPESTATUS[0]}"
A=docs/experiments/artifacts/<date>-sf37-lagrangian-cube-verification/scripts
for part in labels-eps10 labels-o3 v2 v1 v3b; do scripts/remote --increment SF-37 run sf37-$part -- "bash $A/run_sf37.sh build/v100-release $part"; scripts/remote --increment SF-37 wait sf37-$part; done
scripts/remote --increment SF-37 run sf37-ctest-full -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote --increment SF-37 wait sf37-ctest-full
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

Pre-registered verbatim in the activation bitácora row before any run (with the design of "In scope", the
bootstrap settings, `q`, the planes and the times); no threshold or setting is tuned afterwards. `Q` ranges over
`sigma_T^2(x1)` and `var tau(x1)` at every plane, per seed.

- **V1 (tolerance, source B).** At `N` in {256, 512}, for every amplitude, seed and plane:
  `|Q(1e-8) - Q(1e-10)| <= max(1e-3 |Q(1e-10)|, 2 SE_D)` for `Q = sigma_T^2`. Recorded: the whole ladder, the
  reference `(h/32, 1e-10)`, the numerical transverse variance `MS(dx_T)/sigma_T^2` of the paired separations and
  the first plane where `RMS(dx_T) > 0.1 sigma_T`.
- **V2 (grid, source A).** For every amplitude, seed, plane and `Q`: observed order `>= 1.5` on 128 -> 256 -> 512,
  or `|Q(256) - Q(512)| <= max(0.02 |Q(512)|, 2 SE_D)`. **Production `lambda/h`** per amplitude = the smallest
  `lambda/h` of {8, 10, 16, 20, 32, 64} with `|Q(N) - Q(512)| <= max(0.05 |Q(512)|, 2 SE_D)` at every plane, `Q`
  and seed; the study uses the maximum over amplitudes.
- **V3a (O1 anchor, recorded with classification).** For each new seed at `eps = 1`, SF-35 criterion C on its
  gridded labels at 64/128/256 (`accept` on 128 -> 256) decides whether that seed's labels are a verified
  reference. Where verified: RMS and max of `|psi_grid(x_p) - psi_p|` per plane at each `N`, classified
  `concordant` if the RMS decreases on both refinements; otherwise `not concordant`.
- **V3b (independent representation, recorded with classification).** `D_P(N) = sigma_T^2(Pollock, N) -
  sigma_T^2(O6, N)` per plane: `concordant` if `|D_P|` decreases on every refinement 64 -> 128 -> 256 -> 512 and
  `|D_P(512)| <= max(0.05 sigma_T^2(O6, 512), 2 SE_D)`; otherwise `discordant`. Not blocking (Pollock's own
  convergence is not established); a `discordant` at any amplitude is reported to the owner in the PR.
- **V4 (Lagrangian `e_v`).** For every amplitude and seed, at every plane `x1 <= 4`: RMS `e_v` decreases on every
  refinement 64 -> 128 -> 256 -> 512 with observed order `>= 1.8` on 256 -> 512.
- **V5 (recorded, no threshold).** Covector percentiles and rates per plane; the decorrelation fits of item 7 and
  the named better fit per amplitude.
- **O3 (blind, recorded with verdict, not blocking for the consumer).** At `eps` 1.5 and 2 per seed: O3a `accept`
  if `e_v^(q)` decreases on every refinement with observed order `>= 1.8` on 256 -> 512, `reject` if it changes
  `< 10 %` between 256 and 512 while above 1e-12, `unresolved` otherwise; O3b `shrinking` if `f_half` decreases on
  every refinement, otherwise `not shrinking`; O3c recorded.
- **R. Regression.** `lagrangian_face_flux_repack` green; SF-35 and SF-36 outputs byte-identical with
  `keep_face_fluxes = false`; full V100 `ctest` green.
- **Gate 4 mapping.** Tolerance refinement = V1; grid refinement = V2, V4; independent comparator = V3b; label
  confinement where gridded labels are verified = V3a; the transverse classification itself is SF-39.

## Regression surface

- `build_production_stage` (new option), the driver, `ctest` wall time; nothing else in production code.

## Failure and rollback policy

- V1 or V2 not met at `sigma^2 = 4` -> stop rule R1: the consumer is not defensible at that amplitude; the increment
  returns to the owner with the ladders; nothing is tuned.
- V1 or V2 not met at `sigma^2` 1 or 2.25 -> stop; owner (the study's amplitudes depend on it).
- V4 not met -> representation defect or under-resolution; stop; owner.
- V5 names the exponential fit -> long-range criteria in SF-39 are statistical only (anticipated, R2); if V1/V2 then
  pass only through their `2 SE_D` branch at every far plane, that is recorded explicitly.
- V3a `not concordant` or a seed's labels not verified (C not `accept`) -> recorded; it narrows the O1 anchor to the
  seeds where it holds; owner informed.
- V3b `discordant` -> recorded; owner informed; not blocking.
- O3 `reject`/`unresolved`/`not shrinking` -> recorded; the gridded labels stay a diagnostic inside the O1 envelope.
- Inlet backflow on a new seed -> recorded (the consumer does not need D-1 labels; V3a for that seed is
  `reference not available`).

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

`SF-39` becomes eligible after this increment and `SF-38` are merged and marked `done` on the default branch.

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-17T19:30Z | not started | Specification created on the SF-35 closure PR (#53) after the owner decision of 2026-10-17 (`docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`). Thresholds V1, V2, V4 from the planning values with the statistical branch `2 SE_D` of the decision's item 8 written into V1/V2; V3a, V3b, O3 given recorded classifications; production-candidate grids `N` 40/80 added so that `lambda/h` 10 and 20 are qualified directly. | Depends on SF-36 `done` on `master`. | Activate only when the checker reports it READY; the activation row contains the design and thresholds verbatim and the base commit before any run. |
