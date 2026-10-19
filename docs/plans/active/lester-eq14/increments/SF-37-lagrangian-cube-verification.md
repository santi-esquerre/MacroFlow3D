# SF-37 — Verification of the Lagrangian consumer on the cube at sigma^2 <= 4, with blind O3

- State: `active`
- Goal: `Verificar en el cubo periódico, con semillas nuevas y umbrales pre-registrados, que las estadísticas de la pluma del tracker lagrangiano convergen en tolerancia y en h para sigma^2 en {1, 2.25, 4}, contrastarlas con las etiquetas en grilla verificadas de SF-35 (sigma_Y <= 1) y con el comparador de Pollock sobre las caras de SF-19, fijar con evidencia el lambda/h de producción y aplicar a ciegas la métrica robusta O3 de las etiquetas en grilla.`
- Depends on: `SF-36`
- Unlocks: `SF-39`
- Branch: `science/lester-sf37-lagrangian-cube-verification`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 4 (tolerance and grid refinement, independent representation comparator) + Gate 3A diagnostics of the gridded labels (O3)`
- Human review: `required`
- Owner: `Claude Opus 5.5 orchestrator (session 2026-10-19)`
- Started: `2026-10-19T13:35Z on base 44d4697`
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
| 2026-10-19T13:35Z | activation on `master=44d4697` (PR #55 squash-merged 2026-10-19T12:36Z, SF-36 `done`; checker OK ready=SF-34 SF-37 SF-38, nonterminal=none); delivery branch `science/lester-sf37-lagrangian-cube-verification`; remote `scripts/remote --increment SF-37` (one mirror, one job at a time), outputs under `~/sf37_campaign` on the V100 host | UNDERSTAND (runtime record `.claude/orchestration/SF-37-lagrangian-cube-verification/understanding.md`; scientific-rigor categories B + C; plan reviewed adversarially by three independent reviewers - scientific, feasibility, compliance - before this row). **Pre-registration before any run** (no threshold, design value, Goal or failure-policy text changed; reading rules resolve gaps inside the listed scope, SF-35/SF-36 precedent): PRE. Pre-registered, unchanged, verbatim by reference: SF-37 spec blob e6e0acb98753dbb0a32a1c268acf7cb4f8a9b063 (identical at 040e3ce and 44d4697): "In scope" Design bullet, V1-V5/O3/V3a/V3b ladder bullets, "Implementation specification" items 1-8, every "Acceptance thresholds" bullet incl. the Gate 4 mapping, "Failure and rollback policy". Key values restated: SF-18 Gaussian ell = 1/4, sigma2 = 1 normalized (raw variance recorded), seeds 3101/3102/3103, eps {1, 1.5, 2}, N {32, 40, 64, 80, 128, 256, 512}, M = 256 (65536), flux weights, planes {0.25, 0.5, 1, 2, 4, 8, 16} (unwrapped), times {0.25, 0.5, 1, 2, 4, 8, 16}, primary (h/8, 1e-8); V1 at N {256, 512} tol {1e-6..1e-10} at h/8 + reference (h/32, 1e-10); bootstrap blocks = 16 ell x ell squares (64 x 64 seeds), B = 2000, generator seed 20261017 -> committed index matrix analysis/bootstrap_blocks.npy (int64 (2000,16) = default_rng(20261017).integers(0,16,size=(2000,16)); sha256 16c1a5b9d9fe062576d5f18fc9d54127db76472c7fbe64493201d8cffc66af39, identical on numpy 2.5.0 and 1.26.4); q = 1e-3; V1 1e-3 / 2 SE_D; V2 order >= 1.5 or 2 % / 2 SE_D; lambda/h 5 % / 2 SE_D; V3b 5 % / 2 SE_D; V4 order >= 1.8; O3a order >= 1.8, reject < 10 %. RR1 (contributors). A particle contributes at plane m (time n) iff it landed there in that run (n_reached > m, as the SF-36 moments); seed_backflow has weight 0; each setting uses its own contributors and flux weights renormalized over them (biased weighted variance). Digest moments must reproduce moments_planes.csv / moments_time.csv (and the Pollock CSVs) to <= 1e-12 relative (analysis consistency, not a science threshold). RR2 (quantities). sigma_T^2 = (var x2 + var x3)/2 (unwrapped); var tau; per time var x1, x2, x3. Paired separation of a (test) vs b (reference): dx_T^2 = ((x2a - x2b)^2 + (x3a - x3b)^2)/2 over common contributors, MS(dx_T) weighted with b's weights renormalized over them, RMS(dx_T) = sqrt(MS); sigma_T = sqrt(sigma_T^2(b)). "Far plane" of a pair = the first plane with RMS(dx_T) > 0.1 sigma_T and every later plane. RR3 (SE_D). Block of p = (m2 // 64) * 4 + (m3 // 64), p = m2 M + m3. The committed 2000 x 16 index matrix is used for every paired comparison; per resample Q_a*, Q_b* recomputed on the resampled multiset (own contributors, weights renormalized); SE_D = std(D*, ddof = 1). Per-block shifted weighted sums; sequential accumulation in index order. RR4 (orders). Statistic Q on a ratio-2 triple: spec item 4; nan iff a difference is exactly 0 or the two differences have opposite signs. Error quantity vanishing in the limit (V4 RMS e_v, O3a e_v^(q)) on a pair (N, 2N): log2(e_N / e_2N) (SF-35 orders_digest convention); "decreases on every refinement" = e_2N < e_N on every consecutive pair; a missing or non-finite grid value -> not_evaluable (never unresolved). RR5 (production lambda/h). Literal spec rule, primary: the smallest candidate of {8, 10, 16, 20, 32, 64} satisfying the inequality at every plane, Q in {sigma_T^2, var tau} and seed, SE_D of the pair (N, 512), evaluated on contributors. A candidate is not qualifiable only if a needed run has no data (darcy_failed, exit 18, exception); non-ok fractions are reported, never used to disqualify. Recorded: the monotone variant (smallest candidate such that it and every larger candidate pass); a difference -> owner. "none <= 64" -> stop handling (RR14). RR6 (V3a). Criterion C on the eps = 1 labels at N {64, 128, 256} by the SF-35 orders_digest.py invoked in place (--expected-ns 64,128,256; global e_v, e_i1, e_i2, e_div, r_F of the --summary JSON); exit 19 or 17 -> not evaluable. Interpolation on grid plane j = x1 N (x1 in {0.25, 0.5, 0.75, 1}): U1 = psi1 - x2_m, U2 = psi2 - x3_m ((N, N), no duplicate), scipy.ndimage.map_coordinates(U, [(x2 mod 1) N, (x3 mod 1) N], order = 3, mode = 'grid-wrap', prefilter = True); psi_grid = (x2_u + U1, x3_u + U2) with unwrapped coordinates; error = Euclidean norm of psi_grid - psi_p (psi_p = labels.npy at the seed). Classified quantity: UNWEIGHTED RMS and max over contributing particles with finite psi_p (concordant iff the RMS decreases on 64 -> 128 and 128 -> 256 at that plane). O6 data: dedicated runs (part v3a: eps 1, N {64, 128, 256}, planes 0.25, 0.5, 0.75, 1, primary setting), since 0.75 is not a ladder plane. RR7 (V3b). Pollock and O6 are paired by particle index inside the same --comparator pollock process (same stage, keep_face_fluxes = true); Pollock weights = max(RT0 v1 at the seed in the cell chosen by pollock_init_state, 0), normalized; SE_D by RR3; Pollock non-active particles (12/14/15) contribute only where they landed and are counted. RR8 (V5). From the V1 runs (--covectors --save-seed-velocity). Covector norm percentiles 50/90/99/max of \|a\|, \|b\| per plane, unweighted over contributors (flux-weighted recorded too); rate per particle per plane log(\|a\|/\|a0\|)/tau with a0 = (-v2, v1, 0), b0 = (-v3/v1, 0, 1) from seed_v.npy. Fits per (eps, seed, N) on the planes with 1e-12 < RMS(dx_T) < 0.1 sigma_T (pair 1e-8 vs 1e-10), >= 3 eligible planes else insufficient; least squares of log RMS(dx_T) on x1 (exponential) and on log x1 (algebraic), RSS reported; amplitude name = majority over the evaluable (seed, N) cases, mixed on a tie, insufficient if fewer than 2 evaluable. RR9 (O3). Pair (N/2, N): common vertices = the N/2 lattice on interior coarse planes j = 1..N/2-1 with all (N/2)^2 transverse vertices; c by label_diagnostics.py section 3 on each lattice; d and RMS\|c_N\| over the common vertices; M_q(N) = the ceil(q n_common) largest d (ties by vertex index); O3a e_v^(q)(N) = RMS over unmarked common vertices of \|c_N - vD_N\| / v_rms(N), v_rms(N) = RMS\|vD_N\| over ALL common vertices; vD_N from the forward-mode stage export (--save-vd in the v2 runs at the same field, eps, N; the stage is the same deterministic build_production_stage); O3b f_half(N) = label_diagnostics frac for the pair (cross-checked against label_diagnostics.py on the same exports) and the fraction with d > 0.1; O3c per plane x1 in {0.25, 0.5}: weighted fraction (O6 weights renormalized over contributors, v2 runs at the same eps, seed, N) of landings within one coarse cell (periodic \|dx2\|, \|dx3\| <= 2/N) of a marked vertex of that plane, plus the pooled fraction over both planes; x1 = 1 is the outlet coarse plane (no marks by construction): reported structurally empty. A construction with exit 17 or 19, or a missing N -> O3 not evaluable for that seed and amplitude. o3.py and RR9 are frozen (commit recorded) before any eps 1.5/2 export of seeds 3101-3103 is read; until then the orchestrator observes the label jobs by status/wait only. RR10 (scope clarifications, opt-in, default outputs byte-identical). The SF-35 trace path is not modified. Driver --track forward gains: --comparator pollock (spec); --save-vd DIR (vD1/vD2/vD3.npy (N+1, N, N) slab layout + vD.json, the stage's reference vD) for O3a; --save-particles DIR --save-seed-velocity (seed_v.npy (Np,3) = k g at the seeds, host stage.direction_field()) for V5; exit 23 comparator_fail. Library: FaceFluxRepack.{cuh,cu} (spec) and PollockComparator.{cuh,cu} (new, beyond the spec's file list: the per-particle Pollock lattice tracker shared by the ctest and the driver). Disclosed in the PR as files beyond the list. RR11 (settings). All forward runs: --sigma2 1 --ell 0.25 --seed S --np-side 256 --max-arclength 1000 (the default 4*16+10 = 74 is a guard too small for the time pass to t = 16; the cap never alters an accepted trajectory; cap hits are counted). Primary --tol 1e-8 --hmax-div 8. v2: all N (N = 40, 80 first), --save-vd at eps 1.5/2 for N {128, 256, 512}. v1: N {256, 512}, --tol {1e-6, 1e-7, 1e-8, 1e-9, 1e-10} --hmax-div 8 and --tol 1e-10 --hmax-div 32, all --covectors --save-seed-velocity. v3b: --comparator pollock at N {64, 128, 256, 512}. Constructions (V3a, O3): --construct trace --tracer gpu --threads 32 --sigma2 1 --ell 0.25 --seed S --save-labels --summary with the SF-35 defaults (h/8, 1e-8, max_roundtrip 1e-8), run with the base-commit binary (md5 recorded; criterion R proves the head equal on this path); exit 21 accepted (labels saved), 17/19 -> not evaluable. Pollock: SF-32 core defaults. RR12 (failures). No setting changes inside a job; a scientific failure is never re-run; an infrastructure interruption (ssh drop, host reboot, script crash before a case's exit code exists) is resumed with identical settings (cases with outputs and an exit code are skipped), disclosed, superseded attempts excluded. A run with exit 22/23 contributes its contributors and is listed. RR13 (criterion R). Head vs ~/sf37_base_refs with identical arguments: --construct trace (32/64 x eps 0.5/1.0 x host/gpu, 64 eps 2 gpu): psi1/psi2.npy cmp, labels.json without transport.seconds_*, lines CASE PLANE BACKFLOW GROWTH RESIDUAL TRANSPORT_STATUS STATUS, exit code; --production 32 eps 0.25 --oracle-ladder: CASE ORACLE ORACLE_SUMMARY EXTRA lines; --track forward (generic3d 16 smoke, generic3d 32 cov, gauss 64 eps 1 cov, eps 2, eps 2 tol 1e-10 h/32 arclength 1000, gauss 40 eps 1, gauss 80 eps 2): every .npy, both CSVs, track.json without timings_s, lines SETUP TRACK BACKFLOW TPLANE TTIME TSTATUS STATUS ("built in" masked), exit code; the same forward cases with --comparator pollock / --save-vd / --save-seed-velocity: every pre-existing file and line identical; SF-27 trio (final job). Never compared: summary command, TIMING/MEMORY lines, stage seconds, nvidia-smi peaks. Binary check after the head build: md5 differs from base and the usage text lists --comparator. RR14 (stop handling). All pre-registered parts run to completion whatever the intermediate verdicts (no outcome-dependent stopping, nothing re-run or tuned). After the final audit: if V1 or V2 is not met at any amplitude, or V4 is not met, or the production lambda/h is "none <= 64", or a blocking cell is not evaluable -> State `blocked`, a bitácora row naming the rule (R1 for V1 at sigma^2 = 4), checklist item 2 unchecked, PR for the owner with the stop in title and body. Otherwise `awaiting_review`, with every non-blocking outcome reported explicitly (V3a not concordant / not verified, V3b discordant, O3 reject / unresolved / not shrinking, V5 exponential, V1/V2 passing only through 2 SE_D at every far plane, V1 vacuous, any shadow classification not met). RR15 (V1 vacuity, scientific review F1). Because the step is capped by h_max (SF-36 J), the 1e-8 and 1e-10 runs can take bitwise-identical step sequences; V1 is then met with D = 0 without measuring integration error. Recorded per (N, eps, seed, tol): track.json pass totals (nfev, accepted, rejected, landing steps/rejections), the weight fraction of particles with records bitwise identical to the 1e-10 run, the first plane with D != 0. If every record is identical, V1 is labelled "met, vacuous (tol-independent step sequence)" and is not presented as evidence that the integration error is controlled. Shadow classifications (recorded, not blocking; reported to the owner when V1 is met and a shadow is not): the V1 inequality, MS(dx_T)/sigma_T^2 and the first 0.1 sigma_T plane for (h/8, 1e-10) vs (h/32, 1e-10) (pure h_max effect) and (h/8, 1e-8) vs (h/32, 1e-10). V5 fallback when the 1e-8/1e-10 pair is insufficient: the same fits on those two pairs and on (1e-6, 1e-10), reported as fallback, never as the named result. | Recorded diagnostics, no thresholds: D1 (multiplicity, F2/F3): per cell z = D/SE_D, binding branch (det / stat / order / fail), 2 SE_D/\|Q_ref\|; number of cells in the statistical regime and expected chance exceedances (~5-6.5 % per cell for t_15); sign agreement of failing cells across seeds and adjacent planes; SE_D with 4 and 64 blocks (sensitivity); effective sample size (sum w)^2 / sum w^2 per plane; seed-pooled z; Richardson residual for V2's order branch and the flag "order from differences below 2 SE_D". D2 (pairing, F4): applied_scale and raw_variance per (seed, N) (equal to <= 1e-12 rel. across N); nfev per unit length. D3 (attribution, F5/F7): at N 256/512, \|Q(h/8,1e-8) - Q(h/32,1e-10)\| vs \|Q(256) - Q(512)\|; V2 inequality and V4 order on the reference runs; \|RMS e_v(h/8) - RMS e_v(h/32)\| / RMS e_v (passive-I error). D4 (V4 robustness, F6): flux-weighted percentiles 50/90/99/99.9/max of \|e_v\| and \|I\|; RMS I; share of sum w e^2 carried by the top 10 particles and top 0.1 % weight; V4 test on RMS I and on the 99.9 %-trimmed RMS; block-bootstrap SE of log2(RMS_256/RMS_512); correlation of \|I\| with tau; triple order. D5 (V3a, F8): interpolation proxy (U_N interpolated at the 2N lattice of the same plane vs U_2N), V3a RMS as a multiple of it; flux-weighted RMS; fraction above 10x proxy; constructor exit/round-trip; stage identity trace vs forward (G, inlet_vmin, v_rms at print precision). D6 (V3b, F9): D_P split into weight part sigma^2(P-traj, P-w) - sigma^2(P-traj, O6-w) and trajectory part sigma^2(P-traj, O6-w) - sigma^2(O6), each with SE_D; signed D_P per N; noise-limited flag per refinement; slope of D_P in x1 and its scaling with N (the §5A leakage measurement); D_P on common contributors; Pollock non-active weight. D7 (O3b, F10): N_half and its growth factor per doubling (~8 uniform, ~4 sheet, ~2 curve), d_max and location, RMS d over unmarked vertices with pair order; reading note: f_half "shrinking" means concentration, not convergence. D8 (SF-39 estimator, F13): shadow V1/V2/lambda-h classifications on var x_k(t), on Delta var_k / Delta t and on Delta sigma_T^2 / Delta x1; var x2 and var x3 separately; per-plane/per-time RMS(dx_T)/sigma_T budget for the tol, h_max, grid and Pollock pairs; local growth exponents. D9: non-contributing weight fraction per plane and setting; differences recomputed on common contributors; cap hits (count, weight) per pass; bitwise identity of V1 (1e-8, h/8) vs V2 primary positions/tau/I/statuses (reproducibility). DAG: ACT -> J0 (orchestrator job `sf37-base-refs`: base build of 44d4697 on the SF-37 mirror before any SF-37 source reaches it, R references, 512^3 forward feasibility) ; {N2 keep_face_fluxes \| N3 FaceFluxRepack + PollockComparator + `lagrangian_face_flux_repack` \| N5 digests + selftests \| N6a `run_sf37.sh` label parts} -> J1 (orchestrator jobs `sf37-labels-eps10`, `sf37-labels-o3` with the base binary) ; N4 driver (after N2, N3) -> N6b ladder/bytecmp/digest parts -> I1 -> [J1 done] -> J2 (orchestrator jobs `sf37-bytecmp`, `sf37-v2`, `sf37-v3a`, `sf37-v3b`, `sf37-v1`, `sf37-digest` at H1; order and extra parts deviate from the spec's validation loop, recorded) -> N7 evidence import -> N8 experiment note -> I2 -> J3 (`sf37-final`: full V100 ctest + SF-27 trio) -> FINAL_AUDIT -> State per RR14 -> PR. Workers never use the remote; V100 jobs are orchestrator jobs launched from clean worktrees at accepted commits running the committed scripts (departure from the SF-35/SF-36 worker-campaign precedent, disclosed). Estimated ~46 GPU-h, ~50 h wall serialized; using both GPUs concurrently would need an explicit owner authorization (not given). Persistent-goal feature not available in this runtime. | Launch J0 and the G1 workers (N2, N3, N5, N6a); J1 after J0 and N6a are accepted. |
