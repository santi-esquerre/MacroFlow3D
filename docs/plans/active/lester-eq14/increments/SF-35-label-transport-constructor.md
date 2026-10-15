# SF-35 — Label transport by backward streamline tracing as the production constructor

- State: `pending`
- Goal: `Establecer el trazado regresivo de líneas de corriente desde la cara de entrada como constructor de producción de las etiquetas de Darcy en el slab, verificado sin trazar (controles con etiquetas exactas, métricas por diferencias finitas, residuo de la ecuación (14) como diagnóstico y concordancia con la solución elíptica donde existe) y con órdenes observados en 32^3-256^3 para sigma_Y en {0.5, 1, 1.5}.`
- Depends on: `SF-33`
- Unlocks: `SF-34`
- Branch: `science/lester-sf35-label-transport-constructor`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 2 + Gate 3A (tracing-constructor mapping, see "Acceptance thresholds")`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Owner decision of 2026-10-07 (`docs/decisions/2026-10-07-label-transport-constructor.md`): the Newton–Krylov solve of
equation (14) on the slab (SF-33) is not a viable production constructor (linear-solver plateau at 128^3 for
`eps >= 0.14`; 32^3 `eps = 0.5` fails; `docs/experiments/2026-10-06-sf33-gpu-inlet-labels.md`, Result). SF-35 adopts
label transport along backward-traced streamlines of the SF-19 Darcy flow (SF-28 spline, SF-30 DP5(4) integrator with
Hénon landing, D-1 inlet labels at the feet) as the production constructor of the inlet-anchored Darcy labels
`psi1`, `psi2` on the `x1`-slab, and is allowed to establish two claims:

- (a) the traced labels pass checks that do not trace: exact-label positive controls, grid finite-difference
  metrics `e_v`, `e_i`, `e_div`, the equation-(14) residual `r_F` as an independently derived necessary condition,
  and agreement with the elliptic SF-33 solution where that solution exists;
- (b) on the production field (SF-18 Gaussian, `ell = 1/4`, seed 3001, `sigma2 = 1`, `eps` in {0.5, 1, 1.5}, i.e.
  `sigma_Y` = `eps`, `sigma^2` in {0.25, 1, 2.25}) the FD metrics of the traced labels converge under refinement
  32/64/128/256, with the error growth along `x1` and the backflow statistics measured.

Equation (14) is a diagnostic here (`r_F`, `r_out` at the constructed labels), never an acceptance quantity by itself.
No claim about transport, the return map (SF-34) or `alpha_T`. Independence rule: decision record item 3.

## Preconditions

- `SF-33` is `done` on the default branch (the slab module `src/physics/streamfunctions/inlet_slab/`: production
  oracle `compute_oracle`/`trace_to_plane`, `InletLabels`, `SlabProductionSetup`, `SlabMetrics`, `SlabResidual`, the
  Newton–Krylov solver kept as an instrument, 6 fast tests).
- `SF-28`, `SF-30` are `done` (spline; `apps/closure_gate/streamline_integrator.hpp`, untouched by this increment).
- The decision record `docs/decisions/2026-10-07-label-transport-constructor.md` is `accepted` on the default branch.
- The Gaussian-covariance config PR is not a dependency (the periodic medium uses SF-18).

## In scope

- `construct_labels_by_tracing` (host backend) wrapping `compute_oracle`, decoupled from any Newton solve, with
  failure accounting (status histogram over `closure_gate::StreamlineStatus`, `seed_backflow`, `backflow_encounters`,
  `landings_discarded`, `min_g1_hat`), per-plane statistics, and export (`psi1.npy`, `psi2.npy` shape `(N+1, N, N)`
  float64 in slab layout, plus `labels.json` with field, `eps`, `N`, `h`, `h_max`, `tol`, backend, statuses, round
  trips, commit).
- Driver mode `inlet_slab --construct trace` (no Newton), with `--solve` (cross-construction with the unchanged SF-33
  solver at small `N`), `--ladder` (`h_max`/`tol` ladder), `--tracer host|gpu`, `--save-labels DIR`.
- Per-plane metrics (`e_v_j`, `e_i1_j`, `e_i2_j`, `min_c_j`, `vD1_min_j`, `vD1_nonpos_j`) as an additive function in
  `SlabMetrics`; `PLANE`, `BACKFLOW`, `GROWTH`, `LABELDIFF`, `XCONS` output lines; CASE line `cand=trace`.
- Positive controls with exactly known labels: `k = 1`; `k = exp(0.5 cos 2 pi x2)` (straight streamlines, non-trivial
  D-1 labels, via `SlabFieldSource::spectral`); `control2d` (`psi2 = x3`); a synthetic failure field (backflow,
  stagnation) for the status accounting.
- Cross-construction against the SF-33 elliptic solution at (`eps`, `N`) = (0.25, 32), (0.25, 64), (0.5, 32).
- GPU port of the tracer (non-blocking node): device copies of the DP5(4) arclength step and Hénon landing with the
  host expression order, `DeviceSplineDirectionField` on `evaluate_point`, one thread per seed, host/GPU equivalence.
- V100 campaign (matrix 3 amplitudes x 4 grids, ladders at 32/64, per-plane growth, backflow statistics, timing and
  memory), experiment note and artifact.

## Out of scope

- The frozen periodic stack (`src/physics/streamfunctions/*` outside `inlet_slab/`), `src/numerics/`, `src/multigrid/`,
  SF-18/SF-19, `apps/closure_gate/*` (SF-30 instrument: the device port is a copy, not a modification),
  `src/physics/particles/streamline_tracker/*` (read-only reuse allowed).
- Changes to the Newton–Krylov path beyond what `--solve` needs to call it unchanged.
- Consumer adaptation (SF-31 takes periodic labels; legacy PSPTA needs cell-centred float32): only the export format
  is defined here.
- The return map against SF-30 (SF-34), the long domain, transport, `alpha_T`.
- Handling of inlet backflow (`v1 <= 0` on the inlet face): detected (`inlet_backflow`), recorded, not handled.
  Interior backflow and stagnation are followed and counted; nothing is clamped or regularized.

## Files and symbols

- Extend (existing outputs bitwise unchanged): `src/physics/streamfunctions/inlet_slab/SlabOracle.{cuh,cu}`
  (accounting fields on `SlabOraclePlane`/`SlabOracleResult`, `table()`), `SlabMetrics.{cuh,cu}`
  (`SlabPlaneMetrics`, `evaluate_plane_metrics`), `apps/inlet_slab/inlet_slab_driver.cuh` (`run_trace_construct`,
  options, usage, exit codes 0 ok / 17 `inlet_backflow` / 18 `darcy_failed` / 19 `trace_fail`), `CMakeLists.txt`.
- New: `src/physics/streamfunctions/inlet_slab/SlabLabelTransport.{cuh,cu}` (`TracerBackend`,
  `LabelTransportStatus {ok, trace_fail, roundtrip_exceeded}`, `LabelTransportOptions {tol 1e-8, h_max 0 -> h/8,
  min_step 1e-13, max_arclength 100, threads, max_roundtrip 1e-8, backend}`, `LabelTransportResult`,
  `construct_labels_by_tracing(ctx, stage, options, psi1, psi2)`, `save_labels`);
  `src/physics/streamfunctions/inlet_slab/SlabTraceGpu.{cuh,cu}` (`DeviceSplineDirectionField`, `trace_dev::{dp5_step,
  land_on_plane, trace_to_plane}`, `SlabGpuTracer {prepare, trace}`; TU compiled with `--fmad=false`);
  `tests/inlet_slab/slab_trace_tests.cu` (`inlet_slab_trace16`), `tests/inlet_slab/slab_controls_tests.cu`
  (`inlet_slab_controls`), `tests/inlet_slab/slab_trace_gpu_tests.cu` (`inlet_slab_trace_gpu`), each `<= 60 s`;
  `docs/experiments/artifacts/2026-10-07-sf35-label-transport/{README.md, scripts/, logs/, raw/, analysis/}`;
  `docs/experiments/2026-10-07-sf35-label-transport.md`.
- Read-only reuse: `SlabProductionSetup` (`build_production_stage`, `SlabFieldSource::{gaussian, analytic,
  spectral}`, `SlabSplineDirectionField`), `InletLabels::evaluate_labels`, `SlabResidual` (`evaluate_residual`,
  `labels_to_periodic_parts`), `SlabNewtonKrylov` (only under `--solve`), `NpyIo.hpp`,
  `apps/closure_gate/streamline_integrator.hpp` (`dp5_step`, `land_on_plane`, `ArclengthRhs`),
  `src/numerics/interpolation/PeriodicTricubicBSpline.cuh` (`evaluate_point`, `__host__ __device__`),
  `src/physics/particles/streamline_tracker/ReferenceRkTracker.cuh` (`dp54_trial_step`, template only).

## Implementation specification

1. **Constructor.** `construct_labels_by_tracing` requires `stage.ok()` and device-prepared inlet labels; maps
   `LabelTransportOptions` to `SlabOracleOptions` (`h_max = 0` means `grid.h / 8`, logged); calls `compute_oracle`
   (host) or `SlabGpuTracer` (gpu); fills the full labels on planes 0..N; `status = trace_fail` if any streamline is
   non-`ok`, `roundtrip_exceeded` if any plane's round trip exceeds `max_roundtrip`; failed seeds get NaN labels.
   Outputs are a pure function of `(stage, options)` and bitwise independent of the thread count.
2. **Diagnostics at the traced labels.** `labels_to_periodic_parts` -> `U1, U2`; `evaluate_residual` -> `r_F`,
   `r_out`; `evaluate_metrics` with `vD` only -> CASE line `cand=trace` (`r_F` filled, `its=nan`);
   `evaluate_plane_metrics` -> `PLANE j=<j> x1=<x1> e_v=<> e_i1=<> e_i2=<> min_c=<> vD1_min=<> vD1_nonpos=<>
   roundtrip=<> nfev=<> nfev_rt=<> non_ok=<> backflow_enc=<> landings_discarded=<>`; `BACKFLOW field=.. eps=.. N=..
   inlet_vmin=<> Q0=<> vD1_min=<> vD1_nonpos=<> seed_backflow=<> backflow_enc=<> landings_discarded=<> min_g1_hat=<>
   status_hist=<ok,seed_backflow,cap,stagnation,underflow,landing_failed,invalid_k>`; `GROWTH` line with
   `e_v_N / e_v_1`, `e_i_N / e_i_1`, `max_j roundtrip_j`.
3. **Driver.** `--construct trace --n N --eps E [--sigma2 S --ell L --seed R | --analytic F | --cells P] [--hmax-div 8]
   [--tol 1e-8] [--max-roundtrip 1e-8] [--ladder] [--tracer host|gpu] [--threads T] [--solve] [--save-labels DIR]`.
   One stage build, no continuation; `--ladder` adds `(h/16, tol)`, `(h/16, 1e-10)`, `(h/32, 1e-10)` with `LABELDIFF`
   lines (max `|dpsi|` vs the primary); `--solve` runs the unchanged SF-33 Newton–Krylov with its C3/C4 defaults and
   reports the CASE `cand=i1o4` line with `e_psi*`/`a_psi*` against the traced labels plus `XCONS` per-plane
   `a_psi_j`. `--production`, `--proto`, `--sf19-crosscheck`, `--linear-probe` untouched.
4. **Positive control `k(x2)`.** `SlabFieldSource::spectral` with `Y_cell = cos(2 pi (j + 1/2) h)`, `eps = 0.5`: the
   SF-19 potential fluctuation vanishes to PCG level, `g = G` exactly, streamlines are straight, so
   `psi_i(j, m2, m3) = psi_i(0, m2, m3)` on every plane. `k = 1` and `control2d` as in the existing tests. A synthetic
   analytic functor with a backflow region and a stagnation point exercises the accounting (counts asserted).
5. **GPU tracer.** Identical expression order to the host `dp5_step`, `land_on_plane`, `step_factor`,
   `ArclengthRhs`; `--fmad=false` on the TU; `prepare(grid)` is the only allocating call; the kernel writes per-seed
   outputs only (feet, round trip, status, `nfev`, counters); the host does the per-plane reductions in index order
   exactly as `compute_oracle`; labels at the feet via the existing `detail::oracle_evaluate_labels`.
6. **Campaign.** Field: SF-18 Gaussian, `sigma2 = 1`, `ell = 0.25`, seed 3001, `normalize_variance`,
   `k = exp(eps Y)`. Matrix: `eps` in {0.5, 1.0, 1.5} x `N` in {32, 64, 128, 256}; primary `(h/8, 1e-8)`; `--ladder` at
   32 and 64; `--save-labels` at every point; `/usr/bin/time -v`. Cross-construction: `--solve` at (0.25, 32),
   (0.25, 64), (0.5, 32). Controls at 16/32/64. Regression: `--production --n 32 --eps 0.25 --sigma2 1 --ell 0.25
   --seed 3001 --oracle-ladder` diffed against SF-33 `logs/ladder_0.25_c4/N32.log` on the CASE/ORACLE/ORACLE_SUMMARY/
   EXTRA lines (timings excluded). If the GPU node is accepted: `--tracer gpu` at 128^3 and 256^3 `eps = 0.5` vs host.
   Informative: `--sigma2 2.25 --eps 1` vs `--sigma2 1 --eps 1.5` at 32^3. Digests under `analysis/`.
7. **Failure behaviour.** Distinct exits; NaN/Inf propagate; no retry with changed settings inside a job; the ladder
   settings are fixed before the runs (activation bitácora row).
8. **Conventions.** Slab layout `idx = m3 + N (m2 + N j)`, `(N+1) N^2` per field, `x = m/N`; labels
   `psi1 = x2 + U1`, `psi2 = x3 + U2`; no regularization anywhere; double precision throughout.

## Expected numerical effect

No existing output changes (`--production` CASE/ORACLE lines byte-identical; the 6 SF-33 tests unchanged). Expected:
the traced labels' `e_v` at `eps = 0.25` equals SF-33's `oracle_fd4` ceiling (2.247e-3 / 5.684e-4 / 1.427e-4 at
32/64/128, order ~2, bounded by the 2nd-order SF-19 inlet flux and spline potential); `e_i`, `e_div` orders 2.5-4;
`r_F` of the traced labels decreasing with order `>= 2` (order 4 is not expected: the inputs are 2nd order); per-plane
`e_v_j` dominated by the flow's discretization error, not by the tracer (round trip 1e-9 << 1e-3); `eps = 1.5`: not
predicted (backflow possible; SF-30 saw none at `sigma^2 <= 1` for `ell` 1/8-1/16; `ell = 1/4` at 2.25 is unmeasured).

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R 'inlet_slab'          # 6 existing + 3 new, each <= 60 s
./build/wsl-debug/inlet_slab --construct trace --analytic generic3d --n 16 --eps 0.25
scripts/remote --increment SF-35 sync
scripts/remote --increment SF-35 exec -- "cmake --preset v100-release && cmake --build build/v100-release -j 2>&1 | tail -n 20; echo BUILD_EXIT=${PIPESTATUS[0]}"
scripts/remote --increment SF-35 run sf35-build -- "ctest --test-dir build/v100-release --output-on-failure -R inlet_slab"
scripts/remote --increment SF-35 wait sf35-build
A=docs/experiments/artifacts/2026-10-07-sf35-label-transport/scripts
scripts/remote --increment SF-35 run sf35-controls -- "bash $A/run_campaign.sh build/v100-release controls"
scripts/remote --increment SF-35 wait sf35-controls
scripts/remote --increment SF-35 run sf35-xcons -- "bash $A/run_campaign.sh build/v100-release xcons"
scripts/remote --increment SF-35 wait sf35-xcons
scripts/remote --increment SF-35 run sf35-regress-prod32 -- "bash $A/run_campaign.sh build/v100-release regress"
scripts/remote --increment SF-35 wait sf35-regress-prod32
for e in 05 10 15; do scripts/remote --increment SF-35 run sf35-matrix-eps$e -- "bash $A/run_campaign.sh build/v100-release matrix $e"; scripts/remote --increment SF-35 wait sf35-matrix-eps$e; done
scripts/remote --increment SF-35 run sf35-gpu-equiv -- "bash $A/run_campaign.sh build/v100-release gpu"   # only if the GPU node is accepted
scripts/remote --increment SF-35 wait sf35-gpu-equiv
scripts/remote --increment SF-35 run sf35-ctest-full-bytecmp -- "ctest --test-dir build/v100-release --output-on-failure; <SF-27 trio byte-compare vs base refs>"
scripts/remote --increment SF-35 wait sf35-ctest-full-bytecmp
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

Pre-registered verbatim in the activation bitácora row before any run; the ladder settings `(h/8, 1e-8)` primary,
`(h/16, 1e-8)`, `(h/16, 1e-10)`, `(h/32, 1e-10)` are fixed before the runs. No threshold is tuned afterwards.

- **A. Positive controls** (fast tests at 16/32; V100 adds 64). A1 `k = 1`: feet `= (x2, x3)` and labels affine to
  `<= 1e-13` on every plane. A2 `k = exp(0.5 cos 2 pi x2)`: `|psi_i(j, m2, m3) - psi_i(0, m2, m3)| <= 1e-10` on every
  plane; `e_v` and `r_F` of these labels decrease 16 -> 32 (-> 64) with observed order `>= 3.5`. A3 `control2d`
  `eps = 0.5`: `|psi2 - x3| <= 1e-10` on every plane. A4 synthetic failure field: status histogram,
  `landings_discarded`, NaN labels and `trace_fail` exactly as constructed.
- **B. Integration certificate.** Round trip `<= 1e-8` on every plane at `(h/8, 1e-8)` for every matrix point;
  `non_ok = 0`; at 32 and 64 the ladder gives max `|dpsi|` between the primary and `(h/32, 1e-10)` `<= 1e-8`.
- **C. Convergence per amplitude** (32/64/128/256). `e_v`, `e_i1`, `e_i2`, `e_div`, `r_F` each decrease on every
  refinement and show observed order `>= 1.8` on 128 -> 256: `accept`; any of them changing `< 10 %` between 128 and
  256 while above 1e-12: `reject` (floor); otherwise `unresolved`. Rationale for 1.8 on `r_F`: the traced labels are
  the exact labels of the spline flow, whose consistency with the spectral `k` is O(h^2); a 4th-order threshold would
  presuppose an exact Darcy flow. Blocking: `eps = 0.5` and `1.0` must be `accept`; `eps = 1.5` must be `accept`
  unless the stage reports `inlet_backflow` (then recorded; owner decision). `r_out`, `min_c`, percentiles recorded.
- **D. Cross-construction.** The `--solve` runs converge (`r_F <= 1e-10`) and `e_psi / e_psi1 / e_psi2` (elliptic vs
  traced, C-i4 normalization) reproduce SF-33 within `1e-6` relative: 32^3 `eps = 0.25` `1.250e-2 / 1.250e-2 /
  1.140e-2`; 64^3 `eps = 0.25` `1.555e-3 / 1.483e-3 / 1.555e-3`; 32^3 `eps = 0.5` `3.837e-2 / 3.837e-2 / 2.756e-2`
  (`analysis/ladder_orders_025_c4.md`, `analysis/prod32_05.md`, `logs/ladder_0.25_c4/N{32,64}.log` of the SF-33
  artifact); `a_psi1`, `a_psi2` observed order 32 -> 64 `>= 2.5` (SF-33: 3.07 / 2.87 — the two constructions agree
  faster than either agrees with `vD`).
- **E. Recorded, no threshold.** Per-plane `e_v_j`, `e_i_j`, `roundtrip_j`, `nfev_j` and the `GROWTH` line for every
  point; the `BACKFLOW` line for every point, tabulated vs `eps` and `N`; the `--sigma2 2.25 --eps 1` vs `--sigma2 1
  --eps 1.5` label difference at 32^3 (expected `<= 1e-12`).
- **F. GPU tracer (non-blocking).** With `--fmad=false`: max `|dfoot|`, `|dpsi|` host vs device `<= 1e-12` at 16^3
  and 32^3 on {`k = 1`, `generic3d` 0.25, `gauss` 0.5}; per-plane `nfev` and status histograms identical (otherwise the
  number of streamlines with a different step sequence is reported; the 1e-12 bound remains the criterion); two
  launches bitwise identical; no device allocation in `trace` after `prepare`; on V100 at 128^3 (and 256^3)
  `eps = 0.5`: `<= 1e-12` vs host, wall time recorded; the `--fmad=true` difference reported as INFO.
- **G. Regression.** CASE/ORACLE/ORACLE_SUMMARY/EXTRA lines of the `--production` 32^3 `eps = 0.25` run identical to
  SF-33's `N32.log` (timings excluded); `git diff <base>..<head>` empty on the frozen stack, `src/numerics`,
  `src/multigrid`, `src/physics/particles`, `apps/closure_gate`; SF-27 trio byte-compare clean; full V100 `ctest`
  green; the 6 SF-33 tests unchanged.
- **H.** Wall time, host RSS and peak device memory at 256^3 recorded per amplitude.
- **Gate 3A mapping.** `r_F`, `r_out` = residuals of the equation and outlet rows evaluated at the constructed labels
  (necessary condition; never an acceptance quantity alone); `e_v`, `e_i`, `e_div`, `|c|` percentiles as in
  `SlabMetrics.cuh`; denominator regularization: none; gauge: none (labels anchored by D-1); "Newton failures" ->
  tracing status histogram and round trips; grid convergence = C; positive controls (A) precede acceptance (C).

## Regression surface

- `inlet_slab --production/--proto/--sf19-crosscheck/--linear-probe` outputs; `SlabOracleResult` existing fields;
  `SlabMetrics` existing outputs; the 6 `inlet_slab_*` ctest entries; the frozen periodic stack;
  `apps/closure_gate/streamline_integrator.hpp` (untouched); SF-31 tracker (untouched); default pipeline configs;
  `ctest` wall time.

## Failure and rollback policy

- A or D not met: implementation or shared-code defect (BLOCKING); the increment stays active until fixed (D failing
  means the traced labels or the solver changed: regression G locates it).
- B not met at a matrix point: recorded `trace_fail` / `roundtrip_exceeded`; no setting is changed; exclusion only by
  owner decision.
- C `reject` / `unresolved` at `eps = 0.5` or `1.0`: stop and return to the owner with the ladders; the decision
  record's validity envelope is written with the measured amplitudes only; nothing is tuned.
- C at `eps = 1.5` with `inlet_backflow`: recorded; owner decision.
- F not met: the GPU node is dropped (`--tracer gpu` removed or left throwing); the increment closes with the host
  constructor.
- Any change to the frozen stack or to the SF-30 integrator header: reverted.

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

`SF-34` (re-specified as acceptance of the SF-35 labels against the SF-30 return map) becomes eligible after this
increment is merged and marked `done` on the default branch.

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-07T16:40Z | not started | Specification created on the SF-33 closure PR (#51) after the owner decision of 2026-10-07 (SF-33 closed `done` with claim (b) not established; decision record `docs/decisions/2026-10-07-label-transport-constructor.md`). Intra-increment DAG planned: N1 constructor API + driver + fast test -> {N2 controls, scripts, digests ∥ N3 GPU tracer} -> N4 V100 campaign (serial, the only remote node) -> N5 experiment note; one integrator. | Depends on SF-33 `done` on `master`. | Activate only when the checker reports it READY; the activation row must contain thresholds A-H verbatim, the ladder settings and the base commit before any run. |
