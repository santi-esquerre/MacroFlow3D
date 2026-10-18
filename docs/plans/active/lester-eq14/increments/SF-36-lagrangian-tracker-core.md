# SF-36 — Lagrangian-label tracker core on the cube production stage

- State: `awaiting_review`
- Goal: `Construir el tracker lagrangiano de producción sobre el stage del cubo (trazado hacia adelante DP5(4) con tolerancia controlada sobre el spline v_s = k_s grad Phi_s, etiquetas por partícula, transporte de covectores y divergencia acumulada, momentos por plano y por tiempo) y verificarlo con controles exactos, equivalencia con el trazador host de SF-30 y la identidad c = v_s exp(-int div v_s dt), sin cambiar ningún bit de las salidas de SF-35.`
- Depends on: `SF-35`
- Unlocks: `SF-37, SF-38`
- Branch: `science/lester-sf36-lagrangian-tracker-core`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 2 + Gate 4 (method controls of a Lagrangian-label consumer, see "Acceptance thresholds")`
- Human review: `required`
- Owner: `Claude Fable 5.1 orchestrator (session 2026-10-17)`
- Started: `2026-10-17T19:48Z on base 040e3ce`
- Completed: `not completed`
- PR: `not opened`
- Commit: `e81b11833412149027643293d7e11ff989643189` (source-bearing audited head: integrated chain `200cbf0..e81b118`); evidence/docs-only commits above it on the PR branch (`fc3e5b4` N8, `ba81e4b` N9, FINAL_AUDIT metadata)

## Scientific or engineering intent

Owner decision of 2026-10-17 (`docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`, items 2, 4, 9): the
production consumer at `sigma_Y^2 >= 4` is Lagrangian — particles traced forward on the spline flow
`v_s = k_s grad Phi_s` (`Phi_s = G.x + s_h`, `k_s = exp(s_Y)`), each carrying its own labels. SF-36 builds the
consumer's core on the existing cube stage (SF-18 field, SF-19 flow, SF-28 splines, SF-35 device integrator) and
is allowed to establish one claim: **the implemented forward tracker integrates the intended equations** —
positions (DP5(4) arclength with Hénon landing, as SF-30), travel time, the accumulated divergence
`I = int div v_s dt`, and covectors transported by `d(grad psi)/ds = -(grad u)^T grad psi` — as shown by controls
with exact answers, host/GPU equivalence, finite differences of the flow map and the identity
`grad psi1 x grad psi2 = v_s exp(-I)`.

No claim about the convergence of plume statistics, the production `lambda/h`, `alpha_L` or `alpha_T` (SF-37,
SF-39). The tracker carries no structural guarantee against integration drift (source (B) of theory note §5); that
is measured in SF-37 (Rule 3).

## Preconditions

- `SF-35` is `done` on the default branch (tracing constructor, `SlabTraceGpu` device copy of the SF-30 integrator
  compiled with `--fmad=false`, `inlet_slab --construct trace`, 9 `inlet_slab_*` tests).
- `SF-28`, `SF-30` are `done` (spline; `apps/closure_gate/streamline_integrator.hpp`, untouched).
- The decision record `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md` is `accepted` on the default
  branch; the probe diff `docs/experiments/artifacts/2026-10-17-sf35-sigma4-feasibility/probe.patch` is the source
  of P2.

## In scope

- **Shared device integrator header.** Move `trace_dev::{DP5, EvalCode, LandingOutcome, kDim, DevResult, finite3,
  hmax, hmin, ArclengthRhs, LandingRhs, dp5_step, step_factor, status_from, map_code, land_on_plane,
  trace_to_plane}` from `SlabTraceGpu.cu` to a header without changing any expression, constant or branch; every TU
  that includes it is compiled with `--fmad=false`.
- **P2.** Release of the dead `N^3` temporaries in `build_production_stage` exactly as the probe (`K_dev`, `v`, `w`
  after the SF-19 solve; `u` after the inlet samples; the SF-19 workspace and `lnK_dev` after the host copy of the
  potential), without the probe's `MF_PROBE_KEEP_TEMPS` switch (no environment-controlled behaviour).
- **Stage without inlet labels.** `ProductionStageOptions::require_inlet_labels` (default `true` = current
  behaviour, including the early return on `inlet_backflow`); with `false`, an `InletBackflowError` is recorded
  (status `inlet_backflow`, `labels_available = false`) and steps (iv)-(v) (splines, `vD`) are still built.
- **Spline Hessian.** `interpolation::evaluate_point_hessian` (`__host__ __device__`, additive): value, gradient and
  the 6 second derivatives from second-derivative weights `d2w` per axis; `evaluate_point` untouched.
- **Forward tracker** (new module `src/physics/particles/lagrangian_labels/`): flux-weighted injection on a fixed
  stratified lattice, per-particle attributes, forward DP5(4) arclength tracing with landings on a list of planes in
  unwrapped `x1` and on a list of snapshot times, passive integration of `tau`, `I` and (optional) covectors,
  statuses as SF-30, weighted moments per plane and per time, `.npy`/CSV/JSON export.
- **Driver mode** `inlet_slab --track forward` following the `run_trace_construct` pattern (Mode enum, `parse_args`,
  validation, `run_track_forward`, dispatch case).
- Fast tests (each `<= 60 s`), V100 control and regression jobs, experiment note and artifact.

## Out of scope

- Plume-statistics convergence, tolerance/grid ladders, Pollock comparator, O3 (SF-37); non-cubic stage, ensemble
  runner, Par2 baseline (SF-38); the study (SF-39).
- Any change to the frozen periodic stack (`src/physics/streamfunctions/*` outside `inlet_slab/`), `src/multigrid/`,
  `src/numerics/` other than the additive Hessian function, `apps/closure_gate/*`, SF-18, SF-19,
  `src/physics/particles/{pspta, par2_adapter, streamline_tracker}`.
- Any change to SF-35 outputs (byte-identity is criterion F), to the SF-35 constructor's behaviour or to its exit
  codes.
- Pseudo-symplectic tracking, finite Péclet number, interpolation of gridded labels at particle positions (SF-37).

## Files and symbols

- New: `src/physics/streamfunctions/inlet_slab/SlabTraceDevice.cuh` (moved `trace_dev`, verbatim);
  `src/physics/particles/lagrangian_labels/LagrangianTracker.{cuh,cu}` (`InjectionSpec`, `TrackOptions`,
  `ParticleStatus`, `LagrangianTrackResult`, `LagrangianTracker {prepare, track}`, `write_track_outputs`), TU with
  `--fmad=false`; `tests/lagrangian_labels/lagrangian_controls_tests.cu` (`lagrangian_tracker_controls`),
  `tests/lagrangian_labels/lagrangian_gpu_host_tests.cu` (`lagrangian_tracker_gpu_host`),
  `tests/interpolation/spline_hessian_tests.cu` (`interpolation_hessian`);
  `docs/experiments/<date>-sf36-lagrangian-tracker-core.md` and its artifact directory.
- Extend: `SlabTraceGpu.cu` (includes the header), `SlabProductionSetup.{cuh,cu}` (P2,
  `require_inlet_labels`, `labels_available`), `src/numerics/interpolation/PeriodicTricubicBSpline.cuh`
  (`AxisStencil` gains `d2w[4]` filled only by a new `axis_stencil2`; `evaluate_point_hessian`),
  `apps/inlet_slab/inlet_slab_driver.cuh` (`Mode::track_forward`, options, usage, `run_track_forward`, exit codes),
  `CMakeLists.txt` (sources, `--fmad=false`, tests).
- Read-only reuse: `DeviceSplineDirectionField`, `InletLabels::evaluate_labels`, `closure_gate::StreamlineStatus`,
  `SlabOracle.cuh::trace_to_plane` (host reference), `NpyIo.hpp` (`write_npy`), `io::TimeSeriesPoint` /
  `io::CsvTimeSeriesWriter` (`src/io/writers/CsvTimeSeriesWriter.hpp`), `compute_macrodispersion`
  (`src/analysis/macrodispersion/MacrodispersionAnalysis.hpp`, used only as a downstream reader test).

## Implementation specification

1. **Header move.** `SlabTraceDevice.cuh` holds the `trace_dev` symbols verbatim; `SlabTraceGpu.cu` includes it
   and keeps its kernel and `SlabGpuTracer`. The CMake property `--fmad=false` covers `SlabTraceGpu.cu` and
   `LagrangianTracker.cu`. Control: criterion F.
2. **P2 and stage options.** As in "In scope"; `stage.ok()` keeps its meaning; a new `stage.flow_ready()` is true
   when the Darcy solve converged and the splines exist (also under `inlet_backflow` with
   `require_inlet_labels = false`). The SF-35 and SF-33 driver paths keep the default.
3. **Hessian.** Per axis at relative coordinate `u` in [0, 1): `d2w = {1 - u, 3u - 2, 1 - 3u, u} / h^2` (second
   derivative of the uniform cubic B-spline weights in the order of `w`, `dw`); mixed derivatives from products
   `dw x dw`. Output order `H = (xx, yy, zz, xy, xz, yz)`. Same periodic reduction and stencil as `evaluate_point`.
4. **Injection.** Plane `x1 = X0` (default 0); seeds on an `M x M` stratified lattice
   `((m2 + 1/2)/M, (m3 + 1/2)/M)` mapped to the window (default the whole periodic face): **fixed, grid-independent
   positions** so that tolerance and grid ladders are paired particle by particle. Weight
   `w_p = max(v_s1(x_p), 0)` from the spline flow at the seed, normalized to sum 1 (flux weighting, B&dD "injected
   proportionally to the flow", realized by weights rather than by sampling density to keep the pairing exact);
   seeds with `v_s1 <= 0` get status `seed_backflow`, weight 0, and are counted. Attributes: seed, weight, D-1
   labels at the seed when `labels_available` (else NaN). Initial covectors `a = (-v_s2, v_s1, 0)`,
   `b = (-v_s3/v_s1, 0, 1)`, so `a x b = v_s(x0)` exactly and `a.v_s = b.v_s = 0`.
5. **Integration.** Forward arclength DP5(4) (`sigma = +1`) with the SF-30 step control, `tol` (default 1e-8),
   `h_max` (default `h/8`), `min_step` 1e-13, `max_arclength` (default `4 (X_last - X0) + 10`). The error norm uses
   the **position components only**; `tau` (`dtau/ds = 1/(k_s |g|)`), `I` (`dI/ds = (grad s_Y . g + lap s_h) /
   |g|`, i.e. `div v_s / |v_s|`) and the covectors (`da/ds = -H (a - (u.a) u) / |g|`, `H` = Hessian of `s_h`,
   `u = g/|g|`; same for `b`) ride passively on the accepted steps, so positions are bitwise identical with
   covectors on or off. The moved `trace_dev::dp5_step` has `kDim = 4` and is not modified: the forward tracker uses
   its own DP5 stepper templated on the state dimension (4 without, 11 with covectors and `I`), with the same
   tableau, `step_factor` and expression order for the position components, so that criterion C holds. Planes: strictly increasing unwrapped `x1` values `> X0`; Hénon landing on each in turn
   (first forward crossing; further crossings counted). Snapshot times: strictly increasing `t_n > 0`; Hénon landing
   with `tau` as independent variable (`dx/dtau = k_s g`). Statuses: `closure_gate::StreamlineStatus` (which already
   contains `seed_backflow`); interior backflow followed and counted; nothing clamped or regularized.
6. **Moments.** Over particles `ok` at that plane/time, with the normalized weights renormalized over them:
   weighted mean and **biased** weighted variance (`sum w (x - m)^2`, as B&dD eq. 4 and Par2 `var_estimator:
   biased`) of `x1, x2, x3` at each snapshot -> `io::TimeSeriesPoint<double>{time, mean[3], var[3], active}`
   (`active` = number of ok particles); per plane: weighted mean/variance of `x2`, `x3`, `tau`, and RMS / max of the
   Lagrangian `e_v = 1 - exp(-I)`. Coordinates are unwrapped.
7. **Determinism and ownership.** `prepare` is the only allocating call; `track` launches one thread per particle,
   writes per-particle outputs only, and the host reduces in index order; results are bitwise independent of the
   launch configuration and of repeated launches.
8. **Outputs** (`--save-particles DIR`): `seeds.npy (Np,3)`, `weights.npy (Np)`, `labels.npy (Np,2)`,
   `status.npy (Np) uint8`, `planes_x23.npy (Np,P,2)`, `planes_tau.npy (Np,P)`, `planes_I.npy (Np,P)`,
   `times_x.npy (Np,T,3)`, `times_I.npy (Np,T)`, and with covectors `planes_cov.npy (Np,P,6)`,
   `times_cov.npy (Np,T,6)`; `moments_time.csv` (`CsvTimeSeriesWriter` format), `moments_planes.csv`,
   `track.json` (field, `eps`, `N`, `h`, `X0`, `M`, planes, times, `tol`, `h_max`, status histogram, counts,
   commit). Driver lines `TRACK`, `TPLANE`, `TTIME`, `TSTATUS`. Exit codes: 0 ok; 18 `darcy_failed`;
   22 `track_fail` (any particle non-ok other than `seed_backflow`); inlet backflow is recorded, not an exit
   condition.
9. **Conventions.** Double precision throughout; cube units (`[0,1]^3`, mean flux `e1`, porosity 1, so the mean
   velocity is `e1` and one period takes unit time on average); no regularization anywhere.

## Expected numerical effect

No existing output changes (criterion F). Expected: exact controls at roundoff; `lester2021` one-period return
`~1e-9`-`1e-8` at `tol = 1e-8` (SF-30: `R <= 7.2e-9` with 1024 seeds); GPU vs host 0 to `1e-12`; covector FD
defect of order 2 in `delta`; identity residual `<= 1e-8` at `tol = 1e-10` while the covectors stay moderate; at
`eps = 2` the covector norms can grow strongly (stretching), not predicted.

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R 'inlet_slab|lagrangian_tracker|interpolation'
./build/wsl-debug/inlet_slab --track forward --analytic generic3d --n 16 --eps 0.25 --np-side 16 --planes 0.25,0.5,1 --times 0.5,1 --covectors
scripts/remote --increment SF-36 sync
scripts/remote --increment SF-36 exec -- "cmake --preset v100-release && cmake --build build/v100-release -j 2>&1 | tail -n 20; echo BUILD_EXIT=${PIPESTATUS[0]}"
A=docs/experiments/artifacts/<date>-sf36-lagrangian-tracker-core/scripts
scripts/remote --increment SF-36 run sf36-base-refs -- "bash $A/base_refs.sh"          # base-commit build: SF-35 outputs for F, peak memory for G
scripts/remote --increment SF-36 wait sf36-base-refs
scripts/remote --increment SF-36 run sf36-controls -- "bash $A/run_sf36.sh build/v100-release controls"   # A-E at 64^3
scripts/remote --increment SF-36 wait sf36-controls
scripts/remote --increment SF-36 run sf36-bytecmp -- "bash $A/run_sf36.sh build/v100-release bytecmp"     # F, G
scripts/remote --increment SF-36 wait sf36-bytecmp
scripts/remote --increment SF-36 run sf36-cost -- "bash $A/run_sf36.sh build/v100-release cost"           # J
scripts/remote --increment SF-36 wait sf36-cost
scripts/remote --increment SF-36 run sf36-ctest-full-bytecmp -- "ctest --test-dir build/v100-release --output-on-failure; <SF-27 trio byte-compare vs base refs>"
scripts/remote --increment SF-36 wait sf36-ctest-full-bytecmp
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

Pre-registered verbatim in the activation bitácora row before any run; no threshold is tuned afterwards. Fast
tests at 16^3/32^3; V100 repeats at 64^3. Defaults: `tol = 1e-8`, `h_max = h/8` unless stated.

- **A. Exact controls.** A1 `k = 1`: on every plane and snapshot, `|x2 - x2_0|`, `|x3 - x3_0| <= 1e-12`; `tau` at
  plane `X` equals `X - X0` to `<= 1e-12` relative; covectors equal their initial values to `<= 1e-12`; `|I| <=
  1e-14`. A2 layered `k = exp(0.5 cos 2 pi x1)` (`SlabFieldSource::spectral`): transverse positions and covectors
  constant to `<= 1e-12`; `tau` and `I` at each plane against an independent adaptive 1-D quadrature of
  `1/v_s1(x1)` and `div v_s / v_s1` on the same spline `<= 1e-9` relative (`tau`, `I` are passive quadratures on the
  error-controlled steps; the planning value `1e-12` is not attainable by construction and is not used). A3
  synthetic backflow/stagnation field (the SF-35 A4 functor): status histogram and counters exactly as constructed.
- **B. Closed streamlines.** `lester2021`: after one period every particle returns to its seed (transverse,
  unwrapped minus `e1`) within `10 tol` at `tol = 1e-8`, all statuses `ok`, at 32^3 and 64^3.
- **C. GPU vs host.** Single-plane forward traces against the host `trace_to_plane` (SF-30/SF-33) on {`k = 1`,
  `generic3d` 0.25, `gauss` 0.5, `gauss` 2.0} at 16^3/32^3 (V100: 64^3): max `|dx| <= 1e-12`; per-particle `nfev`
  and status identical (otherwise the count of differing step sequences is reported; the 1e-12 bound remains the
  criterion).
- **D. Covectors vs finite differences of the flow map.** 64^3, fields `generic3d` 0.25 and `gauss` 1.0, 256 seeds,
  `tol = 1e-12`, `h_max = h/32`, snapshots `tau` in {0.5, 2}: the RMS over seeds of
  `|a(tau)^T DPhi_FD(delta) - a(0)^T|` (and `b`), central differences with `delta` in {8, 4, 2} x 1e-3, has observed
  order `>= 1.8` on both `delta` pairs.
- **E. Lagrangian identity.** `|a x b - v_s(x) exp(-I)| / |v_s(x)| <= 1e-8` at every plane and snapshot for every
  `ok` particle up to `x1 = 4`, at `tol = 1e-10`, on `generic3d` 0.25 and `gauss` 1.0 (64^3): blocking. On `gauss`
  2.0: recorded with the covector norms (informs SF-37 V5), not blocking.
- **F. SF-35 byte identity and regression.** Against the base-commit build on the same mirror: `--construct trace`
  at 32^3/64^3, `eps` in {0.5, 1.0}, host and GPU: `psi1.npy`, `psi2.npy` and the CASE/PLANE/BACKFLOW/GROWTH/
  RESIDUAL/TRANSPORT_STATUS/STATUS lines byte-identical (timing fields masked); `--production` 32^3 `eps = 0.25`
  CASE/ORACLE/ORACLE_SUMMARY/EXTRA lines identical to SF-33 `logs/ladder_0.25_c4/N32.log` (as SF-35 G); the 9 `inlet_slab_*` tests unchanged and
  green; full V100 `ctest` green; SF-27 trio byte-compare clean; `git diff <base>..<head>` empty on the frozen
  stack, `src/multigrid`, `apps/closure_gate`, `src/physics/particles/{pspta,par2_adapter,streamline_tracker}`, and
  additive-only in `PeriodicTricubicBSpline.cuh`.
- **G. P2.** Byte identity is part of F (also at 64^3/128^3 `eps = 2`); device peak by nvidia-smi (200 ms) at
  256^3 `eps = 2` `--construct trace` for the head and the base-commit build: head `<=` base, and head within 2 % of
  the probe value with P2 (3769 MiB; the CUDA context varies by a few MiB between runs).
- **H. Spline Hessian.** Exact to `<= 1e-12` relative where the coefficient lattice is a cubic polynomial over the
  `4^3` support (interior, non-wrapping); central FD of the analytic gradient at non-knot points with `delta` in
  {1e-3, 5e-4} `h`: observed order `>= 1.8`; host vs device `<= 1e-14` relative; `interpolation_periodic_tricubic`
  green and `evaluate_point` outputs bitwise unchanged.
- **I. Determinism and allocation.** Two launches bitwise identical; positions, `tau` and statuses bitwise
  identical with covectors on vs off; no device allocation in `track` after `prepare` (cudaMemGetInfo before/after
  equal); moments reproduced by `compute_macrodispersion` reading the written CSV.
- **J. Recorded, no threshold.** Wall time, field evaluations per particle per unit length and peak device memory
  at 64/128/256^3 for `eps` 1 and 2 (`M = 256`, planes to `x1 = 4`), with and without covectors.
- **Owner amendment (2026-10-17, after the N6 measurements; post-run, disclosed).** The pre-registered criteria
  above stay as written and their verdicts are recorded as **not met** where they failed: A2's `I` sub-criterion
  (rel. `1.16e-3` at 32^3), E (`1.99e-4` generic3d, `8.8e-4` gauss 1.0 at 16^3) and D at (gauss 1.0, `tau = 2`)
  (orders `0.73-1.27` at 64^3). Root causes established by the N6 diagnostics: on the C^2 spline the Hessian is only
  C^0, and the passive `I` and covectors carry an `O(h_max^~2)` error independent of `tol` (on a C^inf analytic field
  with exact derivatives the identity holds to `5.3e-11`); the pre-registered `delta` ladder is pre-asymptotic for the
  stretched flow at `tau = 2` (orders `1.95-2.03` with `delta <= 1e-3` in all four cases). The owner approved these
  amended criteria for acceptance:
  - **E1 (blocking):** identity `<= 1e-8` at `tol = 1e-10` on a C^inf analytic control field with exact `g`, `H`,
    `grad ln k` (Phi = x1 + 0.05 sin 2 pi x2 cos 2 pi x1 + 0.05 sin 2 pi x3 sin 2 pi x1, k = exp(0.5 cos 2 pi (x1 + x2 +
    x3)), `h_max = 1/128`; planes and times 0.5, 1, 2, 4). **E2 (recorded, no threshold):** the identity residual on
    `generic3d` 0.25, `gauss` 1.0 and 2.0 with an `h_max` ladder `h/8 ... h/64` at `tol = 1e-12`.
  - **A2:** transverse positions, covectors and `tau` unchanged (`tau <= 1e-9` relative, evaluated at `N >= 32`); the
    `I` comparison is recorded with its `h_max` ladder (no threshold).
  - **D:** `delta` in {1, 0.5, 0.25} x 1e-3 (chosen after the pre-registered ladder was seen to be pre-asymptotic),
    same fields, seeds, `tol`, `h_max`, `tau` and order `>= 1.8` on both pairs.
  - Consequence recorded for SF-37: the passive `I` and covectors are accurate to `O(h_max^~2)` on the spline (at
    32^3, `|dI|` between `h/8` and `h/64` is `2-5e-5` against RMS `I` of `2.3e-3` to `0.21`).
- **Gate 4 mapping.** Exact controls (A, B), equivalence (C), covector consistency (D), divergence identity (E)
  precede any plume-statistics claim (SF-37); the tolerance and grid ladders, the independent Pollock comparator
  and the transverse classification are SF-37/SF-39.

## Regression surface

- SF-35 `--construct trace` and SF-33 `--production` outputs (shared device header, stage changes); the 9
  `inlet_slab_*` tests; every spline consumer (SF-30, SF-31, SF-32, SF-35) through `PeriodicTricubicBSpline.cuh`;
  `build_production_stage` callers; `ctest` wall time.

## Failure and rollback policy

- A, B, C, H or I not met -> implementation defect (BLOCKING); the increment stays active until fixed.
- D not met -> covector transport or FD design defect; BLOCKING (SF-37 V5 depends on it); no threshold tuned.
- E not met on `generic3d`/`gauss` 1.0 -> passive-component accuracy defect; BLOCKING. E on `gauss` 2.0 -> recorded.
- F or G not met -> regression in shared code; the change is reverted or redone until byte-identical; BLOCKING.
- Any change to the frozen stack, the SF-30 header or SF-35 behaviour -> reverted.
- A finding that the passive (non-error-controlled) covectors are too inaccurate for SF-37 V5 -> recorded; adding
  them to the error norm is a separate owner-approved change (it alters the step sequence).

## Completion checklist

<!-- completion-checklist:start -->
- [x] Implementation matches the scope and contains no unrelated changes.
- [x] Targeted validation passes and its evidence is recorded.
- [x] Required regression tests pass.
- [x] Scientific or engineering findings are appended to the bitácora.
- [ ] Required human review is recorded.
- [ ] PR and commit identifiers are recorded.
- [ ] The master checklist entry is checked in this branch and `check-lester-increments.sh` passes.
<!-- completion-checklist:end -->

## Advancement rule

`SF-37` and `SF-38` become eligible after this increment is merged and marked `done` on the default branch.

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-17T19:30Z | not started | Specification created on the SF-35 closure PR (#53) after the owner decision of 2026-10-17 (`docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`; probes `docs/experiments/2026-10-17-sf35-sigma4-feasibility.md`). Thresholds A-J drafted from the planning values; deviations from them written into the criteria: A2's `tau`/`I` at `1e-9` (passive quadratures), E blocking at `eps <= 1` and recorded at 2. | Depends on SF-35 `done` on `master`. | Activate only when the checker reports it READY; the activation row contains thresholds A-J verbatim and the base commit before any run. |
| 2026-10-17T19:48Z | activation on `master=040e3ce` (PR #53 squash-merged 2026-10-17T19:43Z; tree identical to the audited head `662a8b8`; checker OK ready=SF-34 SF-36, nonterminal=none); delivery branch `science/lester-sf36-lagrangian-tracker-core`; per-increment mirror `scripts/remote --increment SF-36`, synced only from clean worktrees | UNDERSTAND (record `.claude/orchestration/SF-36-lagrangian-tracker-core/understanding.md`; scientific-rigor skill, categories B + C). **Pre-registration before any run.** Acceptance thresholds A-J and the Gate 4 mapping: exactly the text of the section "Acceptance thresholds" of this file at commit `040e3ce`, unchanged (verbatim by reference; no threshold is tuned afterwards). Defaults: `tol = 1e-8`, `h_max = h/8`, `min_step = 1e-13`. Orchestrator reading rules (clarifications for the reviewer; no threshold changed): (1) each particle is traced in two deterministic passes — a PLANE pass (`integrate_streamline` generalized to an increasing list of planes; its first plane is the `trace_to_plane` of criterion C) and a TIME pass (same arclength stepping, Hénon landing with `tau` as independent variable) — because plane and time events can coincide exactly (`k = 1`: `x1 = X` exactly when `tau = X`) and one pass has no ordering rule for them; criterion I's on/off identity holds per pass; (2) the tracker uses its own DP5(4) stepper templated on the state dimension (4, or 11 with `I` and the covectors), with the tableau, `step_factor` and expression order of the SF-30 stepper for the position components and the error norm on positions only; the moved `trace_dev` code is not modified; (3) `evaluate_point_hessian` returns value and gradient bitwise equal to `evaluate_point` (needed by I); (4) arclength cap = total from the seed per pass, default `4 max(X_last - X0, t_last) + 10` (cube units), logged; (5) A2 references = independent host adaptive 1-D quadrature (`<= 1e-13`) of `1/v_s1` and `div v_s / v_s1` along the straight streamline on the same spline; (6) D perturbs the seed by `+-delta e_j` (j = 1..3) and uses the TIME pass to `tau`; (7) G compares the head to the base build measured by job `sf36-base-refs` (and to the probe value 3769 MiB within 2 %); (8) `--track forward` exit 22 = `track_fail` (17-21 are taken); the forward mode builds the stage with `require_inlet_labels = false`; (9) F's base = a `040e3ce` build on the SF-36 mirror (job `sf36-base-refs`, before any SF-36 source reaches the mirror), including the SF-27 trio. | Human-review increment (spec). DAG: {N1 device-integrator header move ∥ N2 P2 + stage options ∥ N3 spline Hessian} -> N4 forward tracker + GPU/host test -> {N5 driver mode ∥ N6 exact-control tests} -> N7 campaign scripts -> I1 integration -> N8 V100 campaign (only remote node) -> N9 experiment note -> I2 -> FINAL_AUDIT (orchestrator job `sf36-ctest-full-bytecmp`). | Launch N1, N2, N3; run `sf36-base-refs` on V100. |
| 2026-10-17T22:01Z | N1 `737d664`, N2 `88ff82b`, N3 `31eaa5e`, N4 `e64ace7` (on cherry-picks `6f88d42`/`cdce24b`/`4e0fe03`, patch-ids equal), N5 `e8d0210`, N6 `58d2c48` accepted (audits under `.claude/orchestration/SF-36-lagrangian-tracker-core/audits/`); orchestrator job `sf36-base-refs` exit 0 (base `040e3ce` on the SF-36 mirror; base 256^3 eps 2 device peak 6239 MiB) | Exact controls and equivalence pass: A1 all errors 0; A3 histograms exactly as constructed; B return `6.2e-10`; C max `|dx| = 0` with identical status/nfev on 8/8 cases (16/32: `k = 1`, `generic3d` 0.25, `gauss` 0.5, 2.0); I bitwise (repeat, block 64/256, covectors on/off) and pointer-stable. **Not met as pre-registered:** A2 `I` (rel. `1.16e-3`), E (`1.99e-4` / `8.8e-4`), D at (gauss 1.0, `tau = 2`) (64^3 orders `0.73-1.27`). Diagnostics (N6): identity on a C^inf field with exact derivatives `5.3e-11` (implementation verified); spline residual falls ~`h_max^1-2` (regularity of the C^2 spline: C^0 Hessian, passive `I`/covectors); D orders `1.95-2.03` with `delta <= 1e-3`. Clarification of reading rule (2): the state dimension is 5 (`x`, `tau`, `I`) or 11 with covectors — `I` is always integrated. Pre-existing bug found (out of scope): `analysis::read_timeseries_csv` reads one extra all-zero row because `CsvTimeSeriesWriter` writes two header lines (affects `AnalysisRunner` re-reads; for SF-38). | Owner decision (orchestrator session 2026-10-17): amend E, A2 and D with the evidence (section "Acceptance thresholds", owner amendment); the original verdicts stay recorded as not met. | Corrective C1 (tests to the amended criteria; `--n` / 64^3 mode for the GPU/host test) ∥ N7 campaign scripts -> I1 -> N8 V100 campaign. |
| 2026-10-18T00:47Z | FINAL_AUDIT positive; State `awaiting_review`; source-bearing audited head `e81b118` (I1 chain `200cbf0` N1, `727fd0e` N2, `3ebff2e` N3, `dd99c0f` N4, `0e3963b` N5, `e271fa3` N6, `fef20a3` C1, `e81b118` N7; cherry-picks with patch-ids equal to the accepted node commits, no integration changes); above it only evidence/docs: `fc3e5b4` (N8), `ba81e4b` (N9) and this commit | N8 V100 campaign at `e81b118` (64^3, `--full`): A1 every error 0; A2 transverse `9.7e-17`, covectors `1.2e-13`, `tau` `5.65e-11` relative (`<= 1e-9`; 32^3 `4.77e-10`; 16^3 `4.6e-9` is below the amended `N >= 32` scope), `I` recorded `7.4e-4`; A3 histograms and counters exact; B `3.97e-11` (`<= 1e-7`); C 26/26, max abs dx 0, `tau` `<= 2.84e-14`, status and `nfev` identical; D (amended `delta`) order `>= 1.8` on both pairs in all four cases (pre-registered ladder: not met at gauss 1.0, `tau = 2`); E1 `5.29e-11` (`<= 1e-8`); E2 recorded `6.68e-6` / `4.61e-5` / `1.54e-3` (generic3d 0.25, gauss 1.0, 2.0 at `h/8`); F 12/12 identical (10 `--construct trace` cases with `psi1/psi2.npy` and masked lines; `--production` 32^3 102/102 lines vs the base and SF-33 `N32.log`); G head `3771 MiB` vs base `6239 MiB`, inside `3769 +- 2 %`; H and I green in `interpolation_hessian` / `lagrangian_tracker_gpu_host`; J 12 runs recorded (fevals per particle per unit length 3286 / 6556 / 13096 at 64 / 128 / 256^3, eps 1); stress (`sigma_Y^2 = 16`, 16^3) exit 22 after 1228 s, `ev_rms` overflows. FINAL job `sf36-ctest-full-bytecmp` (tree `fc3e5b4`, GPU 0, 2026-10-17T23:21Z-2026-10-18T00:10Z): build exit 0; full `ctest` 37/37 in 2881.9 s; SF-27 trio outputs byte-identical to the base build (manifests identical modulo time lines). `git diff 040e3ce..` empty on the frozen stack, `src/multigrid`, `apps/closure_gate`, `src/physics/particles/{pspta,par2_adapter,streamline_tracker}`; no removed line in `PeriodicTricubicBSpline.cuh`. Open findings: MINOR two unused variables in `lagrangian_controls_tests.cu` (warnings only); INFORMATIONAL `inlet_slab_trace_gpu` flaky locally under a shared GPU (device-wide memory probe), green on V100; pre-existing `analysis::read_timeseries_csv` extra zero row (for SF-38). | Goal met under the owner-amended criteria; the original A2-`I`, E and D verdicts stay recorded as not met. Consequence for SF-37: the passive `I` and covectors are accurate to `O(h_max^~2)` on the spline. Evidence: `docs/experiments/2026-10-17-sf36-lagrangian-tracker-core.md`, artifact `docs/experiments/artifacts/2026-10-17-sf36-lagrangian-tracker-core/` (job log `logs/jobs/sf36-ctest-full-bytecmp.log`), orchestration audits. Checker OK. | Push the delivery branch, open the PR and record it; human review of the audited head (no `done`, master-checklist entry unchecked until approval). |
