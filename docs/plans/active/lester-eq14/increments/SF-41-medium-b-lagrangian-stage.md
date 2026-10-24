# SF-41 — Medium (b) for the Lagrangian consumer: lean Dirichlet stage, non-periodic-in-x1 spline and x-exit handling

- State: `pending`
- Goal: `Llevar el consumidor lagrangiano al medio (b) de Dirichlet en x: stage austero (solve_head con la ubicación de buffers de SF-40, cabeza bit a bit igual a la de SF-38, pico de memoria de dispositivo medido <= 30000 MiB en 3328x384x384 y 6656x256x256), spline cúbico no periódico en x1 del potencial, salida explícita por las caras x en el tracker y en el comparador Pollock, e integración en el runner de ensemble; verificado con controles exactos, representación cruzada contra el spline periódico y escalera de e_v, sin cambio de bytes en los caminos periódicos.`
- Depends on: `SF-38`
- Unlocks: `SF-39`
- Branch: `science/lester-sf41-medium-b-lagrangian-stage`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 (new representation verified on exact controls, refinement and an independent comparator; periodic paths: byte identity)`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Decision records `docs/decisions/2026-10-24-sf41-sf39-planning-decisions.md` (owner decisions E1-E3 of 2026-10-24),
`docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md` (items 2, 4, 7; stop rules R3, R4) and
`docs/decisions/2026-10-23-sf38-activation-decisions.md` (D1, D4, D9, D11). SF-38 selected medium (b) (`solve_head`, Dirichlet in
`x`, periodic `y`/`z`, harmonic face `K`: the B&dD and Par2 configuration) by its pre-registered equivalence test
(`docs/experiments/2026-10-23-sf38-long-domain-infrastructure.md`). The Lagrangian consumer traces `v_s = k_s grad Phi_s`
(SF-28 splines; helicity-free pointwise by its form, independently of periodicity); on medium (b) the potential is not periodic in
`x1`, the tracker must stop particles at the `x` faces, and medium (b) as built (92.571 B/cell at its solve peak) does not fit the
`sigma^2 >= 2.25` production box. SF-41 builds and verifies that path and is allowed to establish:

- (a) a lean medium (b) stage whose head is bit for bit SF-38's `DirichletMedium` head, and whether it fits 3328x384x384 and the
  SF-39 narrow-ladder rung 6656x256x256 within 30000 MiB (measured);
- (b) a cubic B-spline non-periodic in `x1` (periodic in `x2`, `x3`) that satisfies the SF-28 unit contracts and coincides with the
  periodic SF-28 spline in the interior when both represent the same periodic samples, up to an end effect that decays from the faces;
- (c) explicit `x`-exit handling in the forward tracker and the Pollock comparator for non-periodic media, with the periodic paths
  byte-identical;
- (d) the consumer on medium (b) passes exact controls and its Lagrangian `e_v` converges under grid refinement at `sigma^2` 1 and 4;
- (e) the ensemble runner runs medium (b) with the Gate 5 outputs (twin series, Pollock bridge) and the SF-39 options.

Not allowed: any `alpha` estimate (SF-39; Rule 1), any change of SF-18/SF-19/SF-28 numerics or of the periodic paths' outputs, any
claim about the production `lambda/h` beyond the transfer argument of E2 (SF-39 re-verifies it at the production length).

## Preconditions

- `SF-38` is `done` on the default branch (medium (b) selected; `DirichletMedium`, `EulerianFaceStats`, `EnsembleRunner`,
  `lagrangian_medium_ab`, the Par2 configs; its post-hoc readings P1-P6 and the Pollock discrepancy disclosed in its note).
- SF-37 and SF-40 are `done` (production `lambda/h`; lean stage S+ and its placement tricks).

## In scope

1. **Lean medium (b) stage** (`DirichletLongBoxStage`; E1). Same inputs as SF-38 medium (b): SF-18 `Y` on the box (`ny == nz`,
   `h = 1/ny`, even extents), `K = exp(eps Y)` built on the host, `solve_head` operator (matrix-free harmonic face `K`, Dirichlet
   `G1 L1` / 0 at `x = 0` / `L1`, periodic `y`/`z`), PCG + MG (`mg_levels` = the box auto rule), `rtol = 1e-10` on `||r|| / ||r_0||`.
   Placement (SF-40 tricks, bitwise by construction): `K` uploaded straight into the MG level-0 coefficient; the right-hand side
   assembled into the dead level-0 residual; `z`, `Ap` and level-0 `b` share one buffer through a **new** non-projected
   external-vector PCG (same arithmetic and order as `pcg_solve`; the existing functions are not changed); MG/PCG buffers released
   before the velocity faces; the head kept and turned in place into `f = -H - G1 x + G1 L1`; per-phase exact bytes and
   `MEM_PHASE` lines as SF-40. Estimated solve peak 60.57 B/cell (decision record, *estimate*). **Fallback (E1, pre-authorized):** if
   the measured PID peak of a production grid exceeds 30000 MiB, the head (PCG `x`) lives in pinned mapped host memory (bitwise);
   otherwise R3 (owner). Velocity faces (`compute_velocity_from_head`) kept for the Pollock bridge and the flow statistics; inlet
   samples from `U[0]` (D-1 labels optional, as in (a)).
   *Pre-activation precisions (decision 2026-10-24, addendum E12; code facts at `a3d98ae`):* `pcg_solve` reads its right-hand side
   once (`r = b - A x0`, `src/numerics/solvers/pcg.cuh:120-122`) and never recomputes a true residual, so no `b_hat` is needed and the
   right-hand side may live in a dead MG level-0 buffer (`r` or `x`; not level-0 `b`, which is `z = Ap`); the solve peak is then
   `8 [7 n + 4 sum_{l>=1} n_l]` bytes plus scratch (60.571 B/cell nominal at 7 levels, *derived from code*). `K` stays alive through the
   velocity phase (`compute_velocity_from_head` reads `K` and the head together), in the level-0 coefficient buffer. The lean stage
   reproduces `DirichletMedium`'s flux-balance and divergence diagnostics verbatim (file-local kernels at `DirichletMedium.cu:21-84`,
   host sums in index order) and its relative-residual convention `rel = initial > 0 ? final / initial : 0` (`DirichletMedium.cu:215`),
   so that the `B_SOLVER` lines can be identical. For medium (b) without an (a) solve, `G1 = 1`; `cg_max_iter = 2000` (as the A/B
   application).
2. **Non-periodic-in-`x1` spline** of the potential fluctuation `f`. Uniform cubic B-spline (C^2, the SF-28 degree and cell-centred
   sample lattice), periodic in `x2`, `x3` exactly as SF-28; in `x1` a non-cyclic coefficient solve on the cell-centred sample line.
   **Closure fixed before activation (decision 2026-10-24, addendum E12):** not-a-knot on the raw cell-centred samples, **no ghost
   samples** (unknowns `c_{-1}..c_{nx}`; rows: third-derivative continuity at the knots `x_1` and `x_{nx-2}`, i.e.
   `-c_{-1} + 4 c_0 - 6 c_1 + 4 c_2 - c_3 = 0` and its mirror, and the `nx` interpolation rows), plus one continuation coefficient per end
   (`c_{-2} = 4 c_{-1} - 6 c_0 + 4 c_1 - c_2` and its mirror: the end cubic pieces continued). It accepts general face data (the
   cross-representation test) and reproduces cubic polynomials in `x1` (constant in `x2`, `x3`); its accuracy is `O(h^4)` / `O(h^3)` /
   `O(h^2)` for value / gradient / Hessian up to the faces. The odd-extension ghost of the finite-volume Dirichlet data
   (`f(-h/2) = -f(h/2)`) is not used: an odd extension of `f` is only `C^1` at a face where `f'' != 0`, and thresholds A and B verify the
   raw-data closure, which is therefore the production closure. Evaluation (value, gradient, Hessian) is defined on the half-open domain
   `[-h/2, L1 + h/2)` (margin `h/2 >= h_max` for `--hmax-div >= 2`); outside it the field functor reports the position without reading
   outside the coefficients (the tracker turns it into an exit status, item 3); a particle is outside the medium when `x1 < 0` or
   `x1 > L1` (strict: C2 seeds lie on the face `x1 = 0`). Prefilter on the device: the not-a-knot line solve in `x` and the SF-28 FFT
   symbol in `y`/`z` (batched two-dimensional cuFFT per `x` plane); the CPU mirror solves the same `x` system and uses cyclic tridiagonal
   line solves in `y`/`z` (as the SF-28 CPU mirror). Reasons (design emulation and SF-38 evidence): cyclic line solves leave round-off
   variations across `x2`, `x3` on data constant there (emulated spurious transverse Hessian about 8 times the FFT's), which the layered
   C2 control's transverse-position and covector gates (`1e-12`) must not see, whereas the SF-28 cuFFT path kept the covectors of the
   layered SF-38 F2 control constant to 2.985e-14; and B's round-off margin is larger with FFT in `y`/`z` (Hessian 2.2e-12 against 3.4e-12).
   The prefilter's phases stay below the solve-phase peak (threshold E). The periodic SF-28 types, functions and outputs are not changed:
   **new types** (no template parameter is added to `PeriodicTricubicBSpline`, which stays read-only). **`ln k` uses the same non-periodic
   `x` geometry and closure** (not-a-knot on the raw `eps Y` samples; `Y` is periodic on the box, so its interior representation coincides
   with the periodic one up to the end effect, as threshold B measures for the potential): a periodic `ln k` spline of the non-periodic C2
   profile fails C2 (design emulation: `tau` 0.93e-9 / 1.35e-9 against `1e-9`, and a wrong `v_s` near the faces), and one geometry keeps the
   fused path (`FusedSplineLagrangianField` requires identical geometry for both splines, `FusedSplineLagrangianField.cuh:174-180`).
3. **`x`-exit handling** (E3), only for non-periodic media; the periodic paths produce identical bytes:
   - forward tracker: new statuses `exit_inlet` (crosses `x1 = 0` upstream) and `exit_outlet` (crosses `x1 = L1`); a DP5 step whose
     stages leave the evaluation domain is handled without reading outside the coefficients; plane pass: a particle that exits before the
     last plane is non-ok with its status, counted per plane; time pass: an exited particle stops and is excluded from later times with
     the counts per time recorded; plane lists bounded by `L1 - 5 lambda`; the existing `backflow_encounter` flag stays;
   - Pollock comparator / tracker core: a non-periodic mode in which crossing `x = 0` upstream or the outlet face ends the particle with a
     non-ok status (no wrap); the periodic mode is bitwise unchanged;
   - the SF-38 A/B (`lagrangian_medium_ab`) uses the exit-aware Pollock for medium (b).
   *Pre-activation precisions (decision 2026-10-24, addendum E12; code facts at `a3d98ae`):*
   - the tracker's status histogram is a fixed 7-slot array (`LagrangianTrackerTypes.hpp:84`; statuses `>= 7` fold into slot 6,
     `LagrangianTracker.cu:121`) printed in `TSTATUS` lines, `status.json` and the `inlet_slab` outputs: it is **not** enlarged; the exit
     statuses are defined in `lagrangian_labels` (the `StreamlineStatus` enum lives in the frozen closure-gate header) and are counted in
     separate fields written only in non-periodic mode;
   - Pollock exit codes are values in `0..15` not used by the Pollock core (its histograms have 16 slots and clamp `>= 16` into slot 15,
     `PollockComparator.cu:294-298`); the non-periodic mode needs the outlet face plane `U[nx]`, which the periodic repack drops
     (`FaceFluxRepack.cuh:14-19`): a new, additive non-periodic face view keeps it;
   - the seed rule is unchanged in both modes: a seed with non-positive flux is `seed_backflow`, weight 0, not traced by the forward
     tracker (`LagrangianTrackerDevice.cuh:485-489, 639-642`), still traced by Pollock (weight 0), and a seed set without any positive flux
     is an error in both (`LagrangianTracker.cu:105-107`, `PollockComparator.cu:293`);
   - with Dirichlet heads in `x` the discrete operator is an irreducible M-matrix, so the inlet face fluxes `U[0]` are positive and an RT0
     particle cannot reach `x = 0` moving upstream: Pollock exits through `x = 0` on medium (b) are expected to be zero up to solver error
     (*derivation*, measured by H); the spline tracker can in principle exit there (its `v_s` is not RT0).
4. **Runner integration** (`lagrangian_ensemble`, SF-38 `EnsembleRunner`): `--medium b` with injection plane `X0 = 5 lambda` (default
   for (b)), window `[0.1, 0.9]^2`, `M = 100`, the RR12 seed-face assertion, `--planes auto` / `--times auto` bounded by `L1 - 5 lambda`
   and by exits, the Par2 travel plane `X0 + L1 - 10 lambda`, twin series and Pollock bridge (on the (b) faces, exit-aware); multi-pass
   `--hmax-div 8,16,32` re-tracking the same stage build (outputs per pass; for SF-39's `h_max` ladder); per realization
   `flow_stats.json`: fraction of `x`-faces with `u <= 0`, percentiles (0, 0.1, 1, 50 %) of the cell-centre speed `|v| / qbar`, the
   interior (`[5 lambda, L1 - 5 lambda]`) mean face-flux vector and its tilt `theta_2`, `theta_3`; checkpoint/restart and byte
   reproducibility as SF-38 C-RUNNER (everything wall-clock only in `timings.json`).
   *Pre-activation precisions (decision 2026-10-24, addendum E12):* for `--medium b`, `G1 = 1` and `--times auto` = `n lambda / 2` up to
   `(L1 - 10 lambda) / qbar` with `qbar` the measured inlet mean flux (the mean advection time from `X0` to `L1 - 5 lambda`), exits
   recorded per time; medium (a) with a single `--hmax-div` keeps its file set, its `DONE` signature (`EnsembleRunner.cu:252-253, 900-909`)
   and its bytes (`flow_stats.json` and the per-pass outputs exist only for `--medium b` or a multi-pass list).
5. **Verification** (E2): exact controls, spline contracts, the cross-representation test, the `e_v` ladder on (b), production
   feasibility, regression; thresholds below.
6. Experiment note and artifact (`docs/experiments/<date>-sf41-medium-b-lagrangian-stage.md`, pre-registration in the artifact).

## Out of scope

- Any `alpha` estimate; SF-39 runs; plume-statistic ladders of the SF-37 type on medium (b) (E2: SF-39's production-length ladders own
  `lambda/h`).
- Changes to SF-18, SF-19, SF-28 numerics, to medium (a), to `solve_head`'s operator, MG or smoother, to Par2; float storage or any
  non-bitwise memory saving; D-1 labels for `X0 > 0`; a pseudo-symplectic consumer; medium (a) runs beyond the cross-representation test.

## Files and symbols

- New: the non-periodic spline (`src/numerics/interpolation/`, new files next to `PeriodicTricubicBSpline.{cuh,cu}`), the lean stage
  `src/physics/particles/lagrangian_labels/DirichletLongBoxStage.{cuh,cu}`, the external-vector non-projected PCG (an addition in
  `src/numerics/solvers/`; `pcg_solve` and `projected_pcg_solve_external` unchanged), tests under `tests/interpolation/` and
  `tests/lagrangian_labels/`, the experiment note and artifact.
- Changed (additions; periodic behaviour byte-identical): `LagrangianTracker` device code and types (exit statuses, non-periodic
  field functor), `FusedSplineLagrangianField` (only if the fused path is extended), `PollockTracker` / `PollockComparator` (non-periodic
  mode), `EnsembleRunner` and `apps/lagrangian_ensemble` (`--medium b`, multi-pass `h_max`, `flow_stats.json`),
  `apps/lagrangian_medium_ab` (exit-aware Pollock for (b)), `CMakeLists.txt`.
- Changed by owner decision 2026-10-24 (addendum E9), additive and opt-in: `DirichletMedium` gains `keep_host_head` (default `false`):
  when set, the head is downloaded to the host right before it is freed (`DirichletMedium.cu:294-295`), so that threshold D compares the
  real reference head; with the flag off nothing changes (threshold G checks the `lagrangian_medium_ab` outputs byte for byte).
  `FaceFluxRepack` / the Pollock face view: an additive non-periodic view keeping the outlet plane `U[nx]` (In scope 3). New tests go into
  new test executables, so the existing ones (`interpolation_periodic_tricubic`, `interpolation_hessian`, ...) are unchanged.
- Read-only: `PeriodicTricubicBSpline`, `solve_head` and its operator/MG/smoother, `compute_velocity_from_head`, `DirichletMedium`'s
  computation (the bitwise reference of the lean stage; only the opt-in head copy above is added), `LongBoxStage`, SF-18, SF-19, Par2.

## Implementation specification

1. The activation row fixes, before any run: the exit-status codes; the fields, seeds (4001 for the production
   grids, as SF-38 B; fresh seeds 6101-6103 for the `e_v` ladder, 6001 for the cross-representation test) and boxes of every threshold;
   the reading rules. Fixed before activation by decision 2026-10-24 (addendum E12, In scope 2): the `x` closure (not-a-knot on the raw
   samples, no ghost), the evaluation domain `[-h/2, L1 + h/2)`, `ln k` on the same non-periodic geometry (fused path available), the
   prefilter variant (`x` line solve with FFT in `y`/`z` on the device; the CPU mirror with cyclic line solves in `y`/`z`).
2. Bitwise reference for the lean stage: SF-38 `DirichletMedium` on the same `Y` (one SF-18 call feeds both), `eps`, `G1`, grid and
   `mg_levels` (head bytes through `keep_host_head`, `B_SOLVER` line); `DirichletMedium` and the lean stage run one after the other in the
   same process (their device peaks do not add).
3. Device peaks measured as SF-38 (nvidia-smi per PID, 200 ms, plus the device-wide `memory.used`; host RSS by `/usr/bin/time -v`).
4. Heavy work (production grids, the `e_v` ladder, the SF-38 A/B re-run, full `ctest`, byte-compare sets) as detached V100 jobs,
   serialized on the SF-41 mirror, one GPU. Every check on a grid larger than `32^3` (including A's ladder, C2, C4 at `h = 1/32` and D's
   128x32x32 and 512x64x64) runs on the V100 (owner rule: no local grid above `32^3`); locally only builds through
   `/home/sesquerre/.local/bin/mf3d-build` and new test cases `<= 32^3` selected by explicit `-R`.
5. The base binaries (SF-27 trio, the SF-38 20-run R set, the medium (a) runner and A/B references of G) are built and run from a clean
   tree at the increment base on the SF-41 mirror before any SF-41 source reaches it (SF-38 RR1 precedent); every build on the V100 is a
   detached `run` job (`exec` takes no GPU lock).

## Expected numerical effect

None on existing outputs (periodic spline, cube and medium (a) paths, closure gate, inlet slab, Par2: byte-identical). The SF-38 A/B
outputs of medium (b) change only for particles that leave through `x = 0` (expected none; counted). New medium (b) path: interior
coincidence with the periodic representation up to an end effect decaying geometrically from the faces (for the cubic B-spline the
per-knot factor of the homogeneous solution is `2 - sqrt(3) ~ 0.268`, *derivation*); `e_v` of order ~2 in `h` as in SF-37 V4.

## Validation commands

```bash
# local (light only: builds through mf3d-build; new SF-41 test cases <= 32^3 by explicit -R; nothing larger runs locally)
cmake --preset wsl-debug
/home/sesquerre/.local/bin/mf3d-build build/wsl-debug
nice -n 15 taskset -c 10-15 ctest --test-dir build/wsl-debug --output-on-failure -R '<new SF-41 fast cases, each <= 32^3>'
# V100 (detached jobs, one at a time, one GPU, SF-41 mirror; no sync while a job runs)
A=docs/experiments/artifacts/<date>-sf41-medium-b-lagrangian-stage/scripts
# base references first, from a clean tree at the increment base, before any SF-41 source reaches the mirror
scripts/remote --increment SF-41 sync   # tree = increment base
scripts/remote --increment SF-41 run sf41-basebuild -- "cmake --preset v100-release && cmake --build build/v100-release -j 2>&1 | tail -n 20; echo BUILD_EXIT=\${PIPESTATUS[0]}"
scripts/remote --increment SF-41 wait sf41-basebuild
scripts/remote --increment SF-41 run sf41-baserefs -- "bash <base-refs script committed with the activation> build/v100-release"
scripts/remote --increment SF-41 wait sf41-baserefs
# head
scripts/remote --increment SF-41 sync   # tree = source-bearing head
scripts/remote --increment SF-41 run sf41-build -- "cmake --preset v100-release && cmake --build build/v100-release -j 2>&1 | tail -n 20; echo BUILD_EXIT=\${PIPESTATUS[0]}"
scripts/remote --increment SF-41 wait sf41-build
for part in controls cross prodgrid evladder abrerun final; do scripts/remote --increment SF-41 run sf41-$part -- "bash $A/run_sf41.sh build/v100-release $part"; scripts/remote --increment SF-41 wait sf41-$part; done
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

Pre-registered verbatim in the activation bitácora row before any run; no threshold is tuned afterwards. The quantities, norms, fields
and boxes below were made explicit before activation by decision 2026-10-24 (addendum E8-E12) from a pre-activation review with design
emulations (`docs/experiments/2026-10-24-sf41-pre-activation-review.md`; numpy, not SF-41 evidence).

- **A. Spline contracts** (non-periodic spline; boxes with `nx != ny == nz`, `ny` a power of two, including points within one cell of
  the `x` faces; boxes above `32^3` on the V100). Every relative error is the maximum over the point set of the error divided by
  the maximum over the point set of the reference magnitude (value: absolute value; gradient: Euclidean norm per point; Hessian: the six
  components):
  - node interpolation: `max |s - f| / max |f| <= 1e-12` at every cell centre, including the cells adjacent to the faces;
  - partition of unity: the constant `3.0`, `max |s - 3| <= 1e-14` (absolute) on `[0, L1]` including points within one cell of the faces;
    the gradient and the margin points `[-h/2, 0)`, `[L1, L1 + h/2)` recorded (SF-28 gated the gradient at `1e-13`; with the not-a-knot closure
    it is emulated at 2.9e-14 to 3.9e-13 depending on `h` and on the `x` solver, so it is recorded, not gated);
  - exact reproduction of the closure's class (polynomials of degree `<= 3` in `x1`, constant in `x2`, `x3`) through the prefilter, on 64x16x16 and
    128x16x16: value and gradient `<= 1e-12` relative (emulated <= 4.4e-15 and <= 2.7e-13); the Hessian recorded (its relative round-off
    grows like `nx^2`: emulated 7.5e-13 to 7.7e-12 at `nx = 64`, 1.3e-12 to 2.9e-11 at `nx = 128`);
  - order ladder (closure on the raw data, no ghost samples) on the boxes 32x16x16, 64x32x32, 128x64x64 (`L1 = 2`; `16/32/64` = cells per
    unit length) and a fixed physical point set (1e4 random points in `[0, L1] x [0, 1)^2` plus points within one coarsest cell of each face,
    all inside `[0, L1]`): the minimum observed order over the pairs 16->32 and 32->64 is value `>= 3.8` and gradient `>= 2.8` in both the
    max and the RMS norm, Hessian `>= 1.8` in the max norm. The field is smooth, non-periodic in `x1`, periodic in `x2`, `x3`, and exercises
    the `x` closure; it is fixed before activation with its orders predicted by an independent exact emulation (owner decision 2026-10-24,
    addendum E11: the near-face convergence is pre-asymptotic at 16/32/64 and the outcome depends on the field, e.g. 1.77 for the Hessian
    of a natural `x`-dominated field): `f = exp(x1) sin(3 x1) (1 + 0.05 sin(2 pi x2) cos(2 pi x3))`, whose finest-level value and Hessian
    errors are 95 % from the `x` closure and lie at the face; predicted minimum orders 3.9024 (value, max) / 4.3659 (value, RMS) / 2.9042
    (gradient, max) / 3.3858 (gradient, RMS) / 1.9154 (Hessian), i.e. 0.10-0.12 above the gates (the implementation must reproduce this
    mathematics up to round-off, far below the errors 9.4e-6 / 2.1e-3 / 0.27 at `h = 1/64`). Recorded: the `x`-only near-face sub-ladder
    16/32/64/128 (predicted 64->128 orders 3.98 / 2.98 / 1.99) and the difference between the measured and the emulated errors;
  - CPU vs GPU `<= 1e-13` relative, on 32x16x16 and 64x32x32: the value for the full pipeline (GPU prefilter and evaluation against the
    CPU mirror), and value, gradient and Hessian for the evaluation alone on the same coefficients. Recorded: the coefficient difference
    (relative to `max |c|`) and the full-pipeline gradient and Hessian, inside, near the faces and in the margins (the not-a-knot end rows
    amplify a round-off difference between two prefilter algorithms by up to about 33 at `c_{-2}` and 8 at `c_{-1}`; design emulation of
    the y/z FFT against the cyclic tridiagonal CPU mirror: full-pipeline gradient up to 1.9e-13 and Hessian up to 1.6e-12 at `h = 1/32`,
    and even the SF-28 periodic pair gives a full-pipeline Hessian of 3.2e-13 there; SF-28 gated only value and gradient);
  - Hessian-path value and gradient bitwise equal (`memcmp`) to the value path, on the host and on the device;
  - no device allocation during evaluation (`cudaMemGetInfo` free bytes equal before and after, value and Hessian paths);
  - `y`/`z` shift by whole periods (`q` in {-2, -1, 1, 2, 3}) on dyadic coordinates `k / 2^16` (the SF-28 method): value, gradient and
    Hessian `<= 1e-14` absolute (bitwise equality expected);
  - evaluation domain: points in the margin `[-h/2, 0)`, `[L1, L1 + h/2)` evaluate finite and equal to the continued end cubic pieces
    (`<= 1e-12` relative); points outside `[-h/2, L1 + h/2)` are reported without reading outside the coefficients.
- **B. Cross-representation** (E2). On the SF-19 potential fluctuation of a medium (a) field (SF-18, `sigma^2 = 4`, `lambda/h = 32`, box
  1664x128x128, seed 6001) represent the same samples with the periodic SF-28 spline and with the non-periodic spline (closure on the raw
  face data). At every evaluation point (a lattice of at least 1e5 points plus the cell centres of three `x1` lines) with distance `>= 2 lambda`
  from both `x` faces: `max |value_np - value_p| / max |value_p|`, the same for the gradient (Euclidean norm per point) and the Hessian
  (six components), `<= 1e-11`. Recorded: the difference against the distance from the faces along the three `x1` lines over their full
  length and its fitted per-knot decay factor (expected ~0.268), fitted on the knots whose difference exceeds `1e3` times the interior
  round-off floor; the same comparison for the `ln k` spline (recorded). Design emulation: the interior difference is a round-off floor
  growing like `eps (lambda/h)^2` for the Hessian: 2.2e-12 with FFT in `y`/`z` (3.4e-12 with cyclic line solves) for the smoothest
  surrogate (linearized head), down to about 3e-14 for a K-like spectrum: a worst-case margin of about 4.6 with the chosen prefilter.
- **C. Exact controls on medium (b)** (64x16x16 and 128x16x16 unless stated). The representation and tracker controls C1-C4 use the exact
  head written into the sample array (the analytic linear head for `k = 1`; the exact discrete one-dimensional solution for layered `k`), so they
  test the spline, the field functor and the tracker free of the solver's iteration error (a solved head carries `O(rtol)` transverse variations from
  the red-black smoother, which a `1e-12` threshold must not see); C5 tests the solver against the same exact heads. `G1 = 1` and dyadic
  `h`, `L1` throughout C (with `k = 1` from `eps Y = 0`, so that `exp` gives exactly 1, the written linear head gives `f = 0` exactly and
  the PCG residual `r_0 = b - A x_0` of the linear initial guess is exactly 0). Tracking settings: `tol = 1e-8`, `h_max = h/8`:
  - C1 `k = 1`: `f = 0` exactly and `v_s = G1 e1`; transverse displacement `<= 1e-12`; `tau` against `Delta x1 / G1` `<= 1e-9` relative;
    `|I| <= 1e-14`; covectors constant `<= 1e-12`;
  - C2 layered non-periodic `k(x1)` (a smooth `k` whose `x1` profile is not periodic over `L1`) on 64x32x32 and 128x32x32: transverse positions and
    covectors constant `<= 1e-12`; `tau` against an independent quadrature of `int dx1 / v_s1` `<= 1e-9` relative; `I` recorded (SF-36 A2 as
    amended). Pre-registered (addendum E10): `ln k = 0.5 cos(2 pi 0.6 x1 + 0.3)` (not periodic over `L1` = 2 or 4), the exact discrete
    one-dimensional head (harmonic face `K`, Dirichlet faces at `h/2`, constant face flux), both splines on the non-periodic geometry,
    `X0 = 0` (seeds on the inlet face), planes {0.25, 0.5, 1, 1.5, 1.75} (`L1 = 2`) and {0.25, 0.5, 1, 2, 3, 3.75} (`L1 = 4`), the quadrature
    of the SF-36/SF-38 controls helper (adaptive Gauss-Kronrod on the host spline functor of the same field); design emulation: `tau` about
    5e-11 (a margin of about 20; the SF-38 F2 profile at the same `h` gave 4.771e-10);
  - C3 shift by `+1` in `x2` or `x3`: same field `<= 1e-14`. On the C1 and C2 fields every transverse shift is trivially exact, so C3 runs on
    a written field with transverse structure (addendum E10), on 64x16x16 and 128x16x16:
    `f = 0.1 cos(2 pi x2) cos(2 pi x3) exp(-2 pi sqrt(2) x1)`, `ln k = 0.1 sin(2 pi x2) cos(2 pi x3) + 0.2 cos(2 pi 0.6 x1 + 0.3)`
    (both splines on the non-periodic geometry; it exercises the `x2` and `x3` wrap of both splines and the not-a-knot `ln k`; `|v_s|` stays
    in about [0.2, 2.2]); compared quantity: the velocity components `v_s` of the field functor at `p` and `p + e_2`, `p + e_3`; gate per
    component `|d| <= 1e-14` and `|d| <= 1e-14 |v_s(p)|` (the SF-38 F3 method) on 4096 random points (every fourth pushed to transverse faces
    or edges) and on dyadic points `k / 2^16` (bitwise expected). Design emulation: max abs about 1.1e-15 and max rel about 3.7e-15; with
    amplitudes 0.5 / 0.3 the per-point relative gate is predicted to fail where `|v_s|` is about 0.05 (1.1e-14), which is why the amplitudes
    are small;
  - C4 exits (amended by owner decision 2026-10-24, addendum E8: the reversed head drop is not measurable, because a seed with non-positive
    flux is `seed_backflow` and not traced, and a seed set without any positive flux is an error in the tracker and in the Pollock lattice
    API):
    - C4a outlet: `k = 1`, `G1 = 1`, 64x16x16 and 128x16x16: particles traced past `L1` (a plane beyond `L1`; times beyond `L1 / G1`) end
      `exit_outlet` in the tracker's plane and time passes and in the Pollock non-periodic plane and time modes (all of them; no non-finite
      reached record or final state; unreached records are NaN by design); counts exact;
    - C4b inlet: the harmonic `k = 1` field written into the sample array, `Phi = G1 x1 + a cos(2 pi x2) exp(-2 pi x1)` (`a > 1/(2 pi)`: a
      backflow lobe at the inlet), with the exact two-dimensional stream function `psi = G1 x2 - a exp(-2 pi x1) sin(2 pi x2)` (separatrix
      `psi = 0` mod 1), on 64x32x32 and 128x32x32: `a = 0.5`, `G1 = 1`, `X0 = 0.1`, the `64 x 64` seed lattice of the unit section
      (`x2`, `x3` = `(m + 1/2) / 64`), keeping the seeds with analytic `v1 >= 0.05` and `|psi0 - round(psi0)| >= 0.02` (40 of the 64 `x2`
      values, 2560 seeds: 768 `exit_inlet`, 1792 downstream; the same on both grids); plane at `L1 / 2`; times up to `T = 3.0` (`L1 = 2`) and
      `5.5` (`L1 = 4`). Design emulation: no disagreement between the exact classification, a DP5 emulation on the not-a-knot spline field
      and an RT0 emulation on the analytic face fluxes (Pollock misclassifies only within 4.6e-4 of the separatrix in `psi`, the margin is
      0.02); each seed classified
      `exit_inlet` or reaching the plane `L1 / 2` (plane pass) / `exit_outlet` (time pass) by the exact stream function with an independent
      integration of the analytic field; the tracker (both passes) and the Pollock non-periodic mode on the analytically integrated face fluxes
      give that classification for every seed; counts exact;
    - recorded, no threshold: under a reversed head drop every seed has non-positive flux and both APIs report the error by design;
    - the tracker and Pollock status codes differ: the comparison uses the pre-registered semantic map (`exit_inlet`, `exit_outlet`,
      reached / ok);
  - C5 solver on the C1 and C2 media (through the lean stage): relative residual `<= 1e-10` (`rel = initial > 0 ? final / initial : 0`, the
    `DirichletMedium` convention: on C1 the initial residual is exactly 0 and the PCG returns at once); flux balance `<= 1e-8`; the maximum
    relative difference between the solved and the exact head recorded (no threshold).
- **D. Lean stage bitwise.** The lean stage's head equals SF-38 `DirichletMedium`'s head byte for byte (and the `B_SOLVER` line is identical)
  on 128x32x32 and 512x64x64 and on 3328x256x256, `eps = 2`, `ell = 0.125`, seed 4001 (the SF-38 C0 case); the same if the mapped-host
  fallback is used. All three on the V100 (addendum E10: the spec's earlier "(local)" for the two small sizes conflicts with the owner's
  rule of no local grid above `32^3`). The reference head is read through `DirichletMedium`'s opt-in `keep_host_head` (addendum E9); the
  mapped-host head path is compared at the two small sizes whether or not a production grid needs it. `G1 = 1` at the two small sizes; at
  3328x256x256, `G1 = G[0]` of the medium (a) lean stage of the same field (the SF-38 C0 input), so that both `B_SOLVER` lines are also
  compared with SF-38's committed C0 line (`iterations=50 initial=6.724494e+06 ... rel=9.705149e-11`; an environment control, recorded).
- **E. Production feasibility** (seed 4001, SF-41 runner with `--medium b`, `1e4` particles over the full measurement length, plane and time
  passes, twin series, Pollock bridge, `flow_stats.json`): E1 2048x256x256 `eps = 1` `ell = 10/256`; E2 3328x384x384 `eps = 1.5`
  `ell = 16/384`; E3 3328x384x384 `eps = 2` `ell = 32/384`; E4 6656x256x256 `eps = 2` `ell = 64/256` (the SF-39 narrow-ladder rung at
  `sigma^2 = 4`). In each: the solve reaches `||r|| / ||r_0|| <= 1e-10` (fallback `<= 1e-8` with balance met, recorded, as SF-38 C0); flux
  balance `|Q(x=0) - Q(x=L1)| / |Q(x=0)| <= 1e-8`; PID peak `<= 30000 MiB` (with the fallback if needed, recorded); exit and non-ok counts,
  wall time, host RSS and per-phase bytes recorded. Settings (addendum E12): `G1 = 1`, `cg_max_iter = 2000`, `--x0` = `5 lambda`,
  `--planes auto` / `--times auto` as In scope 4, `--hmax-div 8`, `--fused-splines --single-launch` if the fused path is built, window
  `[0.1, 0.9]^2`, `M = 100`. Every non-solve phase is designed to stay at or below the solve-phase exact bytes; its exact bytes are
  reported per phase.
- **F. Lagrangian `e_v` on medium (b)** (SF-37 V4 analogue). Seeds 6101-6103; injection at `5 lambda`, planes `X0 + m lambda` up to
  `L1 - 5 lambda`; box `24 x 2 x 2 lambda`, `lambda/h` in {16, 32, 64, 128} at both amplitudes (the SF-37 V4 range; largest rung
  3072x256x256). For every amplitude, seed and plane, RMS `e_v` decreases on every refinement with observed
  order `>= 1.8` on the last pair. Recorded: `D_P = sigma_T^2(Pollock) - sigma_T^2(consumer)` per plane and `lambda/h` with the SF-37 V3b
  classification (non-blocking). Parameters (addendum E10): SF-18 Gaussian `ell = 0.5` (transverse period 1 = `2 lambda`), `normalize_variance`
  on, `eps` 1 and 2 (`sigma^2` 1 and 4); grids 384x32x32, 768x64x64, 1536x128x128, 3072x256x256 (`L1 = 12`); `X0 = 5 lambda`; planes
  `X0 + m lambda`, `m = 1..14` (the last at `L1 - 5 lambda`, inclusive), the same physical planes at every rung; window `[0.1, 0.9]^2`,
  `M = 100` (1e4 seeds, the runner default: with `M = 256` the window puts seeds exactly on transverse faces at `ny = 256`, which the
  RR12 seed-face assertion rejects; with `M = 100` no seed lies on a face for `ny` a power of two, decision D9), flux weights; `tol = 1e-8`, `h_max = h/8`, `G1 = 1`; RMS `e_v` = the tracker's flux-weighted RMS
  of `e_v = 1 - exp(-I)` over the particles that reached the plane; observed order on a pair `(N, 2N)` = `log2(e_N / e_2N)` and "decreases" =
  `e_2N < e_N` on every consecutive pair (SF-37 RR4); a missing or non-finite value is "not met". `D_P` uses the exit-aware Pollock on the
  same faces and seeds, `SE_D` by a block bootstrap over 4 blocks of 50 x 50 seeds (`B = 2000`, generator seed fixed in the activation
  row). Reference: SF-37 V4 on the cube, last-pair orders 1.996-2.007 in 45 of 45 cells, no floor at `lambda/h = 128`.
- **G. Regression.** Full V100 `ctest` green; the SF-27 trio and the SF-38 20-run R set byte-identical to base binaries (SF-38 F method); a
  medium (a) `lagrangian_ensemble` run (twin and bridge on) and a `lagrangian_medium_ab` medium (a) run byte-identical to base binaries
  (`timings.json` excluded; the byte-compare sets are listed in the activation row); `interpolation_periodic_tricubic` and `interpolation_hessian` unchanged
  and green. Reading (addendum E10): `lagrangian_medium_ab` always runs both media, so "a medium (a) run" is a full A/B run whose every
  output except `timings.json` is compared byte for byte (new exit data, if any, only in new files listed as allowed extras); the medium (a)
  `lagrangian_ensemble` comparison includes the `DONE` files; the base binaries come from the base build of Implementation 5.
- **H. Recorded: SF-38 A/B re-run with the exit-aware Pollock** (both amplitudes, seeds 4001-4010): per realization and plane the number of
  medium (b) particles that left through `x = 0`; where it is zero, `e2_b.csv` is byte-identical to SF-38's; where it is not, the SF-38
  `C_STAT` lines are recomputed and recorded beside the originals (the SF-38 verdict is not re-decided). Reading (addendum E10): the exit
  counts go to a new per-realization file (the `e2_b.csv` format is unchanged; its `non_ok` column is a per-run total); the comparison is
  against the 20 committed SF-38 files (`docs/experiments/artifacts/2026-10-23-sf38-long-domain-infrastructure/raw/sf38_campaign/ab/ab_s{1,4}/realizations/r_*/e2_b.csv`),
  with `e2_a.csv` (periodic Pollock, unchanged code) as the environment control; expected (*derivation*, In scope 3): zero exits through
  `x = 0` and 20 of 20 identical files.

## Regression surface

- The periodic spline module and every periodic consumer (cube, medium (a), closure gate, inlet slab); the tracker device code
  (status enum, landing logic); the Pollock core; `EnsembleRunner` and `lagrangian_ensemble` (medium (a) outputs byte-identical);
  `ctest` wall time.
- Compiler contraction: code added to templated device functions can change inlining and FMA contraction of the periodic instantiations;
  the non-periodic paths are separate instantiations or types, and the SF-38 R set (inlet slab: tracker and Pollock) is the canary (G).
- `DirichletMedium` (opt-in head copy only; flag off = unchanged bytes, G).

## Failure and rollback policy

- A, B, C, D or G not met -> defect (BLOCKING), corrective in this increment.
- E: solve not converged -> C0 fallback rule; otherwise stop (owner). Peak above 30000 MiB in the solve phase -> mapped-host head (E1);
  still above -> R3, owner decision on the domain of SF-39, recorded. A non-solve phase (spline prefilter, velocity, tracking, Pollock
  bridge repack) whose peak exceeds the solve-phase peak or 30000 MiB is a placement defect (BLOCKING), corrected in this increment before
  R3 is considered (addendum E10: the pre-authorized fallback addresses only the solve phase).
- F not met at an amplitude -> the consumer is not defensible on medium (b) at that amplitude (as R1); stop, owner.
- H shows exits -> recorded; the SF-38 decision is not re-opened unless the recomputed `C_STAT` makes all ten statistics equivalent
  (then: owner). H with zero exits but `e2_b.csv` different while the environment control `e2_a.csv` is identical -> defect of the
  exit-aware Pollock (BLOCKING); `e2_a.csv` different -> environment or build difference, investigated and recorded before any reading.

**Compute (decision 2026-10-24 E1-E3).** One V100, every remote job serialized on the SF-41 mirror; GPU-hours reported per part.

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

`SF-39` becomes eligible after this increment, `SF-37` and `SF-38` are merged and marked `done` on the default branch.

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-24T06:00Z | not started | Specification created on the docs-only governance PR (branch `docs/lester-sf41-sf39-planning`) after SF-38 selected medium (b) (R4) and the owner decisions E1-E3 of 2026-10-24 (`docs/decisions/2026-10-24-sf41-sf39-planning-decisions.md`): lean medium (b) stage with a pre-authorized mapped-host fallback, non-periodic-in-`x1` spline verified by representation and cross-representation, `x`-exit handling for the tracker and Pollock (closing the SF-38 MINOR discrepancy), runner integration with the SF-39 options. Figures used: SF-38 post-hoc P1-P6 (exploratory), the orchestrator's code reconnaissance (lean stage 60.57 B/cell, *estimate*). | Depends on SF-38 `done` on `master`. | Activate only when the checker reports it READY; the activation row fixes the items of "Implementation specification" 1 before any run. |
| 2026-10-24T21:00Z | not started | Amended before activation on a second docs-only governance PR (branch `docs/lester-sf41-activation-governance`) by the owner decisions of 2026-10-24, addendum E8-E12 of `docs/decisions/2026-10-24-sf41-sf39-planning-decisions.md`, after a pre-activation review of every threshold against the code at `a3d98ae` (five code-reading reports, two adversarial reviews, numpy design emulations: `docs/experiments/2026-10-24-sf41-pre-activation-review.md` and its artifact). Found: C4's reversed-head-drop case unmeasurable (non-positive seeds are `seed_backflow` untraced; both APIs throw without a positive seed) -> E8 two exact controls (C4a outlet, C4b harmonic inlet lobe with exact stream function); D's reference head inaccessible (`DirichletMedium` frees a local head) -> E9 opt-in `keep_host_head`; A's ladder pre-asymptotic near the faces (natural field Hessian 1.77) -> E11 field `exp(x1) sin(3 x1) (1 + 0.05 S)` with emulated orders; E10 clarifications (D small sizes on the V100; non-solve-phase peaks are defects; C3 on a written transverse field with small amplitudes, since the SF-38 F3 per-point relative gate would fail at amplitude 0.5; A's gated quantities: full-pipeline value and evaluation-only derivatives for CPU vs GPU, value and gradient for reproduction, value for partition of unity; C2 profile, planes, `X0`; F's `M = 256`, weighting, order convention, blocks; G and H readings; base build and references, detached builds, local work `<= 32^3`); E12 design items fixed (not-a-knot raw closure without ghost, domain `[-h/2, L1 + h/2)`, `ln k` on the same geometry, device prefilter `x` line solve + FFT `y/z`, new types, exit statuses outside the 7-slot histogram, Pollock codes `0..15` and a non-periodic face view with `U[nx]`, `G1 = 1`, `cg_max_iter = 2000`, `--times auto` bound). The Goal, the threshold values and the failure policy's stop rules are unchanged except where the decision record says. The previous text is in git history at `a3d98ae`. | Docs only; no SF-41 run; the emulations are design evidence, not SF-41 evidence. | Activate only after this PR is merged and the checker reports SF-41 READY; the activation row pre-registers the thresholds verbatim. |
