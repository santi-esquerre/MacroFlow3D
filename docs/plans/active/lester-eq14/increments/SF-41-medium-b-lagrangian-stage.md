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
2. **Non-periodic-in-`x1` spline** of the potential fluctuation `f`. Uniform cubic B-spline (C^2, the SF-28 degree and cell-centred
   sample lattice), periodic in `x2`, `x3` exactly as SF-28; in `x1` a non-cyclic coefficient solve on the cell-centred sample line with a
   boundary closure chosen in the DAG and fixed in the activation row (for example not-a-knot), optionally extended by one ghost sample per
   face carrying the finite-volume Dirichlet data (for `f`, which vanishes on the faces: `f(-h/2) = -f(h/2)`, the ghost the cell-centred
   Dirichlet stencil uses); the choice and its near-face accuracy are recorded (an odd extension of `f` is only `C^1` at a face where
   `f'' != 0`, so near-face order may drop; particles are injected `5 lambda` from the inlet and measured up to `5 lambda` from the outlet).
   The closure must accept general face data (for the cross-representation
   test). Evaluation (value, gradient, Hessian) defined on `[0, L1]` plus a margin of at least `h_max` (padded coefficients); outside,
   the field functor reports the position (the tracker turns it into an exit status, item 3). Prefilter on the device (separable line
   solves, or FFT in `y`/`z` plus tridiagonal in `x`; decided in the DAG with its memory) with a CPU mirror. The periodic SF-28 types,
   functions and outputs are not changed (new types or a template parameter with the periodic instantiation bitwise unchanged). The
   `ln k` spline stays periodic (`Y` is periodic on the box) or uses the non-periodic `x` geometry to keep the fused path (decided in the
   DAG with byte and accuracy evidence).
3. **`x`-exit handling** (E3), only for non-periodic media; the periodic paths produce identical bytes:
   - forward tracker: new statuses `exit_inlet` (crosses `x1 = 0` upstream) and `exit_outlet` (crosses `x1 = L1`); a DP5 step whose
     stages leave the evaluation domain is handled without reading outside the coefficients; plane pass: a particle that exits before the
     last plane is non-ok with its status, counted per plane; time pass: an exited particle stops and is excluded from later times with
     the counts per time recorded; plane lists bounded by `L1 - 5 lambda`; the existing `backflow_encounter` flag stays;
   - Pollock comparator / tracker core: a non-periodic mode in which crossing `x = 0` upstream or the outlet face ends the particle with a
     non-ok status (no wrap); the periodic mode is bitwise unchanged;
   - the SF-38 A/B (`lagrangian_medium_ab`) uses the exit-aware Pollock for medium (b).
4. **Runner integration** (`lagrangian_ensemble`, SF-38 `EnsembleRunner`): `--medium b` with injection plane `X0 = 5 lambda` (default
   for (b)), window `[0.1, 0.9]^2`, `M = 100`, the RR12 seed-face assertion, `--planes auto` / `--times auto` bounded by `L1 - 5 lambda`
   and by exits, the Par2 travel plane `X0 + L1 - 10 lambda`, twin series and Pollock bridge (on the (b) faces, exit-aware); multi-pass
   `--hmax-div 8,16,32` re-tracking the same stage build (outputs per pass; for SF-39's `h_max` ladder); per realization
   `flow_stats.json`: fraction of `x`-faces with `u <= 0`, percentiles (0, 0.1, 1, 50 %) of the cell-centre speed `|v| / qbar`, the
   interior (`[5 lambda, L1 - 5 lambda]`) mean face-flux vector and its tilt `theta_2`, `theta_3`; checkpoint/restart and byte
   reproducibility as SF-38 C-RUNNER (everything wall-clock only in `timings.json`).
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
- Read-only: `PeriodicTricubicBSpline`, `solve_head` and its operator/MG/smoother, `compute_velocity_from_head`, `DirichletMedium` (the
  bitwise reference of the lean stage), `LongBoxStage`, SF-18, SF-19, Par2.

## Implementation specification

1. The activation row fixes, before any run: the `x` boundary closure and the ghost rule; the evaluation margin; the prefilter variant;
   the fused/unfused choice for `ln k`; the exit-status codes; the fields, seeds (4001 for the production grids, as SF-38 B; fresh seeds
   6101-6103 for the `e_v` ladder, 6001 for the cross-representation test) and boxes of every threshold; the reading rules.
2. Bitwise reference for the lean stage: SF-38 `DirichletMedium` on the same `Y`, `eps`, `G1`, grid and `mg_levels` (head bytes,
   `B_SOLVER` line).
3. Device peaks measured as SF-38 (nvidia-smi per PID, 200 ms, plus the device-wide `memory.used`; host RSS by `/usr/bin/time -v`).
4. Heavy work (production grids, the `e_v` ladder, the SF-38 A/B re-run, full `ctest`, byte-compare sets) as detached V100 jobs,
   serialized on the SF-41 mirror, one GPU.

## Expected numerical effect

None on existing outputs (periodic spline, cube and medium (a) paths, closure gate, inlet slab, Par2: byte-identical). The SF-38 A/B
outputs of medium (b) change only for particles that leave through `x = 0` (expected none; counted). New medium (b) path: interior
coincidence with the periodic representation up to an end effect decaying geometrically from the faces (for the cubic B-spline the
per-knot factor of the homogeneous solution is `2 - sqrt(3) ~ 0.268`, *derivation*); `e_v` of order ~2 in `h` as in SF-37 V4.

## Validation commands

```bash
cmake --preset wsl-debug
/home/sesquerre/.local/bin/mf3d-build build/wsl-debug
nice -n 15 taskset -c 10-15 ctest --test-dir build/wsl-debug --output-on-failure -R 'interpolation|lagrangian'
scripts/remote --increment SF-41 sync
scripts/remote --increment SF-41 exec -- "cmake --preset v100-release && cmake --build build/v100-release -j 2>&1 | tail -n 20; echo BUILD_EXIT=${PIPESTATUS[0]}"
A=docs/experiments/artifacts/<date>-sf41-medium-b-lagrangian-stage/scripts
for part in controls cross prodgrid evladder abrerun final; do scripts/remote --increment SF-41 run sf41-$part -- "bash $A/run_sf41.sh build/v100-release $part"; scripts/remote --increment SF-41 wait sf41-$part; done
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

Pre-registered verbatim in the activation bitácora row before any run; no threshold is tuned afterwards.

- **A. Spline contracts** (non-periodic spline; boxes with `nx != ny == nz`, including points within one cell of the `x` faces):
  node interpolation `<= 1e-12` relative; partition of unity `<= 1e-14`; exact reproduction (`<= 1e-12`) of the polynomial class the chosen
  closure reproduces (stated in the activation row); order ladder on a smooth non-periodic field on 16/32/64 (closure on the raw data, no ghost samples): value `>= 3.8`, gradient
  `>= 2.8`, Hessian `>= 1.8`; CPU vs GPU `<= 1e-13`; Hessian-path value and gradient bitwise equal to the value path; no device allocation
  during evaluation; `y`/`z` shift by one period `<= 1e-14`.
- **B. Cross-representation** (E2). On the SF-19 potential fluctuation of a medium (a) field (SF-18, `sigma^2 = 4`, `lambda/h = 32`, box
  1664x128x128, seed 6001) represent the same samples with the periodic SF-28 spline and with the non-periodic spline (closure on the raw
  face data). At every evaluation point (a lattice of at least 1e5 points plus the cell centres of three `x1` lines) with distance `>= 2 lambda`
  from both `x` faces: `max |value_np - value_p| / max |value_p|`, the same for the gradient and the Hessian, `<= 1e-11`. Recorded: the
  difference against the distance from the faces and its fitted per-knot decay factor (expected ~0.268).
- **C. Exact controls on medium (b)** (64x16x16 and 128x16x16 unless stated). The representation and tracker controls C1-C4 use the exact
  head written into the sample array (the analytic linear head for `k = 1`; the exact discrete one-dimensional solution for layered `k`), so they
  test the spline, the field functor and the tracker free of the solver's iteration error (a solved head carries `O(rtol)` transverse variations from
  the red-black smoother, which a `1e-12` threshold must not see); C5 tests the solver against the same exact heads:
  - C1 `k = 1`: `f = 0` exactly and `v_s = G1 e1`; transverse displacement `<= 1e-12`; `tau` against `Delta x1 / G1` `<= 1e-9` relative;
    `|I| <= 1e-14`; covectors constant `<= 1e-12`;
  - C2 layered non-periodic `k(x1)` (a smooth `k` whose `x1` profile is not periodic over `L1`) on 64x32x32 and 128x32x32: transverse positions and
    covectors constant `<= 1e-12`; `tau` against an independent quadrature of `int dx1 / v_s1` `<= 1e-9` relative; `I` recorded (SF-36 A2 as
    amended);
  - C3 shift by `+1` in `x2` or `x3`: same field `<= 1e-14`;
  - C4 exits: with `k = 1`, particles traced past `L1` end `exit_outlet` (all of them, no non-finite value), particles started with a reversed
    head drop end `exit_inlet`; the Pollock non-periodic mode gives the same statuses; counts exact;
  - C5 solver on the C1 and C2 media (through the lean stage): relative residual `<= 1e-10`; flux balance `<= 1e-8`; the maximum relative
    difference between the solved and the exact head recorded (no threshold).
- **D. Lean stage bitwise.** The lean stage's head equals SF-38 `DirichletMedium`'s head byte for byte (and the `B_SOLVER` line is identical)
  on 128x32x32 and 512x64x64 (local) and on 3328x256x256, `eps = 2`, `ell = 0.125`, seed 4001 (the SF-38 C0 case, V100); the same if the
  mapped-host fallback is used.
- **E. Production feasibility** (seed 4001, SF-41 runner with `--medium b`, `1e4` particles over the full measurement length, plane and time
  passes, twin series, Pollock bridge, `flow_stats.json`): E1 2048x256x256 `eps = 1` `ell = 10/256`; E2 3328x384x384 `eps = 1.5`
  `ell = 16/384`; E3 3328x384x384 `eps = 2` `ell = 32/384`; E4 6656x256x256 `eps = 2` `ell = 64/256` (the SF-39 narrow-ladder rung at
  `sigma^2 = 4`). In each: the solve reaches `||r|| / ||r_0|| <= 1e-10` (fallback `<= 1e-8` with balance met, recorded, as SF-38 C0); flux
  balance `|Q(x=0) - Q(x=L1)| / |Q(x=0)| <= 1e-8`; PID peak `<= 30000 MiB` (with the fallback if needed, recorded); exit and non-ok counts,
  wall time, host RSS and per-phase bytes recorded.
- **F. Lagrangian `e_v` on medium (b)** (SF-37 V4 analogue). Seeds 6101-6103; injection at `5 lambda`, planes `X0 + m lambda` up to
  `L1 - 5 lambda`; box `24 x 2 x 2 lambda`, `lambda/h` in {16, 32, 64, 128} at both amplitudes (the SF-37 V4 range; largest rung
  3072x256x256). For every amplitude, seed and plane, RMS `e_v` decreases on every refinement with observed
  order `>= 1.8` on the last pair. Recorded: `D_P = sigma_T^2(Pollock) - sigma_T^2(consumer)` per plane and `lambda/h` with the SF-37 V3b
  classification (non-blocking).
- **G. Regression.** Full V100 `ctest` green; the SF-27 trio and the SF-38 20-run R set byte-identical to base binaries (SF-38 F method); a
  medium (a) `lagrangian_ensemble` run (twin and bridge on) and a `lagrangian_medium_ab` medium (a) run byte-identical to base binaries
  (`timings.json` excluded; the byte-compare sets are listed in the activation row); `interpolation_periodic_tricubic` and `interpolation_hessian` unchanged
  and green.
- **H. Recorded: SF-38 A/B re-run with the exit-aware Pollock** (both amplitudes, seeds 4001-4010): per realization and plane the number of
  medium (b) particles that left through `x = 0`; where it is zero, `e2_b.csv` is byte-identical to SF-38's; where it is not, the SF-38
  `C_STAT` lines are recomputed and recorded beside the originals (the SF-38 verdict is not re-decided).

## Regression surface

- The periodic spline module and every periodic consumer (cube, medium (a), closure gate, inlet slab); the tracker device code
  (status enum, landing logic); the Pollock core; `EnsembleRunner` and `lagrangian_ensemble` (medium (a) outputs byte-identical);
  `ctest` wall time.

## Failure and rollback policy

- A, B, C, D or G not met -> defect (BLOCKING), corrective in this increment.
- E: solve not converged -> C0 fallback rule; otherwise stop (owner). Peak above 30000 MiB -> mapped-host head (E1); still above -> R3,
  owner decision on the domain of SF-39, recorded.
- F not met at an amplitude -> the consumer is not defensible on medium (b) at that amplitude (as R1); stop, owner.
- H shows exits -> recorded; the SF-38 decision is not re-opened unless the recomputed `C_STAT` makes all ten statistics equivalent
  (then: owner).

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
