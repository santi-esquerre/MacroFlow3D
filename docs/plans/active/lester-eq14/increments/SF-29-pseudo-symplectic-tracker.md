# SF-29 — Pseudo-symplectic particle tracker (GPU)

- State: `pending`
- Goal: `Implementar el tracker pseudo-simpléctico en GPU con interpolación spline periódica y Newton 2x2 por partícula sobre los invariantes psi1, psi2, más trackers de referencia RK45 y tipo Pollock sobre el mismo campo.`
- Depends on: `SF-28`
- Unlocks: `SF-30`
- Branch: `science/lester-sf29-pseudo-symplectic-tracker`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + Gate 4`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Lester et al. (2023) §5.4: trajectories are computed by holding the two
invariants `psi1(x) = psi1,0`, `psi2(x) = psi2,0` exactly and integrating the
1D advection along the streamline (eqs. 37-38). This increment implements
that method as a new consumer of the accepted streamfunctions (not inside the
legacy PSPTA engine), with one GPU thread per particle and a 2x2 Newton
projection, plus conventional trackers on the SAME interpolated field so that
SF-30 can reproduce the paper's spurious-dispersion results (Figs. 3-4).

## Preconditions

- SF-28 `done` (accepted 256^3 or 128^3 streamfunctions with Gate 3A record).

## In scope

- New module `src/physics/particles/streamline_tracker/`:
  - `PeriodicTricubicSpline`: B-spline prefilter (cuFFT deconvolution or
    3-pass recursive filter) of `u1`, `u2` + affine parts; device evaluation of
    `psi_i` and `grad psi_i` (C^2), double precision.
  - `PseudoSymplecticTracker`: per particle, invariants sampled at injection;
    step = arclength predictor `x* = x + ds * v/|v|`, `v = grad psi1 x grad psi2`,
    then Newton on `r = (psi1(x) - psi1,0, psi2(x) - psi2,0)` with least-norm
    update `delta = J^T (J J^T)^-1 (-r)`, `J = [grad psi1; grad psi2]` (2x3,
    `J J^T` 2x2), until `|r| <= tol_psi` (default `1e-12 * psi_scale`) or
    `max_iter`; `t += ds / |v|` (Simpson over the step); periodic wrapping with
    unwrapped displacement bookkeeping; per-particle failure counters and
    trust-region clamp on `|delta|`. Alternative `x1`-parametrized mode
    (eq. 38) selectable for paper parity where `v1 > 0`.
  - `ReferenceRkTracker`: RK45 with tolerance on `dx/dt = v(x)` from the same
    spline field.
  - `StokesFaceVelocity` + Pollock-like cellwise linear tracker (eq. 32-33:
    face fluxes from line integrals of `psi1 grad psi2` around faces; exactly
    divergence-free per cell).
- Diagnostics: invariant drift `delta psi_i`, `delta_x2, delta_x3` after one
  period vs the pseudo-symplectic trajectory (eqs. 34-35), Newton failure
  histogram, particle status summary; positions/times emitted in the layout
  consumed by `EnsembleRunner`/`AnalysisRunner` (adapter, additive).

## Out of scope

- Local dispersion / Langevin noise (SF-30 optional); modifying `pspta/` or
  Par2; macrodispersion interpretation (SF-30).

## Files and symbols

- New: `src/physics/particles/streamline_tracker/{PeriodicTricubicSpline,PseudoSymplecticTracker,ReferenceRkTracker,StokesFaceVelocity}.{cu,cuh}`,
  `tests/particles/streamline_tracker_gpu_cases.cu`, config surface in `src/io/`,
  `CMakeLists.txt`.
- Reference only (no reuse of code): `src/physics/particles/pspta/PsptaEngine.hpp`
  (float, trilinear, (y,z) 2x2 Newton per x sub-step — the same idea).

## Implementation specification

1. Spline: verify interpolation of the exact pair (SF-26 fixture) reproduces
   `psi`, `grad psi` to `O(h^3)` / `O(h^2)`; periodic wrap exact.
2. Tracker kernels: no allocations per step; particle arrays SoA; RNG-free.
3. Tests: exact pair (`v = e1`): trajectories are straight lines with exact
   travel time; a homogeneous shear-free rotation-type manufactured pair
   (`psi1 = x2 + a sin(2 pi x1)`, `psi2 = x3`) — invariants preserved to
   `tol_psi`, positions vs analytic streamline `O(ds^2)`.
4. On the SF-28 field: 1e5 particles, flux-weighted injection over a plane,
   one domain period; record diagnostics.

## Expected numerical effect

- Pseudo-symplectic: `|delta psi_i| <= tol_psi` for all particles; RK45:
  `sigma^2(delta_x2), sigma^2(delta_x3) ~ sqrt(tol)`; Pollock-like:
  `~ (Delta/Delta_0)^2` (paper Figs. 3c, 4b).

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamline_tracker
scripts/remote run sf29-track -- "./build/v100-release/macroflow3d_pipeline apps/config_tracker_reference_256.yaml"
```

## Acceptance thresholds

- Manufactured cases as above; spline orders observed `>= 2.9` / `>= 1.9`.
- Reference field: Newton failure rate `< 1e-6` of steps; `max |delta psi_i| <= 10 tol_psi`;
  RK45/Pollock-like error scalings observed within a factor 2 of the expected
  exponents over >= 3 tolerance/grid levels.
- Throughput and memory at 1e5-1e6 particles recorded.

## Regression surface

- `AnalysisRunner`/`EnsembleRunner` inputs (adapter additive); Par2 path untouched.

## Failure and rollback policy

- Backflow regions (`v1 < 0`) are handled by the arclength mode; if the
  `x1` mode is used for parity it must report and skip such particles, never
  silently continue.
- Stagnation-adjacent cells (`|c|` below the diagnostic threshold) are
  counted, not regularized.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Periodic tricubic spline implemented and order-verified.
- [ ] Pseudo-symplectic tracker with 2x2 Newton projection implemented and verified on manufactured pairs.
- [ ] RK45 and Stokes-face/Pollock-like references implemented on the same field.
- [ ] Reference-field run recorded (drift, failures, scalings, throughput).
- [ ] Gate 4 interpretation and human review recorded.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-29 complete and selects SF-30.
<!-- completion-checklist:end -->

## Advancement rule

SF-30 may run the transverse-macrodispersion validation with these trackers.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-09-30T00:00Z | created by the owner-directed re-sequencing (decision 2026-09-30) | Replaces the former "SF-29 — V100 benchmark". The legacy PSPTA engine already used a (y,z) 2x2 Newton projection per x sub-step; this increment re-implements the idea as a new double-precision consumer with spline interpolation and arclength parametrization. | `docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md` | Activate only when named by `NEXT`. |
