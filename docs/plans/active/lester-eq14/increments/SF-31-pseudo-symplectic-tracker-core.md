# SF-31 — Pseudo-symplectic tracker core and RK reference

- State: `active`
- Goal: `Implementar el tracker pseudo-simpléctico (predictor de longitud de arco y Newton 2x2 de mínima norma sobre las etiquetas) y una referencia RK adaptativa, verificados en pares analíticos independientes del solver.`
- Depends on: `SF-28`
- Unlocks: `SF-32`
- Branch: `science/lester-sf31-pseudo-symplectic-tracker-core`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 4`
- Human review: `required`
- Owner: `Claude Fable 5.1 orchestrator session (2026-10-05)`
- Started: `2026-10-05T15:20Z on master=9b10f1a`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

A correct tracker wherever exact invariants exist, verified on analytic pairs and independent of the solver.

Context: `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md` and `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`.

## Preconditions

- `SF-28` is `done` on the default branch.

## In scope

- New module `src/physics/particles/streamline_tracker/` (`PseudoSymplecticTracker.{cu,cuh}`, `ReferenceRkTracker.{cu,cuh}`).
- Per particle: labels sampled at injection from the SF-28 spline; arclength predictor `x* = x + ds v/|v|`, `v = grad psi1 x grad psi2`; 2x2 least-norm Newton on `r = (psi1(x) - psi1,0, psi2(x) - psi2,0)` with `delta = J^T (J J^T)^-1 (-r)`, `J = [grad psi1; grad psi2]`, until `|r| <= tol_psi` or `max_iter`, trust-region clamp; `t += ds/|v|` (Simpson); periodic wrapping with unwrapped bookkeeping; per-particle failure counters.
- RK adaptive reference on `dx/dt = v(x)` from the same spline.
- The engine contract of `src/runtime/ensemble/EnsembleRunner.cu` (about line 765: `bind_particles, inject_box, ensure_tracking, prepare, step, particles(), compute_unwrapped, synchronize`) implemented but NOT wired into the runner.

## Out of scope

- Runner wiring, face-flux trackers (SF-32), any solver change.
- Anything under `pspta/` or Par2.

## Files and symbols

- `src/physics/particles/streamline_tracker/*` (new)
- `CMakeLists.txt` and new fast tests under `tests/`
- `src/runtime/ensemble/EnsembleRunner.cu` is read only

## Implementation specification

1. Implement the tracker with the algorithm above, per-particle state only, preallocated workspace.
2. Implement the RK reference with tolerance control.
3. Test on the pair `psi1 = x2 + a sin 2pi x1, psi2 = x3` and a second pair with both labels curved, plus the uniform pair `psi1 = x2, psi2 = x3`.
4. Confirm determinism and allocation freedom.

## Expected numerical effect

New capability only; no existing result changes.

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R tracker
scripts/remote sync
scripts/remote run sf31-tracker -- "ctest --test-dir build/v100-release --output-on-failure -R tracker"
scripts/remote wait sf31-tracker
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

- Labels conserved to `tol_psi` for every particle on the analytic pairs.
- Position error of order 2 in the step on the curved pairs.
- Exact travel time on the uniform pair (`v = e1`).
- RK reference error scales with tolerance.
- No allocations per step (`cudaMemGetInfo` unchanged).
- Determinism bitwise across two runs.

## Regression surface

- None existing; the new module is consumed by SF-32 and later phases.

## Failure and rollback policy

- Unmet orders or label conservation leave the increment active.

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

`SF-32` becomes eligible after this increment is merged and marked `done` on the default branch.

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-02T00:00Z | not started | Specification created by the 2026-10-02 re-sequencing (decision record `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`). | Replaces the cancelled SF-27..SF-30 specifications (git history at `4670fb5`). | Activate only when the checker reports it READY. |
| 2026-10-05T15:20Z | activation on `master=9b10f1a` (PR #46 merged; checker OK ready=SF-29 SF-31, nonterminal=none); delivery branch `science/lester-sf31-pseudo-symplectic-tracker-core` | UNDERSTAND: SF-28 `done` on the default branch; SF-29 runs concurrently in its own session (own delivery branch, mirror `--increment SF-29`), so activating SF-31 gives two nonterminal increments. Human-review increment (code that consumes `psi1`/`psi2` as invariants; autonomy policy). Scientific-rigor skill invoked (numerical-method and GPU-implementation contracts; no physical claim, `alpha_T` neither measured nor presupposed). Source check (Lester 2023 sections 5.3-5.4, eqs. 37-38): the paper's method inverts `(x2, x3) = X(psi1, psi2, x1)` and integrates `t = int dx1 / v1`; the spec's method (decision record R3) is a projection method (arclength predictor + least-norm Newton onto the label curve) and is a project design, not the paper's algorithm. Numerical contract fixed before any implementation (orchestration record `understanding.md` section 3): labels `psi_i = gbar_i . x_u + s_i(x)` with `s_i` the SF-28 periodic spline and `x_u` the unwrapped position; one arclength panel `ds` = two projected half-steps, clock by Simpson of `1/|c|` on three on-curve points, `c = grad psi1 x grad psi2`; Newton residual in max-norm against an absolute `tol_psi`, `det(J J^T) = |c|^2` never regularized; an unconverged position is never committed (reject, count, freeze; no step-halving retry); `step(dt)` uses per-particle clocks with one final partial panel and a banked second-order clock mismatch; RK reference = Dormand-Prince 5(4) FSAL with the SF-30 controller, exact landing on the target time, no projection. Derived expectation: a panel advances `ds (1 - kappa^2 ds^2 / 12)` of true arc, hence global order 2 with a known constant. DAG: N0 shared label view and particle bookkeeping -> N1 pseudo-symplectic tracker and N2 RK reference (parallel) -> N3 and N4 contract tests -> one integrator; orchestrator-owned detached V100 jobs on `--increment SF-31`. | Pre-registered readings for the reviewer: (T1) label conservation is gated on the spline labels the tracker is given (max-norm, `tol_psi` plus a `1e-14` roundoff allowance), analytic labels reported with the measured interpolation error; (T2) order gate = every consecutive two-level estimate in `[1.8, 2.2]` over >= 4 halvings of `ds`, on the analytic position error and on the interpolation-floor-free clock error `t_p - (x1_u - x1_0)` (pairs with `c1 = 1`); (T3) uniform pair bitwise on dyadic inputs, `1e-13` relative otherwise; (T4) RK label drift and `x1` error strictly decreasing over `tol = 1e-4 .. 1e-10` with log-log slope in `[0.7, 1.3]`; (T5) `cudaMemGetInfo` unchanged over >= 100 steps; (T6) `memcmp` identity of positions, wraps, status, clocks and counters across two fresh runs. Test pairs: uniform; `psi1 = x2 + a sin 2 pi x1, psi2 = x3` (spec); helix `psi2 = x3 + a cos 2 pi x1` (constant curvature: predicted clock error `S kappa^2 ds^2 / (12 |c|)`); `psi2 = x3 + b sin 2 pi x2` (transverse dependence); a generic 3-D pair for the cross-check against RK. Decisions D-1..D-10 of the orchestration record go to the PR for review. | Commit activation; start the base build on the SF-31 mirror; launch N0. |
| 2026-10-05T15:28Z | `ff3ec81` (activation); no worker launched yet | PLAN, orchestrator prototype before any candidate result (numpy, CPU; analytic labels and a numpy periodic cubic B-spline in the SF-28 conventions; runtime record `audits/tools/prototype_*.py`). Confirmed: panel scheme order 1.997-2.000 on the three curved pairs (analytic and splined labels, `ds = 1/16 .. 1/256`); helix clock error / `S kappa^2 ds^2 / (12 |c|)` = 0.9972, 0.9993, 0.9998, 1.0000, 1.0000; uniform pair bitwise on dyadic inputs; engine-mode clock mismatch 0.03-0.2 of the bound `2.5 ds_max^2`, not growing over 200 calls; pseudo-symplectic against RK on the generic 3-D pair: order 2.00. Two findings on the RK reading T4 of the activation row: (1) for pairs with `c1 = 1` any RK integrates `x1` exactly, so the `x1` error is identically zero and cannot be a scaling observable; with fluctuations of one coordinate RK is a quadrature in `t` and is superconvergent over whole periods (helix at roundoff for any fixed step at `T = 2`); (2) on SPLINED labels the velocity `grad psi1 x grad psi2` is only C^1 and the DP5(4) label drift does not follow the tolerance: generic pair at 32^3, `tol = 1e-4 .. 1e-10`: `2.96e-4, 2.54e-5, 4.01e-6, 2.92e-6, 2.05e-6, 2.25e-7, 3.90e-8` (fitted slope 0.57, a plateau over two decades); 64^3: slope 0.55 with one non-monotone pair; fixed-step order about 2-2.5 instead of 5. On analytic labels at `T = 1.7` the same integrator gives slopes 1.16, 1.05, 1.20 and fixed-step order 5.5-6.1. | T4 reading REVISED before any worker result (the original stays in the row above): (a) integrator core on analytic labels (host instantiation of the same core the kernels use), `T = 1.7`, generic, helix and transverse pairs: label drift strictly decreasing over `tol = 1e-4 .. 1e-10` with log-log slope in `[0.8, 1.5]`, fixed-step order on the generic pair `>= 4.5`, `x1` exact to `1e-13` where `c1 = 1`; (b) GPU engine on splined labels: ladder and step counts reported, gate only `drift(1e-10) <= drift(1e-4) / 100`, slope not gated. Design consequence: both integrators are `__host__ __device__` cores templated on a label evaluator, verified on analytic labels on the host and run by the GPU engines on splines. Hand-over for SF-32: its RK exponent is measured on a C^1 spline velocity, where the smoothness of the interpolant, not the tolerance, limits the error. T1-T3, T5, T6 unchanged. | Commit; launch N0. |
