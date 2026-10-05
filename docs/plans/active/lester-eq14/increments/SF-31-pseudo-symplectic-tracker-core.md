# SF-31 — Pseudo-symplectic tracker core and RK reference

- State: `awaiting_review`
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
- PR: `#47` (https://github.com/santi-esquerre/MacroFlow3D/pull/47), delivery branch `science/lester-sf31-pseudo-symplectic-tracker-core`
- Commit: `9ba13ae` (audited source-bearing head: integration of N0 `153b81a`, N1 `900217d`, C1 `8904930`, N2 `f68d353`, N3 `76902c6`, C2 `f25cc55` + `9661152`, N4 `c740dae` + `f84cd2d` on the delivery head `fcb8214`, base `9b10f1a`; later commits are docs/metadata only)

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
- [x] Implementation matches the scope and contains no unrelated changes.
- [x] Targeted validation passes and its evidence is recorded.
- [x] Required regression tests pass.
- [x] Scientific or engineering findings are appended to the bitácora.
- [ ] Required human review is recorded.
- [x] PR and commit identifiers are recorded.
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
| 2026-10-05T16:15Z | worker commits on `db32ce4`: N0 `153b81a`, N1 `900217d` + corrective C1 `8904930`, N2 `f68d353` (isolated worktrees; not yet integrated) | EXECUTE/AUDIT. N0: `StreamlineTrackerCommon.{cuh,cu}` (label evaluator `psi_i = gbar_i . (xi + w L) + s_i(xi)` taking `(xi, w)`, wrap helper, status codes 10-14, 53-bit hash injection, unwrapped positions) and the `STREAMLINE_TRACKER_SOURCES` list. N1: `PseudoSymplecticTracker.{cuh,cu}` (host/device core templated on the label evaluator: least-norm Newton projection, two-half-step Simpson panel, advance-to-time with banked clock; GPU engine with the runner contract plus `step_arclength`). N2: `ReferenceRkTracker.{cuh,cu}` (host/device Dormand-Prince 5(4) FSAL core on a velocity functor, SF-30 controller, exact landing, no projection; GPU engine). Orchestrator audits (runtime record `audits/`): every diff read in full; each core re-executed with the orchestrator's own analytic evaluators and compared with the numpy prototype, which shares no code with the workers. N1: the 15 ladder rows (pairs A, H, B; four fixed start points; `ds = 1/16 .. 1/256`) agree with the prototype in all 7 printed digits, e.g. helix clock error 4.404127e-03, 1.103318e-03, 2.759724e-04, 6.900205e-05, 1.725107e-05. N2: 21/21 tolerance-ladder rows within 3.2e-5 relative with identical accepted/rejected counts; fixed-step orders on the generic pair 5.62, 5.94, 5.31; GPU landing bitwise; tableau equal to the SF-30 one. Finding F1 (MAJOR, N1, first reported by the worker's own extra probe and reproduced): on the GPU an exactly degenerate pair (`psi2 = psi1`) left 291 of 1024 particles active with clocks up to 4.1e31 (104 Newton failures, 629 degenerate) while the host core flags all 1024. Cause: the contract's test `det > 0` on a Gram determinant computed with cancellation; under FMA contraction the device value is a rounding residual of either sign. The defect was in the orchestrator's contract, not in the transcription. | Decision D-11 (for review against the AGENTS.md rule on `|grad psi1 x grad psi2|` denominators): degeneracy is also declared when `det <= min_cross_sin2 * |grad psi1|^2 |grad psi2|^2`, `min_cross_sin2` configurable with default `16 eps` (rounding error of the determinant is about `10 eps` of that product; `0` restores the raw test). It is a failure threshold on the certified sign of the determinant; `det` is used as computed in every division and nothing is added to it. C1 after the change: 1024/1024 degenerate on the GPU, nothing moved, clocks 0; regular results identical digit for digit; a tilted pair with `sin^2 = 9e-16` (computed determinant 8.9e-16 > 0) is degenerate with the default and active with `0`, on host and device. All four commits compile with nvcc 11.4 on the SF-31 mirror (detached jobs `sf31-n0-build`, `sf31-n2-build`, `sf31-n1-build2`, `sf31-c1-build`; `sf31-n1-build` is excluded: stale `build.ninja` after a sync from another worktree and an exit code masked by a pipe, an orchestrator procedure error). Informational: on splined labels with ten calls of `dt = 0.17` the RK decrease `drift(1e-10)/drift(1e-4)` is 5.4e-3 and 5.6e-3, inside the T4(b) gate 1e-2 by a factor 1.8; the gate is not changed. | N3 (contract tests of the pseudo-symplectic tracker, independent author) running; then N4, integration, V100. |
| 2026-10-05T16:39Z | worker commits (isolated worktrees, not yet integrated): N3 `76902c6` + corrective C2 `f25cc55`, `9661152`; N4 `c740dae` + amendment `f84cd2d` | EXECUTE/AUDIT of the contract tests (authors different from the implementers; gates fixed beforehand). N3: `tests/streamline_tracker/analytic_pairs.hpp` and `pseudo_symplectic_tracker_tests.cu`, ctest `streamline_tracker_pseudo_symplectic`. N4: `reference_rk_tracker_tests.cu`, ctest `streamline_tracker_reference_rk`. Orchestrator audits: gate constants and observable definitions read against the pre-registered readings; both executables re-run locally and on V100 (nvcc 11.4, detached jobs `sf31-n3-pstest`, `sf31-n4-rktest`): N3 225/225 on both with identical printed numbers; N4 69/69 on both. The analytic pairs of the test header were written independently of the orchestrator's and print the prototype's digits (four fixed start points). Three findings, all defects of the orchestrator's test specification, none a deviation of the workers: F2 (reported by the N3 worker) in the determinism and no-allocation cases the `step(dt)` calls came after arclength calls that had already moved the clocks past the target time, so the engine-contract path advanced nothing; F3 the engine's `compute_unwrapped` was never called; F4 the GPU RK ladder advanced with ten calls of `step(0.17)`, which clips every step, so at `tol = 1e-4, 1e-5` the controller was inactive (11 accepted, 0 rejected steps per particle) and the gate `drift(1e-10) <= drift(1e-4)/100` held by a factor 1.6-1.8 only while the tight end moved by up to 24% between the local GPU and V100. | Correctives. C2: `step` blocks first, with checks that every clock reached the target time (mismatch 6.4e-5 and 9.5e-5 against the bound 6.1e-4); determinism compared at two snapshots; no-allocation measured separately for `step` and `step_arclength`; engine `compute_unwrapped` bitwise against `fma(wrap, L, x)`; 237 checks, 0 failed. N4 amendment (disclosed change of a test configuration after candidate results; the gate keeps its form): the ladder uses one controller-governed `step(1.7)` per tolerance, the configuration the reading T4(b) was derived from, and a separate case checks the bitwise landing over ten calls; drift generic pair `1.85e-3 .. 8.29e-8` (factor 220 inside the gate), transverse pair `2.53e-3 .. 3.34e-8` (factor 760), fitted slopes 0.65 and 0.82 (reported, not gated; plateau and non-monotone tight end as recorded for a C^1 spline velocity); 71 checks, 0 failed. Measured on analytic labels (host core): RK label drift strictly decreasing over `tol = 1e-4 .. 1e-10` with slopes 1.14, 1.05, 1.19; fixed-step orders 6.07, 5.78, 5.03; `x1` exact to 1.6e-15 where `c1 = 1`; pseudo-symplectic against RK on the generic pair: orders 1.993, 1.998, 1.999 (host cores) and engine against engine within 0.67 of the derived bound. | Accepted set: N0 `153b81a`, N1 `900217d`, C1 `8904930`, N2 `f68d353`, N3 `76902c6`, C2 `f25cc55` + `9661152`, N4 `c740dae` + `f84cd2d`. Next: single integrator, then V100 evidence and FINAL_AUDIT. |
| 2026-10-05T17:34Z | FINAL_AUDIT positive on `9ba13ae`; State -> `awaiting_review` (human-review increment) | INTEGRATE: one integrator, nine cherry-picks on the delivery head `fcb8214` in the order N0, N1, C1, N2, N3, C2 (two commits), N4 (two commits); one textual conflict (`STREAMLINE_TRACKER_SOURCES` in `CMakeLists.txt`, both lines kept); no integration-only change. FINAL_AUDIT by the orchestrator against the original Goal: every file of the increment is blob-identical to its last accepted worker version; full diff from `9b10f1a` = 11 files (+5268/-3: six module files, three test files, `CMakeLists.txt` +20, this spec); the diff over `src/runtime`, `src/physics/{streamfunctions,flow,stochastic,common}`, `src/physics/particles/{pspta,par2_adapter}`, `src/numerics`, `src/multigrid`, `src/core`, `src/io`, `apps`, `scripts` is empty (`EnsembleRunner.cu` untouched: engines implemented, not wired). V100 (`scripts/remote --increment SF-31`, mirror `~/MacroFlow3D-SF-31`, Tesla V100, nvcc 11.4, preset `v100-release`, detached jobs returning the exit code of the build/test): `sf31-build` reconfigure + full build exit 0, no diagnostic in the new files, `ctest -N` = 20; **`sf31-tracker` (`ctest -R tracker`): 2/2 passed in 9.05 s, 237 + 71 checks, 0 failed**; **`sf31-ctest-full`: 20/20 passed, `Total Test time (real) = 2745.40 sec`**; `sf31-smoke` (`config_pspta_small`) exit 0. Local: `compute-sanitizer --tool memcheck --leak-check full` on both test executables: 0 errors, 0 bytes leaked; the pseudo-symplectic executable prints the same digits locally (RTX 3050, CUDA 13) and on V100. Logs under the runtime record `integration/`. | Acceptance thresholds (V100 numbers): (1) labels conserved to `tol_psi`: PASS, <= 9.997e-9 at `1e-8` and <= 9.98e-13 at `1e-12`, 1024/1024 active on four pairs with wraps in all three directions, reading on the spline labels; (2) position error of order 2 on the curved pairs: PASS, two-level orders 1.9944 .. 2.0000 (host core and GPU, spec pair, helix, transverse pair), helix clock error over the predicted `S kappa^2 ds^2 / (12 |c|)` = 0.99983, 0.99996, 0.99999; (3) exact travel time on the uniform pair: PASS, bitwise on dyadic inputs (clock, unwrapped `x1`, `x2`, `x3`), <= 1.1e-14 otherwise; (4) RK reference error scales with tolerance: PASS under the reading revised before implementation: analytic labels slopes 1.14, 1.05, 1.19 and fixed-step orders 6.07, 5.78, 5.03; splined labels `1.85e-3 -> 8.37e-8` and `2.53e-3 -> 3.34e-8` with slopes 0.65, 0.82 (reported); (5) no allocations per step: PASS, `cudaMemGetInfo` delta 0 for `step` and `step_arclength` (pseudo-symplectic, measured separately) and `step` (RK); (6) bitwise determinism: PASS for both engines. Gate 1 + Gate 2 + Gate 4 satisfied; Gate 4 statement: no transverse growth beyond `tol_psi` for the pseudo-symplectic tracker on the analytic pairs; the RK label drift is numerical; no macrodispersion quantity computed, `alpha_T` neither presupposed nor measured. Open for the human reviewer: D-11 (roundoff-derived degeneracy threshold, against the AGENTS.md denominator rule), D-6 (banked clock, no landing iteration), D-5 (no retry), D-3, D-2, the revised reading of the RK threshold, and the limits: analytic pairs only, no `x1`-non-periodic labels, near-degenerate regime not studied. | Push the delivery branch and open the PR; stop at AWAIT_HUMAN_REVIEW; do not set `done` or check the master entry. |
| 2026-10-05T17:35Z | PR #47 opened at head `55b4636` (docs/metadata above the audited source head `9ba13ae`) | PUBLISH_PR: delivery branch pushed (over HTTPS with the existing `gh` login: SSH port 22 to GitHub was unreachable from the development machine during the whole session; same repository, remote configuration untouched) and PR opened against `master`. The PR body lists Goal, module and test layout, the numerical contract, the DAG and the four corrected findings, the detached V100 jobs, the acceptance table and the decisions for the reviewer (D-11, D-6, D-5, D-3, D-2, the revised reading of the RK threshold). Human-review increment: `done` and the master-checklist entry are deferred to the closure-only metadata commit after explicit approval of the source head `9ba13ae`. | Checklist items 1-4 and 6 checked; items 5 and 7 open. `check-lester-increments.sh` OK (nonterminal=SF-31). | AWAIT_HUMAN_REVIEW. On approval: closure metadata commit on this PR; after merge, `scripts/remote remove-mirror SF-31`. |
