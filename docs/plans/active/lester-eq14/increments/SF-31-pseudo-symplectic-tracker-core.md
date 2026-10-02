# SF-31 — Pseudo-symplectic tracker core and RK reference

- State: `pending`
- Goal: `Implementar el tracker pseudo-simpléctico (predictor de longitud de arco y Newton 2x2 de mínima norma sobre las etiquetas) y una referencia RK adaptativa, verificados en pares analíticos independientes del solver.`
- Depends on: `SF-28`
- Unlocks: `SF-32`
- Branch: `science/lester-sf31-pseudo-symplectic-tracker-core`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 4`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
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
