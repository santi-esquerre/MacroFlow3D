# SF-32 — Face-flux reference trackers and the paper's scalings

- State: `pending`
- Goal: `Reproducir las figuras 3 y 4 de Lester 2023, mostrando la dispersión transversal espuria de los trackers convencionales frente al pseudo-simpléctico en un campo de invariantes exactos.`
- Depends on: `SF-31`
- Unlocks: `none`
- Branch: `science/lester-sf32-reference-trackers-scalings`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 4`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Reproduce Lester 2023 figures 3-4 (spurious transverse spreading of conventional trackers vs the pseudo-symplectic one) on a field whose invariants are exact.

Context: `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md` and `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`.

## Preconditions

- `SF-31` is `done` on the default branch.

## In scope

- `StokesFaceVelocity`: face fluxes from line integrals of `psi1 grad psi2` around faces (Lester 2023 eqs. 32-33), exactly divergence-free per cell.
- A Pollock-type cellwise-linear tracker in the same module.
- Streamfunctions of the Lester (2021) field obtained with the frozen stack; if it does not converge, reduce the amplitude, then fall back to an analytic pair, never touching the solver.
- Diagnostics: `delta psi_i`, `delta x2, delta x3` after one period vs the pseudo-symplectic trajectory (eqs. 34-35).

## Out of scope

- Runner wiring, ensembles, macrodispersion interpretation.

## Files and symbols

- `src/physics/particles/streamline_tracker/*` (extended)
- New tests under `tests/` and an experiment note `docs/experiments/<date>-sf32-spurious-spreading.md`

## Implementation specification

1. Implement `StokesFaceVelocity` and the Pollock-type tracker.
2. Obtain the invariants (frozen stack, reduced amplitude, or analytic pair) and record which route was used.
3. Run grid and tolerance ladders as detached V100 jobs and fit the exponents.
4. Record exponents outside the bands without tuning.

## Expected numerical effect

New capability only; no existing result changes.

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R tracker
scripts/remote sync
scripts/remote run sf32-scalings -- "<scaling ladders command>"
scripts/remote wait sf32-scalings
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

- Pollock variance proportional to `Delta^p` with `p = 2 +- 0.3` over >= 3 grids.
- RK with `p = 0.5 +- 0.15` over >= 4 tolerances.
- Pseudo-symplectic drift <= 10x the Newton tolerance.
- An exponent outside the band is recorded, not tuned.

## Regression surface

- None existing; consumes SF-31 and the frozen stack.

## Failure and rollback policy

- Out-of-band exponents are reported as findings; the increment stays active until the human review decides.

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

No increment is unlocked directly; its outcome feeds the later phases described in the dashboard.

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-02T00:00Z | not started | Specification created by the 2026-10-02 re-sequencing (decision record `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`). | Replaces the cancelled SF-27..SF-30 specifications (git history at `4670fb5`). | Activate only when the checker reports it READY. |
