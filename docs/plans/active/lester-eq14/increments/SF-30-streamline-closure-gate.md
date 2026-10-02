# SF-30 — Streamline-closure gate on the production stack

- State: `pending`
- Goal: `Repetir en la pila de producción (campo SF-18 y solve Darcy SF-19, parámetros del paper) la medición de cierre de líneas de corriente y clasificar si cierran, como oráculo independiente de etiquetas.`
- Depends on: `SF-28`
- Unlocks: `none`
- Branch: `science/lester-sf30-streamline-closure-gate`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 4 (qualified form: classification, not a presupposed `D_T`)`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Repeat the 2026-10-02 closure measurement on the project's own SF-18 field and SF-19 Darcy solve at the paper's parameters. It is the label-independent oracle for every later tracker or solver claim.

Context: `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md` and `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`.

## Preconditions

- `SF-28` is `done` on the default branch.

## In scope

- Additive const accessor to the periodic potential fluctuation in `src/physics/flow/AffinePeriodicFlowSolver.cuh` (`h_tilde_` is private; `G` is already in the report).
- Streamlines of `grad phi` (direction; `k` cancels) on the SF-28 spline of the potential, adaptive integrator (RK45/DOP853 class with tolerance ladder) parametrized by `x1`, first return at `x1 + 1`, counts of backflow (`v1 <= 0`) encounters.
- Controls: a 2-D field (must close to the instrument level); the Lester (2021) field `sin 2pi x1 cos 2pi x2 sin 2pi x3 + (2/5) sin 2pi x1 sin 8pi x3` (must close); its symmetry-broken version `sin(2pi x1 + 0.9)` in the second term (must not close).
- Matrix `sigma^2` in {0.25, 1, 4} x `ell` in {1/8, 1/16}: five realizations at 128^3, three at 256^3 for `(4, 1/16)`, a 64/128/256 ladder on one continuum field.
- Also: `D_T` by re-injection (Lester 2023 eq. 36) and the deterministic many-period iteration from the return map; and `e_v(h)` of the current frozen stack on one Gaussian fixture (prediction: amplitude-squared plateau) and on the Lester 2021 field (prediction: converges).

## Out of scope

- Any solver change; the tracker (SF-31); wiring into the runner.

## Files and symbols

- `src/physics/flow/AffinePeriodicFlowSolver.cuh` (const accessor only)
- A closure-gate executable or test under `tests/` and `apps/`, or a script driving it
- `docs/experiments/<date>-sf30-streamline-closure-gate.md` (new experiment note with the full matrix)

## Implementation specification

1. Add the accessor and a fast test confirming it returns the stored fluctuation.
2. Implement the streamline integrator on the spline with a tolerance ladder and the first-return map.
3. Fix the decision rule below before running the matrix.
4. Run every matrix case as a detached V100 job and record the return map, backflow counts, and the controls.
5. Report `D_T` by re-injection and the many-period iteration, and `e_v(h)` for the two fixtures.

## Expected numerical effect

The accessor is additive; no existing result changes. The experiment produces a classification of the production fields.

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R closure   # accessor and 16^3 controls only
scripts/remote sync
scripts/remote run sf30-<case> -- "<closure command for the case>"
scripts/remote wait sf30-<case>
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

- Decision rule fixed before running: `does not close` if the non-uniform RMS displacement changes < 10 % between the two finest grids and exceeds 10x the controls' error; `closes` if it decays to the instrument level; anything else stops and returns to the owner.
- Controls behave as stated (2-D and Lester 2021 close; the symmetry-broken field does not).
- If it `closes` at `(4, 1/16)`, the later phases are re-planned.
- Experiment note contains the full matrix; human review recorded.

## Regression surface

- `AffinePeriodicFlowSolver` callers (accessor only); the later periodic-medium acceptance, which compares its return map with this one.

## Failure and rollback policy

- An ambiguous classification stops the increment and returns it to the owner; no threshold is tuned.

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
