# SF-30 — Streamline-closure gate on the production stack

- State: `active`
- Goal: `Repetir en la pila de producción (campo SF-18 y solve Darcy SF-19, parámetros del paper) la medición de cierre de líneas de corriente y clasificar si cierran, como oráculo independiente de etiquetas.`
- Depends on: `SF-28`
- Unlocks: `none`
- Branch: `science/lester-sf30-streamline-closure-gate`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 4 (qualified form: classification, not a presupposed D_T)`
- Human review: `required`
- Owner: `Claude Fable 5.1 orchestrator session (2026-10-05)`
- Started: `2026-10-05T11:30Z on master=0dbb65b`
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
| 2026-10-05T11:30Z | activation on `master=0dbb65b` (PR #45 merged; checker OK ready=SF-29 SF-30 SF-31, nonterminal=none); delivery branch `science/lester-sf30-streamline-closure-gate` | UNDERSTAND: SF-28 `done` on the default branch; SF-29 runs concurrently in its own session (own delivery branch, mirror `--increment SF-29`), so activating SF-30 gives two nonterminal increments (SF-31 stays blocked by the limit). Human-review increment (`src/physics/flow` accessor + a scientific classification). Scientific-rigor skill invoked. Code inspected: SF-18 gives the same continuum field on every grid (stateless mode hash; `normalize_variance = true` is grid independent to roundoff for `ell >= 1/16`, `N >= 64`); SF-19 stores `h_tilde_ = sum_d G_d w_d` privately and computes `v = K_f (G + grad h_tilde)`; SF-28 provides the host/device `evaluate_point`. **Doc/code discrepancy found:** the SF-19 header says `h = -G.x + h_tilde` while the code's potential is `phi = -(G.x + h_tilde)`; the direction field of the streamlines is `g = G + grad h_tilde`; the accessor documents the code's convention, the header sentence is corrected (comment only) and the accessor test checks the sign. Instrument fixed before any code: `g = G + grad s_h` with `s_h` the SF-28 spline of `h_tilde` (GPU prefilter, host evaluation); Dormand-Prince 5(4), one absolute tolerance, step <= `h`; statistic `R` = RMS over seeds of the displacement minus its mean (the probes' non-uniform RMS); 1024 fixed seeds on `x1 = 0` (stateless hash, the same in every run); controls: the five analytic probe fields against the 2026-10-02 spectral probe values, and a matched 2-D control per matrix point. DAG: N1 accessor + integrator + analytic field definitions + contract tests -> N2 closure-gate executable + 16^3 controls ctest, N3 `e_v(h)` driver on the frozen stack (parallel with N2) -> N4 analysis script -> one integrator -> orchestrator-owned detached V100 jobs on `--increment SF-30` -> experiment note. | Pre-registered readings for the reviewer, fixed before any run. (D-1) Deviation from "parametrized by `x1`": that ODE is singular at `v1 = 0` and would discard the streamlines whose backflow the spec asks to count; one integrator in arclength form (`dx/ds = g/abs(g)`, same curves) is used for every streamline, and the return point is landed with `x1` as the independent variable (Henon's device) so that it lies on `x1 + 1` to roundoff; statuses `ok / seed_backflow / cap_exceeded / stagnation / step_underflow / landing_failed` and backflow encounters are counted, nothing is clamped. (D-3) Matched 2-D control: the `x3`-average of the same SF-18 realization (its `m3 = 0` plane: 2-D Gaussian covariance, same `ell`, same continuum field on every grid), zero mean, rescaled to variance `sigma^2`, extruded; its `R` is the instrument error `E_ctrl(sigma^2, ell, N)`. Derived predictions: `lester2021` closes to the PCG/roundoff level at every resolution (the mirror `x1 -> 1/2 - x1` maps cell centres onto cell centres); `control2d` is exactly planar but closes in-plane only at `O(h^2)`. (D-4) Matrix = the spec's matrix plus a 64/128/256 ladder on realization 3001 of EVERY case (the spec asks for one ladder), seeds 3001..3005 shared by all cases, `(4, 1/16)` at 256^3 on 3001..3003, tolerances {1e-6, 1e-8, 1e-10} (working 1e-8), many-period iteration 64 periods at 128^3 and 16 at 256^3 for `(4, 1/16)`. (D-5) Decision rule per case on realization 3001, `N_f = 256`, `N_c = 128`, `E_ctrl(N)` = max of the matched control's `R` and `lester2021`'s `R`: validity = PCG converged, integrator error (1e-8 vs 1e-10) `<= 0.01 R(N_f)` or `<= E_ctrl(N_f)`, non-`ok` streamlines `<= 1 %`; `does not close` = `abs(R(N_f) - R(N_c))/R(N_f) < 0.10` AND `R(N_f) > 10 E_ctrl(N_f)` AND every 128^3 realization has `R > 10 E_ctrl(128)`; `closes` = `R(N_f) <= 10 E_ctrl(N_f)` AND `R(N_f) < R(N_c)`; anything else `ambiguous` -> owner. `(4, 1/16)` is classified only if 3001, 3002, 3003 agree. (D-6) `D_T` by re-injection = `var(d_i)/(2 <tau>)` (Lester 2023 eq. 36) from the one-period map, uniform and flux-weighted, reported as a property of the protocol, not as an accepted coefficient (O3); many-period variance from one continuous integration. (D-7) `e_v(h)` of the frozen stack, non-gating: direct solve at `lambda = eta = 1`, `epsilon = 1e-6`, Anderson on, Newton off, `r_F` always reported; Lester (2021) field eps 0.25 at 32/64/128 (prediction: converges, order about 2); Gaussian seed 3001, `ell = L/4`, `sigma^2` = 0.0625 and 0.25 at 32/64/128 (prediction: plateau, ratio about 4, `r_F` at a floor that falls with `h`). (D-8) ctest gains two fast entries (`streamline_closure_integrator`, `streamline_closure_controls16`); one 32^3 `control2d` solve is used to show the in-plane `R` falls >= 2.5x from 16^3. Predictions to confront: probe values reproduced on `lester_brk` (8.69e-3), `generic3d` (5.402e-2), `two_mode` (2.85e-2); `R(sigma^2 = 1)/R(0.25)` about 4; `R` stable within 10 % from 128^3 to 256^3 at `sigma^2 <= 1`. Not predicted: the class at `(4, 1/16)`, the backflow fraction at `sigma^2 = 4`, the many-period growth law. | Commit activation; launch N1. |
