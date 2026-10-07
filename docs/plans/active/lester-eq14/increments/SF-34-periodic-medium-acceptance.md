# SF-34 — Acceptance of the inlet labels in the periodic medium against the closure gate

- State: `pending`
- Goal: `Aceptar las etiquetas de Darcy construidas por transporte (SF-35) en el medio periódico: e_v(h) e invariancia convergentes y mapa de retorno de las etiquetas en la cara de salida concordante con la medición de cierre de SF-30 para sigma^2 en {0.25, 1, 2.25} y ell en {1/8, 1/16}, con (4, 1/16, 256^3) no bloqueante.`
- Depends on: `SF-35`
- Unlocks: `none`
- Branch: `science/lester-sf34-periodic-medium-acceptance`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 3A + Gate 4 (qualified form: label-oracle agreement, not a presupposed D_T)`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Dashboard "Later phases" item 2. Establish, on the production stack and at the project's Gaussian parameters, that
the inlet labels constructed by SF-35 (label transport along backward-traced streamlines; decision
`docs/decisions/2026-10-07-label-transport-constructor.md`) are the labels of the Darcy flow at the resolutions of
SF-30: `e_v(h)` and the invariance residual converge under refinement, and the labels' outlet return map (the outlet
point carrying the same pair of labels as an inlet point) agrees with the first-return map measured by the
label-independent SF-30 closure gate. This is the flow-level check of the decision record's independence rule (item
3 (iv)): SF-35 verifies the constructor on one realization at `ell = 1/4` without tracing; SF-34 checks it against the
flow itself on SF-30's seeds and `ell`. The move to the long domain requires this
(`docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`, delegated decision "Move to the long
domain"). No claim about transverse macrodispersion or `alpha_T` is made.

Context: `docs/decisions/2026-10-07-label-transport-constructor.md`,
`docs/experiments/2026-10-05-sf30-streamline-closure-gate.md`, the SF-35 experiment note.

## Preconditions

- `SF-35` is `done` on the default branch (tracing constructor, `inlet_slab --construct trace`, `.npy` label export,
  per-plane metrics, measured validity envelope at `ell = 1/4`).
- `SF-30` is `done` on the default branch (closure-gate executable, matrix and return maps).

## In scope

- Construction of the labels by `inlet_slab --construct trace` on SF-18 fields with SF-30's seeds and `ell` (`ell` in
  {1/8, 1/16}, realization 3001 for the ladders), `sigma^2` in {0.25, 1, 2.25}, grid ladder 64/128/256 (SF-35
  reports the 256^3 footprint and wall time), with SF-35's production settings (`h_max = h/8`, `tol = 1e-8`) and
  `--save-labels`.
- The labels' outlet return map: for the SF-30 seeds on `x1 = 0`, the point on `x1 = 1` with the same
  `(psi1, psi2)` (2x2 Newton on the outlet plane, SF-28 spline of the labels on the outlet plane), compared with the
  SF-30 first-return point of the same seed.
- `sigma^2 = 2.25` is not in SF-30's matrix: its reference return map is produced here with the unchanged SF-30
  `closure_gate` executable and SF-30's instrument settings (seeds, tolerance ladder, matched control).
- `(sigma^2, ell, N) = (4, 1/16, 256^3)`: attempted, non-blocking; SF-30 found backflow there, so the outcome may be
  `inlet_backflow`, `trace_fail` or a failure, recorded as such.
- Experiment note with the full matrix.

## Out of scope

- Any change to the SF-35 constructor beyond bug fixes found by this increment (a fix is a recorded corrective with
  its own evidence); any change to the SF-30 instrument.
- The long domain, runner wiring, tracker, transport, `D_T` or `alpha_T` claims.

## Files and symbols

- Read-only use: `src/physics/streamfunctions/inlet_slab/*` (SF-33/SF-35), `apps/closure_gate/*` (SF-30),
  `src/numerics/interpolation/` (SF-28), SF-18/SF-19.
- New: a return-map comparison driver (under `apps/` or the experiment artifact `scripts/`) reading the SF-35
  `psi1.npy`/`psi2.npy` export, and `docs/experiments/<date>-sf34-periodic-medium-acceptance.md` with its artifact
  directory.

## Implementation specification

1. Fix the decision rule below in the activation bitácora row before any run; refinements are allowed only before
   the first matrix run and are recorded as such.
2. For each `(sigma^2, ell)` in {0.25, 1, 2.25} x {1/8, 1/16}, realization 3001: construct the labels on 64/128/256
   with `--construct trace` (record every `BACKFLOW`/`GROWTH` line and the round trips); record `e_v(h)`, the
   invariance residual `RMS(v_D . grad psi_i) / (v_D,rms grad psi_i,rms)`, `e_div`, the Gate 3A metrics and the
   equation-(14) residual `r_F` at the labels (diagnostic, recorded).
3. Labels' return map on the SF-30 seeds; distance to the SF-30 first-return points of the same seed and grid
   (`D_ret(N)` = RMS over seeds of the point distance, on the periodic outlet face) and its ratio to SF-30's
   non-uniform displacement `R(N)`.
4. `sigma^2 = 2.25`: run SF-30's `closure_gate` with its settings on the same realization and grids first, and record
   its classification.
5. Attempt `(4, 1/16, 256^3)`, non-blocking.

Decision rule (default; confirmed or refined before running, never after):

- `accept` for a `(sigma^2, ell)` iff every streamline of the construction is `ok` with round trip `<= 1e-8` on every
  grid, `e_v` and the invariance residual decrease on every refinement with observed order `>= 1.8` on the finest
  pair, and `D_ret` decreases with observed order `>= 1.8` on the finest pair with `D_ret(N_f) <= 0.1 R(N_f)`;
- `reject` if any quantity shows a floor (change < 10 % between the two finest grids while above the threshold);
- anything else (including pre-asymptotic orders between) is `unresolved` and returns to the owner.
- `r_F` is recorded with its observed order; it is not an acceptance quantity (decision record item 2).

## Expected numerical effect

No change to any existing output. The experiment produces a classification per `(sigma^2, ell)`.

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
scripts/remote --increment SF-34 sync
scripts/remote --increment SF-34 exec -- "cmake --preset v100-release && cmake --build build/v100-release -j 2>&1 | tail -n 20; echo BUILD_EXIT=${PIPESTATUS[0]}"
scripts/remote --increment SF-34 run sf34-closure225 -- "<SF-30 closure_gate at sigma^2 = 2.25, SF-30 settings>"
scripts/remote --increment SF-34 wait sf34-closure225
scripts/remote --increment SF-34 run sf34-matrix -- "<inlet_slab --construct trace --save-labels + return-map driver over the matrix>"
scripts/remote --increment SF-34 wait sf34-matrix
scripts/remote --increment SF-34 run sf34-paper -- "<(4, 1/16, 256^3) attempt>"
scripts/remote --increment SF-34 wait sf34-paper
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

- Decision rule fixed before running (activation bitácora row).
- `accept` for every `(sigma^2, ell)` with `sigma^2` in {0.25, 1, 2.25} that SF-30 (or step 4) classifies
  `does_not_close` or `closes`; `(4, 1/16)` recorded, non-blocking.
- Experiment note contains the full matrix (label metrics, round trips and backflow statistics, return-map distances,
  the 2.25 closure measurement); human review recorded.

## Regression surface

- None in production code (read-only use of SF-35/SF-30); the long-domain phase depends on this outcome.

## Failure and rollback policy

- `reject` or `unresolved` at any `sigma^2 <= 2.25`: the increment stops and returns to the owner with the
  evidence; no threshold is tuned. The long-domain phase is not specified until this passes.
- `inlet_backflow` or `trace_fail` on a planned `(sigma^2, ell) <= 2.25` point: recorded; the case is excluded only
  with an owner decision.

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

No increment is unlocked directly; acceptance enables the specification of the long-domain phase (dashboard
"Later phases" item 3).

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-06T00:00Z | not started | Specification created by the SF-29 closure PR (decision record `docs/decisions/2026-10-06-eq14-inlet-label-formulation.md`; dashboard "Later phases" item 2). | Depends on SF-33; SF-30 is `done` on `master`. | Activate only when the checker reports it READY. |
| 2026-10-07T16:40Z | not started | Re-specified on the SF-33 closure PR (#51) after the owner decision of 2026-10-07: the labels come from the SF-35 tracing constructor (`--construct trace`, `.npy` export), not from the SF-33 Newton–Krylov solve; `r_F` becomes a recorded diagnostic; the construction's round trips and backflow statistics join the accept condition; the decision rule on `e_v`, invariance and `D_ret` is unchanged (precedent for re-specifying a pending increment in place: SF-27..SF-30 on 2026-10-02). | `docs/decisions/2026-10-07-label-transport-constructor.md`; depends on SF-35. | Activate only when the checker reports it READY. |
