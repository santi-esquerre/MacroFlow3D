# SF-29 — CPU prototype: equation (14) with `x1` non-periodic and inlet labels

- State: `active`
- Goal: `Determinar con un prototipo CPU si la ecuación (14) generalizada a x1 no periódica, con etiquetas fijadas en la cara de entrada, reproduce las etiquetas del flujo de Darcy real con e_v convergente bajo refinamiento y sin floor.`
- Depends on: `SF-26`
- Unlocks: `none`
- Branch: `science/lester-sf29-eq14-inlet-labels-prototype`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 2 + Gate 3A (CPU, no production code)`
- Human review: `required`
- Owner: `Claude Fable 5.1 orchestrator session (2026-10-02, second session, parallel to SF-27)`
- Started: `2026-10-02T23:55Z on master=81cd612`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

The riskiest hypothesis first: that eq. (14) generalized to `x1` non-periodic, with the labels fixed on the inlet face in flow coordinates, yields the labels of the actual Darcy flow with `e_v` converging under refinement and no near-null Jacobian cluster or residual floor. The periodic-fluctuation stack is frozen (SF-02..SF-26); this prototype decides the formulation of the next phase.

Context: `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md` and `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`.

## Preconditions

- `SF-26` is `done` on the default branch.
- The closure probes of `docs/experiments/artifacts/2026-10-02-closure-probes/scripts/closure_probe.py` are available (`FIELDS`: `control2d`, `lester2021`, `lester_brk`, `two_mode`, `generic3d`, `gauss`).

## In scope

- numpy/scipy prototype under `docs/experiments/artifacts/<date>-sf29-inlet-labels/scripts/` (no project binary), grids 16^3 to 48^3.
- Test fields: the analytic `FIELDS` above plus one constant-head-faces case (Dirichlet potential on `x1 = 0, 1`, periodic in `x2, x3`).
- Inlet labels in flow coordinates (triangular construction on the face `v1`: `psi1` = cumulative face flux in `x2` at fixed `x3`, `psi2 = x3`, so that `grad psi1 x grad psi2 . e1 = v1` on the face).
- Candidate (i): non-divergence form `L_i psi_i = S_i` with one-sided stencils at the `x1` faces.
- Candidate (ii): energy fit (minimize the `1/k` dissipation of `grad psi1 x grad psi2`) with face fluxes that are discretely solenoidal; the collocated version was shown unsound in the 2026-10-02 probes and must not be used.
- Oracle: labels obtained by tracing streamlines of the spectral/independent Darcy potential from the inlet face (DOP853, rtol 1e-12).

## Out of scope

- GPU code, `src/` changes, runner wiring, the long-domain study.

## Files and symbols

- `docs/experiments/artifacts/<date>-sf29-inlet-labels/scripts/*.py` (new)
- `docs/experiments/<date>-sf29-inlet-labels.md` (new experiment note)
- `docs/decisions/<date>-eq14-inlet-label-formulation.md` (new decision record)
- New increment files under `docs/plans/active/lester-eq14/increments/` for the next phase, plus dashboard checklist entries, in the closure PR

## Implementation specification

1. Fix the five criteria below in the experiment note before running anything.
2. Implement both candidates on the fields above at amplitudes 0.25, 0.5, 1 over three grids per case.
3. Compute the oracle labels and compare; compute the dense Jacobian at 12^3-16^3 and report its relative singular-value spectrum.
4. Write the decision record choosing formulation and nonlinear method, or recording that none qualifies.
5. Write the specifications of the next phase as new increment files in the same PR: GPU generalization of `src/physics/streamfunctions/` to `x1` non-periodic reusing PCG/MG, SF-18, SF-19 and `Diagnostics.cuh`; acceptance in the periodic medium with `e_v(h)`, invariance, and return map vs SF-30 at `sigma^2` in {0.25, 1, 2.25}, with `(4, 1/16, 256^3)` non-blocking. The closure PR also writes the locked decisions of the chosen formulation.

## Expected numerical effect

No change to any project output. Produces evidence and the formulation decision only.

## Validation commands

```bash
# CPU only; scripts live under the artifact directory
python docs/experiments/artifacts/<date>-sf29-inlet-labels/scripts/run_all.py
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

- Criteria fixed before running, at amplitudes 0.25, 0.5, 1: (1) `e_v(h)` decreases with observed order >= 1.8 over three grids.
- (2) Labels agree with the oracle to the same order.
- (3) Relative nonlinear residual <= 1e-10 with no floor.
- (4) Dense Jacobian at 12^3-16^3 shows no near-null cluster: no gap-separated group below 1e-3 relative beyond the gauge modes explicitly fixed by the inlet data (report the relative singular-value spectrum).
- (5) Criteria (1)-(4) also hold in the constant-head case.

## Regression surface

- None in production code; the planned GPU phase depends on the decision record.

## Failure and rollback policy

- If no candidate meets (1)-(5), the increment stops `blocked` with the evidence and returns to the owner; no criterion is relaxed.

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

No increment is unlocked directly; the later GPU-phase specifications are created by this increment's closure PR.

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-02T00:00Z | not started | Specification created by the 2026-10-02 re-sequencing (decision record `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`). | Replaces the cancelled SF-27..SF-30 specifications (git history at `4670fb5`). | Activate only when the checker reports it READY. |
| 2026-10-02T23:55Z | activation on `master=81cd612` (checker OK ready=SF-29, nonterminal=SF-27 -> now SF-27 SF-29); delivery branch `science/lester-sf29-eq14-inlet-labels-prototype` | UNDERSTAND (scientific-rigor skill invoked). Object: Darcy labels transported from the inlet face (exist for every case with `v1 > 0`; oracle = backward streamline tracing of the spectral Darcy potential, label-independent). Analysis of eq. (14) on the slab, linearized about `k = 1`: principal symbol determinant `xi1^2 |xi|^2` (not elliptic); per transverse mode four `x1`-modes (relabeling gauge, an `x1`-linear shear whose flow is helical and satisfies (14) exactly, two potential modes); inlet Dirichlet fixes two. Pre-registered predictions: P1 the spec-literal candidate (i) (no outlet condition, one-sided (14) rows at the outlet) has a null cluster of ~2(N_perp^2-1) modes and fails criterion (4); P2 constant-head case: (14) + inlet Dirichlet + outlet Neumann `d1 psi_i = 0` is well-posed and its solution is the Darcy labels; P3 periodic-flow case: flow periodicity removes the potential modes but not the shear; the outlet condition `grad psi1 x grad psi2 x e1 = v_D x e1 |_inlet` (uses only inlet-face data + periodicity; reduces to P2's Neumann when `v_perp = 0`) closes it — run as candidate (i-1), spec-literal as control (i-0); P5/P6 candidate (ii) = dissipation energy with Whitney/mimetic discretely solenoidal face fluxes (collocated version excluded), free outlet in the constant-head case, outlet-flux constraint `c1(1,.) = v1(0,.)` in the periodic case (Kelvin principle); its Euler-Lagrange equations are (14) + natural outlet condition. Deviations recorded for the reviewer: D-1 the triangular inlet construction is normalized (`psi2 = int_0^x3 Q`, `psi1 = int_0^x2 v1 / Q(x3)`) so both labels are affine + periodic in (x2, x3) (the spec's literal `psi2 = x3` gives an `x3`-dependent jump of `psi1`); D-2 outlet condition for (i) as above; D-3 outlet flux constraint for (ii) in the periodic case; D-4 constant-head case realized by the mirror trick `k(g(x1), x2, x3)`, `g = (1 - cos pi x1)/2` on a length-2 periodic cell (odd potential -> constant head, `v_perp = 0` on the faces), reusing the closure-probe spectral solver. DAG: N1 reference + inlet labels + oracle + note skeleton (criteria fixed, status planned); N2 candidate (i) || N3 candidate (ii); N4 full sweep as a detached CPU job on the V100 host (`scripts/remote --increment SF-29 run sf29-run-all`, 7 fields x eps {0.25, 0.5, 1} x N {16, 32, 48} x {i-0, i-1, ii}, dense spectra at 12^3/16^3); N5 note + decision record + next-phase specs (if a formulation qualifies). | Criteria (1)-(5) of this spec, unchanged; predictions above fixed before any candidate run. Human-review increment. | Activation commit; launch N1 worker. |
