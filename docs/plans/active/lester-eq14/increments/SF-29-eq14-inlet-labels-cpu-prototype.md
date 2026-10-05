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
| 2026-10-03T03:05Z | N1 accepted: `356146a` (reference, inlet labels, oracle, note skeleton) fast-forwarded onto the delivery branch | Orchestrator audit of N1 (scope exact; selftest re-run: face Jacobian 1.2e-15, round trip 2.5e-13, constant-head faces `max|v_perp|` 2e-16; oracle self-consistency orders 1.95/1.98 at `eps = 0.25` for `gauss` and `gauss_ch`; return maps equal to the 2026-10-02 closure note to 4 digits; `N_phi(field, eps)` table with differences <= 4.9e-9, two rows need 64). Refinement of P1 before any candidate run (orchestrator instrument, linearized (14) at `k = 1`, dense SVD at N = 8..16): the spec-literal outlet has exactly `2N` null modes (shear modes with `xi2 = 0` or `xi3 = 0`; the other continuum shear/potential modes are null only to O(h^2) discretely), the outlet-condition variant has none and its smallest relative singular value scales as `h^2` (2.0e-4 at 16^3, below the spec's 1e-3 without a gap): criterion (4) is read on the gap-separated cluster, as its text says. Finding of N1 (consequential): the exact oracle labels differentiated with the same second-order FD are pre-asymptotic at `eps = 1` on 16/32/48 (`gauss` mid-slab orders 0.62..1.76 up to 128^3, `gauss_ch` 0.23..1.62; `max|grad_h psi|` up to 13) and marginal at `eps = 0.5` (~1.78 on 16->32), so no candidate can meet criterion (1) literally there. | D-5 (pre-registered before any candidate run; criteria unchanged): the sweep reports the oracle-FD ceiling `e_v^or(h)` next to every candidate; (1)/(2) are evaluated as written; a cell that misses 1.8 is classed `ceiling-limited` (not a pass) iff `e_v <= 1.5 e_v^or` on every grid and the orders differ by <= 0.15; `e_v > 1.5 e_v^or` or a floor is FAIL. D-6: `N = 64` (96 if < 2 h per case) added for `gauss`, `gauss_ch` at `eps = 1` with (i-1) and (ii) as evidence beyond the spec's 48^3. The reviewer adjudicates. Minor findings: `psi2^0 = Q0 x3 + periodic` with `|Q0 - 1| <= 2e-10` (negligible); the `N_phi` table and the 128^3 mid-slab probe ran locally (26 + 31 min, single process) at the orchestrator's request. | Launch N2 (candidate (i), variants i-0/i-1) and N3 (candidate (ii)) in parallel from this head. |
| 2026-10-03T06:40Z | N2 accepted: `f8c7e31` (candidate (i), variants i0/i1); N3 accepted as implementation: `fd2d6bd` (candidate (ii), Whitney/avg energy); corrective C-ii commissioned | N2 (orchestrator audit + own probe): i1 converges to `r_F` 1e-15 (no floor); consistency at the oracle labels order 1.88/1.96; Jacobian FD-verified; dense spectra at 12^3: `k = 1` i0 has `2N` exact nulls and i1 none (§8 confirmed); at `eps = 0.25` i0's nulls are lifted to ~1e-5 by `grad ln k` (no roundoff cluster) but Newton never converges (steps amplified, line search collapses at every `eps >= 1/16`): the spec-literal outlet has no solution, P1 confirmed in substance. i1 at `eps = 0.25`: `e_v` second order and 1.14-1.65x the oracle-FD ceiling, but `e_psi` only ~0.9-1.3 on 16->32. Orchestrator probe on `control2d` (2D: candidate (i) reduces to a linear elliptic BVP): `e_psi` orders 1.09, 1.47, 1.67, 1.79 over 16/24/32/48/64 (fit `a h^2 - |b| h^3`, `b/a ~ -9`), `e_v` order 2.0 on every pair at 1.14-1.17x the ceiling: asymptotically second order, pre-asymptotic labels from the boundary treatment; no implementation defect. N3: the Whitney/edge-averaged energy has a checkerboard (hourglass) kernel of exactly `2N^2` label modes that change no face flux (288 at 12^3): Newton plateaus (`g_rel` 1e-10..1e-3), `e_psi` 0.16-0.41 non-convergent, Hessian indefinite — a defect of the discretization the orchestrator specified, not of the energy; the flux converges to the TPFA Kelvin flux `c_K` (`E_h(u) - E_K = 1/2|c - c_K|^2_W` verified to 8e-16). The orchestrator's D-3 constraint also leaves the mean transverse flux free (floor `e_v` ~5e-3 for `gauss`; two added scalar constraints give order 1.96 in a scratch check). | Decisions: (a) corrective C-ii = Q1 (trilinear) in-cell energy with full Gauss quadrature — pointwise divergence-free, normal-continuous, face averages identical to `c_f` — plus the two mean-transverse-flux constraints (D-3 completed); the Whitney form stays as the kernel control. (b) Sweep plan: i0 capped at 16^3; i1 on 16/32/48 everywhere and 64 for `gauss`, `gauss_ch`, `control2d` (D-6 extended to all amplitudes for those fields); (ii) in the corrected form. (c) Criterion (2) at 16/32/48 is expected to read ~1.3-1.7 for the labels at every amplitude; the note shows the ladders and the D-5 ceiling ratios; no criterion is changed. | Audit C-ii; then N4 sweep as a detached CPU job on the SF-29 mirror. |
| 2026-10-05T14:30Z | C-ii accepted: `83737b5` (Q1 in-cell energy + mean-transverse-flux rows); N4 raw data accepted as factual record: `fa34232` (driver) + `0132180` (505 raw files, summary, timeout digest); owner directive on the corrective scope | C-ii: the Q1 energy removes the hourglass kernel (`k = 1`, 12^3: 0 modes below 1e-10 vs 288 for the Whitney control; positive curvature along both checkerboard directions), the two mean-transverse-flux rows remove the D-3 floor (TPFA `e_v` orders 1.98/1.99), Newton reaches 1e-15, spectra show no cluster. Sweep `sf29-run-all` (detached CPU job on the V100 host, mirror `~/MacroFlow3D-SF-29`, 2026-10-03T06:19Z-20:59Z, 52 822 s, exit 0, 22 workers x 3 threads, GPU lock 0 held): 305 cells done, 33 timeout (i1 at 48^3 for `gauss`:{0.5,1}, `gauss_ch`:{0.25,0.5,1} and five analytic cases at `eps` 1 or 0.5, every 64^3 `gauss`/`gauss_ch` cell; (ii) at 32^3 in 17 cells). 85-95% of the i1 time is GMRES (median 900 iterations per Newton step, caps and stagnation stops; the `k = 1` mode preconditioner degrades with amplitude); `i1-gauss-0.5-N32` ended STAGNATED at `r_F = 8.6e-3` after 5 bisections (linear vs nonlinear failure not separated). What finished: i1 criterion (4) PASS 21/21; i0 criterion (3) FAIL 17/21; i1 `control2d:0.5` `e_v` order 2.0 on every pair at 1.14-1.17x the ceiling, `e_psi` orders 1.09/1.47/1.67/1.79 (16..64); i1 `gauss:0.25` `e_v` orders 1.27/1.29/1.40 (16/24/32/48) with `e_v`/ceiling 1.04 -> 2.09 and `e_psi` 0.217 -> 0.068 (orders 0.81/1.05/1.28): FAIL under D-5 (ratio > 1.5), rising orders; (ii)-Q1 `control2d:0.5` orders 1.99/1.98, `gauss`/`gauss_ch` at 0.25 `e_v` 1.35-1.37 and `e_psi` 1.23-1.25 on 16 -> 24. Orchestrator audit finding: `e_psi ~ 0.46` (flat) for `control2d` at `eps` 0.25 and 1.0 (i1 and (ii)) is a metric artifact — `psi2`'s true periodic part is zero there and the stored one is `(Q0 - 1) x3 ~ 1e-13`, above the 1e-12 relative fallback of `metrics.fd_metrics` (measured denominators 2.2e-2 vs 2.4e-13); 3D fields and `control2d:0.5` unaffected. The first N4 worker stalled after the job had finished and was stopped by the owner; a bounded retrieval worker pulled the results. | Owner directive (2026-10-05, explicit choice among three options presented by the orchestrator): corrective cycle PLUS a 4th-order variant of candidate (i-1) — a scope extension beyond the spec's candidates, recorded as D-7. Corrective DAG: C-i4 (metric fix with per-label `e_psi`; saved solutions; `i1` with 4th-order stencils, 4th-order outlet row, 4th-order reconstruction for `e_v` and its ceiling; direct linear solves up to 32^3) -> N4c reduced sweep (`gauss`, `gauss_ch`, `control2d` x 3 amplitudes x 16/24/32, 48 where affordable; 2nd-order i1 reruns with direct solves; (ii) 32^3 reruns) launched detached and monitored by the orchestrator -> N5. Criteria (1)-(5) unchanged; ladders 16/24/32 are within the spec's 16^3-48^3 range. | Launch C-i4. |
