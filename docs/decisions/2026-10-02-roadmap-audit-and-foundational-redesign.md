# Roadmap audit (SF-21..SF-26) and foundational redesign toward the pseudo-symplectic tracker

- Status: accepted (2026-10-02, owner). O1-O3 are decided below. This record changes
  the roadmap and the premise documents; the dashboard and increment specifications
  are updated by the orchestrator's roadmap run.
- Date: 2026-10-02
- Deciders: owner (direction), Claude Code orchestrator (audit and analysis)
- Evidence: `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md`
  (new probes), `docs/experiments/2026-10-01-sf26-pairing-correction-gates.md`,
  `docs/experiments/2026-08-15-sf25-terminal-solver-campaign-report.md`,
  Lester et al. (2023) and Lester et al. (2021) read in full at the cited locations.
- Refutes the target of: `2026-07-13-lester-eq14-streamfunction-solver.md` (the target
  itself); closes the open decisions D1-D7 of `2026-10-01-eta1-residual-floor-gauge-degeneracy.md`.

## Context

The owner asked for an audit of the plan and of the last increments, a fact check of
the recorded findings, and a roadmap rebuilt from the foundations upward, with one
fixed objective: a correct and functional pseudo-symplectic transport implementation.

State on `master = 4670fb5`: SF-00..SF-26 `done`, `NEXT = SF-27` with a guard (its
specification is invalid, D4). No accepted `psi1`, `psi2` exist for any heterogeneous
Gaussian field. The V100 full suite is 17/21 with four entries red on the default
branch and takes 66 700 s.

## Fact check of the recorded findings

| # | Recorded finding | Verdict | Basis |
|---|---|---|---|
| 1 | The correct system is same-index, `L_i = S_i` | Confirmed | Re-derived independently. Provenance located: Lester (2021) eq. (2.16) prints `+B` (the identity gives `-B`); its eq. (2.20) is nevertheless the correct one and implies `L2 = a1`, `L1 = a2`; eqs. (2.22)-(2.23) then print `L1 = a1`, `L2 = a2`, and the 2023 eq. (14) copies that swap. |
| 2 | "The paper's code solved the same-index system; the print is a typo" (2026-09-30 record) | Not verifiable | Two papers carry the same swap. Supporting but not conclusive: the same-index pseudo-time flow is a descent flow of the dissipation (below), the crossed one is not, and the 2021 paper calls its explicit method "robust and stable". |
| 3 | The crossed pairing is the root cause of the SF-21/SF-25 wall | Partly | It is a real defect and it is fixed. Its prespecified prediction (the wall disappears, `e_v` converges) was falsified by SF-26. The `e_v ~ 2.5 %` plateau attributed to it is explained by item 9. |
| 4 | eta = 1 residual floor of the code's scheme: method-independent, epsilon-independent, falls with `h`, grows with amplitude (C3) | Measured; accepted as recorded | Raw outputs preserved. Not re-measured in this audit. |
| 5 | The equation is solvable when discretized accurately (C4) | Confirmed, and qualified | In the new probes the pseudo-spectral system converges to `1e-11` on three smooth fields and is still decreasing at `2e-8` and `2e-6` on two more after 3000 iterations. The solution is not the Darcy flow (item 9). |
| 6 | Most of the floor comes from the linear operator's discretization (C5) | Measured with instrument limits already recorded | Several table entries are iteration plateaus. Not on the critical path any more. |
| 7 | Near-null Jacobian cluster from label recombination (C6) | Confirmed in kind | The continuum symmetry group is every area-preserving relabeling `(psi1, psi2) -> (f, g)` with unit Jacobian, an infinite-dimensional family; the mean-zero projection removes two dimensions of it. |
| 8 | "Gate 3A physics at the floor is clean" | Already retracted by the SF-26 note (N2) | Physics was measured only at `lambda <= 0.025`. |
| 9 | The target `grad psi1 x grad psi2 = v_D` with periodic fluctuations exists for smooth Gaussian fields (assumed by every increment; theory note §2-§4) | **Refuted for the fields tested** | New probes, next section. |

## New foundational finding

Established by the new probes (numbers and limits in the experiment note):

1. A nondegenerate affine + periodic invariant pair forces every Darcy streamline to be
   closed on the torus. For smooth scalar periodic `k` without a special symmetry they
   are not: the return map of the face `x1 = 0` differs from the identity at second
   order in the amplitude, independently of resolution (Gaussian-covariance field,
   `L/ell = 4`: 4.7e-3, 2.0e-2, 8.6e-2 per period at amplitude 0.25, 0.5, 1). A 2-D
   control closes to `1e-12` with the same code.
2. The field of Lester (2021) §3, the only published comparison between the
   streamfunction velocity and the potential velocity, is mirror-symmetric about
   `x1 = 1/4`; the symmetry forces closure (measured `1e-14`). Breaking it with a phase
   gives 8.7e-3 per period.
3. Equation (14) is the pair of components of `curl(c/k) = 0` across `c`. The component
   along `c` (zero helicity, `B . c = 0`) is not imposed. Equivalently (derivation) it
   is the Euler-Lagrange system of the dissipation `E = 1/2 <|c|^2 / k>` over affine +
   periodic pairs, and `1/2 <|c - v_D|^2 / k> = E[c] - E[v_D]`. Its periodic solution is
   the closed-streamline field nearest to the Darcy flow. Measured: at `r_F` down to
   `1e-11`, `e_v` = 3.5e-3 .. 1.4e-2, the same to four digits on two grids,
   proportional to amplitude squared, with `curl(c/k)` along `c` nonzero.
4. The transverse displacement grows without bound over many periods in the cases run,
   so the boundedness of the label fluctuations assumed in Lester (2023) §4 fails there.

What follows for the project:

- No solver, tolerance, discretization, or continuation applied to the periodic
  equation (14) can produce invariants of the Darcy flow on a generic Gaussian field.
  The recorded `e_v ~ 2.5 %` at amplitude 0.5 (SF-21, SF-25 "F-SAT") is this distance;
  the probe field gives 2.0e-2 at that amplitude.
- A pseudo-symplectic tracker fed with the periodic solution of (14) preserves the
  streamlines of a different flow and returns zero transverse dispersion by
  construction. That is what Lester (2023) §5 does: the velocity is defined from the
  interpolated streamfunctions and treated as exact.
- "Purely advective transverse macrodispersion is zero in this regime" is no longer a
  usable oracle for Gate 4 or SF-30. The rule that positive transverse spreading is not
  automatically physical stands; its converse is not established either.

Not established: the same measurements at the paper's parameters (`sigma^2 = 4`,
`ell = 1/16`, 256^3) and on the production stack; the value of the transverse
macrodispersion coefficient in a random, non-periodic medium; whether the production
eta = 1 floor is related to item 3.

## Flaws, from foundational to superficial

1. **Scientific premise.** The existence of the object being computed was taken from
   the paper and written into the theory note as a hard constraint. It was never tested
   independently of the solver. Twenty-five increments built on it.
2. **Problem formulation.** Equation (14) omits the helicity condition, so a small
   `r_F` certifies membership in a family that contains the Darcy flow only when the
   Darcy flow has closed streamlines. The plan's own rule ("a low algebraic residual
   alone never completes a scientific increment") was not enforced by any gate:
   the heterogeneity gates were `r_F <= 1e-6` and `lambda = 1`.
3. **Verification design.** No exact control with a nonzero source existed until SF-26;
   no refinement study of `e_v` at a meaningful amplitude exists to date. The `e_v`
   plateau was seen in SF-21 and in SF-25 and was explained twice without that study.
4. **Method sequencing.** Anderson, Jacobian-vector products, GMRES, Newton-Krylov, the
   terminal solver and its campaign (SF-20..SF-25) were built to cross a wall whose
   cause was upstream. The small CPU probes of SF-26 and of this audit settled in
   minutes what V100 campaigns of days did not.
5. **Pending plan.** SF-27 (cross-validation against solutions that do not exist),
   SF-28 (convergence of `e_v` at 256^3, `sigma^2 = 4`), SF-30 (demonstrate zero
   transverse macrodispersion) cannot meet their Goals as written. SF-29 is sound in
   its tracker mechanics and wrong only in its dependency on SF-28.
6. **Validation loop.** Full suite 18.5 h; four red entries on the default branch;
   science experiments registered as ctest gates.
7. **Documents.** Overview "Estado actual" says the solver is not implemented; the
   dashboard's "Benchmark progression" excludes a tracker that SF-29/SF-30 include;
   Gate 4 and the theory note state the zero-dispersion claim without the limits above.

## Decision (accepted)

Freeze the periodic equation (14) solver stack as it is (no deletion, no further
solver work), and replace pending SF-27..SF-30 by the sequence below. Each phase
unlocks the next; phase 0 and phase 1 do not depend on the owner's choice in O1.
The R-numbering below is the original proposal; "Resulting roadmap" gives the
increment mapping and the order decided by the owner (O1 = B, O2).

**Phase 0 — settle the premise on the production stack (no new solver).**

- R0. Accept this record (done 2026-10-02); qualify theory note §2-§4, Gate 4,
  `ARCHITECTURE.md` §4.3, the overview and the dashboard accordingly; retire D1-D7 with
  the dispositions below; re-tier the four red ctest entries as recorded experiments.
- R1. Periodic tricubic B-spline interpolation of cell-centered periodic fields
  (value and gradient, double precision, order-verified). Shared by R2 and by the
  tracker. Engineering increment.
- R2. Closure gate: return map of the SF-19 Darcy flow, streamlines taken from the
  spline of the potential (direction of `grad phi`; `k` cancels), with a grid ladder on
  one continuum field and a tolerance ladder. Positive controls: a 2-D field and the
  Lester (2021) field (must close). Cases: Gaussian `sigma^2` in {0.25, 1, 4}, including
  `ell = 1/16` at 256^3. Recorded alongside: `e_v(h)` of the existing stack on the
  Lester (2021) field (prediction: converges) and on one Gaussian fixture (prediction:
  amplitude-squared plateau). Human review. The outcome is a measurement, not a
  pass/fail on the physics.

**Phase 1 — tracker core on fields whose invariants are exact (current SF-29, decoupled).**

- R3. Pseudo-symplectic tracker (arclength predictor, 2x2 least-norm Newton on the two
  labels, failure accounting) and the RK and Pollock-like references, verified on
  analytic pairs with known streamlines and on the Lester (2021) field with `psi` from
  the existing stack. This delivers a correct and functional tracker wherever
  invariants exist, and reproduces the paper's tracker comparisons on a field where
  its construction is valid.

**Phase 2 — invariants of the actual Darcy flow (after O1).**

- R4. CPU prototype at 16^3-32^3, two candidates side by side, selected by `e_v(h)`,
  invariance, and agreement with the R2 return map: (i) labels carried from an inlet
  face along the Darcy flow (a linear problem); (ii) the elliptic system on a slab that
  is not periodic in `x1`, labels fixed at the inlet. For constant-head faces the
  unconstrained dissipation minimizer is the Darcy flow, so (14) is exact there
  (derivation, not tested); for the
  periodic Darcy flow the face conditions use the Darcy potential. Any discretization
  of the functional must use face fluxes that are discretely divergence-free (the
  collocated version was tried in the probes and is unsound).
- R5. GPU implementation of the selected construction, reusing the verified operators,
  PCG/MG, SF-18, SF-19 and the SF-11 diagnostics.
- R6. Tracker integration: relabeling at the periodic face (the return map), or a long
  non-periodic domain.

**Phase 3 — science.**

- R7. Transverse spreading with the three trackers under grid and tolerance
  refinement; every observed growth classified as physical, numerical or unresolved;
  comparison protocol with Beaudoin and de Dreuzy (2013) stated. Gate 4 and Gate 5
  rewritten in R0 before this runs.

Disposition of D1-D7 if this record is accepted: D1, D3, D5 moot (they tune a system
whose solution is not the target); D2 retired with this record as the reason; D4
resolved by cancelling the pseudo-time increment (it would remain the natural solver
only under O1-A); D6 resolved in R0; D7 resolved by adopting the Lester (2021) field as
the positive control, with the paper's `1e-16` left unreconciled and non-blocking.

Process rules proposed for the plan: an independent existence or positive-control check
before any solver for a new target; acceptance on a physics metric under refinement,
never on a residual; a CPU prototype before a GPU increment; ctest holds fast contract
tests only, science runs are experiment notes.

## Owner decisions (resolved 2026-10-02)

- **O1 = B — What the tracker preserves.** The tracker preserves the labels of the
  **actual Darcy flow**, non-periodic in `x1`, anchored at the inlet face (`x1 = 0`).
  It does not preserve the paper's periodic surrogate, which gives `D_T = 0` by
  construction and says nothing about Darcy flow. The Lester (2021) field stays as the
  positive control and tracker verification case.
- **O2 — Order.** The probes are accepted as they stand. The re-sequenced roadmap
  starts SF-27 (ctest validation-tier hygiene) and SF-29 (CPU prototype of equation
  (14) with `x1` non-periodic and inlet labels) in parallel. The streamline-closure
  gate on the production stack (SF-30) follows SF-28 (periodic tricubic spline).
- **O3 — Scope of the scientific claim.** The project withdraws "zero purely advective
  transverse macrodispersion" as a regime expectation and as an acceptance oracle now.
  It stays recorded as the paper's claim under the paper's assumptions (bounded label
  fluctuations). No acceptance criterion presupposes the value of `alpha_T`. The rule
  "positive transverse spreading is not automatically physical" stays; its converse is
  not established either.

Delegated decisions, resolved with the plan:

| Theme | Decision | Reason |
|---|---|---|
| Formulation | One formulation (operator, right-hand sides, residual, sources generalized to `x1` non-periodic with labels fixed on the inlet face) serves both the periodic cell and the long domain (Dirichlet in `x`, periodic in `y`, `z`). | Avoids two solvers whose agreement would itself need validation. |
| Characteristics | Out of production. The streamline integrator of the closure gate is the label-independent oracle. | An oracle must not share the construction it checks. |
| Periodic-cell study | `D_T` by re-injection (eq. 36 of the 2023 paper) and the deterministic many-period iteration are read off the return map that the closure gate measures. | The return map is the quantity that decides closure; no separate machinery. |
| Move to the long domain | Requires `e_v(h)` convergent and the labels' return map agreeing with the closure gate up to `sigma^2 = 2.25`; `(4, 1/16, 256^3)` is attempted, non-blocking. | Existence and agreement are checked where affordable before the expensive domain; the paper's parameters are not established. |
| Covariance | Gaussian only. | Smooth fields are the regime where invariants are meaningful (AGENTS.md hard rule). |
| x-marching | The `x`-marching construction is discarded. | Legacy PSPTA construction; superseded. |
| Paper figures | Figures 3-4 of the 2023 paper now; figure 5 later. | Tracker comparisons first; local dispersion needs the long domain. |
| Order of implementation | The CPU prototype runs before any GPU code; runner wiring is deferred past the prototype. | Process rule: a CPU prototype before a GPU increment. |

## Resulting roadmap

| Record item | Increment | Content | Depends on |
|---|---|---|---|
| R0 | this record + the premise documents | Accept, qualify theory note, Gate 3A/4/5, `ARCHITECTURE.md`, `AGENTS.md`, overview, dashboard; D1-D7 dispositions below; re-tier red ctest entries (SF-27) | SF-26 |
| (R0 tiering) | SF-27 | ctest validation-tier hygiene | SF-26 |
| R1 | SF-28 | Periodic tricubic B-spline interpolation | SF-27 |
| R2 | SF-30 | Streamline-closure gate on the production stack | SF-28 |
| R3 | SF-31 | Pseudo-symplectic tracker core + RK reference | SF-28 |
| R3 (references) | SF-32 | Face-flux reference trackers + the paper's scalings | SF-31 |
| R4 | SF-29 | CPU prototype of equation (14), `x1` non-periodic, inlet labels | SF-26 |
| R5-R7 | later phases | Specs created when SF-29 closes (below) | SF-29 |

Later phases (prose; specifications are written when SF-29 closes):

- GPU generalization of `src/physics/streamfunctions/` to `x1` non-periodic.
- Acceptance in the periodic medium: `e_v(h)`, invariance, and the return map against
  SF-30 at `sigma^2 = 0.25, 1, 2.25`.
- Long domain (2048 x 256 x 256, `lambda/h = 10`): `alpha_L` must match RWPT;
  `alpha_T` is reported with grid and tolerance convergence without presupposing its
  value.
- Local dispersion and `D_T(Pe)`.

Process rules adopted: an independent existence or positive-control check before any
solver for a new target; acceptance on a physics metric under refinement, never on a
residual alone; a CPU prototype before a GPU increment; `ctest` holds fast contract
tests only, science runs are experiment notes.

Disposition of D1-D7 of `2026-10-01-eta1-residual-floor-gauge-degeneracy.md`: D1, D3,
D5 moot (they tune a system whose solution is not the target); D2 retired with this
record as the reason; D4 resolved by cancelling the pseudo-time increment; D6 resolved
by SF-27 (heavy entries leave ctest and become documented experiments); D7 resolved by
adopting the Lester (2021) field as the positive control, the paper's `1e-16` left
unreconciled and non-blocking.

## Consequences

- The next increments are small and each one is checkable against an independent
  measurement; the tracker no longer waits for a 256^3 solve; the solver stack
  (11.5 k lines, 26 k lines of tests) stays frozen as verified infrastructure and as
  the producer of invariants on symmetric controls.
- The project stops reproducing Lester (2023) §5 as a statement about Darcy flow. A
  reproduction as a statement about the paper's surrogate remains possible with the
  Lester (2021) field as control.
- No acceptance criterion presupposes `alpha_T`; Gate 4 and Gate 5, the theory note,
  `ARCHITECTURE.md`, `AGENTS.md` and the overview are qualified accordingly.
- Risk: the closure gate at the paper's parameters and on the production stack (SF-30)
  could show something the small probes do not; the long-domain move is gated on it.

## Classification (docs/AGENTS.md)

- Confirmed by derivation: fact-check items 1 and 7; equation (14) as the across-`c`
  part of `curl(c/k) = 0`; the dissipation identity.
- Confirmed in runs (CPU probes, smooth fields, amplitude <= 1, `L/ell = 4`): the new
  finding, items 1-4.
- Accepted scope: R0-R7 as re-sequenced in "Resulting roadmap", the process rules, O1-O3.
- Open question: the paper's parameters and the production stack (SF-30); transverse
  macrodispersion in a random non-periodic medium.
