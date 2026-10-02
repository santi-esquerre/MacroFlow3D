# Roadmap audit (SF-21..SF-26) and foundational redesign toward the pseudo-symplectic tracker

- Status: proposed — owner decision required (O1-O3 below). Nothing in the dashboard,
  the increment specifications, the gates, or the code is changed by this record.
- Date: 2026-10-02
- Deciders: owner (direction), Claude Code orchestrator (audit and analysis)
- Evidence: `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md`
  (new probes), `docs/experiments/2026-10-01-sf26-pairing-correction-gates.md`,
  `docs/experiments/2026-08-15-sf25-terminal-solver-campaign-report.md`,
  Lester et al. (2023) and Lester et al. (2021) read in full at the cited locations.
- Puts in question: `2026-07-13-lester-eq14-streamfunction-solver.md` (the target
  itself); the open decisions D1-D7 of `2026-10-01-eta1-residual-floor-gauge-degeneracy.md`.

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

## Decision (proposed)

Freeze the periodic equation (14) solver stack as it is (no deletion, no further
solver work), and replace pending SF-27..SF-30 by the sequence below. Each phase
unlocks the next; phase 0 and phase 1 do not depend on the owner's choice in O1.

**Phase 0 — settle the premise on the production stack (no new solver).**

- R0. Accept or reject this record; if accepted, qualify theory note §2-§4, Gate 4,
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

## Owner decisions

- **O1 — What the tracker preserves.** (A) the paper's construction: the periodic
  solution of (14), a closed-streamline surrogate of the Darcy flow, zero transverse
  dispersion by construction, `e_v` reported as its distance to Darcy; or (B) labels of
  the actual Darcy flow, anchored at an inlet and not periodic in `x1`. Recommendation:
  B, keeping the Lester (2021) field from A as the tracker's verification case.
- **O2 — Order.** Run R2 before anything else (recommended: it repeats the refutation
  on the project's own stack and at the paper's parameters), or accept the probes and
  start R1/R3 in parallel with it.
- **O3 — Scope of the scientific claim.** Whether the project's stated regime
  expectation (zero transverse macrodispersion) is withdrawn now or after R2.

## Consequences

- If accepted: the next increments are small and each one is checkable against an
  independent measurement; the tracker no longer waits for a 256^3 solve; the solver
  stack (11.5 k lines, 26 k lines of tests) stays as verified infrastructure and as the
  producer of invariants on symmetric controls.
- The project stops reproducing Lester (2023) §5 as a statement about Darcy flow. A
  reproduction as a statement about the paper's surrogate remains possible under O1-A.
- Risk: R2 at the paper's parameters could show something the small probes do not.
  That is why R2 precedes Phase 2 and why this record stays `proposed` until the owner
  decides.

## Classification (docs/AGENTS.md)

- Confirmed by derivation: fact-check items 1 and 7; equation (14) as the across-`c`
  part of `curl(c/k) = 0`; the dissipation identity.
- Confirmed in runs (CPU probes, smooth fields, amplitude <= 1, `L/ell = 4`): the new
  finding, items 1-4.
- Proposed architecture: R0-R7, the process rules.
- Open question: the paper's parameters and the production stack (R2); transverse
  macrodispersion in a random non-periodic medium; O1-O3.
