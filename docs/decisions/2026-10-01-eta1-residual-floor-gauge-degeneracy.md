# eta = 1 residual floor of the corrected equation (14): findings and open decisions

- Status: superseded by `2026-10-02-roadmap-audit-and-foundational-redesign.md`
  (D1–D7 dispositions recorded there)
- Date: 2026-10-01 (updated 2026-10-02)
- Deciders: owner (direction), Claude Code orchestrator (analysis)
- Relates to: `2026-09-30-eq14-source-pairing-root-cause.md` (confirmed),
  `2026-08-14-manifold-robust-terminal-solver.md` (its gauge-manifold hypothesis H4
  holds for the corrected system)
- Evidence: `docs/experiments/2026-10-01-sf26-pairing-correction-gates.md` and
  `docs/experiments/artifacts/2026-10-01-sf26-probes/`

## Context

SF-26 corrected the source pairing to the derived same-index form and proved it
with exact-pair contract tests. Re-running the unchanged heterogeneity gates
falsified the prespecified prediction that the SF-21/SF-25 wall would disappear:
every method in the stack stalls exactly at eta = 1, now for a different reason.

## Findings (V100 runs on head `58898bf` and independent numpy probes)

1. The corrected equation is equivariant under label recombination
   (`psi1 -> psi1 + Phi(psi2)` etc.), so solutions form a manifold. Its discrete
   Jacobian carries a near-null cluster at eta = 1 (relative singular values
   1e-4..1e-6, dozens of modes, sharpening with refinement). The crossed system
   had none. Restarted GMRES(10) stagnates on that cluster; a full-recurrence
   GMRES converges.
2. With the code's scheme, random Gaussian fields give a residual floor at
   eta = 1 that is the same for Picard, Anderson, restarted Newton-GMRES, a
   dense Newton with an exact linear solve, and the explicit flow, and that does
   not depend on the regularization epsilon. It falls under refinement (x5.8 for
   the same field 24^3 -> 48^3) and grows with amplitude: 1.9e-2 at 32^3 and
   3.5e-3 at 64^3 for sigma^2 = 1, lambda = 1 (numpy probe).
3. The per-stage tolerance `r_F <= 1e-6` is therefore not reached on the
   current fixtures: both 32^3 smokes exhaust the lambda floor (sigma^2 = 0.25
   at lambda = 0.0125; sigma^2 = 1 at lambda = 0), with every eta = 1 stage at
   `r_F = 1.50e-6`. Disabling Newton does not change the outcome.
4. The equation is solvable when discretized accurately: a pseudo-spectral
   residual reaches 1e-11..1e-12 at lambda*sigma = 0.1 and 6.5e-8 at 0.3; a
   smooth analytic field converges to 6e-16 with the code's scheme.
5. Most of the code scheme's floor comes from the discretization of the linear
   operator: a 2nd-order non-divergence form using `grad(lambda Y)`, with the
   same sources and order, is 2.6x lower at lambda*sigma = 0.1 and 15x lower at
   lambda*sigma = 1 (64^3).
6. The crossed system's explicit flow is unstable in every probe; on the
   corrected system the flow converges on a smooth field, and a fixed-step flow
   on a rough field decays to 1.2e-4 and then destabilizes.
7. No degeneracy appears: `|grad psi1 x grad psi2| >= 0.05` up to
   lambda*sigma = 1.

## Decision

None of the options below is adopted. By owner directive (2026-10-02) SF-26 is
closed with the pairing correction and its contract tests accepted, the
heterogeneity gates recorded as unmet and untuned, and decisions D1-D7 left
open for the owner.

## Interpretation (consistent with the data, not proven)

The floor is a truncation-type inconsistency of the discrete equations that the
gauge-degenerate system cannot absorb. Finding 5 shows that plain truncation of
the linear operator on a lognormal `K` carries most of it; how much is due to
the gauge cluster itself is not separated.

## Not established

- Whether any discretization reaches `r_F <= 1e-6` at sigma^2 = 1, lambda = 1
  (the 4th-order numbers, 6.8e-5 at 64^3 and 3.4e-5 at 96^3, are upper bounds
  that mix floor and iteration stagnation; the pseudo-spectral iteration failed
  at high amplitude with a Laplacian preconditioner).
- The physics at the floor: `e_v`, invariance and `e_div` were measured only at
  lambda <= 0.025. Nothing was measured at full amplitude.
- Whether a divergence-form variant with log-space (geometric-mean) face
  coefficients recovers the gain of finding 5 while keeping `A` symmetric and
  multigrid-compatible.
- Whether an error-controlled pseudo-time integrator converges on rough fields.
- How the paper obtains a "finite difference residual 1e-16" at 256^3,
  sigma^2 = 4 (scheme and residual definition unspecified).
- Robustness across realizations (one seed in the order/spectral probes).

## Open decisions (owner)

These decisions are closed by `2026-10-02-roadmap-audit-and-foundational-redesign.md`
(accepted 2026-10-02). Disposition: D1, D3, D5 moot (they tune a system whose
solution is not the target); D2 retired with that record as the reason; D4 resolved by
cancelling the pseudo-time increment; D6 resolved by SF-27 (heavy entries leave ctest
and become documented experiments); D7 resolved by adopting the Lester (2021) field as
the positive control, the paper's `1e-16` left unreconciled and non-blocking. The text
below is kept as the historical record.

- **D1 — Acceptance at eta = 1.** Keep the locked algebraic tolerance
  (`r_F <= 1e-6` per accepted stage) or accept at the measured floor on Gate 3A
  physics and its convergence under refinement. Touches the dashboard's "Locked
  nonlinear and continuation policy". Blocked on the missing physics
  measurement at full amplitude.
- **D2 — The unmet heterogeneity gates.** The SF-21 prespecification (32^3
  sigma^2 = 1 smoke; 64^3 suite sigma^2 in {0.25, 1, 2.25, 4}), moved verbatim
  SF-21 -> SF-25 (old) -> SF-26, is not met and was not weakened. After SF-26
  closes it is not attached to any pending increment. Decide whether to
  re-impose it in a later increment (after D1/D3), restate it, or retire it
  with a recorded reason.
- **D3 — Discretization of the residual.** Keep the current scheme (floor ~h^2,
  3.5e-3 at 64^3 for sigma^2 = 1); move the linear operator to log-space or
  non-divergence form (15x lower in the probe; conflicts with the locked rules
  "harmonic mean of q at faces" and the choice not to difference `grad(log K)`);
  raise the order; or go pseudo-spectral (needs a preconditioner that controls
  the second-derivative source terms).
- **D4 — SF-27 before activation.** Its specification is invalid as written:
  it requires agreement with implicit-stack solutions at tol 1e-10 on the
  sigma^2 in {0.25, 1} 32^3 fixtures, which do not converge at eta = 1; it
  estimates O(1e3-1e4) steps where rough fields need tau >> 1; and its step
  cap bounds only the linear part, while the fixed-step flow destabilized.
  Decide the revised specification, and whether SF-27 stays next or a
  discretization/acceptance increment goes first.
- **D5 — Newton-Krylov at eta = 1.** The SF-25 E1 wiring enables Newton in the
  smoke fixture and the five benchmark YAMLs; at eta = 1 it takes the stage and
  exhausts GMRES(10)/100. Options: deactivate it at eta = 1, fix the gauge,
  deflate or lengthen the recurrence. It is not the root cause of the gate
  failure.
- **D6 — Red ctest entries.** `streamfunction_heterogeneity_smoke`,
  `streamfunction_anderson_stall`, `streamfunction_newton_difficult` and
  `streamfunction_gauge_recombination_heavy` are red on V100 at `58898bf` and
  will be red on the default branch after merge. Keep them red as recorded
  science gates (SF-25 precedent), re-baseline, or reclassify as evidence
  recorders. `anderson_stall` and `newton_difficult` assume a stall that
  Anderson/Newton then cure; on the corrected system both arms stall.
- **D7 — Reference resolution and paper parity.** `ell/h` (8 in the fixtures,
  16 in the paper), the unreconciled 1e-16, and whether to adopt the
  deterministic field of Lester et al. (2021) section 3 as a paper-parity
  control.

## Options considered (none adopted)

- A. Truncation-aware acceptance for the implicit stack, with the floor tracked
  under refinement.
- B. Pseudo-time (SF-27) with error control that also bounds the nonlinear
  stiffness and budgets sized for tau >> 1. It integrates the same discrete
  residual, so it does not remove the floor.
- C. Gauge-fixing / bordered formulation that removes the manifold.
- D. Reference resolution `ell/h = 16` with the floor measured under refinement.
- E. A more accurate residual discretization (log-space or non-divergence
  linear operator, 4th order, or pseudo-spectral) with the existing PCG/MG on
  `A` as preconditioner.

## Proposed next measurement (not run)

A probe that solves Darcy on the same field and reports `e_v` and invariance
next to `r_F`, adds a divergence-form variant with geometric-mean faces, and
uses fixed iteration budgets instead of a stagnation exit. It would address the
first three items under "Not established" and inform D1 and D3.

## Consequences

- The heterogeneity gates stay unmet, recorded, untuned.
- No locked decision is changed by this record. D1 and D3 put three of them in
  question: the 1e-6 nonlinear tolerance, harmonic-mean face coefficients, and
  not differencing `grad(log K)`.

## Classification (docs/AGENTS.md)

- Confirmed in code/runs: findings 1-7.
- Proposed architecture: options A-E; the proposed next measurement.
- Open question: every item under "Not established"; decisions D1-D7.
