# eta = 1 residual floor of the corrected equation (14): gauge degeneracy of the discrete system

- Status: proposed (orchestrator finding, SF-26; awaiting owner decision)
- Date: 2026-10-01
- Deciders: owner (direction), Claude Code orchestrator (analysis)
- Relates to: `2026-09-30-eq14-source-pairing-root-cause.md` (confirmed),
  `2026-08-14-manifold-robust-terminal-solver.md` (H4, now partially vindicated for the correct system)

## Context

SF-26 corrected the source pairing to the derived same-index form and proved it
with exact-pair contract tests. Re-running the unchanged heterogeneity gates
(`docs/experiments/2026-10-01-sf26-pairing-correction-gates.md`) falsified the
prespecified prediction: every method in the stack stalls exactly at eta = 1.

## Finding (confirmed by code runs on V100 and by independent numpy probes)

1. The corrected equation is exactly equivariant under label recombination
   (`psi1 -> psi1 + Phi(psi2)` etc.; continuum: `F' = M F`), so every solution
   lies on a manifold and the Jacobian is singular there in the continuum.
2. Discretely, J carries a near-null cluster (relative singular values
   1e-4..1e-6, dozens of modes at 24^3-32^3) that sharpens with refinement.
3. On smooth analytic fields the discrete system still has exact zeros
   (full Newton to 6e-16). On random Gaussian fields (ell/h = 8..16) it does
   NOT have a reachable zero: residual floor ~1e-4 (lambda = 0.11, sigma^2 = 1)
   independent of the solver (Picard, Anderson, restarted and full-recurrence
   Newton, explicit pseudo-time) and of the denominator regularization epsilon;
   it decreases only ~x2 per doubling of ell/h.
4. The crossed (pre-SF-26) system had isolated solutions at small amplitude
   (fast Anderson) but an unstable explicit flow and an unsolvable shelf at
   finite amplitude; the corrected system has the opposite profile: stable
   flow, slow damped maps, and an algebraic floor set by discretization +
   gauge degeneracy. Gate 3A physics at the floor is clean.

## Options for decision (not decided here)

- A. Truncation-aware acceptance for the implicit stack: stop at the
  measured floor; accept on Gate 3A physics + h-convergence of e_v,
  invariance, e_div (the floor itself tracked under refinement).
- B. SF-27 pseudo-time as planned, with error-controlled stepping that bounds
  the NONLINEAR stiffness (the linear Gershgorin cap is insufficient) and
  step budgets sized for tau >> 1.
- C. Gauge-fixing / bordered formulation removing the manifold (restores a
  nonsingular J; Newton-Krylov becomes applicable again).
- D. Reference resolution ell/h = 16 (paper) with the floor measured under
  refinement before any 256^3 attempt.

## Consequences

- The locked nonlinear tolerance (1e-6 per stage) is unreachable on the
  current fixtures for structural reasons; the heterogeneity gates stay
  unmet, recorded, untuned.
- SF-27's prespecified expectations (`O(1e3-1e4)` steps, agreement with the
  implicit stack to 1e-5 at tol 1e-10) must be revised by the owner.

## Classification (docs/AGENTS.md)

- Confirmed in code/runs: items 1-4 above (V100 suite + numpy probes).
- Proposed: options A-D.
- Open: scaling of the floor with ell/h beyond 16; whether a discrete
  gauge-fixing restores exact solvability.
