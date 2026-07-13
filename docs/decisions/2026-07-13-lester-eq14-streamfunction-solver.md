# Lester equation (14) streamfunction solver direction

- Status: proposed
- Date: 2026-07-13

## Context

MacroFlow3D previously documented the transport-near-nullspace / eigensolver route as the main invariant-construction path for PSPTA. The project direction is now to design and integrate a solver for the coupled nonlinear Lester et al. equation (14) system to recover two streamfunctions `psi1`, `psi2`.

This decision record separates confirmed project choices from hypotheses that still require code-level validation.

## Decisions

1. The initial implementation focuses on smooth scalar locally isotropic conductivity fields.
   - Status: accepted for scope.

2. The primary benchmark uses Gaussian-covariance log-conductivity fields.
   - Status: accepted for validation scope.
   - Note: existing smoke/reference configs still use exponential covariance; those are not equivalent smooth-invariant validation cases.

3. The solver should prioritize the divergent form
   `Delta psi - grad(log k).grad psi = k div((1/k) grad psi)`.
   - Status: accepted.
   - Rationale: avoids explicit finite differences of `grad(log k)` and exposes a variable-coefficient diffusion operator.

4. Define `q=1/k` and `A psi = -div(q grad psi)` so decoupled iterations solve `A psi1 = -q S2` and `A psi2 = -q S1`.
   - Status: accepted as mathematical formulation.

5. Reuse of the existing PCG/MG stack is the priority architecture hypothesis.
   - Status: proposed, verify before relying on it.
   - Must check sign, coefficient placement, `q` coarsening, boundary conditions, gauge, SPD behavior, and consistency with residual evaluation.

6. The first nonlinear solver is damped Picard.
   - Status: accepted.
   - Do not start with Newton-Krylov before residual and operator tests are validated.

7. Anderson acceleration is a planned extension after basic Picard is working.
   - Status: proposed.

8. Matrix-free Newton-Krylov is the intended robust production direction.
   - Status: proposed.
   - Preconditioner target: block diagonal with two `A` blocks.

9. Continuation is part of the solver design.
   - Status: accepted.
   - Axes: heterogeneity `lambda`, nonlinearity `eta`, and grid resolution.

10. Denominator regularization for `|grad psi1 x grad psi2|^2` must be explicit, configurable, logged, and studied as epsilon tends to zero.
    - Status: accepted.

11. Validation must include velocity reconstruction, Darcy invariance, reconstructed-flow divergence, denominator percentiles, grid convergence, and invariant conservation along trajectories.
    - Status: accepted.
    - PDE residual alone is insufficient.

## Consequences

- `docs/plans/active/lester-eq14-streamfunction-solver-plan.md` is authoritative for new invariant-construction work.
- `docs/plans/archive/pspta-execution-plan.md` remains useful historical and PSPTA-transport context, but it no longer governs the new streamfunction construction strategy.
- Future solver tasks must begin on `16^3`/`32^3` controls and operator tests, not production grids.
- Exponential-covariance fields require explicit regularization/smoothness discussion before being used for invariant-existence claims.
- Tensorial or locally anisotropic conductivity is out of initial scope.

## Open verification items

- Confirm whether `VarCoeffLaplacian`/MG can be adapted directly to `q=1/k`.
- Confirm compatible discrete gradients and Hessian-vector products.
- Confirm gauge strategy for periodic and throughflow domains.
- Confirm whether construction should use double storage and cast only for PSPTA transport.
