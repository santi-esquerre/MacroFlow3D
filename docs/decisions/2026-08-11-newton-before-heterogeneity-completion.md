# Newton-Krylov before heterogeneity completion (SF-21 partial closure)

- Status: accepted
- Date: 2026-08-11
- Owner decision: option (a) — maximum solver robustness

## Context

SF-21 (heterogeneity continuation re-activation) delivered and validated the
complete lambda/eta-rescue continuation machinery with Anderson acceleration
wired into the stage solver. Its PRESPECIFIED 32^3 smoke gates split:

- `sigma_Y^2=0.25`: PASS — `lambda=1` in 8/8 accepted stages, zero rescues,
  488 s on V100 (pre-Anderson: floor death at `lambda=0.373` after 7.75 h).
- `sigma_Y^2=1`: FAIL — `lambda_floor_exhausted` at `lambda=0.5` (pre-Anderson:
  `lambda=0.10`). New signature: eta-rescue ramps accept
  `eta=0.98125/0.996875` to `r_F<=1e-6`, but attempts AT `eta=1` exit via the
  stagnation detector with `r_F` plateauing at ~1e-3. The damped
  Picard/Anderson fixed-point map is NON-CONTRACTIVE at
  `(sigma^2=1, lambda~0.5, eta=1, epsilon=1e-2)`; physics stayed clean at
  every accepted stage (zero unexplained degeneracy, smooth Gate-3A metrics).

Two successive re-sequencings have now confirmed the same structural fact:
fixed-point iteration (however damped/accelerated/safeguarded) is not
sufficient for terminal convergence of the full nonlinear coupling on
physical Gaussian fields at moderate variance. The plan always reserved
Newton-Krylov for terminal convergence; the empirical boundary simply
arrived earlier than the original sequencing assumed.

## Decision

1. Close SF-21 as a PARTIAL increment via a versioned gate amendment (owner
   instruction): the `sigma^2=0.25` 32^3 gate stands as passed; the
   `sigma^2=1` 32^3 gate and the entire 64^3 suite move UNCHANGED to a new
   post-Newton increment. All SF-21 machinery, tests, pipeline surface, and
   partial evidence are delivered and frozen at audited head `463753d`.
2. Pull the Newton-Krylov phase forward. New sequence:
   - SF-22 — Matrix-free Jacobian-vector product (was SF-25)
   - SF-23 — Restarted GMRES + block preconditioner (was SF-26)
   - SF-24 — Globalized Newton-Krylov (was SF-27)
   - SF-25 — Heterogeneity completion (NEW: the moved SF-21 gates, verbatim)
   - SF-26 — Grid continuation (was SF-22)
   - SF-27 — GPU optimization (was SF-23)
   - SF-28 — V100 benchmark (was SF-24)
   - SF-29 — Mixed-precision study (was SF-28)
3. Robustness doctrine for the solver stack (the point of option (a)):
   layered defense with explicit handoffs — Picard (permanent fallback,
   correctness reference) -> Anderson (safeguarded acceleration of the
   contractive regime) -> Newton-Krylov (terminal quadratic convergence
   inside its basin, activation-thresholded, Armijo-globalized, with
   reproducible fallback to the fixed-point layers) — all wrapped by the
   lambda/eta/epsilon continuation with bitwise rollback. Every layer keeps
   its own guards; no layer bypasses another's safeguards.

## Consequences

- Grid continuation, GPU optimization, and the V100 benchmark now run AFTER
  the full solver stack exists, so they exercise and optimize the real
  production path (confirmed in code: the benchmark/optimization increments
  no longer target a Picard-only solver that would be obsoleted).
- The Lester reference case (256^3, sigma^2=4) is only attempted with the
  complete stack — maximum robustness before scale.
- SF-21's partial closure is honest and versioned: gates were never tuned;
  the unmet gates move verbatim with their evidence trail to SF-25.
- Sequence length grows by one increment (SF-29 total).
