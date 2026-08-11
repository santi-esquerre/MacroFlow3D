# Anderson acceleration before heterogeneity continuation

- Status: accepted
- Date: 2026-08-11
- Deciders: repository owner (explicit instruction: "opción (a): re-secuenciá
  el plan con Anderson primero"), on evidence assembled by the orchestrator.

## Context

The original Lester increment sequence ran the heterogeneity (lambda)
continuation (then SF-20) and grid continuation (then SF-21) before Anderson
acceleration (then SF-22), assuming plain adaptive Picard would converge
small smooth Gaussian cases.

The SF-20-era attempt falsified that assumption honestly [confirmed in code
and recorded evidence]: both PRESPECIFIED 32^3 physical smokes (ell=8, seed
12345, dashboard-locked solver defaults, run verbatim on the remote V100)
ended `lambda_floor_exhausted` — sigma_Y^2=0.25 stalled at lambda~0.373 and
sigma_Y^2=1 at lambda~0.10. The failure mechanism is a pure asymptotic
convergence-rate stall of the Picard fixed-point map at full nonlinear
coupling: eta-rescue ramps converge easily to eta=0.95 (~80 iterations) and
iteration counts diverge as eta->1 (~150 at 0.98125, ~450 at 0.996875,
budget-exhausted >500 at 1.0 with r_F stuck at 1-5x tolerance). No
divergence, no degeneracy, no guard trips; every accepted stage honored
r_F <= 1e-6. The continuation machinery itself (lambda stepping, eta-rescue
ordering, hierarchy lifecycle, rollback, Gate-3A records) operated exactly
as specified through 161 stages.

The plan's own architecture already reserved Anderson acceleration for
exactly this coupled fixed-point map (overview §3); the empirical finding is
that it is needed EARLIER than the original sequencing assumed — already at
sigma_Y^2=0.25 on 32^3.

## Decision

Rotate increment slots 20/21/22 [accepted scope]:

- SF-20 = Anderson acceleration (was SF-22); depends on SF-19; gains one
  prespecified acceptance threshold targeting the recorded stall fixtures
  (converge them within the standard budget where plain Picard exhausted it).
- SF-21 = Heterogeneity continuation (was SF-20); gates UNCHANGED; its
  already-audited machinery (T01 `88076a0`, T02 `dcabc25`, T03 `8305c10` on
  activation base `0177ead`) is parked in Claude worker worktrees for reuse
  at re-activation.
- SF-22 = Grid continuation (was SF-21); content unchanged.
- SF-23+ unchanged (dependency prose updated).

The checker derives sequence from file numbering, so the rotation is
implemented by renaming the increment files and updating Depends/Unlocks/
titles/links; all bitácoras are preserved append-only.

## Consequences

- The Picard-only stall evidence remains the versioned scientific record in
  the SF-21 bitácora; nothing was relabeled or tuned.
- Anderson now has a sharply defined, already-measured acceptance target
  [accepted scope]; if Anderson also fails those fixtures, that is a new
  material finding for the Newton-Krylov phase (open question).
- The heterogeneity re-activation must wire Anderson into its stage solver
  and expects a rebase of the parked commits over the merged Anderson work
  (proposed architecture).
- The 64^3 suite (including sigma_Y^2=4) stays deferred until SF-21
  re-activation.
