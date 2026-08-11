# SF-20 — Anderson acceleration

- State: `pending`
- Goal: `Incorporar Anderson acceleration con profundidad configurable y salvaguardas.`
- Depends on: `SF-19`
- Unlocks: `SF-21`
- Branch: `science/lester-sf20-anderson`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Reduce Picard iterations without weakening the validated residual, gauge,
degeneracy, or rollback safeguards.

## Preconditions

- SF-19 provides periodic `Y` fields, affine-periodic Darcy flow, and the
  SF-17 warm-started continuation machinery (including the SF-20-era
  heterogeneity driver and `CoefficientState` extension, parked pending
  re-activation of the heterogeneity increment).
- Re-sequencing motivation (2026-08-11 owner decision, see
  `docs/decisions/2026-08-11-anderson-before-heterogeneity.md`): plain
  adaptive Picard stalls asymptotically at full coupling `eta=1` on
  physical Gaussian fields (32^3, sigma_Y^2=0.25 stalls at lambda~0.37;
  sigma_Y^2=1 at lambda~0.10) with iteration counts diverging as
  `eta->1` — the fixed-point map's spectral radius approaches 1.

## In scope

- Anderson depth 3–8, default 5, start iteration 5, coupled history, small dense
  least-squares solve, conditioning reset, and Picard fallback.

## Out of scope

- Newton, concurrent block solves, and mixed precision.

## Files and symbols

- Add `AndersonAccelerator.cuh/.cu` and optional config/history fields.
- Store coupled `Delta X` and `Delta F`, four scalar fields per history level.

## Implementation specification

1. Form dot products on GPU and transfer only the small dense matrix/vector.
2. Solve the least-squares problem by pivoted QR; reject/reset if condition
   estimate exceeds `1e12`.
3. Project accelerated candidates and pass them through the same residual and
   degeneracy safeguard as Picard.
4. On any failed acceleration, clear history and accept/retry the normal Picard
   candidate according to SF-15.

## Expected numerical effect

The fixed benchmark suite converges in no more nonlinear iterations than Picard
and normally fewer, with the same final solution and diagnostics.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_anderson
scripts/remote exec -- "<fixed-picard-vs-anderson-benchmark>"
```

## Acceptance thresholds

- Depth 3/5/8 memory equals `4*m` scalar fields plus small dense storage.
- Ill-conditioned and rejected controls fall back safely.
- Default depth five does not increase iteration count on the acceptance suite
  and final fields agree with Picard within nonlinear tolerance.
- On the SF-20-era stall fixtures (32^3 physical Gaussian, ell=8, seed 12345,
  sigma_Y^2 = 0.25 and 1.0, eta=1, epsilon=1e-2), Anderson-accelerated
  Picard converges `r_F <= 1e-6` within the standard 500-iteration budget
  where plain Picard exhausted it.

## Regression surface

- GPU memory, history ordering, dense solve robustness, projection, and
  continuation rollback.

## Failure and rollback policy

- Anderson remains disabled by default if it does not improve the fixed suite.
- Never accept an accelerated candidate that fails the Picard safeguard.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Coupled history and pivoted least-squares solve are implemented.
- [ ] Projection, conditioning reset, rejection, and fallback tests pass.
- [ ] Memory for depths 3, 5, and 8 is measured.
- [ ] Fixed-suite comparison against Picard is recorded.
- [ ] Gate 3A review and full regressions pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-20 complete and selects SF-21.
<!-- completion-checklist:end -->

## Advancement rule

SF-21 may resume the heterogeneity continuation using Anderson-accelerated
Picard as the stage solver.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-11T14:10Z | re-sequenced into slot SF-20 (was SF-22) | Owner decision (option (a), 2026-08-11): pull Anderson acceleration BEFORE the heterogeneity continuation, motivated by the SF-20-era honest BLOCKED evidence (asymptotic Picard stall at eta=1 on physical Gaussian fields; see the SF-21 heterogeneity bitácora and `docs/decisions/2026-08-11-anderson-before-heterogeneity.md`). Spec adjustments in this re-sequencing: Depends on SF-19; a new PRESPECIFIED acceptance threshold targeting the recorded stall fixtures; preconditions updated. All other gates unchanged. | The stall fixtures give Anderson a sharply defined, already-measured target: plain Picard needed >500 iterations at eta=1 (r_F stuck at 1-5x tolerance) where eta=0.95 needed only ~80. | Activate only when named by `NEXT` after this re-sequencing PR is merged. |
