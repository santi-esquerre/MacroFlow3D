# SF-22 — Anderson acceleration

- State: `pending`
- Goal: `Incorporar Anderson acceleration con profundidad configurable y salvaguardas.`
- Depends on: `SF-21`
- Unlocks: `SF-23`
- Branch: `science/lester-sf22-anderson`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf22-anderson`
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

- SF-21 provides a robust accepted Picard/continuation baseline and histories.

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
- [ ] Dashboard marks SF-22 complete and selects SF-23.
<!-- completion-checklist:end -->

## Advancement rule

SF-23 may optimize kernels using Picard and Anderson outputs as correctness
baselines.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
