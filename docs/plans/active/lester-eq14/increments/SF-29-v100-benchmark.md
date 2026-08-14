# SF-29 — V100 benchmark

- State: `pending`
- Goal: `Validar robustez, memoria y convergencia del solver Picard en V100 hasta 256 cubed.`
- Depends on: `SF-28`
- Unlocks: `SF-30`
- Branch: `science/lester-sf-29-v100-benchmark`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf24-v100-benchmark`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + Gate 4`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Establish the accepted double-precision Picard baseline at the target physical
resolution before introducing Newton or mixed precision.

## Preconditions

- SF-27 provides validated kernels and measured memory behavior on V100.

## In scope

- Reproducible `128^3`/`256^3` Gaussian benchmarks, memory/runtime profiles,
  mesh comparison, and selected multi-realization robustness.

## Out of scope

- `sigma_Y^2=6.25`, exponential covariance, macrodispersion production, Newton,
  and consumer integration.

## Files and symbols

- Add immutable benchmark configs and notes under `docs/experiments/`.
- Use `scripts/remote` for all V100 execution and artifact collection.

## Implementation specification

1. Run `[0,1]^3`, periodic, physical `ell=1/16`, `sigma_Y^2=4`, mean velocity
   one, fixed seed, and the same realization at `128^3` and `256^3`.
2. Preserve complete continuation, linear, nonlinear, denominator, physical,
   runtime, and peak-memory histories.
3. Run 5–10 fixed seeds at validated `128^3` settings for robustness.
4. Compare restricted/prolongated fields and physical metrics, not only
   residuals.

## Expected numerical effect

The solver fits a V100, reaches the accepted Picard tolerance, and shows
physically consistent mesh behavior for smooth isotropic Darcy flow.

## Validation commands

```bash
scripts/remote sync
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j && ctest --test-dir build/v100-release --output-on-failure"
scripts/remote run lester-128 -- "<128-cubed-benchmark-command>"
scripts/remote wait lester-128
scripts/remote run lester-256 -- "<256-cubed-benchmark-command>"
scripts/remote wait lester-256
```

## Acceptance thresholds

- Peak memory fits the actual V100 with safety margin and no OOM retry policy.
- Linear residual `<=1e-10` and nonlinear `r_F<=1e-6` initially.
- No hidden fixed-epsilon claim; final epsilon and degeneracy populations are
  reported.
- `e_v`, invariance, divergence, and cross-gradient statistics are reviewed for
  convergence, with unexplained degradation blocking acceptance.

## Regression surface

- Remote reproducibility, seed/grid identity, peak memory, long-run stability,
  and interpretation of low-speed regions.

## Failure and rollback policy

- A failed realization remains part of the robustness result; do not delete it.
- A `256^3` failure returns to the responsible earlier increment through a new
  plan revision rather than changing benchmark physics.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] `128^3` and `256^3` same-realization runs are complete.
- [ ] Peak memory, runtime, and iteration profiles are recorded.
- [ ] All Gate 3A metrics and mesh comparisons are reviewed.
- [ ] Selected 5–10 realization robustness run is documented.
- [ ] Gate 4 human interpretation accepts or explicitly bounds remaining risk.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-29 complete and selects SF-30.
<!-- completion-checklist:end -->

## Advancement rule

SF-29 may study mixed precision using this full-stack baseline as the
fallback and performance reference.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
