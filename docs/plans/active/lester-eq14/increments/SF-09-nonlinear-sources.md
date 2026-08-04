# SF-09 — Nonlinear sources

- State: `pending`
- Goal: `Implementar c, S1, S2 y la regularización explícita del denominador.`
- Depends on: `SF-08`
- Unlocks: `SF-10`
- Branch: `science/lester-sf09-nonlinear-sources`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf09-nonlinear-sources`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Construct the two nonlinear Lester sources with transparent, dimensionless
regularization and enough diagnostics to distinguish degeneracy from low Darcy
speed.

## Preconditions

- SF-08 validates gradients, Hessian-vector products, and `B`.

## In scope

- Cross-gradient `c`, regularized denominator, `S1`, `S2`, source RHS terms,
  finite-value flags, basic degeneracy counts, and CPU/GPU tests.

## Out of scope

- Picard, adaptive epsilon continuation, full physical metrics, and kernel
  optimization.

## Files and symbols

- Add `src/physics/streamfunctions/NonlinearSources.cuh/.cu` and source workspace
  types.
- Extend test references with cross product and Lester source formulas.

## Implementation specification

1. Compute `c=g1 cross g2` and
   `d=|c|^2+(epsilon*v_rms)^2` in double.
2. Compute each `S_i=((B cross g_i) dot c)/d` and write only the two source
   arrays plus requested diagnostic accumulators.
3. Default `epsilon=1e-2`; never insert an unreported hard-coded floor.
4. Count nonfinite cells and `|c|/v_rms` below configured thresholds.

## Expected numerical effect

Smooth nondegenerate controls yield finite second-order sources; regularization
dependence is explicit rather than hidden.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Homogeneous affine control has `S1,S2` at roundoff.
- Manufactured source errors converge with L2 order at least 1.8 away from
  degeneracy.
- CPU/GPU nonfinite flags and threshold counts agree exactly.

## Regression surface

- Source index pairing (`A psi1` uses `S2`), units of epsilon, and denominator
  behavior near stagnation.

## Failure and rollback policy

- A nonfinite output or unexplained denominator collapse fails the increment.
- Do not increase epsilon merely to make a failing manufactured case pass.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] c, both sources, and explicit regularization are implemented.
- [ ] Source pairing and sign are covered by tests.
- [ ] Homogeneous and manufactured thresholds pass.
- [ ] Nonfinite and degeneracy diagnostics agree with CPU reference.
- [ ] Full regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-09 complete and selects SF-10.
<!-- completion-checklist:end -->

## Advancement rule

SF-10 may assemble the coupled residual from these validated sources.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
