# SF-11 — Physical diagnostics

- State: `pending`
- Goal: `Reconstruir v_psi sobre CompactMAC y calcular los diagnósticos físicos obligatorios.`
- Depends on: `SF-10`
- Unlocks: `SF-12`
- Branch: `science/lester-sf11-physical-diagnostics`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf11-physical-diagnostics`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Prevent acceptance based solely on algebraic residual by measuring velocity
reconstruction, invariance, divergence, angle, magnitude, and nondegeneracy.

## Preconditions

- SF-10 provides total cell gradients, `c`, and reduction infrastructure.

## In scope

- CompactMAC face reconstruction and all Gate 3A physical metrics.

## Out of scope

- Exact discrete curl formulations, output writers, and trajectory invariance.

## Files and symbols

- Add `src/physics/streamfunctions/Diagnostics.cuh/.cu`.
- Reuse `VelocityField` CompactMAC layout and natural MAC divergence.

## Implementation specification

1. At each normal face, use the normal centered derivative and consistently
   interpolated tangential derivatives before forming the cross product.
2. Compare directly with Darcy CompactMAC components.
3. Report L2/Linf by component, magnitude error, correlation, and robust angular
   error; exclude only explicitly counted near-zero pairs from angles.
4. Compute Darcy invariance at one documented common location and split
   cross-gradient degeneracy by Darcy-speed threshold.

## Expected numerical effect

The homogeneous flow reconstructs exactly; manufactured fields exhibit
second-order physical-metric convergence.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Uniform velocity reconstruction relative error `<=1e-13`.
- Manufactured velocity and divergence errors have L2 order at least 1.8.
- Invariance metrics agree with independent CPU calculations within `1e-12`
  relative plus spatial truncation error.

## Regression surface

- CompactMAC indexing, interpolation placement, periodic boundary faces, and
  treatment of low-speed cells.

## Failure and rollback policy

- Do not claim algebraic divergence freedom for this initial reconstruction.
- If face interpolation fails second-order convergence, stop before adding
  output or solver acceptance logic.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] CompactMAC reconstruction is implemented and documented.
- [ ] Required velocity, invariance, divergence, and degeneracy metrics exist.
- [ ] Uniform and manufactured thresholds pass.
- [ ] Low-speed exclusions are explicit and counted.
- [ ] Full regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-11 complete and selects SF-12.
<!-- completion-checklist:end -->

## Advancement rule

SF-12 may define the stable public ownership/API around the accepted numerical
primitives and reports.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
