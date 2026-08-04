# SF-17 — Eta and epsilon continuation

- State: `pending`
- Goal: `Implementar continuación adaptativa en eta y epsilon con rollback.`
- Depends on: `SF-16`
- Unlocks: `SF-18`
- Branch: `science/lester-sf17-eta-epsilon-continuation`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf17-eta-epsilon-continuation`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Reach the unregularized nonlinear problem progressively while preserving the
last accepted state and making every homotopy decision reproducible.

## Preconditions

- SF-16 exposes adaptive Picard, histories, config, and output.

## In scope

- Reusable continuation-stage controller for `eta` and epsilon, adaptive steps,
  rollback, stage histories, and configuration.

## Out of scope

- `lambda`, stochastic generation, and grid continuation.

## Files and symbols

- Add `ContinuationController.hpp/.cu` under streamfunctions.
- Extend config/output only with eta and epsilon continuation fields.

## Implementation specification

1. For eta, use initial step `0.1`, minimum `0.0125`, maximum `0.25`, halve on
   failure, and grow by `1.5` after two easy stages.
2. Keep an accepted-state buffer and never overwrite it until a stage passes.
3. After eta one converges, reduce epsilon by decades from `1e-2` to `1e-6`;
   make `1e-8` configurable but not required.
4. Record parameter, attempted step, iterations, omega, residual, degeneracy,
   acceptance, and failure reason for every stage.

## Expected numerical effect

Controlled problems reach `eta=1` and lower epsilon without unrecoverable state
corruption.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_continuation
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Deterministic control reaches `eta=1` and `epsilon=1e-6`.
- Injected failures halve the step and restore the accepted state exactly.
- Minimum-step exhaustion returns a structured, logged failure.

## Regression surface

- State-buffer memory, parameter ordering, config serialization, and histories.

## Failure and rollback policy

- A stage failure never mutates the accepted solution.
- Do not skip failed parameter intervals or claim the fixed-epsilon result as
  the original system solution.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Eta stepping, epsilon stepping, and rollback are implemented.
- [ ] Step growth/reduction and minimum-step tests pass.
- [ ] Stage history contains every required diagnostic.
- [ ] Target continuation control reaches final parameters.
- [ ] Gate 3A regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-17 complete and selects SF-18.
<!-- completion-checklist:end -->

## Advancement rule

SF-18 may create the smooth periodic random fields required by the first
physical continuation benchmarks.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
