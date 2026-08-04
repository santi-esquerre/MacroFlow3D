# SF-01 — Reference test harness

- State: `pending`
- Goal: `Crear un harness de pruebas independiente y referencias CPU para los operadores de streamfunctions.`
- Depends on: `SF-00`
- Unlocks: `SF-02`
- Branch: `science/lester-sf01-reference-tests`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf01-reference-tests`
- Acceptance gate: `Gate 1 + Gate 2 scaffold`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Provide an independent oracle and executable for verifying each new discrete
operator before any nonlinear solver is trusted.

## Preconditions

- SF-00 is `done` on the default branch.

## In scope

- A new streamfunction-specific CTest executable, CPU reference helpers,
  periodic trigonometric fixtures, convergence-rate utilities, and reports.

## Out of scope

- Production GPU operators, PCG/MG changes, solver code, and PSPTA tests.

## Files and symbols

- Add `tests/streamfunctions/` or the closest existing test location.
- Add `apps/run_streamfunction_tests.cu` only if the CMake test pattern requires
  an application entry point.
- Extend `CMakeLists.txt` with a separate `streamfunction_operator_tests` target.

## Implementation specification

1. Implement CPU index wrapping, centered first/second/mixed derivatives,
   divergence-form diffusion, cross products, RMS/Linf norms, and observed
   convergence rates.
2. Supply deterministic sine/cosine fields on anisotropic domain lengths while
   retaining isotropic grid spacing for current production compatibility.
3. Make each case independently selectable and print grid, norm, expected
   order, observed order, and pass threshold.
4. Do not include or extend `OperatorTestHarness` under legacy PSPTA.

## Expected numerical effect

None in production.  Tests establish independent reference values.

## Validation commands

```bash
cmake --preset wsl-debug
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- CPU analytic controls pass at `16^3` and `32^3`.
- Intentional sign and periodic-index perturbation controls are detected.
- The current PSPTA operator test remains unchanged and passing.

## Regression surface

- CMake target registration, CUDA separable compilation, and test runtime.

## Failure and rollback policy

- Keep production code untouched if the reference convention is disputed.
- Record discrepancies as test findings; do not adapt the oracle to current
  implementation output without an analytic justification.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Independent reference helpers and deterministic fixtures are implemented.
- [ ] Separate CTest target is registered.
- [ ] Positive and intentional-failure controls behave as expected.
- [ ] Full local test suite passes.
- [ ] Scientific review confirms the reference formulas.
- [ ] Evidence, PR, and commit are recorded in the bitácora.
- [ ] Dashboard marks SF-01 complete and selects SF-02.
<!-- completion-checklist:end -->

## Advancement rule

SF-02 may start after this target and its analytic reference are merged.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
