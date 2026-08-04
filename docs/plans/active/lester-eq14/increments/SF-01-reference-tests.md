# SF-01 — Reference test harness

- State: `active`
- Goal: `Crear un harness de pruebas independiente y referencias CPU para los operadores de streamfunctions.`
- Depends on: `SF-00`
- Unlocks: `SF-02`
- Branch: `science/lester-sf01-reference-tests`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf01-reference-tests`
- Acceptance gate: `Gate 1 + Gate 2 scaffold`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-04T21:24Z`
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
| 2026-08-04T21:24Z | `e593ea8`, active | Verified clean and fast-forwarded `master`, created the exact persistent Goal, and created the canonical SF-01 worktree. | `origin/master=e593ea8`; increment checker passed with `next=SF-01`; SF-00 is `done`; required central documents and local rules were read completely. | Inspect the existing test/CMake architecture, define the task DAG, and dispatch the first independent tasks. |
| 2026-08-04T21:30Z | `8694aa8`, active | Read-only tasks T01-T03 fixed the analytic CPU contract, isolated CMake/CTest pattern, CLI, and intentional mutation controls. | Use cell-centered periodic references and positive `A=-div(q grad)` with harmonic face `q`; preserve literal `16^3/32^3` controls and add anisotropic-domain fixtures at `16x24x32/32x48x64` with isotropic spacing; keep legacy `operator_tests` untouched. | Implement reference helpers in an isolated task worktree. |
| 2026-08-04T21:30Z | baseline `8694aa8` | Established the pre-implementation local baseline on GNU 16.1.1, CUDA 13.3.73, architecture 86. | `cmake --preset wsl-debug`, 86-target build, full CTest (1/1), and `config_pspta_small.yaml` smoke passed; smoke completed 500 steps with 387 active, 113 exited, and zero Newton failures. | Compare integrated SF-01 validation against this baseline. |
| 2026-08-04T21:36Z | task `3238082` | T04 added CPU-only periodic references and deterministic cubic/anisotropic fixtures; an initial direct-compile attempt exposed one unused helper warning and a malformed smoke command, both corrected before commit. | Independent `-std=c++17 -Wall -Wextra -Wpedantic -Werror` compilation passed; measured first-derivative RMS `0.114196 -> 0.0287146`, order `1.99166`; negative wrapping and invalid `q`/NaN inputs were rejected; task tree is clean. | Build the selectable analytic and intentional-mutation runner on T04. |
