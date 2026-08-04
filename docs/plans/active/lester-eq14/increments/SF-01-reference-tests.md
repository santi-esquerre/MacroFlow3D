# SF-01 — Reference test harness

- State: `done`
- Goal: `Crear un harness de pruebas independiente y referencias CPU para los operadores de streamfunctions.`
- Depends on: `SF-00`
- Unlocks: `SF-02`
- Branch: `science/lester-sf01-reference-tests`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf01-reference-tests`
- Acceptance gate: `Gate 1 + Gate 2 scaffold`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-04T21:24Z`
- Completed: `2026-08-04T21:58Z`
- PR: `https://github.com/santi-esquerre/MacroFlow3D/pull/4`
- Commit: `8884df9`

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
- [x] Independent reference helpers and deterministic fixtures are implemented.
- [x] Separate CTest target is registered.
- [x] Positive and intentional-failure controls behave as expected.
- [x] Full local test suite passes.
- [x] Scientific review confirms the reference formulas.
- [x] Evidence, PR, and commit are recorded in the bitácora.
- [x] Dashboard marks SF-01 complete and selects SF-02.
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
| 2026-08-04T21:40Z | task `8d7d983` | T05 added the selectable CPU runner and positive/intentional-mutation matrix without changing the oracle or build system. | Strict direct compilation and all cases passed; observed orders were `1.925` (literal cubes), `1.981` (first derivatives), `1.985` (second/mixed), and `1.988` (diffusion); sign error `1.92280` vs correct `0.08574` and wrap error `0.26085` vs correct `0.02550` were detected; invalid CLI exits `2`. | Register the isolated target and exercise it through CTest. |
| 2026-08-04T21:46Z | integration `6467073` | Independent integration cherry-picked task commits `3238082 -> 8d7d983 -> ddcb0da` as `5fb293d -> cf3e0c6 -> 6467073`; no conflicts, missing commits, duplicate APIs/buffers, production changes, or legacy `OperatorTestHarness`/`run_operator_tests` diff were found. | Checker passed (`29`, `next=SF-01`); configure/build passed on GCC 16.1.1, CMake 4.4.2, CUDA 13.3.73, RTX 3050 Laptop 4 GiB (sm_86). CTest: new target `0.52 s`, legacy `1.91 s`, full `2/2` in `0.73 s`; PSPTA smoke passed: 500 steps, active/exited `387/113`, Newton failures `0`. The first T06 selection attempt used nonexistent `--case periodic_index` and failed as expected; integration instead enumerated the authoritative `--list` cases, each passed, and missing/unknown `--case` both exited `2`. An initial `/usr/bin/time` wrapper was unavailable and was rerun with the canonical command; CMake reported only the pre-existing yaml-cpp deprecation warning. Gate 2 scaffold: positive `A=-div(q grad)` with exact harmonic `q_f(1,4)=1.6`, all-axis periodic wrapping, cubes `16^3->32^3` order `1.925`, anisotropic `16x24x32->32x48x64` orders first/second-mixed/diffusion `1.981/1.985/1.988`; sign mutant `1.9227972` versus `0.085742492`, wrap mutant `0.26084964` versus `0.025504642`. Gate 3A, Gate 4, and V100 are not applicable: this is a CPU-only test-local oracle with no production streamfunction solver, physical fields, or V100 benchmark. | Master-agent audit and required human review; do not advance state or dashboard. |
| 2026-08-04T21:47Z | integration follow-up | The first new-target CTest attempt reported that `streamfunction_operator_tests` was not yet found immediately after the initial build invocation; no source defect was inferred or changed. | Re-ran the canonical build, verified the explicit `streamfunction_operator_tests` target (`ninja: no work to do`), then CTest passed in `0.52 s`; all later direct and full-suite checks passed. This is retained as integration evidence rather than silently discarded. | Preserve for master audit; investigate only if reproducible in a clean build. |
| 2026-08-04T21:50Z | master audit PASS at `e069a39` | Audited the complete diff and commit chain from `master`, re-derived the discrete formulas, and independently reran the required positive, negative, regression, and error-path checks. | Scope is test/CMake/state only; `A=-div(q grad)` sign, harmonic face `q`, cell-centered indexing, periodicity, anisotropic isotropic-spacing fixtures, analytic derivatives, norms, thresholds, and CLI are consistent. Checker, configure, 48-step rebuild, target CTest, all/individual cases, invalid exit codes, legacy CTest, full `2/2` CTest, and 500-step smoke passed; ASan/UBSan passed and the CPU target has no project/CUDA link. The transient missing executable did not reproduce. Gate 3A/4/V100 remain not applicable to this non-production scaffold. | Freeze implementation, open the draft PR, and request mandatory human scientific review. |
| 2026-08-04T21:52Z | PR `#4`, awaiting review | Published `science/lester-sf01-reference-tests` and opened the SF-01 pull request against `master` with the complete scope, commands, numerical results, risks, and intentionally untouched files. | PR: `https://github.com/santi-esquerre/MacroFlow3D/pull/4`; implementation is frozen after the master PASS and mandatory human review is now the only advancement condition. | Human reviewer verifies formulas and thresholds; do not merge, complete the runtime Goal, or start SF-02 before explicit approval. |
| 2026-08-04T21:58Z | `8884df9`, done | Human review explicitly approved the analytic formulas and thresholds; PR `#4` was merged through GitHub and the implementation merge is visible on `master`. | Final evidence: checker, configure/build, dedicated and full CTest, all selectable controls, invalid CLI paths, legacy operator test, 500-step smoke, and ASan/UBSan passed; observed orders were `1.925–1.988`; sign/wrap mutants were rejected. Residual scope risk is limited to the intended fact that this is a CPU test oracle, not a production operator. | Merge this procedural closure through a follow-up PR, verify `NEXT=SF-02` on default, then complete the SF-01 runtime Goal. |
