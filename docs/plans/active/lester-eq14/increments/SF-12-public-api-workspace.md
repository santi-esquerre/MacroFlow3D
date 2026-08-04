# SF-12 — Public API and workspace

- State: `pending`
- Goal: `Definir la API pública, ownership y workspace reutilizable del solver.`
- Depends on: `SF-11`
- Unlocks: `SF-13`
- Branch: `science/lester-sf12-public-api-workspace`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf12-public-api-workspace`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Establish explicit data ownership and reusable GPU storage before nonlinear
loops make allocation and coupling behavior difficult to change.

## Preconditions

- SF-11 completes the numerical primitive and report inventory.

## In scope

- Public problem/config/result types, streamfunction fields, persistent
  workspace, memory estimator, and module boundaries.

## Out of scope

- Pipeline YAML, Picard behavior, field export, and transport consumption.

## Files and symbols

- Add `StreamfunctionTypes.hpp`, `StreamfunctionWorkspace.cuh/.cu`, and
  `StreamfunctionSolver.cuh` under `src/physics/streamfunctions/`.
- Define `AffineGauge`, `StreamfunctionProblemView`, `StreamfunctionFields`,
  `StreamfunctionSolverConfig`, and `StreamfunctionSolveReport`.

## Implementation specification

1. Accept nonowning grid, `K` or `Y`, reference Darcy velocity, BCs, and gauge;
   reject inconsistent dimensions and nonperiodic v1 benchmark BCs.
2. Own `u1/u2` in `StreamfunctionFields`; own all scratch and solver vectors in
   `StreamfunctionWorkspace`.
3. Allocate for sequential block solves by default and expose no concurrent
   solve mode yet.
4. Report fine-grid-equivalent field count and exact allocated bytes.

## Expected numerical effect

None beyond existing primitives; memory allocation becomes predictable and
hot-loop-safe.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- API type/size validation tests pass.
- Repeated workspace use performs no allocation after construction/resizing.
- Estimated bytes equal actual owned `DeviceBuffer` capacities.

## Regression surface

- Include dependencies, CUDA ownership, move semantics, and peak memory.

## Failure and rollback policy

- Do not expose unstable internal kernels in the public interface.
- If the 24.6-field budget cannot be met, record each extra field and redesign
  ownership before SF-13.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Public types and ownership boundaries are implemented.
- [ ] Workspace covers all accepted primitives without hot-loop allocation.
- [ ] Memory estimator is tested at all target grid sizes.
- [ ] API validation and full regression tests pass.
- [ ] Human review and evidence are recorded.
- [ ] Dashboard marks SF-12 complete and selects SF-13.
<!-- completion-checklist:end -->

## Advancement rule

SF-13 may implement the first complete homogeneous solve through this API.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
