# SF-23 — GPU optimization

- State: `pending`
- Goal: `Fusionar kernels y reducir tráfico de memoria sin cambiar resultados.`
- Depends on: `SF-22`
- Unlocks: `SF-24`
- Branch: `science/lester-sf23-gpu-optimization`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf23-gpu-optimization`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + performance evidence`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Reduce V100 memory traffic and synchronization only after the unfused double
implementation supplies an accepted numerical oracle.

## Preconditions

- SF-22 establishes stable Picard/Anderson correctness and memory baselines.

## In scope

- Source-kernel fusion, shared-memory stencil tiling, buffer lifetime reuse,
  reduction staging, and sequential/concurrent solve measurement.

## Out of scope

- Changing formulas, derivative order, solver tolerances, or precision.

## Files and symbols

- Optimize `DifferentialOperators`, `NonlinearSources`, reductions, workspace,
  and profiler ranges.
- Retain reference/unfused test path for direct comparison.

## Implementation specification

1. Fuse loads, gradients, Hessian-vector products, `B`, `c`, and `S1/S2`; write
   only sources and indispensable diagnostics.
2. Start with an `8x8x4` interior tile plus one-cell halo for both psi fields
   and required coefficient data; tune only with V100 measurements.
3. Reuse RHS/residual/trial buffers when lifetimes do not overlap.
4. Remove hidden hot-loop allocations and avoid host synchronization except
   convergence scalars.
5. Enable concurrent block solves only if duplicated mutable workspaces fit the
   budget and improve wall time measurably.

## Expected numerical effect

No accepted metric changes beyond floating-point reduction ordering; lower
kernel time and/or peak memory traffic on V100.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j && ctest --test-dir build/v100-release --output-on-failure"
scripts/remote run lester-sf23 -- "<profile-command>"
scripts/remote wait lester-sf23
```

## Acceptance thresholds

- Fused/unfused fields and metrics agree within predeclared double tolerances.
- No allocation or full-field transfer occurs in the nonlinear hot loop.
- At least one measured V100 improvement in source time, total time, or peak
  memory is demonstrated; otherwise retain the simpler reference path.

## Regression surface

- Halo loads, register occupancy, reduction order, buffer aliasing, and older
  GPU compilation.

## Failure and rollback policy

- Revert any optimization without a correctness match and measurable benefit.
- V100 is the tuning authority; K40 limitations do not dictate the design.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Fused kernel matches the reference suite.
- [ ] Buffer reuse and synchronization audit is documented.
- [ ] V100 profile and memory measurements are preserved.
- [ ] Only measured beneficial optimizations remain enabled.
- [ ] Gate 3A and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-23 complete and selects SF-24.
<!-- completion-checklist:end -->

## Advancement rule

SF-24 may run the production-scale V100 acceptance benchmarks.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
