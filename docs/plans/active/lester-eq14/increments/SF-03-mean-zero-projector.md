# SF-03 — Mean-zero projector

- State: `active`
- Goal: `Implementar una proyección GPU robusta al subespacio de media cero.`
- Depends on: `SF-02`
- Unlocks: `SF-04`
- Branch: `science/lester-sf03-mean-zero-projector`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf03-mean-zero-projector`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-05T00:23Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Remove the periodic constant null mode without pinning a physical cell or
destroying symmetry.

## Preconditions

- SF-02 positive periodic operator contract is accepted.

## In scope

- Reusable GPU mean reduction/subtraction, reusable workspace, and projector
  tests for cell-centered fields.

## Out of scope

- PCG integration, multigrid level projection, and source compatibility policy.

## Files and symbols

- Add `src/numerics/constraints/MeanZeroProjector.cuh/.cu` or the closest
  non-PSPTA numerical namespace.
- Reuse `DeviceBuffer`, `DeviceSpan`, `CudaContext`, and existing BLAS reductions
  where their synchronization behavior is explicit.

## Implementation specification

1. Accept a preallocated reduction workspace and CUDA stream/context.
2. Compute the double-precision mean and subtract it in place.
3. Expose a diagnostic mean query without allocating device memory.
4. Test constants, shifted trigonometric fields, repeated projection, and sizes
   not divisible by the CUDA block size.

## Expected numerical effect

Fields retain only their zero-mean component; nonconstant Fourier modes remain
unchanged to reduction roundoff.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- `|mean(Px)| <= 100*epsilon_machine*max(RMS(x),1)`.
- `RMS(P(Px)-Px)` is at roundoff scale.
- No allocation occurs inside `project()`.

## Regression surface

- Shared reduction primitives, CUDA stream ordering, and device synchronization.

## Failure and rollback policy

- Do not reuse a reduction that accumulates in float or hides allocation.
- If deterministic and fastest reductions differ, accept the stable double
  reduction first and defer optimization to SF-23.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Projector and explicit workspace are implemented.
- [ ] Accuracy and idempotence thresholds pass.
- [ ] Odd-size and periodic-grid cases pass.
- [ ] Allocation/synchronization behavior is documented.
- [ ] Full regression suite and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-03 complete and selects SF-04.
<!-- completion-checklist:end -->

## Advancement rule

SF-04 may integrate the projector into PCG after this increment is merged.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-05T00:23Z | `802f5d2`, active | Verified the reviewed SF-02 implementation and procedural closeout on the default branch, created the exact SF-03 runtime Goal, and created the canonical SF-03 worktree. | `master=origin/master=802f5d2`; increment checker passed with `next=SF-03`; SF-02 is `done`; SF-03 depends only on SF-02. | Audit existing reduction primitives and stream/allocation contracts, then construct the SF-03 task DAG. |
| 2026-08-05T00:30Z | `d248bf0`, active | Accepted read-only audits T01-T03 and the explicit SF-03 task DAG; established the pre-implementation baseline. | Use CUB `DeviceReduce::Sum` in double through the existing `blas::ReductionWorkspace`, wrapped by an exact-size `MeanZeroWorkspace`; `prepare()` alone may allocate, `mean_device()` and `project()` stay ordered on `CudaContext::cuda_stream()` without allocation or host synchronization, and `mean_host()` synchronizes explicitly. Tests use a periodic cell-centered `17x19x23` fixture (`7429` cells), CPU long-double reference, the literal mean bound `100*epsilon*max(RMS(x),1)`, and `200*epsilon` comparison/idempotence bounds. Baseline configure/build, CTest 2/2, and the 500-step PSPTA smoke passed on RTX 3050 Laptop GPU; head residual `1.77e-13`, zero stalls/failures. | Implement the reusable double sum primitive with caller-prepared storage; do not add projector or tests yet. |
