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
| 2026-08-05T01:14Z | `22baf06`, integrated | Integrated accepted T04 (`a80cf93`): a CUB `DeviceReduce::Sum` primitive accumulating `real=double` with caller-prepared exact-size `ReductionWorkspace` storage. | `sum_device()` only enqueues work on the supplied stream; `sum_host()` is explicitly diagnostic and synchronizing. Earlier overlapping/polled T04 build attempts are recorded as **NO-EVIDENCE**; the accepted recovery was one serial `-j1` build. The prior task report typo is corrected by reproduced smoke evidence: `active=387`, `exited=113` (not `exited=0`). | Integrate the projector implementation. |
| 2026-08-05T01:14Z | `9567864`, integrated | Integrated accepted T05 (`40c3be9`): stateless `MeanZeroProjector` and move-only exact-size `MeanZeroWorkspace`. | `prepare()` owns backend query/allocation; `mean_device()` and `project()` have no allocation or host synchronization, use `CudaContext::cuda_stream()`, and reject a mismatched workspace before enqueue. Earlier overlapping/polled T05 build attempts are **NO-EVIDENCE**; the accepted recovery was one serial clean configure and `-j1` build. | Integrate and execute the isolated projector tests. |
| 2026-08-05T01:14Z | `15a82a3`, integrated | Integrated accepted T06 (`fff3d93`): seven GPU projector contract cases registered alongside the existing 19 SF-02 cases. | The deterministic cell-centered periodic fixture is `17x19x23` (`N=7429`); long-double CPU reference, mean bound `100*eps*max(RMS,1)`, comparison/idempotence bound `200*eps*max(RMS,1)`. Earlier overlapping/polled T06 build attempts are **NO-EVIDENCE**; only the serial recovery evidence below is accepted. | Perform clean integration validation. |
| 2026-08-05T01:14Z | `15a82a3`, validating evidence | Independent integration validation PASS (not acceptance): checker passed; `cmake --fresh --preset wsl-debug` passed; exactly one serial `cmake --build build/wsl-debug -j1` completed, followed by clean Ninja dry-run; all seven SF-03 cases and full `26/26` suite passed; targeted CTest `1/1`, full CTest `2/2`, and the PSPTA smoke passed. | Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, driver `610.43.03`, 4096 MiB; CUDA `13.3.73`, GNU C++ `16.1.1`, Debug preset. Metrics: shifted trig mean `7.83e-18`, CPU/GPU RMS `2.57e-18`; idempotence `6.08e-18`; diagnostic post-mean `9.56e-18`; double-vs-CPU RMS `8.59e-10` while the float-accumulator mutant is `6416`; stable workspace capacity `2815` bytes; stream-ordering RMS `2.57e-18`. Smoke (`apps/config_pspta_small.yaml`, seed `42`, grid `64x32x32`, legacy covariance) had head residual `1.02e+01 -> 1.77e-13`, `active=387`, `exited=113`, zero stalls/failures. Seed/covariance are N/A for the deterministic projector fixture. Gate 1 and Gate 2 pass; V100, Gate 3A and Gate 4 do not apply because SF-03 neither solves Eq. (14) nor changes physical reconstruction/tracking. CLI control: unknown case rejects (`exit=2`); registry duplicate names reject; repeated `--case` selection executes the named case twice (non-blocking harness behavior); `--list` cannot be combined with a case (`exit=2`). No allocations/synchronizations appear in `project()`/`mean_device()` by source audit; `mean_host()` is explicitly synchronizing. | Master-agent audit, then mandatory human review if it passes. |
