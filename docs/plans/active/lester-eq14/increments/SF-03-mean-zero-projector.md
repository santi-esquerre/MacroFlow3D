# SF-03 — Mean-zero projector

- State: `done`
- Goal: `Implementar una proyección GPU robusta al subespacio de media cero.`
- Depends on: `SF-02`
- Unlocks: `SF-04`
- Branch: `science/lester-sf03-mean-zero-projector`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf03-mean-zero-projector`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-05T00:23Z`
- Completed: `2026-08-05T12:05Z`
- PR: `#8`
- Commit: `5f3eec0`

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
- [x] Projector and explicit workspace are implemented.
- [x] Accuracy and idempotence thresholds pass.
- [x] Odd-size and periodic-grid cases pass.
- [x] Allocation/synchronization behavior is documented.
- [x] Full regression suite and human review pass.
- [x] Evidence, PR, and commit are recorded.
- [x] Dashboard marks SF-03 complete and selects SF-04.
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
| 2026-08-05T01:20Z | `13e6478`, corrective C01 | Corrected the workspace allocation documentation after the master audit found that its wording made `prepare()` sound like the sole possible allocation point. | `blas::ReductionWorkspace` construction may reserve its device result scalar, while `MeanZeroWorkspace::prepare()` may query and reserve or resize temporary reduction storage.  The accepted hot-path contract is unchanged: after exact-size preparation, `project()` and `mean_device()` allocate nothing and do not synchronize the host.  This comments-only correction has no numerical effect. | Re-run the checker and diff audit, then reintegrate the corrective commit before repeating independent validation. |
| 2026-08-05T01:28Z | `f2a2fbb`, corrective C02 | Removed the unused const `ProjectorFixture::context()` accessor identified during corrective review. | Test-only one-line removal; it eliminates the new-test unused-accessor warning and has no numerical, allocation, stream, or production-code effect. | Reintegrate C01/C02 in topological order and repeat independent validation. |
| 2026-08-05T01:28Z | `f2a2fbb`, corrective reintegration evidence | Independent corrective reintegration of C01 `136881e -> 20a261a` and C02 `cf32d29 -> f2a2fbb` passed. | REWORK cause: allocation documentation was inaccurate because `ReductionWorkspace` construction can reserve `d_scalar`, and the test had an unused accessor. C01 now documents construction plus prepare semantics truthfully; source audit confirms `project()`/`mean_device()` have no allocation or host sync after exact-size preparation, while `mean_host()` explicitly synchronizes. Checker, fresh Debug configure, one serial `-j1` build, all 7 projector cases, full `26/26` executable suite, focal CTest `1/1`, full CTest `2/2`, and PSPTA smoke passed. The new projector test source emitted no warning; one unrelated legacy `prolong_3d.cu:107` unused `NZ` warning remains. Metrics reproduced: shifted mean `7.83e-18`, CPU/GPU RMS `2.57e-18`, idempotence `6.08e-18`, diagnostic post-mean `9.56e-18`, double RMS `8.59e-10` vs float mutant `6416`, workspace `2815` B, stream RMS `2.57e-18`; smoke head residual `1.02e+01 -> 1.77e-13`, active/exited `387/113`, no stalls/failures. Gate 1 and Gate 2 pass; V100, Gate 3A, and Gate 4 are N/A because this increment does not run the Eq. (14) solver or change reconstruction/tracking. `git diff --check master...HEAD` passed; scope remains SF-03 only. | Master audit and mandatory human review; state, checklist, NEXT, Goal, and PR metadata remain unchanged. |
| 2026-08-05T01:33Z | `2de232d`, validating | Master audit repeated from the complete diff after corrective reintegration and classified **PASS**; implementation is frozen pending human review. | Personally inspected all nine branch commits and the full 13-file diff against `master=802f5d2`: arithmetic cell-centered gauge, double reduction, exact-size workspace, stream ordering, error paths, ownership, hot-path allocation/synchronization, odd-size coverage, idempotence, and absence of PCG/MG/PSPTA/SF-04 changes are consistent with SF-03. A fresh configure and serial build completed with exit 0 and no warning from the new projector test; 7/7 projector cases, 26/26 executable cases, CTest 2/2, checker, Gate 1, Gate 2, and PSPTA smoke passed. Metrics reproduced exactly: shifted mean `7.83e-18`, CPU/GPU RMS `2.57e-18`, idempotence `6.08e-18`, diagnostic post-mean `9.56e-18`, double RMS `8.59e-10` vs float mutant `6416`, workspace `2815` B; smoke residual `1.02e+01 -> 1.77e-13`, active/exited `387/113`, zero stalls/failures. Gate 3A, Gate 4, and V100 remain N/A for this isolated projector increment. | Open the scientific-core PR, record it, and request mandatory human review without merging. |
| 2026-08-05T01:34Z | `20c199c`, awaiting_review | Opened ready-for-review PR #8 after the repeated master audit passed and froze implementation changes. | PR #8 targets `master` from `science/lester-sf03-mean-zero-projector`, is open, non-draft, and mergeable. Its description records scope, commands, metrics, residual risks, applicable gates, and files intentionally untouched. GitGuardian security check passed at creation; scientific-core human review remains mandatory. | Await explicit human approval; do not merge, close the Goal, mark SF-03 done, or advance NEXT. |
| 2026-08-05T12:13Z | closeout reassigned | The first C04 closeout attempt was interrupted after producing no diff or report; this procedural closeout was reassigned on a clean worktree. | No technical hypothesis, implementation, or validation evidence is attributed to the interrupted attempt. | Record the confirmed GitHub merge factually and close SF-03 state. |
| 2026-08-05T12:13Z | `5f3eec0`, done | Closed SF-03 after the user confirmed review completion and merge of PR #8. | `gh pr view 8` reports `state=MERGED`, `mergedAt=2026-08-05T12:05:51Z`, merge `5f3eec0`, and successful GitGuardian check; it reports `reviews=[]`, so no formal review record is invented. Versioned validation: serial Debug build, 7/7 projector cases, 26/26 executable cases, CTest 2/2, checker, Gates 1/2, and PSPTA smoke passed; shifted mean `7.83e-18`, CPU/GPU RMS `2.57e-18`, idempotence `6.08e-18`, double RMS `8.59e-10` vs float mutant `6416`, workspace `2815` B. | SF-03 complete; residual risk is CUB floating-sum reproducibility across GPU architectures, deferred to SF-23. |
