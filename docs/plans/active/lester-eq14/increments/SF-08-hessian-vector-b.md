# SF-08 — Hessian-vector products and B

- State: `awaiting_review`
- Goal: `Implementar productos Hessiano-vector y la construcción de B sin almacenar Hessianos.`
- Depends on: `SF-07`
- Unlocks: `SF-09`
- Branch: `science/lester-sf08-hessian-vector-b`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf08-hessian-vector-b`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A operator subset`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-06T02:33Z`
- Completed: `not completed`
- PR: [#18 — SF-08: add register-only Hessian-vector B operator](https://github.com/santi-esquerre/MacroFlow3D/pull/18)
- Commit: `not recorded`

## Scientific or engineering intent

Evaluate the directional curvature required by Lester equation (14) with a
small periodic stencil and without a nine-component Hessian memory cost.

## Preconditions

- SF-07 total gradients and derivative conventions are accepted.

## In scope

- Direct `H(psi2)*grad(psi1)`, `H(psi1)*grad(psi2)`, and their difference `B`.

## Out of scope

- `c`, `S1`, `S2`, denominator regularization, and source fusion.

## Files and symbols

- Extend `DifferentialOperators` with a reference CUDA kernel.
- Add CPU analytic Hessian-vector and `B` controls.

## Implementation specification

1. Differentiate only periodic fluctuations in the Hessian; affine parts have
   zero Hessian.
2. Use centered diagonal and mixed second derivatives.
3. Load the radius-one union stencil: center, six axial neighbors, and twelve
   edge-diagonal neighbors per field.
4. Form products and `B=H(psi2)g1-H(psi1)g2` in registers.

## Expected numerical effect

Directional curvature converges at second order without persistent Hessian
buffers.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Each Hessian-vector component has L2 order at least 1.8.
- `B` matches CPU reference within the measured discretization error.
- Analytic controls with parallel/constant gradients produce `B` at roundoff.

## Regression surface

- Mixed-derivative signs, periodic diagonal indexing, register pressure, and
  future source fusion.

## Failure and rollback policy

- Keep the unfused kernel and componentwise diagnostics if a fused register-only
  implementation obscures an error.
- Do not allocate full Hessian fields as a workaround.

## Completion checklist

<!-- completion-checklist:start -->
- [x] Direct Hessian-vector products are implemented without Hessian storage.
- [x] B construction and analytic controls pass.
- [x] Component convergence order is at least 1.8.
- [x] Temporary memory is measured and documented.
- [ ] Full regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-08 complete and selects SF-09.
<!-- completion-checklist:end -->

## Advancement rule

SF-09 may construct regularized nonlinear sources from the accepted `B` and
gradient definitions.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-06T02:33Z | active; master=`origin/master=d3c8ca796ea6cdeb21ddf6bd335d9f1e59575a69` | Activated SF-08 documentation state. | Preflight verified on the default-branch closure: clean master and origin/master at `d3c8ca796ea6cdeb21ddf6bd335d9f1e59575a69`; SF-07 is done; checker PASS (`29 increments, next=SF-08`); persistent Goal `Implementar productos Hessiano-vector y la construcción de B sin almacenar Hessianos.` created; branch `science/lester-sf08-hessian-vector-b`; worktree `~/src/MacroFlow3D/.agents/worktrees/lester-sf08-hessian-vector-b`. | Build and execute the SF-08 DAG. |
| 2026-08-06T02:57Z | measurement; `ccc42d2db5ef2b2dfd83d105d74a6e6757462438` | Measured the accepted SF-08 kernel resource and storage contract on local `sm_86` Debug. | `src/core/Scalar.hpp` fixes `sizeof(real)=8` bytes. Reproducible artifact: `cmake --preset wsl-debug`; `ninja -C build/wsl-debug -j 1 CMakeFiles/macroflow3d_lib.dir/src/physics/streamfunctions/DifferentialOperators.cu.o` (exit 0); `cuobjdump --dump-resource-usage build/wsl-debug/CMakeFiles/macroflow3d_lib.dir/src/physics/streamfunctions/DifferentialOperators.cu.o` (exit 0) reports the mangled `streamfunction_hessian_vector_b_kernel` in `sm_86` ELF with `REG:64 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:528`. Thus the kernel has no compiler-reported stack, local, or shared temporary storage; the 64 32-bit registers/thread are execution resources, not caller-visible persistent fields. Source audit (`rg -n 'cuda(Malloc|MallocAsync|Memcpy|MemcpyAsync|StreamSynchronize|DeviceSynchronize|Free)' src/physics/streamfunctions/DifferentialOperators.cu`) found none; the only CUDA runtime operation after launch is `cudaGetLastError`, so enqueue allocates, copies, and synchronizes no memory. Caller-owned storage is distinct: one `real` field is 32,768 B (0.000030517578 GiB) at 16^3 and 134,217,728 B (0.125 GiB) at 256^3; inputs (2 fluctuations + 6 total gradients = 8 fields) are 262,144 B (0.000244140625 GiB) / 1,073,741,824 B (1.000 GiB); nine diagnostic outputs (6 HVP + 3 B) are 294,912 B (0.000274658203 GiB) / 1,207,959,552 B (1.125 GiB); their 17-field combined caller allocation is 557,056 B (0.000518798828 GiB) / 2,281,701,376 B (2.125 GiB). The avoided materialization of two scalar Hessians is 12 fields if symmetry is stored (393,216 B / 0.000366210938 GiB; 1,610,612,736 B / 1.500 GiB) or 18 fields if all 3x3 entries are stored (589,824 B / 0.000549316406 GiB; 2,415,919,104 B / 2.250 GiB); these are avoided Hessian-component fields, not a claim that caller-owned diagnostic outputs disappear. Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, CC 8.6, 4096 MiB, driver 610.43.03, CUDA/nvcc/cuobjdump 13.3.73. Checker PASS (exit 0, `29 increments, next=SF-08`). Limitation: this is only Debug `sm_86`; it neither measures nor infers V100 resources. | Integrate the measurement evidence with the complete SF-08 validation report. |
| 2026-08-06T03:13Z | integration attempt; invalid evidence | Earlier T02/T03 build attempts were duplicated or interrupted and did not yield a real successful completion; the integration command first observed no linked test binary (exit 127), then a duplicate build was interrupted (exit 130). A direct positional test-case invocation was also invalid syntax (exit 2; the executable requires `--case`). None is used as validation evidence. | Preserve these failures as reproducibility evidence; use only the subsequent single focal build and correctly invoked commands below. |
| 2026-08-06T03:13Z | integration validation; `cb9865efcffbfdcd101b7f7eda175edfe7b43aef` | Independent integration cherry-picked T02 `cbdeb4404762603d2c74a05a2a5dfbca83a025dc`, T03 `859754f6f5b74c116feb0282465aff7c0e050572`, T04 `ccc42d2db5ef2b2dfd83d105d74a6e6757462438`, T05 `a23b96b7504148250ae1d9980a995d002cae3147`, and T06 `55dc86a1a2d0c4552935bdde1571dbcd9fb35f36` without conflict, yielding `e56b304`, `18ae84b`, `48d03d9`, `a63fd3b`, `cb9865e`; clean tree and `git diff --check master...HEAD` exit 0. Exact evidence: checker exit 0 (`29 increments, next=SF-08`); configure exit 0; single focal `ninja -C build/wsl-debug -j1 streamfunction_operator_tests; rc=$?; echo BUILD_EXIT=$rc; exit $rc` exit 0 (`BUILD_EXIT=0`); five correct `--case` runs exit 0; targeted CTest exit 0 (1/1, 7.71 s); full CTest exit 0 (2/2, 7.79 s); PSPTA-small smoke exit 0. GPU/CPU oracle worst normalized RMS `4.412206416377133e-16`, boundary Linf `3.126255864145324e-15` (threshold `5e-11`). All nine 16/32 and 32/64 L2 orders are >=1.8; minimum `1.894419699630846`, maximum `1.990902192137669`; all Linf errors strictly decrease. Pure-affine all HVP/B Linf `0` <= `3.552713678800501e-15`; parallel scale-2 `B` Linf `0` <= `1.020494255444177e-11` with HVP scale `359.0544959806348`. Contracts: 69 invalid span/grid/alias cases reject with `invalid_argument`; read-only input/input overlap is accepted and finite (Linf `74.43725987038972`); four test-only mutants exceed their explicit thresholds (0.13234, 0.23688, 0.30383, 1.05514). API audit confirms the 19-point periodic stencil, centered mixed signs, `B=H(psi2)g1-H(psi1)g2`, same-state total-gradient view, output/input anti-aliasing, and no enqueue allocation/copy/sync. Gate 1 PASS; Gate 2 PASS; SF-08 Gate 3A operator subset PASS. Coupled residual, reconstruction, invariance, divergence, denominator/gauge physical metrics and Gate 4/V100 are N/A for this pre-source, pre-coupled operator increment, not inferred. Hardware Debug sm86: RTX 3050 Laptop GPU, driver 610.43.03, CUDA 13.3.73; deterministic manufactured grids 16^3/32^3/64^3 (no random seed). Smoke legacy metrics: head 10 iters, `1.02e+01 -> 1.77e-13`; divergence min/max `-8.1175e-14`/`8.2979e-14`; particles active/exited `387/113`, nonzero/max failures `0/0`; elapsed 1.4 s. | Master audit and human-review preparation; do not mark accepted or advance NEXT. |
| 2026-08-06T03:31Z | root audit PASS; validating | Root audited `git diff master...7bd02f8` (10 files, 7 commits) for exact SF-08 scope: no future increment work, direct register-only HVPs/B, centered diagonal and mixed signs, periodic radius-one 19-point union stencil, total-gradient use with affine Hessian zero, output/input anti-aliasing, and no allocation/copy/synchronization in enqueue. Root reproduced checker PASS (`29 increments, next=SF-08`), configure, exact `cmake --build build/wsl-debug -j` exit 0, five `--case` executions exit 0, targeted/full CTest 1/1 and 2/2 exit 0, and smoke exit 0. Metrics: oracle `4.4122064e-16`, periodic-boundary Linf `3.1262559e-15`, minimum component order `1.8944197`, pure/parallel control zeros `0`, 69 rejects plus 1 accepted read-only overlap, four mutants detected, and resources `REG:64`, `STACK/LOCAL/SHARED:0`; caller diagnostic outputs are 1.125 GiB at 256^3. Gate 1, Gate 2, and Gate 3A operator subset PASS. Full coupled physical metrics/Gate 4/V100 are N/A because SF-08 is pre-source/pre-coupled and not a V100 increment. Earlier Ninja recovery warnings arose from invalid interrupted attempts already recorded; the final exact build is valid. Automatic classification PASS; implementation frozen pending required human review. | Open/update PR and request human review; do not advance NEXT. |
| 2026-08-06T03:23Z | awaiting_review; PR [#18](https://github.com/santi-esquerre/MacroFlow3D/pull/18); initial published head `9bf658fa0c8bd0885b1e040c760605fdeac8e7b7` | Published the frozen SF-08 implementation for mandatory human review. The PR records scope, API/kernel/reference/tests, exact exit-0 commands, oracle/boundary/order/zero/contract/mutant/resource/memory/smoke evidence, Gate 1/2/3A operator-subset PASS, and invalid attempts excluded from acceptance evidence. | Implementation remains frozen. Residual risks: the explicit nine caller-owned diagnostic outputs cost 1.125 GiB at 256^3; `REG:64` and zero stack/local/shared storage were measured only in local Debug `sm_86`, with no V100 inference. Await human review; do not advance dashboard, checklist, Goal, or NEXT. |
| 2026-08-06T03:25Z | metadata correction; append-only | The preceding root-audit row is labeled `2026-08-06T03:31Z`, which is inexact/future-dated and is not rewritten. Authoritative timestamps are the validating/root-audit commit `9bf658f` at `2026-08-06T03:20:38Z`, PR #18 creation at `2026-08-06T03:23:04Z`, and the awaiting-review commit `8f5f9ba` at `2026-08-06T03:23:28Z`; therefore the correct order is audit/validating → PR created → awaiting-review commit. | Metadata-only correction with zero numerical effect. PASS classification, all metrics, increment state, PR, NEXT, and persistent Goal are unchanged. |
