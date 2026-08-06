# SF-08 — Hessian-vector products and B

- State: `active`
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
- PR: `not opened`
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
- [ ] Direct Hessian-vector products are implemented without Hessian storage.
- [ ] B construction and analytic controls pass.
- [ ] Component convergence order is at least 1.8.
- [ ] Temporary memory is measured and documented.
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
