# Performance Contract
> Auto-generated from Etapa 9 audit. Keep in sync with code changes.

## Sync & Alloc Map

### Sync Points

| Location | Type | Path | Why | Cadence |
|----------|------|------|-----|---------|
| `PipelineRunner.cu:364` | `cudaStreamSynchronize` | HOT (gated) | Wait for async stats reduction + unwrapped compute before host-side read | Every `sample_every` steps (typ. 10) |
| `IOScheduler.hpp:166` | `cudaStreamSynchronize` | WARM | Wait for D2H staging of snapshot pinned buffers before disk write | Every `snapshot_every` steps (typ. 200+) |
| `StageProfiler.cuh:106` | `cudaEventSynchronize` | COLD | Measure elapsed time between CUDA events (profiler.stop()) | Once per stage per realization; compiled out when `RWPT_ENABLE_PROFILING=0` |
| `CudaContext.cu:43` | `cudaStreamSynchronize` | COLD | Explicit `ctx.synchronize()` — called at engine.synchronize() after hot loop | Once per realization |
| `diagnostics.cuh:110` | `cudaStreamSynchronize` | COLD | Debug sync check | Only when `RWPT_ENABLE_SYNC_CHECKS=1` |
| `cg.cuh:108-134` | `cudaMemcpyAsync + sync` | WARM | CG convergence check (D2H `is_valid` + `rr_new`) | Every `check_every` CG iters (typ. 10) |
| `velocity_from_head.cu:910-1044` | `cudaMemcpy` (sync) | COLD | Diagnostic functions (`compute_norm`, `check_no_nans`, `compute_sum`) | Only when `verify_velocity=true` or diagnostics enabled |

### D2H Transfers

| Location | Type | Path | Why |
|----------|------|------|-----|
| `PinnedHostBuffer.hpp:67` | `cudaMemcpyAsync` D2H | WARM | Stage particle positions to pinned host buffer for snapshots |
| `cg.cuh:108` | `cudaMemcpyAsync` D2H | WARM | Check convergence flag every `check_every` CG iterations |
| `cg.cuh:133` | `cudaMemcpyAsync` D2H | WARM | Read `rr_new_host` for convergence test |
| `stochastic.cu:429-434` | `cudaMemcpyAsync` D2H ×3 | COLD | K field statistics (min/max/sum), once per realization |

### H2D Transfers

| Location | Type | Path | Why |
|----------|------|------|-----|
| `dot.cu:17` | `cudaMemcpyAsync` H2D | HOT | Zero `d_result` before cublasDdot (POINTER_MODE_DEVICE) |
| `nrm2.cu:16` | `cudaMemcpyAsync` H2D | HOT | Zero `d_result` before cublasDnrm2 |

### Allocations

All allocations occur in **cold paths** (constructors, `allocate()`, setup):

| Location | Type | Path | Why |
|----------|------|------|-----|
| `DeviceBuffer.cuh:19,83` | `cudaMalloc` | COLD | Device buffer constructor / `resize()` grow |
| `PinnedHostBuffer.hpp:58` | `cudaHostAlloc` | COLD | One-time snapshot staging allocation |
| `fields.cuh:60-65,134-136,267-269` | `DeviceBuffer::resize` | COLD | ScalarField / VelocityField construction |
| `workspaces.cuh:69-80,137-140` | `DeviceBuffer::resize` | COLD | Stochastic + Flow workspace init |
| `mg_types.hpp:42-68` | `DeviceBuffer::resize` | COLD | MG hierarchy setup |
| `IOScheduler.hpp:111` | `vector::reserve(4096)` | COLD | Pre-allocate stats series capacity |

**Zero hot-loop allocations confirmed.**

### cuBLAS

| Call | Sync mode | Notes |
|------|-----------|-------|
| `cublasDdot` (dot.cu) | `POINTER_MODE_DEVICE` | No implicit host sync; result stays on device |
| `cublasDnrm2` (nrm2.cu) | `POINTER_MODE_DEVICE` | No implicit host sync |
| `cublasCreate/Destroy` | N/A | Cold init/teardown only |

### Known Tech-Debt (post-release)

1. **CG convergence D2H**: `cg.cuh:108-134` does `cudaMemcpyAsync` + stream sync every `check_every` iterations. Could be replaced by a device-side convergence flag read via mapped memory or persistent kernel.
2. **dot/nrm2 H2D zero**: The per-call H2D zero of `d_result` in `dot.cu:17` / `nrm2.cu:16` adds ~1μs per BLAS call. Could be moved to a persistent workspace zero at CG start.
3. **Sync `cudaMemcpy` in diagnostics**: `velocity_from_head.cu:910-1044` uses synchronous `cudaMemcpy`; acceptable for diagnostics but would block if ever called in hot path.
4. **StageProfiler `cudaEventSynchronize`**: When profiling is ON, `profiler.stop()` blocks. This is expected but prevents overlap of stages. Consider deferred report in future.
