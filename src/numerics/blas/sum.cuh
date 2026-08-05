#pragma once

#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"
#include "reduction_workspace.cuh"

#include <cstddef>

namespace macroflow3d {
namespace blas {

// Query the caller-owned CUB storage needed to sum n double-precision values.
// This does not allocate or synchronize. n must fit the current BLAS-size
// contract (INT_MAX); an empty sum requires no temporary storage.
std::size_t sum_workspace_bytes(std::size_t n);

// The only preparation API allowed to resize a ReductionWorkspace for sum.
// It records an exact input size, so sum_device never allocates or re-queries.
void prepare_sum_workspace(std::size_t n, ReductionWorkspace& ws);

// True only when ws was prepared for exactly n values and still owns all
// required storage. This check itself does not invoke CUB or allocate.
bool sum_workspace_ready(std::size_t n, const ReductionWorkspace& ws) noexcept;

// Enqueue a CUB double-precision sum on ctx.cuda_stream(). d_result[0] is
// written on the device; it must not alias x. The workspace must have been
// prepared for exactly x.size(). This function never allocates or synchronizes.
void sum_device(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> d_result,
                ReductionWorkspace& ws);

// Diagnostic-only host result. It uses a preprepared workspace and explicitly
// synchronizes ctx after the asynchronous device-to-host copy. It never
// allocates or prepares storage.
real sum_host(CudaContext& ctx, DeviceSpan<const real> x, ReductionWorkspace& ws);

} // namespace blas
} // namespace macroflow3d
