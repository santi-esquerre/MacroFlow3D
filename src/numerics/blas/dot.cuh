#pragma once

#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"
#include "reduction_workspace.cuh"

namespace macroflow3d {
namespace blas {

// Dot product (device result, no sync)
void dot_device(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<const real> y,
                DeviceSpan<real> d_result, ReductionWorkspace& ws);

// Dot product (host result, includes synchronization)
// WARNING: Use only for debugging/reporting, not in hot loops
real dot_host(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<const real> y,
              ReductionWorkspace& ws);

} // namespace blas
} // namespace macroflow3d
