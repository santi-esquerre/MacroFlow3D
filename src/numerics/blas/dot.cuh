#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "reduction_workspace.cuh"

namespace rwpt {
namespace blas {

// Dot product (device result, no sync)
void dot_device(CudaContext& ctx, 
                DeviceSpan<const real> x, 
                DeviceSpan<const real> y, 
                DeviceSpan<real> d_result, 
                ReductionWorkspace& ws);

// Dot product (host result, includes synchronization)
// WARNING: Use only for debugging/reporting, not in hot loops
real dot_host(CudaContext& ctx, 
              DeviceSpan<const real> x, 
              DeviceSpan<const real> y, 
              ReductionWorkspace& ws);

} // namespace blas
} // namespace rwpt
