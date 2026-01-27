#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "reduction_workspace.cuh"

namespace rwpt {
namespace blas {

// L2 norm (device result, no sync)
void nrm2_device(CudaContext& ctx, 
                 DeviceSpan<const real> x, 
                 DeviceSpan<real> d_result, 
                 ReductionWorkspace& ws);

// L2 norm (host result, includes synchronization)
// WARNING: Use only for debugging/reporting, not in hot loops
real nrm2_host(CudaContext& ctx, 
               DeviceSpan<const real> x, 
               ReductionWorkspace& ws);

} // namespace blas
} // namespace rwpt
