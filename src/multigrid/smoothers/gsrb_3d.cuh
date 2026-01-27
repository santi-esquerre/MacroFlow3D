#pragma once

#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/BCSpec.hpp"
#include "../../runtime/CudaContext.cuh"

namespace rwpt {
namespace multigrid {

// Gauss-Seidel Red-Black smoother
// Solves A*x = b using red-black ordering
// Red cells: (i+j+k) % 2 == 0
// Black cells: (i+j+k) % 2 == 1
void gsrb_smooth_3d(
    CudaContext& ctx,
    const Grid3D& grid,
    DeviceSpan<real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    int num_iters,
    const BCSpec& bc
);

} // namespace multigrid
} // namespace rwpt
