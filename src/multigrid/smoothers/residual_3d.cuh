#pragma once

#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/BCSpec.hpp"
#include "../../runtime/CudaContext.cuh"

namespace rwpt {
namespace multigrid {

// Compute residual: r = b - A*x
// A is 7-point stencil with variable coefficients K (conductivity)
// Uses harmonic mean for face conductivities
void compute_residual_3d(
    CudaContext& ctx,
    const Grid3D& grid,
    DeviceSpan<const real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    DeviceSpan<real> r,
    const BCSpec& bc
);

} // namespace multigrid
} // namespace rwpt
