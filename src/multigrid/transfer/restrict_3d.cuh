#pragma once

#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../runtime/CudaContext.cuh"

namespace macroflow3d {
namespace multigrid {

// Restriction: fine grid -> coarse grid (averaging)
// phiCoarse[I,J,K] = (1/8) * sum of 8 fine cells that cover coarse cell [I,J,K]
// Fine grid has 2*NX, 2*NY, 2*NZ cells
// Coarse grid has NX, NY, NZ cells
void restrict_3d(
    CudaContext& ctx,
    const Grid3D& fine_grid,
    const Grid3D& coarse_grid,
    DeviceSpan<const real> r_fine,
    DeviceSpan<real> b_coarse
);

} // namespace multigrid
} // namespace macroflow3d
