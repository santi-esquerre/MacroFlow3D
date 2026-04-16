#pragma once

#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../runtime/CudaContext.cuh"

namespace macroflow3d {
namespace multigrid {

// Prolongation: coarse grid -> fine grid (injection + addition)
// For each fine cell, copy value from corresponding coarse cell
// phiFine[i,j,k] += phiCoarse[i/2, j/2, k/2]
// This is "additive" prolongation (used for error correction)
void prolong_3d_add(CudaContext& ctx, const Grid3D& coarse_grid, const Grid3D& fine_grid,
                    DeviceSpan<const real> x_coarse, DeviceSpan<real> x_fine);

} // namespace multigrid
} // namespace macroflow3d
