#pragma once

#include "../core/DeviceSpan.cuh"
#include "../core/Grid3D.hpp"
#include "../runtime/CudaContext.cuh"
#include "mg_types.hpp"

namespace macroflow3d {
namespace multigrid {

/**
 * Coarsen a positive cell-centered coefficient with the geometric mean over
 * each 2x2x2 fine-cell block.
 */
void coarsen_coefficient_geometric(CudaContext& ctx, const Grid3D& coarse_grid,
                                   const Grid3D& fine_grid,
                                   DeviceSpan<const real> fine_coefficient,
                                   DeviceSpan<real> coarse_coefficient);

/**
 * Copy a finest-level positive coefficient and populate every coarser MG
 * level once. The hierarchy must be isotropic and geometrically 2:1 between
 * adjacent levels. Device values are assumed finite and strictly positive.
 */
void populate_coefficient_hierarchy(CudaContext& ctx, MGHierarchy& hierarchy,
                                    DeviceSpan<const real> fine_coefficient);

} // namespace multigrid
} // namespace macroflow3d
