#pragma once

/**
 * @file coarsen_K.cuh
 * @brief Coarsen (homogenize) conductivity field K for multigrid levels
 *
 * This file provides the physics-specific coarsening of K for Darcy flow.
 * Uses geometric mean homogenization (legacy: CompactHomogenizationKtensor).
 *
 * ## Location Rationale
 *
 * Although this is used by the multigrid hierarchy, it lives in physics/flow/
 * because:
 * 1. The coarsening method is physics-specific (geometric mean for log-normal K)
 * 2. Other physics problems might require different coarsening strategies
 * 3. It's tightly coupled with the flow equation's coefficient structure
 *
 * The generic MG transfer operators (restrict, prolong) are in multigrid/transfer/.
 */

#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../runtime/CudaContext.cuh"

namespace macroflow3d {
namespace physics {

/**
 * @brief Coarsen conductivity field K from fine to coarse level.
 *
 * Uses geometric mean over 2x2x2 blocks of fine cells, matching legacy
 * CompactHomogenizationKtensor implementation.
 *
 * For log-normal K fields:
 *   K_coarse = exp(mean(log(K_fine))) = (K₁·K₂·...·K₈)^(1/8)
 *
 * This preserves the harmonic mean behavior needed for Darcy flow
 * with heterogeneous conductivity.
 *
 * @param ctx CUDA context
 * @param coarse_grid Coarse grid dimensions (must be fine_grid/2)
 * @param fine_grid Fine grid dimensions
 * @param K_fine Input fine conductivity field
 * @param K_coarse Output coarse conductivity field
 */
void coarsen_K(CudaContext& ctx, const Grid3D& coarse_grid, const Grid3D& fine_grid,
               DeviceSpan<const real> K_fine, DeviceSpan<real> K_coarse);

} // namespace physics
} // namespace macroflow3d
