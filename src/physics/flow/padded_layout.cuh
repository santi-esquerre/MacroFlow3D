#pragma once

/**
 * @file padded_layout.cuh
 * @brief Shared device helper for PaddedFaceField indexing
 * @ingroup physics_flow
 *
 * Provides the canonical padded_idx function used by both
 * velocity_from_head.cu and velocity_diagnostics.cu.
 *
 * Layout: merge_id(ix,iy,iz) = iz*(ny+1)*(nx+1) + iy*(nx+1) + ix
 * Buffer size per component: (nx+1)*(ny+1)*(nz+1)
 *
 * Returns size_t to avoid overflow on large grids where
 * (nx+1)*(ny+1)*(nz+1) exceeds INT_MAX (e.g. 1600^3 → ~4e9 > 2^31).
 */

#include <cstddef>
#include <cuda_runtime.h>

namespace macroflow3d {
namespace physics {

/**
 * @brief Linear index into a padded face-field component.
 *
 * Compatible with Par2_Core's FaceFieldView / VelocityView.
 * All three velocity components (U, V, W) share this index space.
 */
__host__ __device__ __forceinline__ size_t padded_idx(int ix, int iy, int iz, int nx, int ny) {
    return static_cast<size_t>(iz) * static_cast<size_t>(ny + 1) * static_cast<size_t>(nx + 1) +
           static_cast<size_t>(iy) * static_cast<size_t>(nx + 1) + static_cast<size_t>(ix);
}

} // namespace physics
} // namespace macroflow3d
