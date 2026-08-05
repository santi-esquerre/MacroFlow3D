#pragma once

/**
 * @file face_coefficient.cuh
 * @brief Shared cell-centered positive coefficient face policy.
 */

#include "../../core/Scalar.hpp"

namespace macroflow3d {
namespace operators {

/**
 * Harmonic face coefficient for a positive cell-centered coefficient: legacy
 * conductivity K or Lester coefficient q.
 */
__host__ __device__ inline real harmonic_mean_positive_cell_coefficient(real center,
                                                                         real neighbor) {
    return 2.0 / (1.0 / center + 1.0 / neighbor);
}

} // namespace operators
} // namespace macroflow3d
