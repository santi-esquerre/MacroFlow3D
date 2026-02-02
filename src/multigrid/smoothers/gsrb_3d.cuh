#pragma once

#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/BCSpec.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../../numerics/pin_spec.hpp"

namespace rwpt {
namespace multigrid {

/**
 * @brief Gauss-Seidel Red-Black smoother
 * 
 * Solves A*x = b using red-black ordering:
 *   Red cells: (i+j+k) % 2 == 0
 *   Black cells: (i+j+k) % 2 == 1
 * 
 * @param ctx       CUDA context
 * @param grid      Grid specification
 * @param x         Solution (in/out)
 * @param b         Right-hand side
 * @param K         Conductivity field
 * @param num_iters Number of iterations (each = red + black sweep)
 * @param bc        Boundary conditions
 * @param pin       Pin specification (optional, default = no pin)
 */
void gsrb_smooth_3d(
    CudaContext& ctx,
    const Grid3D& grid,
    DeviceSpan<real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    int num_iters,
    const BCSpec& bc,
    PinSpec pin = {}
);

} // namespace multigrid
} // namespace rwpt
