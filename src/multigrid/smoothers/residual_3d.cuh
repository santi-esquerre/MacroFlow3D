#pragma once

#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/BCSpec.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../../numerics/pin_spec.hpp"

namespace macroflow3d {
namespace multigrid {

/**
 * @brief Compute residual: r = b - A*x
 * 
 * A is 7-point stencil with variable coefficients K (conductivity).
 * Uses harmonic mean for face conductivities.
 * 
 * If pin is enabled:
 *   r[pin.index] = b[pin.index] - x[pin.index]  (identity row residual)
 * 
 * @param ctx   CUDA context
 * @param grid  Grid specification
 * @param x     Current solution
 * @param b     Right-hand side
 * @param K     Conductivity field
 * @param r     Output residual
 * @param bc    Boundary conditions
 * @param pin   Pin specification (optional, default = no pin)
 */
void compute_residual_3d(
    CudaContext& ctx,
    const Grid3D& grid,
    DeviceSpan<const real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    DeviceSpan<real> r,
    const BCSpec& bc,
    PinSpec pin = {}
);

} // namespace multigrid
} // namespace macroflow3d
