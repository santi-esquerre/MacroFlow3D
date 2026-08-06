#pragma once

/**
 * @file DifferentialOperators.cuh
 * @brief Cell-centered differential operators for periodic streamfunctions.
 */

#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../runtime/CudaContext.cuh"
#include "affine_gauge.cuh"

namespace macroflow3d {
namespace streamfunctions {

/**
 * Caller-owned device buffers for the total, cell-centered gradients of both
 * streamfunctions. The components use the x-fastest layout
 * `i + nx * (j + ny * k)` of Grid3D.
 */
struct TotalStreamfunctionGradientOutput {
    DeviceSpan<real> psi1_x;
    DeviceSpan<real> psi1_y;
    DeviceSpan<real> psi1_z;
    DeviceSpan<real> psi2_x;
    DeviceSpan<real> psi2_y;
    DeviceSpan<real> psi2_z;
};

/**
 * Enqueue total gradients of the two affine-periodic streamfunctions.
 *
 * At every cell, this applies a second-order centered difference to each
 * periodic fluctuation with independent periodic wrapping in x, y, and z,
 * then adds the matching constant component of `gauge`. No affine scalar ramp
 * is stored or differentiated. Directional spacings are used independently.
 *
 * All spans must reference device-resident, non-null storage with exactly
 * `grid.num_cells()` elements. The six outputs must not overlap each other or
 * either fluctuation input; `u1` and `u2` may overlap because they are read
 * only. Grid extents and spacings must be finite and strictly positive, and
 * affine-gradient components must be finite. These preconditions are checked
 * on the host before enqueueing work.
 *
 * The inputs and outputs must remain alive and unchanged until work previously
 * queued on `ctx.cuda_stream()` has completed. This function performs no
 * allocation, device-to-host transfer, or host synchronization.
 */
void enqueue_total_streamfunction_gradients(
    CudaContext& ctx, const Grid3D& grid,
    const PeriodicStreamfunctionFluctuations& fluctuations, const AffineGauge& gauge,
    const TotalStreamfunctionGradientOutput& output);

} // namespace streamfunctions
} // namespace macroflow3d
