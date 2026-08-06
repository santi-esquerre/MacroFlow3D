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
 * Read-only view of total, cell-centered streamfunction gradients produced
 * from the same fluctuation state by `enqueue_total_streamfunction_gradients`.
 * The affine contribution is already present in these six components.
 */
struct TotalStreamfunctionGradientView {
    DeviceSpan<const real> psi1_x;
    DeviceSpan<const real> psi1_y;
    DeviceSpan<const real> psi1_z;
    DeviceSpan<const real> psi2_x;
    DeviceSpan<const real> psi2_y;
    DeviceSpan<const real> psi2_z;
};

/**
 * Caller-owned components of the two directional Hessian products and their
 * difference `B = H(psi2) grad(psi1) - H(psi1) grad(psi2)`.
 *
 * These are diagnostic/reference outputs, not stored Hessian components.
 */
struct HessianVectorBOutput {
    DeviceSpan<real> hessian_psi2_times_grad_psi1_x;
    DeviceSpan<real> hessian_psi2_times_grad_psi1_y;
    DeviceSpan<real> hessian_psi2_times_grad_psi1_z;
    DeviceSpan<real> hessian_psi1_times_grad_psi2_x;
    DeviceSpan<real> hessian_psi1_times_grad_psi2_y;
    DeviceSpan<real> hessian_psi1_times_grad_psi2_z;
    DeviceSpan<real> b_x;
    DeviceSpan<real> b_y;
    DeviceSpan<real> b_z;
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

/**
 * Enqueue register-only periodic Hessian-vector products and their `B` field.
 *
 * The Hessians differentiate only the periodic fluctuation inputs. The affine
 * streamfunction parts are represented solely by `total_gradients`, and hence
 * correctly contribute zero Hessian. At each cell the kernel loads the
 * radius-one 19-point union stencil (center, six axial, twelve edge-diagonal
 * neighbors) for each fluctuation, uses centered diagonal and mixed
 * derivatives with independent spacings, forms both Hessian-vector products
 * in registers, then writes their difference. No Hessian field is allocated
 * or materialized.
 *
 * All two fluctuation inputs, six total-gradient inputs, and nine outputs
 * must be non-null device spans of exactly `grid.num_cells()` elements.
 * Outputs must not overlap one another or any input; input/input overlap is
 * permitted because every input is read-only. Grid extents and spacings must
 * be finite and strictly positive. The total gradients must correspond to
 * these fluctuations at the same state.
 *
 * Every span must remain alive and unchanged until prior work on
 * `ctx.cuda_stream()` completes. This enqueue is asynchronous and performs no
 * allocation, transfer, or host synchronization.
 */
void enqueue_streamfunction_hessian_vector_b(
    CudaContext& ctx, const Grid3D& grid,
    const PeriodicStreamfunctionFluctuations& fluctuations,
    const TotalStreamfunctionGradientView& total_gradients,
    const HessianVectorBOutput& output);

} // namespace streamfunctions
} // namespace macroflow3d
