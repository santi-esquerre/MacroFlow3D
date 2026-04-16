#pragma once

/**
 * @file rhs_head.cuh
 * @brief Build RHS for head equation (Darcy flow)
 *
 * Legacy reference: RHS_head_3D.cu
 *
 * ## RHS Construction
 *
 * For the standard problem with no volumetric sources:
 *   RHS = 0  (interior cells)
 *
 * Dirichlet BC contributions (boundary cells):
 *   RHS[cell] -= 2 * K[cell] * H_bc / dx²
 *
 * Periodic and Neumann BCs do NOT contribute to the RHS.
 *
 * ## Note on Singular Systems
 *
 * This module only builds the RHS. For singular systems (all periodic/Neumann),
 * the pin mechanism is handled separately in the operator and smoother kernels.
 * See pin_spec.hpp for the pin documentation.
 */

#include "../../core/BCSpec.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"

namespace macroflow3d {
namespace physics {

/**
 * @brief Build RHS for head equation
 *
 * Constructs the right-hand side vector for the Darcy flow equation.
 *
 * Algorithm:
 *   1. Fill RHS with zeros (no volumetric sources)
 *   2. Add Dirichlet BC contributions: RHS -= 2*K*H_bc/dx²
 *
 * @param rhs   Output: right-hand side vector (device, size = num_cells)
 * @param K     Input: conductivity field (device, size = num_cells)
 * @param grid  Grid specification
 * @param bc    Boundary conditions
 * @param ctx   CUDA context
 */
void build_rhs_head(DeviceSpan<real> rhs, DeviceSpan<const real> K, const Grid3D& grid,
                    const BCSpec& bc, const CudaContext& ctx);

} // namespace physics
} // namespace macroflow3d
