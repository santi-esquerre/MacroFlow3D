#pragma once

/**
 * @file velocity_from_head.cuh
 * @brief Compute staggered velocity field from cell-centered head and conductivity
 * @ingroup physics_flow
 *
 * Physics: Darcy's law with harmonic mean conductivity
 *   q = -K_eff * grad(H)
 *
 * where K_eff between two cells is the harmonic mean:
 *   K_eff = 2 / (1/K_a + 1/K_b) = 2*K_a*K_b / (K_a + K_b)
 *
 * Layout (standard staggered grid):
 *   U: x-velocity at x-faces, dims (nx+1, ny, nz)
 *   V: y-velocity at y-faces, dims (nx, ny+1, nz)
 *   W: z-velocity at z-faces, dims (nx, ny, nz+1)
 *
 * Boundary conditions:
 *   - Neumann homogeneous: flux = 0 at boundary face
 *   - Dirichlet: one-sided gradient with distance h/2, using K_cell (not harmonic)
 *   - Periodic: wrap to opposite side, use harmonic mean as interior
 *
 * Reference: legacy/compute_velocity_from_head_for_par2.cu
 * Semantics: identical to legacy, but with clean modular implementation
 */

#include "../../core/BCSpec.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../common/fields.cuh"

namespace macroflow3d {
namespace physics {

// ============================================================================
// Main API
// ============================================================================

/**
 * @brief Compute velocity field from head using Darcy's law
 *
 * Writes U, V, W in-place (already allocated in vel).
 * No memory allocation inside this function.
 *
 * @param vel       Output velocity field (U, V, W already allocated)
 * @param head      Input head field (cell-centered)
 * @param K         Input conductivity field (cell-centered)
 * @param grid      Grid dimensions and spacing
 * @param bc        Boundary conditions for all 6 faces
 * @param ctx       CUDA context for synchronization
 */
void compute_velocity_from_head(VelocityField& vel, const HeadField& head, const KField& K,
                                const Grid3D& grid, const BCSpec& bc, CudaContext& ctx);

/**
 * @brief Compute velocity field in padded facefield layout
 *
 * Same physics (Darcy + harmonic mean) but writes each component
 * into an (nx+1)*(ny+1)*(nz+1) array using merge_id indexing,
 * compatible with Par2_Core's VelocityView / FaceFieldView.
 * No memory allocation inside this function.
 *
 * @param vel       Output padded velocity field (U, V, W already allocated)
 * @param head      Input head field (cell-centered)
 * @param K         Input conductivity field (cell-centered)
 * @param grid      Grid dimensions and spacing
 * @param bc        Boundary conditions for all 6 faces
 * @param ctx       CUDA context for synchronization
 */
void compute_velocity_from_head(PaddedVelocityField& vel, const HeadField& head, const KField& K,
                                const Grid3D& grid, const BCSpec& bc, CudaContext& ctx);

// ============================================================================
// Checksum/validation utilities
// ============================================================================

/**
 * @brief Compute L2 norm (sqrt of sum of squares) for a device array
 */
real compute_norm2(DeviceSpan<const real> data, CudaContext& ctx);

/**
 * @brief Check if any values are NaN in a device array
 * @return true if no NaNs found (all values are valid)
 */
bool check_no_nans(DeviceSpan<const real> data, CudaContext& ctx);

/**
 * @brief Compute sum of all elements in a device array
 */
real compute_sum(DeviceSpan<const real> data, CudaContext& ctx);

/**
 * @brief Compute mean of all elements in a device array
 */
real compute_mean(DeviceSpan<const real> data, CudaContext& ctx);

/**
 * @brief Compute and print checksums for velocity field
 *
 * Prints L2 norms and NaN status for U, V, W
 */
void print_velocity_checksums(const VelocityField& vel, CudaContext& ctx);

/**
 * @brief Verify mean U velocity against theoretical Darcy value
 *
 * For Dirichlet west-east with periodic elsewhere:
 *   u_theory = K_eff * (H_west - H_east) / Lx
 *
 * where K_eff is the effective (harmonic) conductivity in x-direction.
 *
 * This function computes:
 *   1. Mean U velocity from the computed field
 *   2. Theoretical velocity using harmonic mean K
 *   3. Relative error
 *
 * @param vel       Computed velocity field
 * @param K         Conductivity field
 * @param grid      Grid dimensions
 * @param bc        Boundary conditions
 * @param ctx       CUDA context
 */
void verify_mean_velocity_darcy(const VelocityField& vel, const KField& K, const Grid3D& grid,
                                const BCSpec& bc, CudaContext& ctx);

} // namespace physics
} // namespace macroflow3d
