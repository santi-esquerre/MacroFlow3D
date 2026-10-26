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

} // namespace physics
} // namespace macroflow3d
