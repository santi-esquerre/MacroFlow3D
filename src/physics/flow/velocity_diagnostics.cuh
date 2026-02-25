#pragma once

/**
 * @file velocity_diagnostics.cuh
 * @brief GPU diagnostics for the velocity field: divergence, vorticity, helicity
 * @ingroup physics_flow
 *
 * Computes cell-centered diagnostic quantities from a PaddedVelocityField:
 *   - Divergence  div(i,j,k) using natural MAC face differences
 *   - Vorticity   |omega(i,j,k)| via centered derivatives of face-averaged velocities
 *   - Helicity    h(i,j,k) = v_center · omega
 *
 * Evaluation domain: cell centers (i,j,k), i in [0,nx-1], j in [0,ny-1], k in [0,nz-1].
 * Curl/helicity interior-only (i in [1,nx-2], ...) to avoid padding contamination.
 *
 * All outputs are cell-centered ScalarFields of size nx*ny*nz.
 * Statistics (min/max/mean) are printed to stdout via compute_field_stats.
 */

#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../common/fields.cuh"

namespace macroflow3d {
namespace physics {

// ============================================================================
// Diagnostic output container
// ============================================================================

/**
 * @brief Cell-centered diagnostic fields computed from velocity.
 *
 * All fields have dimensions (nx, ny, nz) and linear index i + nx*(j + ny*k).
 */
struct VelocityDiagnostics {
    ScalarField divergence;   ///< div(v) at cell centers
    ScalarField vorticity_mag; ///< |curl(v)| at cell centers
    ScalarField helicity;     ///< v_center · curl(v) at cell centers

    VelocityDiagnostics() = default;

    explicit VelocityDiagnostics(const Grid3D& grid)
        : divergence(grid)
        , vorticity_mag(grid)
        , helicity(grid)
    {}

    void resize(const Grid3D& grid) {
        divergence.resize(grid);
        vorticity_mag.resize(grid);
        helicity.resize(grid);
    }

    bool empty() const { return divergence.empty(); }
};

// ============================================================================
// Main API
// ============================================================================

/**
 * @brief Compute velocity diagnostics (divergence, vorticity magnitude, helicity).
 *
 * One kernel launch over all cell centers. Divergence is computed for every
 * cell; curl and helicity are computed only in the interior
 * (i in [1,nx-2], j in [1,ny-2], k in [1,nz-2]) and set to 0 at boundaries.
 *
 * If any grid dimension < 3, curl/helicity are skipped (zeroed) and a
 * warning is printed.
 *
 * @param vel    Input padded velocity field (read-only).
 * @param diag   Output diagnostics (must be pre-allocated to grid size).
 * @param grid   Grid specification (nx, ny, nz, dx, dy, dz).
 * @param ctx    CUDA context (stream).
 */
void compute_velocity_diagnostics(
    const PaddedVelocityField& vel,
    VelocityDiagnostics&       diag,
    const Grid3D&              grid,
    const CudaContext&         ctx);

/**
 * @brief Print velocity diagnostic statistics to stdout.
 *
 * Uses compute_field_stats for each field and prints one line per quantity:
 *   [diag] r=<rid> div:  min=... max=... mean=...
 *   [diag] r=<rid> |ω|:  min=... max=... mean=...
 *   [diag] r=<rid> h:    min=... max=... mean=...
 *
 * @param diag            Diagnostic fields (device memory).
 * @param realization_id  Realization index (for labeling).
 * @param ctx             CUDA context (stream).
 */
void print_velocity_diagnostics(
    const VelocityDiagnostics& diag,
    int                        realization_id,
    const CudaContext&         ctx);

} // namespace physics
} // namespace macroflow3d
