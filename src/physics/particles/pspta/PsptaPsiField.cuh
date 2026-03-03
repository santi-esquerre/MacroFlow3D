#pragma once

/**
 * @file PsptaPsiField.cuh
 * @brief GPU-owned streamline invariant fields ψ1, ψ2 for PSPTA transport.
 *
 * Stores two cell-centered scalar fields ψ1(x,y,z) and ψ2(x,y,z) satisfying
 * v·∇ψi = 0 along streamlines of the Darcy velocity field (helicity-free 3D).
 *
 * @par Index convention
 * Cell-centered 3D arrays use x-fastest (column-major) order:
 *   idx(i,j,k) = i + nx*(j + ny*k)
 * consistent with Grid3D::idx.
 *
 * @par Storage type
 * ψ1 and ψ2 are stored as float32 to reduce memory footprint.
 * All intermediate arithmetic is performed in double.
 *
 * @par Level A precompute
 * precompute_levelA() builds ψ by marching in x using a semi-Lagrangian
 * backtrace on the CompactMAC (VelocityField) velocity.
 * Periodic bilinear interpolation uses "periodic lifting" to avoid
 * artefacts at the y/z periodic boundaries.
 *
 * @note This module is part of the PSPTA (pseudo-symplectic particle tracking
 * algorithm) implementation.  The engine (PsptaEngine) uses this field to
 * enforce ψ-invariance per particle via Newton projection.
 *
 * @ingroup physics_particles_pspta
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/Grid3D.hpp"
#include "../../common/fields.cuh"   // VelocityField (CompactMAC / real = double)
#include <cuda_runtime.h>
#include <cstdint>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// Report returned by precompute_levelA
// ============================================================================

/**
 * @brief Diagnostic counters returned by PsptaPsiField::precompute_levelA().
 *
 * n_vx_clamped: number of (i,j,k) cells (across all marching steps) where
 *               the x-component of the midpoint velocity was <= eps_vx and
 *               was clamped to eps_vx to avoid division by zero.  A large
 *               fraction of n_total = (nx-1)*ny*nz indicates the velocity
 *               field has many stagnation regions; PSPTA accuracy may degrade.
 */
struct PsptaPrecomputeReport {
    long long n_vx_clamped = 0;  ///< cells where vx_avg was clamped to eps_vx
    long long n_total      = 0;  ///< total advance-plane cells = (nx-1)*ny*nz
};

/**
 * @brief v⋅∇ψ residual norms computed by PsptaPsiField::compute_psi_quality().
 *
 * For each cell c = (i,j,k):
 *   r1(c) = v_cc(c) · ∇ψ1(c)
 *   r2(c) = v_cc(c) · ∇ψ2(c)
 * where v_cc is the cell-centered velocity (average of adjacent CompactMAC faces)
 * and ∇ψ is computed via central FDs (one-sided at x-boundary, periodic-lifted in y,z).
 *
 * RMS = sqrt(mean(r^2)),  max = max|r|  over all nx*ny*nz cells.
 */
struct PsiQualityReport {
    double rms_r1  = 0.0;  ///< RMS of v·∇ψ1
    double max_r1  = 0.0;  ///< max |v·∇ψ1|
    double rms_r2  = 0.0;  ///< RMS of v·∇ψ2
    double max_r2  = 0.0;  ///< max |v·≧ψ2|
    long long n_cells = 0; ///< number of cells evaluated
};

// ============================================================================
// PsptaPsiField
// ============================================================================

/**
 * @brief Streamline invariant fields ψ1, ψ2 stored on the GPU.
 *
 * Usage:
 * @code
 *   PsptaPsiField psi;
 *   psi.resize(grid);
 *   auto report = psi.precompute_levelA(vel, grid, stream);
 *   // psi.psi1_ptr(), psi.psi2_ptr() now hold valid float32 arrays
 * @endcode
 */
struct PsptaPsiField {

    // ── Grid metadata ───────────────────────────────────────────────────────
    int    nx = 0, ny = 0, nz = 0;
    double dx = 0.0, dy = 0.0, dz = 0.0;

    // ── Storage ─────────────────────────────────────────────────────────────
    DeviceBuffer<float> psi1;  ///< ψ1 cell-centered, size nx*ny*nz, float32
    DeviceBuffer<float> psi2;  ///< ψ2 cell-centered, size nx*ny*nz, float32

    // ── Lifetime ────────────────────────────────────────────────────────────
    PsptaPsiField() = default;

    /**
     * @brief Resize buffers to match grid.  Only reallocates if larger.
     * Safe to call multiple times; idempotent when grid is unchanged.
     */
    void resize(const Grid3D& grid);

    // ── Accessors ───────────────────────────────────────────────────────────

    /// Device pointer to ψ1 (float32, size = nx*ny*nz).
    float*       psi1_ptr()       { return psi1.data(); }
    const float* psi1_ptr() const { return psi1.data(); }

    /// Device pointer to ψ2 (float32, size = nx*ny*nz).
    float*       psi2_ptr()       { return psi2.data(); }
    const float* psi2_ptr() const { return psi2.data(); }

    /// Total number of cell-centered values.
    size_t size() const { return static_cast<size_t>(nx) * ny * nz; }

    /// Base grid.
    Grid3D grid() const {
        return Grid3D(nx, ny, nz,
                      static_cast<real>(dx),
                      static_cast<real>(dy),
                      static_cast<real>(dz));
    }

    // ── Level A precompute ──────────────────────────────────────────────────

    /**
     * @brief Build ψ1, ψ2 by marching in x (semi-Lagrangian Level A).
     *
     * Algorithm:
     *   - Inlet (i=0):  ψ1(0,j,k) = (j+0.5)*dy,  ψ2(0,j,k) = (k+0.5)*dz
     *   - Advance i→i+1: for each (j,k) in destination plane:
     *       1. Reconstruct cell-center velocity at (i,j,k) and (i+1,j,k) from
     *          CompactMAC faces; average to get midpoint estimate v_avg.
     *       2. Compute uy = vy_avg / max(vx_avg, eps_vx),
     *                  uz = vz_avg / max(vx_avg, eps_vx)
     *       3. Backtrace: y* = y - uy*dx,  z* = z - uz*dx
     *       4. Wrap y*, z* periodically into [0, Ly) × [0, Lz)
     *       5. Bilinear sample ψ on plane i with periodic lifting (see note)
     *       6. Wrap interpolated result back into [0, L) and store as float.
     *
     * @par Periodic lifting (critical for correctness)
     * When two adjacent cells straddle the periodic seam (e.g., j=0 and
     * j=ny-1), their raw ψ values may differ by ~L even though the true
     * invariant is continuous.  Lifting shifts the three non-reference corners
     * by the nearest multiple of L to the reference corner before averaging.
     * This is equivalent to computing the average in the lifted/unwrapped
     * coordinate space.
     *
     * @param vel      CompactMAC velocity field (VelocityField, real=double).
     * @param grid     Grid metadata.  Must match vel dimensions.
     * @param stream   CUDA stream. All kernels are launched on this stream.
     *                 The function synchronizes the stream internally to read
     *                 the diagnostic counter before returning.
     * @param eps_vx   Minimum clamped x-velocity to avoid division by zero.
     *                 Default 1e-10 (velocity units).
     *
     * @return PsptaPrecomputeReport with n_vx_clamped and n_total.
     */
    PsptaPrecomputeReport precompute_levelA(
        const VelocityField& vel,
        const Grid3D&        grid,
        cudaStream_t         stream,
        double               eps_vx = 1e-10
    );

    // ── Diagnostics ─────────────────────────────────────────────────────────

    /**
     * @brief Compute v⋅∇ψ residual norms (RMS + max) over all cells.
     *
     * Uses cell-centered velocity from CompactMAC faces and central FDs with
     * periodic lifting in y (for ψ1) and z (for ψ2) to handle periodic seams.
     * One-sided FDs at x=0 and x=nx-1.
     *
     * Runs one GPU kernel, reduces to a 5-element scratch buffer, copies to host.
     * Synchronizes the stream before returning.
     * Suitable for once-per-realization diagnostics (not inside hot loop).
     *
     * @param vel    CompactMAC velocity field (must match this field's grid).
     * @param grid   Grid metadata.
     * @param stream CUDA stream.
     * @return PsiQualityReport with RMS and max for r1 and r2.
     */
    PsiQualityReport compute_psi_quality(
        const VelocityField& vel,
        const Grid3D&        grid,
        cudaStream_t         stream
    ) const;

    // ── Optional diagnostics ────────────────────────────────────────────────

    /**
     * @brief Compute v·∇ψ residuals at all cell centers.
     *
     * Uses central differences (periodic in y,z; one-sided at x boundaries).
     * Reconstructs cell-center velocity from CompactMAC faces.
     *
     * @note Near y/z periodic seams, the finite-difference ∂ψ/∂y and ∂ψ/∂z
     * computations do NOT apply periodic lifting.  Residuals near the seam
     * may be artificially large.  Use this diagnostic only to check interior
     * cells or after verifying that ψ is smooth across the seam.
     *
     * @param vel    CompactMAC velocity field (must match this field's grid).
     * @param out_r1 Device pointer to output buffer for v·∇ψ1, size nx*ny*nz.
     * @param out_r2 Device pointer to output buffer for v·∇ψ2, size nx*ny*nz.
     * @param stream CUDA stream.
     */
    void compute_vdotgradpsi_norms(
        const VelocityField& vel,
        float*               out_r1,
        float*               out_r2,
        cudaStream_t         stream
    ) const;

private:
    /// Device int counter accumulated by kernel_advance_plane.
    DeviceBuffer<int> d_vx_clamped_counter_;
    /// 5-element double scratch for compute_psi_quality reduction [sumsq1, sumsq2, maxabs1, maxabs2, count].
    mutable DeviceBuffer<double> d_psi_quality_buf_;
};

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
