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

#include "../../core/BCSpec.hpp"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../common/fields.cuh"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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
    ScalarField divergence;    ///< div(v) at cell centers
    ScalarField vorticity_mag; ///< |curl(v)| at cell centers
    ScalarField helicity;      ///< v_center · curl(v) at cell centers

    VelocityDiagnostics() = default;

    explicit VelocityDiagnostics(const Grid3D& grid)
        : divergence(grid), vorticity_mag(grid), helicity(grid) {}

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
void compute_velocity_diagnostics(const PaddedVelocityField& vel, VelocityDiagnostics& diag,
                                  const Grid3D& grid, const CudaContext& ctx);

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
void print_velocity_diagnostics(const VelocityDiagnostics& diag, int realization_id,
                                const CudaContext& ctx);

// ============================================================================
// Sampled backend diagnostics for FACE vs KH comparison
// ============================================================================

enum class VelocityEvalDiagnosticMode : uint8_t {
    FaceTrilinear = 0,
    KhLinear = 1,
    KhCubicPotentialReconstruction = 2,
    KhLogKCubicPotentialReconstruction = 3,
    KhPotentialReconstruction = KhLinear
};

struct VelocityDiagnosticSample {
    real speed = 0;
    real div_abs = 0;
    real curl_mag = 0;
    real helicity = 0;
    real helicity_abs = 0;
    real helicity_norm = 0;
    real k_interp = 0;
    real logk_interp = 0;
    uint8_t finite = 0;
    uint8_t has_k_interp = 0;
    uint8_t has_logk_interp = 0;
    uint8_t k_nonpositive = 0;
    uint8_t k_clamped = 0;
};

struct VelocityComparisonSample {
    real diff_mag = 0;
    real rel_diff = 0;
    real face_sq = 0;
    real kh_sq = 0;
    real dot = 0;
    uint8_t finite = 0;
};

struct VelocityEvalDiagnosticsSummary {
    int realization_id = 0;
    std::string backend;
    size_t n_samples = 0;
    size_t sample_stride = 1;
    int finite_count = 0;
    int invalid_count = 0;
    real speed_mean = 0;
    real speed_max = 0;
    real div_abs_mean = 0;
    real div_abs_max = 0;
    real curl_mag_mean = 0;
    real curl_mag_max = 0;
    real helicity_mean = 0;
    real helicity_abs_mean = 0;
    real helicity_abs_max = 0;
    real helicity_norm_mean = 0;
    real helicity_norm_std = 0;
    real helicity_norm_p50 = 0;
    real helicity_norm_p95 = 0;
    real helicity_norm_max = 0;
    real k_interp_min = 0;
    real k_interp_max = 0;
    real k_interp_mean = 0;
    int k_interp_nonpositive_count = 0;
    int k_interp_clamped_count = 0;
    real logk_interp_min = 0;
    real logk_interp_max = 0;
};

struct VelocityBackendComparisonSummary {
    int realization_id = 0;
    std::string backend;
    size_t n_samples = 0;
    size_t sample_stride = 1;
    int finite_count = 0;
    int invalid_count = 0;
    real rel_l2_diff = 0;
    real diff_mean = 0;
    real diff_std = 0;
    real diff_p50 = 0;
    real diff_p95 = 0;
    real diff_max = 0;
    real rel_diff_mean = 0;
    real rel_diff_p95 = 0;
    real rel_diff_max = 0;
    real vector_correlation = 0;
};

struct VelocityEvalDiagnosticsWorkspace {
    DeviceBuffer<VelocityDiagnosticSample> samples;
    DeviceBuffer<VelocityComparisonSample> comparison_samples;
    std::vector<VelocityDiagnosticSample> host_samples;
    std::vector<VelocityComparisonSample> host_comparison_samples;
};

VelocityEvalDiagnosticsSummary compute_velocity_eval_diagnostics(
    VelocityEvalDiagnosticMode mode, const PaddedVelocityField& face_velocity, const KField& K,
    const HeadField& head, const Grid3D& grid, const BCSpec& bc,
    VelocityEvalDiagnosticsWorkspace& workspace, const CudaContext& ctx, int realization_id,
    size_t max_samples = 262144);

VelocityBackendComparisonSummary compute_velocity_backend_comparison(
    const PaddedVelocityField& face_velocity, const KField& K, const HeadField& head,
    const Grid3D& grid, const BCSpec& bc, VelocityEvalDiagnosticMode mode,
    VelocityEvalDiagnosticsWorkspace& workspace, const CudaContext& ctx, int realization_id,
    size_t max_samples = 262144);

} // namespace physics
} // namespace macroflow3d
