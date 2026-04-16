/**
 * @file PsptaInvariantField.cuh
 * @brief Unified container for streamline invariants psi1, psi2 with metadata.
 *
 * This is the NEW unified interface for invariant storage, decoupled from
 * how the invariants are constructed (legacy marching, Strategy A eigensolver,
 * Strategy C refinement, etc.).
 *
 * The PsptaEngine binds to this interface and does not need to know the
 * construction method. This enables a clean separation between:
 *   - Invariant construction (multiple backends possible)
 *   - Invariant consumption (Newton projection in transport)
 *
 * @par Storage
 * - psi1, psi2: cell-centered, float32 by default for memory efficiency
 * - All metadata in host structs, lightweight GPU scratch for quality metrics
 *
 * @par Compatibility
 * For backward compatibility, the legacy PsptaPsiField can be wrapped via
 * LegacyMarchingInvariantBuilder which populates a PsptaInvariantField.
 *
 * @ingroup physics_particles_pspta
 */

#pragma once

#include "../../../../core/DeviceBuffer.cuh"
#include "../../../../core/Grid3D.hpp"
#include "../../../../core/Scalar.hpp"
#include "../../../common/fields.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace macroflow3d {

// Forward declaration for modal quality diagnostics
class CudaContext;

namespace physics {
namespace particles {
namespace pspta {

// Forward declaration (defined in EigensolverBackend.cuh)
struct EigensolverResult;

// ============================================================================
// Quality metrics for invariance diagnostics
// ============================================================================

/**
 * @brief v*grad(psi) residual statistics.
 *
 * For a perfect invariant, v*grad(psi) = 0 everywhere.
 * RMS and max norms quantify departure from exact invariance.
 */
struct InvarianceQuality {
    double rms_r1 = 0.0;   ///< RMS of v*grad(psi1) over all cells
    double max_r1 = 0.0;   ///< max |v*grad(psi1)|
    double rms_r2 = 0.0;   ///< RMS of v*grad(psi2)
    double max_r2 = 0.0;   ///< max |v*grad(psi2)|
    long long n_cells = 0; ///< number of cells evaluated
};

/**
 * @brief Cross-product reconstruction quality: ||v - grad(psi1) x grad(psi2)||.
 *
 * For ideal streamfunctions, v = grad(psi1) x grad(psi2).
 * This metric measures how well the invariants reconstruct the velocity.
 */
struct CrossProductQuality {
    double rms_mismatch = 0.0;     ///< RMS of ||v - grad_psi1 x grad_psi2||
    double max_mismatch = 0.0;     ///< max ||v - grad_psi1 x grad_psi2||
    double rel_rms_mismatch = 0.0; ///< RMS / mean(|v|)
    long long n_cells = 0;
};

/**
 * @brief Independence metric: correlation between grad(psi1) and grad(psi2).
 *
 * Independent invariants should have orthogonal gradients.
 * degeneracy_score measures |grad(psi1) . grad(psi2)| / (|grad_psi1|
 * |grad_psi2|).
 */
struct IndependenceQuality {
    double mean_cos_angle = 0.0;   ///< mean of |cos(angle)| between gradients
    double max_cos_angle = 0.0;    ///< max |cos(angle)|
    double degeneracy_score = 0.0; ///< fraction of cells with |cos| > 0.9
    long long n_cells = 0;
};

/**
 * @brief Combined quality report for invariants.
 */
struct InvariantQualityReport {
    InvarianceQuality invariance;
    CrossProductQuality cross_product;
    IndependenceQuality independence;

    double masked_fraction = 0.0; ///< fraction of cells masked (e.g., low velocity)
    bool valid = false;           ///< true if quality was computed successfully
};

// ============================================================================
// Modal quality diagnostics (per-eigenvector, computed at ingestion time)
// ============================================================================

/**
 * @brief Per-mode quality metrics from the eigensolver.
 *
 * Computed at eigenvector ingestion time (double precision, before f64→f32
 * cast) to give the most accurate diagnostics possible.
 *
 * These metrics determine whether the modes are suitable for downstream
 * gauge fixing and particle transport.
 */
struct ModalQualityReport {
    int n_modes = 0;                    ///< number of ingested modes (typically 2)
    std::vector<double> eigenvalues;    ///< λ_i from eigensolver (ascending)
    std::vector<double> residual_norms; ///< ||Aψ_i − λ_i ψ_i|| from eigensolver
    std::vector<double> l2_norms;       ///< ||ψ_i||₂ (should ≈ 1 if normalized)
    double orthogonality = 0.0;         ///< |<ψ_1,ψ_2>|/(||ψ_1||·||ψ_2||), 0=perfect
    bool gauge_ready = false;           ///< true if modes usable for gauge fixing
};

// ============================================================================
// Construction metadata
// ============================================================================

/**
 * @brief Method used to construct the invariants.
 */
enum class InvariantConstructionMethod : uint8_t {
    Unknown = 0,
    LegacyMarching = 1, ///< Semi-Lagrangian x-marching (legacy)
    StrategyA = 2,      ///< Eigenproblem near-nullspace
    StrategyAC = 3,     ///< Strategy A + Strategy C refinement
    StrategyC = 4,      ///< Strategy C only (refinement from seed)
    Custom = 5          ///< User-provided
};

/**
 * @brief Grid snapshot at construction time.
 *
 * Captures essential grid parameters so the invariant field can verify
 * compatibility with velocity fields used later.
 */
struct InvariantGridInfo {
    int nx = 0, ny = 0, nz = 0;
    double dx = 0.0, dy = 0.0, dz = 0.0;
    double Lx = 0.0, Ly = 0.0, Lz = 0.0;

    bool matches(const Grid3D& g) const { return nx == g.nx && ny == g.ny && nz == g.nz; }
};

/**
 * @brief Metadata about how the invariants were constructed.
 */
struct InvariantConstructionInfo {
    InvariantConstructionMethod method = InvariantConstructionMethod::Unknown;
    InvariantGridInfo grid_info;

    // Strategy A parameters (if applicable)
    double mu = 0.0;                 ///< regularization parameter
    int n_eigenvectors_computed = 0; ///< K in the subspace
    std::vector<double> eigenvalues; ///< smallest K eigenvalues
    int eigensolver_iterations = 0;
    double eigensolver_residual = 0.0;
    std::string eigensolver_backend; ///< e.g., "LOBPCG", "SLEPc"

    // Strategy C parameters (if applicable)
    int refinement_iterations = 0;
    double refinement_omega = 0.0;
    double refinement_final_rms = 0.0;
    std::string refinement_stop_reason;

    // Legacy marching parameters (if applicable)
    double legacy_eps_vx = 0.0;
    long long legacy_vx_clamped = 0;

    // Gauge fixing info
    bool inlet_gauge_applied = false; ///< psi1=y, psi2=z at x=0
    std::string gauge_method;         ///< e.g., "inlet", "none"

    // General metadata
    double construction_time_ms = 0.0;
    long long n_masked_cells = 0; ///< cells where invariant is invalid
    double masked_fraction = 0.0; ///< n_masked / total cells
    std::string notes;
};

// ============================================================================
// Main container: PsptaInvariantField
// ============================================================================

/**
 * @brief Unified container for streamline invariants psi1, psi2.
 *
 * This is the primary interface for invariant storage. The transport engine
 * binds to this and does not depend on construction details.
 *
 * @par Usage
 * @code
 * PsptaInvariantField inv;
 * inv.resize(grid);
 *
 * // Option 1: Use legacy marching
 * LegacyMarchingInvariantBuilder builder;
 * builder.build(inv, vel, grid, stream);
 *
 * // Option 2: Use Strategy A+C (future)
 * // InvariantSolverAC solver;
 * // solver.solve(inv, vel, grid, stream, config);
 *
 * // Use in engine
 * engine.bind_invariants(&inv);
 * @endcode
 */
class PsptaInvariantField {
  public:
    // ── Lifecycle ──────────────────────────────────────────────────────────

    PsptaInvariantField() = default;
    ~PsptaInvariantField() = default;

    // Non-copyable but movable
    PsptaInvariantField(PsptaInvariantField&&) = default;
    PsptaInvariantField& operator=(PsptaInvariantField&&) = default;
    PsptaInvariantField(const PsptaInvariantField&) = delete;
    PsptaInvariantField& operator=(const PsptaInvariantField&) = delete;

    /**
     * @brief Resize buffers to match grid. Only reallocates if larger.
     */
    void resize(const Grid3D& grid);

    /**
     * @brief Clear all data and reset to empty state.
     */
    void clear();

    // ── Grid metadata ──────────────────────────────────────────────────────

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    double dx() const { return dx_; }
    double dy() const { return dy_; }
    double dz() const { return dz_; }
    double Lx() const { return nx_ * dx_; }
    double Ly() const { return ny_ * dy_; }
    double Lz() const { return nz_ * dz_; }

    Grid3D grid() const {
        return Grid3D(nx_, ny_, nz_, static_cast<real>(dx_), static_cast<real>(dy_),
                      static_cast<real>(dz_));
    }

    size_t num_cells() const { return static_cast<size_t>(nx_) * ny_ * nz_; }

    bool is_valid() const { return nx_ > 0 && ny_ > 0 && nz_ > 0; }

    // ── Primary data access ────────────────────────────────────────────────

    /// Device pointer to psi1 (float32, cell-centered, size = nx*ny*nz)
    float* psi1_ptr() { return d_psi1_.data(); }
    const float* psi1_ptr() const { return d_psi1_.data(); }

    /// Device pointer to psi2 (float32, cell-centered, size = nx*ny*nz)
    float* psi2_ptr() { return d_psi2_.data(); }
    const float* psi2_ptr() const { return d_psi2_.data(); }

    /// Direct buffer access (for builders)
    DeviceBuffer<float>& psi1_buffer() { return d_psi1_; }
    DeviceBuffer<float>& psi2_buffer() { return d_psi2_; }
    const DeviceBuffer<float>& psi1_buffer() const { return d_psi1_; }
    const DeviceBuffer<float>& psi2_buffer() const { return d_psi2_; }

    // ── Optional cached derivatives (for advanced diagnostics) ─────────────

    /**
     * @brief Allocate and compute cached derivatives if not already done.
     *
     * After calling, dpsi1_dx_ptr(), dpsi1_dy_ptr(), etc. are available.
     * Uses central finite differences with periodic lifting in y,z.
     * Uses one-sided FD at x boundaries.
     *
     * The VelocityField is NOT used by this function but kept in signature
     * for potential future velocity-weighted derivative computation.
     *
     * @note This is optional; most use cases don't need cached derivatives.
     *       The transport engine does NOT use these (it uses trilinear +
     *       analytic).
     */
    void ensure_cached_derivatives(const VelocityField& vel, cudaStream_t stream);
    bool has_cached_derivatives() const { return has_cached_derivs_; }

    // X-derivatives (one-sided at boundaries)
    float* dpsi1_dx_ptr() { return d_dpsi1_dx_.data(); }
    float* dpsi2_dx_ptr() { return d_dpsi2_dx_.data(); }
    const float* dpsi1_dx_ptr() const { return d_dpsi1_dx_.data(); }
    const float* dpsi2_dx_ptr() const { return d_dpsi2_dx_.data(); }

    // Y-derivatives (periodic with lifting)
    float* dpsi1_dy_ptr() { return d_dpsi1_dy_.data(); }
    float* dpsi2_dy_ptr() { return d_dpsi2_dy_.data(); }
    const float* dpsi1_dy_ptr() const { return d_dpsi1_dy_.data(); }
    const float* dpsi2_dy_ptr() const { return d_dpsi2_dy_.data(); }

    // Z-derivatives (periodic with lifting)
    float* dpsi1_dz_ptr() { return d_dpsi1_dz_.data(); }
    float* dpsi2_dz_ptr() { return d_dpsi2_dz_.data(); }
    const float* dpsi1_dz_ptr() const { return d_dpsi1_dz_.data(); }
    const float* dpsi2_dz_ptr() const { return d_dpsi2_dz_.data(); }

    // ── Quality metrics ────────────────────────────────────────────────────

    /**
     * @brief Compute comprehensive quality metrics.
     *
     * Synchronizes the stream before returning.
     */
    InvariantQualityReport compute_quality(const VelocityField& vel, cudaStream_t stream) const;

    /**
     * @brief Get the most recently computed quality report.
     */
    const InvariantQualityReport& quality() const { return quality_; }

    /**
     * @brief Update cached quality (called by builders after construction).
     */
    void set_quality(const InvariantQualityReport& q) { quality_ = q; }

    /**
     * @brief Get the modal quality report (computed at eigenvector ingestion).
     */
    const ModalQualityReport& modal_quality() const { return modal_quality_; }

    // ── Construction metadata ──────────────────────────────────────────────

    const InvariantConstructionInfo& construction_info() const { return construction_info_; }

    void set_construction_info(const InvariantConstructionInfo& info) { construction_info_ = info; }

    InvariantConstructionMethod method() const { return construction_info_.method; }

    // ── Strategy A eigenvector ingestion ───────────────────────────────────

    /**
     * @brief Ingest eigenvectors from the eigensolver into psi1/psi2.
     *
     * Performs a GPU-side float64→float32 cast of the two leading eigenvectors
     * and populates the construction metadata for Strategy A provenance.
     *
     * This is the single, canonical entry point for converting eigensolver
     * output into the invariant field. No scattered casts elsewhere.
     *
     * @param ev1     First eigenvector (float64, device buffer, size >=
     * num_cells)
     * @param ev2     Second eigenvector (float64, device buffer)
     * @param result  EigensolverResult with eigenvalues, residuals, iterations
     * @param mu      Regularization parameter used in A = D†WD + μL
     * @param backend_name  Solver backend name (e.g. "slepc")
     * @param ctx     CudaContext for blas reductions (modal quality diagnostics)
     * @param stream  CUDA stream for the cast kernel
     *
     * @pre  resize(grid) must have been called first.
     * @post psi1_ptr()/psi2_ptr() contain the cast eigenvectors.
     *       construction_info().method == StrategyA.
     *       modal_quality() is populated.
     *       Cached derivatives are invalidated.
     */
    void ingest_eigenvectors(const DeviceBuffer<real>& ev1, const DeviceBuffer<real>& ev2,
                             const EigensolverResult& result, double mu,
                             const std::string& backend_name, CudaContext& ctx,
                             cudaStream_t stream);

  private:
    // Grid metadata
    int nx_ = 0, ny_ = 0, nz_ = 0;
    double dx_ = 0.0, dy_ = 0.0, dz_ = 0.0;

    // Primary data: psi1, psi2 (float32 for memory efficiency)
    DeviceBuffer<float> d_psi1_;
    DeviceBuffer<float> d_psi2_;

    // Optional cached derivatives (allocated on demand)
    bool has_cached_derivs_ = false;
    DeviceBuffer<float> d_dpsi1_dx_;
    DeviceBuffer<float> d_dpsi1_dy_;
    DeviceBuffer<float> d_dpsi1_dz_;
    DeviceBuffer<float> d_dpsi2_dx_;
    DeviceBuffer<float> d_dpsi2_dy_;
    DeviceBuffer<float> d_dpsi2_dz_;

    // Cached quality metrics
    mutable InvariantQualityReport quality_;
    ModalQualityReport modal_quality_;

    // Construction metadata
    InvariantConstructionInfo construction_info_;

    // Scratch buffer for quality reduction
    mutable DeviceBuffer<double> d_quality_scratch_;
};

// ============================================================================
// Helper: linear index for cell-centered fields (host and device)
// ============================================================================

__host__ __device__ inline int invariant_idx(int i, int j, int k, int nx, int ny) {
    return i + nx * (j + ny * k);
}

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
