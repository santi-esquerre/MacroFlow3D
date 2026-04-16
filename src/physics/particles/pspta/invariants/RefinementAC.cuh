/**
 * @file RefinementAC.cuh
 * @brief Interface for Strategy C refinement of invariant fields.
 *
 * This is a SKELETON interface for future implementation.
 *
 * @par Strategy C Overview
 * After Strategy A provides initial invariants, Strategy C refines them
 * using alternating optimization to better satisfy:
 *
 *   v = grad(psi1) x grad(psi2)
 *
 * @par Algorithm
 * For each refinement iteration:
 *
 * 1. Fix psi2, optimize psi1:
 *    - Compute target gradient: g1_target = (v x grad(psi2)) / |grad(psi2)|^2
 *    - Project to integrable: solve Poisson ∇²(delta_psi1) = div(g1_target)
 *    - Update: psi1 <- psi1 + omega * delta_psi1
 *
 * 2. Fix psi1, optimize psi2:
 *    - Compute target gradient: g2_target = (v x grad(psi1)) / |grad(psi1)|^2
 *    - Project to integrable: solve Poisson ∇²(delta_psi2) = div(g2_target)
 *    - Update: psi2 <- psi2 + omega * delta_psi2
 *
 * 3. Backtracking line search on cross-product quality metric
 *
 * 4. Reapply gauge fixing
 *
 * @ingroup physics_particles_pspta
 */

#pragma once

#include "../../../../core/Scalar.hpp"
#include "../../../../runtime/CudaContext.cuh"
#include "../../../common/fields.cuh"
#include "GaugeFixer.cuh"
#include "PsptaInvariantField.cuh"
#include "TransportOperator3D.cuh"
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// Refinement configuration
// ============================================================================

/**
 * @brief Configuration for Strategy C refinement.
 */
struct RefinementACConfig {
    bool enabled = false;
    int max_iterations = 10;        ///< Maximum alternating iterations
    double omega = 0.5;             ///< Initial step size
    double omega_min = 1e-6;        ///< Minimum step size (backtracking limit)
    int max_backtracks = 10;        ///< Maximum backtracking steps
    double stop_rel_quality = 0.1;  ///< Stop if quality improves < 10%
    double stop_abs_quality = 1e-6; ///< Stop if absolute quality < threshold
    double poisson_tol = 1e-8;      ///< Tolerance for inner Poisson solves
    int poisson_max_iter = 100;     ///< Max iterations for Poisson solves
    bool verbose = false;
    bool save_history = false;
};

/**
 * @brief Per-iteration diagnostics from refinement.
 */
struct RefinementIterReport {
    int iter = 0;
    double omega_accepted = 0.0;
    int backtracks = 0;
    bool accepted = false;
    InvariantQualityReport quality_before;
    InvariantQualityReport quality_after;
    double rel_improvement = 0.0; ///< (q_before - q_after) / q_before
    double poisson_residual_1 = 0.0;
    double poisson_residual_2 = 0.0;
};

/**
 * @brief Full report from Strategy C refinement.
 */
struct RefinementACReport {
    bool enabled = false;
    bool converged = false;
    int iterations_done = 0;
    std::string stop_reason;
    InvariantQualityReport initial_quality;
    InvariantQualityReport final_quality;
    double total_time_ms = 0.0;
    std::vector<RefinementIterReport> history;
};

// ============================================================================
// RefinementAC interface
// ============================================================================

/**
 * @brief Strategy C refinement for invariant fields.
 *
 * @note This is a SKELETON for future implementation (Phase 5).
 */
class RefinementAC {
  public:
    /**
     * @brief Construct refinement engine.
     *
     * @param grid   Grid metadata
     * @param vel    Velocity field (must outlive the engine)
     * @param config Refinement configuration
     */
    RefinementAC(const Grid3D& grid, const VelocityField* vel,
                 const RefinementACConfig& config = {});

    ~RefinementAC() = default;

    /**
     * @brief Apply Strategy C refinement to invariant fields.
     *
     * @param inv    Invariant field (modified in place)
     * @param ctx    CUDA context
     * @return Refinement report with diagnostics
     *
     * @note The gauge is reapplied after refinement using InletPlane method.
     */
    RefinementACReport refine(PsptaInvariantField& inv, CudaContext& ctx);

    /**
     * @brief Set gauge fixer for post-refinement normalization.
     */
    void set_gauge_fixer(std::unique_ptr<GaugeFixer> gf);

    const RefinementACConfig& config() const { return config_; }

  private:
    Grid3D grid_;
    const VelocityField* vel_;
    RefinementACConfig config_;
    std::unique_ptr<GaugeFixer> gauge_fixer_;

    // Work buffers for Poisson solves
    DeviceBuffer<real> d_delta_psi1_;
    DeviceBuffer<real> d_delta_psi2_;
    DeviceBuffer<real> d_rhs_;
    DeviceBuffer<real> d_work_;
};

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
