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

enum class RefinementACStrategy {
    AlternatingProjection,
    SubspaceQuadraticMap,
    SubspaceQuadraticGaussNewton,
    SubspaceQuadraticGaussNewtonProjectionProxy,
    SubspaceQuadraticGaussNewtonEngineProxy,
};

enum class EngineProxySelectorMode {
    CombinedScore,
    FailFractionLexicographic,
};

// ============================================================================
// Refinement configuration
// ============================================================================

/**
 * @brief Configuration for Strategy C refinement.
 */
struct RefinementACConfig {
    bool enabled = false;
    RefinementACStrategy strategy = RefinementACStrategy::AlternatingProjection;
    int max_iterations = 10;                    ///< Maximum alternating iterations
    double omega = 0.5;                         ///< Initial step size
    double omega_min = 1e-6;                    ///< Minimum step size (backtracking limit)
    int max_backtracks = 10;                    ///< Maximum backtracking steps
    double stop_rel_quality = 1e-3;             ///< Stop if quality improves < threshold
    double stop_abs_quality = 1e-6;             ///< Stop if absolute quality < threshold
    double poisson_tol = 1e-8;                  ///< Tolerance for inner Poisson solves
    int poisson_max_iter = 200;                 ///< Max iterations for inner Poisson solves
    double invariance_weight = 1.0;             ///< Weight for v·grad penalties in local fit
    double local_tikhonov = 1e-6;               ///< Tikhonov regularization for local 3x3 solves
    double max_invariance_growth = 0.25;        ///< Reject trial if invariance grows too much
    double max_degeneracy_growth = 0.10;        ///< Reject trial if degeneracy grows too much
    double min_relative_gradient_rms = 0.2;     ///< Anti-collapse guard relative to initial
    double min_relative_field_range = 0.2;      ///< Anti-collapse guard relative to initial
    double subspace_initial_step = 0.25;        ///< Initial coefficient step for subspace search
    double subspace_min_step = 1e-3;            ///< Minimum coefficient step for subspace search
    double gn_lambda_initial = 1e-2;            ///< Initial LM damping
    double gn_lambda_up = 5.0;                  ///< Damping growth on rejected trust step
    double gn_lambda_down = 0.5;                ///< Damping shrink on accepted trust step
    double gn_fd_relative_step = 1e-3;          ///< Relative finite-difference coefficient step
    double gn_fd_absolute_step = 1e-4;          ///< Absolute finite-difference coefficient step
    double gn_trust_radius_initial = 0.5;       ///< Initial coefficient trust radius
    double gn_trust_radius_min = 1e-3;          ///< Minimum trust radius before stopping
    double gn_trust_radius_max = 2.0;           ///< Maximum trust radius after repeated accepts
    double projection_proxy_yz_weight = 0.25;   ///< Relative weight for vy/vz mismatch in proxy GN
    double projection_proxy_cond_weight = 0.50; ///< Weight for Jacobian conditioning barrier
    double projection_proxy_cond_floor = 0.15;  ///< Reciprocal-condition floor for barrier
    double projection_proxy_acceptance_weight = 0.25; ///< Weight in accepted-candidate score
    int engine_proxy_sample_count = 64;           ///< Number of sampled particles in engine proxy
    int engine_proxy_sample_steps = 1;            ///< Number of surrogate PSPTA steps to emulate
    double engine_proxy_fail_weight = 1.0;        ///< Weight on mean sampled fail count
    double engine_proxy_iter_weight = 0.10;       ///< Weight on mean Newton iteration count
    double engine_proxy_residual_weight = 0.10;   ///< Weight on normalized final Newton residual
    double engine_proxy_low_recip_weight = 0.10;  ///< Weight on low-condition stage fraction
    double engine_proxy_acceptance_weight = 0.25; ///< Weight in accepted-candidate score
    EngineProxySelectorMode engine_proxy_selector_mode =
        EngineProxySelectorMode::CombinedScore; ///< Candidate selector for engine-aware GN
    bool verbose = false;
    bool save_history = false;
};

struct ProjectionProxyReport {
    double rel_rms_vx_det_mismatch = 0.0;
    double mean_recip_condition = 0.0;
    double min_recip_condition = 0.0;
    double low_recip_condition_fraction = 0.0;
    double combined_score = 0.0;
    bool valid = false;
};

struct EngineSampledProxyReport {
    double fail_fraction = 0.0;
    double mean_fail_count = 0.0;
    double fail_x_fraction = 0.0;
    double fail_mid_fraction = 0.0;
    double fail_new_fraction = 0.0;
    double mean_newton_iterations = 0.0;
    double mean_normalized_final_residual = 0.0;
    double low_recip_condition_fraction = 0.0;
    double combined_score = 0.0;
    bool valid = false;
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
    std::string best_trial_phase;
    std::string best_trial_rejection_reason;
    double best_trial_rel_mismatch = 0.0;
    double best_trial_invariance_sum = 0.0;
    double best_trial_degeneracy = 0.0;
    double best_trial_min_gradient_rms = 0.0;
    double best_trial_min_field_range = 0.0;
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
    ProjectionProxyReport initial_projection;
    ProjectionProxyReport final_projection;
    EngineSampledProxyReport initial_engine;
    EngineSampledProxyReport final_engine;
    double initial_min_gradient_rms = 0.0;
    double initial_min_field_range = 0.0;
    double final_min_gradient_rms = 0.0;
    double final_min_field_range = 0.0;
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

    /**
     * @brief Provide host-side Strategy A modal fields for subspace-constrained refinement.
     *
     * Each entry is a cell-centered scalar field with `num_cells()` values. The first four
     * modes are used as the authoritative A->C handoff object on `darcy_small`.
     */
    void set_subspace_basis_host(std::vector<std::vector<float>> basis_modes);

    const RefinementACConfig& config() const { return config_; }

  private:
    RefinementACReport refine_alternating_projection(PsptaInvariantField& inv, CudaContext& ctx);
    RefinementACReport refine_subspace_quadratic_map(PsptaInvariantField& inv, CudaContext& ctx);
    RefinementACReport refine_subspace_quadratic_gauss_newton(PsptaInvariantField& inv,
                                                              CudaContext& ctx);
    RefinementACReport
    refine_subspace_quadratic_gauss_newton_projection_proxy(PsptaInvariantField& inv,
                                                            CudaContext& ctx);
    RefinementACReport refine_subspace_quadratic_gauss_newton_engine_proxy(PsptaInvariantField& inv,
                                                                           CudaContext& ctx);

    Grid3D grid_;
    const VelocityField* vel_;
    RefinementACConfig config_;
    std::unique_ptr<GaugeFixer> gauge_fixer_;

    // Work buffers for Poisson solves
    DeviceBuffer<real> d_delta_psi1_;
    DeviceBuffer<real> d_delta_psi2_;
    DeviceBuffer<real> d_rhs_;
    DeviceBuffer<real> d_work_;
    DeviceBuffer<real> d_target_gx_;
    DeviceBuffer<real> d_target_gy_;
    DeviceBuffer<real> d_target_gz_;
    DeviceBuffer<float> d_trial_psi1_;
    DeviceBuffer<float> d_trial_psi2_;
    DeviceBuffer<float> d_base_psi1_;
    DeviceBuffer<float> d_base_psi2_;
    std::vector<std::vector<float>> subspace_basis_host_;
};

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
