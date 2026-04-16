/**
 * @file LegacyMarchingInvariantBuilder.cuh
 * @brief Wrapper for legacy semi-Lagrangian marching to build invariants.
 *
 * This class adapts the existing PsptaPsiField::precompute_levelA() to the
 * new PsptaInvariantField interface, enabling backward compatibility while
 * allowing gradual migration to Strategy A+C.
 *
 * @par Usage
 * @code
 * PsptaInvariantField inv;
 * inv.resize(grid);
 *
 * LegacyMarchingInvariantBuilder builder;
 * auto report = builder.build(inv, vel, grid, stream);
 *
 * // inv now contains psi1, psi2 built by legacy marching
 * engine.bind_invariants(&inv);
 * @endcode
 *
 * @ingroup physics_particles_pspta
 */

#pragma once

#include "../../../../core/Grid3D.hpp"
#include "../../../../core/Scalar.hpp"
#include "../../../common/fields.cuh"
#include "../invariants/PsptaInvariantField.cuh"
#include "../PsptaPsiField.cuh" // Legacy implementation
#include <cuda_runtime.h>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// Legacy builder configuration
// ============================================================================

/**
 * @brief Configuration for legacy marching builder.
 */
struct LegacyMarchingConfig {
    double eps_vx = 1e-10;      ///< Minimum clamped vx to avoid division by zero
    bool enable_refine = false; ///< Enable legacy defect-correction refinement
    PsiRefineConfig refine_cfg; ///< Refinement parameters (if enabled)
};

/**
 * @brief Report from legacy marching builder.
 */
struct LegacyMarchingReport {
    PsptaPrecomputeReport precompute;
    PsiRefineReport refine;
    double total_time_ms = 0.0;
};

// ============================================================================
// LegacyMarchingInvariantBuilder
// ============================================================================

/**
 * @brief Build invariants using legacy semi-Lagrangian marching.
 *
 * This is a thin wrapper around the existing PsptaPsiField implementation.
 * It provides the interface expected by the new architecture while delegating
 * to proven legacy code.
 */
class LegacyMarchingInvariantBuilder {
  public:
    explicit LegacyMarchingInvariantBuilder(const LegacyMarchingConfig& cfg = {});
    ~LegacyMarchingInvariantBuilder() = default;

    /**
     * @brief Build invariants into the provided field.
     *
     * @param out    Output invariant field (resized internally if needed)
     * @param vel    CompactMAC velocity field
     * @param grid   Grid metadata
     * @param stream CUDA stream
     * @return Report with diagnostics
     */
    LegacyMarchingReport build(PsptaInvariantField& out, const VelocityField& vel,
                               const Grid3D& grid, cudaStream_t stream);

    /**
     * @brief Get reference to internal legacy field (for debugging).
     */
    const PsptaPsiField& legacy_field() const { return legacy_; }

    const LegacyMarchingConfig& config() const { return config_; }

  private:
    LegacyMarchingConfig config_;
    PsptaPsiField legacy_; ///< Internal legacy implementation
};

// ============================================================================
// Adapter to use PsptaInvariantField with existing PsptaEngine
// ============================================================================

/**
 * @brief Lightweight adapter to make PsptaInvariantField compatible with
 * PsptaEngine.
 *
 * PsptaEngine expects a PsptaPsiField* with specific interface. This adapter
 * provides that interface while backing onto a PsptaInvariantField.
 *
 * @note For full integration, PsptaEngine::bind_psifield() should eventually
 * be renamed/overloaded to bind_invariants(PsptaInvariantField*).
 */
class InvariantFieldAdapter {
  public:
    /**
     * @brief Construct adapter wrapping an invariant field.
     *
     * @param inv PsptaInvariantField to adapt (must outlive adapter)
     */
    explicit InvariantFieldAdapter(PsptaInvariantField* inv);

    // ── Interface matching PsptaPsiField ───────────────────────────────────

    int nx() const { return inv_->nx(); }
    int ny() const { return inv_->ny(); }
    int nz() const { return inv_->nz(); }
    double dx() const { return inv_->dx(); }
    double dy() const { return inv_->dy(); }
    double dz() const { return inv_->dz(); }

    float* psi1_ptr() { return inv_->psi1_ptr(); }
    const float* psi1_ptr() const { return inv_->psi1_ptr(); }
    float* psi2_ptr() { return inv_->psi2_ptr(); }
    const float* psi2_ptr() const { return inv_->psi2_ptr(); }

    Grid3D grid() const { return inv_->grid(); }

  private:
    PsptaInvariantField* inv_;
};

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
