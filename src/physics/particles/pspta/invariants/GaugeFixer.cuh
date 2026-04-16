/**
 * @file GaugeFixer.cuh
 * @brief Interface for gauge fixing of invariant fields.
 *
 * Partial implementation.
 *
 * Current status:
 * - Implemented: InletPlane gauge (sets psi1=y, psi2=z at x=0)
 * - Not implemented yet: MeanZero, ScaledPeriodic
 *
 * @par Gauge Ambiguity
 * Lagrangian invariants psi1, psi2 are defined up to arbitrary transformations:
 *
 *   psi1 -> f1(psi1)
 *   psi2 -> f2(psi2)
 *
 * and linear combinations. Gauge fixing selects a canonical representative.
 *
 * @par Common Gauge Choices
 *
 * 1. **Inlet normalization**: At x=0, set psi1 = y, psi2 = z
 *    (current legacy approach, compatible with existing engine)
 *
 * 2. **Mean zero**: Subtract mean so integral over domain is zero
 *
 * 3. **Scaled to physical coordinates**: psi1 in [0, Ly), psi2 in [0, Lz)
 *
 * @ingroup physics_particles_pspta
 */

#pragma once

#include "../../../../core/Scalar.hpp"
#include "../../../common/fields.cuh"
#include "PsptaInvariantField.cuh"
#include <cuda_runtime.h>
#include <string>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// Gauge fixing types
// ============================================================================

/**
 * @brief Gauge fixing method.
 */
enum class GaugeMethod : uint8_t {
    None = 0,          ///< No gauge fixing
    InletPlane = 1,    ///< Set psi1=y, psi2=z at inlet plane (x=0)
    MeanZero = 2,      ///< Subtract mean from each field
    ScaledPeriodic = 3 ///< Scale to [0, Ly) x [0, Lz) with wrapping
};

/**
 * @brief Configuration for gauge fixing.
 */
struct GaugeFixerConfig {
    GaugeMethod method = GaugeMethod::InletPlane;
    double inlet_x = 0.0; ///< x-coordinate of inlet plane for InletPlane method
    bool verbose = false;
};

// ============================================================================
// GaugeFixer interface
// ============================================================================

/**
 * @brief Apply gauge fixing to invariant fields.
 *
 * @note Partially implemented. See class-level status above.
 */
class GaugeFixer {
  public:
    explicit GaugeFixer(const GaugeFixerConfig& config = {});
    ~GaugeFixer() = default;

    /**
     * @brief Apply gauge fixing in-place.
     *
     * @param inv    Invariant field (modified in place)
     * @param vel    Velocity field (for diagnostics)
     * @param stream CUDA stream
     */
    void apply(PsptaInvariantField& inv, const VelocityField& vel, cudaStream_t stream);

    /**
     * @brief Apply inlet normalization specifically.
     *
     * Sets psi1(0,j,k) = (j+0.5)*dy, psi2(0,j,k) = (k+0.5)*dz.
     * This is the gauge used by the legacy marching and expected by PsptaEngine.
     */
    void apply_inlet_gauge(PsptaInvariantField& inv, cudaStream_t stream);

    GaugeMethod method() const { return config_.method; }

  private:
    GaugeFixerConfig config_;
};

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
