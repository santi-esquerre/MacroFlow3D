#pragma once

/**
 * @file pin_spec.hpp
 * @brief Pin specification and utilities for singular linear systems
 * @ingroup numerics
 *
 * When solving the Darcy flow equation with all periodic or Neumann BCs,
 * the system is singular (constant mode in null space). Pinning one cell
 * breaks this degeneracy by anchoring the solution.
 *
 * ## Legacy semantics: "diagonal doubling"
 *
 * For the pinned cell (always index 0 = cell [0,0,0]):
 *   - Operator: aC *= 2 (doubles the diagonal coefficient contribution)
 *   - Smoother (GSRB): aC *= 2 before computing update
 *   - Residual: aC *= 2 before computing residual
 *   - RHS: NOT modified
 *
 * This effectively anchors H[0,0,0] towards the mean of its neighbors,
 * removing the constant mode from the null space.
 *
 * ## Usage
 *
 * 1. Use `needs_pin(bc)` to detect if system is singular
 * 2. Use `pin_enabled(mode, bc)` to respect user config (auto/on/off)
 * 3. Pass `PinSpec(enabled)` to operators, smoothers, and residual kernels
 *
 * ## Files that use pin:
 *
 * - src/numerics/operators/varcoeff_laplacian.cu  (outer operator)
 * - src/multigrid/smoothers/gsrb_3d.cu            (smoother)
 * - src/multigrid/smoothers/residual_3d.cu        (residual)
 * - src/multigrid/cycle/v_cycle.cu                (propagates to all levels)
 * - src/numerics/solvers/mg_preconditioner.cu     (preconditioner)
 * - src/physics/flow/solve_head.cu                (constructs PinSpec)
 */

#include "../core/BCSpec.hpp"
#include "../core/Scalar.hpp"
#include <cstddef>

namespace macroflow3d {

// ============================================================================
// Pin Mode (user configuration)
// ============================================================================

/**
 * @brief Pin mode for singular system handling
 *
 * - Auto: enable pin only when system is singular (no Dirichlet BCs)
 * - On: always enable pin (useful for testing)
 * - Off: never pin (may fail/drift if system is actually singular)
 */
enum class PinMode {
    Auto, // Enable only when needed (default)
    On,   // Always enable
    Off   // Never enable
};

// ============================================================================
// Pin Specification (runtime parameter)
// ============================================================================

/**
 * @brief Pin specification passed to kernels
 *
 * Simple POD structure that travels through the numeric stack.
 * When enabled, kernels apply diagonal doubling at cell 0.
 */
struct PinSpec {
    bool enabled = false; // Whether pin is active
    size_t index = 0;     // Always 0 (legacy: cell [0,0,0])

    PinSpec() = default;
    explicit PinSpec(bool en) : enabled(en), index(0) {}
};

// ============================================================================
// Pin Utilities
// ============================================================================

/**
 * @brief Check if system needs a pin (is singular)
 *
 * A system is singular when there's no Dirichlet BC to anchor the solution.
 * Periodic and Neumann BCs do NOT fix the gauge → system is singular.
 *
 * @param bc Boundary conditions
 * @return true if ALL faces are Periodic or Neumann (system is singular)
 */
inline bool needs_pin(const BCSpec& bc) {
    auto is_dirichlet = [](const BCFace& f) { return f.type == BCType::Dirichlet; };

    // If ANY face is Dirichlet, system is NOT singular
    bool has_dirichlet = is_dirichlet(bc.xmin) || is_dirichlet(bc.xmax) || is_dirichlet(bc.ymin) ||
                         is_dirichlet(bc.ymax) || is_dirichlet(bc.zmin) || is_dirichlet(bc.zmax);

    return !has_dirichlet;
}

/**
 * @brief Determine if pin should be enabled based on mode and BCs
 *
 * @param mode User-configured pin mode (auto/on/off)
 * @param bc Boundary conditions
 * @return true if pin should be applied
 */
inline bool pin_enabled(PinMode mode, const BCSpec& bc) {
    switch (mode) {
    case PinMode::On:
        return true;
    case PinMode::Off:
        return false;
    case PinMode::Auto:
    default:
        return needs_pin(bc);
    }
}

/**
 * @brief Get pin mode as string for logging
 */
inline const char* pin_mode_str(PinMode mode) {
    switch (mode) {
    case PinMode::On:
        return "on";
    case PinMode::Off:
        return "off";
    case PinMode::Auto:
        return "auto";
    default:
        return "unknown";
    }
}

} // namespace macroflow3d
