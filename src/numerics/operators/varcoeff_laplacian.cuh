#pragma once

/**
 * @file varcoeff_laplacian.cuh
 * @brief Variable-coefficient Laplacian operator for CG/MG comparison
 * 
 * This operator implements the SAME discrete Laplacian used by the multigrid
 * smoother and residual computation:
 *   (A*h)_C = sum_6faces( K_face * (h_C - h_neighbor) ) / dx²
 * 
 * where K_face is the harmonic mean:
 *   K_face = 2 / (1/K_C + 1/K_neighbor)
 * 
 * This allows direct comparison between CG solve and MG solve on the same
 * mathematical operator.
 * 
 * ## Pin support (for singular systems)
 * 
 * When enabled, replaces one row with identity: A(x)[p] = x[p]
 * Combined with RHS[p] = pin_value, this gives x[p] = pin_value exactly.
 */

#include "../../runtime/CudaContext.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/BCSpec.hpp"
#include "../../core/Scalar.hpp"
#include "../pin_spec.hpp"

namespace macroflow3d {
namespace operators {

// Re-export PinSpec in operators namespace for backward compat
using macroflow3d::PinSpec;

/**
 * Variable-coefficient Laplacian operator: -∇·(K∇h)
 * 
 * Stores references to grid, K field, and boundary conditions.
 * The K field must remain valid for the lifetime of the operator.
 */
class VarCoeffLaplacian {
public:
    VarCoeffLaplacian() = default;
    
    VarCoeffLaplacian(
        const Grid3D& grid,
        DeviceSpan<const real> K,
        const BCSpec& bc,
        PinSpec pin = {}  // Optional pin spec
    );
    
    // Matrix-free apply: y = A*x
    // Uses harmonic mean for face conductivities
    // Handles Dirichlet/Neumann/Periodic BCs
    // If pin enabled: y[pin_index] = x[pin_index] (identity row)
    void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const;
    
    // Accessors
    const Grid3D& grid() const { return grid_; }
    size_t size() const { return grid_.num_cells(); }
    const PinSpec& pin() const { return pin_; }
    
private:
    Grid3D grid_;
    DeviceSpan<const real> K_;
    BCSpec bc_;
    PinSpec pin_;
};

} // namespace operators
} // namespace macroflow3d
