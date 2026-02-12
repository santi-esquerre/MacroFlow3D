#pragma once

/**
 * @file operator_concept.cuh
 * @brief Unified operator concept for linear solvers (CG, MG, etc.)
 * 
 * This header defines a lightweight "concept" (template pattern) for operators
 * without using virtual dispatch (HPC-first design).
 * 
 * An operator must provide:
 *   - apply(x, y, ctx, stream): compute y = A * x
 *   - Optional: diag(), metadata for preconditioners/smoothers
 * 
 * Current implementations:
 *   - Poisson3DOperator: constant-coefficient -Δ (for CG)
 *   - MGOperator: variable-coefficient -∇·(K∇) (for MG)
 * 
 * Future: unify both under this concept so CG and MG can share operators.
 */

#include "../../core/Scalar.hpp"
#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/BCSpec.hpp"
#include "../../runtime/CudaContext.cuh"

namespace macroflow3d {
namespace operators {

/**
 * @brief Operator concept (compile-time polymorphism, no vtable overhead)
 * 
 * Any type T satisfying this concept must provide:
 * 
 * void apply(
 *     const CudaContext& ctx,
 *     DeviceSpan<const real> x,
 *     DeviceSpan<real> y
 * ) const;
 * 
 * Optional:
 * DeviceSpan<const real> diag() const;  // Diagonal for preconditioners
 * const Grid3D& grid() const;           // Grid metadata
 * const BCSpec& boundary_conditions() const;  // Boundary conditions
 * 
 * Template-based design allows inlining and zero-cost abstraction.
 */

// Example: Operator wrapper for MG (variable-coefficient Laplacian)
// This will be implemented when unifying CG and MG operators in future tasks
/*
struct MGOperator {
    Grid3D grid;
    BCSpec bc;
    DeviceSpan<const real> K;  // Conductivity field
    
    void apply(
        const CudaContext& ctx,
        DeviceSpan<const real> x,
        DeviceSpan<real> y
    ) const {
        // Call compute_residual-style kernel: y = A*x
        // (not b - A*x, just A*x)
    }
    
    const Grid3D& get_grid() const { return grid; }
    const BCSpec& get_bc() const { return bc; }
};
*/

// Note: Poisson3DOperator already exists in numerics/operators/poisson3d_operator.cuh
// It needs minor refactoring to fit this concept (currently has apply_add, not apply)

} // namespace operators
} // namespace macroflow3d
