#pragma once

/**
 * @file preconditioner.cuh
 * @brief Minimal preconditioner interface for iterative solvers
 * 
 * Legacy correspondence: PCCMG_CG structure and Precond_CCMG_Vcycle* routines
 * 
 * Design constraints (HPC):
 * - apply() must NOT allocate memory (all workspace pre-allocated)
 * - apply() is called every iteration, must be efficient
 * - Interface is "conceptual" - uses template duck-typing, not virtual
 */

#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"

namespace macroflow3d {
namespace solvers {

/**
 * @brief Identity preconditioner (M = I, so z = r)
 * 
 * Used as a baseline and for plain CG (no preconditioning).
 */
struct IdentityPreconditioner {
    /**
     * @brief Apply preconditioner: z = M^{-1} r = r (identity)
     * 
     * @param ctx  CUDA context
     * @param r    Input residual vector (device)
     * @param z    Output preconditioned residual (device)
     */
    void apply(CudaContext& ctx, DeviceSpan<const real> r, DeviceSpan<real> z) const;
};

/**
 * @brief Concept documentation for Preconditioner
 * 
 * Any preconditioner must provide:
 * 
 *   void apply(CudaContext& ctx, DeviceSpan<const real> r, DeviceSpan<real> z) const;
 * 
 * Where:
 *   - r is the residual vector (input, device memory)
 *   - z is the preconditioned residual (output, device memory, same size as r)
 *   - apply computes z = M^{-1} r where M approximates the operator A
 *   - apply must NOT allocate memory (use pre-allocated workspace)
 *   - apply should be efficient (called every PCG iteration)
 */

} // namespace solvers
} // namespace macroflow3d
