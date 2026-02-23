#pragma once

/**
 * @file pcg.cuh
 * @brief Preconditioned Conjugate Gradient solver
 * @ingroup numerics_solvers
 * 
 * Legacy correspondence:
 *   - solver_CG(AH, PCCMG_CG, ...) where PCCMG_CG is MG preconditioner
 *   - rz = dot(r, z), NOT dot(r, r) as in unpreconditioned CG
 * 
 * Design (HPC):
 *   - Zero allocations in solve loop
 *   - Workspace passed from outside
 *   - Device-first: sync only every check_every iterations
 *   - Preconditioner is template (duck-typing, not virtual)
 */

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/DeviceBuffer.cuh"
#include "../../core/Scalar.hpp"
#include "../../numerics/blas/blas.cuh"

namespace macroflow3d {
namespace solvers {

// PCG configuration
struct PCGConfig {
    int max_iter = 1000;
    int check_every = 10;  // Check convergence every N iterations
    real rtol = 1e-6;      // Relative tolerance
    bool verbose = false;
};

// PCG result
struct PCGResult {
    int iterations = 0;
    real initial_residual = 0.0;
    real final_residual = 0.0;
    bool converged = false;
};

// Workspace for PCG (pre-allocated buffers)
struct PCGWorkspace {
    DeviceBuffer<real> r;  // Residual
    DeviceBuffer<real> z;  // Preconditioned residual
    DeviceBuffer<real> p;  // Search direction
    DeviceBuffer<real> Ap; // A * p
    
    // Reduction workspace for dot/nrm2 (persistent, avoids per-solve alloc)
    blas::ReductionWorkspace red;
    
    void ensure(size_t n) {
        if (r.size() != n) {
            r.resize(n);
            z.resize(n);
            p.resize(n);
            Ap.resize(n);
        }
    }
};

/**
 * @brief Preconditioned Conjugate Gradient solver
 * 
 * Solves A*x = b using PCG with preconditioner M.
 * 
 * Algorithm:
 *   r0 = b - A*x0
 *   z0 = M^{-1} * r0
 *   p0 = z0
 *   for k = 0, 1, ...
 *       alpha_k = (r_k, z_k) / (p_k, A*p_k)
 *       x_{k+1} = x_k + alpha_k * p_k
 *       r_{k+1} = r_k - alpha_k * A*p_k
 *       if ||r_{k+1}|| / ||r_0|| < rtol: converged
 *       z_{k+1} = M^{-1} * r_{k+1}
 *       beta_k = (r_{k+1}, z_{k+1}) / (r_k, z_k)
 *       p_{k+1} = z_{k+1} + beta_k * p_k
 * 
 * Template parameters:
 *   - Operator: must have `apply(ctx, x, Ax)` method
 *   - Preconditioner: must have `apply(ctx, r, z)` method
 * 
 * @param ctx    CUDA context
 * @param A      Linear operator with apply(ctx, x, Ax)
 * @param M      Preconditioner with apply(ctx, r, z)
 * @param b      Right-hand side (device)
 * @param x      Initial guess / solution (device, in/out)
 * @param cfg    PCG configuration
 * @param ws     Pre-allocated workspace
 * @return PCGResult with convergence info
 */
template<typename Operator, typename Preconditioner>
PCGResult pcg_solve(
    CudaContext& ctx,
    const Operator& A,
    const Preconditioner& M,
    DeviceSpan<const real> b,
    DeviceSpan<real> x,
    const PCGConfig& cfg,
    PCGWorkspace& ws
) {
    PCGResult result;
    const size_t n = x.size();
    
    // Ensure workspace
    ws.ensure(n);
    
    // Use persistent reduction workspace from PCGWorkspace (no allocation per solve)
    auto& red = ws.red;
    
    // Spans for convenience
    auto r  = ws.r.span();
    auto z  = ws.z.span();
    auto p  = ws.p.span();
    auto Ap = ws.Ap.span();
    
    // Initial residual: r0 = b - A*x0
    A.apply(ctx, DeviceSpan<const real>(x), Ap);  // Ap = A*x (temporary use)
    blas::copy(ctx, b, r);                         // r = b
    blas::axpy(ctx, -1.0, DeviceSpan<const real>(Ap), r);  // r = b - A*x
    
    // Initial residual norm
    result.initial_residual = blas::nrm2_host(ctx, DeviceSpan<const real>(r), red);
    
    if (result.initial_residual < 1e-15) {
        // Already converged
        result.converged = true;
        result.final_residual = result.initial_residual;
        return result;
    }
    
    // Apply preconditioner: z0 = M^{-1} * r0
    M.apply(ctx, DeviceSpan<const real>(r), z);
    
    // p0 = z0
    blas::copy(ctx, DeviceSpan<const real>(z), p);
    
    // rz_old = (r, z)
    real rz_old = blas::dot_host(ctx, DeviceSpan<const real>(r), DeviceSpan<const real>(z), red);
    
    // PCG iteration
    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        result.iterations = iter + 1;
        
        // Ap = A * p
        A.apply(ctx, DeviceSpan<const real>(p), Ap);
        
        // alpha = rz_old / (p, Ap)
        real pAp = blas::dot_host(ctx, DeviceSpan<const real>(p), DeviceSpan<const real>(Ap), red);
        real alpha = rz_old / pAp;
        
        // x = x + alpha * p
        blas::axpy(ctx, alpha, DeviceSpan<const real>(p), x);
        
        // r = r - alpha * Ap
        blas::axpy(ctx, -alpha, DeviceSpan<const real>(Ap), r);
        
        // Check convergence every check_every iterations
        if ((iter + 1) % cfg.check_every == 0 || iter == cfg.max_iter - 1) {
            result.final_residual = blas::nrm2_host(ctx, DeviceSpan<const real>(r), red);
            real relative_res = result.final_residual / result.initial_residual;
            
            if (relative_res < cfg.rtol) {
                result.converged = true;
                return result;
            }
        }
        
        // z = M^{-1} * r
        M.apply(ctx, DeviceSpan<const real>(r), z);
        
        // rz_new = (r, z)
        real rz_new = blas::dot_host(ctx, DeviceSpan<const real>(r), DeviceSpan<const real>(z), red);
        
        // beta = rz_new / rz_old
        real beta = rz_new / rz_old;
        
        // p = z + beta * p
        blas::axpby(ctx, 1.0, DeviceSpan<const real>(z), beta, p);
        
        rz_old = rz_new;
    }
    
    // Did not converge within max_iter
    result.final_residual = blas::nrm2_host(ctx, DeviceSpan<const real>(r), red);
    return result;
}

} // namespace solvers
} // namespace macroflow3d
