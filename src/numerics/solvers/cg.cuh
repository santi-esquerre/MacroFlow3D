#pragma once

#include "cg_types.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../blas/blas.cuh"
#include <cmath>
#include <iostream>

namespace rwpt {
namespace solvers {

template<typename Operator>
CGResult cg_solve(CudaContext& ctx,
                  const Operator& A,
                  DeviceSpan<const real> b,
                  DeviceSpan<real> x,
                  const CGConfig& cfg,
                  CGWorkspace& ws) {
    
    CGResult result;
    size_t n = b.size();
    
    // Ensure workspace is allocated
    ws.ensure(n);
    
    auto r_span = ws.r.span();
    auto p_span = ws.p.span();
    auto Ap_span = ws.Ap.span();
    
    // Device scalar spans
    auto d_rr_span = ws.d_rr.span();
    auto d_rr_new_span = ws.d_rr_new.span();
    auto d_pAp_span = ws.d_pAp.span();
    auto d_alpha_span = ws.d_alpha.span();
    auto d_beta_span = ws.d_beta.span();
    
    // r = b - A*x
    A.apply(ctx, x, r_span);
    
    // DEBUG: Print norms before computing residual
    if (cfg.verbose) {
        real b_norm, Ax_norm;
        blas::dot_device(ctx, b, b, d_rr_span, ws.red);
        RWPT_CUDA_CHECK(cudaMemcpy(&b_norm, ws.d_rr.data(), sizeof(real), cudaMemcpyDeviceToHost));
        blas::dot_device(ctx, r_span, r_span, d_rr_span, ws.red);
        RWPT_CUDA_CHECK(cudaMemcpy(&Ax_norm, ws.d_rr.data(), sizeof(real), cudaMemcpyDeviceToHost));
        ctx.synchronize();
        std::cout << "[CG DEBUG] ||b|| = " << std::sqrt(b_norm) << ", ||A*x|| = " << std::sqrt(Ax_norm) << "\n";
    }
    
    blas::axpby(ctx, 1.0, b, -1.0, r_span);
    
    // p = r
    blas::copy(ctx, r_span, p_span);
    
    // rr = dot(r, r) - device only
    blas::dot_device(ctx, r_span, r_span, d_rr_span, ws.red);
    
    // Copy initial rr to host for tolerance check (only once at start)
    real rr_host;
    RWPT_CUDA_CHECK(cudaMemcpyAsync(&rr_host, ws.d_rr.data(), sizeof(real),
                                     cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();
    
    real r0_norm = std::sqrt(rr_host);
    real tol = cfg.atol + r0_norm * cfg.rtol;
    
    result.r0_norm = r0_norm;
    
    // DEBUG: Print first few values
    if (cfg.verbose) {
        std::cout << "[CG] Initial ||r||_2 = " << r0_norm << "\n";
        std::cout << "[CG] Tolerance = " << tol << "\n";
    }
    
    // Check if already converged
    if (r0_norm <= tol) {
        result.converged = true;
        result.iters = 0;
        result.r_norm = r0_norm;
        return result;
    }
    
    // Main CG loop (ALL IN DEVICE, no host sync per iteration)
    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        // Ap = A*p
        A.apply(ctx, p_span, Ap_span);
        
        // pAp = dot(p, Ap) - device only
        blas::dot_device(ctx, p_span, Ap_span, d_pAp_span, ws.red);
        
        // DEBUG: Print values in first few iterations
        if (cfg.verbose && iter < 5) {
            real rr_debug, pAp_debug;
            RWPT_CUDA_CHECK(cudaMemcpy(&rr_debug, ws.d_rr.data(), sizeof(real), cudaMemcpyDeviceToHost));
            RWPT_CUDA_CHECK(cudaMemcpy(&pAp_debug, ws.d_pAp.data(), sizeof(real), cudaMemcpyDeviceToHost));
            ctx.synchronize();
            std::cout << "[CG iter " << iter << "] rr = " << rr_debug << ", pAp = " << pAp_debug 
                      << ", alpha = " << (rr_debug / pAp_debug) << "\n";
        }
        
        // Check for breakdown (pAp ~= 0 or NaN) every check_every
        if ((iter + 1) % cfg.check_every == 0) {
            blas::check_pAp_valid(ctx, d_pAp_span, ws.d_is_valid.span());
            int is_valid_host;
            RWPT_CUDA_CHECK(cudaMemcpyAsync(&is_valid_host, ws.d_is_valid.data(), sizeof(int),
                                             cudaMemcpyDeviceToHost, ctx.cuda_stream()));
            ctx.synchronize();
            
            if (is_valid_host == 0) {
                // Breakdown detected
                result.converged = false;
                result.iters = iter + 1;
                result.r_norm = -1.0;  // Indicate breakdown
                return result;
            }
        }
        
        // alpha = rr / pAp - device only
        blas::compute_alpha(ctx, d_rr_span, d_pAp_span, d_alpha_span);
        
        // x = x + alpha*p, r = r - alpha*Ap - fused, device only
        blas::update_x_and_r(ctx, d_alpha_span, p_span, x, Ap_span, r_span);
        
        // rr_new = dot(r, r) - device only
        blas::dot_device(ctx, r_span, r_span, d_rr_new_span, ws.red);
        
        // Check convergence every check_every iterations (requires host sync)
        if ((iter + 1) % cfg.check_every == 0) {
            real rr_new_host;
            RWPT_CUDA_CHECK(cudaMemcpyAsync(&rr_new_host, ws.d_rr_new.data(), sizeof(real),
                                             cudaMemcpyDeviceToHost, ctx.cuda_stream()));
            ctx.synchronize();
            
            real r_norm = std::sqrt(rr_new_host);
            
            if (r_norm <= tol) {
                result.converged = true;
                result.iters = iter + 1;
                result.r_norm = r_norm;
                return result;
            }
        }
        
        // beta = rr_new / rr - device only
        blas::compute_beta(ctx, d_rr_new_span, d_rr_span, d_beta_span);
        
        // p = r + beta*p - fused, device only
        blas::update_p(ctx, d_beta_span, r_span, p_span);
        
        // Swap rr and rr_new for next iteration (no kernel overhead)
        ws.d_rr.swap(ws.d_rr_new);
    }
    
    // Did not converge - need final sync to get residual
    // After swap in last iteration, current rr is in d_rr (not d_rr_new)
    real rr_final_host;
    RWPT_CUDA_CHECK(cudaMemcpyAsync(&rr_final_host, ws.d_rr.data(), sizeof(real),
                                     cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();
    
    result.converged = false;
    result.iters = cfg.max_iter;
    result.r_norm = std::sqrt(rr_final_host);
    
    return result;
}

} // namespace solvers
} // namespace rwpt
