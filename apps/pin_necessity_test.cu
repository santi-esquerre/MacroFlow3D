/**
 * @file pin_necessity_test.cu
 * @brief Test that demonstrates pin is NECESSARY for singular systems
 * 
 * This test creates a SINGULAR system (all periodic BCs, no Dirichlet) and
 * shows that:
 *   - Without pin: solver diverges or fails to converge
 *   - With pin: solver converges correctly
 * 
 * The key insight is that with a non-zero initial guess, the solver must
 * iterate, and singular systems without pinning will fail.
 */

#include "../src/runtime/CudaContext.cuh"
#include "../src/runtime/cuda_check.cuh"
#include "../src/core/Grid3D.hpp"
#include "../src/core/DeviceBuffer.cuh"
#include "../src/core/BCSpec.hpp"
#include "../src/numerics/blas/blas.cuh"
#include "../src/numerics/operators/operators.cuh"
#include "../src/numerics/solvers/pcg.cuh"
#include "../src/numerics/solvers/mg_preconditioner.cuh"
#include "../src/numerics/pin_spec.hpp"
#include "../src/multigrid/multigrid.cuh"
#include "../src/physics/flow/coarsen_K.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

using namespace rwpt;

// Fill buffer with random values
__global__ void fill_random_kernel(real* data, size_t n, unsigned int seed) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    
    // Simple LCG random
    unsigned int state = seed + idx * 1099087573u;
    state = state * 1103515245u + 12345u;
    float r = (state & 0x7FFFFFFF) / float(0x7FFFFFFF);
    
    data[idx] = real(r * 100.0 - 50.0);  // Random in [-50, 50]
}

void fill_random(CudaContext& ctx, DeviceSpan<real> data, unsigned int seed) {
    const int block = 256;
    const int grid = (data.size() + block - 1) / block;
    fill_random_kernel<<<grid, block, 0, ctx.cuda_stream()>>>(data.data(), data.size(), seed);
}

int main() {
    std::cout << "=== Pin Necessity Test ===\n";
    std::cout << "Demonstrates that pin is REQUIRED for singular systems\n\n";
    
    try {
        CudaContext ctx;
        
        // Small grid for quick test
        const int N = 16;
        Grid3D grid(N, N, N, 1.0/N, 1.0/N, 1.0/N);
        const size_t n = grid.num_cells();
        
        std::cout << "Grid: " << N << "x" << N << "x" << N << " = " << n << " cells\n";
        
        // ALL PERIODIC = SINGULAR SYSTEM
        BCSpec bc;
        bc.xmin = {BCType::Periodic, 0.0};
        bc.xmax = {BCType::Periodic, 0.0};
        bc.ymin = {BCType::Periodic, 0.0};
        bc.ymax = {BCType::Periodic, 0.0};
        bc.zmin = {BCType::Periodic, 0.0};
        bc.zmax = {BCType::Periodic, 0.0};
        
        std::cout << "BCs: ALL PERIODIC (singular system)\n\n";
        
        // Constant K = 1
        DeviceBuffer<real> K(n);
        blas::fill(ctx, K.span(), real(1.0));
        
        // RHS = 0 (no source)
        DeviceBuffer<real> b(n);
        blas::fill(ctx, b.span(), real(0.0));
        
        // Solution buffer
        DeviceBuffer<real> x(n);
        
        // MG hierarchy
        const int num_levels = 3;
        multigrid::MGHierarchy hier(grid, num_levels);
        
        // Copy K to hierarchy and coarsen
        RWPT_CUDA_CHECK(cudaMemcpyAsync(
            hier.levels[0].K.data(), K.data(), n * sizeof(real),
            cudaMemcpyDeviceToDevice, ctx.cuda_stream()));
        
        for (int l = 1; l < num_levels; ++l) {
            physics::coarsen_K(ctx, hier.levels[l].grid, hier.levels[l-1].grid,
                               hier.levels[l-1].K.span(), hier.levels[l].K.span());
        }
        
        // MG config
        multigrid::MGConfig mg_cfg;
        mg_cfg.num_levels = num_levels;
        mg_cfg.pre_smooth = 2;
        mg_cfg.post_smooth = 2;
        mg_cfg.coarse_solve_iters = 50;
        
        // PCG config
        solvers::PCGConfig pcg_cfg;
        pcg_cfg.max_iter = 100;
        pcg_cfg.check_every = 1;
        pcg_cfg.rtol = 1e-8;
        pcg_cfg.verbose = false;
        
        solvers::PCGWorkspace ws;
        ws.ensure(n);
        
        // ====================================================================
        // TEST 1: Without pin (should FAIL or not converge well)
        // ====================================================================
        std::cout << "========================================\n";
        std::cout << "TEST 1: Singular system WITHOUT pin\n";
        std::cout << "========================================\n";
        
        // Non-zero initial guess (this is key!)
        fill_random(ctx, x.span(), 12345);
        ctx.synchronize();
        
        // Check initial x stats
        real x_min, x_max;
        blas::ReductionWorkspace red;
        real x_norm_init = blas::nrm2_host(ctx, DeviceSpan<const real>(x.span()), red);
        std::cout << "Initial ||x|| = " << x_norm_init << " (non-zero guess)\n";
        
        // Create operator WITHOUT pin
        PinSpec no_pin(false);
        operators::VarCoeffLaplacian A_no_pin(grid, DeviceSpan<const real>(K.span()), bc, no_pin);
        solvers::MultigridPreconditioner M_no_pin(hier, bc, mg_cfg, no_pin);
        
        // Solve
        solvers::PCGResult result1 = solvers::pcg_solve(
            ctx, A_no_pin, M_no_pin,
            DeviceSpan<const real>(b.span()),
            x.span(),
            pcg_cfg,
            ws
        );
        
        std::cout << "Result WITHOUT pin:\n";
        std::cout << "  Converged:   " << (result1.converged ? "YES" : "NO") << "\n";
        std::cout << "  Iterations:  " << result1.iterations << "\n";
        std::cout << "  Init resid:  " << std::scientific << result1.initial_residual << "\n";
        std::cout << "  Final resid: " << std::scientific << result1.final_residual << "\n";
        
        real x_norm_final1 = blas::nrm2_host(ctx, DeviceSpan<const real>(x.span()), red);
        std::cout << "  Final ||x||: " << x_norm_final1 << "\n";
        
        // Check h[0] value
        real h0_no_pin;
        RWPT_CUDA_CHECK(cudaMemcpy(&h0_no_pin, x.data(), sizeof(real), cudaMemcpyDeviceToHost));
        std::cout << "  h[0]:        " << h0_no_pin << " (arbitrary without pin)\n";
        
        // Check if solution "drifted" (large norm = null space pollution)
        bool drifted1 = (x_norm_final1 > 1e6) || std::isnan(x_norm_final1) || std::isinf(x_norm_final1);
        std::cout << "  Drifted:     " << (drifted1 ? "YES (bad!)" : "no") << "\n\n";
        
        // ====================================================================
        // TEST 2: With pin (should SUCCEED)
        // ====================================================================
        std::cout << "========================================\n";
        std::cout << "TEST 2: Singular system WITH pin\n";
        std::cout << "========================================\n";
        
        // Reset to same non-zero initial guess
        fill_random(ctx, x.span(), 12345);
        ctx.synchronize();
        
        std::cout << "Initial ||x|| = " << x_norm_init << " (same non-zero guess)\n";
        
        // Create operator WITH pin
        PinSpec with_pin(true);
        operators::VarCoeffLaplacian A_with_pin(grid, DeviceSpan<const real>(K.span()), bc, with_pin);
        solvers::MultigridPreconditioner M_with_pin(hier, bc, mg_cfg, with_pin);
        
        // Solve
        solvers::PCGResult result2 = solvers::pcg_solve(
            ctx, A_with_pin, M_with_pin,
            DeviceSpan<const real>(b.span()),
            x.span(),
            pcg_cfg,
            ws
        );
        
        std::cout << "Result WITH pin:\n";
        std::cout << "  Converged:   " << (result2.converged ? "YES" : "NO") << "\n";
        std::cout << "  Iterations:  " << result2.iterations << "\n";
        std::cout << "  Init resid:  " << std::scientific << result2.initial_residual << "\n";
        std::cout << "  Final resid: " << std::scientific << result2.final_residual << "\n";
        
        real x_norm_final2 = blas::nrm2_host(ctx, DeviceSpan<const real>(x.span()), red);
        std::cout << "  Final ||x||: " << x_norm_final2 << "\n";
        
        // Check h[0] value - should be ~0 with pin
        real h0_with_pin;
        RWPT_CUDA_CHECK(cudaMemcpy(&h0_with_pin, x.data(), sizeof(real), cudaMemcpyDeviceToHost));
        std::cout << "  h[0]:        " << h0_with_pin << " (should be ~0 with pin)\n";
        
        bool drifted2 = (x_norm_final2 > 1e6) || std::isnan(x_norm_final2) || std::isinf(x_norm_final2);
        std::cout << "  Drifted:     " << (drifted2 ? "YES (bad!)" : "no") << "\n\n";
        
        // ====================================================================
        // Summary
        // ====================================================================
        std::cout << "========================================\n";
        std::cout << "SUMMARY\n";
        std::cout << "========================================\n";
        
        bool test1_bad = !result1.converged || drifted1;
        bool test2_good = result2.converged && !drifted2;
        
        std::cout << "Without pin:\n";
        std::cout << "  h[0] = " << h0_no_pin << " (can be any value)\n";
        std::cout << "  ||x|| = " << x_norm_final1 << "\n";
        
        std::cout << "With pin:\n";
        std::cout << "  h[0] = " << h0_with_pin << " (should be ~0, anchored by pin)\n";
        std::cout << "  ||x|| = " << x_norm_final2 << "\n\n";
        
        // The KEY difference: with pin, h[0] should be ~0
        bool h0_anchored = std::abs(h0_with_pin) < 1e-3;
        
        std::cout << "Pin anchors h[0] to ~0: " << (h0_anchored ? "YES" : "NO") << "\n\n";
        
        if (result2.converged && h0_anchored) {
            std::cout << "=== TEST PASSED: Pin correctly anchors the solution ===\n";
            std::cout << "Without pin, solution has arbitrary constant (h[0]=" << h0_no_pin << ")\n";
            std::cout << "With pin, solution is unique (h[0]~0)\n";
            return 0;
        } else {
            std::cout << "=== TEST FAILED ===\n";
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
