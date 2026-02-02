/**
 * @file mg_cg_compare.cu
 * @brief Quantitative regression test: compare MG and CG solvers
 * 
 * This test verifies that:
 * 1. Both MG and CG converge to the same solution (within tolerance)
 * 2. Residual norms decrease as expected
 * 3. The variable-coefficient operator in CG matches MG semantics
 * 
 * Test problem: Poisson equation with manufactured solution
 *   -∇·(K∇u) = f in [0,1]³
 *   u = 0 on boundary (Dirichlet)
 * 
 * With u_exact = sin(πx)sin(πy)sin(πz) and K=1 (homogeneous),
 * the RHS is f = 3π² sin(πx)sin(πy)sin(πz).
 */

#include "../src/runtime/CudaContext.cuh"
#include "../src/runtime/GpuTimer.cuh"
#include "../src/multigrid/multigrid.cuh"
#include "../src/numerics/blas/blas.cuh"
#include "../src/numerics/operators/operators.cuh"
#include "../src/numerics/solvers/solvers.cuh"
#include "../src/physics/flow/coarsen_K.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace rwpt;
using namespace rwpt::multigrid;

// Compute RHS for manufactured solution on host
void compute_manufactured_rhs(
    const Grid3D& grid,
    std::vector<real>& b_host
) {
    const real pi = 3.14159265358979323846;
    const real coef = 3.0 * pi * pi;  // For -Δu = 3π²u when u = sin(πx)sin(πy)sin(πz)
    
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                // Cell-centered coordinates
                real x = (i + 0.5) * grid.dx;
                real y = (j + 0.5) * grid.dy;
                real z = (k + 0.5) * grid.dz;
                
                size_t idx = i + grid.nx * (j + grid.ny * k);
                b_host[idx] = coef * std::sin(pi * x) * std::sin(pi * y) * std::sin(pi * z);
            }
        }
    }
}

// Compute L2 error against exact solution
real compute_l2_error(
    CudaContext& ctx,
    const Grid3D& grid,
    DeviceSpan<const real> x_computed
) {
    const real pi = 3.14159265358979323846;
    size_t n = grid.num_cells();
    
    // Copy solution to host
    std::vector<real> x_host(n);
    cudaMemcpy(x_host.data(), x_computed.data(), n * sizeof(real), cudaMemcpyDeviceToHost);
    
    // Compute L2 error
    real l2_error_sq = 0.0;
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                real x = (i + 0.5) * grid.dx;
                real y = (j + 0.5) * grid.dy;
                real z = (k + 0.5) * grid.dz;
                
                real u_exact = std::sin(pi * x) * std::sin(pi * y) * std::sin(pi * z);
                size_t idx = i + grid.nx * (j + grid.ny * k);
                real diff = x_host[idx] - u_exact;
                l2_error_sq += diff * diff;
            }
        }
    }
    
    real dV = grid.dx * grid.dy * grid.dz;
    return std::sqrt(l2_error_sq * dV);
}

int main() {
    std::cout << "=== MG vs CG Quantitative Comparison Test ===\n\n";
    
    bool all_passed = true;
    
    try {
        CudaContext ctx(0);
        
        // Test grid: 32^3
        const int N = 32;
        Grid3D grid(N, N, N, 1.0/N, 1.0/N, 1.0/N);
        size_t n = grid.num_cells();
        
        std::cout << "Grid: " << N << "³ = " << n << " cells\n";
        std::cout << "dx = " << grid.dx << "\n\n";
        
        // Boundary conditions: Dirichlet zero
        BCSpec bc;
        bc.xmin = BCFace(BCType::Dirichlet, 0.0);
        bc.xmax = BCFace(BCType::Dirichlet, 0.0);
        bc.ymin = BCFace(BCType::Dirichlet, 0.0);
        bc.ymax = BCFace(BCType::Dirichlet, 0.0);
        bc.zmin = BCFace(BCType::Dirichlet, 0.0);
        bc.zmax = BCFace(BCType::Dirichlet, 0.0);
        
        // Allocate buffers
        DeviceBuffer<real> b(n);
        DeviceBuffer<real> x_mg(n);
        DeviceBuffer<real> x_cg(n);
        DeviceBuffer<real> K(n);
        
        // Set K = 1 (homogeneous)
        blas::fill(ctx, K.span(), 1.0);
        
        // Compute manufactured RHS
        std::vector<real> b_host(n);
        compute_manufactured_rhs(grid, b_host);
        cudaMemcpy(b.data(), b_host.data(), n * sizeof(real), cudaMemcpyHostToDevice);
        
        // ========================================
        // Test 1: MG Solve
        // ========================================
        std::cout << "=== Test 1: Multigrid V-cycle ===\n";
        
        // Create MG hierarchy
        int num_levels = 4;  // 32->16->8->4
        MGHierarchy hier(grid, num_levels);
        
        // Copy K and b to finest level
        cudaMemcpy(hier.levels[0].K.data(), K.data(), n * sizeof(real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(hier.levels[0].b.data(), b.data(), n * sizeof(real), cudaMemcpyDeviceToDevice);
        blas::fill(ctx, hier.levels[0].x.span(), 0.0);
        
        // Homogenize K to coarser levels
        for (int l = 1; l < hier.num_levels(); ++l) {
            physics::coarsen_K(ctx, hier.levels[l].grid, hier.levels[l-1].grid,
                               hier.levels[l-1].K.span(), hier.levels[l].K.span());
        }
        
        // Configure MG
        MGConfig mg_cfg;
        mg_cfg.num_levels = num_levels;
        mg_cfg.pre_smooth = 2;
        mg_cfg.post_smooth = 2;
        mg_cfg.coarse_solve_iters = 50;
        mg_cfg.check_convergence_every = 1;
        
        // Solve
        GpuTimer timer;
        timer.start(ctx.cuda_stream());
        auto mg_result = mg_solve(ctx, hier, mg_cfg, bc, 20, 1e-8);
        float mg_time = timer.stop(ctx.cuda_stream());
        ctx.synchronize();
        
        // Copy solution
        cudaMemcpy(x_mg.data(), hier.levels[0].x.data(), n * sizeof(real), cudaMemcpyDeviceToDevice);
        
        std::cout << "  Converged: " << (mg_result.converged ? "YES" : "NO") << "\n";
        std::cout << "  Cycles: " << mg_result.num_cycles << "\n";
        std::cout << "  Initial residual: " << std::scientific << mg_result.initial_residual << "\n";
        std::cout << "  Final residual: " << mg_result.final_residual << "\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(2) << mg_time << " ms\n";
        
        real mg_l2_error = compute_l2_error(ctx, grid, x_mg.span());
        std::cout << "  L2 error vs exact: " << std::scientific << mg_l2_error << "\n\n";
        
        if (!mg_result.converged) {
            std::cout << "  FAIL: MG did not converge\n";
            all_passed = false;
        }
        
        // ========================================
        // Test 2: CG Solve with VarCoeffLaplacian
        // ========================================
        std::cout << "=== Test 2: CG with VarCoeffLaplacian ===\n";
        
        // Create operator
        operators::VarCoeffLaplacian A_varcoeff(grid, K.span(), bc);
        
        // Initialize x = 0
        blas::fill(ctx, x_cg.span(), 0.0);
        
        // Configure CG
        solvers::CGConfig cg_cfg;
        cg_cfg.max_iter = 1000;
        cg_cfg.rtol = 1e-8;
        cg_cfg.atol = 0.0;
        cg_cfg.check_every = 1;  // Check every iteration to catch early convergence
        
        solvers::CGWorkspace ws;
        
        // Solve
        timer.start(ctx.cuda_stream());
        auto cg_result = solvers::cg_solve(ctx, A_varcoeff, b.span(), x_cg.span(), cg_cfg, ws);
        float cg_time = timer.stop(ctx.cuda_stream());
        ctx.synchronize();
        
        std::cout << "  Converged: " << (cg_result.converged ? "YES" : "NO") << "\n";
        std::cout << "  Iterations: " << cg_result.iters << "\n";
        std::cout << "  Initial residual: " << std::scientific << cg_result.r0_norm << "\n";
        std::cout << "  Final residual: " << cg_result.r_norm << "\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(2) << cg_time << " ms\n";
        
        real cg_l2_error = compute_l2_error(ctx, grid, x_cg.span());
        std::cout << "  L2 error vs exact: " << std::scientific << cg_l2_error << "\n\n";
        
        if (!cg_result.converged) {
            std::cout << "  FAIL: CG did not converge\n";
            all_passed = false;
        }
        
        // ========================================
        // Test 3: Compare MG vs CG solutions
        // ========================================
        std::cout << "=== Test 3: Solution comparison ===\n";
        
        // Compute ||x_mg - x_cg||
        blas::ReductionWorkspace red;
        
        // x_cg = x_cg - x_mg (reuse buffer)
        blas::axpy(ctx, -1.0, x_mg.span(), x_cg.span());
        real diff_norm = blas::nrm2_host(ctx, x_cg.span(), red);
        
        // Get norm of MG solution for relative comparison
        real mg_norm = blas::nrm2_host(ctx, x_mg.span(), red);
        real rel_diff = diff_norm / mg_norm;
        
        std::cout << "  ||x_mg - x_cg|| = " << std::scientific << diff_norm << "\n";
        std::cout << "  ||x_mg|| = " << mg_norm << "\n";
        std::cout << "  Relative diff = " << rel_diff << "\n";
        
        // Solutions should match to solver tolerance
        if (rel_diff > 1e-6) {
            std::cout << "  FAIL: Solutions differ too much\n";
            all_passed = false;
        } else {
            std::cout << "  PASS: Solutions match within tolerance\n";
        }
        
        // ========================================
        // Test 4: Verify discretization error
        // ========================================
        std::cout << "\n=== Test 4: Discretization error check ===\n";
        
        // For 2nd-order FD on 32^3 grid with dx~0.031, expect O(dx²) ~ O(1e-3) L2 error
        // The exact value depends on the manufactured solution smoothness
        real expected_max_error = 0.02;  // Generous bound for 32^3
        
        std::cout << "  MG L2 error: " << std::scientific << mg_l2_error << "\n";
        std::cout << "  Expected max: " << expected_max_error << "\n";
        
        if (mg_l2_error > expected_max_error) {
            std::cout << "  FAIL: Discretization error too high\n";
            all_passed = false;
        } else {
            std::cout << "  PASS: Discretization error within expected range\n";
        }
        
        // ========================================
        // Summary
        // ========================================
        std::cout << "\n========================================\n";
        if (all_passed) {
            std::cout << "ALL TESTS PASSED\n";
        } else {
            std::cout << "SOME TESTS FAILED\n";
        }
        std::cout << "========================================\n";
        
        return all_passed ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
