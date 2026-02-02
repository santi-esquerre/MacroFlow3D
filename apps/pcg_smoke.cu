/**
 * @file pcg_smoke.cu
 * @brief Smoke test for PCG with MG preconditioner
 * 
 * Tests:
 * 1. PCG with identity preconditioner (should work with NegatedOperator)
 * 2. PCG with MG preconditioner (should work directly with negative operator)
 */

#include "../src/runtime/CudaContext.cuh"
#include "../src/runtime/GpuTimer.cuh"
#include "../src/runtime/cuda_check.cuh"
#include "../src/core/Grid3D.hpp"
#include "../src/core/DeviceBuffer.cuh"
#include "../src/core/BCSpec.hpp"
#include "../src/numerics/blas/blas.cuh"
#include "../src/numerics/operators/operators.cuh"
#include "../src/numerics/solvers/pcg.cuh"
#include "../src/numerics/solvers/preconditioner.cuh"
#include "../src/numerics/solvers/mg_preconditioner.cuh"
#include "../src/numerics/pin_spec.hpp"
#include "../src/multigrid/multigrid.cuh"
#include "../src/physics/flow/coarsen_K.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace rwpt;

// Compute RHS for manufactured solution
void compute_manufactured_rhs(const Grid3D& grid, std::vector<real>& b_host) {
    const real pi = 3.14159265358979323846;
    const real coef = 3.0 * pi * pi;
    
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                real x = (i + 0.5) * grid.dx;
                real y = (j + 0.5) * grid.dy;
                real z = (k + 0.5) * grid.dz;
                size_t idx = i + grid.nx * (j + grid.ny * k);
                b_host[idx] = coef * std::sin(pi*x) * std::sin(pi*y) * std::sin(pi*z);
            }
        }
    }
}

int main() {
    std::cout << "=== PCG Smoke Test ===\n\n";
    
    try {
        CudaContext ctx(0);
        
        // Grid setup
        const int N = 32;
        Grid3D grid(N, N, N, 1.0/N, 1.0/N, 1.0/N);
        size_t n = grid.num_cells();
        
        std::cout << "Grid: " << N << "^3 = " << n << " cells\n";
        
        // BC: All Dirichlet zero (standard Poisson)
        BCSpec bc;
        bc.xmin = BCFace(BCType::Dirichlet, 0.0);
        bc.xmax = BCFace(BCType::Dirichlet, 0.0);
        bc.ymin = BCFace(BCType::Dirichlet, 0.0);
        bc.ymax = BCFace(BCType::Dirichlet, 0.0);
        bc.zmin = BCFace(BCType::Dirichlet, 0.0);
        bc.zmax = BCFace(BCType::Dirichlet, 0.0);
        
        std::cout << "BCs: All Dirichlet zero\n\n";
        
        // Allocate
        DeviceBuffer<real> K(n);
        DeviceBuffer<real> x(n);
        DeviceBuffer<real> b(n);
        DeviceBuffer<real> b_neg(n);  // Negated RHS for positive operator
        
        // K = 1
        blas::fill(ctx, K.span(), 1.0);
        
        // Compute RHS for manufactured solution
        std::vector<real> b_host(n);
        compute_manufactured_rhs(grid, b_host);
        cudaMemcpy(b.data(), b_host.data(), n * sizeof(real), cudaMemcpyHostToDevice);
        
        // Workspace
        solvers::PCGWorkspace ws;
        blas::ReductionWorkspace red;
        
        // ========================================
        // Test 1: PCG with Identity Preconditioner
        // Need to use NegatedOperator because A is negative definite
        // ========================================
        std::cout << "=== Test 1: PCG + Identity (with NegatedOperator) ===\n";
        
        // Create positive operator: A_pos = -A_neg
        operators::VarCoeffLaplacian A_neg(grid, K.span(), bc);
        operators::NegatedOperator<operators::VarCoeffLaplacian> A_pos(A_neg);
        
        // Negate RHS: b_pos = -b
        blas::copy(ctx, b.span(), b_neg.span());
        blas::scal(ctx, b_neg.span(), -1.0);
        
        // Reset x = 0
        blas::fill(ctx, x.span(), 0.0);
        
        // Identity preconditioner
        solvers::IdentityPreconditioner M_id;
        
        // PCG config
        solvers::PCGConfig pcg_cfg;
        pcg_cfg.max_iter = 200;
        pcg_cfg.check_every = 10;
        pcg_cfg.rtol = 1e-6;
        pcg_cfg.verbose = false;
        
        GpuTimer timer;
        timer.start(ctx.cuda_stream());
        
        solvers::PCGResult result1 = solvers::pcg_solve(
            ctx, A_pos, M_id, b_neg.span(), x.span(), pcg_cfg, ws
        );
        
        float time1 = timer.stop(ctx.cuda_stream());
        
        std::cout << "  Converged: " << (result1.converged ? "YES" : "NO") << "\n";
        std::cout << "  Iterations: " << result1.iterations << "\n";
        std::cout << "  Initial residual: " << std::scientific << result1.initial_residual << "\n";
        std::cout << "  Final residual: " << result1.final_residual << "\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(2) << time1 << " ms\n\n";
        
        // ========================================
        // Test 2: PCG with MG Preconditioner
        // Should work directly with negative operator (both A and M^{-1} are negative)
        // ========================================
        std::cout << "=== Test 2: PCG + MG (direct with negative operator) ===\n";
        
        // Create MG hierarchy
        int num_levels = 4;  // 32->16->8->4
        multigrid::MGHierarchy hier(grid, num_levels);
        
        // Copy K to finest level and coarsen
        cudaMemcpy(hier.levels[0].K.data(), K.data(), n * sizeof(real), cudaMemcpyDeviceToDevice);
        for (int l = 1; l < hier.num_levels(); ++l) {
            physics::coarsen_K(ctx, hier.levels[l].grid, hier.levels[l-1].grid,
                               hier.levels[l-1].K.span(), hier.levels[l].K.span());
        }
        
        // MG config
        multigrid::MGConfig mg_cfg;
        mg_cfg.num_levels = num_levels;
        mg_cfg.pre_smooth = 2;
        mg_cfg.post_smooth = 2;
        mg_cfg.coarse_solve_iters = 50;
        
        // MG preconditioner (pin disabled for this test)
        solvers::MultigridPreconditioner M_mg(hier, bc, mg_cfg, PinSpec{});
        
        // Reset x = 0
        blas::fill(ctx, x.span(), 0.0);
        
        // Note: Using A_neg directly (not A_pos), and original RHS b (not negated)
        // This works because M_mg also "inverts" A_neg, so both are consistent
        timer.start(ctx.cuda_stream());
        
        solvers::PCGResult result2 = solvers::pcg_solve(
            ctx, A_neg, M_mg, b.span(), x.span(), pcg_cfg, ws
        );
        
        float time2 = timer.stop(ctx.cuda_stream());
        
        std::cout << "  Converged: " << (result2.converged ? "YES" : "NO") << "\n";
        std::cout << "  Iterations: " << result2.iterations << "\n";
        std::cout << "  Initial residual: " << std::scientific << result2.initial_residual << "\n";
        std::cout << "  Final residual: " << result2.final_residual << "\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(2) << time2 << " ms\n\n";
        
        // ========================================
        // Test 3: Verify <r, z> sign with MG preconditioner
        // Should be NEGATIVE (both r and z from negative operator)
        // ========================================
        std::cout << "=== Test 3: Sign check for <r, z> with MG ===\n";
        
        DeviceBuffer<real> r_test(n);
        DeviceBuffer<real> z_test(n);
        
        // Set r to a random-ish pattern (use b as proxy)
        blas::copy(ctx, b.span(), r_test.span());
        
        // Apply MG preconditioner: z = M^{-1} r
        M_mg.apply(ctx, r_test.span(), z_test.span());
        
        // Compute <r, z>
        real rz = blas::dot_host(ctx, r_test.span(), z_test.span(), red);
        
        std::cout << "  <r, z> = " << std::scientific << rz << "\n";
        if (rz < 0) {
            std::cout << "  Sign is NEGATIVE -> consistent with negative-definite operator\n";
            std::cout << "  PCG should work because alpha = <r,z>/<p,Ap> = neg/neg = pos\n";
        } else {
            std::cout << "  Sign is POSITIVE -> inconsistent!\n";
            std::cout << "  This may cause PCG to diverge.\n";
        }
        
        std::cout << "\n=== Summary ===\n";
        std::cout << "Test 1 (PCG+Identity with NegatedOp): " << (result1.converged ? "PASS" : "FAIL") << "\n";
        std::cout << "Test 2 (PCG+MG direct): " << (result2.converged ? "PASS" : "FAIL") << "\n";
        
        return (result1.converged && result2.converged) ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
