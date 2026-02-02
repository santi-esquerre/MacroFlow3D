/**
 * @file operator_debug.cu
 * @brief Diagnostic tests for operator equivalence and symmetry
 * 
 * Task 2: Validate mathematical consistency before implementing PCG+MG
 * 
 * Tests:
 * 1. Operator equivalence: ||VarCoeff(x) - MG_residual_based(x)|| / ||VarCoeff(x)||
 * 2. Symmetry check: |u^T A v - v^T A u| / (||u|| ||v||)
 */

#include "../src/runtime/CudaContext.cuh"
#include "../src/runtime/cuda_check.cuh"
#include "../src/core/Grid3D.hpp"
#include "../src/core/DeviceBuffer.cuh"
#include "../src/core/BCSpec.hpp"
#include "../src/numerics/blas/blas.cuh"
#include "../src/numerics/operators/varcoeff_laplacian.cuh"
#include "../src/multigrid/smoothers/residual_3d.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>

using namespace rwpt;

// Fill with random values
void fill_random(std::vector<real>& v, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<real> dist(-1.0, 1.0);
    for (auto& x : v) {
        x = dist(gen);
    }
}

// Compute A*x via MG residual: if r = b - A*x with b=0, then A*x = -r
void compute_Ax_via_residual(
    CudaContext& ctx,
    const Grid3D& grid,
    DeviceSpan<const real> x,
    DeviceSpan<const real> K,
    const BCSpec& bc,
    DeviceSpan<real> Ax,  // output
    DeviceBuffer<real>& b_zero  // workspace (b=0)
) {
    // Set b = 0
    blas::fill(ctx, b_zero.span(), 0.0);
    
    // Compute r = b - A_mg*x = -A_mg*x
    multigrid::compute_residual_3d(ctx, grid, x, b_zero.span(), K, Ax, bc);
    
    // Ax = -r (because r = b - A*x = 0 - A*x = -A*x, so A*x = -r)
    blas::scal(ctx, Ax, -1.0);
}

int main() {
    std::cout << "=== Operator Diagnostic Tests ===\n\n";
    
    try {
        CudaContext ctx(0);
        
        // Grid setup
        const int N = 32;
        Grid3D grid(N, N, N, 1.0/N, 1.0/N, 1.0/N);
        size_t n = grid.num_cells();
        
        std::cout << "Grid: " << N << "^3 = " << n << " cells\n";
        std::cout << "dx = " << grid.dx << "\n\n";
        
        // BC: Dirichlet in X, Periodic in Y/Z (typical flow problem)
        BCSpec bc;
        bc.xmin = BCFace(BCType::Dirichlet, 1.0);
        bc.xmax = BCFace(BCType::Dirichlet, 0.0);
        bc.ymin = BCFace(BCType::Periodic, 0.0);
        bc.ymax = BCFace(BCType::Periodic, 0.0);
        bc.zmin = BCFace(BCType::Periodic, 0.0);
        bc.zmax = BCFace(BCType::Periodic, 0.0);
        
        std::cout << "BCs: Dirichlet(x), Periodic(y,z)\n\n";
        
        // Allocate
        DeviceBuffer<real> K(n);
        DeviceBuffer<real> x(n);
        DeviceBuffer<real> u(n);
        DeviceBuffer<real> v(n);
        DeviceBuffer<real> y_varcoeff(n);
        DeviceBuffer<real> y_mg(n);
        DeviceBuffer<real> Au(n);
        DeviceBuffer<real> Av(n);
        DeviceBuffer<real> b_zero(n);
        
        // Initialize K = 1 (homogeneous for simplicity)
        blas::fill(ctx, K.span(), 1.0);
        
        // Random vectors
        std::vector<real> x_host(n), u_host(n), v_host(n);
        fill_random(x_host, 42);
        fill_random(u_host, 123);
        fill_random(v_host, 456);
        
        cudaMemcpy(x.data(), x_host.data(), n * sizeof(real), cudaMemcpyHostToDevice);
        cudaMemcpy(u.data(), u_host.data(), n * sizeof(real), cudaMemcpyHostToDevice);
        cudaMemcpy(v.data(), v_host.data(), n * sizeof(real), cudaMemcpyHostToDevice);
        
        // ========================================
        // Test 1: Operator Equivalence
        // ========================================
        std::cout << "=== Test 1: Operator Equivalence ===\n";
        
        // VarCoeffLaplacian
        operators::VarCoeffLaplacian A_var(grid, K.span(), bc);
        A_var.apply(ctx, x.span(), y_varcoeff.span());
        
        // MG-based A*x (via residual with b=0)
        compute_Ax_via_residual(ctx, grid, x.span(), K.span(), bc, y_mg.span(), b_zero);
        
        // Compute ||y_varcoeff - y_mg||
        blas::ReductionWorkspace red;
        
        // diff = y_varcoeff - y_mg
        blas::axpy(ctx, -1.0, y_mg.span(), y_varcoeff.span());
        real diff_norm = blas::nrm2_host(ctx, y_varcoeff.span(), red);
        
        // Re-compute y_varcoeff for norm
        A_var.apply(ctx, x.span(), y_varcoeff.span());
        real varcoeff_norm = blas::nrm2_host(ctx, y_varcoeff.span(), red);
        
        real rel_diff = diff_norm / varcoeff_norm;
        
        std::cout << "  ||VarCoeff(x)|| = " << std::scientific << varcoeff_norm << "\n";
        std::cout << "  ||VarCoeff(x) - MG_Ax|| = " << diff_norm << "\n";
        std::cout << "  Relative difference = " << rel_diff << "\n";
        
        if (rel_diff < 1e-10) {
            std::cout << "  PASS: Operators are equivalent\n";
        } else if (rel_diff < 0.01) {
            std::cout << "  WARN: Small difference (BC handling?)\n";
        } else {
            std::cout << "  FAIL: Operators differ significantly!\n";
            std::cout << "  This indicates a sign or formula mismatch.\n";
            
            // Print some values for debugging
            std::vector<real> yv_h(n), ym_h(n);
            cudaMemcpy(yv_h.data(), y_varcoeff.data(), n*sizeof(real), cudaMemcpyDeviceToHost);
            
            // Re-compute MG for inspection
            compute_Ax_via_residual(ctx, grid, x.span(), K.span(), bc, y_mg.span(), b_zero);
            cudaMemcpy(ym_h.data(), y_mg.data(), n*sizeof(real), cudaMemcpyDeviceToHost);
            
            std::cout << "\n  Sample values at interior cell (N/2, N/2, N/2):\n";
            int idx = N/2 + N/2*N + N/2*N*N;
            std::cout << "    VarCoeff[" << idx << "] = " << yv_h[idx] << "\n";
            std::cout << "    MG_Ax[" << idx << "] = " << ym_h[idx] << "\n";
            std::cout << "    Ratio = " << yv_h[idx] / ym_h[idx] << "\n";
        }
        
        std::cout << "\n";
        
        // ========================================
        // Test 2: Symmetry Check
        // ========================================
        std::cout << "=== Test 2: Symmetry Check ===\n";
        
        // Compute A*u and A*v
        A_var.apply(ctx, u.span(), Au.span());
        A_var.apply(ctx, v.span(), Av.span());
        
        // Compute <u, Av> and <v, Au>
        real uAv = blas::dot_host(ctx, u.span(), Av.span(), red);
        real vAu = blas::dot_host(ctx, v.span(), Au.span(), red);
        
        real u_norm = blas::nrm2_host(ctx, u.span(), red);
        real v_norm = blas::nrm2_host(ctx, v.span(), red);
        
        real sym_error = std::abs(uAv - vAu) / (u_norm * v_norm);
        
        std::cout << "  <u, Av> = " << std::scientific << uAv << "\n";
        std::cout << "  <v, Au> = " << vAu << "\n";
        std::cout << "  |<u,Av> - <v,Au>| / (||u|| ||v||) = " << sym_error << "\n";
        
        if (sym_error < 1e-10) {
            std::cout << "  PASS: Operator is symmetric\n";
        } else {
            std::cout << "  FAIL: Operator is NOT symmetric!\n";
        }
        
        std::cout << "\n";
        
        // ========================================
        // Test 3: Definiteness Check
        // ========================================
        std::cout << "=== Test 3: Definiteness Check ===\n";
        
        // Compute <x, Ax> - should be negative for negative-definite operator
        A_var.apply(ctx, x.span(), y_varcoeff.span());
        real xAx = blas::dot_host(ctx, x.span(), y_varcoeff.span(), red);
        real x_norm_sq = blas::dot_host(ctx, x.span(), x.span(), red);
        
        std::cout << "  <x, Ax> = " << std::scientific << xAx << "\n";
        std::cout << "  ||x||^2 = " << x_norm_sq << "\n";
        std::cout << "  <x, Ax> / ||x||^2 = " << xAx / x_norm_sq << "\n";
        
        if (xAx < 0) {
            std::cout << "  Operator is NEGATIVE definite\n";
            std::cout << "  -> CG standard will FAIL (expects SPD)\n";
            std::cout << "  -> Use PCG with compatible preconditioner, or negate system\n";
        } else if (xAx > 0) {
            std::cout << "  Operator is POSITIVE definite\n";
            std::cout << "  -> CG standard should work\n";
        } else {
            std::cout << "  Operator is INDEFINITE or SINGULAR\n";
        }
        
        std::cout << "\n=== Diagnostic Summary ===\n";
        std::cout << "If Test 1 FAILS: VarCoeff and MG use different formulas\n";
        std::cout << "If Test 2 FAILS: BC handling breaks symmetry\n";
        std::cout << "If operator is NEGATIVE: need to adjust CG/PCG approach\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
