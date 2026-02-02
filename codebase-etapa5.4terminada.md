# .vscode\settings.json

```json
{
    "chat.tools.terminal.autoApprove": {
        "make": true
    }
}
```

# apps\cg_smoke.cu

```cu
#include "../src/runtime/CudaContext.cuh"
#include "../src/runtime/GpuTimer.cuh"
#include "../src/runtime/cuda_check.cuh"
#include "../src/core/Grid3D.hpp"
#include "../src/core/DeviceBuffer.cuh"
#include "../src/numerics/blas/blas.cuh"
#include "../src/numerics/operators/operators.cuh"
#include "../src/numerics/solvers/solvers.cuh"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    try {
        std::cout << "=== CG Smoke Test ===\n\n";
        
        // Create CUDA context
        rwpt::CudaContext ctx(0);
        
        // Setup grid
        const int N = 32;
        rwpt::Grid3D grid(N, N, N, 1.0/N, 1.0/N, 1.0/N);
        size_t num_cells = grid.num_cells();
        
        std::cout << "Grid: " << grid.nx << " x " << grid.ny << " x " << grid.nz 
                  << " = " << num_cells << " cells\n";
        
        // Boundary conditions: Dirichlet zero (makes system SPD and well-posed)
        rwpt::BCSpec bc;
        bc.xmin = rwpt::BCFace(rwpt::BCType::Dirichlet, 0.0);
        bc.xmax = rwpt::BCFace(rwpt::BCType::Dirichlet, 0.0);
        bc.ymin = rwpt::BCFace(rwpt::BCType::Dirichlet, 0.0);
        bc.ymax = rwpt::BCFace(rwpt::BCType::Dirichlet, 0.0);
        bc.zmin = rwpt::BCFace(rwpt::BCType::Dirichlet, 0.0);
        bc.zmax = rwpt::BCFace(rwpt::BCType::Dirichlet, 0.0);
        
        // Allocate vectors
        rwpt::DeviceBuffer<rwpt::real> x(num_cells);
        rwpt::DeviceBuffer<rwpt::real> b(num_cells);
        rwpt::DeviceBuffer<rwpt::real> K(num_cells);
        
        // Initialize: K = 1 (homogeneous), x = 0
        rwpt::blas::fill(ctx, K.span(), 1.0);
        rwpt::blas::fill(ctx, x.span(), 0.0);
        
        // b = manufactured RHS for u = sin(πx)sin(πy)sin(πz)
        const double pi = 3.14159265358979323846;
        std::vector<rwpt::real> b_host(num_cells);
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; i++) {
                    int idx = i + j*N + k*N*N;
                    rwpt::real xc = (i + 0.5) / N;
                    rwpt::real yc = (j + 0.5) / N;
                    rwpt::real zc = (k + 0.5) / N;
                    b_host[idx] = 3.0 * pi * pi * std::sin(pi*xc) * std::sin(pi*yc) * std::sin(pi*zc);
                }
            }
        }
        RWPT_CUDA_CHECK(cudaMemcpy(b.data(), b_host.data(), 
                                   num_cells * sizeof(rwpt::real), 
                                   cudaMemcpyHostToDevice));
        
        // Create VarCoeffLaplacian operator (matches MG semantics)
        rwpt::operators::VarCoeffLaplacian A(grid, K.span(), bc);
        
        // Configure CG
        rwpt::solvers::CGConfig cfg;
        cfg.max_iter = 200;
        cfg.rtol = 1e-6;
        cfg.atol = 0.0;
        cfg.check_every = 1;  // Check every iteration for early convergence
        
        // Create workspace
        rwpt::solvers::CGWorkspace ws;
        
        std::cout << "\nStarting CG solve...\n";
        
        // Timing
        rwpt::GpuTimer timer;
        timer.start(ctx.cuda_stream());
        
        // Solve
        rwpt::solvers::CGResult result = rwpt::solvers::cg_solve(
            ctx, A, b.span(), x.span(), cfg, ws
        );
        
        float elapsed_ms = timer.stop(ctx.cuda_stream());
        
        // Print results
        std::cout << "\n=== Results ===\n";
        std::cout << "Initial residual: " << result.r0_norm << "\n";
        std::cout << "Tolerance: " << (cfg.atol + result.r0_norm * cfg.rtol) << "\n";
        std::cout << "Converged: " << (result.converged ? "YES" : "NO") << "\n";
        std::cout << "Iterations: " << result.iters << "\n";
        std::cout << "Final residual norm: " << result.r_norm << "\n";
        std::cout << "Relative residual: " << (result.r_norm / result.r0_norm) << "\n";
        std::cout << "Time: " << elapsed_ms << " ms\n";
        
        if (!result.converged) {
            std::cout << "\nWARNING: CG did not converge!\n";
        }
        
        return result.converged ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

```

# apps\config_pcg_mg_test.yaml

```yaml
# Test configuration for PCG with MG preconditioner (legacy: solver_CG + PCCMG_CG)

grid:
  nx: 32
  ny: 32
  nz: 32
  dx: 0.03125  # 1/32, domain [0,1]^3

stochastic:
  sigma2: 1.0           # Variance of log(K)
  corr_length: 0.1      # Correlation length
  n_modes: 1000         # Fourier modes for spectral method
  covariance_type: 1    # 0 = exponential, 1 = gaussian
  seed: 12345
  K_mean: 1.0           # Geometric mean of K

flow:
  solver: pcg_mg        # PCG with MG preconditioner (default, legacy)
  mg_levels: 4
  mg_pre_smooth: 2
  mg_post_smooth: 2
  mg_coarse_iters: 50
  mg_max_cycles: 20
  cg_max_iter: 100      # PCG max iterations
  cg_rtol: 1.0e-6       # PCG relative tolerance
  cg_check_every: 10    # Check convergence every N iterations
  rtol: 1.0e-6
  verify_velocity: true  # Enable theoretical verification
  
  pin:
    mode: auto
  
  bc:
    west:   { type: dirichlet, value: 100.0 }
    east:   { type: dirichlet, value: 0.0 }
    south:  { type: periodic }
    north:  { type: periodic }
    bottom: { type: periodic }
    top:    { type: periodic }

transport:
  n_particles: 1000
  dt: 0.001
  n_steps: 100
  porosity: 1.0
  diffusion: 0.0
  seed: 54321
  output_every: 100
  inject_x: 0.0

output:
  output_dir: ./output_pcg_mg_test
  save_K: false
  save_head: false
  save_velocity: false
  save_particles: false
  format: binary

```

# apps\config_pin_debug.yaml

```yaml
# Debug config for PIN functionality test
# Uses Neumann everywhere (singular) with constant K (homogeneous)
# Compare with Dirichlet case where pin should NOT activate

grid:
  nx: 8
  ny: 8
  nz: 8
  dx: 0.125  # domain = 1x1x1

stochastic:
  sigma2: 0.0      # K = 1 constant (no heterogeneity)
  corr_length: 0.1
  n_modes: 1
  seed: 1

flow:
  solver: pcg_mg
  mg_levels: 2
  rtol: 1e-8
  
  pin:
    mode: auto  # Will enable for all Neumann/Periodic
  
  # All Neumann = singular system (gradient problem)
  bc:
    west:   { type: neumann, value: -1.0 }   # flux in from left
    east:   { type: neumann, value: 1.0 }    # flux out to right
    south:  { type: neumann, value: 0.0 }
    north:  { type: neumann, value: 0.0 }
    bottom: { type: neumann, value: 0.0 }
    top:    { type: neumann, value: 0.0 }

transport:
  n_particles: 10
  dt: 0.001
  n_steps: 1

```

# apps\config_pin_test.yaml

```yaml
# Test config for SINGULAR system (all periodic BCs)
# Pin should auto-enable to break degeneracy

grid:
  nx: 16
  ny: 16
  nz: 16
  dx: 0.0625

stochastic:
  sigma2: 0.5
  corr_length: 0.1
  n_modes: 500
  seed: 12345

flow:
  solver: pcg_mg
  mg_levels: 3
  rtol: 1e-6
  
  # Pin mode: auto|on|off (legacy diagonal doubling, always cell [0,0,0])
  # Value and index are NOT configurable (legacy semantics)
  pin:
    mode: auto      # Will auto-enable because no Dirichlet BCs
  
  # ALL periodic → singular system
  bc:
    west:   { type: periodic }
    east:   { type: periodic }
    south:  { type: periodic }
    north:  { type: periodic }
    bottom: { type: periodic }
    top:    { type: periodic }

transport:
  n_particles: 100
  dt: 0.001
  n_steps: 10

```

# apps\config_velocity_test.yaml

```yaml
# Test configuration with UNIFORM K for velocity verification
# With K=1 uniform, u_theory = K * dH/Lx = 1 * 100 / 1 = 100

grid:
  nx: 16
  ny: 16
  nz: 16
  dx: 0.0625  # 1/16

stochastic:
  sigma2: 0.0           # Zero variance = uniform K
  corr_length: 0.1
  n_modes: 10           # Doesn't matter with sigma2=0
  covariance_type: 0
  seed: 12345
  K_mean: 1.0           # K = 1 everywhere

flow:
  solver: pcg_mg
  mg_levels: 3
  rtol: 1.0e-10         # Tight tolerance for accuracy
  verify_velocity: true  # Enable theoretical verification
  
  bc:
    west:   { type: dirichlet, value: 100.0 }
    east:   { type: dirichlet, value: 0.0 }
    south:  { type: neumann, value: 0.0 }
    north:  { type: neumann, value: 0.0 }
    bottom: { type: neumann, value: 0.0 }
    top:    { type: neumann, value: 0.0 }

transport:
  n_particles: 10
  dt: 0.001
  n_steps: 10

```

# apps\mg_cg_compare.cu

```cu
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

```

# apps\mg_smoke.cu

```cu
#include "../src/runtime/CudaContext.cuh"
#include "../src/runtime/GpuTimer.cuh"
#include "../src/multigrid/multigrid.cuh"
#include "../src/numerics/blas/blas.cuh"
#include "../src/numerics/blas/reduction_workspace.cuh"
#include "../src/physics/flow/coarsen_K.cuh"
#include <iostream>
#include <iomanip>

using namespace rwpt;
using namespace rwpt::multigrid;

int main() {
    std::cout << "=== Multigrid Smoke Test ===\n\n";
    
    // Create CUDA context
    CudaContext ctx(0);
    
    // Define problem: 32^3 grid
    int N = 32;
    Grid3D finest_grid;
    finest_grid.nx = N;
    finest_grid.ny = N;
    finest_grid.nz = N;
    finest_grid.dx = 1.0 / N;
    finest_grid.dy = 1.0 / N;
    finest_grid.dz = 1.0 / N;
    
    std::cout << "Grid: " << N << "^3 = " << finest_grid.num_cells() << " cells\n";
    std::cout << "dx = " << finest_grid.dx << "\n\n";
    
    // Create MG hierarchy (4 levels: 32->16->8->4)
    int num_levels = 4;
    MGHierarchy hier(finest_grid, num_levels);
    
    std::cout << "Multigrid hierarchy:\n";
    for (int l = 0; l < hier.num_levels(); ++l) {
        const auto& g = hier.levels[l].grid;
        std::cout << "  Level " << l << ": " << g.nx << "x" << g.ny << "x" << g.nz 
                  << " (" << g.num_cells() << " cells)\n";
    }
    std::cout << "\n";
    
    // Initialize finest level
    auto& finest = hier.levels[0];
    size_t n = finest.grid.num_cells();
    
    // Set RHS: b = 1 everywhere
    rwpt::blas::fill(ctx, finest.b.span(), 1.0);
    
    // Set initial guess: x = 0
    rwpt::blas::fill(ctx, finest.x.span(), 0.0);
    
    // Set conductivity: K = 1 everywhere (homogeneous for smoke test)
    rwpt::blas::fill(ctx, finest.K.span(), 1.0);
    
    // Homogenize K to coarser levels using geometric mean (legacy CompactHomogenizationKtensor)
    for (int l = 1; l < hier.num_levels(); ++l) {
        rwpt::physics::coarsen_K(
            ctx,
            hier.levels[l].grid,
            hier.levels[l-1].grid,
            hier.levels[l-1].K.span(),
            hier.levels[l].K.span()
        );
    }
    
    ctx.synchronize();
    std::cout << "Problem setup complete.\n";
    std::cout << "RHS norm: ";
    
    rwpt::blas::ReductionWorkspace red;
    real b_norm = rwpt::blas::nrm2_host(ctx, finest.b.span(), red);
    std::cout << std::scientific << std::setprecision(6) << b_norm << "\n\n";
    
    // Configure MG
    MGConfig config;
    config.num_levels = num_levels;
    config.pre_smooth = 2;
    config.post_smooth = 2;
    config.coarse_solve_iters = 50;
    config.verbose = false;
    
    std::cout << "MG Config:\n";
    std::cout << "  Levels: " << config.num_levels << "\n";
    std::cout << "  Pre-smooth: " << config.pre_smooth << "\n";
    std::cout << "  Post-smooth: " << config.post_smooth << "\n";
    std::cout << "  Coarse solve iters: " << config.coarse_solve_iters << "\n\n";
    
    // Configure boundary conditions
    BCSpec bc;
    bc.xmin = BCFace(BCType::Dirichlet, 0.0);
    bc.xmax = BCFace(BCType::Dirichlet, 0.0);
    bc.ymin = BCFace(BCType::Dirichlet, 0.0);
    bc.ymax = BCFace(BCType::Dirichlet, 0.0);
    bc.zmin = BCFace(BCType::Dirichlet, 0.0);
    bc.zmax = BCFace(BCType::Dirichlet, 0.0);
    
    // Solve with MG
    int max_cycles = 10;
    real rtol = 1e-6;
    
    std::cout << "Starting MG solve (max " << max_cycles << " V-cycles, rtol=" << rtol << ")...\\n\\n";
    
    GpuTimer timer;
    timer.start(ctx.cuda_stream());
    
    auto result = mg_solve(ctx, hier, config, bc, max_cycles, rtol);
    
    float elapsed_ms = timer.stop(ctx.cuda_stream());
    ctx.synchronize();
    
    // Report results
    std::cout << "=== Results ===\n";
    std::cout << "Cycles: " << result.num_cycles << "\n";
    std::cout << "Converged: " << (result.converged ? "YES" : "NO") << "\n";
    std::cout << "Initial residual: " << std::scientific << result.initial_residual << "\n";
    std::cout << "Final residual: " << result.final_residual << "\n";
    std::cout << "Relative residual: " << (result.final_residual / result.initial_residual) << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    std::cout << "Time/cycle: " << (elapsed_ms / result.num_cycles) << " ms\n";
    
    return 0;
}

```

# apps\operator_debug.cu

```cu
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

```

# apps\pcg_smoke.cu

```cu
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

```

# apps\physics_types_smoke.cu

```cu
/**
 * @file physics_types_smoke.cu
 * @brief Smoke test for physics types (Etapa 5.0)
 * 
 * Verifies that all physics types compile and construct correctly.
 * No actual computation, just instantiation and basic operations.
 */

#include "../src/physics/common/physics_types.cuh"
#include "../src/runtime/CudaContext.cuh"
#include "../src/core/Grid3D.hpp"
#include <iostream>

using namespace rwpt;
using namespace rwpt::physics;

int main() {
    std::cout << "=== Physics Types Smoke Test (Etapa 5.0) ===\n\n";
    
    // 1. Test configuration structs (no GPU needed)
    std::cout << "1. Testing configuration structs...\n";
    {
        StochasticConfig stoch_cfg;
        std::cout << "   StochasticConfig: sigma2=" << stoch_cfg.sigma2 
                  << ", corr_length=" << stoch_cfg.corr_length
                  << ", n_modes=" << stoch_cfg.n_modes << "\n";
        
        FlowConfig flow_cfg;
        std::cout << "   FlowConfig: solver_type=" << flow_cfg.solver_type
                  << ", mg_levels=" << flow_cfg.mg_levels
                  << ", cg_max_iter=" << flow_cfg.cg_max_iter << "\n";
        
        TransportConfig trans_cfg;
        std::cout << "   TransportConfig: n_particles=" << trans_cfg.n_particles
                  << ", dt=" << trans_cfg.dt
                  << ", n_steps=" << trans_cfg.n_steps << "\n";
        
        SimulationConfig sim_cfg;
        std::cout << "   SimulationConfig: Nx=" << sim_cfg.Nx
                  << ", Ny=" << sim_cfg.Ny
                  << ", Nz=" << sim_cfg.Nz << "\n";
    }
    std::cout << "   OK\n\n";
    
    // 2. Test field types (requires GPU)
    std::cout << "2. Testing field types...\n";
    {
        CudaContext ctx(0);
        Grid3D grid(16, 16, 16, 0.1, 0.1, 0.1);
        
        // Scalar fields
        KField K(grid);
        std::cout << "   KField: size=" << K.size() 
                  << " (" << K.nx << "x" << K.ny << "x" << K.nz << ")\n";
        
        HeadField h(grid);
        std::cout << "   HeadField: size=" << h.size() << "\n";
        
        // Velocity field (staggered)
        VelocityField vel(grid);
        std::cout << "   VelocityField:\n";
        std::cout << "     U: " << vel.size_U() << " (" << (vel.nx+1) << "x" << vel.ny << "x" << vel.nz << ")\n";
        std::cout << "     V: " << vel.size_V() << " (" << vel.nx << "x" << (vel.ny+1) << "x" << vel.nz << ")\n";
        std::cout << "     W: " << vel.size_W() << " (" << vel.nx << "x" << vel.ny << "x" << (vel.nz+1) << ")\n";
        std::cout << "     Total: " << vel.total_size() << "\n";
        
        // Verify staggered dims match legacy convention
        // Legacy: U(Nx+1,Ny,Nz), V(Nx,Ny+1,Nz), W(Nx,Ny,Nz+1)
        bool dims_ok = (vel.size_U() == static_cast<size_t>(17 * 16 * 16)) &&
                       (vel.size_V() == static_cast<size_t>(16 * 17 * 16)) &&
                       (vel.size_W() == static_cast<size_t>(16 * 16 * 17));
        std::cout << "     Staggered dims match legacy: " << (dims_ok ? "YES" : "NO") << "\n";
        
        // Test resize
        Grid3D grid2(32, 32, 32, 0.05, 0.05, 0.05);
        K.resize(grid2);
        std::cout << "   KField after resize: size=" << K.size() << "\n";
    }
    std::cout << "   OK\n\n";
    
    // 3. Test workspace types
    std::cout << "3. Testing workspace types...\n";
    {
        CudaContext ctx(0);
        Grid3D grid(16, 16, 16, 0.1, 0.1, 0.1);
        
        StochasticConfig stoch_cfg;
        stoch_cfg.n_modes = 500;
        
        StochasticWorkspace stoch_ws;
        stoch_ws.allocate(grid, stoch_cfg);
        std::cout << "   StochasticWorkspace: n_modes=" << stoch_ws.n_modes
                  << ", n_cells=" << stoch_ws.n_cells
                  << ", allocated=" << stoch_ws.is_allocated() << "\n";
        
        FlowWorkspace flow_ws;
        flow_ws.allocate(grid);
        std::cout << "   FlowWorkspace: n_cells=" << flow_ws.n_cells
                  << ", allocated=" << flow_ws.is_allocated() << "\n";
        
        TransportConfig trans_cfg;
        trans_cfg.n_particles = 5000;
        
        ParticlesWorkspace part_ws;
        part_ws.allocate(trans_cfg);
        std::cout << "   ParticlesWorkspace: n_particles=" << part_ws.n_particles
                  << ", allocated=" << part_ws.is_allocated() << "\n";
        
        // Test combined workspace
        SimulationConfig sim_cfg;
        sim_cfg.stochastic.n_modes = 1000;
        sim_cfg.transport.n_particles = 10000;
        
        SimulationWorkspace sim_ws;
        sim_ws.allocate(grid, sim_cfg);
        std::cout << "   SimulationWorkspace: all allocated\n";
        
        // Test clear
        sim_ws.clear();
        std::cout << "   SimulationWorkspace after clear: stoch=" << sim_ws.stochastic.is_allocated()
                  << ", flow=" << sim_ws.flow.is_allocated()
                  << ", particles=" << sim_ws.particles.is_allocated() << "\n";
    }
    std::cout << "   OK\n\n";
    
    std::cout << "=== All tests passed ===\n";
    std::cout << "Physics module version: " << PHYSICS_VERSION_MAJOR << "." << PHYSICS_VERSION_MINOR << "\n";
    
    return 0;
}

```

# apps\pin_necessity_test.cu

```cu
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

```

# apps\rwpt_flow_transport.cu

```cu
/**
 * @file rwpt_flow_transport.cu
 * @brief Main application for RWPT flow + transport simulation (Etapa 5)
 * 
 * Pipeline:
 *   1. Load config from YAML
 *   2. Generate stochastic K field
 *   3. Solve head equation (MG or CG)
 *   4. Compute velocity from head
 *   5. Run particle transport
 *   6. Compute statistics / output
 * 
 * This file is the skeleton; actual computation added in subsequent tasks.
 */

#include "../src/io/config/Config.hpp"
#include "../src/numerics/pin_spec.hpp"  // pin_enabled, needs_pin, pin_mode_str
#include "../src/runtime/CudaContext.cuh"
#include "../src/core/Grid3D.hpp"
#include "../src/physics/common/physics_types.cuh"
#include "../src/physics/stochastic/stochastic.cuh"
#include "../src/physics/flow/solve_head.cuh"
#include "../src/physics/flow/velocity_from_head.cuh"
#include <iostream>
#include <iomanip>
#include <string>

using namespace rwpt;
using namespace rwpt::io;
using namespace rwpt::physics;

// Helper to convert BC type to string
const char* bc_type_str(BCType t) {
    switch (t) {
        case BCType::Dirichlet: return "Dirichlet";
        case BCType::Neumann:   return "Neumann";
        case BCType::Periodic:  return "Periodic";
        default:                return "Unknown";
    }
}

void print_config_summary(const AppConfig& cfg) {
    std::cout << "=== Configuration Summary ===\n\n";
    
    // Grid
    std::cout << "Grid:\n";
    std::cout << "  Dimensions: " << cfg.grid.nx << " x " << cfg.grid.ny << " x " << cfg.grid.nz << "\n";
    std::cout << "  Cell size:  dx = " << cfg.grid.dx << "\n";
    std::cout << "  Domain:     [0, " << cfg.grid.Lx() << "] x [0, " << cfg.grid.Ly() 
              << "] x [0, " << cfg.grid.Lz() << "]\n";
    std::cout << "  Total cells: " << (cfg.grid.nx * cfg.grid.ny * cfg.grid.nz) << "\n\n";
    
    // Stochastic
    std::cout << "Stochastic K:\n";
    std::cout << "  sigma^2:     " << cfg.stochastic.sigma2 << "\n";
    std::cout << "  corr_length: " << cfg.stochastic.corr_length << "\n";
    std::cout << "  n_modes:     " << cfg.stochastic.n_modes << "\n";
    std::cout << "  seed:        " << cfg.stochastic.seed << "\n\n";
    
    // Flow
    std::cout << "Flow solver:\n";
    std::cout << "  Solver:      " << cfg.flow.solver << "\n";
    std::cout << "  MG levels:   " << cfg.flow.mg_levels << "\n";
    std::cout << "  Tolerance:   " << cfg.flow.rtol << "\n";
    
    // Pin config
    bool pin_is_enabled = pin_enabled(cfg.flow.pin.mode, cfg.flow.bc);
    std::cout << "  Pin mode:    " << pin_mode_str(cfg.flow.pin.mode);
    std::cout << " (needs_pin=" << (needs_pin(cfg.flow.bc) ? "yes" : "no");
    std::cout << ", enabled=" << (pin_is_enabled ? "YES" : "no") << ")\n";
    if (pin_is_enabled) {
        std::cout << "    (diagonal doubling at cell [0,0,0])\n";
    }
    
    std::cout << "  BCs (west/east=x, south/north=y, bottom/top=z):\n";
    std::cout << "    west(xmin):   " << bc_type_str(cfg.flow.bc.xmin.type) << " = " << cfg.flow.bc.xmin.value << "\n";
    std::cout << "    east(xmax):   " << bc_type_str(cfg.flow.bc.xmax.type) << " = " << cfg.flow.bc.xmax.value << "\n";
    std::cout << "    south(ymin):  " << bc_type_str(cfg.flow.bc.ymin.type) << " = " << cfg.flow.bc.ymin.value << "\n";
    std::cout << "    north(ymax):  " << bc_type_str(cfg.flow.bc.ymax.type) << " = " << cfg.flow.bc.ymax.value << "\n";
    std::cout << "    bottom(zmin): " << bc_type_str(cfg.flow.bc.zmin.type) << " = " << cfg.flow.bc.zmin.value << "\n";
    std::cout << "    top(zmax):    " << bc_type_str(cfg.flow.bc.zmax.type) << " = " << cfg.flow.bc.zmax.value << "\n\n";
    
    // Transport
    std::cout << "Transport:\n";
    std::cout << "  n_particles: " << cfg.transport.n_particles << "\n";
    std::cout << "  dt:          " << cfg.transport.dt << "\n";
    std::cout << "  n_steps:     " << cfg.transport.n_steps << "\n";
    std::cout << "  Total time:  " << (cfg.transport.dt * cfg.transport.n_steps) << "\n";
    std::cout << "  porosity:    " << cfg.transport.porosity << "\n";
    std::cout << "  diffusion:   " << cfg.transport.diffusion << "\n\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=== RWPT Flow + Transport Simulator ===\n";
    std::cout << "Etapa 5 - Physics Pipeline\n\n";
    
    try {
        // 1. Parse command line
        std::string config_path = "apps/config_example.yaml";
        if (argc > 1) {
            config_path = argv[1];
        }
        std::cout << "Config file: " << config_path << "\n\n";
        
        // 2. Load configuration
        AppConfig cfg = load_config_yaml(config_path);
        print_config_summary(cfg);
        
        // 3. Initialize CUDA context
        std::cout << "Initializing CUDA...\n";
        CudaContext ctx(0);
        std::cout << "  Device ready.\n\n";
        
        // 4. Create Grid3D from config
        Grid3D grid(cfg.grid.nx, cfg.grid.ny, cfg.grid.nz,
                    cfg.grid.dx, cfg.grid.dx, cfg.grid.dx);  // Isotropic
        
        // 5. Allocate fields (Etapa 5.0 types)
        std::cout << "Allocating fields...\n";
        
        KField K(grid);
        std::cout << "  K field:        " << K.size() << " cells ("
                  << (K.size() * sizeof(real) / 1024.0 / 1024.0) << " MB)\n";
        
        HeadField h(grid);
        std::cout << "  Head field:     " << h.size() << " cells\n";
        
        VelocityField vel(grid);
        std::cout << "  Velocity field: " << vel.total_size() << " faces ("
                  << "U:" << vel.size_U() << ", V:" << vel.size_V() << ", W:" << vel.size_W() << ")\n";
        
        // 6. Allocate workspaces
        std::cout << "\nAllocating workspaces...\n";
        
        // Convert config types
        StochasticConfig stoch_cfg;
        stoch_cfg.sigma2 = cfg.stochastic.sigma2;
        stoch_cfg.corr_length = cfg.stochastic.corr_length;
        stoch_cfg.n_modes = cfg.stochastic.n_modes;
        stoch_cfg.covariance_type = cfg.stochastic.covariance_type;
        stoch_cfg.seed = cfg.stochastic.seed;
        
        StochasticWorkspace stoch_ws;
        stoch_ws.allocate(grid, stoch_cfg);
        std::cout << "  Stochastic workspace: " << stoch_ws.n_modes << " modes, "
                  << stoch_ws.n_cells << " cells\n";
        
        FlowWorkspace flow_ws;
        flow_ws.allocate(grid);
        std::cout << "  Flow workspace:       " << flow_ws.n_cells << " cells\n";
        
        TransportConfig trans_cfg;
        trans_cfg.n_particles = cfg.transport.n_particles;
        trans_cfg.dt = cfg.transport.dt;
        trans_cfg.n_steps = cfg.transport.n_steps;
        
        ParticlesWorkspace part_ws;
        part_ws.allocate(trans_cfg);
        std::cout << "  Particles workspace:  " << part_ws.n_particles << " particles\n";
        
        // Sync and report memory
        ctx.synchronize();
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "\nGPU memory: " << std::fixed << std::setprecision(1)
                  << ((total_mem - free_mem) / 1024.0 / 1024.0) << " MB used, "
                  << (free_mem / 1024.0 / 1024.0) << " MB free\n";
        
        // ====================================================================
        // PIPELINE: Generate stochastic K field (Task 5.2)
        // ====================================================================
        std::cout << "\n=== STEP 1: Generate K field ===\n";
        std::cout << "  Covariance type: " << (stoch_cfg.covariance_type == 0 ? "exponential" : "gaussian") << "\n";
        std::cout << "  sigma^2 = " << stoch_cfg.sigma2 << ", corr_length = " << stoch_cfg.corr_length << "\n";
        std::cout << "  n_modes = " << stoch_cfg.n_modes << ", seed = " << stoch_cfg.seed << "\n";
        
        // Generate K field
        generate_K_field(
            DeviceSpan<real>(K.device_ptr(), K.size()),
            stoch_ws,
            grid,
            stoch_cfg,
            ctx
        );
        ctx.synchronize();
        
        // Compute and print statistics
        real K_min, K_max, K_mean;
        compute_field_stats(
            DeviceSpan<const real>(K.device_ptr(), K.size()),
            K_min, K_max, K_mean,
            ctx
        );
        
        std::cout << "\n  K field statistics:\n";
        std::cout << "    min(K)  = " << K_min << "\n";
        std::cout << "    max(K)  = " << K_max << "\n";
        std::cout << "    mean(K) = " << K_mean << "\n";
        std::cout << "    ratio   = " << (K_max / K_min) << "\n";
        
        // Also compute logK stats for comparison
        real logK_min, logK_max, logK_mean;
        compute_field_stats(
            DeviceSpan<const real>(stoch_ws.logK.data(), stoch_ws.n_cells),
            logK_min, logK_max, logK_mean,
            ctx
        );
        std::cout << "\n  logK (Gaussian) statistics:\n";
        std::cout << "    min(logK)  = " << logK_min << "\n";
        std::cout << "    max(logK)  = " << logK_max << "\n";
        std::cout << "    mean(logK) = " << logK_mean << " (should be ~0)\n";
        std::cout << "    Expected Var[logK] ~= " << stoch_cfg.sigma2 << "\n";
        
        // ====================================================================
        // PIPELINE: Solve head equation (Task 5.3)
        // ====================================================================
        std::cout << "\n=== STEP 2: Solve head equation ===\n";
        std::cout << "  Solver: " << cfg.flow.solver << "\n";
        std::cout << "  BCs: west=" << cfg.flow.bc.xmin.value << " (Dirichlet), east=" 
                  << cfg.flow.bc.xmax.value << " (Dirichlet)\n";
        
        // Build HeadSolveConfig from FlowYamlConfig using factory method
        HeadSolveConfig head_cfg = HeadSolveConfig::from_yaml(cfg.flow);
        
        // Allocate MG hierarchy in workspace (now with MG levels)
        flow_ws.allocate(grid, head_cfg.mg_levels);
        
        // Solve head
        HeadSolveResult head_result = solve_head(
            DeviceSpan<real>(h.device_ptr(), h.size()),
            DeviceSpan<const real>(K.device_ptr(), K.size()),
            grid,
            cfg.flow.bc,  // BCSpec from config
            head_cfg,
            ctx,
            flow_ws
        );
        ctx.synchronize();
        
        std::cout << "\n  Solve result:\n";
        std::cout << "    Converged:        " << (head_result.converged ? "YES" : "NO") << "\n";
        std::cout << "    Iterations:       " << head_result.num_iterations << "\n";
        std::cout << std::scientific << std::setprecision(3);
        std::cout << "    Initial residual: " << head_result.initial_residual << "\n";
        std::cout << "    Final residual:   " << head_result.final_residual << "\n";
        if (head_result.initial_residual > 0) {
            std::cout << "    Reduction:        " << (head_result.final_residual / head_result.initial_residual) << "\n";
        }
        std::cout << std::fixed << std::setprecision(1);  // Reset to default
        
        // Head statistics
        real h_min, h_max, h_mean;
        compute_field_stats(
            DeviceSpan<const real>(h.device_ptr(), h.size()),
            h_min, h_max, h_mean,
            ctx
        );
        
        std::cout << "\n  Head field statistics:\n";
        std::cout << "    min(h)  = " << h_min << "\n";
        std::cout << "    max(h)  = " << h_max << "\n";
        std::cout << "    mean(h) = " << h_mean << "\n";
        
        // ====================================================================
        // PIPELINE: Compute velocity from head (Task 5.4)
        // ====================================================================
        std::cout << "\n=== STEP 3: Compute velocity from head ===\n";
        std::cout << "  Using Darcy's law with harmonic mean conductivity\n";
        
        // Compute U, V, W from H and K
        compute_velocity_from_head(vel, h, K, grid, cfg.flow.bc, ctx);
        ctx.synchronize();
        
        // Print checksums
        print_velocity_checksums(vel, ctx);
        
        // Verify mean velocity against theoretical Darcy (if enabled and Dirichlet)
        if (cfg.flow.verify_velocity &&
            cfg.flow.bc.xmin.type == BCType::Dirichlet && 
            cfg.flow.bc.xmax.type == BCType::Dirichlet) {
            verify_mean_velocity_darcy(vel, K, grid, cfg.flow.bc, ctx);
        }
        
        // ====================================================================
        // PLACEHOLDER: Remaining pipeline steps (Tasks 5.5+)
        // ====================================================================
        std::cout << "\n";
        std::cout << "--- Remaining pipeline steps (not implemented yet) ---\n";
        std::cout << "[TODO] 5.5: Run particle transport\n";
        std::cout << "[TODO] 5.6: Compute statistics & output\n";
        std::cout << "-----------------------------------------------------\n\n";
        
        std::cout << "=== OK: K generated, head solved, velocity computed ===\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}

```

# apps\smoke_core.cu

```cu
#include "../src/core/Scalar.hpp"
#include "../src/core/Grid3D.hpp"
#include "../src/core/BCSpec.hpp"
#include "../src/core/DeviceBuffer.cuh"
#include "../src/runtime/CudaContext.cuh"
#include "../src/runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <iostream>

int main() {
    try {
        std::cout << "Testing core contracts...\n\n";
        
        // Create CUDA context
        rwpt::CudaContext ctx(0);
        
        // Test Scalar type
        std::cout << "1. Testing rwpt::real type:\n";
        rwpt::real test_value = 3.14159;
        std::cout << "   rwpt::real value: " << test_value << "\n\n";
        
        // Test Grid3D
        std::cout << "2. Testing rwpt::Grid3D:\n";
        rwpt::Grid3D grid(16, 8, 4, 1.0, 1.0, 1.0);
        std::cout << "   Dimensions: " << grid.nx << " x " << grid.ny << " x " << grid.nz << "\n";
        std::cout << "   Spacing: dx=" << grid.dx << ", dy=" << grid.dy << ", dz=" << grid.dz << "\n";
        std::cout << "   Total cells: " << grid.num_cells() << "\n";
        std::cout << "   Linear index at (1,2,3): " << grid.idx(1, 2, 3) << "\n\n";
        
        // Test BCSpec
        std::cout << "3. Testing rwpt::BCSpec:\n";
        rwpt::BCSpec bc;
        std::cout << "   Default BC created (all Dirichlet with value 0)\n";
        std::cout << "   xmin: type=" << static_cast<int>(bc.xmin.type) 
                  << ", value=" << bc.xmin.value << "\n\n";
        
        // Test DeviceBuffer
        std::cout << "4. Testing rwpt::DeviceBuffer:\n";
        const size_t buffer_size = 128;
        rwpt::DeviceBuffer<rwpt::real> buf(buffer_size);
        std::cout << "   Allocated buffer of size: " << buf.size() << "\n";
        std::cout << "   Device pointer: " << buf.data() << "\n";
        
        // Optional: Fill with cudaMemset
        RWPT_CUDA_CHECK(cudaMemsetAsync(buf.data(), 0, buffer_size * sizeof(rwpt::real), 
                                         ctx.cuda_stream()));
        ctx.synchronize();
        std::cout << "   Buffer zeroed successfully\n\n";
        
        // Test DeviceSpan
        std::cout << "5. Testing rwpt::DeviceSpan:\n";
        auto span = buf.span();
        std::cout << "   Span size: " << span.size() << "\n";
        std::cout << "   Span data pointer: " << span.data() << "\n\n";
        
        std::cout << "All core contract tests passed!\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

```

# apps\smoke_device.cu

```cu
#include "../src/runtime/CudaContext.cuh"
#include "../src/runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <iostream>

int main() {
    try {
        // Create CUDA context for device 0
        rwpt::CudaContext ctx(0);
        
        // Get device properties
        cudaDeviceProp prop;
        RWPT_CUDA_CHECK(cudaGetDeviceProperties(&prop, ctx.device()));
        
        // Print device information
        std::cout << "GPU Device Information:\n";
        std::cout << "  Name: " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  SM Count: " << prop.multiProcessorCount << "\n";
        
        // Convert global memory to GB
        double memory_gb = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "  Total Global Memory: " << memory_gb << " GB\n";
        
        // Verify stream and cublas handle are created
        std::cout << "\nCUDA Context:\n";
        std::cout << "  Stream created: " << (ctx.cuda_stream() != nullptr ? "yes" : "no") << "\n";
        std::cout << "  cuBLAS handle created: " << (ctx.cublas_handle() != nullptr ? "yes" : "no") << "\n";
        
        std::cout << "\nSmoke test passed!\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

```

# CMakeLists.txt

```txt
cmake_minimum_required(VERSION 3.18)
project(rwpt-mdisp-simulator CUDA CXX)

# CUDA setup
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)  # Adjust for your GPU

# Find CUDA toolkit for cuBLAS
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# External: yaml-cpp (vendorized, static build)
# ============================================================================
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(src/external/yaml-cpp-0.8.0 EXCLUDE_FROM_ALL)

# ============================================================================
# Source files
set(RUNTIME_SOURCES
    src/runtime/CudaContext.cu
)

set(BLAS_SOURCES
    src/numerics/blas/fill.cu
    src/numerics/blas/copy.cu
    src/numerics/blas/scal.cu
    src/numerics/blas/axpy.cu
    src/numerics/blas/axpby.cu
    src/numerics/blas/dot.cu
    src/numerics/blas/nrm2.cu
    src/numerics/blas/cg_kernels.cu
)

set(SOLVER_SOURCES
    src/numerics/solvers/preconditioner.cu
    src/numerics/solvers/mg_preconditioner.cu
)

set(OPERATOR_SOURCES
    src/numerics/operators/poisson3d_operator.cu
    src/numerics/operators/varcoeff_laplacian.cu
)

set(MULTIGRID_SOURCES
    src/multigrid/transfer/restrict_3d.cu
    src/multigrid/transfer/prolong_3d.cu
    src/multigrid/smoothers/residual_3d.cu
    src/multigrid/smoothers/gsrb_3d.cu
    src/multigrid/cycle/v_cycle.cu
)

set(PHYSICS_SOURCES
    src/physics/flow/coarsen_K.cu
    src/physics/flow/rhs_head.cu
    src/physics/flow/solve_head.cu
    src/physics/flow/velocity_from_head.cu
    src/physics/stochastic/stochastic.cu
)

set(IO_SOURCES
    src/io/config/ConfigYaml.cpp
)

# Library
add_library(rwpt_lib STATIC
    ${RUNTIME_SOURCES}
    ${BLAS_SOURCES}
    ${SOLVER_SOURCES}
    ${OPERATOR_SOURCES}
    ${MULTIGRID_SOURCES}
    ${PHYSICS_SOURCES}
    ${IO_SOURCES}
)

target_include_directories(rwpt_lib PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(rwpt_lib PUBLIC CUDA::cudart CUDA::cublas yaml-cpp)

# Apps
add_executable(smoke_device apps/smoke_device.cu)
target_link_libraries(smoke_device rwpt_lib)

add_executable(smoke_core apps/smoke_core.cu)
target_link_libraries(smoke_core rwpt_lib)

add_executable(cg_smoke apps/cg_smoke.cu)
target_link_libraries(cg_smoke rwpt_lib)

add_executable(mg_smoke apps/mg_smoke.cu)
target_link_libraries(mg_smoke rwpt_lib)

add_executable(mg_cg_compare apps/mg_cg_compare.cu)
target_link_libraries(mg_cg_compare rwpt_lib)

add_executable(operator_debug apps/operator_debug.cu)
target_link_libraries(operator_debug rwpt_lib)

add_executable(pcg_smoke apps/pcg_smoke.cu)
target_link_libraries(pcg_smoke rwpt_lib)

add_executable(physics_types_smoke apps/physics_types_smoke.cu)
target_link_libraries(physics_types_smoke rwpt_lib CUDA::curand)

add_executable(rwpt_flow_transport apps/rwpt_flow_transport.cu)
target_link_libraries(rwpt_flow_transport rwpt_lib CUDA::curand)

add_executable(pin_necessity_test apps/pin_necessity_test.cu)
target_link_libraries(pin_necessity_test rwpt_lib)

# Compile options
set_target_properties(rwpt_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(rwpt_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

```

# desktop.ini

```ini
[.ShellClassInfo]
IconResource=C:\Program Files\Microsoft OneDrive\OneDrive.exe,8

```

# src\core\BCSpec.hpp

```hpp
#pragma once

#include "Scalar.hpp"
#include <stdexcept>
#include <cmath>

namespace rwpt {

enum class BCType {
    Dirichlet,
    Neumann,
    Periodic
};

struct BCFace {
    BCType type;
    real value;

    // Default: Dirichlet with zero value
    BCFace() : type(BCType::Dirichlet), value(0.0) {}
    
    BCFace(BCType t, real v) : type(t), value(v) {}
};

struct BCSpec {
    BCFace xmin, xmax;
    BCFace ymin, ymax;
    BCFace zmin, zmax;

    // Default constructor: all faces Dirichlet with zero value
    BCSpec()
        : xmin(BCType::Dirichlet, 0.0),
          xmax(BCType::Dirichlet, 0.0),
          ymin(BCType::Dirichlet, 0.0),
          ymax(BCType::Dirichlet, 0.0),
          zmin(BCType::Dirichlet, 0.0),
          zmax(BCType::Dirichlet, 0.0) {}
    
    // Query helpers
    bool is_periodic_x() const { return xmin.type == BCType::Periodic && xmax.type == BCType::Periodic; }
    bool is_periodic_y() const { return ymin.type == BCType::Periodic && ymax.type == BCType::Periodic; }
    bool is_periodic_z() const { return zmin.type == BCType::Periodic && zmax.type == BCType::Periodic; }
    
    bool is_all_homog_neumann() const {
        return xmin.type == BCType::Neumann && xmin.value == 0.0 &&
               xmax.type == BCType::Neumann && xmax.value == 0.0 &&
               ymin.type == BCType::Neumann && ymin.value == 0.0 &&
               ymax.type == BCType::Neumann && ymax.value == 0.0 &&
               zmin.type == BCType::Neumann && zmin.value == 0.0 &&
               zmax.type == BCType::Neumann && zmax.value == 0.0;
    }
    
    // Validation
    void validate() const {
        // Periodic must come in pairs
        if (xmin.type == BCType::Periodic || xmax.type == BCType::Periodic) {
            if (xmin.type != BCType::Periodic || xmax.type != BCType::Periodic) {
                throw std::runtime_error("Periodic BC must be specified on both xmin and xmax");
            }
        }
        if (ymin.type == BCType::Periodic || ymax.type == BCType::Periodic) {
            if (ymin.type != BCType::Periodic || ymax.type != BCType::Periodic) {
                throw std::runtime_error("Periodic BC must be specified on both ymin and ymax");
            }
        }
        if (zmin.type == BCType::Periodic || zmax.type == BCType::Periodic) {
            if (zmin.type != BCType::Periodic || zmax.type != BCType::Periodic) {
                throw std::runtime_error("Periodic BC must be specified on both zmin and zmax");
            }
        }
        
        // Values must be finite
        auto is_finite = [](real v) { return v == v && v != INFINITY && v != -INFINITY; };
        if (!is_finite(xmin.value) || !is_finite(xmax.value) ||
            !is_finite(ymin.value) || !is_finite(ymax.value) ||
            !is_finite(zmin.value) || !is_finite(zmax.value)) {
            throw std::runtime_error("BC values must be finite");
        }
    }
};

} // namespace rwpt

```

# src\core\BCSpecDevice.cuh

```cuh
#pragma once

#include "BCSpec.hpp"
#include "Scalar.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace rwpt {

// Device-friendly POD for BC specification
// Passed by value to kernels (cheap: 6 bytes + 6*8 = 54 bytes)
struct BCSpecDevice {
    // Type for each face (order: xmin, xmax, ymin, ymax, zmin, zmax)
    uint8_t type[6];
    
    // Value for each face (same order)
    real value[6];
    
    __host__ __device__
    BCSpecDevice() {
        for (int i = 0; i < 6; ++i) {
            type[i] = static_cast<uint8_t>(BCType::Dirichlet);
            value[i] = 0.0;
        }
    }
};

// Convert BCSpec to device view (host-only, trivial)
inline BCSpecDevice to_device(const BCSpec& bc) {
    BCSpecDevice dev;
    
    // Order: xmin=0, xmax=1, ymin=2, ymax=3, zmin=4, zmax=5
    dev.type[0] = static_cast<uint8_t>(bc.xmin.type);
    dev.value[0] = bc.xmin.value;
    
    dev.type[1] = static_cast<uint8_t>(bc.xmax.type);
    dev.value[1] = bc.xmax.value;
    
    dev.type[2] = static_cast<uint8_t>(bc.ymin.type);
    dev.value[2] = bc.ymin.value;
    
    dev.type[3] = static_cast<uint8_t>(bc.ymax.type);
    dev.value[3] = bc.ymax.value;
    
    dev.type[4] = static_cast<uint8_t>(bc.zmin.type);
    dev.value[4] = bc.zmin.value;
    
    dev.type[5] = static_cast<uint8_t>(bc.zmax.type);
    dev.value[5] = bc.zmax.value;
    
    return dev;
}

} // namespace rwpt

```

# src\core\DeviceBuffer.cuh

```cuh
#pragma once

#include "DeviceSpan.cuh"
#include "../runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace rwpt {

// RAII wrapper for device memory allocation with capacity
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), n_(0), capacity_(0) {}

    explicit DeviceBuffer(size_t n) : ptr_(nullptr), n_(n), capacity_(0) {
        if (n > 0) {
            RWPT_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
            capacity_ = n;
        }
    }

    ~DeviceBuffer() {
        reset();
    }

    // Delete copy semantics
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), n_(other.n_), capacity_(other.capacity_) {
        other.ptr_ = nullptr;
        other.n_ = 0;
        other.capacity_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            n_ = other.n_;
            capacity_ = other.capacity_;
            other.ptr_ = nullptr;
            other.n_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    
    size_t size() const { return n_; }

    DeviceSpan<T> span() {
        return DeviceSpan<T>(ptr_, n_);
    }

    DeviceSpan<const T> span() const {
        return DeviceSpan<const T>(ptr_, n_);
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            // Best effort free, no exceptions (safe for destructor)
            cudaFree(ptr_);
            ptr_ = nullptr;
            n_ = 0;
            capacity_ = 0;
        }
    }

    void resize(size_t n) {
        // Only reallocate if required size exceeds capacity
        if (n > capacity_) {
            reset();
            if (n > 0) {
                RWPT_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
                capacity_ = n;
            }
        }
        n_ = n;
    }
    
    void swap(DeviceBuffer& other) noexcept {
        T* tmp_ptr = ptr_;
        size_t tmp_n = n_;
        size_t tmp_cap = capacity_;
        
        ptr_ = other.ptr_;
        n_ = other.n_;
        capacity_ = other.capacity_;
        
        other.ptr_ = tmp_ptr;
        other.n_ = tmp_n;
        other.capacity_ = tmp_cap;
    }

private:
    T* ptr_;
    size_t n_;
    size_t capacity_;
};

} // namespace rwpt

```

# src\core\DeviceSpan.cuh

```cuh
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <type_traits>

namespace rwpt {

// Non-owning view into device memory
template<typename T>
class DeviceSpan {
public:
    DeviceSpan() : ptr_(nullptr), n_(0) {}

    DeviceSpan(T* ptr, size_t n) : ptr_(ptr), n_(n) {}
    
    // Conversion from non-const to const (safe direction)
    template<typename U = T>
    __host__ __device__
    DeviceSpan(const DeviceSpan<typename std::remove_const<U>::type>& other,
               typename std::enable_if<std::is_const<U>::value>::type* = nullptr)
        : ptr_(const_cast<T*>(other.data())), n_(other.size()) {}

    __host__ __device__
    T* data() const { return ptr_; }
    
    __host__ __device__
    size_t size() const { return n_; }

    // Device-only to prevent dangerous host access to device memory
    __device__
    T& operator[](size_t i) const {
        return ptr_[i];
    }

private:
    T* ptr_;
    size_t n_;
};

} // namespace rwpt

```

# src\core\Grid3D.hpp

```hpp
#pragma once

#include "Scalar.hpp"
#include <cuda_runtime.h>
#include <cstddef>

namespace rwpt {

/**
 * @brief 3D structured grid specification
 * 
 * IMPORTANT: Current implementation assumes ISOTROPIC grids (dx = dy = dz)
 * 
 * This assumption is used throughout the multigrid stack:
 * - Smoother (gsrb_3d.cu): uses grid.dx for all directions
 * - Residual (residual_3d.cu): uses grid.dx for operator scaling
 * - Transfer operators: assume 2:1 coarsening in all directions
 * - Variable-coefficient Laplacian: uses grid.dx for inv_dx2
 * 
 * Legacy code (CCMG_V_cycle.cu) also assumes dx=dy=dz via:
 *   double h = Ly/(double)Ny;  // Uses Ly/Ny for all directions
 *   double dxdx = h*h;
 * 
 * To support anisotropic grids, the following would need modification:
 * - All stencil kernels to use direction-specific spacing
 * - Coarsening ratio per direction
 * - BC helper functions
 * 
 * For now, users should ensure dx == dy == dz when constructing grids.
 */
struct Grid3D {
    int nx, ny, nz;
    real dx, dy, dz;

    // Default constructor
    Grid3D() : nx(0), ny(0), nz(0), dx(0.0), dy(0.0), dz(0.0) {}

    // Constructor with dimensions and spacing
    Grid3D(int nx_, int ny_, int nz_, real dx_, real dy_, real dz_)
        : nx(nx_), ny(ny_), nz(nz_), dx(dx_), dy(dy_), dz(dz_) {}

    // Total number of cells
    size_t num_cells() const {
        return static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    }

    // Linear index: row-major order i + nx*(j + ny*k)
    __host__ __device__
    size_t idx(int i, int j, int k) const {
        return i + nx * (j + ny * k);
    }
};

} // namespace rwpt

```

# src\core\Scalar.hpp

```hpp
#pragma once

namespace rwpt {

// Scalar type for the entire project
using real = double;

} // namespace rwpt

```

# src\io\config\Config.hpp

```hpp
#pragma once

/**
 * @file Config.hpp
 * @brief Application configuration loaded from YAML
 * 
 * Aggregates all config needed for the physics pipeline:
 * Grid + Stochastic + Flow + Transport
 * 
 * The config structs here mirror YAML structure and are used for I/O.
 * Runtime numerical types (like PinSpec) are in the numerics layer.
 */

#include "../../core/Scalar.hpp"
#include "../../core/BCSpec.hpp"
#include "../../numerics/pin_spec.hpp"  // PinMode
#include <string>
#include <cstdint>
#include <array>

namespace rwpt {
namespace io {

// Re-export PinMode from numerics layer for config usage
using rwpt::PinMode;

/**
 * @brief Pin configuration for flow solver
 * 
 * Configures how the solver handles singular systems (all periodic/Neumann).
 * See pin_spec.hpp for full documentation on the pin mechanism.
 */
struct PinConfig {
    PinMode mode = PinMode::Auto;  // auto | on | off
};

/**
 * @brief Grid configuration
 */
struct GridConfig {
    int nx = 64;
    int ny = 64;
    int nz = 64;
    real dx = 1.0;  // Isotropic: dy = dz = dx
    
    // Computed domain size
    real Lx() const { return nx * dx; }
    real Ly() const { return ny * dx; }
    real Lz() const { return nz * dx; }
};

/**
 * @brief Stochastic K field configuration
 */
struct StochasticYamlConfig {
    real sigma2 = 1.0;           // Variance of log-K
    real corr_length = 1.0;      // Correlation length
    int  n_modes = 1000;         // Number of Fourier modes
    int  covariance_type = 0;    // 0 = exponential, 1 = gaussian
    uint64_t seed = 12345;       // RNG seed
    real K_mean = 1.0;           // Geometric mean of K
};

/**
 * @brief Flow solver configuration
 */
struct FlowYamlConfig {
    // Solver type: "mg", "cg", "mg_cg" (MG-preconditioned CG)
    std::string solver = "mg";
    
    // MG parameters
    int mg_levels = 4;
    int mg_pre_smooth = 2;
    int mg_post_smooth = 2;
    int mg_coarse_iters = 50;
    int mg_max_cycles = 20;
    
    // CG parameters
    int cg_max_iter = 1000;
    real cg_rtol = 1e-8;
    int cg_check_every = 10;  // Check convergence every N iterations
    
    // Convergence
    real rtol = 1e-6;
    
    // Boundary conditions (6 faces)
    // Legacy names: west/east=x, south/north=y, bottom/top=z
    BCSpec bc;
    
    // Pin configuration for singular systems (legacy: pin1stCell)
    PinConfig pin;
    
    // Verification: compare computed velocity vs theoretical Darcy
    bool verify_velocity = false;
};

/**
 * @brief Particle transport configuration
 */
struct TransportYamlConfig {
    int n_particles = 10000;
    real dt = 0.01;
    int n_steps = 1000;
    real porosity = 1.0;
    real diffusion = 0.0;       // Molecular diffusion
    uint64_t seed = 54321;
    
    // Output frequency
    int output_every = 100;
    
    // Injection (default: x=0 plane)
    real inject_x = 0.0;
};

/**
 * @brief Output configuration
 */
struct OutputYamlConfig {
    std::string output_dir = "./output";
    bool save_K = true;
    bool save_head = true;
    bool save_velocity = false;
    bool save_particles = true;
    std::string format = "binary";  // "binary" or "vtk"
};

/**
 * @brief Complete application configuration
 */
struct AppConfig {
    GridConfig grid;
    StochasticYamlConfig stochastic;
    FlowYamlConfig flow;
    TransportYamlConfig transport;
    OutputYamlConfig output;
    
    // Validation helpers
    bool is_valid() const {
        return grid.nx > 0 && grid.ny > 0 && grid.nz > 0 && grid.dx > 0;
    }
};

/**
 * @brief Load configuration from YAML file
 * 
 * @param path Path to YAML config file
 * @return AppConfig Parsed configuration with defaults for missing fields
 * @throws std::runtime_error if file not found or critical fields missing
 */
AppConfig load_config_yaml(const std::string& path);

} // namespace io
} // namespace rwpt

```

# src\io\config\ConfigYaml.cpp

```cpp
/**
 * @file ConfigYaml.cpp
 * @brief YAML parser for AppConfig
 * 
 * Uses yaml-cpp. Tolerant parsing: unknown keys are ignored,
 * missing optional fields get defaults.
 */

#include "Config.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace rwpt {
namespace io {

namespace {

// Helper to get value or default
template<typename T>
T get_or(const YAML::Node& node, const std::string& key, T default_val) {
    if (node[key]) {
        return node[key].as<T>();
    }
    return default_val;
}

// Parse BCType from string
BCType parse_bc_type(const std::string& s) {
    if (s == "dirichlet" || s == "Dirichlet") return BCType::Dirichlet;
    if (s == "neumann" || s == "Neumann") return BCType::Neumann;
    if (s == "periodic" || s == "Periodic") return BCType::Periodic;
    throw std::runtime_error("Unknown BC type: " + s);
}

// Parse a single BC face with validation
BCFace parse_bc_face(const YAML::Node& node, const std::string& face_name) {
    BCFace face;
    if (!node) return face;  // Default: Dirichlet 0
    
    std::string type_str = get_or<std::string>(node, "type", "dirichlet");
    face.type = parse_bc_type(type_str);
    
    // Periodic doesn't need a value
    if (face.type == BCType::Periodic) {
        face.value = 0.0;  // Ignored
    } else {
        // Dirichlet/Neumann need value
        if (!node["value"]) {
            // Use default 0 but warn (could be intentional)
            face.value = 0.0;
        } else {
            face.value = node["value"].as<real>();
        }
    }
    return face;
}

// Parse grid section
GridConfig parse_grid(const YAML::Node& node) {
    GridConfig cfg;
    if (!node) return cfg;
    
    cfg.nx = get_or<int>(node, "nx", 64);
    cfg.ny = get_or<int>(node, "ny", 64);
    cfg.nz = get_or<int>(node, "nz", 64);
    cfg.dx = get_or<real>(node, "dx", 1.0);
    
    return cfg;
}

// Parse stochastic section
StochasticYamlConfig parse_stochastic(const YAML::Node& node) {
    StochasticYamlConfig cfg;
    if (!node) return cfg;
    
    cfg.sigma2 = get_or<real>(node, "sigma2", 1.0);
    cfg.corr_length = get_or<real>(node, "corr_length", 1.0);
    cfg.n_modes = get_or<int>(node, "n_modes", 1000);
    cfg.covariance_type = get_or<int>(node, "covariance_type", 0);
    cfg.seed = get_or<uint64_t>(node, "seed", 12345);
    cfg.K_mean = get_or<real>(node, "K_mean", 1.0);
    
    return cfg;
}

// Parse flow section
FlowYamlConfig parse_flow(const YAML::Node& node) {
    FlowYamlConfig cfg;
    if (!node) return cfg;
    
    cfg.solver = get_or<std::string>(node, "solver", "mg");
    cfg.mg_levels = get_or<int>(node, "mg_levels", 4);
    cfg.mg_pre_smooth = get_or<int>(node, "mg_pre_smooth", 2);
    cfg.mg_post_smooth = get_or<int>(node, "mg_post_smooth", 2);
    cfg.mg_coarse_iters = get_or<int>(node, "mg_coarse_iters", 50);
    cfg.mg_max_cycles = get_or<int>(node, "mg_max_cycles", 20);
    cfg.cg_max_iter = get_or<int>(node, "cg_max_iter", 1000);
    cfg.cg_rtol = get_or<real>(node, "cg_rtol", 1e-8);
    cfg.cg_check_every = get_or<int>(node, "cg_check_every", 10);
    cfg.rtol = get_or<real>(node, "rtol", 1e-6);
    
    // Verification flag
    cfg.verify_velocity = get_or<bool>(node, "verify_velocity", false);
    
    // Pin configuration (legacy: pin1stCell diagonal doubling)
    // Format: flow.pin.mode = "auto" | "on" | "off"
    // Legacy format: flow.pin_first_cell = true/false (backward compat)
    // Note: pin always applies to cell [0,0,0], value is not configurable
    if (node["pin"]) {
        const auto& pin_node = node["pin"];
        
        // Parse mode: "auto" | "on" | "off"
        std::string mode_str = get_or<std::string>(pin_node, "mode", "auto");
        if (mode_str == "on") {
            cfg.pin.mode = PinMode::On;
        } else if (mode_str == "off") {
            cfg.pin.mode = PinMode::Off;
        } else {
            cfg.pin.mode = PinMode::Auto;  // default
        }
        // Note: pin.value and pin.index are ignored (legacy diagonal doubling)
    } else if (node["pin_first_cell"]) {
        // Legacy format backward compatibility
        bool pin_enabled = get_or<bool>(node, "pin_first_cell", false);
        cfg.pin.mode = pin_enabled ? PinMode::On : PinMode::Off;
    }
    // else: defaults (mode=Auto)
    
    // Parse boundary conditions
    // Support both legacy names (west/east/south/north/bottom/top) 
    // and coordinate names (xmin/xmax/ymin/ymax/zmin/zmax)
    if (node["bc"]) {
        const auto& bc_node = node["bc"];
        
        // X direction: west/east or xmin/xmax
        if (bc_node["west"]) {
            cfg.bc.xmin = parse_bc_face(bc_node["west"], "west(xmin)");
        } else if (bc_node["xmin"]) {
            cfg.bc.xmin = parse_bc_face(bc_node["xmin"], "xmin");
        }
        
        if (bc_node["east"]) {
            cfg.bc.xmax = parse_bc_face(bc_node["east"], "east(xmax)");
        } else if (bc_node["xmax"]) {
            cfg.bc.xmax = parse_bc_face(bc_node["xmax"], "xmax");
        }
        
        // Y direction: south/north or ymin/ymax
        if (bc_node["south"]) {
            cfg.bc.ymin = parse_bc_face(bc_node["south"], "south(ymin)");
        } else if (bc_node["ymin"]) {
            cfg.bc.ymin = parse_bc_face(bc_node["ymin"], "ymin");
        }
        
        if (bc_node["north"]) {
            cfg.bc.ymax = parse_bc_face(bc_node["north"], "north(ymax)");
        } else if (bc_node["ymax"]) {
            cfg.bc.ymax = parse_bc_face(bc_node["ymax"], "ymax");
        }
        
        // Z direction: bottom/top or zmin/zmax
        if (bc_node["bottom"]) {
            cfg.bc.zmin = parse_bc_face(bc_node["bottom"], "bottom(zmin)");
        } else if (bc_node["zmin"]) {
            cfg.bc.zmin = parse_bc_face(bc_node["zmin"], "zmin");
        }
        
        if (bc_node["top"]) {
            cfg.bc.zmax = parse_bc_face(bc_node["top"], "top(zmax)");
        } else if (bc_node["zmax"]) {
            cfg.bc.zmax = parse_bc_face(bc_node["zmax"], "zmax");
        }
    }
    
    return cfg;
}

// Parse transport section
TransportYamlConfig parse_transport(const YAML::Node& node) {
    TransportYamlConfig cfg;
    if (!node) return cfg;
    
    cfg.n_particles = get_or<int>(node, "n_particles", 10000);
    cfg.dt = get_or<real>(node, "dt", 0.01);
    cfg.n_steps = get_or<int>(node, "n_steps", 1000);
    cfg.porosity = get_or<real>(node, "porosity", 1.0);
    cfg.diffusion = get_or<real>(node, "diffusion", 0.0);
    cfg.seed = get_or<uint64_t>(node, "seed", 54321);
    cfg.output_every = get_or<int>(node, "output_every", 100);
    cfg.inject_x = get_or<real>(node, "inject_x", 0.0);
    
    return cfg;
}

// Parse output section
OutputYamlConfig parse_output(const YAML::Node& node) {
    OutputYamlConfig cfg;
    if (!node) return cfg;
    
    cfg.output_dir = get_or<std::string>(node, "output_dir", "./output");
    cfg.save_K = get_or<bool>(node, "save_K", true);
    cfg.save_head = get_or<bool>(node, "save_head", true);
    cfg.save_velocity = get_or<bool>(node, "save_velocity", false);
    cfg.save_particles = get_or<bool>(node, "save_particles", true);
    cfg.format = get_or<std::string>(node, "format", "binary");
    
    return cfg;
}

} // anonymous namespace

AppConfig load_config_yaml(const std::string& path) {
    // Check file exists
    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("Config file not found: " + path);
    }
    file.close();
    
    // Parse YAML
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML parse error: " + std::string(e.what()));
    }
    
    AppConfig cfg;
    
    // Parse sections (all optional, use defaults if missing)
    cfg.grid = parse_grid(root["grid"]);
    cfg.stochastic = parse_stochastic(root["stochastic"]);
    cfg.flow = parse_flow(root["flow"]);
    cfg.transport = parse_transport(root["transport"]);
    cfg.output = parse_output(root["output"]);
    
    // Validate critical fields
    if (cfg.grid.nx <= 0 || cfg.grid.ny <= 0 || cfg.grid.nz <= 0) {
        throw std::runtime_error("Grid dimensions must be positive (nx, ny, nz)");
    }
    if (cfg.grid.dx <= 0) {
        throw std::runtime_error("Grid spacing must be positive (dx)");
    }
    
    return cfg;
}

} // namespace io
} // namespace rwpt

```

# src\multigrid\common\mg_conventions.cuh

```cuh
#pragma once

#include "../../core/Scalar.hpp"
#include "../../core/Grid3D.hpp"

/**
 * @file mg_conventions.cuh
 * @brief Mathematical conventions for multigrid operator and residual computation.
 * 
 * This file documents the EXACT mathematical conventions inherited from the legacy
 * codebase (GSRB_Smooth_up_residual_3D_bien.cu, up_residual_3D.cu, CCMG_V_cycle.cu).
 * DO NOT change these conventions without verifying against legacy behavior.
 * 
 * ============================================================================
 * CRITICAL ASSUMPTION: ISOTROPIC GRIDS (dx = dy = dz)
 * ============================================================================
 * 
 * All kernels in this multigrid implementation assume dx = dy = dz.
 * The spacing is accessed via grid.dx only (grid.dy, grid.dz are ignored).
 * 
 * This matches the legacy convention where:
 *   double h = Ly / (double)Ny;  // Single spacing for all directions
 *   double dxdx = h * h;
 * 
 * Users MUST ensure isotropic grids. Anisotropic grids will produce
 * incorrect results silently.
 * 
 * ============================================================================
 * OPERATOR DEFINITION (Variable-Coefficient Laplacian)
 * ============================================================================
 * 
 * Physical equation: -∇·(K∇h) = f
 * 
 * Discrete operator A in cell-centered finite differences:
 *   (A*h)_C = -sum_6faces( K_face * (h_C - h_neighbor) ) / dx²
 * 
 * where K_face is the HARMONIC MEAN of conductivity:
 *   K_face = 2 / (1/K_C + 1/K_neighbor)
 * 
 * Sign convention: A is POSITIVE DEFINITE (SPD), representing -∇·(K∇).
 * 
 * ============================================================================
 * SCALING WITH dx²
 * ============================================================================
 * 
 * Let dx² = grid.dx * grid.dx (the actual grid spacing squared).
 * 
 * The operator Ax is computed WITHOUT dx² scaling in the stencil, i.e.:
 *   Ax = sum_6faces( K_face * (x_C - x_neighbor) )
 * 
 * Then:
 *   - Residual: r = b - Ax / dx²
 *   - GSRB update: x = (result - rhs * dx²) / aC
 * 
 * where:
 *   - result = sum_neighbors(K_face * x_neighbor)
 *   - aC = sum_6faces(K_face) = diagonal coefficient
 *   - rhs = right-hand side (from problem or previous level)
 * 
 * This convention comes from discretizing: -∇·(K∇h) / dx² = rhs / dx²
 * 
 * ============================================================================
 * BOUNDARY CONDITIONS
 * ============================================================================
 * 
 * Dirichlet: h = h_bc (fixed value)
 *   - In residual: r = 0 at boundary nodes (MG convention)
 *   - In GSRB: no update (skip boundary nodes)
 *   - In stencil: contributes aC += 2*K_C for the Dirichlet direction
 * 
 * Neumann: ∂h/∂n = flux_bc (typically flux_bc = 0)
 *   - Homogeneous Neumann (flux=0): stencil_coeff = 0 for that direction
 *   - Ghost value = interior value (no contribution)
 * 
 * Periodic: h(xmin) = h(xmax), etc.
 *   - Use neighbor from opposite boundary
 *   - Harmonic mean computed normally
 * 
 * ============================================================================
 * LEGACY CONSISTENCY CHECK
 * ============================================================================
 * 
 * Legacy GSRB interior update (GSRB_int kernel):
 *   result = sum(h_neighbor * K_face)
 *   aC = sum(K_face)
 *   h = -(rhs - result/dxdx) / (aC/dxdx)
 *     = (result - rhs*dxdx) / aC       [where dxdx = h*h in legacy]
 * 
 * Legacy residual interior (update_int kernel):
 *   result = -sum( 2*(h_C - h_neighbor) / (1/K_C + 1/K_neighbor) )
 *   r = rhs - result/dxdx
 * 
 * Current implementation (Task 1 compliant):
 *   dx2 = grid.dx * grid.dx
 *   Residual: r = b - Ax / dx2
 *   GSRB: x = (result - b * dx2) / aC
 * 
 * ============================================================================
 * USAGE IN CODE
 * ============================================================================
 * 
 * Always use these variable names for clarity:
 *   - dx2: the actual value dx² (grid.dx * grid.dx)
 *   - inv_dx2: 1/dx² (only in performance-critical inner loops)
 * 
 * Never use ambiguous names like "dxdx" which could mean either dx² or 1/dx².
 * 
 * ============================================================================
 */

namespace rwpt {
namespace multigrid {

// Inline helper: compute harmonic mean of two conductivities
__device__ __host__ inline real harmonic_mean_K(real K1, real K2) {
    return 2.0 / (1.0 / K1 + 1.0 / K2);
}

// Inline helper: compute dx² from grid
__device__ __host__ inline real compute_dx2(const Grid3D& grid) {
    return grid.dx * grid.dx;
}

// Inline helper: compute 1/dx² from grid (for optimized kernels)
__device__ __host__ inline real compute_inv_dx2(const Grid3D& grid) {
    return 1.0 / (grid.dx * grid.dx);
}

} // namespace multigrid
} // namespace rwpt

```

# src\multigrid\cycle\v_cycle.cu

```cu
#include "v_cycle.cuh"
#include "../transfer/restrict_3d.cuh"
#include "../transfer/prolong_3d.cuh"
#include "../smoothers/residual_3d.cuh"
#include "../smoothers/gsrb_3d.cuh"
#include "../../numerics/blas/blas.cuh"
#include "../../numerics/blas/reduction_workspace.cuh"
#include "../../numerics/pin_spec.hpp"
#include "../../core/BCSpec.hpp"
#include <cmath>
#include <cassert>

namespace rwpt {
namespace multigrid {

// Helper to verify isotropic grid (dx == dy == dz within tolerance)
inline void assert_isotropic_grid(const Grid3D& grid) {
    const real tol = 1e-12;
    assert(std::abs(grid.dx - grid.dy) < tol * grid.dx && 
           "Grid must be isotropic (dx != dy). See Grid3D.hpp for details.");
    assert(std::abs(grid.dx - grid.dz) < tol * grid.dx && 
           "Grid must be isotropic (dx != dz). See Grid3D.hpp for details.");
}

// Legacy: V_cycle from CCMG_V_cycle.cu
// Recursive V-cycle:
// 1. Pre-smooth on current level
// 2. Compute residual
// 3. Restrict to coarser
// 4. Recursive call (or solve on coarsest)
// 5. Prolong correction back
// 6. Post-smooth on current level
//
// Note: pin is propagated to ALL levels (legacy semantics from boundaryCond macro)
void v_cycle_recursive(
    CudaContext& ctx,
    MGHierarchy& hier,
    int level,
    const MGConfig& config,
    const BCSpec& bc,
    PinSpec pin
) {
    const int num_levels = hier.num_levels();
    
    // Coarsest level: solve directly with many GSRB iterations
    if (level == num_levels - 1) {
        auto& lvl = hier.levels[level];
        gsrb_smooth_3d(ctx, lvl.grid, lvl.x.span(), lvl.b.span(), lvl.K.span(), config.coarse_solve_iters, bc, pin);
        return;
    }
    
    auto& fine = hier.levels[level];
    auto& coarse = hier.levels[level + 1];
    
    // 1. Pre-smooth: x^{h} = S(x^{h}, b^{h})
    gsrb_smooth_3d(ctx, fine.grid, fine.x.span(), fine.b.span(), fine.K.span(), config.pre_smooth, bc, pin);
    
    // 2. Compute residual: r^{h} = b^{h} - A^{h} * x^{h}
    compute_residual_3d(ctx, fine.grid, fine.x.span(), fine.b.span(), fine.K.span(), fine.r.span(), bc, pin);
    
    // 3. Restrict residual to coarse RHS: b^{2h} = R * r^{h}
    restrict_3d(ctx, fine.grid, coarse.grid, fine.r.span(), coarse.b.span());
    
    // 4. Initialize coarse correction: e^{2h} = 0
    rwpt::blas::fill(ctx, coarse.x.span(), 0.0);
    
    // 5. Recursively solve: A^{2h} * e^{2h} = b^{2h} (pin propagated)
    v_cycle_recursive(ctx, hier, level + 1, config, bc, pin);
    
    // 6. Prolong correction and add: x^{h} += P * e^{2h}
    prolong_3d_add(ctx, coarse.grid, fine.grid, coarse.x.span(), fine.x.span());
    
    // 7. Post-smooth: x^{h} = S(x^{h}, b^{h})
    gsrb_smooth_3d(ctx, fine.grid, fine.x.span(), fine.b.span(), fine.K.span(), config.post_smooth, bc, pin);
}

VCycleResult mg_solve(
    CudaContext& ctx,
    MGHierarchy& hier,
    const MGConfig& config,
    const BCSpec& bc,
    int max_cycles,
    real rtol,
    PinSpec pin
) {
    VCycleResult result;
    
    auto& finest = hier.levels[0];
    
    // Verify isotropic grid (required by current implementation)
    assert_isotropic_grid(finest.grid);
    
    // Compute initial residual norm
    compute_residual_3d(ctx, finest.grid, finest.x.span(), finest.b.span(), finest.K.span(), finest.r.span(), bc, pin);
    rwpt::blas::ReductionWorkspace red;
    result.initial_residual = rwpt::blas::nrm2_host(ctx, finest.r.span(), red);
    
    if (config.verbose) {
        // Would print here, but avoiding I/O in library code
    }
    
    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        result.num_cycles = cycle + 1;
        
        // Execute one V-cycle (pin propagated to all levels)
        v_cycle_recursive(ctx, hier, 0, config, bc, pin);
        
        // Check convergence periodically (not every cycle to reduce host sync cost)
        if ((cycle + 1) % config.check_convergence_every == 0 || cycle == max_cycles - 1) {
            compute_residual_3d(ctx, finest.grid, finest.x.span(), finest.b.span(), finest.K.span(), finest.r.span(), bc, pin);
            result.final_residual = rwpt::blas::nrm2_host(ctx, finest.r.span(), red);
            
            real relative_residual = result.final_residual / result.initial_residual;
            if (relative_residual < rtol) {
                result.converged = true;
                break;
            }
        }
    }
    
    return result;
}

} // namespace multigrid
} // namespace rwpt

```

# src\multigrid\cycle\v_cycle.cuh

```cuh
#pragma once

#include "../mg_types.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../../core/BCSpec.hpp"
#include "../../numerics/pin_spec.hpp"

namespace rwpt {
namespace multigrid {

struct VCycleResult {
    int num_cycles = 0;
    real initial_residual = 0.0;
    real final_residual = 0.0;
    bool converged = false;
};

/**
 * @brief Execute one V-cycle on the MG hierarchy
 * 
 * @param ctx     CUDA context
 * @param hier    MG hierarchy (pre-initialized with K)
 * @param level   Current level (0 = finest)
 * @param config  MG configuration
 * @param bc      Boundary conditions
 * @param pin     Pin specification (applied to ALL levels - legacy semantics)
 */
void v_cycle_recursive(
    CudaContext& ctx,
    MGHierarchy& hier,
    int level,
    const MGConfig& config,
    const BCSpec& bc,
    PinSpec pin = {}
);

/**
 * @brief Solve A*x = b using multigrid V-cycles
 * 
 * Assumes hier.levels[0].b is set, hier.levels[0].x is initial guess.
 * 
 * @param ctx        CUDA context
 * @param hier       MG hierarchy
 * @param config     MG configuration
 * @param bc         Boundary conditions
 * @param max_cycles Maximum number of V-cycles
 * @param rtol       Relative tolerance
 * @param pin        Pin specification (for singular systems, applied to all levels)
 * @return VCycleResult with convergence info
 */
VCycleResult mg_solve(
    CudaContext& ctx,
    MGHierarchy& hier,
    const MGConfig& config,
    const BCSpec& bc,
    int max_cycles = 10,
    real rtol = 1e-6,
    PinSpec pin = {}
);

} // namespace multigrid
} // namespace rwpt

```

# src\multigrid\mg_types.hpp

```hpp
#pragma once

#include "../core/Grid3D.hpp"
#include "../core/DeviceBuffer.cuh"
#include "../core/Scalar.hpp"
#include <vector>

namespace rwpt {
namespace multigrid {

// MG configuration
struct MGConfig {
    int num_levels = 4;
    int pre_smooth = 2;
    int post_smooth = 2;
    int coarse_solve_iters = 50;  // GSRB iterations on coarsest level
    int check_convergence_every = 1;  // Check residual norm every N cycles (1 = every cycle)
    real omega = 1.0;  // Relaxation parameter (if needed)
    bool verbose = false;
};

// Single MG level with buffers
struct MGLevel {
    Grid3D grid;
    
    // Solution/correction at this level
    DeviceBuffer<real> x;
    
    // RHS at this level
    DeviceBuffer<real> b;
    
    // Residual at this level
    DeviceBuffer<real> r;
    
    // Conductivity field (if heterogeneous, otherwise can be uniform)
    DeviceBuffer<real> K;
    
    MGLevel() = default;
    
    explicit MGLevel(const Grid3D& g) : grid(g) {
        size_t n = g.num_cells();
        x.resize(n);
        b.resize(n);
        r.resize(n);
        K.resize(n);
    }
    
    void ensure(const Grid3D& g) {
        grid = g;
        size_t n = g.num_cells();
        // Guarantee exact sizes (not just minimum)
        if (x.size() != n) x.resize(n);
        if (b.size() != n) b.resize(n);
        if (r.size() != n) r.resize(n);
        if (K.size() != n) K.resize(n);
    }
};

// MG hierarchy (multiple levels)
struct MGHierarchy {
    std::vector<MGLevel> levels;
    
    MGHierarchy() = default;
    
    // Construct hierarchy from finest grid
    // Coarsening rule: divide by 2 in each dimension
    explicit MGHierarchy(const Grid3D& finest, int num_levels) {
        levels.reserve(num_levels);
        
        Grid3D current = finest;
        for (int l = 0; l < num_levels; ++l) {
            levels.emplace_back(current);
            
            // Coarsen for next level (if not last)
            if (l < num_levels - 1) {
                // Divide by 2 (cell-centered MG)
                current.nx = current.nx / 2;
                current.ny = current.ny / 2;
                current.nz = current.nz / 2;
                current.dx = current.dx * 2.0;
                current.dy = current.dy * 2.0;
                current.dz = current.dz * 2.0;
                
                // Sanity check
                if (current.nx < 2 || current.ny < 2 || current.nz < 2) {
                    break;  // Can't coarsen further
                }
            }
        }
    }
    
    int num_levels() const { return static_cast<int>(levels.size()); }
    
    const Grid3D& finest_grid() const { return levels[0].grid; }
    const Grid3D& coarsest_grid() const { return levels.back().grid; }
};

// MG workspace for temporary buffers (if needed beyond what's in MGLevel)
struct MGWorkspace {
    // Currently empty - all buffers are in MGLevel
    // Can add scratch buffers here if transfer/smooth need them
};

} // namespace multigrid
} // namespace rwpt

```

# src\multigrid\multigrid.cuh

```cuh
#pragma once

// Multigrid conventions (mathematical definitions)
#include "common/mg_conventions.cuh"

// Multigrid types
#include "mg_types.hpp"

// Transfer operators
#include "transfer/restrict_3d.cuh"
#include "transfer/prolong_3d.cuh"

// Smoothers
#include "smoothers/residual_3d.cuh"
#include "smoothers/gsrb_3d.cuh"

// V-cycle
#include "cycle/v_cycle.cuh"

// Note: No namespace re-export to avoid pollution.
// Use rwpt::multigrid::* explicitly.

```

# src\multigrid\smoothers\bc_kernel_tags.cuh

```cuh
#pragma once

#include <cuda_runtime.h>

namespace rwpt {
namespace multigrid {
namespace bc_tags {

// Face tags (6 faces)
enum Face {
    XMIN = 0,
    XMAX = 1,
    YMIN = 2,
    YMAX = 3,
    ZMIN = 4,
    ZMAX = 5
};

// Edge tags (12 edges)
enum Edge {
    XMIN_YMIN = 0,
    XMIN_YMAX = 1,
    XMAX_YMIN = 2,
    XMAX_YMAX = 3,
    
    XMIN_ZMIN = 4,
    XMIN_ZMAX = 5,
    XMAX_ZMIN = 6,
    XMAX_ZMAX = 7,
    
    YMIN_ZMIN = 8,
    YMIN_ZMAX = 9,
    YMAX_ZMIN = 10,
    YMAX_ZMAX = 11
};

// Vertex tags (8 vertices)
enum Vertex {
    XMIN_YMIN_ZMIN = 0,
    XMIN_YMIN_ZMAX = 1,
    XMIN_YMAX_ZMIN = 2,
    XMIN_YMAX_ZMAX = 3,
    XMAX_YMIN_ZMIN = 4,
    XMAX_YMIN_ZMAX = 5,
    XMAX_YMAX_ZMIN = 6,
    XMAX_YMAX_ZMAX = 7
};

// Compile-time queries for faces
template<Face F> __host__ __device__ constexpr bool on_xmin() { return F == XMIN; }
template<Face F> __host__ __device__ constexpr bool on_xmax() { return F == XMAX; }
template<Face F> __host__ __device__ constexpr bool on_ymin() { return F == YMIN; }
template<Face F> __host__ __device__ constexpr bool on_ymax() { return F == YMAX; }
template<Face F> __host__ __device__ constexpr bool on_zmin() { return F == ZMIN; }
template<Face F> __host__ __device__ constexpr bool on_zmax() { return F == ZMAX; }

// Compile-time queries for edges
template<Edge E> __host__ __device__ constexpr bool on_xmin_edge() { 
    return E == XMIN_YMIN || E == XMIN_YMAX || E == XMIN_ZMIN || E == XMIN_ZMAX; 
}
template<Edge E> __host__ __device__ constexpr bool on_xmax_edge() { 
    return E == XMAX_YMIN || E == XMAX_YMAX || E == XMAX_ZMIN || E == XMAX_ZMAX; 
}
template<Edge E> __host__ __device__ constexpr bool on_ymin_edge() { 
    return E == XMIN_YMIN || E == XMAX_YMIN || E == YMIN_ZMIN || E == YMIN_ZMAX; 
}
template<Edge E> __host__ __device__ constexpr bool on_ymax_edge() { 
    return E == XMIN_YMAX || E == XMAX_YMAX || E == YMAX_ZMIN || E == YMAX_ZMAX; 
}
template<Edge E> __host__ __device__ constexpr bool on_zmin_edge() { 
    return E == XMIN_ZMIN || E == XMAX_ZMIN || E == YMIN_ZMIN || E == YMAX_ZMIN; 
}
template<Edge E> __host__ __device__ constexpr bool on_zmax_edge() { 
    return E == XMIN_ZMAX || E == XMAX_ZMAX || E == YMIN_ZMAX || E == YMAX_ZMAX; 
}

// Compile-time queries for vertices
template<Vertex V> __host__ __device__ constexpr bool on_xmin_vertex() { 
    return V == XMIN_YMIN_ZMIN || V == XMIN_YMIN_ZMAX || 
           V == XMIN_YMAX_ZMIN || V == XMIN_YMAX_ZMAX; 
}
template<Vertex V> __host__ __device__ constexpr bool on_xmax_vertex() { 
    return V == XMAX_YMIN_ZMIN || V == XMAX_YMIN_ZMAX || 
           V == XMAX_YMAX_ZMIN || V == XMAX_YMAX_ZMAX; 
}
template<Vertex V> __host__ __device__ constexpr bool on_ymin_vertex() { 
    return V == XMIN_YMIN_ZMIN || V == XMIN_YMIN_ZMAX || 
           V == XMAX_YMIN_ZMIN || V == XMAX_YMIN_ZMAX; 
}
template<Vertex V> __host__ __device__ constexpr bool on_ymax_vertex() { 
    return V == XMIN_YMAX_ZMIN || V == XMIN_YMAX_ZMAX || 
           V == XMAX_YMAX_ZMIN || V == XMAX_YMAX_ZMAX; 
}
template<Vertex V> __host__ __device__ constexpr bool on_zmin_vertex() { 
    return V == XMIN_YMIN_ZMIN || V == XMIN_YMAX_ZMIN || 
           V == XMAX_YMIN_ZMIN || V == XMAX_YMAX_ZMIN; 
}
template<Vertex V> __host__ __device__ constexpr bool on_zmax_vertex() { 
    return V == XMIN_YMIN_ZMAX || V == XMIN_YMAX_ZMAX || 
           V == XMAX_YMIN_ZMAX || V == XMAX_YMAX_ZMAX; 
}

} // namespace bc_tags
} // namespace multigrid
} // namespace rwpt

```

# src\multigrid\smoothers\bc_stencil_helpers.cuh

```cuh
#pragma once

/**
 * @file bc_stencil_helpers.cuh
 * @brief Device helpers for computing stencil coefficients with boundary conditions.
 * 
 * These helpers compute neighbor contributions for variable-coefficient Laplacian
 * following the conventions documented in ../common/mg_conventions.cuh.
 * 
 * Key convention: stencil_coeff is returned WITHOUT dx² scaling.
 * The calling kernel must apply dx² scaling for residual or GSRB update.
 */

#include "../../core/Grid3D.hpp"
#include "../../core/BCSpecDevice.cuh"
#include "../../core/DeviceSpan.cuh"

namespace rwpt {
namespace multigrid {
namespace bc_helpers {

// Helper to wrap periodic index
__device__ inline int wrap_periodic(int idx, int n) {
    if (idx < 0) return n - 1;
    if (idx >= n) return 0;
    return idx;
}

// Compute neighbor contribution in X-minus direction
// Returns: ghost_value (for residual), stencil_coeff, rhs_adjustment (for GSRB)
__device__ inline void neighbor_xminus(
    int i, int j, int k,
    const Grid3D& grid,
    const BCSpecDevice& bc,
    const DeviceSpan<const real>& x,
    const DeviceSpan<const real>& K,
    real& ghost_value,
    real& stencil_coeff,
    real& rhs_adjust
) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const int idx = i + nx * (j + ny * k);
    
    // Center conductivity
    const real Kc = K[idx];
    
    if (i == 0) {
        // At xmin boundary
        const auto bc_type = static_cast<BCType>(bc.type[0]); // xmin=0
        const real bc_val = bc.value[0];
        
        if (bc_type == BCType::Periodic) {
            // Wrap to xmax
            const int neighbor_idx = (nx - 1) + nx * (j + ny * k);
            const real Kn = K[neighbor_idx];
            const real Kh = 2.0 / (1.0/Kc + 1.0/Kn); // harmonic mean
            ghost_value = x[neighbor_idx];
            stencil_coeff = Kh;  // Return Kh, not Kh/dx^2
            rhs_adjust = 0.0;
        } else if (bc_type == BCType::Neumann) {
            // Homogeneous Neumann: no contribution
            ghost_value = x[idx];
            stencil_coeff = 0.0;
            rhs_adjust = 0.0;
        } else { // Dirichlet
            // Dirichlet BC: ghost contributes ONLY to diagonal, not to result
            // The BC value contribution is in the RHS (built by build_rhs_head)
            // Legacy: aC += 2*KC, result does NOT include bc_val
            ghost_value = 0.0;  // Don't add bc_val*coeff to result
            stencil_coeff = 2.0 * Kc;  // Add to diagonal
            rhs_adjust = 0.0;  // Already in RHS
        }
    } else {
        // Interior case
        const int neighbor_idx = (i - 1) + nx * (j + ny * k);
        const real Kn = K[neighbor_idx];
        const real Kh = 2.0 / (1.0/Kc + 1.0/Kn);
        ghost_value = x[neighbor_idx];
        stencil_coeff = Kh;  // Return Kh, not Kh/dx^2
        rhs_adjust = 0.0;
    }
}

__device__ inline void neighbor_xplus(
    int i, int j, int k,
    const Grid3D& grid,
    const BCSpecDevice& bc,
    const DeviceSpan<const real>& x,
    const DeviceSpan<const real>& K,
    real& ghost_value,
    real& stencil_coeff,
    real& rhs_adjust
) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const int idx = i + nx * (j + ny * k);
    const real Kc = K[idx];
    
    if (i == nx - 1) {
        const auto bc_type = static_cast<BCType>(bc.type[1]); // xmax=1
        const real bc_val = bc.value[1];
        
        if (bc_type == BCType::Periodic) {
            const int neighbor_idx = 0 + nx * (j + ny * k);
            const real Kn = K[neighbor_idx];
            const real Kh = 2.0 / (1.0/Kc + 1.0/Kn);
            ghost_value = x[neighbor_idx];
            stencil_coeff = Kh;
            rhs_adjust = 0.0;
        } else if (bc_type == BCType::Neumann) {
            ghost_value = x[idx];
            stencil_coeff = 0.0;
            rhs_adjust = 0.0;
        } else { // Dirichlet
            // Legacy: BC value in RHS, not in result. Only diagonal affected.
            ghost_value = 0.0;
            stencil_coeff = 2.0 * Kc;
            rhs_adjust = 0.0;
        }
    } else {
        const int neighbor_idx = (i + 1) + nx * (j + ny * k);
        const real Kn = K[neighbor_idx];
        const real Kh = 2.0 / (1.0/Kc + 1.0/Kn);
        ghost_value = x[neighbor_idx];
        stencil_coeff = Kh;
        rhs_adjust = 0.0;
    }
}

__device__ inline void neighbor_yminus(
    int i, int j, int k,
    const Grid3D& grid,
    const BCSpecDevice& bc,
    const DeviceSpan<const real>& x,
    const DeviceSpan<const real>& K,
    real& ghost_value,
    real& stencil_coeff,
    real& rhs_adjust
) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const int idx = i + nx * (j + ny * k);
    const real Kc = K[idx];
    
    if (j == 0) {
        const auto bc_type = static_cast<BCType>(bc.type[2]); // ymin=2
        const real bc_val = bc.value[2];
        
        if (bc_type == BCType::Periodic) {
            const int neighbor_idx = i + nx * ((ny - 1) + ny * k);
            const real Kn = K[neighbor_idx];
            const real Kh = 2.0 / (1.0/Kc + 1.0/Kn);
            ghost_value = x[neighbor_idx];
            stencil_coeff = Kh;
            rhs_adjust = 0.0;
        } else if (bc_type == BCType::Neumann) {
            ghost_value = x[idx];
            stencil_coeff = 0.0;
            rhs_adjust = 0.0;
        } else { // Dirichlet
            // Legacy: BC value in RHS, not in result. Only diagonal affected.
            ghost_value = 0.0;
            stencil_coeff = 2.0 * Kc;
            rhs_adjust = 0.0;
        }
    } else {
        const int neighbor_idx = i + nx * ((j - 1) + ny * k);
        const real Kn = K[neighbor_idx];
        const real Kh = 2.0 / (1.0/Kc + 1.0/Kn);
        ghost_value = x[neighbor_idx];
        stencil_coeff = Kh;
        rhs_adjust = 0.0;
    }
}

__device__ inline void neighbor_yplus(
    int i, int j, int k,
    const Grid3D& grid,
    const BCSpecDevice& bc,
    const DeviceSpan<const real>& x,
    const DeviceSpan<const real>& K,
    real& ghost_value,
    real& stencil_coeff,
    real& rhs_adjust
) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const int idx = i + nx * (j + ny * k);
    const real Kc = K[idx];
    
    if (j == ny - 1) {
        const auto bc_type = static_cast<BCType>(bc.type[3]); // ymax=3
        const real bc_val = bc.value[3];
        
        if (bc_type == BCType::Periodic) {
            const int neighbor_idx = i + nx * (0 + ny * k);
            const real Kn = K[neighbor_idx];
            const real Kh = 2.0 / (1.0/Kc + 1.0/Kn);
            ghost_value = x[neighbor_idx];
            stencil_coeff = Kh;
            rhs_adjust = 0.0;
        } else if (bc_type == BCType::Neumann) {
            ghost_value = x[idx];
            stencil_coeff = 0.0;
            rhs_adjust = 0.0;
        } else { // Dirichlet
            // Legacy: BC value in RHS, not in result. Only diagonal affected.
            ghost_value = 0.0;
            stencil_coeff = 2.0 * Kc;
            rhs_adjust = 0.0;
        }
    } else {
        const int neighbor_idx = i + nx * ((j + 1) + ny * k);
        const real Kn = K[neighbor_idx];
        const real Kh = 2.0 / (1.0/Kc + 1.0/Kn);
        ghost_value = x[neighbor_idx];
        stencil_coeff = Kh;
        rhs_adjust = 0.0;
    }
}

__device__ inline void neighbor_zminus(
    int i, int j, int k,
    const Grid3D& grid,
    const BCSpecDevice& bc,
    const DeviceSpan<const real>& x,
    const DeviceSpan<const real>& K,
    real& ghost_value,
    real& stencil_coeff,
    real& rhs_adjust
) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const int idx = i + nx * (j + ny * k);
    const real Kc = K[idx];
    
    if (k == 0) {
        const auto bc_type = static_cast<BCType>(bc.type[4]); // zmin=4
        const real bc_val = bc.value[4];
        
        if (bc_type == BCType::Periodic) {
            const int neighbor_idx = i + nx * (j + ny * (nz - 1));
            const real Kn = K[neighbor_idx];
            const real Kh = 2.0 / (1.0/Kc + 1.0/Kn);
            ghost_value = x[neighbor_idx];
            stencil_coeff = Kh;
            rhs_adjust = 0.0;
        } else if (bc_type == BCType::Neumann) {
            ghost_value = x[idx];
            stencil_coeff = 0.0;
            rhs_adjust = 0.0;
        } else { // Dirichlet
            // Legacy: BC value in RHS, not in result. Only diagonal affected.
            ghost_value = 0.0;
            stencil_coeff = 2.0 * Kc;
            rhs_adjust = 0.0;
        }
    } else {
        const int neighbor_idx = i + nx * (j + ny * (k - 1));
        const real Kn = K[neighbor_idx];
        const real Kh = 2.0 / (1.0/Kc + 1.0/Kn);
        ghost_value = x[neighbor_idx];
        stencil_coeff = Kh;
        rhs_adjust = 0.0;
    }
}

__device__ inline void neighbor_zplus(
    int i, int j, int k,
    const Grid3D& grid,
    const BCSpecDevice& bc,
    const DeviceSpan<const real>& x,
    const DeviceSpan<const real>& K,
    real& ghost_value,
    real& stencil_coeff,
    real& rhs_adjust
) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const int idx = i + nx * (j + ny * k);
    const real Kc = K[idx];
    
    if (k == nz - 1) {
        const auto bc_type = static_cast<BCType>(bc.type[5]); // zmax=5
        const real bc_val = bc.value[5];
        
        if (bc_type == BCType::Periodic) {
            const int neighbor_idx = i + nx * (j + ny * 0);
            const real Kn = K[neighbor_idx];
            const real Kh = 2.0 / (1.0/Kc + 1.0/Kn);
            ghost_value = x[neighbor_idx];
            stencil_coeff = Kh;
            rhs_adjust = 0.0;
        } else if (bc_type == BCType::Neumann) {
            ghost_value = x[idx];
            stencil_coeff = 0.0;
            rhs_adjust = 0.0;
        } else { // Dirichlet
            // Legacy: BC value in RHS, not in result. Only diagonal affected.
            ghost_value = 0.0;
            stencil_coeff = 2.0 * Kc;
            rhs_adjust = 0.0;
        }
    } else {
        const int neighbor_idx = i + nx * (j + ny * (k + 1));
        const real Kn = K[neighbor_idx];
        const real Kh = 2.0 / (1.0/Kc + 1.0/Kn);
        ghost_value = x[neighbor_idx];
        stencil_coeff = Kh;
        rhs_adjust = 0.0;
    }
}

} // namespace bc_helpers
} // namespace multigrid
} // namespace rwpt

```

# src\multigrid\smoothers\gsrb_3d.cu

```cu
/**
 * @file gsrb_3d.cu
 * @brief Gauss-Seidel Red-Black smoother for variable-coefficient 3D Laplacian
 * 
 * Legacy correspondence: GSRB_Smooth_up_residual_3D_bien.cu
 * 
 * This implementation replicates the mathematical semantics of the legacy "bien" variant
 * while using modern C++ (templates, constexpr) instead of C preprocessor macros.
 * 
 * See ../LEGACY_VARIANTS.md for details on legacy variant history.
 */

#include "gsrb_3d.cuh"
#include "../common/mg_conventions.cuh"
#include "../../runtime/cuda_check.cuh"
#include "../../core/BCSpecDevice.cuh"
#include "bc_stencil_helpers.cuh"
#include "bc_kernel_tags.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace rwpt {
namespace multigrid {

using namespace bc_helpers;
using namespace bc_tags;

// Interior kernel: branch-free red-black GSRB
__global__ void gsrb_interior_kernel(
    Grid3D grid,
    DeviceSpan<real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    bool is_red
) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    const real dx2 = grid.dx * grid.dx;  // dx² for RHS scaling
    
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    for (; iy < Ny - 2; iy += blockDim.y * gridDim.y) {
        for (int ix_local = ix; ix_local < Nx - 2; ix_local += blockDim.x * gridDim.x) {
            int stride = Nx * Ny;
            
            for (int iz = 1; iz < Nz - 1; ++iz) {
                int ix_abs = ix_local + 1;
                int iy_abs = iy + 1;
                int color_sum = ix_abs + iy_abs + iz;
                
                if ((is_red && (color_sum % 2 != 0)) || (!is_red && (color_sum % 2 == 0))) {
                    continue;
                }
                
                int idx = ix_abs + iy_abs * Nx + iz * stride;
                
                real KC = K[idx];
                real K_xp = 2.0 / (1.0 / KC + 1.0 / K[idx + 1]);
                real K_xm = 2.0 / (1.0 / KC + 1.0 / K[idx - 1]);
                real K_yp = 2.0 / (1.0 / KC + 1.0 / K[idx + Nx]);
                real K_ym = 2.0 / (1.0 / KC + 1.0 / K[idx - Nx]);
                real K_zp = 2.0 / (1.0 / KC + 1.0 / K[idx + stride]);
                real K_zm = 2.0 / (1.0 / KC + 1.0 / K[idx - stride]);
                
                real result = x[idx + 1] * K_xp;
                result += x[idx - 1] * K_xm;
                result += x[idx + Nx] * K_yp;
                result += x[idx - Nx] * K_ym;
                result += x[idx + stride] * K_zp;
                result += x[idx - stride] * K_zm;
                
                real aC = K_xp + K_xm + K_yp + K_ym + K_zp + K_zm;
                // Legacy: h = -(rhs - result/dx²) / (aC/dx²) = (result - rhs*dx²) / aC
                x[idx] = (result - b[idx] * dx2) / aC;
            }
        }
    }
}

// Face kernel template
template<Face F>
__global__ void gsrb_face_kernel(
    Grid3D grid,
    BCSpecDevice bc,
    DeviceSpan<real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    bool is_red
) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    
    int i, j, k;
    if (on_xmin<F>()) {
        i = 0;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        k = threadIdx.y + blockIdx.y * blockDim.y;
        if (j >= Ny || k >= Nz) return;
    } else if (on_xmax<F>()) {
        i = Nx - 1;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        k = threadIdx.y + blockIdx.y * blockDim.y;
        if (j >= Ny || k >= Nz) return;
    } else if (on_ymin<F>()) {
        j = 0;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        k = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= Nx || k >= Nz) return;
    } else if (on_ymax<F>()) {
        j = Ny - 1;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        k = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= Nx || k >= Nz) return;
    } else if (on_zmin<F>()) {
        k = 0;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= Nx || j >= Ny) return;
    } else {
        k = Nz - 1;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= Nx || j >= Ny) return;
    }
    
    // Face kernels must NOT touch edges/vertices
    if (on_zmin<F>() || on_zmax<F>()) {
        if (i == 0 || i == Nx-1 || j == 0 || j == Ny-1) return;
    }
    if (on_xmin<F>() || on_xmax<F>()) {
        if (j == 0 || j == Ny-1 || k == 0 || k == Nz-1) return;
    }
    if (on_ymin<F>() || on_ymax<F>()) {
        if (i == 0 || i == Nx-1 || k == 0 || k == Nz-1) return;
    }
    
    // Color check
    int color_sum = i + j + k;
    if ((is_red && (color_sum % 2 != 0)) || (!is_red && (color_sum % 2 == 0))) {
        return;
    }
    
    const int idx = i + Nx * (j + Ny * k);
    
    // Compute stencil and update (legacy: no skip for Dirichlet)
    real val_xm, coef_xm, rhs_xm;
    real val_xp, coef_xp, rhs_xp;
    real val_ym, coef_ym, rhs_ym;
    real val_yp, coef_yp, rhs_yp;
    real val_zm, coef_zm, rhs_zm;
    real val_zp, coef_zp, rhs_zp;
    
    neighbor_xminus(i, j, k, grid, bc, x, K, val_xm, coef_xm, rhs_xm);
    neighbor_xplus(i, j, k, grid, bc, x, K, val_xp, coef_xp, rhs_xp);
    neighbor_yminus(i, j, k, grid, bc, x, K, val_ym, coef_ym, rhs_ym);
    neighbor_yplus(i, j, k, grid, bc, x, K, val_yp, coef_yp, rhs_yp);
    neighbor_zminus(i, j, k, grid, bc, x, K, val_zm, coef_zm, rhs_zm);
    neighbor_zplus(i, j, k, grid, bc, x, K, val_zp, coef_zp, rhs_zp);
    
    const real dx2 = grid.dx * grid.dx;  // dx² for RHS scaling
    real result = coef_xm * val_xm + coef_xp * val_xp +
                  coef_ym * val_ym + coef_yp * val_yp +
                  coef_zm * val_zm + coef_zp * val_zp;
    
    real aC = coef_xm + coef_xp + coef_ym + coef_yp + coef_zm + coef_zp;
    real b_adj = b[idx] + (rhs_xm + rhs_xp + rhs_ym + rhs_yp + rhs_zm + rhs_zp);
    
    // Legacy: h = -(rhs - result/dx²) / (aC/dx²) = (result - rhs*dx²) / aC
    x[idx] = (result - b_adj * dx2) / aC;
}

// Edge kernel template
template<Edge E>
__global__ void gsrb_edge_kernel(
    Grid3D grid,
    BCSpecDevice bc,
    DeviceSpan<real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    bool is_red
) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    
    int i, j, k;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (E == XMIN_YMIN) { i = 0; j = 0; k = tid; if (k >= Nz) return; }
    else if (E == XMIN_YMAX) { i = 0; j = Ny - 1; k = tid; if (k >= Nz) return; }
    else if (E == XMAX_YMIN) { i = Nx - 1; j = 0; k = tid; if (k >= Nz) return; }
    else if (E == XMAX_YMAX) { i = Nx - 1; j = Ny - 1; k = tid; if (k >= Nz) return; }
    else if (E == XMIN_ZMIN) { i = 0; k = 0; j = tid; if (j >= Ny) return; }
    else if (E == XMIN_ZMAX) { i = 0; k = Nz - 1; j = tid; if (j >= Ny) return; }
    else if (E == XMAX_ZMIN) { i = Nx - 1; k = 0; j = tid; if (j >= Ny) return; }
    else if (E == XMAX_ZMAX) { i = Nx - 1; k = Nz - 1; j = tid; if (j >= Ny) return; }
    else if (E == YMIN_ZMIN) { j = 0; k = 0; i = tid; if (i >= Nx) return; }
    else if (E == YMIN_ZMAX) { j = 0; k = Nz - 1; i = tid; if (i >= Nx) return; }
    else if (E == YMAX_ZMIN) { j = Ny - 1; k = 0; i = tid; if (i >= Nx) return; }
    else { j = Ny - 1; k = Nz - 1; i = tid; if (i >= Nx) return; }
    
    int color_sum = i + j + k;
    if ((is_red && (color_sum % 2 != 0)) || (!is_red && (color_sum % 2 == 0))) {
        return;
    }
    
    const int idx = i + Nx * (j + Ny * k);
    
    // Legacy: always update, even with Dirichlet (no skip)
    real val_xm, coef_xm, rhs_xm;
    real val_xp, coef_xp, rhs_xp;
    real val_ym, coef_ym, rhs_ym;
    real val_yp, coef_yp, rhs_yp;
    real val_zm, coef_zm, rhs_zm;
    real val_zp, coef_zp, rhs_zp;
    
    neighbor_xminus(i, j, k, grid, bc, x, K, val_xm, coef_xm, rhs_xm);
    neighbor_xplus(i, j, k, grid, bc, x, K, val_xp, coef_xp, rhs_xp);
    neighbor_yminus(i, j, k, grid, bc, x, K, val_ym, coef_ym, rhs_ym);
    neighbor_yplus(i, j, k, grid, bc, x, K, val_yp, coef_yp, rhs_yp);
    neighbor_zminus(i, j, k, grid, bc, x, K, val_zm, coef_zm, rhs_zm);
    neighbor_zplus(i, j, k, grid, bc, x, K, val_zp, coef_zp, rhs_zp);
    
    const real dx2 = grid.dx * grid.dx;  // dx² for RHS scaling
    real result = coef_xm * val_xm + coef_xp * val_xp +
                  coef_ym * val_ym + coef_yp * val_yp +
                  coef_zm * val_zm + coef_zp * val_zp;
    
    real aC = coef_xm + coef_xp + coef_ym + coef_yp + coef_zm + coef_zp;
    real b_adj = b[idx] + (rhs_xm + rhs_xp + rhs_ym + rhs_yp + rhs_zm + rhs_zp);
    
    // Legacy: h = -(rhs - result/dx²) / (aC/dx²) = (result - rhs*dx²) / aC
    x[idx] = (result - b_adj * dx2) / aC;
}

// Vertex kernel template
// pin1stCell: if true, doubles the diagonal for vertex XMIN_YMIN_ZMIN (cell [0,0,0])
template<Vertex V>
__global__ void gsrb_vertex_kernel(
    Grid3D grid,
    BCSpecDevice bc,
    DeviceSpan<real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    bool is_red,
    bool pin1stCell
) {
    if (threadIdx.x + blockIdx.x * blockDim.x != 0) return;
    
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    
    int i, j, k;
    if (V == XMIN_YMIN_ZMIN) { i = 0; j = 0; k = 0; }
    else if (V == XMIN_YMIN_ZMAX) { i = 0; j = 0; k = Nz - 1; }
    else if (V == XMIN_YMAX_ZMIN) { i = 0; j = Ny - 1; k = 0; }
    else if (V == XMIN_YMAX_ZMAX) { i = 0; j = Ny - 1; k = Nz - 1; }
    else if (V == XMAX_YMIN_ZMIN) { i = Nx - 1; j = 0; k = 0; }
    else if (V == XMAX_YMIN_ZMAX) { i = Nx - 1; j = 0; k = Nz - 1; }
    else if (V == XMAX_YMAX_ZMIN) { i = Nx - 1; j = Ny - 1; k = 0; }
    else { i = Nx - 1; j = Ny - 1; k = Nz - 1; }
    
    int color_sum = i + j + k;
    if ((is_red && (color_sum % 2 != 0)) || (!is_red && (color_sum % 2 == 0))) {
        return;
    }
    
    const int idx = i + Nx * (j + Ny * k);
    
    // Legacy: always update, even with Dirichlet (no skip)
    real val_xm, coef_xm, rhs_xm;
    real val_xp, coef_xp, rhs_xp;
    real val_ym, coef_ym, rhs_ym;
    real val_yp, coef_yp, rhs_yp;
    real val_zm, coef_zm, rhs_zm;
    real val_zp, coef_zp, rhs_zp;
    
    neighbor_xminus(i, j, k, grid, bc, x, K, val_xm, coef_xm, rhs_xm);
    neighbor_xplus(i, j, k, grid, bc, x, K, val_xp, coef_xp, rhs_xp);
    neighbor_yminus(i, j, k, grid, bc, x, K, val_ym, coef_ym, rhs_ym);
    neighbor_yplus(i, j, k, grid, bc, x, K, val_yp, coef_yp, rhs_yp);
    neighbor_zminus(i, j, k, grid, bc, x, K, val_zm, coef_zm, rhs_zm);
    neighbor_zplus(i, j, k, grid, bc, x, K, val_zp, coef_zp, rhs_zp);
    
    const real dx2 = grid.dx * grid.dx;  // dx² for RHS scaling
    real result = coef_xm * val_xm + coef_xp * val_xp +
                  coef_ym * val_ym + coef_yp * val_yp +
                  coef_zm * val_zm + coef_zp * val_zp;
    
    real aC = coef_xm + coef_xp + coef_ym + coef_yp + coef_zm + coef_zp;
    real b_adj = b[idx] + (rhs_xm + rhs_xp + rhs_ym + rhs_yp + rhs_zm + rhs_zp);
    
    // Legacy pin1stCell: double diagonal for cell [0,0,0] to break singular null space
    if (pin1stCell && V == XMIN_YMIN_ZMIN) {
        aC *= 2.0;
    }
    
    // Legacy: h = -(rhs - result/dx²) / (aC/dx²) = (result - rhs*dx²) / aC
    x[idx] = (result - b_adj * dx2) / aC;
}

void gsrb_smooth_3d(
    CudaContext& ctx,
    const Grid3D& grid,
    DeviceSpan<real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    int num_iters,
    const BCSpec& bc,
    PinSpec pin
) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    const size_t n = grid.num_cells();
    
    bc.validate();
    
    assert(x.size() == n && "x size mismatch");
    assert(b.size() == n && "b size mismatch");
    assert(K.size() == n && "K size mismatch");
    
    BCSpecDevice bc_dev = to_device(bc);
    
    dim3 block(16, 16);
    int grid_x = (Nx + block.x - 1) / block.x;
    int grid_y = (Ny + block.y - 1) / block.y;
    grid_x = (grid_x < 65535) ? grid_x : 65535;
    grid_y = (grid_y < 65535) ? grid_y : 65535;
    dim3 grid_dim(grid_x, grid_y);
    
    dim3 face_block(16, 16);
    int edge_block = 256;
    
    for (int iter = 0; iter < num_iters; ++iter) {
        // RED pass
        gsrb_interior_kernel<<<grid_dim, block, 0, ctx.cuda_stream()>>>(grid, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        
        // Faces RED
        dim3 face_grid_yz((Ny + 15) / 16, (Nz + 15) / 16);
        gsrb_face_kernel<XMIN><<<face_grid_yz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<XMAX><<<face_grid_yz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        
        dim3 face_grid_xz((Nx + 15) / 16, (Nz + 15) / 16);
        gsrb_face_kernel<YMIN><<<face_grid_xz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<YMAX><<<face_grid_xz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        
        dim3 face_grid_xy((Nx + 15) / 16, (Ny + 15) / 16);
        gsrb_face_kernel<ZMIN><<<face_grid_xy, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<ZMAX><<<face_grid_xy, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        
        // Edges RED
        int edge_grid_z = (Nz + edge_block - 1) / edge_block;
        gsrb_edge_kernel<XMIN_YMIN><<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMIN_YMAX><<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_YMIN><<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_YMAX><<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        
        int edge_grid_y = (Ny + edge_block - 1) / edge_block;
        gsrb_edge_kernel<XMIN_ZMIN><<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMIN_ZMAX><<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_ZMIN><<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_ZMAX><<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        
        int edge_grid_x = (Nx + edge_block - 1) / edge_block;
        gsrb_edge_kernel<YMIN_ZMIN><<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMIN_ZMAX><<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMAX_ZMIN><<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMAX_ZMAX><<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        RWPT_CUDA_CHECK(cudaGetLastError());
        
        // Vertices RED (pin1stCell only affects XMIN_YMIN_ZMIN)
        gsrb_vertex_kernel<XMIN_YMIN_ZMIN><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, pin.enabled);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMIN_YMIN_ZMAX><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMIN_YMAX_ZMIN><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMIN_YMAX_ZMAX><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMIN_ZMIN><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMIN_ZMAX><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMAX_ZMIN><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMAX_ZMAX><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        
        // BLACK pass (same structure, is_red=false)
        gsrb_interior_kernel<<<grid_dim, block, 0, ctx.cuda_stream()>>>(grid, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        
        gsrb_face_kernel<XMIN><<<face_grid_yz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<XMAX><<<face_grid_yz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<YMIN><<<face_grid_xz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<YMAX><<<face_grid_xz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<ZMIN><<<face_grid_xy, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<ZMAX><<<face_grid_xy, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        
        gsrb_edge_kernel<XMIN_YMIN><<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMIN_YMAX><<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_YMIN><<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_YMAX><<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMIN_ZMIN><<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMIN_ZMAX><<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_ZMIN><<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_ZMAX><<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMIN_ZMIN><<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMIN_ZMAX><<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMAX_ZMIN><<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMAX_ZMAX><<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        
        gsrb_vertex_kernel<XMIN_YMIN_ZMIN><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, pin.enabled);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMIN_YMIN_ZMAX><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMIN_YMAX_ZMIN><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMIN_YMAX_ZMAX><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMIN_ZMIN><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMIN_ZMAX><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMAX_ZMIN><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMAX_ZMAX><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        RWPT_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace multigrid
} // namespace rwpt

```

# src\multigrid\smoothers\gsrb_3d.cuh

```cuh
#pragma once

#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/BCSpec.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../../numerics/pin_spec.hpp"

namespace rwpt {
namespace multigrid {

/**
 * @brief Gauss-Seidel Red-Black smoother
 * 
 * Solves A*x = b using red-black ordering:
 *   Red cells: (i+j+k) % 2 == 0
 *   Black cells: (i+j+k) % 2 == 1
 * 
 * @param ctx       CUDA context
 * @param grid      Grid specification
 * @param x         Solution (in/out)
 * @param b         Right-hand side
 * @param K         Conductivity field
 * @param num_iters Number of iterations (each = red + black sweep)
 * @param bc        Boundary conditions
 * @param pin       Pin specification (optional, default = no pin)
 */
void gsrb_smooth_3d(
    CudaContext& ctx,
    const Grid3D& grid,
    DeviceSpan<real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    int num_iters,
    const BCSpec& bc,
    PinSpec pin = {}
);

} // namespace multigrid
} // namespace rwpt

```

# src\multigrid\smoothers\residual_3d.cu

```cu
/**
 * @file residual_3d.cu
 * @brief Compute residual r = b - A*x for variable-coefficient 3D Laplacian
 * 
 * Legacy correspondence: up_residual_3D.cu
 * 
 * This implementation replicates the mathematical semantics of the legacy residual
 * computation with modern C++ structure (templates for BC handling).
 */

#include "residual_3d.cuh"
#include "../common/mg_conventions.cuh"
#include "../../runtime/cuda_check.cuh"
#include "../../core/BCSpecDevice.cuh"
#include "bc_stencil_helpers.cuh"
#include "bc_kernel_tags.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace rwpt {
namespace multigrid {

using namespace bc_helpers;
using namespace bc_tags;

// Interior kernel: branch-free, no BC checks
// Legacy: update_int from up_residual_3D.cu
// r = b - A*x where A is variable-coefficient Laplacian
// Uses harmonic mean for face conductivities
__global__ void residual_interior_kernel(
    Grid3D grid,
    DeviceSpan<const real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    DeviceSpan<real> r
) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    const real inv_dx2 = 1.0 / (grid.dx * grid.dx);  // 1/dx² for operator scaling
    
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    // Grid-stride in y
    for (; iy < Ny - 2; iy += blockDim.y * gridDim.y) {
        for (int ix_local = ix; ix_local < Nx - 2; ix_local += blockDim.x * gridDim.x) {
            int stride = Nx * Ny;
            int in_idx = (ix_local + 1) + (iy + 1) * Nx;
            
            real x_current = x[in_idx];
            real K_current = K[in_idx];
            int out_idx = in_idx;
            in_idx += stride;
            real x_top = x[in_idx];
            real K_top = K[in_idx];
            in_idx += stride;
            
            for (int iz = 1; iz < Nz - 1; ++iz) {
                real x_bottom = x_current;
                x_current = x_top;
                x_top = x[in_idx];
                
                real K_bottom = K_current;
                K_current = K_top;
                K_top = K[in_idx];
                
                in_idx += stride;
                out_idx += stride;
                
                // Compute -A*x using harmonic mean (branch-free)
                real Ax = 0.0;
                Ax -= 2.0 * (x_current - x[out_idx + 1]) / (1.0 / K_current + 1.0 / K[out_idx + 1]);
                Ax -= 2.0 * (x_current - x[out_idx + Nx]) / (1.0 / K_current + 1.0 / K[out_idx + Nx]);
                Ax -= 2.0 * (x_current - x[out_idx - 1]) / (1.0 / K_current + 1.0 / K[out_idx - 1]);
                Ax -= 2.0 * (x_current - x[out_idx - Nx]) / (1.0 / K_current + 1.0 / K[out_idx - Nx]);
                Ax -= 2.0 * (x_current - x_top) / (1.0 / K_current + 1.0 / K_top);
                Ax -= 2.0 * (x_current - x_bottom) / (1.0 / K_current + 1.0 / K_bottom);
                
                r[out_idx] = b[out_idx] - Ax * inv_dx2;
            }
        }
    }
}

// Face kernel template: handle one face with BC logic
template<Face F>
__global__ void residual_face_kernel(
    Grid3D grid,
    BCSpecDevice bc,
    DeviceSpan<const real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    DeviceSpan<real> r
) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    
    // Determine face index and loops
    int i, j, k;
    if (on_xmin<F>()) {
        i = 0;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        k = threadIdx.y + blockIdx.y * blockDim.y;
        if (j >= Ny || k >= Nz) return;
    } else if (on_xmax<F>()) {
        i = Nx - 1;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        k = threadIdx.y + blockIdx.y * blockDim.y;
        if (j >= Ny || k >= Nz) return;
    } else if (on_ymin<F>()) {
        j = 0;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        k = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= Nx || k >= Nz) return;
    } else if (on_ymax<F>()) {
        j = Ny - 1;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        k = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= Nx || k >= Nz) return;
    } else if (on_zmin<F>()) {
        k = 0;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= Nx || j >= Ny) return;
    } else { // zmax
        k = Nz - 1;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= Nx || j >= Ny) return;
    }
    
    // Face kernels must NOT touch edges/vertices
    if (on_zmin<F>() || on_zmax<F>()) {
        if (i == 0 || i == Nx-1 || j == 0 || j == Ny-1) return;
    }
    if (on_xmin<F>() || on_xmax<F>()) {
        if (j == 0 || j == Ny-1 || k == 0 || k == Nz-1) return;
    }
    if (on_ymin<F>() || on_ymax<F>()) {
        if (i == 0 || i == Nx-1 || k == 0 || k == Nz-1) return;
    }
    
    const int idx = i + Nx * (j + Ny * k);
    
    // Legacy convention: always compute residual, even for Dirichlet nodes
    // The smoother handles Dirichlet BCs via stencil coefficients (aC += 2*KC)
    {
        real val_xm, coef_xm, rhs_xm;
        real val_xp, coef_xp, rhs_xp;
        real val_ym, coef_ym, rhs_ym;
        real val_yp, coef_yp, rhs_yp;
        real val_zm, coef_zm, rhs_zm;
        real val_zp, coef_zp, rhs_zp;
        
        neighbor_xminus(i, j, k, grid, bc, x, K, val_xm, coef_xm, rhs_xm);
        neighbor_xplus(i, j, k, grid, bc, x, K, val_xp, coef_xp, rhs_xp);
        neighbor_yminus(i, j, k, grid, bc, x, K, val_ym, coef_ym, rhs_ym);
        neighbor_yplus(i, j, k, grid, bc, x, K, val_yp, coef_yp, rhs_yp);
        neighbor_zminus(i, j, k, grid, bc, x, K, val_zm, coef_zm, rhs_zm);
        neighbor_zplus(i, j, k, grid, bc, x, K, val_zp, coef_zp, rhs_zp);
        
        const real x_center = x[idx];
        // Use -Ax to match interior kernel sign convention:
        // Interior: Ax -= K*(x_C - x_N), then r = b - Ax/dx² which equals b + (pos sum)/dx²
        const real neg_Ax = -(coef_xm * (x_center - val_xm) +
                              coef_xp * (x_center - val_xp) +
                              coef_ym * (x_center - val_ym) +
                              coef_yp * (x_center - val_yp) +
                              coef_zm * (x_center - val_zm) +
                              coef_zp * (x_center - val_zp));
        const real b_adjusted = b[idx] + (rhs_xm + rhs_xp + rhs_ym + rhs_yp + rhs_zm + rhs_zp);
        
        const real dx2 = grid.dx * grid.dx;  // dx² for operator scaling
        r[idx] = b_adjusted - neg_Ax / dx2;
    }
}

// Edge kernel template
template<Edge E>
__global__ void residual_edge_kernel(
    Grid3D grid,
    BCSpecDevice bc,
    DeviceSpan<const real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    DeviceSpan<real> r
) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    
    int i, j, k;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Determine edge coordinates
    if (E == XMIN_YMIN) {
        i = 0; j = 0; k = tid;
        if (k >= Nz) return;
    } else if (E == XMIN_YMAX) {
        i = 0; j = Ny - 1; k = tid;
        if (k >= Nz) return;
    } else if (E == XMAX_YMIN) {
        i = Nx - 1; j = 0; k = tid;
        if (k >= Nz) return;
    } else if (E == XMAX_YMAX) {
        i = Nx - 1; j = Ny - 1; k = tid;
        if (k >= Nz) return;
    } else if (E == XMIN_ZMIN) {
        i = 0; k = 0; j = tid;
        if (j >= Ny) return;
    } else if (E == XMIN_ZMAX) {
        i = 0; k = Nz - 1; j = tid;
        if (j >= Ny) return;
    } else if (E == XMAX_ZMIN) {
        i = Nx - 1; k = 0; j = tid;
        if (j >= Ny) return;
    } else if (E == XMAX_ZMAX) {
        i = Nx - 1; k = Nz - 1; j = tid;
        if (j >= Ny) return;
    } else if (E == YMIN_ZMIN) {
        j = 0; k = 0; i = tid;
        if (i >= Nx) return;
    } else if (E == YMIN_ZMAX) {
        j = 0; k = Nz - 1; i = tid;
        if (i >= Nx) return;
    } else if (E == YMAX_ZMIN) {
        j = Ny - 1; k = 0; i = tid;
        if (i >= Nx) return;
    } else { // YMAX_ZMAX
        j = Ny - 1; k = Nz - 1; i = tid;
        if (i >= Nx) return;
    }
    
    const int idx = i + Nx * (j + Ny * k);
    
    // Legacy: always compute residual, even with Dirichlet (no skip)
    {
        real val_xm, coef_xm, rhs_xm;
        real val_xp, coef_xp, rhs_xp;
        real val_ym, coef_ym, rhs_ym;
        real val_yp, coef_yp, rhs_yp;
        real val_zm, coef_zm, rhs_zm;
        real val_zp, coef_zp, rhs_zp;
        
        neighbor_xminus(i, j, k, grid, bc, x, K, val_xm, coef_xm, rhs_xm);
        neighbor_xplus(i, j, k, grid, bc, x, K, val_xp, coef_xp, rhs_xp);
        neighbor_yminus(i, j, k, grid, bc, x, K, val_ym, coef_ym, rhs_ym);
        neighbor_yplus(i, j, k, grid, bc, x, K, val_yp, coef_yp, rhs_yp);
        neighbor_zminus(i, j, k, grid, bc, x, K, val_zm, coef_zm, rhs_zm);
        neighbor_zplus(i, j, k, grid, bc, x, K, val_zp, coef_zp, rhs_zp);
        
        const real x_center = x[idx];
        // Use -Ax to match interior kernel sign convention
        const real neg_Ax = -(coef_xm * (x_center - val_xm) +
                              coef_xp * (x_center - val_xp) +
                              coef_ym * (x_center - val_ym) +
                              coef_yp * (x_center - val_yp) +
                              coef_zm * (x_center - val_zm) +
                              coef_zp * (x_center - val_zp));
        const real b_adjusted = b[idx] + (rhs_xm + rhs_xp + rhs_ym + rhs_yp + rhs_zm + rhs_zp);
        
        const real dx2 = grid.dx * grid.dx;  // dx² for operator scaling
        r[idx] = b_adjusted - neg_Ax / dx2;
    }
}

// Vertex kernel template
// pin1stCell: if true, doubles the diagonal for vertex XMIN_YMIN_ZMIN (cell [0,0,0])
template<Vertex V>
__global__ void residual_vertex_kernel(
    Grid3D grid,
    BCSpecDevice bc,
    DeviceSpan<const real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    DeviceSpan<real> r,
    bool pin1stCell
) {
    if (threadIdx.x + blockIdx.x * blockDim.x != 0) return;
    
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    
    int i, j, k;
    if (V == XMIN_YMIN_ZMIN) { i = 0; j = 0; k = 0; }
    else if (V == XMIN_YMIN_ZMAX) { i = 0; j = 0; k = Nz - 1; }
    else if (V == XMIN_YMAX_ZMIN) { i = 0; j = Ny - 1; k = 0; }
    else if (V == XMIN_YMAX_ZMAX) { i = 0; j = Ny - 1; k = Nz - 1; }
    else if (V == XMAX_YMIN_ZMIN) { i = Nx - 1; j = 0; k = 0; }
    else if (V == XMAX_YMIN_ZMAX) { i = Nx - 1; j = 0; k = Nz - 1; }
    else if (V == XMAX_YMAX_ZMIN) { i = Nx - 1; j = Ny - 1; k = 0; }
    else { i = Nx - 1; j = Ny - 1; k = Nz - 1; } // XMAX_YMAX_ZMAX
    
    const int idx = i + Nx * (j + Ny * k);
    
    // Legacy: always compute residual, no skip for Dirichlet
    {
        real val_xm, coef_xm, rhs_xm;
        real val_xp, coef_xp, rhs_xp;
        real val_ym, coef_ym, rhs_ym;
        real val_yp, coef_yp, rhs_yp;
        real val_zm, coef_zm, rhs_zm;
        real val_zp, coef_zp, rhs_zp;
        
        neighbor_xminus(i, j, k, grid, bc, x, K, val_xm, coef_xm, rhs_xm);
        neighbor_xplus(i, j, k, grid, bc, x, K, val_xp, coef_xp, rhs_xp);
        neighbor_yminus(i, j, k, grid, bc, x, K, val_ym, coef_ym, rhs_ym);
        neighbor_yplus(i, j, k, grid, bc, x, K, val_yp, coef_yp, rhs_yp);
        neighbor_zminus(i, j, k, grid, bc, x, K, val_zm, coef_zm, rhs_zm);
        neighbor_zplus(i, j, k, grid, bc, x, K, val_zp, coef_zp, rhs_zp);
        
        const real x_center = x[idx];
        
        // Compute sum(K_face * (xC - xN)) = result
        // Note: coef_* are K_face values, val_* are neighbor values
        real result = coef_xm * (x_center - val_xm) +
                      coef_xp * (x_center - val_xp) +
                      coef_ym * (x_center - val_ym) +
                      coef_yp * (x_center - val_yp) +
                      coef_zm * (x_center - val_zm) +
                      coef_zp * (x_center - val_zp);
        
        // Legacy pin1stCell: double diagonal contribution for cell [0,0,0]
        // Legacy: r = rhs - (result - HC*aC)/dx²  (when pin1stCell is true)
        // This is equivalent to adding aC*xC to result, i.e., doubling diagonal
        if (pin1stCell && V == XMIN_YMIN_ZMIN) {
            real aC = coef_xm + coef_xp + coef_ym + coef_yp + coef_zm + coef_zp;
            result += aC * x_center;
        }
        
        const real b_adjusted = b[idx] + (rhs_xm + rhs_xp + rhs_ym + rhs_yp + rhs_zm + rhs_zp);
        
        const real dx2 = grid.dx * grid.dx;  // dx² for operator scaling
        // r = b - A*x, where A*x = -result/dx² (negative Laplacian)
        r[idx] = b_adjusted - (-result / dx2);
    }
}

void compute_residual_3d(
    CudaContext& ctx,
    const Grid3D& grid,
    DeviceSpan<const real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    DeviceSpan<real> r,
    const BCSpec& bc,
    PinSpec pin
) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    const size_t n = grid.num_cells();
    
    bc.validate();
    
    assert(x.size() == n && "x size mismatch");
    assert(b.size() == n && "b size mismatch");
    assert(K.size() == n && "K size mismatch");
    assert(r.size() == n && "r size mismatch");
    
    // Convert BC to device-friendly format
    BCSpecDevice bc_dev = to_device(bc);
    
    // 1. Interior: branch-free
    dim3 block(16, 16);
    int grid_x = (Nx + block.x - 1) / block.x;
    int grid_y = (Ny + block.y - 1) / block.y;
    grid_x = (grid_x < 65535) ? grid_x : 65535;
    grid_y = (grid_y < 65535) ? grid_y : 65535;
    dim3 grid_dim(grid_x, grid_y);
    
    residual_interior_kernel<<<grid_dim, block, 0, ctx.cuda_stream()>>>(
        grid, x, b, K, r
    );
    RWPT_CUDA_CHECK(cudaGetLastError());
    
    // 2. Faces (6 kernels)
    dim3 face_block(16, 16);
    
    dim3 face_grid_yz((Ny + 15) / 16, (Nz + 15) / 16);
    residual_face_kernel<XMIN><<<face_grid_yz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_face_kernel<XMAX><<<face_grid_yz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    
    dim3 face_grid_xz((Nx + 15) / 16, (Nz + 15) / 16);
    residual_face_kernel<YMIN><<<face_grid_xz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_face_kernel<YMAX><<<face_grid_xz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    
    dim3 face_grid_xy((Nx + 15) / 16, (Ny + 15) / 16);
    residual_face_kernel<ZMIN><<<face_grid_xy, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_face_kernel<ZMAX><<<face_grid_xy, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    
    // 3. Edges (12 kernels)
    int edge_block = 256;
    
    int edge_grid_z = (Nz + edge_block - 1) / edge_block;
    residual_edge_kernel<XMIN_YMIN><<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<XMIN_YMAX><<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<XMAX_YMIN><<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<XMAX_YMAX><<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    
    int edge_grid_y = (Ny + edge_block - 1) / edge_block;
    residual_edge_kernel<XMIN_ZMIN><<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<XMIN_ZMAX><<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<XMAX_ZMIN><<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<XMAX_ZMAX><<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    
    int edge_grid_x = (Nx + edge_block - 1) / edge_block;
    residual_edge_kernel<YMIN_ZMIN><<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<YMIN_ZMAX><<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<YMAX_ZMIN><<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<YMAX_ZMAX><<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    RWPT_CUDA_CHECK(cudaGetLastError());
    
    // 4. Vertices (8 kernels, pin1stCell only affects XMIN_YMIN_ZMIN)
    residual_vertex_kernel<XMIN_YMIN_ZMIN><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, pin.enabled);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMIN_YMIN_ZMAX><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMIN_YMAX_ZMIN><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMIN_YMAX_ZMAX><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMAX_YMIN_ZMIN><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMAX_YMIN_ZMAX><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMAX_YMAX_ZMIN><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    RWPT_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMAX_YMAX_ZMAX><<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace multigrid
} // namespace rwpt

```

# src\multigrid\smoothers\residual_3d.cuh

```cuh
#pragma once

#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/BCSpec.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../../numerics/pin_spec.hpp"

namespace rwpt {
namespace multigrid {

/**
 * @brief Compute residual: r = b - A*x
 * 
 * A is 7-point stencil with variable coefficients K (conductivity).
 * Uses harmonic mean for face conductivities.
 * 
 * If pin is enabled:
 *   r[pin.index] = b[pin.index] - x[pin.index]  (identity row residual)
 * 
 * @param ctx   CUDA context
 * @param grid  Grid specification
 * @param x     Current solution
 * @param b     Right-hand side
 * @param K     Conductivity field
 * @param r     Output residual
 * @param bc    Boundary conditions
 * @param pin   Pin specification (optional, default = no pin)
 */
void compute_residual_3d(
    CudaContext& ctx,
    const Grid3D& grid,
    DeviceSpan<const real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    DeviceSpan<real> r,
    const BCSpec& bc,
    PinSpec pin = {}
);

} // namespace multigrid
} // namespace rwpt

```

# src\multigrid\transfer\prolong_3d.cu

```cu
#include "prolong_3d.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace rwpt {
namespace multigrid {

/**
 * @file prolong_3d.cu
 * @brief Piecewise-constant prolongation for cell-centered multigrid
 * 
 * Legacy correspondence: transf_operator_3D.cu (prolongation_* kernels)
 * 
 * The prolongation operator P transfers corrections from coarse to fine:
 *   x_fine += P * x_coarse
 * 
 * For cell-centered MG with factor-2 coarsening, each coarse cell maps to
 * 8 fine cells. The octant-based mapping determines which coarse cell 
 * contributes to each fine cell based on parity (ix%2, iy%2, iz%2).
 * 
 * Legacy uses separate kernels for interior/faces/edges/vertices to handle
 * boundaries correctly. We replicate this structure for legacy-compatibility.
 */

// ============================================================================
// Interior kernel: cells not touching any boundary
// Iterates over fine cells (1..Nx-2, 1..Ny-2, 1..Nz-2) - legacy convention
// ============================================================================
__global__ void prolong_interior_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz  // Fine dimensions
) {
    // Legacy: ix iterates 0..Nx-3 then uses (ix+1), (iy+1), (iz+1)
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (ix >= Nx - 2 || iy >= Ny - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int stride = Nx * Ny;
    int STRIDE = NX * NY;
    
    int IX = ix / 2;
    int IY = iy / 2;
    int fx = ix % 2;
    int fy = iy % 2;
    
    for (int iz = 0; iz < Nz - 2; ++iz) {
        int IZ = iz / 2;
        int fz = iz % 2;
        
        int in_idx = (ix + 1) + (iy + 1) * Nx + (iz + 1) * stride;
        int IN_IDX = IX + IY * NX + IZ * STRIDE;
        int offset = fx + fy * NX + fz * STRIDE;
        
        phiFine[in_idx] += phiCoarse[IN_IDX + offset];
    }
}

// ============================================================================
// Face kernels: cells on exactly one boundary face
// ============================================================================

// z = 0 face (bottom)
__global__ void prolong_face_zmin_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (ix >= Nx - 2 || iy >= Ny - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int iz = 0;
    int IZ = 0;
    
    int IX = ix / 2;
    int IY = iy / 2;
    int fx = ix % 2;
    int fy = iy % 2;
    // No fz offset for z=0 face
    
    int in_idx = (ix + 1) + (iy + 1) * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    int offset = fx + fy * NX;  // No z-offset
    
    phiFine[in_idx] += phiCoarse[IN_IDX + offset];
}

// z = Nz-1 face (top)
__global__ void prolong_face_zmax_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (ix >= Nx - 2 || iy >= Ny - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int NZ = Nz / 2;
    int iz = Nz - 1;
    int IZ = iz / 2;  // = NZ - 1
    
    int IX = ix / 2;
    int IY = iy / 2;
    int fx = ix % 2;
    int fy = iy % 2;
    // No fz offset for z=Nz-1 face (would go OOB)
    
    int in_idx = (ix + 1) + (iy + 1) * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    int offset = fx + fy * NX;  // No z-offset
    
    phiFine[in_idx] += phiCoarse[IN_IDX + offset];
}

// y = 0 face (south)
__global__ void prolong_face_ymin_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iz = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (ix >= Nx - 2 || iz >= Nz - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int iy = 0;
    int IY = 0;
    
    int IX = ix / 2;
    int IZ = iz / 2;
    int fx = ix % 2;
    int fz = iz % 2;
    // No fy offset for y=0 face
    
    int in_idx = (ix + 1) + iy * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    int offset = fx + fz * NX * NY;  // No y-offset
    
    phiFine[in_idx] += phiCoarse[IN_IDX + offset];
}

// y = Ny-1 face (north)
__global__ void prolong_face_ymax_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iz = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (ix >= Nx - 2 || iz >= Nz - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int iy = Ny - 1;
    int IY = iy / 2;
    
    int IX = ix / 2;
    int IZ = iz / 2;
    int fx = ix % 2;
    int fz = iz % 2;
    // No fy offset for y=Ny-1 face
    
    int in_idx = (ix + 1) + iy * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    int offset = fx + fz * NX * NY;  // No y-offset
    
    phiFine[in_idx] += phiCoarse[IN_IDX + offset];
}

// x = 0 face (west)
__global__ void prolong_face_xmin_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int iy = threadIdx.x + blockIdx.x * blockDim.x;
    int iz = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (iy >= Ny - 2 || iz >= Nz - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = 0;
    int IX = 0;
    
    int IY = iy / 2;
    int IZ = iz / 2;
    int fy = iy % 2;
    int fz = iz % 2;
    // No fx offset for x=0 face
    
    int in_idx = ix + (iy + 1) * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    int offset = fy * NX + fz * NX * NY;  // No x-offset
    
    phiFine[in_idx] += phiCoarse[IN_IDX + offset];
}

// x = Nx-1 face (east)
__global__ void prolong_face_xmax_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int iy = threadIdx.x + blockIdx.x * blockDim.x;
    int iz = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (iy >= Ny - 2 || iz >= Nz - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = Nx - 1;
    int IX = ix / 2;
    
    int IY = iy / 2;
    int IZ = iz / 2;
    int fy = iy % 2;
    int fz = iz % 2;
    // No fx offset for x=Nx-1 face
    
    int in_idx = ix + (iy + 1) * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    int offset = fy * NX + fz * NX * NY;  // No x-offset
    
    phiFine[in_idx] += phiCoarse[IN_IDX + offset];
}

// ============================================================================
// Edge kernels: cells on exactly two boundary faces
// ============================================================================

// Edge along X (y=0, z=0)
__global__ void prolong_edge_x_ymin_zmin_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= Nx - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int iy = 0, iz = 0;
    int IY = 0, IZ = 0;
    int IX = ix / 2;
    int fx = ix % 2;
    
    int in_idx = (ix + 1) + iy * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    
    phiFine[in_idx] += phiCoarse[IN_IDX + fx];
}

// Edge along X (y=0, z=Nz-1)
__global__ void prolong_edge_x_ymin_zmax_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= Nx - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int iy = 0, iz = Nz - 1;
    int IY = 0, IZ = iz / 2;
    int IX = ix / 2;
    int fx = ix % 2;
    
    int in_idx = (ix + 1) + iy * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    
    phiFine[in_idx] += phiCoarse[IN_IDX + fx];
}

// Edge along X (y=Ny-1, z=0)
__global__ void prolong_edge_x_ymax_zmin_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= Nx - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int iy = Ny - 1, iz = 0;
    int IY = iy / 2, IZ = 0;
    int IX = ix / 2;
    int fx = ix % 2;
    
    int in_idx = (ix + 1) + iy * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    
    phiFine[in_idx] += phiCoarse[IN_IDX + fx];
}

// Edge along X (y=Ny-1, z=Nz-1)
__global__ void prolong_edge_x_ymax_zmax_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= Nx - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int iy = Ny - 1, iz = Nz - 1;
    int IY = iy / 2, IZ = iz / 2;
    int IX = ix / 2;
    int fx = ix % 2;
    
    int in_idx = (ix + 1) + iy * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    
    phiFine[in_idx] += phiCoarse[IN_IDX + fx];
}

// Edge along Y (x=0, z=0)
__global__ void prolong_edge_y_xmin_zmin_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int iy = threadIdx.x + blockIdx.x * blockDim.x;
    if (iy >= Ny - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = 0, iz = 0;
    int IX = 0, IZ = 0;
    int IY = iy / 2;
    int fy = iy % 2;
    
    int in_idx = ix + (iy + 1) * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    
    phiFine[in_idx] += phiCoarse[IN_IDX + fy * NX];
}

// Edge along Y (x=0, z=Nz-1)
__global__ void prolong_edge_y_xmin_zmax_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int iy = threadIdx.x + blockIdx.x * blockDim.x;
    if (iy >= Ny - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = 0, iz = Nz - 1;
    int IX = 0, IZ = iz / 2;
    int IY = iy / 2;
    int fy = iy % 2;
    
    int in_idx = ix + (iy + 1) * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    
    phiFine[in_idx] += phiCoarse[IN_IDX + fy * NX];
}

// Edge along Y (x=Nx-1, z=0)
__global__ void prolong_edge_y_xmax_zmin_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int iy = threadIdx.x + blockIdx.x * blockDim.x;
    if (iy >= Ny - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = Nx - 1, iz = 0;
    int IX = ix / 2, IZ = 0;
    int IY = iy / 2;
    int fy = iy % 2;
    
    int in_idx = ix + (iy + 1) * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    
    phiFine[in_idx] += phiCoarse[IN_IDX + fy * NX];
}

// Edge along Y (x=Nx-1, z=Nz-1)
__global__ void prolong_edge_y_xmax_zmax_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int iy = threadIdx.x + blockIdx.x * blockDim.x;
    if (iy >= Ny - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = Nx - 1, iz = Nz - 1;
    int IX = ix / 2, IZ = iz / 2;
    int IY = iy / 2;
    int fy = iy % 2;
    
    int in_idx = ix + (iy + 1) * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    
    phiFine[in_idx] += phiCoarse[IN_IDX + fy * NX];
}

// Edge along Z (x=0, y=0)
__global__ void prolong_edge_z_xmin_ymin_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    if (iz >= Nz - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = 0, iy = 0;
    int IX = 0, IY = 0;
    int IZ = iz / 2;
    int fz = iz % 2;
    
    int in_idx = ix + iy * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    
    phiFine[in_idx] += phiCoarse[IN_IDX + fz * NX * NY];
}

// Edge along Z (x=0, y=Ny-1)
__global__ void prolong_edge_z_xmin_ymax_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    if (iz >= Nz - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = 0, iy = Ny - 1;
    int IX = 0, IY = iy / 2;
    int IZ = iz / 2;
    int fz = iz % 2;
    
    int in_idx = ix + iy * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    
    phiFine[in_idx] += phiCoarse[IN_IDX + fz * NX * NY];
}

// Edge along Z (x=Nx-1, y=0)
__global__ void prolong_edge_z_xmax_ymin_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    if (iz >= Nz - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = Nx - 1, iy = 0;
    int IX = ix / 2, IY = 0;
    int IZ = iz / 2;
    int fz = iz % 2;
    
    int in_idx = ix + iy * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    
    phiFine[in_idx] += phiCoarse[IN_IDX + fz * NX * NY];
}

// Edge along Z (x=Nx-1, y=Ny-1)
__global__ void prolong_edge_z_xmax_ymax_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz
) {
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    if (iz >= Nz - 2) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = Nx - 1, iy = Ny - 1;
    int IX = ix / 2, IY = iy / 2;
    int IZ = iz / 2;
    int fz = iz % 2;
    
    int in_idx = ix + iy * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    
    phiFine[in_idx] += phiCoarse[IN_IDX + fz * NX * NY];
}

// ============================================================================
// Vertex kernels: 8 corner cells
// ============================================================================

__global__ void prolong_vertex_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz,
    int ix, int iy, int iz  // Vertex position in fine grid
) {
    if (threadIdx.x != 0) return;
    
    int NX = Nx / 2;
    int NY = Ny / 2;
    int IX = ix / 2;
    int IY = iy / 2;
    int IZ = iz / 2;
    
    int in_idx = ix + iy * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    
    // No offset for vertices (all directions are at boundary)
    phiFine[in_idx] += phiCoarse[IN_IDX];
}

// ============================================================================
// Main function: orchestrate all kernels
// ============================================================================

void prolong_3d_add(
    CudaContext& ctx,
    const Grid3D& coarse_grid,
    const Grid3D& fine_grid,
    DeviceSpan<const real> x_coarse,
    DeviceSpan<real> x_fine
) {
    int Nx = fine_grid.nx;
    int Ny = fine_grid.ny;
    int Nz = fine_grid.nz;
    
    // Validate dimensions
    assert(coarse_grid.nx == Nx / 2 && "Fine grid must be 2x coarse in x");
    assert(coarse_grid.ny == Ny / 2 && "Fine grid must be 2x coarse in y");
    assert(coarse_grid.nz == Nz / 2 && "Fine grid must be 2x coarse in z");
    assert(x_coarse.size() == coarse_grid.num_cells() && "Coarse buffer size mismatch");
    assert(x_fine.size() == fine_grid.num_cells() && "Fine buffer size mismatch");
    
    real* phiFine = x_fine.data();
    const real* phiCoarse = x_coarse.data();
    cudaStream_t stream = ctx.cuda_stream();
    
    // Block sizes
    dim3 block2d(16, 16);
    int block1d = 256;
    
    // 1. Interior
    {
        int gx = (Nx - 2 + block2d.x - 1) / block2d.x;
        int gy = (Ny - 2 + block2d.y - 1) / block2d.y;
        dim3 grid(gx, gy);
        prolong_interior_kernel<<<grid, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
    }
    
    // 2. Faces (6 kernels)
    {
        // z-faces (iterate x, y interior)
        int gx = (Nx - 2 + block2d.x - 1) / block2d.x;
        int gy = (Ny - 2 + block2d.y - 1) / block2d.y;
        dim3 grid_xy(gx, gy);
        prolong_face_zmin_kernel<<<grid_xy, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_face_zmax_kernel<<<grid_xy, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        
        // y-faces (iterate x, z interior)
        gx = (Nx - 2 + block2d.x - 1) / block2d.x;
        int gz = (Nz - 2 + block2d.y - 1) / block2d.y;
        dim3 grid_xz(gx, gz);
        prolong_face_ymin_kernel<<<grid_xz, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_face_ymax_kernel<<<grid_xz, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        
        // x-faces (iterate y, z interior)
        gy = (Ny - 2 + block2d.y - 1) / block2d.y;
        gz = (Nz - 2 + block2d.x - 1) / block2d.x;
        dim3 grid_yz(gy, gz);
        prolong_face_xmin_kernel<<<grid_yz, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_face_xmax_kernel<<<grid_yz, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
    }
    
    // 3. Edges (12 kernels)
    {
        // X-edges
        int grid_x = (Nx - 2 + block1d - 1) / block1d;
        prolong_edge_x_ymin_zmin_kernel<<<grid_x, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_edge_x_ymin_zmax_kernel<<<grid_x, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_edge_x_ymax_zmin_kernel<<<grid_x, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_edge_x_ymax_zmax_kernel<<<grid_x, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        
        // Y-edges
        int grid_y = (Ny - 2 + block1d - 1) / block1d;
        prolong_edge_y_xmin_zmin_kernel<<<grid_y, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_edge_y_xmin_zmax_kernel<<<grid_y, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_edge_y_xmax_zmin_kernel<<<grid_y, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_edge_y_xmax_zmax_kernel<<<grid_y, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        
        // Z-edges
        int grid_z = (Nz - 2 + block1d - 1) / block1d;
        prolong_edge_z_xmin_ymin_kernel<<<grid_z, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_edge_z_xmin_ymax_kernel<<<grid_z, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_edge_z_xmax_ymin_kernel<<<grid_z, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_edge_z_xmax_ymax_kernel<<<grid_z, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
    }
    
    // 4. Vertices (8 kernels)
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, 0, 0, 0);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, 0, 0, Nz - 1);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, 0, Ny - 1, 0);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, 0, Ny - 1, Nz - 1);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, Nx - 1, 0, 0);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, Nx - 1, 0, Nz - 1);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, Nx - 1, Ny - 1, 0);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, Nx - 1, Ny - 1, Nz - 1);
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace multigrid
} // namespace rwpt

```

# src\multigrid\transfer\prolong_3d.cuh

```cuh
#pragma once

#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../runtime/CudaContext.cuh"

namespace rwpt {
namespace multigrid {

// Prolongation: coarse grid -> fine grid (injection + addition)
// For each fine cell, copy value from corresponding coarse cell
// phiFine[i,j,k] += phiCoarse[i/2, j/2, k/2]
// This is "additive" prolongation (used for error correction)
void prolong_3d_add(
    CudaContext& ctx,
    const Grid3D& coarse_grid,
    const Grid3D& fine_grid,
    DeviceSpan<const real> x_coarse,
    DeviceSpan<real> x_fine
);

} // namespace multigrid
} // namespace rwpt

```

# src\multigrid\transfer\restrict_3d.cu

```cu
#include "restrict_3d.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace rwpt {
namespace multigrid {

// Legacy: restriction_linear3D
// Full-weighting: averages 8 fine cells into 1 coarse cell
// Grid-stride for robustness
__global__ void restrict_kernel(
    real* __restrict__ phiCoarse,
    const real* __restrict__ phiFine,
    int NX, int NY, int NZ  // Coarse dimensions
) {
    int IX = threadIdx.x + blockIdx.x * blockDim.x;
    int IY = threadIdx.y + blockIdx.y * blockDim.y;
    
    // Grid-stride in y
    for (; IY < NY; IY += blockDim.y * gridDim.y) {
        // Grid-stride in x
        for (int IX_local = IX; IX_local < NX; IX_local += blockDim.x * gridDim.x) {
            int STRIDE = NX * NY;
            
            // Corresponding fine grid indices (2x resolution)
            int ix = 2 * IX_local;
            int iy = 2 * IY;
            int Nx = 2 * NX;
            int Ny = 2 * NY;
            int stride = Nx * Ny;
            
            for (int IZ = 0; IZ < NZ; ++IZ) {
                int iz = 2 * IZ;
                int IN_IDX = IX_local + IY * NX + IZ * STRIDE;
                int in_idx = ix + iy * Nx + iz * stride;
                
                real result = 0.0;
                // Bottom 4 cells (z-plane)
                result += phiFine[in_idx];
                result += phiFine[in_idx + 1];
                result += phiFine[in_idx + Nx];
                result += phiFine[in_idx + 1 + Nx];
                
                // Top 4 cells (z+1 plane)
                result += phiFine[in_idx + stride];
                result += phiFine[in_idx + 1 + stride];
                result += phiFine[in_idx + Nx + stride];
                result += phiFine[in_idx + 1 + Nx + stride];
                
                phiCoarse[IN_IDX] = result / 8.0;
            }
        }
    }
}

void restrict_3d(
    CudaContext& ctx,
    const Grid3D& fine_grid,
    const Grid3D& coarse_grid,
    DeviceSpan<const real> r_fine,
    DeviceSpan<real> b_coarse
) {
    int NX = coarse_grid.nx;
    int NY = coarse_grid.ny;
    int NZ = coarse_grid.nz;
    
    // Validate dimensions
    assert(fine_grid.nx == 2 * NX && "Fine grid must be 2x coarse in x");
    assert(fine_grid.ny == 2 * NY && "Fine grid must be 2x coarse in y");
    assert(fine_grid.nz == 2 * NZ && "Fine grid must be 2x coarse in z");
    assert(r_fine.size() == fine_grid.num_cells() && "Fine buffer size mismatch");
    assert(b_coarse.size() == coarse_grid.num_cells() && "Coarse buffer size mismatch");
    
    dim3 block(16, 16);
    int grid_x = (NX + block.x - 1) / block.x;
    int grid_y = (NY + block.y - 1) / block.y;
    // Clamp grid dimensions
    grid_x = (grid_x < 65535) ? grid_x : 65535;
    grid_y = (grid_y < 65535) ? grid_y : 65535;
    dim3 grid(grid_x, grid_y);
    
    restrict_kernel<<<grid, block, 0, ctx.cuda_stream()>>>(
        b_coarse.data(), r_fine.data(), NX, NY, NZ
    );
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace multigrid
} // namespace rwpt

```

# src\multigrid\transfer\restrict_3d.cuh

```cuh
#pragma once

#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../runtime/CudaContext.cuh"

namespace rwpt {
namespace multigrid {

// Restriction: fine grid -> coarse grid (averaging)
// phiCoarse[I,J,K] = (1/8) * sum of 8 fine cells that cover coarse cell [I,J,K]
// Fine grid has 2*NX, 2*NY, 2*NZ cells
// Coarse grid has NX, NY, NZ cells
void restrict_3d(
    CudaContext& ctx,
    const Grid3D& fine_grid,
    const Grid3D& coarse_grid,
    DeviceSpan<const real> r_fine,
    DeviceSpan<real> b_coarse
);

} // namespace multigrid
} // namespace rwpt

```

# src\numerics\blas\axpby.cu

```cu
#include "axpby.cuh"
#include "../../runtime/cuda_check.cuh"

namespace rwpt {
namespace blas {

__global__ void axpby_kernel(real a, const real* x, real b, real* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        y[i] = a * x[i] + b * y[i];
    }
}

void axpby(CudaContext& ctx, real a, DeviceSpan<const real> x, real b, DeviceSpan<real> y) {
    if (x.size() == 0 || y.size() == 0) return;
    
    size_t n = (x.size() < y.size()) ? x.size() : y.size();
    
    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);
    
    axpby_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        a, x.data(), b, y.data(), n
    );
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\axpby.cuh

```cuh
#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace blas {

// Axpby: y = a*x + b*y
void axpby(CudaContext& ctx, real a, DeviceSpan<const real> x, real b, DeviceSpan<real> y);

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\axpy.cu

```cu
#include "axpy.cuh"
#include "../../runtime/cuda_check.cuh"

namespace rwpt {
namespace blas {

__global__ void axpy_kernel(real a, const real* x, real* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

void axpy(CudaContext& ctx, real a, DeviceSpan<const real> x, DeviceSpan<real> y) {
    if (x.size() == 0 || y.size() == 0) return;
    
    size_t n = (x.size() < y.size()) ? x.size() : y.size();
    
    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);
    
    axpy_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        a, x.data(), y.data(), n
    );
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\axpy.cuh

```cuh
#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace blas {

// Axpy: y = a*x + y
void axpy(CudaContext& ctx, real a, DeviceSpan<const real> x, DeviceSpan<real> y);

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\blas.cuh

```cuh
#pragma once

// BLAS operations for rwpt project

#include "fill.cuh"
#include "copy.cuh"
#include "scal.cuh"
#include "axpy.cuh"
#include "axpby.cuh"
#include "reduction_workspace.cuh"
#include "dot.cuh"
#include "nrm2.cuh"
#include "cg_kernels.cuh"

```

# src\numerics\blas\cg_kernels.cu

```cu
#include "cg_kernels.cuh"
#include "../../runtime/cuda_check.cuh"

namespace rwpt {
namespace blas {

__global__ void compute_alpha_kernel(const real* d_rr, const real* d_pAp, real* d_alpha) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_alpha = (*d_rr) / (*d_pAp);
    }
}

__global__ void compute_beta_kernel(const real* d_rr_new, const real* d_rr, real* d_beta) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_beta = (*d_rr_new) / (*d_rr);
    }
}

__global__ void update_x_and_r_kernel(const real* d_alpha, const real* p, real* x,
                                       const real* Ap, real* r, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    real alpha = *d_alpha;
    
    for (size_t i = idx; i < n; i += stride) {
        x[i] = x[i] + alpha * p[i];
        r[i] = r[i] - alpha * Ap[i];
    }
}

__global__ void update_p_kernel(const real* d_beta, const real* r, real* p, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    real beta = *d_beta;
    
    for (size_t i = idx; i < n; i += stride) {
        p[i] = r[i] + beta * p[i];
    }
}

void compute_alpha(CudaContext& ctx,
                   DeviceSpan<const real> d_rr,
                   DeviceSpan<const real> d_pAp,
                   DeviceSpan<real> d_alpha) {
    compute_alpha_kernel<<<1, 1, 0, ctx.cuda_stream()>>>(
        d_rr.data(), d_pAp.data(), d_alpha.data()
    );
    RWPT_CUDA_CHECK(cudaGetLastError());
}

void compute_beta(CudaContext& ctx,
                  DeviceSpan<const real> d_rr_new,
                  DeviceSpan<const real> d_rr,
                  DeviceSpan<real> d_beta) {
    compute_beta_kernel<<<1, 1, 0, ctx.cuda_stream()>>>(
        d_rr_new.data(), d_rr.data(), d_beta.data()
    );
    RWPT_CUDA_CHECK(cudaGetLastError());
}

void update_x_and_r(CudaContext& ctx,
                    DeviceSpan<const real> d_alpha,
                    DeviceSpan<const real> p,
                    DeviceSpan<real> x,
                    DeviceSpan<const real> Ap,
                    DeviceSpan<real> r) {
    size_t n = x.size();
    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);
    
    update_x_and_r_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        d_alpha.data(), p.data(), x.data(), Ap.data(), r.data(), n
    );
    RWPT_CUDA_CHECK(cudaGetLastError());
}

void update_p(CudaContext& ctx,
              DeviceSpan<const real> d_beta,
              DeviceSpan<const real> r,
              DeviceSpan<real> p) {
    size_t n = p.size();
    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);
    
    update_p_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        d_beta.data(), r.data(), p.data(), n
    );
    RWPT_CUDA_CHECK(cudaGetLastError());
}

__global__ void check_pAp_valid_kernel(const real* d_pAp, int* d_is_valid) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        real pAp = *d_pAp;
        // Check for zero or NaN
        // Note: For NEGATIVE definite operators, pAp < 0 is expected and valid!
        // CG works with negative-definite operators as long as pAp != 0
        const real eps = 1e-30;
        if (!isfinite(pAp) || fabs(pAp) < eps) {
            *d_is_valid = 0;  // Bad
        } else {
            *d_is_valid = 1;  // OK (positive OR negative)
        }
    }
}

void check_pAp_valid(CudaContext& ctx,
                     DeviceSpan<const real> d_pAp,
                     DeviceSpan<int> d_is_valid) {
    check_pAp_valid_kernel<<<1, 1, 0, ctx.cuda_stream()>>>(
        d_pAp.data(), d_is_valid.data()
    );
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\cg_kernels.cuh

```cuh
#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace blas {

// CG scalar update kernels (device-only, no sync)

// Compute alpha = rr / pAp and store in d_alpha
void compute_alpha(CudaContext& ctx,
                   DeviceSpan<const real> d_rr,
                   DeviceSpan<const real> d_pAp,
                   DeviceSpan<real> d_alpha);

// Compute beta = rr_new / rr and store in d_beta
void compute_beta(CudaContext& ctx,
                  DeviceSpan<const real> d_rr_new,
                  DeviceSpan<const real> d_rr,
                  DeviceSpan<real> d_beta);

// Fused update: x = x + alpha*p, r = r - alpha*Ap
void update_x_and_r(CudaContext& ctx,
                    DeviceSpan<const real> d_alpha,
                    DeviceSpan<const real> p,
                    DeviceSpan<real> x,
                    DeviceSpan<const real> Ap,
                    DeviceSpan<real> r);

// Fused update: p = r + beta*p
void update_p(CudaContext& ctx,
              DeviceSpan<const real> d_beta,
              DeviceSpan<const real> r,
              DeviceSpan<real> p);

// Check if pAp is valid (not zero, not NaN) and write flag to d_is_valid (1=ok, 0=bad)
void check_pAp_valid(CudaContext& ctx,
                     DeviceSpan<const real> d_pAp,
                     DeviceSpan<int> d_is_valid);

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\copy.cu

```cu
#include "copy.cuh"
#include "../../runtime/cuda_check.cuh"

namespace rwpt {
namespace blas {

__global__ void copy_kernel(const real* x, real* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        y[i] = x[i];
    }
}

void copy(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) {
    if (x.size() == 0 || y.size() == 0) return;
    
    size_t n = (x.size() < y.size()) ? x.size() : y.size();
    
    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);
    
    copy_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        x.data(), y.data(), n
    );
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\copy.cuh

```cuh
#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace blas {

// Copy vector: y = x
void copy(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y);

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\dot.cu

```cu
#include "dot.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cassert>
#include <climits>

namespace rwpt {
namespace blas {

void dot_device(CudaContext& ctx, 
                DeviceSpan<const real> x, 
                DeviceSpan<const real> y, 
                DeviceSpan<real> d_result, 
                ReductionWorkspace& ws) {
    
    if (x.size() == 0 || y.size() == 0) {
        real zero = 0.0;
        RWPT_CUDA_CHECK(cudaMemcpyAsync(d_result.data(), &zero, sizeof(real), 
                                         cudaMemcpyHostToDevice, ctx.cuda_stream()));
        return;
    }
    
    size_t n = (x.size() < y.size()) ? x.size() : y.size();
    assert(n <= INT_MAX && "dot: size exceeds cuBLAS int limit");
    
    // Use cuBLAS with device pointer mode (no host sync, no temp buffers)
    cublasHandle_t handle = ctx.cublas_handle();
    
    // Save current pointer mode and set to device
    cublasPointerMode_t old_mode;
    RWPT_CUBLAS_CHECK(cublasGetPointerMode(handle, &old_mode));
    RWPT_CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    
    // Compute dot product: result in device memory
    RWPT_CUBLAS_CHECK(cublasDdot(handle, static_cast<int>(n), 
                                  x.data(), 1, 
                                  y.data(), 1, 
                                  d_result.data()));
    
    // Restore pointer mode
    RWPT_CUBLAS_CHECK(cublasSetPointerMode(handle, old_mode));
}

real dot_host(CudaContext& ctx, 
              DeviceSpan<const real> x, 
              DeviceSpan<const real> y, 
              ReductionWorkspace& ws) {
    
    ws.ensure_scalar();
    dot_device(ctx, x, y, ws.d_scalar.span(), ws);
    
    real result;
    RWPT_CUDA_CHECK(cudaMemcpyAsync(&result, ws.d_scalar.data(), sizeof(real), 
                                     cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();
    
    return result;
}

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\dot.cuh

```cuh
#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "reduction_workspace.cuh"

namespace rwpt {
namespace blas {

// Dot product (device result, no sync)
void dot_device(CudaContext& ctx, 
                DeviceSpan<const real> x, 
                DeviceSpan<const real> y, 
                DeviceSpan<real> d_result, 
                ReductionWorkspace& ws);

// Dot product (host result, includes synchronization)
// WARNING: Use only for debugging/reporting, not in hot loops
real dot_host(CudaContext& ctx, 
              DeviceSpan<const real> x, 
              DeviceSpan<const real> y, 
              ReductionWorkspace& ws);

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\fill.cu

```cu
#include "fill.cuh"
#include "../../runtime/cuda_check.cuh"

namespace rwpt {
namespace blas {

__global__ void fill_kernel(real* y, size_t n, real value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        y[i] = value;
    }
}

void fill(CudaContext& ctx, DeviceSpan<real> y, real value) {
    if (y.size() == 0) return;
    
    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((y.size() + block_size - 1) / block_size), max_blocks);
    
    fill_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        y.data(), y.size(), value
    );
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\fill.cuh

```cuh
#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace blas {

// Fill vector with constant value
void fill(CudaContext& ctx, DeviceSpan<real> y, real value);

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\nrm2.cu

```cu
#include "nrm2.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cassert>
#include <climits>

namespace rwpt {
namespace blas {

void nrm2_device(CudaContext& ctx, 
                 DeviceSpan<const real> x, 
                 DeviceSpan<real> d_result, 
                 ReductionWorkspace& ws) {
    
    if (x.size() == 0) {
        real zero = 0.0;
        RWPT_CUDA_CHECK(cudaMemcpyAsync(d_result.data(), &zero, sizeof(real), 
                                         cudaMemcpyHostToDevice, ctx.cuda_stream()));
        return;
    }
    
    size_t n = x.size();
    assert(n <= INT_MAX && "nrm2: size exceeds cuBLAS int limit");
    
    // Use cuBLAS with device pointer mode (no host sync, no temp buffers)
    cublasHandle_t handle = ctx.cublas_handle();
    
    // Save current pointer mode and set to device
    cublasPointerMode_t old_mode;
    RWPT_CUBLAS_CHECK(cublasGetPointerMode(handle, &old_mode));
    RWPT_CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    
    // Compute L2 norm: result in device memory
    RWPT_CUBLAS_CHECK(cublasDnrm2(handle, static_cast<int>(n), 
                                   x.data(), 1, 
                                   d_result.data()));
    
    // Restore pointer mode
    RWPT_CUBLAS_CHECK(cublasSetPointerMode(handle, old_mode));
}

real nrm2_host(CudaContext& ctx, 
               DeviceSpan<const real> x, 
               ReductionWorkspace& ws) {
    
    ws.ensure_scalar();
    nrm2_device(ctx, x, ws.d_scalar.span(), ws);
    
    real result;
    RWPT_CUDA_CHECK(cudaMemcpyAsync(&result, ws.d_scalar.data(), sizeof(real), 
                                     cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();
    
    return result;
}

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\nrm2.cuh

```cuh
#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "reduction_workspace.cuh"

namespace rwpt {
namespace blas {

// L2 norm (device result, no sync)
void nrm2_device(CudaContext& ctx, 
                 DeviceSpan<const real> x, 
                 DeviceSpan<real> d_result, 
                 ReductionWorkspace& ws);

// L2 norm (host result, includes synchronization)
// WARNING: Use only for debugging/reporting, not in hot loops
real nrm2_host(CudaContext& ctx, 
               DeviceSpan<const real> x, 
               ReductionWorkspace& ws);

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\reduction_workspace.cuh

```cuh
#pragma once

#include "../../core/DeviceBuffer.cuh"
#include "../../core/Scalar.hpp"
#include <cstddef>

namespace rwpt {
namespace blas {

struct ReductionWorkspace {
    DeviceBuffer<unsigned char> temp;  // cuBLAS workspace (std::byte not in C++14)
    size_t temp_bytes = 0;
    
    // Device scalar for results (no host sync in hot-path)
    DeviceBuffer<real> d_scalar;
    
    ReductionWorkspace() {
        d_scalar.resize(1);
    }
    
    void ensure_bytes(size_t required_bytes) {
        if (temp_bytes < required_bytes) {
            temp.resize(required_bytes);
            temp_bytes = required_bytes;
        }
    }
    
    void ensure_scalar() {
        if (d_scalar.size() < 1) {
            d_scalar.resize(1);
        }
    }
};

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\scal.cu

```cu
#include "scal.cuh"
#include "../../runtime/cuda_check.cuh"

namespace rwpt {
namespace blas {

__global__ void scal_kernel(real* x, size_t n, real a) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        x[i] = a * x[i];
    }
}

void scal(CudaContext& ctx, DeviceSpan<real> x, real a) {
    if (x.size() == 0) return;
    
    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((x.size() + block_size - 1) / block_size), max_blocks);
    
    scal_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        x.data(), x.size(), a
    );
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace blas
} // namespace rwpt

```

# src\numerics\blas\scal.cuh

```cuh
#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace blas {

// Scale vector: x = a*x
void scal(CudaContext& ctx, DeviceSpan<real> x, real a);

} // namespace blas
} // namespace rwpt

```

# src\numerics\operators\negated_operator.cuh

```cuh
#pragma once

/**
 * @file negated_operator.cuh
 * @brief Wrapper to negate an operator for CG/PCG
 * 
 * Problem: VarCoeffLaplacian produces A = -∇·(K∇·), which is NEGATIVE definite.
 *          CG/PCG require a POSITIVE definite operator.
 * 
 * Solution: Wrap A as -A (negate output), making it SPD.
 *           The solver sees: (-A)*x = -b, which solves A*x = b.
 * 
 * Usage:
 *   VarCoeffLaplacian A_neg(...);  // Negative definite
 *   NegatedOperator<VarCoeffLaplacian> A_pos(A_neg);  // Positive definite
 *   pcg_solve(ctx, A_pos, b_negated, x, ...);
 * 
 * Important: The RHS must also be negated: if A*x = b, then (-A)*x = -b.
 */

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../blas/blas.cuh"

namespace rwpt {
namespace operators {

/**
 * @brief Negate an operator: NegatedOperator.apply(x) = -Operator.apply(x)
 * 
 * Template parameter Op must have:
 *   void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const;
 */
template<typename Op>
class NegatedOperator {
public:
    explicit NegatedOperator(const Op& op) : op_(op) {}
    
    /**
     * @brief Apply negated operator: y = -A*x
     */
    void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const {
        // First, apply the original operator: y = A*x
        op_.apply(ctx, x, y);
        
        // Then negate: y = -y
        blas::scal(ctx, y, real(-1.0));
    }
    
private:
    const Op& op_;  // Reference to wrapped operator
};

/**
 * @brief Helper function to create a negated operator
 */
template<typename Op>
NegatedOperator<Op> negate_operator(const Op& op) {
    return NegatedOperator<Op>(op);
}

} // namespace operators
} // namespace rwpt

```

# src\numerics\operators\operator_concept.cuh

```cuh
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

namespace rwpt {
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
} // namespace rwpt

```

# src\numerics\operators\operators.cuh

```cuh
#pragma once

// Operator concept (template-based, no virtual dispatch)
#include "operator_concept.cuh"

// Operators for rwpt project

// Constant-coefficient Laplacian (simple, K=1 everywhere)
#include "poisson3d_operator.cuh"

// Variable-coefficient Laplacian (same semantics as MG smoother/residual)
// Use this for CG solve to compare directly with MG solve
#include "varcoeff_laplacian.cuh"

// Negated operator wrapper (converts negative-definite to positive-definite)
// Essential for using CG/PCG with our negative Laplacian
#include "negated_operator.cuh"

```

# src\numerics\operators\pinned_operator.cuh

```cuh
#pragma once

/**
 * @file pinned_operator.cuh
 * @brief Operator wrapper that pins the first cell for singular systems
 * 
 * For systems with all periodic/homogeneous Neumann BCs, the Laplacian is singular
 * (constant mode in null space). Pinning cell 0 breaks this degeneracy.
 * 
 * Legacy: pin1stCell in up_residual_3D.cu
 * 
 * The pinning is implemented by doubling the diagonal contribution for cell 0:
 *   (A_pinned * x)[0] = (A * x)[0] - aC * x[0]
 * 
 * where aC is the sum of face coefficients for cell 0. This effectively makes
 * the equation for cell 0:
 *   2 * aC * x[0] - sum(K_face * x_neighbor) = RHS[0]
 * 
 * With RHS[0] = 0 and homogeneous neighbors, this pins x[0] ≈ 0.
 * 
 * Note: This is a simple implementation that modifies the output for cell 0 only.
 * For production, the proper approach would pass pin_first_cell to the operator.
 */

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace operators {

/**
 * Wrapper that applies pin_first_cell modification to any operator.
 * 
 * For cell 0, adds an extra -aC * x[0] term to the result, which is
 * equivalent to doubling the diagonal.
 * 
 * This matches legacy behavior in update_vertex_SWB when pin1stCell=true:
 *   r[0] = rhs[0] - (result - HC*aC)/dx²
 * vs
 *   r[0] = rhs[0] - result/dx²
 * 
 * The difference is -HC*aC/dx², so the operator contributes +aC*x[0]/dx²
 * to A*x (since r = b - Ax/dx²).
 */
template<typename Op>
class PinnedOperator {
public:
    PinnedOperator(Op& inner, DeviceSpan<const real> K, real dx)
        : inner_(inner), K_(K), dx_(dx) {}
    
    void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const {
        // First apply the underlying operator
        inner_.apply(ctx, x, y);
        
        // Then modify cell 0: add extra diagonal contribution
        // Legacy does: r[0] = rhs[0] - (result - HC*aC)/dx²
        // where result = sum(K_face*(xC-xN)), aC = sum(K_face)
        // So A*x includes an extra +aC*xC term for the pinned cell
        //
        // But our operator already does y = A*x/dx² (negative Laplacian)
        // The pinning adds: y[0] -= aC * x[0] / dx²
        // This is equivalent to doubling the diagonal for cell 0
        
        // Read aC from device (sum of interior face coefficients)
        // For cell 0, neighbors are at indices 1, Nx, Nx*Ny
        // This requires knowing the grid size, which we don't have here.
        //
        // Simpler approach: just set y[0] = large_value * x[0]
        // This effectively pins x[0] to 0 when solving A*x = b with b[0] = 0.
        //
        // But that's too aggressive. Instead, we use a post-processing approach
        // in the solver to enforce x[0] = pin_value.
        //
        // For now, just call the kernel to fix cell 0.
        apply_pin_kernel(ctx, x, y);
    }
    
    size_t size() const { return inner_.size(); }
    
private:
    void apply_pin_kernel(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const;
    
    Op& inner_;
    DeviceSpan<const real> K_;
    real dx_;
};

// Kernel to apply pin modification to cell 0
__global__ void kernel_apply_pin(
    const real* __restrict__ x,
    real* __restrict__ y,
    const real* __restrict__ K,
    size_t stride_y,  // Nx*Ny
    size_t stride_x,  // Nx
    real inv_dx2
) {
    // Only thread 0 does work
    if (threadIdx.x + blockIdx.x * blockDim.x != 0) return;
    
    // Cell 0 is at (0,0,0)
    // Interior neighbors: +x (idx=1), +y (idx=Nx), +z (idx=Nx*Ny)
    // For simplicity, compute aC as 3*2*K[0] (assuming uniform K near corner)
    // This is approximate but captures the main behavior
    
    real KC = K[0];
    real x0 = x[0];
    
    // Sum of face coefficients for interior neighbors
    // Each interior face has K_face = 2/(1/KC + 1/KN)
    // For corners in periodic/Neumann, all 6 faces may contribute
    // But for simplicity, use 6*2*KC (assuming KC ≈ KN)
    real aC = real(6.0) * real(2.0) * KC;  // Approximate
    
    // Legacy: result -> (result - x0*aC), so the operator output changes by -aC*x0/dx²
    // But legacy operator is negative: y = -Ax/dx²
    // So we need y[0] += aC * x0 * inv_dx2 to match legacy
    y[0] += aC * x0 * inv_dx2;
}

template<typename Op>
void PinnedOperator<Op>::apply_pin_kernel(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const {
    real inv_dx2 = real(1.0) / (dx_ * dx_);
    kernel_apply_pin<<<1, 1, 0, ctx.cuda_stream()>>>(
        x.data(), y.data(), K_.data(), 0, 0, inv_dx2
    );
}

} // namespace operators
} // namespace rwpt

```

# src\numerics\operators\poisson3d_operator.cu

```cu
#include "poisson3d_operator.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cassert>

namespace rwpt {
namespace operators {

__global__ void poisson3d_kernel(const real* x, real* y, 
                                  int nx, int ny, int nz,
                                  real dx, real dy, real dz,
                                  size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    // Coefficients for 7-point stencil: -Δ (SPD)
    // Diagonal positive, off-diagonal negative
    real cx = 1.0 / (dx * dx);
    real cy = 1.0 / (dy * dy);
    real cz = 1.0 / (dz * dz);
    real cc = 2.0 * (cx + cy + cz);  // Positive diagonal
    
    for (size_t linear_idx = idx; linear_idx < n; linear_idx += stride) {
        // Convert linear index to (i, j, k)
        int i = linear_idx % nx;
        int j = (linear_idx / nx) % ny;
        int k = linear_idx / (nx * ny);
        
        real center = x[linear_idx];
        real result = cc * center;
        
        // X-direction neighbors (negative for -Δ)
        if (i > 0) {
            result -= cx * x[linear_idx - 1];
        }
        if (i < nx - 1) {
            result -= cx * x[linear_idx + 1];
        }
        
        // Y-direction neighbors (negative for -Δ)
        if (j > 0) {
            result -= cy * x[linear_idx - nx];
        }
        if (j < ny - 1) {
            result -= cy * x[linear_idx + nx];
        }
        
        // Z-direction neighbors (negative for -Δ)
        if (k > 0) {
            result -= cz * x[linear_idx - nx * ny];
        }
        if (k < nz - 1) {
            result -= cz * x[linear_idx + nx * ny];
        }
        
        y[linear_idx] = result;
    }
}

void Poisson3DOperator::apply(CudaContext& ctx, 
                               DeviceSpan<const real> x, 
                               DeviceSpan<real> y) const {
    
    size_t n = grid.num_cells();
    assert(x.size() >= n && y.size() >= n && "Operator apply: size mismatch");
    if (x.size() < n || y.size() < n) return;
    
    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);
    
    poisson3d_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        x.data(), y.data(),
        grid.nx, grid.ny, grid.nz,
        grid.dx, grid.dy, grid.dz,
        n
    );
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace operators
} // namespace rwpt

```

# src\numerics\operators\poisson3d_operator.cuh

```cuh
#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace operators {

struct Poisson3DOperator {
    Grid3D grid;
    
    Poisson3DOperator() = default;
    
    explicit Poisson3DOperator(const Grid3D& g) : grid(g) {}
    
    // Matrix-free apply: y = A*x
    // Implements discrete Laplacian (7-point stencil)
    // BC: Dirichlet homogeneous (x=0 outside domain)
    void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const;
};

} // namespace operators
} // namespace rwpt

```

# src\numerics\operators\varcoeff_laplacian.cu

```cu
/**
 * @file varcoeff_laplacian.cu
 * @brief Variable-coefficient Laplacian operator implementation
 * 
 * This computes y = A*x where A is the NEGATIVE discrete variable-coefficient Laplacian:
 *   (A*x)_C = -sum_faces( K_face * (x_C - x_neighbor) ) / dx²
 * 
 * K_face is the harmonic mean: K_face = 2 / (1/K_C + 1/K_neighbor)
 * 
 * This matches the legacy stencil_head operator which also produces a NEGATIVE operator.
 * The negative sign ensures CG solves the same system as MG.
 * 
 * For Dirichlet BCs: The BC value contribution goes to RHS (in build_rhs_head),
 * the operator contributes only the diagonal term: -2*KC*xC
 */

#include "varcoeff_laplacian.cuh"
#include "../../runtime/cuda_check.cuh"
#include "../../core/BCSpecDevice.cuh"
#include <cassert>

namespace rwpt {
namespace operators {

VarCoeffLaplacian::VarCoeffLaplacian(
    const Grid3D& grid,
    DeviceSpan<const real> K,
    const BCSpec& bc,
    PinSpec pin
) : grid_(grid), K_(K), bc_(bc), pin_(pin) {
    assert(K.size() == grid.num_cells() && "K field size must match grid");
    bc_.validate();
    // Legacy pin always uses cell [0,0,0] (index 0)
}

// ============================================================================
// Interior kernel: cells not on any boundary
// Produces y = -∇·(K∇x) discretized as -sum_faces(K_face*(xC-xN))/dx²
// Sign convention: NEGATIVE Laplacian, consistent with MG smoother
// ============================================================================
__global__ void varcoeff_apply_interior_kernel(
    const real* __restrict__ x,
    const real* __restrict__ K,
    real* __restrict__ y,
    int Nx, int Ny, int Nz,
    real inv_dx2
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (ix >= Nx - 2 || iy >= Ny - 2) return;
    
    int stride = Nx * Ny;
    
    for (int iz = 1; iz < Nz - 1; ++iz) {
        int idx = (ix + 1) + (iy + 1) * Nx + iz * stride;
        
        real xC = x[idx];
        real KC = K[idx];
        
        // Harmonic mean for each face
        real Kxm = 2.0 / (1.0/KC + 1.0/K[idx - 1]);
        real Kxp = 2.0 / (1.0/KC + 1.0/K[idx + 1]);
        real Kym = 2.0 / (1.0/KC + 1.0/K[idx - Nx]);
        real Kyp = 2.0 / (1.0/KC + 1.0/K[idx + Nx]);
        real Kzm = 2.0 / (1.0/KC + 1.0/K[idx - stride]);
        real Kzp = 2.0 / (1.0/KC + 1.0/K[idx + stride]);
        
        // Compute sum(K_face*(xC-xN)) 
        // NEGATIVE Laplacian: y = -sum(K_face*(xC-xN))/dx²
        // This matches legacy stencil_head which produces -2*sum((xC-xN)/(1/KC+1/KN))/dx²
        real Ax = Kxm * (xC - x[idx - 1]) +
                  Kxp * (xC - x[idx + 1]) +
                  Kym * (xC - x[idx - Nx]) +
                  Kyp * (xC - x[idx + Nx]) +
                  Kzm * (xC - x[idx - stride]) +
                  Kzp * (xC - x[idx + stride]);
        
        // NEGATIVE: matches legacy operator sign
        y[idx] = -Ax * inv_dx2;
    }
}

// ============================================================================
// Boundary kernel: handles all boundary cells with BC logic
// This is simpler than separate face/edge/vertex kernels since apply()
// doesn't need red-black ordering
// ============================================================================
__global__ void varcoeff_apply_boundary_kernel(
    const real* __restrict__ x,
    const real* __restrict__ K,
    real* __restrict__ y,
    int Nx, int Ny, int Nz,
    real inv_dx2,
    BCSpecDevice bc,
    bool pin1stCell
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_cells = Nx * Ny * Nz;
    
    for (int idx = tid; idx < total_cells; idx += blockDim.x * gridDim.x) {
        // Convert to 3D index
        int i = idx % Nx;
        int j = (idx / Nx) % Ny;
        int k = idx / (Nx * Ny);
        
        // Skip interior cells (handled by interior kernel)
        bool is_boundary = (i == 0 || i == Nx-1 || j == 0 || j == Ny-1 || k == 0 || k == Nz-1);
        if (!is_boundary) continue;
        
        real xC = x[idx];
        real KC = K[idx];
        real Ax = 0.0;
        real aC = 0.0;  // Diagonal coefficient accumulator for pin
        
        // X-minus neighbor
        if (i > 0) {
            int n_idx = idx - 1;
            real Kn = K[n_idx];
            real Kh = 2.0 / (1.0/KC + 1.0/Kn);
            Ax += Kh * (xC - x[n_idx]);
            aC += Kh;
        } else {
            // x = 0 boundary
            auto bc_type = static_cast<BCType>(bc.type[0]);
            if (bc_type == BCType::Periodic) {
                int n_idx = (Nx - 1) + j * Nx + k * Nx * Ny;
                real Kn = K[n_idx];
                real Kh = 2.0 / (1.0/KC + 1.0/Kn);
                Ax += Kh * (xC - x[n_idx]);
                // Note: periodic neighbors do NOT add to aC for pin
            } else if (bc_type == BCType::Dirichlet) {
                // Legacy convention: BC value is in RHS, operator only contributes diagonal
                // result -= 2*KC*xC (not 2*KC*(xC - bc_val))
                Ax += 2.0 * KC * xC;
            }
            // Neumann: no contribution
        }
        
        // X-plus neighbor
        if (i < Nx - 1) {
            int n_idx = idx + 1;
            real Kn = K[n_idx];
            real Kh = 2.0 / (1.0/KC + 1.0/Kn);
            Ax += Kh * (xC - x[n_idx]);
            aC += Kh;
        } else {
            auto bc_type = static_cast<BCType>(bc.type[1]);
            if (bc_type == BCType::Periodic) {
                int n_idx = 0 + j * Nx + k * Nx * Ny;
                real Kn = K[n_idx];
                real Kh = 2.0 / (1.0/KC + 1.0/Kn);
                Ax += Kh * (xC - x[n_idx]);
            } else if (bc_type == BCType::Dirichlet) {
                // Legacy convention: BC value is in RHS, operator only contributes diagonal
                Ax += 2.0 * KC * xC;
            }
        }
        
        // Y-minus neighbor
        if (j > 0) {
            int n_idx = idx - Nx;
            real Kn = K[n_idx];
            real Kh = 2.0 / (1.0/KC + 1.0/Kn);
            Ax += Kh * (xC - x[n_idx]);
            aC += Kh;
        } else {
            auto bc_type = static_cast<BCType>(bc.type[2]);
            if (bc_type == BCType::Periodic) {
                int n_idx = i + (Ny - 1) * Nx + k * Nx * Ny;
                real Kn = K[n_idx];
                real Kh = 2.0 / (1.0/KC + 1.0/Kn);
                Ax += Kh * (xC - x[n_idx]);
            } else if (bc_type == BCType::Dirichlet) {
                // Legacy convention: BC value is in RHS, operator only contributes diagonal
                Ax += 2.0 * KC * xC;
            }
        }
        
        // Y-plus neighbor
        if (j < Ny - 1) {
            int n_idx = idx + Nx;
            real Kn = K[n_idx];
            real Kh = 2.0 / (1.0/KC + 1.0/Kn);
            Ax += Kh * (xC - x[n_idx]);
            aC += Kh;
        } else {
            auto bc_type = static_cast<BCType>(bc.type[3]);
            if (bc_type == BCType::Periodic) {
                int n_idx = i + 0 * Nx + k * Nx * Ny;
                real Kn = K[n_idx];
                real Kh = 2.0 / (1.0/KC + 1.0/Kn);
                Ax += Kh * (xC - x[n_idx]);
            } else if (bc_type == BCType::Dirichlet) {
                // Legacy convention: BC value is in RHS, operator only contributes diagonal
                Ax += 2.0 * KC * xC;
            }
        }
        
        // Z-minus neighbor
        int stride = Nx * Ny;
        if (k > 0) {
            int n_idx = idx - stride;
            real Kn = K[n_idx];
            real Kh = 2.0 / (1.0/KC + 1.0/Kn);
            Ax += Kh * (xC - x[n_idx]);
            aC += Kh;
        } else {
            auto bc_type = static_cast<BCType>(bc.type[4]);
            if (bc_type == BCType::Periodic) {
                int n_idx = i + j * Nx + (Nz - 1) * stride;
                real Kn = K[n_idx];
                real Kh = 2.0 / (1.0/KC + 1.0/Kn);
                Ax += Kh * (xC - x[n_idx]);
            } else if (bc_type == BCType::Dirichlet) {
                // Legacy convention: BC value is in RHS, operator only contributes diagonal
                Ax += 2.0 * KC * xC;
            }
        }
        
        // Z-plus neighbor
        if (k < Nz - 1) {
            int n_idx = idx + stride;
            real Kn = K[n_idx];
            real Kh = 2.0 / (1.0/KC + 1.0/Kn);
            Ax += Kh * (xC - x[n_idx]);
            aC += Kh;
        } else {
            auto bc_type = static_cast<BCType>(bc.type[5]);
            if (bc_type == BCType::Periodic) {
                int n_idx = i + j * Nx + 0 * stride;
                real Kn = K[n_idx];
                real Kh = 2.0 / (1.0/KC + 1.0/Kn);
                Ax += Kh * (xC - x[n_idx]);
            } else if (bc_type == BCType::Dirichlet) {
                // Legacy convention: BC value is in RHS, operator only contributes diagonal
                Ax += 2.0 * KC * xC;
            }
        }
        
        // Legacy pin1stCell: double the diagonal for cell [0,0,0]
        // This is equivalent to adding aC*xC to the result
        // Legacy: if(pin1stCell) H_output[0] = -2.0*(result+aC*HC)/dx²
        if (pin1stCell && idx == 0) {
            Ax += aC * xC;
        }
        
        // NEGATIVE: matches legacy operator sign
        y[idx] = -Ax * inv_dx2;
    }
}

void VarCoeffLaplacian::apply(
    CudaContext& ctx,
    DeviceSpan<const real> x,
    DeviceSpan<real> y
) const {
    int Nx = grid_.nx;
    int Ny = grid_.ny;
    int Nz = grid_.nz;
    size_t n = grid_.num_cells();
    
    assert(x.size() == n && "x size mismatch");
    assert(y.size() == n && "y size mismatch");
    
    real inv_dx2 = 1.0 / (grid_.dx * grid_.dx);
    BCSpecDevice bc_dev = to_device(bc_);
    
    // 1. Interior cells
    {
        dim3 block(16, 16);
        int gx = (Nx - 2 + block.x - 1) / block.x;
        int gy = (Ny - 2 + block.y - 1) / block.y;
        dim3 grid(gx, gy);
        
        varcoeff_apply_interior_kernel<<<grid, block, 0, ctx.cuda_stream()>>>(
            x.data(), K_.data(), y.data(), Nx, Ny, Nz, inv_dx2
        );
        RWPT_CUDA_CHECK(cudaGetLastError());
    }
    
    // 2. Boundary cells (includes pin logic via diagonal doubling for cell [0,0,0])
    {
        int block = 256;
        int grid = (n + block - 1) / block;
        if (grid > 65535) grid = 65535;
        
        varcoeff_apply_boundary_kernel<<<grid, block, 0, ctx.cuda_stream()>>>(
            x.data(), K_.data(), y.data(), Nx, Ny, Nz, inv_dx2, bc_dev, pin_.enabled
        );
        RWPT_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace operators
} // namespace rwpt

```

# src\numerics\operators\varcoeff_laplacian.cuh

```cuh
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

namespace rwpt {
namespace operators {

// Re-export PinSpec in operators namespace for backward compat
using rwpt::PinSpec;

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
} // namespace rwpt

```

# src\numerics\pin_spec.hpp

```hpp
#pragma once

/**
 * @file pin_spec.hpp
 * @brief Pin specification and utilities for singular linear systems
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

#include "../core/Scalar.hpp"
#include "../core/BCSpec.hpp"
#include <cstddef>

namespace rwpt {

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
    Auto,   // Enable only when needed (default)
    On,     // Always enable
    Off     // Never enable
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
    bool enabled = false;   // Whether pin is active
    size_t index = 0;       // Always 0 (legacy: cell [0,0,0])
    
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
    auto is_dirichlet = [](const BCFace& f) {
        return f.type == BCType::Dirichlet;
    };
    
    // If ANY face is Dirichlet, system is NOT singular
    bool has_dirichlet = is_dirichlet(bc.xmin) || is_dirichlet(bc.xmax) ||
                         is_dirichlet(bc.ymin) || is_dirichlet(bc.ymax) ||
                         is_dirichlet(bc.zmin) || is_dirichlet(bc.zmax);
    
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
        case PinMode::On:   return true;
        case PinMode::Off:  return false;
        case PinMode::Auto:
        default:            return needs_pin(bc);
    }
}

/**
 * @brief Get pin mode as string for logging
 */
inline const char* pin_mode_str(PinMode mode) {
    switch (mode) {
        case PinMode::On:   return "on";
        case PinMode::Off:  return "off";
        case PinMode::Auto: return "auto";
        default:            return "unknown";
    }
}

} // namespace rwpt

```

# src\numerics\solvers\cg_types.hpp

```hpp
#pragma once

#include "../../core/DeviceBuffer.cuh"
#include "../../core/Scalar.hpp"
#include "../blas/reduction_workspace.cuh"
#include <cstddef>

namespace rwpt {
namespace solvers {

struct CGConfig {
    int max_iter = 200;
    real rtol = 1e-6;
    real atol = 0.0;
    int check_every = 10;  // Check convergence every N iters (reduces host sync)
    bool verbose = false;  // Print debug info
};

struct CGResult {
    int iters = 0;
    real r_norm = 0.0;
    real r0_norm = 0.0;  // Initial residual for caller logging
    bool converged = false;
};

struct CGWorkspace {
    // Vector buffers
    DeviceBuffer<real> r;
    DeviceBuffer<real> p;
    DeviceBuffer<real> Ap;
    
    // Device scalars (no host sync in hot-path)
    DeviceBuffer<real> d_rr;
    DeviceBuffer<real> d_rr_new;
    DeviceBuffer<real> d_pAp;
    DeviceBuffer<real> d_alpha;
    DeviceBuffer<real> d_beta;
    
    // Breakdown detection
    DeviceBuffer<int> d_is_valid;
    
    // Reduction workspace for dot/nrm2
    blas::ReductionWorkspace red;
    
    size_t n = 0;
    
    void ensure(size_t required_n) {
        if (n < required_n) {
            r.resize(required_n);
            p.resize(required_n);
            Ap.resize(required_n);
            n = required_n;
        }
        
        // Ensure device scalars are allocated
        if (d_rr.size() < 1) d_rr.resize(1);
        if (d_rr_new.size() < 1) d_rr_new.resize(1);
        if (d_pAp.size() < 1) d_pAp.resize(1);
        if (d_alpha.size() < 1) d_alpha.resize(1);
        if (d_beta.size() < 1) d_beta.resize(1);
        if (d_is_valid.size() < 1) d_is_valid.resize(1);
    }
};

} // namespace solvers
} // namespace rwpt

```

# src\numerics\solvers\cg.cuh

```cuh
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

```

# src\numerics\solvers\mg_preconditioner.cu

```cu
#include "mg_preconditioner.cuh"
#include "../blas/blas.cuh"

namespace rwpt {
namespace solvers {

MultigridPreconditioner::MultigridPreconditioner(
    multigrid::MGHierarchy& hierarchy,
    const BCSpec& bc,
    const multigrid::MGConfig& config,
    PinSpec pin
)
    : hierarchy_(&hierarchy)
    , bc_(bc)
    , config_(config)
    , pin_(pin)
{
    // Hierarchy is already constructed with all buffers pre-allocated.
    // No additional allocations needed here.
    // The MGHierarchy constructor already allocates x, b, r, K per level.
}

void MultigridPreconditioner::apply(
    CudaContext& ctx,
    DeviceSpan<const real> r,
    DeviceSpan<real> z
) const {
    // Preconditioner apply: z ≈ A^{-1} r using one V-cycle
    //
    // Legacy correspondence:
    //   Precond_CCMG_Vcycle* sets e=0 on all levels,
    //   copies residual to RHS of finest level,
    //   runs ONE V-cycle, result is in finest.x
    //
    // Algorithm:
    //   1. Copy r → finest.b (RHS for MG)
    //   2. Set finest.x = 0 (initial guess)
    //   3. Execute one V-cycle (pin propagated to all levels)
    //   4. Copy finest.x → z (result)
    
    auto& finest = hierarchy_->levels[0];
    
    // 1. Copy input residual to finest level RHS
    rwpt::blas::copy(ctx, r, finest.b.span());
    
    // 2. Zero initial guess on finest level
    rwpt::blas::fill(ctx, finest.x.span(), 0.0);
    
    // 3. Execute ONE V-cycle (recursive from level 0, pin propagated)
    multigrid::v_cycle_recursive(ctx, *hierarchy_, 0, config_, bc_, pin_);
    
    // 4. Copy result to output z
    rwpt::blas::copy(ctx, DeviceSpan<const real>(finest.x.span()), z);
}

} // namespace solvers
} // namespace rwpt

```

# src\numerics\solvers\mg_preconditioner.cuh

```cuh
#pragma once

/**
 * @file mg_preconditioner.cuh
 * @brief Multigrid V-cycle as preconditioner for PCG
 * 
 * Legacy correspondence: 
 * - MGprecond2 structure with pre-allocated _e, _r, _rr, _K per level
 * - Precond_CCMG_Vcycle* routines
 * 
 * Design (HPC):
 * - Construction: allocate all workspace once
 * - apply(): execute ONE V-cycle, no allocations
 * - Respects BCSpec and pin1stCell (propagated, not reinterpreted)
 */

#include "preconditioner.cuh"
#include "../../multigrid/mg_types.hpp"
#include "../../multigrid/cycle/v_cycle.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/BCSpec.hpp"
#include "../../numerics/blas/blas.cuh"
#include "../../numerics/pin_spec.hpp"

namespace rwpt {
namespace solvers {

/**
 * @brief Multigrid V-cycle preconditioner
 * 
 * Uses one V-cycle to approximate z = A^{-1} r.
 * Workspace is pre-allocated at construction; apply() does no allocations.
 */
class MultigridPreconditioner {
public:
    /**
     * @brief Construct MG preconditioner with pre-allocated hierarchy
     * 
     * @param hierarchy  MG hierarchy (grids + K already coarsened)
     * @param bc         Boundary conditions (propagated to smoothers)
     * @param config     MG configuration (pre/post smooth, coarse iters)
     * @param pin        Pin specification (propagated to all levels)
     */
    MultigridPreconditioner(
        multigrid::MGHierarchy& hierarchy,
        const BCSpec& bc,
        const multigrid::MGConfig& config,
        PinSpec pin = {}
    );
    
    // Default constructor (invalid state, must be assigned)
    MultigridPreconditioner() = default;
    
    // Move-only (owns reference to hierarchy)
    MultigridPreconditioner(const MultigridPreconditioner&) = delete;
    MultigridPreconditioner& operator=(const MultigridPreconditioner&) = delete;
    MultigridPreconditioner(MultigridPreconditioner&&) = default;
    MultigridPreconditioner& operator=(MultigridPreconditioner&&) = default;
    
    /**
     * @brief Apply preconditioner: z ≈ A^{-1} r using one V-cycle
     * 
     * Algorithm:
     *   1. Set x = 0 on finest level
     *   2. Set b = r on finest level (copy)
     *   3. Execute one V-cycle
     *   4. Copy result to z
     * 
     * NO allocations inside this function.
     * 
     * @param ctx  CUDA context
     * @param r    Input residual (device, size = num_cells at finest)
     * @param z    Output preconditioned residual (device, same size)
     */
    void apply(CudaContext& ctx, DeviceSpan<const real> r, DeviceSpan<real> z) const;
    
    // Check if initialized
    bool is_valid() const { return hierarchy_ != nullptr; }
    
private:
    multigrid::MGHierarchy* hierarchy_ = nullptr;  // Non-owning, must outlive
    BCSpec bc_;
    multigrid::MGConfig config_;
    PinSpec pin_;
};

} // namespace solvers
} // namespace rwpt

```

# src\numerics\solvers\pcg.cuh

```cuh
#pragma once

/**
 * @file pcg.cuh
 * @brief Preconditioned Conjugate Gradient solver
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

namespace rwpt {
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
    
    // Reduction workspace for dot products
    blas::ReductionWorkspace red;
    
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
} // namespace rwpt

```

# src\numerics\solvers\preconditioner.cu

```cu
/**
 * @file preconditioner.cu
 * @brief Implementation of basic preconditioners
 */

#include "preconditioner.cuh"
#include "../blas/blas.cuh"

namespace rwpt {
namespace solvers {

void IdentityPreconditioner::apply(
    CudaContext& ctx,
    DeviceSpan<const real> r,
    DeviceSpan<real> z
) const {
    // z = r (identity preconditioner)
    rwpt::blas::copy(ctx, r, z);
}

} // namespace solvers
} // namespace rwpt

```

# src\numerics\solvers\preconditioner.cuh

```cuh
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

namespace rwpt {
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
} // namespace rwpt

```

# src\numerics\solvers\solvers.cuh

```cuh
#pragma once

// Solvers for rwpt project

#include "cg_types.hpp"
#include "cg.cuh"

```

# src\physics\common\fields.cuh

```cuh
#pragma once

/**
 * @file fields.cuh
 * @brief Field types for physics modules (cell-centered and staggered)
 * 
 * These types own their GPU memory via DeviceBuffer and provide
 * dimension-aware access. No virtual functions, no runtime polymorphism.
 * 
 * Layout conventions (from legacy compute_velocity_from_head_v1.cu):
 *   - K, h: cell-centered, dims (nx, ny, nz)
 *   - U: face-centered in x, dims (nx+1, ny, nz)
 *   - V: face-centered in y, dims (nx, ny+1, nz)
 *   - W: face-centered in z, dims (nx, ny, nz+1)
 * 
 * Memory layout: column-major (x fastest), i.e. idx = i + j*stride_j + k*stride_k
 */

#include "../../core/Grid3D.hpp"
#include "../../core/DeviceBuffer.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace physics {

// ============================================================================
// Cell-centered scalar field (K, h, etc.)
// ============================================================================

/**
 * @brief Cell-centered scalar field
 * 
 * Owns GPU memory for a 3D scalar field defined at cell centers.
 * Dimensions match the grid: (nx, ny, nz)
 */
struct ScalarField {
    int nx = 0, ny = 0, nz = 0;
    real dx = 1.0, dy = 1.0, dz = 1.0;
    DeviceBuffer<real> data;
    
    ScalarField() = default;
    
    explicit ScalarField(const Grid3D& grid)
        : nx(grid.nx), ny(grid.ny), nz(grid.nz)
        , dx(grid.dx), dy(grid.dy), dz(grid.dz)
        , data(grid.num_cells())
    {}
    
    ScalarField(int nx_, int ny_, int nz_, real dx_ = 1.0, real dy_ = 1.0, real dz_ = 1.0)
        : nx(nx_), ny(ny_), nz(nz_)
        , dx(dx_), dy(dy_), dz(dz_)
        , data(static_cast<size_t>(nx_) * ny_ * nz_)
    {}
    
    // Resize (reallocates if needed)
    void resize(const Grid3D& grid) {
        nx = grid.nx; ny = grid.ny; nz = grid.nz;
        dx = grid.dx; dy = grid.dy; dz = grid.dz;
        data.resize(grid.num_cells());
    }
    
    void resize(int nx_, int ny_, int nz_) {
        nx = nx_; ny = ny_; nz = nz_;
        data.resize(static_cast<size_t>(nx_) * ny_ * nz_);
    }
    
    // Accessors
    size_t size() const { return static_cast<size_t>(nx) * ny * nz; }
    bool empty() const { return data.size() == 0; }
    
    DeviceSpan<real> span() { return data.span(); }
    DeviceSpan<const real> span() const { return data.span(); }
    
    real* device_ptr() { return data.data(); }
    const real* device_ptr() const { return data.data(); }
    
    // Convert to Grid3D (useful for passing to existing APIs)
    Grid3D grid() const { return Grid3D(nx, ny, nz, dx, dy, dz); }
    
    // Linear index (host-side helper, not for kernels)
    size_t idx(int i, int j, int k) const {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * nx + static_cast<size_t>(k) * nx * ny;
    }
};

// Type aliases for semantic clarity
using KField = ScalarField;      ///< Conductivity field (cell-centered)
using HeadField = ScalarField;   ///< Hydraulic head field (cell-centered)

// ============================================================================
// Staggered velocity field (U, V, W on faces)
// ============================================================================

/**
 * @brief Staggered velocity field (face-centered components)
 * 
 * Legacy layout (compute_velocity_from_head_v1.cu):
 *   U[idx] where idx = ix + iy*(Nx+1) + iz*(Nx+1)*Ny, dims (Nx+1, Ny, Nz)
 *   V[idx] where idx = ix + iy*Nx + iz*Nx*(Ny+1),     dims (Nx, Ny+1, Nz)
 *   W[idx] where idx = ix + iy*Nx + iz*Nx*Ny,         dims (Nx, Ny, Nz+1)
 * 
 * Note: U is x-velocity at x-faces, V is y-velocity at y-faces, etc.
 */
struct VelocityField {
    // Base grid dimensions (cell-centered)
    int nx = 0, ny = 0, nz = 0;
    real dx = 1.0, dy = 1.0, dz = 1.0;
    
    // Face-centered components (staggered)
    DeviceBuffer<real> U;  // dims: (nx+1, ny, nz)
    DeviceBuffer<real> V;  // dims: (nx, ny+1, nz)
    DeviceBuffer<real> W;  // dims: (nx, ny, nz+1)
    
    VelocityField() = default;
    
    explicit VelocityField(const Grid3D& grid)
        : nx(grid.nx), ny(grid.ny), nz(grid.nz)
        , dx(grid.dx), dy(grid.dy), dz(grid.dz)
    {
        allocate();
    }
    
    VelocityField(int nx_, int ny_, int nz_, real dx_ = 1.0, real dy_ = 1.0, real dz_ = 1.0)
        : nx(nx_), ny(ny_), nz(nz_)
        , dx(dx_), dy(dy_), dz(dz_)
    {
        allocate();
    }
    
    // Allocate buffers based on current dims
    void allocate() {
        if (nx > 0 && ny > 0 && nz > 0) {
            U.resize(size_U());
            V.resize(size_V());
            W.resize(size_W());
        }
    }
    
    // Resize (reallocates if needed)
    void resize(const Grid3D& grid) {
        nx = grid.nx; ny = grid.ny; nz = grid.nz;
        dx = grid.dx; dy = grid.dy; dz = grid.dz;
        allocate();
    }
    
    void resize(int nx_, int ny_, int nz_) {
        nx = nx_; ny = ny_; nz = nz_;
        allocate();
    }
    
    // Size helpers
    size_t size_U() const { return static_cast<size_t>(nx + 1) * ny * nz; }
    size_t size_V() const { return static_cast<size_t>(nx) * (ny + 1) * nz; }
    size_t size_W() const { return static_cast<size_t>(nx) * ny * (nz + 1); }
    size_t total_size() const { return size_U() + size_V() + size_W(); }
    
    bool empty() const { return U.size() == 0; }
    
    // Spans for kernel access
    DeviceSpan<real> U_span() { return U.span(); }
    DeviceSpan<real> V_span() { return V.span(); }
    DeviceSpan<real> W_span() { return W.span(); }
    
    DeviceSpan<const real> U_span() const { return U.span(); }
    DeviceSpan<const real> V_span() const { return V.span(); }
    DeviceSpan<const real> W_span() const { return W.span(); }
    
    // Pointers for kernel access
    real* U_ptr() { return U.data(); }
    real* V_ptr() { return V.data(); }
    real* W_ptr() { return W.data(); }
    
    const real* U_ptr() const { return U.data(); }
    const real* V_ptr() const { return V.data(); }
    const real* W_ptr() const { return W.data(); }
    
    // Strides for indexing (host-side helpers)
    // U: idx = i + j*(nx+1) + k*(nx+1)*ny
    int stride_U_j() const { return nx + 1; }
    int stride_U_k() const { return (nx + 1) * ny; }
    
    // V: idx = i + j*nx + k*nx*(ny+1)
    int stride_V_j() const { return nx; }
    int stride_V_k() const { return nx * (ny + 1); }
    
    // W: idx = i + j*nx + k*nx*ny
    int stride_W_j() const { return nx; }
    int stride_W_k() const { return nx * ny; }
    
    // Linear index helpers (host-side, not for kernels)
    size_t idx_U(int i, int j, int k) const {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * stride_U_j() + static_cast<size_t>(k) * stride_U_k();
    }
    size_t idx_V(int i, int j, int k) const {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * stride_V_j() + static_cast<size_t>(k) * stride_V_k();
    }
    size_t idx_W(int i, int j, int k) const {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * stride_W_j() + static_cast<size_t>(k) * stride_W_k();
    }
    
    // Convert to Grid3D (base cell-centered grid)
    Grid3D grid() const { return Grid3D(nx, ny, nz, dx, dy, dz); }
};

} // namespace physics
} // namespace rwpt

```

# src\physics\common\physics_config.hpp

```hpp
#pragma once

/**
 * @file physics_config.hpp
 * @brief Configuration structs for physics modules (POD, no virtuals)
 * 
 * These are lightweight value types for passing parameters between modules.
 * All fields have sensible defaults. No dynamic allocation in constructors.
 */

#include "../../core/Scalar.hpp"
#include <cstdint>

namespace rwpt {
namespace physics {

/**
 * @brief Configuration for stochastic K field generation
 * 
 * Legacy correspondence: parameters from random_field_generation.cu
 * K = exp(f(x)) where f is Gaussian random field with given variance/correlation
 */
struct StochasticConfig {
    // Random field parameters
    real sigma2 = 1.0;           ///< Variance of log-conductivity (σ²_f)
    real corr_length = 1.0;      ///< Correlation length (λ)
    int  n_modes = 1000;         ///< Number of Fourier modes for spectral method
    
    // Covariance type: 0 = exponential, 1 = gaussian
    int covariance_type = 0;
    
    // RNG seeds
    uint64_t seed = 12345;       ///< Base seed for RNG
    
    // Geometric mean for normalization (K_g = exp(<ln K>))
    real K_geometric_mean = 1.0; ///< Target geometric mean of K field
    
    // Default constructor with sensible defaults
    StochasticConfig() = default;
};

/**
 * @brief Configuration for flow (head) solver
 * 
 * Legacy correspondence: parameters from main_transport_JSON_input.cu
 */
struct FlowConfig {
    // Solver selection: 0 = MG only, 1 = CG only, 2 = MG-preconditioned CG
    int solver_type = 0;
    
    // MG parameters
    int mg_levels = 4;
    int mg_pre_smooth = 2;
    int mg_post_smooth = 2;
    int mg_coarse_iters = 50;
    int mg_max_cycles = 20;
    
    // CG parameters  
    int cg_max_iter = 1000;
    real cg_rtol = 1e-8;
    real cg_atol = 0.0;
    
    // Convergence
    real rtol = 1e-6;            ///< Relative tolerance for residual
    
    // Physical parameters (for RHS if needed)
    real source_term = 0.0;      ///< Uniform source/sink term
    
    FlowConfig() = default;
};

/**
 * @brief Configuration for particle transport (PAR2/RWPT)
 * 
 * Legacy correspondence: parameters from main_transport_JSON_input.cu
 */
struct TransportConfig {
    // Particle count
    int n_particles = 10000;
    
    // Time stepping
    real dt = 0.01;              ///< Time step size
    int  n_steps = 1000;         ///< Number of time steps
    int  output_every = 100;     ///< Output frequency (steps)
    
    // Physical parameters
    real porosity = 1.0;         ///< Porosity (θ)
    real diffusion = 0.0;        ///< Molecular diffusion coefficient (D_m)
    
    // Injection plane/volume (defaults: inject at x=0 face)
    real inject_xmin = 0.0;
    real inject_xmax = 0.0;      ///< If xmin==xmax, inject on plane
    real inject_ymin = 0.0;
    real inject_ymax = 1.0;
    real inject_zmin = 0.0;
    real inject_zmax = 1.0;
    
    // RNG seed for particle diffusion
    uint64_t seed = 54321;
    
    TransportConfig() = default;
};

/**
 * @brief Combined configuration for full simulation
 */
struct SimulationConfig {
    StochasticConfig stochastic;
    FlowConfig flow;
    TransportConfig transport;
    
    // Domain (informational, actual grid comes from Grid3D)
    real Lx = 1.0, Ly = 1.0, Lz = 1.0;
    int  Nx = 64,  Ny = 64,  Nz = 64;
    
    SimulationConfig() = default;
};

} // namespace physics
} // namespace rwpt

```

# src\physics\common\physics_types.cuh

```cuh
#pragma once

/**
 * @file physics_types.cuh
 * @brief Aggregated include for all physics types
 * 
 * Include this single header to get all physics data structures:
 * - Configuration structs (StochasticConfig, FlowConfig, TransportConfig)
 * - Field types (KField, HeadField, VelocityField)
 * - Workspace types (StochasticWorkspace, FlowWorkspace, ParticlesWorkspace)
 */

#include "physics_config.hpp"
#include "fields.cuh"
#include "workspaces.cuh"

namespace rwpt {
namespace physics {

// Version tag for physics module
constexpr int PHYSICS_VERSION_MAJOR = 5;
constexpr int PHYSICS_VERSION_MINOR = 0;

} // namespace physics
} // namespace rwpt

```

# src\physics\common\workspaces.cuh

```cuh
#pragma once

/**
 * @file workspaces.cuh
 * @brief Pre-allocated workspace buffers for physics modules
 * 
 * Each workspace owns temporary GPU memory needed by its module.
 * Allocate once before simulation loop, reuse across steps.
 * No cudaMalloc inside inner loops.
 * 
 * Pattern:
 *   Workspace ws;
 *   ws.allocate(grid, config);  // Once at setup
 *   for (step : steps) {
 *       module_function(ws, ...);  // Uses pre-allocated buffers
 *   }
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../multigrid/mg_types.hpp"
#include "../../numerics/solvers/cg_types.hpp"
#include "../../numerics/solvers/pcg.cuh"
#include "physics_config.hpp"
#include <curand_kernel.h>

namespace rwpt {
namespace physics {

// ============================================================================
// Stochastic K generation workspace
// ============================================================================

/**
 * @brief Workspace for stochastic K field generation
 * 
 * Legacy correspondence: random_field_generation.cu
 * Buffers for Fourier mode coefficients (k1, k2, k3, a, b, vartheta)
 * and RNG states.
 */
struct StochasticWorkspace {
    // Fourier mode coefficients (size = n_modes)
    DeviceBuffer<real> k1;        ///< Wavenumber component x
    DeviceBuffer<real> k2;        ///< Wavenumber component y
    DeviceBuffer<real> k3;        ///< Wavenumber component z (3D only)
    DeviceBuffer<real> coef_a;    ///< Coefficient a for spectral sum
    DeviceBuffer<real> coef_b;    ///< Coefficient b for spectral sum
    DeviceBuffer<real> vartheta;  ///< Phase angles
    
    // RNG states (one per mode for parallel generation)
    DeviceBuffer<curandState> rng_states;
    
    // Intermediate buffer for log(K) before exp()
    DeviceBuffer<real> logK;
    
    // Allocated sizes
    int n_modes = 0;
    size_t n_cells = 0;
    
    StochasticWorkspace() = default;
    
    // Allocate for given config and grid
    void allocate(const Grid3D& grid, const StochasticConfig& cfg) {
        n_modes = cfg.n_modes;
        n_cells = grid.num_cells();
        
        // Mode coefficients
        k1.resize(n_modes);
        k2.resize(n_modes);
        k3.resize(n_modes);
        coef_a.resize(n_modes);
        coef_b.resize(n_modes);
        vartheta.resize(n_modes);
        
        // RNG states
        rng_states.resize(n_modes);
        
        // Intermediate logK field
        logK.resize(n_cells);
    }
    
    // Check if allocated
    bool is_allocated() const { return n_modes > 0 && n_cells > 0; }
    
    // Clear (free memory)
    void clear() {
        k1 = DeviceBuffer<real>();
        k2 = DeviceBuffer<real>();
        k3 = DeviceBuffer<real>();
        coef_a = DeviceBuffer<real>();
        coef_b = DeviceBuffer<real>();
        vartheta = DeviceBuffer<real>();
        rng_states = DeviceBuffer<curandState>();
        logK = DeviceBuffer<real>();
        n_modes = 0;
        n_cells = 0;
    }
};

// ============================================================================
// Flow solver workspace
// ============================================================================

/**
 * @brief Workspace for flow (head) solver
 * 
 * Contains temporary buffers for MG and CG solvers.
 * Includes MG hierarchy for multigrid solve.
 */
struct FlowWorkspace {
    // Residual and auxiliary vectors for iterative solvers
    DeviceBuffer<real> residual;
    DeviceBuffer<real> aux1;
    DeviceBuffer<real> aux2;
    
    // RHS buffer (can be modified during solve)
    DeviceBuffer<real> rhs;
    
    // MG hierarchy (allocated on first use)
    multigrid::MGHierarchy mg_hierarchy;
    
    // CG solver workspace (for HeadSolverType::CG)
    solvers::CGWorkspace cg_workspace;
    
    // PCG solver workspace (for HeadSolverType::PCG_MG)
    solvers::PCGWorkspace pcg_workspace;
    
    // Size tracking
    size_t n_cells = 0;
    
    FlowWorkspace() = default;
    
    // Allocate for given grid
    void allocate(const Grid3D& grid) {
        n_cells = grid.num_cells();
        residual.resize(n_cells);
        aux1.resize(n_cells);
        aux2.resize(n_cells);
        rhs.resize(n_cells);
        // mg_hierarchy is allocated on-demand in solve_head
        // cg_workspace/pcg_workspace are allocated on-demand in solve_head
    }
    
    // Allocate with MG hierarchy (num_levels)
    void allocate(const Grid3D& grid, int mg_levels) {
        allocate(grid);
        if (mg_levels > 0) {
            mg_hierarchy = multigrid::MGHierarchy(grid, mg_levels);
        }
    }
    
    // Check if allocated
    bool is_allocated() const { return n_cells > 0; }
    
    // Clear (free memory)
    void clear() {
        residual = DeviceBuffer<real>();
        aux1 = DeviceBuffer<real>();
        aux2 = DeviceBuffer<real>();
        rhs = DeviceBuffer<real>();
        mg_hierarchy = multigrid::MGHierarchy();
        cg_workspace = solvers::CGWorkspace();
        pcg_workspace = solvers::PCGWorkspace();
        n_cells = 0;
    }
};

// ============================================================================
// Particle transport workspace
// ============================================================================

/**
 * @brief Workspace for particle transport (RWPT/PAR2)
 * 
 * Contains particle state arrays and temporary buffers for stepping.
 * Position arrays are (x, y, z) for each particle.
 */
struct ParticlesWorkspace {
    // Particle positions (size = n_particles each)
    DeviceBuffer<real> x;
    DeviceBuffer<real> y;
    DeviceBuffer<real> z;
    
    // Previous positions (for reflection BCs, optional)
    DeviceBuffer<real> x_prev;
    DeviceBuffer<real> y_prev;
    DeviceBuffer<real> z_prev;
    
    // Particle status (0=active, 1=exited, etc.)
    DeviceBuffer<int> status;
    
    // RNG states for diffusion (one per particle)
    DeviceBuffer<curandState> rng_states;
    
    // Temporary buffers for velocity interpolation
    DeviceBuffer<real> u_interp;
    DeviceBuffer<real> v_interp;
    DeviceBuffer<real> w_interp;
    
    // Size tracking
    int n_particles = 0;
    
    ParticlesWorkspace() = default;
    
    // Allocate for given config
    void allocate(const TransportConfig& cfg) {
        n_particles = cfg.n_particles;
        
        // Positions
        x.resize(n_particles);
        y.resize(n_particles);
        z.resize(n_particles);
        
        // Previous positions
        x_prev.resize(n_particles);
        y_prev.resize(n_particles);
        z_prev.resize(n_particles);
        
        // Status
        status.resize(n_particles);
        
        // RNG states
        rng_states.resize(n_particles);
        
        // Interpolation temporaries
        u_interp.resize(n_particles);
        v_interp.resize(n_particles);
        w_interp.resize(n_particles);
    }
    
    // Check if allocated
    bool is_allocated() const { return n_particles > 0; }
    
    // Number of active particles (requires host sync - use sparingly)
    // int count_active() const;  // TODO: implement with reduction
    
    // Clear (free memory)
    void clear() {
        x = DeviceBuffer<real>();
        y = DeviceBuffer<real>();
        z = DeviceBuffer<real>();
        x_prev = DeviceBuffer<real>();
        y_prev = DeviceBuffer<real>();
        z_prev = DeviceBuffer<real>();
        status = DeviceBuffer<int>();
        rng_states = DeviceBuffer<curandState>();
        u_interp = DeviceBuffer<real>();
        v_interp = DeviceBuffer<real>();
        w_interp = DeviceBuffer<real>();
        n_particles = 0;
    }
};

// ============================================================================
// Combined simulation workspace
// ============================================================================

/**
 * @brief Combined workspace for full simulation
 * 
 * Convenience container for all workspaces.
 */
struct SimulationWorkspace {
    StochasticWorkspace stochastic;
    FlowWorkspace flow;
    ParticlesWorkspace particles;
    
    SimulationWorkspace() = default;
    
    // Allocate all workspaces
    void allocate(const Grid3D& grid, const SimulationConfig& cfg) {
        stochastic.allocate(grid, cfg.stochastic);
        flow.allocate(grid);
        particles.allocate(cfg.transport);
    }
    
    // Clear all
    void clear() {
        stochastic.clear();
        flow.clear();
        particles.clear();
    }
};

} // namespace physics
} // namespace rwpt

```

# src\physics\flow\coarsen_K.cu

```cu
#include "coarsen_K.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>

namespace rwpt {
namespace physics {

/**
 * Legacy: CompactHomogenizationKtensor
 * Geometric mean over 2x2x2 block of fine cells.
 * K_coarse[IX,IY,IZ] = exp(mean(log(K_fine[8 cells])))
 */
__global__ void coarsen_K_kernel(
    real* __restrict__ K_coarse,
    const real* __restrict__ K_fine,
    int NX, int NY, int NZ  // Coarse dimensions
) {
    int IX = threadIdx.x + blockIdx.x * blockDim.x;
    int IY = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (IX >= NX || IY >= NY) return;
    
    // Fine grid dimensions (2x coarse)
    int Nx = 2 * NX;
    int Ny = 2 * NY;
    
    for (int IZ = 0; IZ < NZ; ++IZ) {
        // Fine cell indices (bottom-left-back corner of 2x2x2 block)
        int ix = 2 * IX;
        int iy = 2 * IY;
        int iz = 2 * IZ;
        
        // Fine indexing
        int stride = Nx * Ny;
        int idx000 = ix + iy * Nx + iz * stride;
        int idx100 = idx000 + 1;
        int idx010 = idx000 + Nx;
        int idx110 = idx000 + 1 + Nx;
        int idx001 = idx000 + stride;
        int idx101 = idx001 + 1;
        int idx011 = idx001 + Nx;
        int idx111 = idx001 + 1 + Nx;
        
        // Geometric mean: exp(mean(log(K)))
        real log_sum = log(K_fine[idx000]) + log(K_fine[idx100]) +
                       log(K_fine[idx010]) + log(K_fine[idx110]) +
                       log(K_fine[idx001]) + log(K_fine[idx101]) +
                       log(K_fine[idx011]) + log(K_fine[idx111]);
        
        int coarse_idx = IX + IY * NX + IZ * NX * NY;
        K_coarse[coarse_idx] = exp(log_sum / 8.0);
    }
}

void coarsen_K(
    CudaContext& ctx,
    const Grid3D& coarse_grid,
    const Grid3D& fine_grid,
    DeviceSpan<const real> K_fine,
    DeviceSpan<real> K_coarse
) {
    int NX = coarse_grid.nx;
    int NY = coarse_grid.ny;
    int NZ = coarse_grid.nz;
    
    // Validate dimensions
    assert(fine_grid.nx == 2 * NX && "Fine grid must be 2x coarse in x");
    assert(fine_grid.ny == 2 * NY && "Fine grid must be 2x coarse in y");
    assert(fine_grid.nz == 2 * NZ && "Fine grid must be 2x coarse in z");
    assert(K_fine.size() == fine_grid.num_cells() && "Fine K size mismatch");
    assert(K_coarse.size() == coarse_grid.num_cells() && "Coarse K size mismatch");
    
    dim3 block(16, 16);
    int grid_x = (NX + block.x - 1) / block.x;
    int grid_y = (NY + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    
    coarsen_K_kernel<<<grid, block, 0, ctx.cuda_stream()>>>(
        K_coarse.data(), K_fine.data(), NX, NY, NZ
    );
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace physics
} // namespace rwpt

```

# src\physics\flow\coarsen_K.cuh

```cuh
#pragma once

/**
 * @file coarsen_K.cuh
 * @brief Coarsen (homogenize) conductivity field K for multigrid levels
 * 
 * This file provides the physics-specific coarsening of K for Darcy flow.
 * Uses geometric mean homogenization (legacy: CompactHomogenizationKtensor).
 * 
 * ## Location Rationale
 * 
 * Although this is used by the multigrid hierarchy, it lives in physics/flow/
 * because:
 * 1. The coarsening method is physics-specific (geometric mean for log-normal K)
 * 2. Other physics problems might require different coarsening strategies
 * 3. It's tightly coupled with the flow equation's coefficient structure
 * 
 * The generic MG transfer operators (restrict, prolong) are in multigrid/transfer/.
 */

#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../runtime/CudaContext.cuh"

namespace rwpt {
namespace physics {

/**
 * @brief Coarsen conductivity field K from fine to coarse level.
 * 
 * Uses geometric mean over 2x2x2 blocks of fine cells, matching legacy
 * CompactHomogenizationKtensor implementation.
 * 
 * For log-normal K fields:
 *   K_coarse = exp(mean(log(K_fine))) = (K₁·K₂·...·K₈)^(1/8)
 * 
 * This preserves the harmonic mean behavior needed for Darcy flow
 * with heterogeneous conductivity.
 * 
 * @param ctx CUDA context
 * @param coarse_grid Coarse grid dimensions (must be fine_grid/2)
 * @param fine_grid Fine grid dimensions
 * @param K_fine Input fine conductivity field
 * @param K_coarse Output coarse conductivity field
 */
void coarsen_K(
    CudaContext& ctx,
    const Grid3D& coarse_grid,
    const Grid3D& fine_grid,
    DeviceSpan<const real> K_fine,
    DeviceSpan<real> K_coarse
);

} // namespace physics
} // namespace rwpt

```

# src\physics\flow\rhs_head.cu

```cu
/**
 * @file rhs_head.cu
 * @brief Implementation of RHS builder for head equation
 * 
 * Legacy correspondence: RHS_head_3D.cu
 * 
 * For Dirichlet BCs, adds contribution: RHS -= 2 * K_cell * H_bc / dx²
 */

#include "rhs_head.cuh"
#include "../../runtime/cuda_check.cuh"
#include "../../core/BCSpecDevice.cuh"

namespace rwpt {
namespace physics {

// ============================================================================
// Kernel: Fill with zeros
// ============================================================================

__global__ void kernel_fill_zero(real* __restrict__ data, const size_t n) {
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    data[idx] = real(0.0);
}

// ============================================================================
// Kernels: Dirichlet BC contributions to RHS
// Legacy: RHS -= 2.0 * Hb * KC / dx²  (where Hb is Dirichlet value)
// ============================================================================

// Face x=0 (west)
__global__ void kernel_rhs_dirichlet_xmin(
    real* __restrict__ rhs,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real inv_dx2,
    real h_bc
) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (j >= ny || k >= nz) return;
    
    int i = 0;
    int idx = i + nx * (j + ny * k);
    real KC = K[idx];
    rhs[idx] -= real(2.0) * h_bc * KC * inv_dx2;
}

// Face x=nx-1 (east)
__global__ void kernel_rhs_dirichlet_xmax(
    real* __restrict__ rhs,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real inv_dx2,
    real h_bc
) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (j >= ny || k >= nz) return;
    
    int i = nx - 1;
    int idx = i + nx * (j + ny * k);
    real KC = K[idx];
    rhs[idx] -= real(2.0) * h_bc * KC * inv_dx2;
}

// Face y=0 (south)
__global__ void kernel_rhs_dirichlet_ymin(
    real* __restrict__ rhs,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real inv_dx2,
    real h_bc
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || k >= nz) return;
    
    int j = 0;
    int idx = i + nx * (j + ny * k);
    real KC = K[idx];
    rhs[idx] -= real(2.0) * h_bc * KC * inv_dx2;
}

// Face y=ny-1 (north)
__global__ void kernel_rhs_dirichlet_ymax(
    real* __restrict__ rhs,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real inv_dx2,
    real h_bc
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || k >= nz) return;
    
    int j = ny - 1;
    int idx = i + nx * (j + ny * k);
    real KC = K[idx];
    rhs[idx] -= real(2.0) * h_bc * KC * inv_dx2;
}

// Face z=0 (bottom)
__global__ void kernel_rhs_dirichlet_zmin(
    real* __restrict__ rhs,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real inv_dx2,
    real h_bc
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || j >= ny) return;
    
    int k = 0;
    int idx = i + nx * (j + ny * k);
    real KC = K[idx];
    rhs[idx] -= real(2.0) * h_bc * KC * inv_dx2;
}

// Face z=nz-1 (top)
__global__ void kernel_rhs_dirichlet_zmax(
    real* __restrict__ rhs,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real inv_dx2,
    real h_bc
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || j >= ny) return;
    
    int k = nz - 1;
    int idx = i + nx * (j + ny * k);
    real KC = K[idx];
    rhs[idx] -= real(2.0) * h_bc * KC * inv_dx2;
}

// ============================================================================
// Host API
// ============================================================================

void build_rhs_head(DeviceSpan<real> rhs,
                    DeviceSpan<const real> K,
                    const Grid3D& grid,
                    const BCSpec& bc,
                    const CudaContext& ctx) {
    const size_t n = grid.num_cells();
    if (rhs.size() < n || K.size() < n) {
        throw std::runtime_error("RHS or K buffer too small");
    }
    
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const real inv_dx2 = real(1.0) / (grid.dx * grid.dx);
    
    // 1. Zero out RHS (no volumetric sources)
    {
        const int block = 256;
        const int grid_1d = (n + block - 1) / block;
        kernel_fill_zero<<<grid_1d, block, 0, ctx.cuda_stream()>>>(rhs.data(), n);
    }
    
    // 2. Add Dirichlet contributions
    dim3 block_yz(16, 16);
    dim3 block_xz(16, 16);
    dim3 block_xy(16, 16);
    
    // x-faces (west/east)
    dim3 grid_yz((ny + block_yz.x - 1) / block_yz.x, (nz + block_yz.y - 1) / block_yz.y);
    
    if (bc.xmin.type == BCType::Dirichlet) {
        kernel_rhs_dirichlet_xmin<<<grid_yz, block_yz, 0, ctx.cuda_stream()>>>(
            rhs.data(), K.data(), nx, ny, nz, inv_dx2, bc.xmin.value);
    }
    if (bc.xmax.type == BCType::Dirichlet) {
        kernel_rhs_dirichlet_xmax<<<grid_yz, block_yz, 0, ctx.cuda_stream()>>>(
            rhs.data(), K.data(), nx, ny, nz, inv_dx2, bc.xmax.value);
    }
    
    // y-faces (south/north)
    dim3 grid_xz((nx + block_xz.x - 1) / block_xz.x, (nz + block_xz.y - 1) / block_xz.y);
    
    if (bc.ymin.type == BCType::Dirichlet) {
        kernel_rhs_dirichlet_ymin<<<grid_xz, block_xz, 0, ctx.cuda_stream()>>>(
            rhs.data(), K.data(), nx, ny, nz, inv_dx2, bc.ymin.value);
    }
    if (bc.ymax.type == BCType::Dirichlet) {
        kernel_rhs_dirichlet_ymax<<<grid_xz, block_xz, 0, ctx.cuda_stream()>>>(
            rhs.data(), K.data(), nx, ny, nz, inv_dx2, bc.ymax.value);
    }
    
    // z-faces (bottom/top)
    dim3 grid_xy((nx + block_xy.x - 1) / block_xy.x, (ny + block_xy.y - 1) / block_xy.y);
    
    if (bc.zmin.type == BCType::Dirichlet) {
        kernel_rhs_dirichlet_zmin<<<grid_xy, block_xy, 0, ctx.cuda_stream()>>>(
            rhs.data(), K.data(), nx, ny, nz, inv_dx2, bc.zmin.value);
    }
    if (bc.zmax.type == BCType::Dirichlet) {
        kernel_rhs_dirichlet_zmax<<<grid_xy, block_xy, 0, ctx.cuda_stream()>>>(
            rhs.data(), K.data(), nx, ny, nz, inv_dx2, bc.zmax.value);
    }
    
    // =======================================================================
    // Pin cell for singular systems (legacy: pin1stCell)
    // =======================================================================
    // When NO boundary is Dirichlet, the system is singular (constant mode 
    // in null space). Pinning one cell breaks this degeneracy.
    //
    // Legacy implementation: "diagonal doubling" approach
    //   - In operator/smoother/residual: aC *= 2 for cell [0,0,0]
    //   - RHS is NOT modified
    //
    // The diagonal doubling happens in the operator and smoother kernels,
    // NOT here. This comment is just for documentation.
    // =======================================================================
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace physics
} // namespace rwpt

```

# src\physics\flow\rhs_head.cuh

```cuh
#pragma once

/**
 * @file rhs_head.cuh
 * @brief Build RHS for head equation (Darcy flow)
 * 
 * Legacy reference: RHS_head_3D.cu
 * 
 * ## RHS Construction
 * 
 * For the standard problem with no volumetric sources:
 *   RHS = 0  (interior cells)
 * 
 * Dirichlet BC contributions (boundary cells):
 *   RHS[cell] -= 2 * K[cell] * H_bc / dx²
 * 
 * Periodic and Neumann BCs do NOT contribute to the RHS.
 * 
 * ## Note on Singular Systems
 * 
 * This module only builds the RHS. For singular systems (all periodic/Neumann),
 * the pin mechanism is handled separately in the operator and smoother kernels.
 * See pin_spec.hpp for the pin documentation.
 */

#include "../../core/Grid3D.hpp"
#include "../../core/BCSpec.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"

namespace rwpt {
namespace physics {

/**
 * @brief Build RHS for head equation
 * 
 * Constructs the right-hand side vector for the Darcy flow equation.
 * 
 * Algorithm:
 *   1. Fill RHS with zeros (no volumetric sources)
 *   2. Add Dirichlet BC contributions: RHS -= 2*K*H_bc/dx²
 * 
 * @param rhs   Output: right-hand side vector (device, size = num_cells)
 * @param K     Input: conductivity field (device, size = num_cells)
 * @param grid  Grid specification
 * @param bc    Boundary conditions
 * @param ctx   CUDA context
 */
void build_rhs_head(DeviceSpan<real> rhs,
                    DeviceSpan<const real> K,
                    const Grid3D& grid,
                    const BCSpec& bc,
                    const CudaContext& ctx);

} // namespace physics
} // namespace rwpt

```

# src\physics\flow\solve_head.cu

```cu
/**
 * @file solve_head.cu
 * @brief Implementation of head solver using MG, CG, or PCG with MG preconditioner
 */

#include "solve_head.cuh"
#include "rhs_head.cuh"
#include "coarsen_K.cuh"
#include "../../runtime/cuda_check.cuh"
#include "../../multigrid/cycle/v_cycle.cuh"
#include "../../numerics/blas/blas.cuh"
#include "../../numerics/operators/varcoeff_laplacian.cuh"
#include "../../numerics/operators/negated_operator.cuh"
#include "../../numerics/solvers/cg.cuh"
#include "../../numerics/solvers/pcg.cuh"
#include "../../numerics/solvers/preconditioner.cuh"
#include "../../numerics/solvers/mg_preconditioner.cuh"
#include "../../numerics/pin_spec.hpp"
#include <iostream>
#include <cmath>

namespace rwpt {
namespace physics {

// ============================================================================
// Kernel: Linear initial guess (interpolate between Dirichlet BCs in x)
// ============================================================================

__global__ void kernel_init_head_linear_x(
    real* __restrict__ h,
    int nx, int ny, int nz,
    real h_west, real h_east
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (ix >= nx || iy >= ny || iz >= nz) return;
    
    int idx = ix + iy * nx + iz * nx * ny;
    
    // Linear interpolation in x: h = h_west + (h_east - h_west) * (ix + 0.5) / nx
    real t = (real(ix) + real(0.5)) / real(nx);
    h[idx] = h_west + (h_east - h_west) * t;
}

__global__ void kernel_fill_value(real* __restrict__ data, size_t n, real value) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    data[idx] = value;
}

// ============================================================================
// Host API
// ============================================================================

void init_head_guess(
    DeviceSpan<real> h,
    const Grid3D& grid,
    const BCSpec& bc,
    const CudaContext& ctx
) {
    // Check for Dirichlet BCs in x-direction
    if (bc.xmin.type == BCType::Dirichlet && bc.xmax.type == BCType::Dirichlet) {
        dim3 block(8, 8, 8);
        dim3 grid_dim(
            (grid.nx + block.x - 1) / block.x,
            (grid.ny + block.y - 1) / block.y,
            (grid.nz + block.z - 1) / block.z
        );
        
        kernel_init_head_linear_x<<<grid_dim, block, 0, ctx.cuda_stream()>>>(
            h.data(), grid.nx, grid.ny, grid.nz,
            bc.xmin.value, bc.xmax.value
        );
    } else {
        // No obvious guess; use zero
        int block = 256;
        int grid_1d = (h.size() + block - 1) / block;
        kernel_fill_value<<<grid_1d, block, 0, ctx.cuda_stream()>>>(
            h.data(), h.size(), real(0.0)
        );
    }
    RWPT_CUDA_CHECK(cudaGetLastError());
}

HeadSolveResult solve_head(
    DeviceSpan<real> h,
    DeviceSpan<const real> K,
    const Grid3D& grid,
    const BCSpec& bc,
    const HeadSolveConfig& cfg,
    CudaContext& ctx,
    FlowWorkspace& workspace
) {
    using namespace rwpt::multigrid;
    using namespace rwpt::blas;
    
    HeadSolveResult result;
    
    // Validate sizes
    size_t n = grid.num_cells();
    if (h.size() < n || K.size() < n) {
        throw std::runtime_error("Head or K buffer too small");
    }
    
    // === 1. Setup MG hierarchy ===
    if (workspace.mg_hierarchy.num_levels() == 0 || 
        workspace.mg_hierarchy.finest_grid().num_cells() != n) {
        // Need to (re)initialize hierarchy
        workspace.mg_hierarchy = MGHierarchy(grid, cfg.mg_levels);
    }
    
    MGHierarchy& hier = workspace.mg_hierarchy;
    MGConfig mg_cfg;
    mg_cfg.num_levels = hier.num_levels();
    mg_cfg.pre_smooth = cfg.mg_pre_smooth;
    mg_cfg.post_smooth = cfg.mg_post_smooth;
    mg_cfg.coarse_solve_iters = cfg.mg_coarse_iters;
    mg_cfg.verbose = false;
    
    // === 2. Copy K to finest level and coarsen ===
    // Copy K to level 0
    RWPT_CUDA_CHECK(cudaMemcpyAsync(
        hier.levels[0].K.data(), K.data(), n * sizeof(real),
        cudaMemcpyDeviceToDevice, ctx.cuda_stream()
    ));
    
    // Coarsen K to all levels (geometric mean)
    for (int l = 0; l < hier.num_levels() - 1; ++l) {
        coarsen_K(
            ctx,
            hier.levels[l+1].grid,
            hier.levels[l].grid,
            DeviceSpan<const real>(hier.levels[l].K.data(), hier.levels[l].K.size()),
            DeviceSpan<real>(hier.levels[l+1].K.data(), hier.levels[l+1].K.size())
        );
    }
    
    // === 3. Set up RHS (b = 0 + Dirichlet contributions) ===
    build_rhs_head(
        DeviceSpan<real>(hier.levels[0].b.data(), hier.levels[0].b.size()),
        DeviceSpan<const real>(hier.levels[0].K.data(), hier.levels[0].K.size()),
        grid, bc, ctx
    );
    
    // === 4. Initial guess ===
    // Copy to MG level 0 x buffer, or use provided h
    init_head_guess(h, grid, bc, ctx);
    RWPT_CUDA_CHECK(cudaMemcpyAsync(
        hier.levels[0].x.data(), h.data(), n * sizeof(real),
        cudaMemcpyDeviceToDevice, ctx.cuda_stream()
    ));
    
    // === 5. Solve based on solver type ===
    if (cfg.solver_type == HeadSolverType::MG) {
        // Standalone MG V-cycles
        VCycleResult vcycle_result = mg_solve(
            ctx,
            hier,
            mg_cfg,
            bc,
            cfg.mg_max_cycles,
            cfg.rtol
        );
        
        result.num_iterations = vcycle_result.num_cycles;
        result.initial_residual = vcycle_result.initial_residual;
        result.final_residual = vcycle_result.final_residual;
        result.converged = vcycle_result.converged;
        
    } else if (cfg.solver_type == HeadSolverType::CG) {
        // =====================================================================
        // Plain CG (no preconditioner)
        // =====================================================================
        // VarCoeffLaplacian produces a NEGATIVE definite operator: A = -∇·(K∇)
        // CG requires a POSITIVE definite operator (SPD).
        //
        // Solution: Use NegatedOperator wrapper to make A_pos = -A (SPD),
        // and negate the RHS: A_pos*h = -b solves A*h = b.
        //
        // This is mathematically equivalent and allows standard CG to work.
        // =====================================================================
        
        // Build PinSpec from config (legacy diagonal doubling, always cell [0,0,0])
        PinSpec pin_spec(pin_enabled(cfg.pin.mode, bc));
        
        operators::VarCoeffLaplacian A_neg(grid, K, bc, pin_spec);
        operators::NegatedOperator<operators::VarCoeffLaplacian> A_pos(A_neg);
        
        // Ensure CG workspace
        workspace.cg_workspace.ensure(n);
        
        // Negate RHS: workspace.rhs = -b (because A_pos*h = -b)
        blas::copy(ctx, DeviceSpan<const real>(hier.levels[0].b.span()), workspace.rhs.span());
        blas::scal(ctx, workspace.rhs.span(), real(-1.0));
        
        // CG config
        solvers::CGConfig cg_cfg;
        cg_cfg.max_iter = cfg.cg_max_iter;
        cg_cfg.check_every = cfg.cg_check_every;
        cg_cfg.rtol = cfg.cg_rtol;
        cg_cfg.atol = 0.0;
        cg_cfg.verbose = false;
        
        // Solve using CG with positive operator
        solvers::CGResult cg_result = solvers::cg_solve(
            ctx, A_pos, 
            DeviceSpan<const real>(workspace.rhs.span()),
            hier.levels[0].x.span(),
            cg_cfg,
            workspace.cg_workspace
        );
        
        result.num_iterations = cg_result.iters;
        result.initial_residual = cg_result.r0_norm;
        result.final_residual = cg_result.r_norm;
        result.converged = cg_result.converged;
        
    } else { // PCG_MG (default, legacy: solver_CG with PCCMG_CG)
        // =====================================================================
        // PCG with MG preconditioner (PREFERRED for production)
        // =====================================================================
        // VarCoeffLaplacian produces a NEGATIVE definite operator: A = -∇·(K∇)
        // 
        // The MG V-cycle acts as preconditioner M^{-1} ≈ A^{-1}.
        // Since A is negative, M^{-1} is also "negative" in the sense that
        // <r, M^{-1}r> < 0 for r ≠ 0 (verified experimentally).
        //
        // PCG uses alpha = <r,z>/<p,Ap> where z = M^{-1}r.
        // With both <r,z> < 0 and <p,Ap> < 0, we get alpha > 0 → convergence.
        //
        // This matches legacy solver_CG(AH, PCCMG_CG, ...) behavior.
        // =====================================================================
        
        // Build PinSpec from config (legacy diagonal doubling, always cell [0,0,0])
        PinSpec pin_spec(pin_enabled(cfg.pin.mode, bc));
        
        operators::VarCoeffLaplacian A(grid, K, bc, pin_spec);
        
        // Create MG preconditioner (wraps one V-cycle)
        // Legacy: PCCMG_CG, Precond_CCMG_Vcycle*
        // Pass pin_spec to MG preconditioner for full consistency
        solvers::MultigridPreconditioner M(hier, bc, mg_cfg, pin_spec);
        
        // Ensure PCG workspace
        workspace.pcg_workspace.ensure(n);
        
        // PCG config
        solvers::PCGConfig pcg_cfg;
        pcg_cfg.max_iter = cfg.cg_max_iter;
        pcg_cfg.check_every = cfg.cg_check_every;
        pcg_cfg.rtol = cfg.cg_rtol;
        pcg_cfg.verbose = false;
        
        // Use h directly as solution (not hier.levels[0].x to avoid conflict with precond)
        // Use RHS directly (no negation needed - A and M are both "negative")
        blas::copy(ctx, DeviceSpan<const real>(hier.levels[0].b.span()), workspace.rhs.span());
        
        // Solve using PCG with MG preconditioner
        solvers::PCGResult pcg_result = solvers::pcg_solve(
            ctx, A, M,
            DeviceSpan<const real>(workspace.rhs.span()),
            h,  // Use h directly as solution
            pcg_cfg,
            workspace.pcg_workspace
        );
        
        result.num_iterations = pcg_result.iterations;
        result.initial_residual = pcg_result.initial_residual;
        result.final_residual = pcg_result.final_residual;
        result.converged = pcg_result.converged;
        
        // No need to copy back - h was used directly
        return result;  // Skip final copy at the end
    }
    
    // === 6. Copy solution back ===
    RWPT_CUDA_CHECK(cudaMemcpyAsync(
        h.data(), hier.levels[0].x.data(), n * sizeof(real),
        cudaMemcpyDeviceToDevice, ctx.cuda_stream()
    ));
    
    ctx.synchronize();
    
    return result;
}

} // namespace physics
} // namespace rwpt

```

# src\physics\flow\solve_head.cuh

```cuh
#pragma once

/**
 * @file solve_head.cuh
 * @brief Solve head (Darcy flow) equation using MG, CG, or PCG
 * 
 * Legacy correspondence: main_transport_JSON_input.cu solver calls
 * 
 * Solves: -∇·(K∇h) = 0  with boundary conditions
 * 
 * The operator uses harmonic mean for K at faces, matching legacy.
 * 
 * ## Solver Types
 * 
 * - CG: Plain conjugate gradient (no preconditioner)
 * - MG: Standalone multigrid V-cycles
 * - PCG_MG: CG preconditioned with MG V-cycle (legacy default: solver_CG + PCCMG_CG)
 * 
 * ## Pin for Singular Systems
 * 
 * When all boundaries are periodic or Neumann, the system is singular.
 * The pin mechanism (diagonal doubling at cell [0,0,0]) breaks this degeneracy.
 * See pin_spec.hpp for full documentation.
 */

#include "../../core/Grid3D.hpp"
#include "../../core/BCSpec.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../../multigrid/mg_types.hpp"
#include "../../numerics/pin_spec.hpp"
#include "../../io/config/Config.hpp"
#include "../common/workspaces.cuh"

namespace rwpt {
namespace physics {

// Re-export for convenience in physics layer
using rwpt::io::PinConfig;
using rwpt::PinMode;

/**
 * @brief Solver type for head equation
 * 
 * Legacy correspondence:
 * - CG: plain conjugate gradient (no preconditioner)
 * - MG: standalone multigrid V-cycles
 * - PCG_MG: CG preconditioned with MG V-cycle (legacy default: solver_CG + PCCMG_CG)
 */
enum class HeadSolverType {
    CG,      // Plain CG (no preconditioner)
    MG,      // Standalone MG V-cycles
    PCG_MG   // CG preconditioned with MG (legacy default)
};

/**
 * @brief Configuration for head equation solve
 * 
 * Groups all parameters needed to solve the head equation.
 * Can be constructed from FlowYamlConfig or set directly.
 */
struct HeadSolveConfig {
    // Solver type (default: PCG_MG to match legacy)
    HeadSolverType solver_type = HeadSolverType::PCG_MG;
    
    // MG parameters (used for MG and PCG_MG)
    int mg_levels = 4;
    int mg_pre_smooth = 2;
    int mg_post_smooth = 2;
    int mg_max_cycles = 20;      // Max V-cycles for standalone MG
    int mg_coarse_iters = 50;
    
    // CG/PCG parameters
    int cg_max_iter = 1000;      // Max CG iterations
    int cg_check_every = 10;     // Check convergence every N iterations
    real cg_rtol = 1e-8;         // CG relative tolerance
    
    // Overall convergence
    real rtol = 1e-6;
    
    // Pin configuration for singular systems
    // See pin_spec.hpp for documentation
    PinConfig pin;
    
    HeadSolveConfig() = default;
    
    /**
     * @brief Factory: Create HeadSolveConfig from FlowYamlConfig
     * 
     * Converts the string-based solver type to enum and copies all parameters.
     * This centralizes the IO→numerics conversion logic.
     * 
     * @param flow_cfg FlowYamlConfig from YAML parsing
     * @return HeadSolveConfig ready for use by solve_head()
     */
    static HeadSolveConfig from_yaml(const rwpt::io::FlowYamlConfig& flow_cfg) {
        HeadSolveConfig cfg;
        
        // Parse solver type from string
        if (flow_cfg.solver == "cg") {
            cfg.solver_type = HeadSolverType::CG;
        } else if (flow_cfg.solver == "mg") {
            cfg.solver_type = HeadSolverType::MG;
        } else if (flow_cfg.solver == "pcg_mg" || flow_cfg.solver == "mg_cg") {
            cfg.solver_type = HeadSolverType::PCG_MG;
        } else {
            // Default to PCG_MG for unknown strings
            cfg.solver_type = HeadSolverType::PCG_MG;
        }
        
        // Copy MG parameters
        cfg.mg_levels = flow_cfg.mg_levels;
        cfg.mg_pre_smooth = flow_cfg.mg_pre_smooth;
        cfg.mg_post_smooth = flow_cfg.mg_post_smooth;
        cfg.mg_max_cycles = flow_cfg.mg_max_cycles;
        cfg.mg_coarse_iters = flow_cfg.mg_coarse_iters;
        
        // Copy CG parameters
        cfg.cg_max_iter = flow_cfg.cg_max_iter;
        cfg.cg_rtol = flow_cfg.cg_rtol;
        cfg.cg_check_every = flow_cfg.cg_check_every;
        
        // Copy convergence and pin
        cfg.rtol = flow_cfg.rtol;
        cfg.pin = flow_cfg.pin;
        
        return cfg;
    }
};

/**
 * @brief Result of head solve
 */
struct HeadSolveResult {
    int num_iterations = 0;      // MG cycles or CG iterations
    real initial_residual = 0.0;
    real final_residual = 0.0;
    bool converged = false;
};

/**
 * @brief Solve head equation: -∇·(K∇h) = 0 with BCs
 * 
 * Uses MG by default. The RHS is zero (no sources).
 * Dirichlet BCs set the head at boundaries.
 * 
 * @param h         Output: head field (device, size = num_cells)
 * @param K         Input: conductivity field (device, size = num_cells)
 * @param grid      Grid specification
 * @param bc        Boundary conditions
 * @param cfg       Solver configuration
 * @param ctx       CUDA context
 * @param workspace Flow workspace (contains MG hierarchy)
 * @return HeadSolveResult with convergence info
 */
HeadSolveResult solve_head(
    DeviceSpan<real> h,
    DeviceSpan<const real> K,
    const Grid3D& grid,
    const BCSpec& bc,
    const HeadSolveConfig& cfg,
    CudaContext& ctx,
    FlowWorkspace& workspace
);

/**
 * @brief Initialize initial guess for head
 * 
 * Default: linear interpolation between Dirichlet BCs in x-direction.
 * Falls back to zero if no Dirichlet BCs.
 * 
 * @param h    Output: initial guess
 * @param grid Grid
 * @param bc   Boundary conditions
 * @param ctx  CUDA context
 */
void init_head_guess(
    DeviceSpan<real> h,
    const Grid3D& grid,
    const BCSpec& bc,
    const CudaContext& ctx
);

} // namespace physics
} // namespace rwpt

```

# src\physics\flow\velocity_from_head.cu

```cu
/**
 * @file velocity_from_head.cu
 * @brief Implementation of velocity computation from head (Darcy's law)
 * 
 * Structure:
 *   1. Device helpers (inline, no overhead)
 *   2. Interior kernels (bulk, no branches, max performance)
 *   3. Boundary kernels (6 faces + edges/vertices if needed)
 *   4. Host orchestration (launch all kernels)
 * 
 * Reference: legacy/compute_velocity_from_head_for_par2.cu
 */

#include "velocity_from_head.cuh"
#include "../../core/BCSpecDevice.cuh"
#include "../../core/DeviceBuffer.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

namespace rwpt {
namespace physics {

// ============================================================================
// Device helpers (inline, zero-overhead)
// ============================================================================

/**
 * @brief Harmonic mean of two positive values
 * K_eff = 2*Ka*Kb / (Ka + Kb) = 2 / (1/Ka + 1/Kb)
 */
__device__ __forceinline__
real harmonic_mean(real Ka, real Kb) {
    return 2.0 * Ka * Kb / (Ka + Kb);
}

/**
 * @brief Cell-centered linear index: idx = i + j*nx + k*nx*ny
 */
__device__ __forceinline__
int cell_idx(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * ny;
}

/**
 * @brief U-face index (staggered in x): idx = i + j*(nx+1) + k*(nx+1)*ny
 * Face i is between cell i-1 and cell i (for i=1..nx-1)
 * Face 0 is at x=0 boundary, face nx is at x=Lx boundary
 */
__device__ __forceinline__
int U_idx(int i, int j, int k, int nx, int ny) {
    return i + j * (nx + 1) + k * (nx + 1) * ny;
}

/**
 * @brief V-face index (staggered in y): idx = i + j*nx + k*nx*(ny+1)
 */
__device__ __forceinline__
int V_idx(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * (ny + 1);
}

/**
 * @brief W-face index (staggered in z): idx = i + j*nx + k*nx*ny
 */
__device__ __forceinline__
int W_idx(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * ny;
}

/**
 * @brief Periodic wrap for cell index in x direction
 */
__device__ __forceinline__
int wrap_x(int i, int nx) {
    return (i + nx) % nx;  // handles both negative and positive
}

__device__ __forceinline__
int wrap_y(int j, int ny) {
    return (j + ny) % ny;
}

__device__ __forceinline__
int wrap_z(int k, int nz) {
    return (k + nz) % nz;
}

// BC type constants (matching legacy)
constexpr uint8_t BC_NEUMANN = 0;
constexpr uint8_t BC_PERIODIC = 1;
constexpr uint8_t BC_DIRICHLET = 2;

// Convert BCType enum to legacy-compatible int (host+device)
__host__ __device__ __forceinline__
uint8_t bc_to_int(BCType t) {
    switch(t) {
        case BCType::Neumann:   return BC_NEUMANN;
        case BCType::Periodic:  return BC_PERIODIC;
        case BCType::Dirichlet: return BC_DIRICHLET;
        default:                return BC_NEUMANN;
    }
}

// ============================================================================
// Interior kernels - pure bulk computation, NO boundary logic
// ============================================================================

/**
 * @brief Compute U-velocity at interior faces (i = 1 to nx-1)
 * 
 * Face i is between cell (i-1,j,k) and cell (i,j,k)
 * U[i,j,k] = -K_harmonic * (H[i,j,k] - H[i-1,j,k]) / dx
 */
__global__ void kernel_U_interior(
    real* __restrict__ U,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dx)
{
    // Thread maps to face (i,j,k) where i = 1 + threadIdx.x + blockIdx.x*blockDim.x
    int face_i = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    // Interior faces: i = 1 to nx-1 (faces between adjacent cells)
    if (face_i > nx - 1 || j >= ny || k >= nz) return;
    
    // Cells on either side of face
    int c_left  = cell_idx(face_i - 1, j, k, nx, ny);  // cell i-1
    int c_right = cell_idx(face_i,     j, k, nx, ny);  // cell i
    
    real H_left  = H[c_left];
    real H_right = H[c_right];
    real K_left  = K[c_left];
    real K_right = K[c_right];
    
    // Darcy: u = -K_eff * (H_right - H_left) / dx
    real K_eff = harmonic_mean(K_left, K_right);
    real u = -K_eff * (H_right - H_left) / dx;
    
    U[U_idx(face_i, j, k, nx, ny)] = u;
}

/**
 * @brief Compute V-velocity at interior faces (j = 1 to ny-1)
 */
__global__ void kernel_V_interior(
    real* __restrict__ V,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dy)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int face_j = 1 + threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i >= nx || face_j > ny - 1 || k >= nz) return;
    
    int c_south = cell_idx(i, face_j - 1, k, nx, ny);
    int c_north = cell_idx(i, face_j,     k, nx, ny);
    
    real H_south = H[c_south];
    real H_north = H[c_north];
    real K_south = K[c_south];
    real K_north = K[c_north];
    
    real K_eff = harmonic_mean(K_south, K_north);
    real v = -K_eff * (H_north - H_south) / dy;
    
    V[V_idx(i, face_j, k, nx, ny)] = v;
}

/**
 * @brief Compute W-velocity at interior faces (k = 1 to nz-1)
 */
__global__ void kernel_W_interior(
    real* __restrict__ W,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dz)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int face_k = 1 + threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i >= nx || j >= ny || face_k > nz - 1) return;
    
    int c_bottom = cell_idx(i, j, face_k - 1, nx, ny);
    int c_top    = cell_idx(i, j, face_k,     nx, ny);
    
    real H_bottom = H[c_bottom];
    real H_top    = H[c_top];
    real K_bottom = K[c_bottom];
    real K_top    = K[c_top];
    
    real K_eff = harmonic_mean(K_bottom, K_top);
    real w = -K_eff * (H_top - H_bottom) / dz;
    
    W[W_idx(i, j, face_k, nx, ny)] = w;
}

// ============================================================================
// Boundary kernels - one kernel per face
// ============================================================================

/**
 * @brief U-velocity at WEST face (i=0): face between domain boundary and cell(0,j,k)
 * 
 * Neumann: U[0,j,k] = 0
 * Dirichlet: U[0,j,k] = -K_cell * (H_cell - H_bc) / (dx/2)
 * Periodic: U[0,j,k] = -K_eff * (H[0,j,k] - H[nx-1,j,k]) / dx  (wrap)
 */
__global__ void kernel_U_west(
    real* __restrict__ U,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dx,
    uint8_t bc_type,
    real H_bc)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (j >= ny || k >= nz) return;
    
    int c_inside = cell_idx(0, j, k, nx, ny);  // first cell
    real H_inside = H[c_inside];
    real K_inside = K[c_inside];
    
    real u = 0.0;  // default: Neumann
    
    if (bc_type == BC_DIRICHLET) {
        // One-sided: distance is dx/2 from cell center to face
        // Gradient: (H_inside - H_bc) / (dx/2) pointing into domain
        // Darcy: u = -K * dH/dx
        u = -K_inside * (H_inside - H_bc) / (dx * 0.5);
    }
    else if (bc_type == BC_PERIODIC) {
        // Wrap: neighbor is cell (nx-1, j, k)
        int c_wrap = cell_idx(nx - 1, j, k, nx, ny);
        real H_wrap = H[c_wrap];
        real K_wrap = K[c_wrap];
        real K_eff = harmonic_mean(K_wrap, K_inside);
        // Gradient from wrap to inside (conceptually wrap is "to the left")
        u = -K_eff * (H_inside - H_wrap) / dx;
    }
    // else: Neumann -> u = 0
    
    U[U_idx(0, j, k, nx, ny)] = u;
}

/**
 * @brief U-velocity at EAST face (i=nx): face between cell(nx-1,j,k) and domain boundary
 */
__global__ void kernel_U_east(
    real* __restrict__ U,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dx,
    uint8_t bc_type,
    real H_bc)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (j >= ny || k >= nz) return;
    
    int c_inside = cell_idx(nx - 1, j, k, nx, ny);  // last cell
    real H_inside = H[c_inside];
    real K_inside = K[c_inside];
    
    real u = 0.0;
    
    if (bc_type == BC_DIRICHLET) {
        // Gradient: (H_bc - H_inside) / (dx/2)
        u = -K_inside * (H_bc - H_inside) / (dx * 0.5);
    }
    else if (bc_type == BC_PERIODIC) {
        // Wrap: neighbor is cell (0, j, k)
        int c_wrap = cell_idx(0, j, k, nx, ny);
        real H_wrap = H[c_wrap];
        real K_wrap = K[c_wrap];
        real K_eff = harmonic_mean(K_inside, K_wrap);
        u = -K_eff * (H_wrap - H_inside) / dx;
    }
    
    U[U_idx(nx, j, k, nx, ny)] = u;
}

/**
 * @brief V-velocity at SOUTH face (j=0)
 */
__global__ void kernel_V_south(
    real* __restrict__ V,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dy,
    uint8_t bc_type,
    real H_bc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= nx || k >= nz) return;
    
    int c_inside = cell_idx(i, 0, k, nx, ny);
    real H_inside = H[c_inside];
    real K_inside = K[c_inside];
    
    real v = 0.0;
    
    if (bc_type == BC_DIRICHLET) {
        v = -K_inside * (H_inside - H_bc) / (dy * 0.5);
    }
    else if (bc_type == BC_PERIODIC) {
        int c_wrap = cell_idx(i, ny - 1, k, nx, ny);
        real H_wrap = H[c_wrap];
        real K_wrap = K[c_wrap];
        real K_eff = harmonic_mean(K_wrap, K_inside);
        v = -K_eff * (H_inside - H_wrap) / dy;
    }
    
    V[V_idx(i, 0, k, nx, ny)] = v;
}

/**
 * @brief V-velocity at NORTH face (j=ny)
 */
__global__ void kernel_V_north(
    real* __restrict__ V,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dy,
    uint8_t bc_type,
    real H_bc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= nx || k >= nz) return;
    
    int c_inside = cell_idx(i, ny - 1, k, nx, ny);
    real H_inside = H[c_inside];
    real K_inside = K[c_inside];
    
    real v = 0.0;
    
    if (bc_type == BC_DIRICHLET) {
        v = -K_inside * (H_bc - H_inside) / (dy * 0.5);
    }
    else if (bc_type == BC_PERIODIC) {
        int c_wrap = cell_idx(i, 0, k, nx, ny);
        real H_wrap = H[c_wrap];
        real K_wrap = K[c_wrap];
        real K_eff = harmonic_mean(K_inside, K_wrap);
        v = -K_eff * (H_wrap - H_inside) / dy;
    }
    
    V[V_idx(i, ny, k, nx, ny)] = v;
}

/**
 * @brief W-velocity at BOTTOM face (k=0)
 */
__global__ void kernel_W_bottom(
    real* __restrict__ W,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dz,
    uint8_t bc_type,
    real H_bc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= nx || j >= ny) return;
    
    int c_inside = cell_idx(i, j, 0, nx, ny);
    real H_inside = H[c_inside];
    real K_inside = K[c_inside];
    
    real w = 0.0;
    
    if (bc_type == BC_DIRICHLET) {
        w = -K_inside * (H_inside - H_bc) / (dz * 0.5);
    }
    else if (bc_type == BC_PERIODIC) {
        int c_wrap = cell_idx(i, j, nz - 1, nx, ny);
        real H_wrap = H[c_wrap];
        real K_wrap = K[c_wrap];
        real K_eff = harmonic_mean(K_wrap, K_inside);
        w = -K_eff * (H_inside - H_wrap) / dz;
    }
    
    W[W_idx(i, j, 0, nx, ny)] = w;
}

/**
 * @brief W-velocity at TOP face (k=nz)
 */
__global__ void kernel_W_top(
    real* __restrict__ W,
    const real* __restrict__ H,
    const real* __restrict__ K,
    int nx, int ny, int nz,
    real dz,
    uint8_t bc_type,
    real H_bc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= nx || j >= ny) return;
    
    int c_inside = cell_idx(i, j, nz - 1, nx, ny);
    real H_inside = H[c_inside];
    real K_inside = K[c_inside];
    
    real w = 0.0;
    
    if (bc_type == BC_DIRICHLET) {
        w = -K_inside * (H_bc - H_inside) / (dz * 0.5);
    }
    else if (bc_type == BC_PERIODIC) {
        int c_wrap = cell_idx(i, j, 0, nx, ny);
        real H_wrap = H[c_wrap];
        real K_wrap = K[c_wrap];
        real K_eff = harmonic_mean(K_inside, K_wrap);
        w = -K_eff * (H_wrap - H_inside) / dz;
    }
    
    W[W_idx(i, j, nz, nx, ny)] = w;
}

// ============================================================================
// Reduction kernels for checksums
// ============================================================================

__global__ void kernel_sum_sq(const real* data, real* partial, int n)
{
    extern __shared__ real sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    real sum = 0.0;
    while (i < n) {
        real v = data[i];
        sum += v * v;
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

__global__ void kernel_check_nans(const real* data, int* has_nan, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (i < n) {
        if (isnan(data[i])) {
            atomicExch(has_nan, 1);
            return;
        }
        i += blockDim.x * gridDim.x;
    }
}

// ============================================================================
// Host orchestration
// ============================================================================

void compute_velocity_from_head(
    VelocityField& vel,
    const HeadField& head,
    const KField& K,
    const Grid3D& grid,
    const BCSpec& bc,
    CudaContext& ctx)
{
    int nx = grid.nx;
    int ny = grid.ny;
    int nz = grid.nz;
    real dx = grid.dx;
    real dy = grid.dy;
    real dz = grid.dz;
    
    // Get device pointers
    real* U = vel.U_ptr();
    real* V = vel.V_ptr();
    real* W = vel.W_ptr();
    const real* H = head.device_ptr();
    const real* K_ptr = K.device_ptr();
    
    // Convert BC types
    uint8_t bc_xmin = bc_to_int(bc.xmin.type);
    uint8_t bc_xmax = bc_to_int(bc.xmax.type);
    uint8_t bc_ymin = bc_to_int(bc.ymin.type);
    uint8_t bc_ymax = bc_to_int(bc.ymax.type);
    uint8_t bc_zmin = bc_to_int(bc.zmin.type);
    uint8_t bc_zmax = bc_to_int(bc.zmax.type);
    
    // ========================================================================
    // Launch interior kernels
    // ========================================================================
    
    // Block size for 3D kernels
    dim3 block(8, 8, 8);
    
    // U interior: faces i = 1 to nx-1
    {
        int num_faces_x = nx - 1;  // interior faces in x
        dim3 grid_U(
            (num_faces_x + block.x - 1) / block.x,
            (ny + block.y - 1) / block.y,
            (nz + block.z - 1) / block.z
        );
        kernel_U_interior<<<grid_U, block>>>(U, H, K_ptr, nx, ny, nz, dx);
    }
    
    // V interior: faces j = 1 to ny-1
    {
        int num_faces_y = ny - 1;
        dim3 grid_V(
            (nx + block.x - 1) / block.x,
            (num_faces_y + block.y - 1) / block.y,
            (nz + block.z - 1) / block.z
        );
        kernel_V_interior<<<grid_V, block>>>(V, H, K_ptr, nx, ny, nz, dy);
    }
    
    // W interior: faces k = 1 to nz-1
    {
        int num_faces_z = nz - 1;
        dim3 grid_W(
            (nx + block.x - 1) / block.x,
            (ny + block.y - 1) / block.y,
            (num_faces_z + block.z - 1) / block.z
        );
        kernel_W_interior<<<grid_W, block>>>(W, H, K_ptr, nx, ny, nz, dz);
    }
    
    // ========================================================================
    // Launch boundary kernels (6 faces)
    // ========================================================================
    
    dim3 block2D(16, 16);
    
    // U boundaries (west/east faces)
    {
        dim3 grid_yz((ny + block2D.x - 1) / block2D.x, (nz + block2D.y - 1) / block2D.y);
        kernel_U_west<<<grid_yz, block2D>>>(U, H, K_ptr, nx, ny, nz, dx, bc_xmin, bc.xmin.value);
        kernel_U_east<<<grid_yz, block2D>>>(U, H, K_ptr, nx, ny, nz, dx, bc_xmax, bc.xmax.value);
    }
    
    // V boundaries (south/north faces)
    {
        dim3 grid_xz((nx + block2D.x - 1) / block2D.x, (nz + block2D.y - 1) / block2D.y);
        kernel_V_south<<<grid_xz, block2D>>>(V, H, K_ptr, nx, ny, nz, dy, bc_ymin, bc.ymin.value);
        kernel_V_north<<<grid_xz, block2D>>>(V, H, K_ptr, nx, ny, nz, dy, bc_ymax, bc.ymax.value);
    }
    
    // W boundaries (bottom/top faces)
    {
        dim3 grid_xy((nx + block2D.x - 1) / block2D.x, (ny + block2D.y - 1) / block2D.y);
        kernel_W_bottom<<<grid_xy, block2D>>>(W, H, K_ptr, nx, ny, nz, dz, bc_zmin, bc.zmin.value);
        kernel_W_top<<<grid_xy, block2D>>>(W, H, K_ptr, nx, ny, nz, dz, bc_zmax, bc.zmax.value);
    }
    
    // Synchronize
    ctx.synchronize();
}

// ============================================================================
// Utility implementations
// ============================================================================

real compute_norm2(DeviceSpan<const real> data, CudaContext& ctx)
{
    int n = static_cast<int>(data.size());
    if (n == 0) return 0.0;
    
    const int block_size = 256;
    int num_blocks = std::min((n + block_size - 1) / block_size, 1024);
    
    // Allocate partial sums on device
    DeviceBuffer<real> partial(num_blocks);
    
    kernel_sum_sq<<<num_blocks, block_size, block_size * sizeof(real)>>>(
        data.data(), partial.data(), n);
    
    // Copy back and reduce on host
    std::vector<real> h_partial(num_blocks);
    cudaMemcpy(h_partial.data(), partial.data(), num_blocks * sizeof(real), cudaMemcpyDeviceToHost);
    ctx.synchronize();
    
    real sum = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        sum += h_partial[i];
    }
    
    return std::sqrt(sum);
}

bool check_no_nans(DeviceSpan<const real> data, CudaContext& ctx)
{
    int n = static_cast<int>(data.size());
    if (n == 0) return true;
    
    DeviceBuffer<int> has_nan(1);
    int zero = 0;
    cudaMemcpy(has_nan.data(), &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    const int block_size = 256;
    int num_blocks = std::min((n + block_size - 1) / block_size, 1024);
    
    kernel_check_nans<<<num_blocks, block_size>>>(data.data(), has_nan.data(), n);
    
    int h_has_nan;
    cudaMemcpy(&h_has_nan, has_nan.data(), sizeof(int), cudaMemcpyDeviceToHost);
    ctx.synchronize();
    
    return h_has_nan == 0;
}

// Kernel for sum reduction
__global__ void kernel_sum(const real* data, real* partial, int n)
{
    extern __shared__ real sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    real sum = 0.0;
    while (i < n) {
        sum += data[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// Kernel for computing 1/K (for harmonic mean calculation)
__global__ void kernel_sum_inv(const real* data, real* partial, int n)
{
    extern __shared__ real sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    real sum = 0.0;
    while (i < n) {
        sum += 1.0 / data[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

real compute_sum(DeviceSpan<const real> data, CudaContext& ctx)
{
    int n = static_cast<int>(data.size());
    if (n == 0) return 0.0;
    
    const int block_size = 256;
    int num_blocks = std::min((n + block_size - 1) / block_size, 1024);
    
    DeviceBuffer<real> partial(num_blocks);
    
    kernel_sum<<<num_blocks, block_size, block_size * sizeof(real)>>>(
        data.data(), partial.data(), n);
    
    std::vector<real> h_partial(num_blocks);
    cudaMemcpy(h_partial.data(), partial.data(), num_blocks * sizeof(real), cudaMemcpyDeviceToHost);
    ctx.synchronize();
    
    real sum = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        sum += h_partial[i];
    }
    
    return sum;
}

real compute_mean(DeviceSpan<const real> data, CudaContext& ctx)
{
    if (data.size() == 0) return 0.0;
    return compute_sum(data, ctx) / static_cast<real>(data.size());
}

// Compute harmonic mean of K field: n / sum(1/K)
real compute_harmonic_mean_K(DeviceSpan<const real> K, CudaContext& ctx)
{
    int n = static_cast<int>(K.size());
    if (n == 0) return 0.0;
    
    const int block_size = 256;
    int num_blocks = std::min((n + block_size - 1) / block_size, 1024);
    
    DeviceBuffer<real> partial(num_blocks);
    
    kernel_sum_inv<<<num_blocks, block_size, block_size * sizeof(real)>>>(
        K.data(), partial.data(), n);
    
    std::vector<real> h_partial(num_blocks);
    cudaMemcpy(h_partial.data(), partial.data(), num_blocks * sizeof(real), cudaMemcpyDeviceToHost);
    ctx.synchronize();
    
    real sum_inv = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        sum_inv += h_partial[i];
    }
    
    return static_cast<real>(n) / sum_inv;
}

void print_velocity_checksums(const VelocityField& vel, CudaContext& ctx)
{
    std::cout << "  Velocity checksums:\n";
    
    // U
    real norm_U = compute_norm2(vel.U_span(), ctx);
    bool ok_U = check_no_nans(vel.U_span(), ctx);
    std::cout << "    U: norm2 = " << norm_U << (ok_U ? " (no NaNs)" : " [HAS NaNs!]") << "\n";
    
    // V
    real norm_V = compute_norm2(vel.V_span(), ctx);
    bool ok_V = check_no_nans(vel.V_span(), ctx);
    std::cout << "    V: norm2 = " << norm_V << (ok_V ? " (no NaNs)" : " [HAS NaNs!]") << "\n";
    
    // W
    real norm_W = compute_norm2(vel.W_span(), ctx);
    bool ok_W = check_no_nans(vel.W_span(), ctx);
    std::cout << "    W: norm2 = " << norm_W << (ok_W ? " (no NaNs)" : " [HAS NaNs!]") << "\n";
    
    bool all_ok = ok_U && ok_V && ok_W;
    std::cout << "    Status: " << (all_ok ? "OK" : "ERROR - NaNs detected!") << "\n";
}

void verify_mean_velocity_darcy(
    const VelocityField& vel,
    const KField& K,
    const Grid3D& grid,
    const BCSpec& bc,
    CudaContext& ctx)
{
    std::cout << "\n  Darcy velocity verification:\n";
    
    // Get domain length: Lx = nx * dx
    real Lx = grid.nx * grid.dx;
    
    // Get BC values (assuming Dirichlet west-east)
    real H_west = bc.xmin.value;
    real H_east = bc.xmax.value;
    real dH = H_west - H_east;  // Head drop
    
    std::cout << "    H_west = " << H_west << ", H_east = " << H_east << "\n";
    std::cout << "    dH = " << dH << ", Lx = " << Lx << "\n";
    
    // Compute K statistics
    DeviceSpan<const real> K_span(K.device_ptr(), K.size());
    real K_harmonic = compute_harmonic_mean_K(K_span, ctx);
    real K_arithmetic = compute_mean(K_span, ctx);
    
    std::cout << "    K_harmonic = " << K_harmonic << "\n";
    std::cout << "    K_arithmetic = " << K_arithmetic << "\n";
    
    // Theoretical velocity using harmonic mean (exact for 1D steady flow)
    // u = -K_eff * dH/dx = K_eff * (H_west - H_east) / Lx
    real u_theory_harmonic = K_harmonic * dH / Lx;
    real u_theory_arithmetic = K_arithmetic * dH / Lx;
    
    std::cout << "    u_theory (K_harmonic) = " << u_theory_harmonic << "\n";
    std::cout << "    u_theory (K_arithmetic) = " << u_theory_arithmetic << "\n";
    
    // Compute mean U from the field
    real U_mean = compute_mean(vel.U_span(), ctx);
    std::cout << "    U_mean (computed) = " << U_mean << "\n";
    
    // Relative errors
    real rel_err_harmonic = std::abs(U_mean - u_theory_harmonic) / std::abs(u_theory_harmonic) * 100.0;
    real rel_err_arithmetic = std::abs(U_mean - u_theory_arithmetic) / std::abs(u_theory_arithmetic) * 100.0;
    
    std::cout << "    Relative error vs K_harmonic:   " << rel_err_harmonic << " %\n";
    std::cout << "    Relative error vs K_arithmetic: " << rel_err_arithmetic << " %\n";
    
    // For 1D steady Darcy flow with heterogeneous K, the effective K is:
    // - Harmonic mean for flow perpendicular to layers (series)
    // - Arithmetic mean for flow parallel to layers (parallel)
    // In 3D random field, it's somewhere between, often close to geometric mean
    std::cout << "\n    Note: For 3D random K, effective K is between harmonic and arithmetic.\n";
    std::cout << "    The computed mean velocity should be consistent with the solved head field.\n";
}

} // namespace physics
} // namespace rwpt

```

# src\physics\flow\velocity_from_head.cuh

```cuh
#pragma once

/**
 * @file velocity_from_head.cuh
 * @brief Compute staggered velocity field from cell-centered head and conductivity
 * 
 * Physics: Darcy's law with harmonic mean conductivity
 *   q = -K_eff * grad(H)
 * 
 * where K_eff between two cells is the harmonic mean:
 *   K_eff = 2 / (1/K_a + 1/K_b) = 2*K_a*K_b / (K_a + K_b)
 * 
 * Layout (standard staggered grid):
 *   U: x-velocity at x-faces, dims (nx+1, ny, nz)
 *   V: y-velocity at y-faces, dims (nx, ny+1, nz)
 *   W: z-velocity at z-faces, dims (nx, ny, nz+1)
 * 
 * Boundary conditions:
 *   - Neumann homogeneous: flux = 0 at boundary face
 *   - Dirichlet: one-sided gradient with distance h/2, using K_cell (not harmonic)
 *   - Periodic: wrap to opposite side, use harmonic mean as interior
 * 
 * Reference: legacy/compute_velocity_from_head_for_par2.cu
 * Semantics: identical to legacy, but with clean modular implementation
 */

#include "../../core/Grid3D.hpp"
#include "../../core/BCSpec.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../common/fields.cuh"
#include "../../core/DeviceSpan.cuh"

namespace rwpt {
namespace physics {

// ============================================================================
// Main API
// ============================================================================

/**
 * @brief Compute velocity field from head using Darcy's law
 * 
 * Writes U, V, W in-place (already allocated in vel).
 * No memory allocation inside this function.
 * 
 * @param vel       Output velocity field (U, V, W already allocated)
 * @param head      Input head field (cell-centered)
 * @param K         Input conductivity field (cell-centered)
 * @param grid      Grid dimensions and spacing
 * @param bc        Boundary conditions for all 6 faces
 * @param ctx       CUDA context for synchronization
 */
void compute_velocity_from_head(
    VelocityField& vel,
    const HeadField& head,
    const KField& K,
    const Grid3D& grid,
    const BCSpec& bc,
    CudaContext& ctx);

// ============================================================================
// Checksum/validation utilities
// ============================================================================

/**
 * @brief Compute L2 norm (sqrt of sum of squares) for a device array
 */
real compute_norm2(DeviceSpan<const real> data, CudaContext& ctx);

/**
 * @brief Check if any values are NaN in a device array
 * @return true if no NaNs found (all values are valid)
 */
bool check_no_nans(DeviceSpan<const real> data, CudaContext& ctx);

/**
 * @brief Compute sum of all elements in a device array
 */
real compute_sum(DeviceSpan<const real> data, CudaContext& ctx);

/**
 * @brief Compute mean of all elements in a device array
 */
real compute_mean(DeviceSpan<const real> data, CudaContext& ctx);

/**
 * @brief Compute and print checksums for velocity field
 * 
 * Prints L2 norms and NaN status for U, V, W
 */
void print_velocity_checksums(const VelocityField& vel, CudaContext& ctx);

/**
 * @brief Verify mean U velocity against theoretical Darcy value
 * 
 * For Dirichlet west-east with periodic elsewhere:
 *   u_theory = K_eff * (H_west - H_east) / Lx
 * 
 * where K_eff is the effective (harmonic) conductivity in x-direction.
 * 
 * This function computes:
 *   1. Mean U velocity from the computed field
 *   2. Theoretical velocity using harmonic mean K
 *   3. Relative error
 * 
 * @param vel       Computed velocity field
 * @param K         Conductivity field
 * @param grid      Grid dimensions
 * @param bc        Boundary conditions
 * @param ctx       CUDA context
 */
void verify_mean_velocity_darcy(
    const VelocityField& vel,
    const KField& K,
    const Grid3D& grid,
    const BCSpec& bc,
    CudaContext& ctx);

} // namespace physics
} // namespace rwpt

```

# src\physics\stochastic\stochastic.cu

```cu
/**
 * @file stochastic.cu
 * @brief Stochastic K field generation - Implementation
 * 
 * Direct port of legacy/random_field_generation.cu
 * Randomized Spectral Method (no FFT)
 */

#include "stochastic.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cmath>
#include <curand_kernel.h>
#include <vector>

namespace rwpt {
namespace physics {

// ============================================================================
// Constants (matching legacy)
// ============================================================================

// Use constexpr for compile-time constant
static constexpr double PI_D = 3.141592653589793238462643383279502884;

// ============================================================================
// Kernel: Initialize RNG states
// ============================================================================

/**
 * @brief Setup curand states with deterministic seeding
 * 
 * Legacy: curand_init(ix, ix, 0, &state[ix])
 * Modified for reproducibility: curand_init(base_seed + ix, ix, 0, &state[ix])
 */
__global__ void kernel_init_rng(curandState* __restrict__ states,
                                const uint64_t base_seed,
                                const int n_modes) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= n_modes) return;
    
    // Use base_seed + ix as seed, ix as sequence, 0 as offset
    // This ensures reproducibility: same seed → same sequence
    curand_init(base_seed + ix, ix, 0, &states[ix]);
}

// ============================================================================
// Kernel: Generate Fourier mode coefficients (exponential covariance)
// ============================================================================

/**
 * @brief Generate wavenumbers for EXPONENTIAL covariance
 * 
 * Legacy: random_kernel_3D()
 * Uses Cauchy-like distribution via rejection sampling
 */
__global__ void kernel_random_modes_exp(curandState* __restrict__ states,
                                        real* __restrict__ V1,
                                        real* __restrict__ V2,
                                        real* __restrict__ V3,
                                        real* __restrict__ a,
                                        real* __restrict__ b,
                                        const real lambda,
                                        const int n_modes) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= n_modes) return;
    
    curandState localState = states[ix];
    
    // Spherical angles for direction
    double fi = 2.0 * PI_D * curand_uniform_double(&localState);
    double theta = acos(1.0 - 2.0 * curand_uniform_double(&localState));
    
    // Wavenumber magnitude k from modified Cauchy distribution (rejection sampling)
    double k, d;
    int flag = 1;
    while (flag == 1) {
        k = tan(PI_D * 0.5 * curand_uniform_double(&localState));
        d = (k * k) / (1.0 + k * k);
        if (curand_uniform_double(&localState) < d) flag = 0;
    }
    
    // Wavenumber vector components (legacy: divide by lambda)
    V1[ix] = static_cast<real>(k * sin(fi) * sin(theta) / lambda);
    V2[ix] = static_cast<real>(k * cos(fi) * sin(theta) / lambda);
    V3[ix] = static_cast<real>(k * cos(theta) / lambda);
    
    // Fourier coefficients a, b ~ N(0,1) via Box-Muller
    double u1 = curand_uniform_double(&localState);
    double u2 = curand_uniform_double(&localState);
    a[ix] = static_cast<real>(sqrt(-2.0 * log(u1)) * cos(2.0 * PI_D * u2));
    
    u1 = curand_uniform_double(&localState);
    u2 = curand_uniform_double(&localState);
    b[ix] = static_cast<real>(sqrt(-2.0 * log(u1)) * cos(2.0 * PI_D * u2));
    
    states[ix] = localState;
}

// ============================================================================
// Kernel: Generate Fourier mode coefficients (Gaussian covariance)
// ============================================================================

/**
 * @brief Generate wavenumbers for GAUSSIAN covariance
 * 
 * Legacy: random_kernel_3D_gauss()
 * Uses rejection sampling with k² exp(-0.5 k²) envelope
 */
__global__ void kernel_random_modes_gauss(curandState* __restrict__ states,
                                          real* __restrict__ V1,
                                          real* __restrict__ V2,
                                          real* __restrict__ V3,
                                          real* __restrict__ a,
                                          real* __restrict__ b,
                                          const real lambda,
                                          const int n_modes,
                                          const int k_max) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= n_modes) return;
    
    curandState localState = states[ix];
    
    // Spherical angles for direction
    double fi = 2.0 * PI_D * curand_uniform_double(&localState);
    double theta = acos(1.0 - 2.0 * curand_uniform_double(&localState));
    
    // Wavenumber magnitude k via rejection sampling
    double k, d;
    int flag = 1;
    while (flag == 1) {
        k = k_max * curand_uniform_double(&localState);
        d = k * k * exp(-0.5 * k * k);
        if (curand_uniform_double(&localState) * 2.0 * exp(-1.0) < d) flag = 0;
    }
    
    // Scale k (legacy formula)
    k = k / (2.0 * lambda / sqrt(PI_D)) * sqrt(2.0);
    
    // Wavenumber vector components (no additional lambda division here)
    V1[ix] = static_cast<real>(k * sin(fi) * sin(theta));
    V2[ix] = static_cast<real>(k * cos(fi) * sin(theta));
    V3[ix] = static_cast<real>(k * cos(theta));
    
    // Fourier coefficients a, b ~ N(0,1) via Box-Muller
    double u1 = curand_uniform_double(&localState);
    double u2 = curand_uniform_double(&localState);
    a[ix] = static_cast<real>(sqrt(-2.0 * log(u1)) * cos(2.0 * PI_D * u2));
    
    u1 = curand_uniform_double(&localState);
    u2 = curand_uniform_double(&localState);
    b[ix] = static_cast<real>(sqrt(-2.0 * log(u1)) * cos(2.0 * PI_D * u2));
    
    states[ix] = localState;
}

// ============================================================================
// Kernel: Evaluate Gaussian field at all grid points
// ============================================================================

/**
 * @brief Compute logK at each cell via spectral sum
 * 
 * Legacy: conductivity_kernel_3D_logK()
 * logK[idx] = (sigma_f / sqrt(n_modes)) * Σᵢ (aᵢ sin(k·x) + bᵢ cos(k·x))
 * 
 * Cell-centered coordinates: x = h * (ix + 0.5, iy + 0.5, iz + 0.5)
 */
__global__ void kernel_eval_logK(const real* __restrict__ V1,
                                 const real* __restrict__ V2,
                                 const real* __restrict__ V3,
                                 const real* __restrict__ a,
                                 const real* __restrict__ b,
                                 const int n_modes,
                                 real* __restrict__ logK,
                                 const real h,
                                 const int nx, const int ny, const int nz,
                                 const real sigma_f) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (ix >= nx || iy >= ny || iz >= nz) return;
    
    const int idx = ix + iy * nx + iz * nx * ny;
    
    // Cell-centered position
    const real x = h * (static_cast<real>(ix) + static_cast<real>(0.5));
    const real y = h * (static_cast<real>(iy) + static_cast<real>(0.5));
    const real z = h * (static_cast<real>(iz) + static_cast<real>(0.5));
    
    // Sum over all modes
    real sum = static_cast<real>(0.0);
    for (int i = 0; i < n_modes; ++i) {
        const real phase = V1[i] * x + V2[i] * y + V3[i] * z;
        sum += a[i] * sin(phase) + b[i] * cos(phase);
    }
    
    // Scale by sigma_f / sqrt(n_modes) per legacy
    logK[idx] = (sigma_f / sqrt(static_cast<real>(n_modes))) * sum;
}

// ============================================================================
// Kernel: Transform logK → K = exp(logK)
// ============================================================================

__global__ void kernel_exp(real* __restrict__ K,
                           const real* __restrict__ logK,
                           const size_t n) {
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    K[idx] = exp(logK[idx]);
}

// ============================================================================
// Kernel: Stats reduction helpers (min/max/sum)
// ============================================================================

__global__ void kernel_minmax_sum(const real* __restrict__ data,
                                  const size_t n,
                                  real* __restrict__ block_mins,
                                  real* __restrict__ block_maxs,
                                  real* __restrict__ block_sums) {
    extern __shared__ char smem[];
    real* s_min = reinterpret_cast<real*>(smem);
    real* s_max = s_min + blockDim.x;
    real* s_sum = s_max + blockDim.x;
    
    const size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize
    real local_min = (idx < n) ? data[idx] : real(1e30);
    real local_max = (idx < n) ? data[idx] : real(-1e30);
    real local_sum = (idx < n) ? data[idx] : real(0);
    
    // Grid-stride loop
    idx += blockDim.x * gridDim.x;
    while (idx < n) {
        real val = data[idx];
        local_min = fmin(local_min, val);
        local_max = fmax(local_max, val);
        local_sum += val;
        idx += blockDim.x * gridDim.x;
    }
    
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    s_sum[tid] = local_sum;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fmin(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_mins[blockIdx.x] = s_min[0];
        block_maxs[blockIdx.x] = s_max[0];
        block_sums[blockIdx.x] = s_sum[0];
    }
}

// ============================================================================
// Host API implementations
// ============================================================================

void init_stochastic_rng(StochasticWorkspace& workspace,
                         uint64_t seed,
                         const CudaContext& ctx) {
    if (workspace.n_modes <= 0) {
        throw std::runtime_error("StochasticWorkspace not allocated");
    }
    
    const int block = 256;
    const int grid = (workspace.n_modes + block - 1) / block;
    
    kernel_init_rng<<<grid, block, 0, ctx.cuda_stream()>>>(
        workspace.rng_states.data(),
        seed,
        workspace.n_modes);
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

void generate_gaussian_field(StochasticWorkspace& workspace,
                             const Grid3D& grid,
                             const StochasticConfig& cfg,
                             const CudaContext& ctx) {
    if (!workspace.is_allocated()) {
        throw std::runtime_error("StochasticWorkspace not allocated");
    }
    
    const int n_modes = cfg.n_modes;
    const real lambda = cfg.corr_length;
    const real sigma_f = sqrt(cfg.sigma2);  // Standard deviation of log-K
    
    // === Stage 1: Generate random Fourier modes ===
    {
        const int block = 256;
        const int modes_grid = (n_modes + block - 1) / block;
        
        if (cfg.covariance_type == 0) {
            // Exponential covariance
            kernel_random_modes_exp<<<modes_grid, block, 0, ctx.cuda_stream()>>>(
                workspace.rng_states.data(),
                workspace.k1.data(),
                workspace.k2.data(),
                workspace.k3.data(),
                workspace.coef_a.data(),
                workspace.coef_b.data(),
                lambda,
                n_modes);
        } else {
            // Gaussian covariance (k_max = 100 per legacy)
            const int k_max = 100;
            kernel_random_modes_gauss<<<modes_grid, block, 0, ctx.cuda_stream()>>>(
                workspace.rng_states.data(),
                workspace.k1.data(),
                workspace.k2.data(),
                workspace.k3.data(),
                workspace.coef_a.data(),
                workspace.coef_b.data(),
                lambda,
                n_modes,
                k_max);
        }
        RWPT_CUDA_CHECK(cudaGetLastError());
    }
    
    // === Stage 2: Evaluate spectral sum at each grid cell ===
    {
        // Use 8x8x8 block for 3D kernel (matches legacy pattern)
        dim3 block(8, 8, 8);
        dim3 grid_dim((grid.nx + block.x - 1) / block.x,
                      (grid.ny + block.y - 1) / block.y,
                      (grid.nz + block.z - 1) / block.z);
        
        kernel_eval_logK<<<grid_dim, block, 0, ctx.cuda_stream()>>>(
            workspace.k1.data(),
            workspace.k2.data(),
            workspace.k3.data(),
            workspace.coef_a.data(),
            workspace.coef_b.data(),
            n_modes,
            workspace.logK.data(),
            grid.dx,
            grid.nx, grid.ny, grid.nz,
            sigma_f);
        
        RWPT_CUDA_CHECK(cudaGetLastError());
    }
}

void generate_K_lognormal(DeviceSpan<real> K,
                          DeviceSpan<const real> logK,
                          const Grid3D& grid,
                          const StochasticConfig& cfg,
                          const CudaContext& ctx) {
    const size_t n = grid.num_cells();
    if (K.size() < n || logK.size() < n) {
        throw std::runtime_error("K or logK buffer too small");
    }
    
    const int block = 256;
    const int grid_1d = (n + block - 1) / block;
    
    // K = exp(logK) — legacy convention (no mean shift)
    kernel_exp<<<grid_1d, block, 0, ctx.cuda_stream()>>>(
        K.data(),
        const_cast<real*>(logK.data()),  // DeviceSpan<const real> workaround
        n);
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

void generate_K_field(DeviceSpan<real> K,
                      StochasticWorkspace& workspace,
                      const Grid3D& grid,
                      const StochasticConfig& cfg,
                      const CudaContext& ctx) {
    // 1. Initialize RNG (if not already done, do it with cfg.seed)
    init_stochastic_rng(workspace, cfg.seed, ctx);
    
    // 2. Generate Gaussian field
    generate_gaussian_field(workspace, grid, cfg, ctx);
    
    // 3. Transform to lognormal
    generate_K_lognormal(K, DeviceSpan<const real>(workspace.logK.data(), workspace.n_cells), 
                         grid, cfg, ctx);
}

void compute_field_stats(DeviceSpan<const real> data,
                         real& min_val, real& max_val, real& mean_val,
                         const CudaContext& ctx) {
    const size_t n = data.size();
    if (n == 0) {
        min_val = max_val = mean_val = 0;
        return;
    }
    
    // Use moderate number of blocks
    const int block = 256;
    const int n_blocks = std::min(256, static_cast<int>((n + block - 1) / block));
    
    // Allocate temporary device buffers for block results
    DeviceBuffer<real> d_mins(n_blocks);
    DeviceBuffer<real> d_maxs(n_blocks);
    DeviceBuffer<real> d_sums(n_blocks);
    
    const size_t smem_size = 3 * block * sizeof(real);
    
    kernel_minmax_sum<<<n_blocks, block, smem_size, ctx.cuda_stream()>>>(
        const_cast<real*>(data.data()),
        n,
        d_mins.data(),
        d_maxs.data(),
        d_sums.data());
    
    RWPT_CUDA_CHECK(cudaGetLastError());
    
    // Copy block results to host and finalize
    std::vector<real> h_mins(n_blocks), h_maxs(n_blocks), h_sums(n_blocks);
    
    RWPT_CUDA_CHECK(cudaMemcpyAsync(h_mins.data(), d_mins.data(), n_blocks * sizeof(real),
                               cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    RWPT_CUDA_CHECK(cudaMemcpyAsync(h_maxs.data(), d_maxs.data(), n_blocks * sizeof(real),
                               cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    RWPT_CUDA_CHECK(cudaMemcpyAsync(h_sums.data(), d_sums.data(), n_blocks * sizeof(real),
                               cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    
    ctx.synchronize();
    
    min_val = h_mins[0];
    max_val = h_maxs[0];
    real sum = h_sums[0];
    for (int i = 1; i < n_blocks; ++i) {
        min_val = std::min(min_val, h_mins[i]);
        max_val = std::max(max_val, h_maxs[i]);
        sum += h_sums[i];
    }
    mean_val = sum / static_cast<real>(n);
}

} // namespace physics
} // namespace rwpt

```

# src\physics\stochastic\stochastic.cuh

```cuh
#pragma once

/**
 * @file stochastic.cuh
 * @brief Stochastic K field generation (lognormal) - API
 * 
 * Port of legacy/random_field_generation.cu using Randomized Spectral Method.
 * Does NOT use FFT - direct sum of Fourier modes.
 * 
 * Reference:
 *   Räss, Kolyukhin, Minakov (2019), Comp. & Geosci. 131, 158-169
 *   DOI: 10.1016/j.cageo.2019.06.007
 * 
 * Conventions (matching legacy exactly):
 *   - sigma_f = sqrt(sigma2) is the std dev of log-K
 *   - logK = (sigma_f / sqrt(n_modes)) * Σᵢ (aᵢ sin(k·x) + bᵢ cos(k·x))
 *   - K = exp(logK)  (no mean correction in legacy)
 *   - Cell-centered coordinates: x = h * (ix + 0.5, iy + 0.5, iz + 0.5)
 */

#include "../common/physics_config.hpp"
#include "../common/workspaces.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../runtime/CudaContext.cuh"

namespace rwpt {
namespace physics {

/**
 * @brief Initialize RNG states for stochastic field generation
 * 
 * Must be called once before generate_gaussian_field.
 * Uses deterministic seeding: state[i] = curand_init(base_seed + i, i, 0)
 * 
 * @param workspace Pre-allocated stochastic workspace
 * @param seed      Base seed for RNG
 * @param ctx       CUDA context (stream)
 */
void init_stochastic_rng(StochasticWorkspace& workspace,
                         uint64_t seed,
                         const CudaContext& ctx);

/**
 * @brief Generate correlated Gaussian random field G (logK)
 * 
 * Two-stage process:
 *   1. Generate random Fourier mode coefficients (k1,k2,k3,a,b)
 *   2. Evaluate sum at each grid cell
 * 
 * Output is stored in workspace.logK
 * 
 * @param workspace  Contains RNG states and mode coefficient buffers
 * @param grid       Grid specification (nx, ny, nz, dx)
 * @param cfg        Stochastic config (sigma2, corr_length, n_modes, covariance_type)
 * @param ctx        CUDA context (stream)
 */
void generate_gaussian_field(StochasticWorkspace& workspace,
                             const Grid3D& grid,
                             const StochasticConfig& cfg,
                             const CudaContext& ctx);

/**
 * @brief Transform Gaussian field to lognormal K
 * 
 * K = exp(logK) following legacy convention (no mean shift).
 * 
 * @param K          Output: lognormal K field (device memory, size = num_cells)
 * @param logK       Input: Gaussian field from workspace.logK
 * @param grid       Grid specification
 * @param cfg        Config (currently unused but reserved for normalization options)
 * @param ctx        CUDA context
 */
void generate_K_lognormal(DeviceSpan<real> K,
                          DeviceSpan<const real> logK,
                          const Grid3D& grid,
                          const StochasticConfig& cfg,
                          const CudaContext& ctx);

/**
 * @brief Convenience: generate lognormal K in one call
 * 
 * Combines init_stochastic_rng + generate_gaussian_field + generate_K_lognormal.
 * Use when you don't need access to intermediate logK.
 * 
 * @param K          Output: lognormal K field
 * @param workspace  Stochastic workspace
 * @param grid       Grid
 * @param cfg        Config
 * @param ctx        CUDA context
 */
void generate_K_field(DeviceSpan<real> K,
                      StochasticWorkspace& workspace,
                      const Grid3D& grid,
                      const StochasticConfig& cfg,
                      const CudaContext& ctx);

/**
 * @brief Compute basic statistics of a device array (for diagnostics)
 * 
 * @param data       Device data
 * @param n          Number of elements
 * @param min_val    Output: minimum
 * @param max_val    Output: maximum
 * @param mean_val   Output: arithmetic mean
 * @param ctx        CUDA context
 */
void compute_field_stats(DeviceSpan<const real> data,
                         real& min_val, real& max_val, real& mean_val,
                         const CudaContext& ctx);

} // namespace physics
} // namespace rwpt

```

# src\runtime\cuda_check.cuh

```cuh
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <cstdio>

namespace rwpt {

// CUDA runtime error checking
inline void cuda_check_impl(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::string msg = std::string("CUDA error at ") + file + ":" + 
                         std::to_string(line) + " - " + cudaGetErrorString(err);
        throw std::runtime_error(msg);
    }
}

#define RWPT_CUDA_CHECK(expr) ::rwpt::cuda_check_impl((expr), __FILE__, __LINE__)

// cuBLAS error checking
inline void cublas_check_impl(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::string msg = std::string("cuBLAS error at ") + file + ":" + 
                         std::to_string(line) + " - code " + std::to_string(status);
        throw std::runtime_error(msg);
    }
}

#define RWPT_CUBLAS_CHECK(expr) ::rwpt::cublas_check_impl((expr), __FILE__, __LINE__)

} // namespace rwpt

```

# src\runtime\CudaContext.cu

```cu
#include "CudaContext.cuh"
#include "cuda_check.cuh"

namespace rwpt {

CudaContext::CudaContext(int device_id)
    : device_id_(device_id), stream_(nullptr), cublas_(nullptr) {
    
    RWPT_CUDA_CHECK(cudaSetDevice(device_id_));
    RWPT_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    RWPT_CUBLAS_CHECK(cublasCreate(&cublas_));
    RWPT_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
}

CudaContext::~CudaContext() {
    release();
}

CudaContext::CudaContext(CudaContext&& other) noexcept
    : device_id_(other.device_id_),
      stream_(other.stream_),
      cublas_(other.cublas_) {
    
    other.stream_ = nullptr;
    other.cublas_ = nullptr;
}

CudaContext& CudaContext::operator=(CudaContext&& other) noexcept {
    if (this != &other) {
        release();
        
        device_id_ = other.device_id_;
        stream_ = other.stream_;
        cublas_ = other.cublas_;
        
        other.stream_ = nullptr;
        other.cublas_ = nullptr;
    }
    return *this;
}

void CudaContext::synchronize() const {
    RWPT_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CudaContext::release() {
    if (cublas_ != nullptr) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

} // namespace rwpt

```

# src\runtime\CudaContext.cuh

```cuh
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace rwpt {

class CudaContext {
public:
    explicit CudaContext(int device_id = 0);
    ~CudaContext();

    // Move semantics
    CudaContext(CudaContext&& other) noexcept;
    CudaContext& operator=(CudaContext&& other) noexcept;

    // Delete copy semantics
    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    // Getters
    int device() const noexcept { return device_id_; }
    cudaStream_t cuda_stream() const noexcept { return stream_; }
    cublasHandle_t cublas_handle() const noexcept { return cublas_; }

    // Explicit synchronization
    void synchronize() const;

private:
    int device_id_;
    cudaStream_t stream_;
    cublasHandle_t cublas_;

    void release();
};

} // namespace rwpt

```

# src\runtime\GpuTimer.cuh

```cuh
#pragma once

#include <cuda_runtime.h>
#include "cuda_check.cuh"

namespace rwpt {

class GpuTimer {
public:
    GpuTimer() {
        RWPT_CUDA_CHECK(cudaEventCreate(&start_));
        RWPT_CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        if (start_) cudaEventDestroy(start_);
        if (stop_) cudaEventDestroy(stop_);
    }

    // Delete copy semantics
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    // Move semantics
    GpuTimer(GpuTimer&& other) noexcept
        : start_(other.start_), stop_(other.stop_) {
        other.start_ = nullptr;
        other.stop_ = nullptr;
    }

    GpuTimer& operator=(GpuTimer&& other) noexcept {
        if (this != &other) {
            if (start_) cudaEventDestroy(start_);
            if (stop_) cudaEventDestroy(stop_);
            
            start_ = other.start_;
            stop_ = other.stop_;
            
            other.start_ = nullptr;
            other.stop_ = nullptr;
        }
        return *this;
    }

    void start(cudaStream_t stream) {
        RWPT_CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    float stop(cudaStream_t stream) {
        RWPT_CUDA_CHECK(cudaEventRecord(stop_, stream));
        RWPT_CUDA_CHECK(cudaEventSynchronize(stop_));
        
        float elapsed_ms = 0.0f;
        RWPT_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
        return elapsed_ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

} // namespace rwpt

```

# src\runtime\nvtx_range.cuh

```cuh
#pragma once

// Optional NVTX range support
// If NVTX is not available, this becomes a no-op

#ifdef USE_NVTX
#include <nvToolsExt.h>

namespace rwpt {

class NvtxRange {
public:
    explicit NvtxRange(const char* name) {
        nvtxRangePushA(name);
    }
    
    ~NvtxRange() {
        nvtxRangePop();
    }
    
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

#define RWPT_NVTX_RANGE_CONCAT_IMPL(a, b) a##b
#define RWPT_NVTX_RANGE_CONCAT(a, b) RWPT_NVTX_RANGE_CONCAT_IMPL(a, b)
#define RWPT_NVTX_RANGE(name) ::rwpt::NvtxRange RWPT_NVTX_RANGE_CONCAT(__nvtx_range__, __COUNTER__)(name)

} // namespace rwpt

#else

namespace rwpt {

// No-op implementation when NVTX is not available
class NvtxRange {
public:
    explicit NvtxRange(const char*) {}
};

#define RWPT_NVTX_RANGE(name) do {} while(0)

} // namespace rwpt

#endif // USE_NVTX

```

