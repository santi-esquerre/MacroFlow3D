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
