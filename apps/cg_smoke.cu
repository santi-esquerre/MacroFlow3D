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

int main() {
    try {
        std::cout << "=== CG Smoke Test ===\n\n";
        
        // Create CUDA context
        rwpt::CudaContext ctx(0);
        
        // Setup grid
        const int n = 32;
        rwpt::Grid3D grid(n, n, n, 1.0, 1.0, 1.0);
        size_t num_cells = grid.num_cells();
        
        std::cout << "Grid: " << grid.nx << " x " << grid.ny << " x " << grid.nz 
                  << " = " << num_cells << " cells\n";
        
        // Create operator
        rwpt::operators::Poisson3DOperator A(grid);
        
        // Allocate vectors
        rwpt::DeviceBuffer<rwpt::real> x(num_cells);
        rwpt::DeviceBuffer<rwpt::real> b(num_cells);
        
        // Initialize: x = 0, b = 1 (constant RHS)
        rwpt::blas::fill(ctx, x.span(), 0.0);
        rwpt::blas::fill(ctx, b.span(), 1.0);
        
        // Alternatively: set b as delta function at center
        // std::vector<rwpt::real> b_host(num_cells, 0.0);
        // size_t center_idx = grid.idx(n/2, n/2, n/2);
        // b_host[center_idx] = 1.0;
        // RWPT_CUDA_CHECK(cudaMemcpy(b.data(), b_host.data(), 
        //                            num_cells * sizeof(rwpt::real), 
        //                            cudaMemcpyHostToDevice));
        
        // Configure CG
        rwpt::solvers::CGConfig cfg;
        cfg.max_iter = 200;
        cfg.rtol = 1e-6;
        cfg.atol = 0.0;
        cfg.check_every = 10;  // Check convergence every 10 iters to reduce host sync
        
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
