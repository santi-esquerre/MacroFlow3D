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
