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
