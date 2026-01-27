#include "v_cycle.cuh"
#include "../transfer/restrict_3d.cuh"
#include "../transfer/prolong_3d.cuh"
#include "../smoothers/residual_3d.cuh"
#include "../smoothers/gsrb_3d.cuh"
#include "../../numerics/blas/blas.cuh"
#include "../../core/BCSpec.hpp"
#include <cmath>

namespace rwpt {
namespace multigrid {

// Legacy: V_cycle from CCMG_V_cycle.cu
// Recursive V-cycle:
// 1. Pre-smooth on current level
// 2. Compute residual
// 3. Restrict to coarser
// 4. Recursive call (or solve on coarsest)
// 5. Prolong correction back
// 6. Post-smooth on current level
void v_cycle_recursive(
    CudaContext& ctx,
    MGHierarchy& hier,
    int level,
    const MGConfig& config
) {
    const int num_levels = hier.num_levels();
    
    // Coarsest level: solve directly with many GSRB iterations
    if (level == num_levels - 1) {
        auto& lvl = hier.levels[level];
        BCSpec bc;  // Default Dirichlet 0
        gsrb_smooth_3d(ctx, lvl.grid, lvl.x.span(), lvl.b.span(), lvl.K.span(), config.coarse_solve_iters, bc);
        return;
    }
    
    auto& fine = hier.levels[level];
    auto& coarse = hier.levels[level + 1];
    BCSpec bc;  // Default Dirichlet 0
    
    // 1. Pre-smooth: x^{h} = S(x^{h}, b^{h})
    gsrb_smooth_3d(ctx, fine.grid, fine.x.span(), fine.b.span(), fine.K.span(), config.pre_smooth, bc);
    
    // 2. Compute residual: r^{h} = b^{h} - A^{h} * x^{h}
    compute_residual_3d(ctx, fine.grid, fine.x.span(), fine.b.span(), fine.K.span(), fine.r.span(), bc);
    
    // 3. Restrict residual to coarse RHS: b^{2h} = R * r^{h}
    restrict_3d(ctx, fine.grid, coarse.grid, fine.r.span(), coarse.b.span());
    
    // 4. Initialize coarse correction: e^{2h} = 0
    rwpt::blas::fill(ctx, coarse.x.span(), 0.0);
    
    // 5. Recursively solve: A^{2h} * e^{2h} = b^{2h}
    v_cycle_recursive(ctx, hier, level + 1, config);
    
    // 6. Prolong correction and add: x^{h} += P * e^{2h}
    prolong_3d_add(ctx, coarse.grid, fine.grid, coarse.x.span(), fine.x.span());
    
    // 7. Post-smooth: x^{h} = S(x^{h}, b^{h})
    gsrb_smooth_3d(ctx, fine.grid, fine.x.span(), fine.b.span(), fine.K.span(), config.post_smooth, bc);
}

VCycleResult mg_solve(
    CudaContext& ctx,
    MGHierarchy& hier,
    const MGConfig& config,
    int max_cycles,
    real rtol
) {
    VCycleResult result;
    
    auto& finest = hier.levels[0];
    BCSpec bc;
    
    // Compute initial residual norm
    compute_residual_3d(ctx, finest.grid, finest.x.span(), finest.b.span(), finest.K.span(), finest.r.span(), bc);
    ReductionWorkspace red;
    red.ensure_for_nrm2(finest.r.size());
    result.initial_residual = rwpt::blas::nrm2_host(ctx, finest.r.span(), red);
    
    if (config.verbose) {
        // Would print here, but avoiding I/O in library code
    }
    
    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        result.num_cycles = cycle + 1;
        
        // Execute one V-cycle
        v_cycle_recursive(ctx, hier, 0, config);
        
        // Check convergence every cycle
        compute_residual_3d(ctx, finest.grid, finest.x.span(), finest.b.span(), finest.K.span(), finest.r.span(), bc);
        result.final_residual = rwpt::blas::nrm2_host(ctx, finest.r.span(), red);
        
        real relative_residual = result.final_residual / result.initial_residual;
        if (relative_residual < rtol) {
            result.converged = true;
            break;
        }
    }
    
    return result;
}

} // namespace multigrid
} // namespace rwpt
