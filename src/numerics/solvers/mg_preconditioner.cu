#include "mg_preconditioner.cuh"
#include "../blas/blas.cuh"

namespace macroflow3d {
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
    macroflow3d::blas::copy(ctx, r, finest.b.span());
    
    // 2. Zero initial guess on finest level
    macroflow3d::blas::fill(ctx, finest.x.span(), 0.0);
    
    // 3. Execute ONE V-cycle (recursive from level 0, pin propagated)
    multigrid::v_cycle_recursive(ctx, *hierarchy_, 0, config_, bc_, pin_);
    
    // 4. Copy result to output z
    macroflow3d::blas::copy(ctx, DeviceSpan<const real>(finest.x.span()), z);
}

} // namespace solvers
} // namespace macroflow3d
