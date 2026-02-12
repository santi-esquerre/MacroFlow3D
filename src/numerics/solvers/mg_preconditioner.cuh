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

namespace macroflow3d {
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
} // namespace macroflow3d
