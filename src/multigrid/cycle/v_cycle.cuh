#pragma once

#include "../../core/BCSpec.hpp"
#include "../../numerics/pin_spec.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../mg_types.hpp"

namespace macroflow3d {
namespace multigrid {

struct VCycleResult {
    int num_cycles = 0;
    real initial_residual = 0.0;
    real final_residual = 0.0;
    bool converged = false;
};

/**
 * @brief Execute one V-cycle on the MG hierarchy
 * @ingroup multigrid_cycle
 *
 * @param ctx     CUDA context
 * @param hier    MG hierarchy (pre-initialized with K)
 * @param level   Current level (0 = finest)
 * @param config  MG configuration
 * @param bc      Boundary conditions
 * @param pin     Pin specification (applied to ALL levels - legacy semantics)
 */
void v_cycle_recursive(CudaContext& ctx, MGHierarchy& hier, int level, const MGConfig& config,
                       const BCSpec& bc, PinSpec pin = {});

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
VCycleResult mg_solve(CudaContext& ctx, MGHierarchy& hier, const MGConfig& config, const BCSpec& bc,
                      int max_cycles = 10, real rtol = 1e-6, PinSpec pin = {});

} // namespace multigrid
} // namespace macroflow3d
