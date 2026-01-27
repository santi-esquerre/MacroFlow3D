#pragma once

#include "../mg_types.hpp"
#include "../../runtime/CudaContext.cuh"

namespace rwpt {
namespace multigrid {

struct VCycleResult {
    int num_cycles = 0;
    real initial_residual = 0.0;
    real final_residual = 0.0;
    bool converged = false;
};

// Execute one V-cycle on the MG hierarchy
// Assumes MGHierarchy is already initialized with K fields
void v_cycle_recursive(
    CudaContext& ctx,
    MGHierarchy& hier,
    int level,
    const MGConfig& config
);

// Solve A*x = b using multigrid V-cycles
// Assumes hier.levels[0].b is set, hier.levels[0].x is initial guess
VCycleResult mg_solve(
    CudaContext& ctx,
    MGHierarchy& hier,
    const MGConfig& config,
    int max_cycles = 10,
    real rtol = 1e-6
);

} // namespace multigrid
} // namespace rwpt
