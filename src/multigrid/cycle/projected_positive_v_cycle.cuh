#pragma once

#include "../../numerics/constraints/MeanZeroProjector.cuh"
#include "../mg_types.hpp"

#include <vector>

namespace macroflow3d {
namespace multigrid {

// Executes one mean-zero, symmetric-by-construction V-cycle for the positive
// operator A(q) = -div(q grad).  The underlying legacy kernels remain the
// negative operator L(q) = div(q grad), so callers provide b = -r.
void projected_positive_v_cycle(CudaContext& ctx, MGHierarchy& hierarchy,
                                const MGConfig& config,
                                std::vector<constraints::MeanZeroWorkspace>& mean_zero_workspaces);

// Host-only construction-time validation shared by the Lester adapter.
void validate_projected_positive_hierarchy(const MGHierarchy& hierarchy, const MGConfig& config);

} // namespace multigrid
} // namespace macroflow3d
