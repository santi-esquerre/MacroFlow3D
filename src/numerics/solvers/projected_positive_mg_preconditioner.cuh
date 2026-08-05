#pragma once

#include "../../multigrid/cycle/projected_positive_v_cycle.cuh"
#include "preconditioner.cuh"

#include <vector>

namespace macroflow3d {
namespace solvers {

// Non-owning adapter for a coefficient hierarchy already populated with q.
// apply() mutates hierarchy work buffers and its reduction workspaces, so this
// object is intentionally not reentrant or safe for concurrent streams.
class ProjectedPositiveMGPreconditioner {
  public:
    ProjectedPositiveMGPreconditioner(multigrid::MGHierarchy& hierarchy,
                                      const multigrid::MGConfig& config);

    ProjectedPositiveMGPreconditioner(const ProjectedPositiveMGPreconditioner&) = delete;
    ProjectedPositiveMGPreconditioner& operator=(const ProjectedPositiveMGPreconditioner&) = delete;
    ProjectedPositiveMGPreconditioner(ProjectedPositiveMGPreconditioner&&) = default;
    ProjectedPositiveMGPreconditioner& operator=(ProjectedPositiveMGPreconditioner&&) = default;

    // z = P A(q)^-1 P r, approximately.  The input r is never mutated.
    void apply(CudaContext& ctx, DeviceSpan<const real> r, DeviceSpan<real> z) const;

  private:
    multigrid::MGHierarchy* hierarchy_;
    multigrid::MGConfig config_;
    mutable std::vector<constraints::MeanZeroWorkspace> mean_zero_workspaces_;
};

} // namespace solvers
} // namespace macroflow3d
