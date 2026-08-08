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

    // Additive, behavior-neutral byte introspection (SF-12). Exact sum of the
    // owned per-level mean-zero workspace capacities (the hierarchy itself is
    // referenced, not owned, and is accounted separately by the caller).
    // Never allocates.
    [[nodiscard]] std::size_t allocated_device_bytes() const noexcept;

    // Host-only prediction of allocated_device_bytes() after constructing a
    // preconditioner over `hierarchy`; kept colocated with the constructor so
    // it cannot drift. Does not validate `hierarchy`.
    [[nodiscard]] static std::size_t estimate_device_bytes(const multigrid::MGHierarchy& hierarchy);

  private:
    multigrid::MGHierarchy* hierarchy_;
    multigrid::MGConfig config_;
    mutable std::vector<constraints::MeanZeroWorkspace> mean_zero_workspaces_;
};

} // namespace solvers
} // namespace macroflow3d
