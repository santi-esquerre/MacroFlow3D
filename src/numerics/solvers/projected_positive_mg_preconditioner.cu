#include "projected_positive_mg_preconditioner.cuh"

#include "../blas/blas.cuh"

#include <stdexcept>

namespace macroflow3d {
namespace solvers {

ProjectedPositiveMGPreconditioner::ProjectedPositiveMGPreconditioner(
    multigrid::MGHierarchy& hierarchy, const multigrid::MGConfig& config)
    : hierarchy_(&hierarchy), config_(config), mean_zero_workspaces_(hierarchy.levels.size()) {
    multigrid::validate_projected_positive_hierarchy(hierarchy, config);
    for (std::size_t level = 0; level < hierarchy.levels.size(); ++level) {
        mean_zero_workspaces_[level].prepare(hierarchy.levels[level].grid.num_cells());
    }
}

void ProjectedPositiveMGPreconditioner::apply(CudaContext& ctx, DeviceSpan<const real> r,
                                              DeviceSpan<real> z) const {
    auto& finest = hierarchy_->levels.front();
    if (r.size() != finest.grid.num_cells() || z.size() != finest.grid.num_cells()) {
        throw std::invalid_argument("Projected positive MG vector size does not match finest grid");
    }

    // Legacy multigrid solves L(q)x = b for L = div(q grad) = -A(q).
    // Hence b = -P r gives x approximately A(q)^-1 P r.
    blas::copy(ctx, r, finest.b.span());
    constraints::MeanZeroProjector{}.project(ctx, finest.b.span(), mean_zero_workspaces_.front());
    blas::scal(ctx, finest.b.span(), real(-1.0));
    blas::fill(ctx, finest.x.span(), real(0.0));
    multigrid::projected_positive_v_cycle(ctx, *hierarchy_, config_, mean_zero_workspaces_);
    constraints::MeanZeroProjector{}.project(ctx, finest.x.span(), mean_zero_workspaces_.front());
    blas::copy(ctx, DeviceSpan<const real>(finest.x.span()), z);
}

std::size_t ProjectedPositiveMGPreconditioner::allocated_device_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const auto& workspace : mean_zero_workspaces_) {
        bytes += workspace.allocated_device_bytes();
    }
    return bytes;
}

std::size_t ProjectedPositiveMGPreconditioner::estimate_device_bytes(
    const multigrid::MGHierarchy& hierarchy) {
    // Mirrors the constructor exactly: one MeanZeroWorkspace prepared for
    // each hierarchy level's exact cell count.
    std::size_t bytes = 0;
    for (const auto& level : hierarchy.levels) {
        bytes += constraints::MeanZeroWorkspace::estimate_device_bytes(level.grid.num_cells());
    }
    return bytes;
}

} // namespace solvers
} // namespace macroflow3d
