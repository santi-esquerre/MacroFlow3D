#include "BlockDiagonalMGPreconditioner.cuh"

#include "../../multigrid/smoothers/residual_3d.cuh"
#include "../../numerics/blas/axpy.cuh"
#include "../../numerics/blas/blas.cuh"

#include <stdexcept>

namespace macroflow3d {
namespace streamfunctions {

namespace {

// Same periodic boundary spec projected_positive_v_cycle.cu's own (private)
// periodic_bc() constructs; duplicated here (public BCSpec/BCType types
// only, no multigrid file touched) because compute_residual_3d needs an
// explicit BCSpec and the hierarchy carries none of its own. The accepted
// Lester streamfunction discretization is periodic-only (see
// ResidualEvaluator.cuh / affine_gauge.cuh); this file does not introduce a
// new boundary policy.
BCSpec periodic_bc() {
    const BCFace periodic{BCType::Periodic, real(0.0)};
    BCSpec bc;
    bc.xmin = bc.xmax = bc.ymin = bc.ymax = bc.zmin = bc.zmax = periodic;
    return bc;
}

} // namespace

BlockDiagonalMGPreconditioner::BlockDiagonalMGPreconditioner(multigrid::MGHierarchy& hierarchy,
                                                              const multigrid::MGConfig& config)
    : hierarchy_(&hierarchy), config_(config), mean_zero_workspaces_(hierarchy.levels.size()) {
    multigrid::validate_projected_positive_hierarchy(hierarchy, config);
    for (std::size_t level = 0; level < hierarchy.levels.size(); ++level) {
        mean_zero_workspaces_[level].prepare(hierarchy.levels[level].grid.num_cells());
    }
    const std::size_t n = hierarchy.finest_grid().num_cells();
    x1_scratch_.resize(n);
    residual_scratch_.resize(n);
}

void BlockDiagonalMGPreconditioner::apply_component(CudaContext& ctx, DeviceSpan<const real> r,
                                                     DeviceSpan<real> z) const {
    auto& finest = hierarchy_->levels.front();
    const constraints::MeanZeroProjector projector;
    auto& finest_workspace = mean_zero_workspaces_.front();

    // ---- Cycle 1 (identical to ProjectedPositiveMGPreconditioner::apply) ----
    // Legacy multigrid solves L(q) x = b for L(q) = div(q grad) = -A(q).
    // Hence b0 = -P(r) gives x1 ~= A(q)^-1 P(r).
    blas::copy(ctx, r, finest.b.span());
    projector.project(ctx, finest.b.span(), finest_workspace);
    blas::scal(ctx, finest.b.span(), real(-1.0));
    blas::fill(ctx, finest.x.span(), real(0.0));
    multigrid::projected_positive_v_cycle(ctx, *hierarchy_, config_, mean_zero_workspaces_);
    projector.project(ctx, finest.x.span(), finest_workspace);

    // Stash x1 (finest.x is about to be reused by the second cycle).
    blas::copy(ctx, DeviceSpan<const real>(finest.x.span()), x1_scratch_.span());

    // ---- Defect correction: res = b0 - L(q) x1, the TRUE level-0 residual
    // of the first cycle's solve (NOT the stale mid-cycle snapshot cycle()
    // leaves in finest.r; see the file header). ----
    multigrid::compute_residual_3d(ctx, finest.grid, DeviceSpan<const real>(finest.x.span()),
                                   DeviceSpan<const real>(finest.b.span()),
                                   DeviceSpan<const real>(finest.coefficient.span()),
                                   residual_scratch_.span(), periodic_bc());
    projector.project(ctx, residual_scratch_.span(), finest_workspace);

    // ---- Cycle 2: solve L(q) dx ~= res from a FRESH zero initial guess. ----
    blas::copy(ctx, DeviceSpan<const real>(residual_scratch_.span()), finest.b.span());
    projector.project(ctx, finest.b.span(), finest_workspace);
    blas::fill(ctx, finest.x.span(), real(0.0));
    multigrid::projected_positive_v_cycle(ctx, *hierarchy_, config_, mean_zero_workspaces_);
    projector.project(ctx, finest.x.span(), finest_workspace);

    // z = P(x1 + dx).
    blas::copy(ctx, DeviceSpan<const real>(x1_scratch_.span()), z);
    blas::axpy(ctx, real(1.0), DeviceSpan<const real>(finest.x.span()), z);
    projector.project(ctx, z, finest_workspace);
}

void BlockDiagonalMGPreconditioner::apply(CudaContext& ctx, ConstCoupledVectorView r,
                                          CoupledVectorView z) const {
    const std::size_t n = hierarchy_->finest_grid().num_cells();
    if (r.c1.size() != n || r.c2.size() != n || z.c1.size() != n || z.c2.size() != n) {
        throw std::invalid_argument(
            "BlockDiagonalMGPreconditioner requires coupled components matching the finest grid");
    }
    // Sequential per component: apply() mutates the SAME hierarchy work
    // buffers (see the file header) and is not reentrant.
    apply_component(ctx, r.c1, z.c1);
    apply_component(ctx, r.c2, z.c2);
}

std::size_t BlockDiagonalMGPreconditioner::allocated_device_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const auto& workspace : mean_zero_workspaces_) {
        bytes += workspace.allocated_device_bytes();
    }
    bytes += x1_scratch_.capacity() * sizeof(real);
    bytes += residual_scratch_.capacity() * sizeof(real);
    return bytes;
}

std::size_t BlockDiagonalMGPreconditioner::estimate_device_bytes(
    const multigrid::MGHierarchy& hierarchy) {
    std::size_t bytes = 0;
    for (const auto& level : hierarchy.levels) {
        bytes += constraints::MeanZeroWorkspace::estimate_device_bytes(level.grid.num_cells());
    }
    const std::size_t n = hierarchy.finest_grid().num_cells();
    bytes += 2 * n * sizeof(real);
    return bytes;
}

} // namespace streamfunctions
} // namespace macroflow3d
