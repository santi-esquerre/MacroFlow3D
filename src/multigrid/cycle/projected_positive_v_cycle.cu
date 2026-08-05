#include "projected_positive_v_cycle.cuh"

#include "../../numerics/blas/blas.cuh"
#include "../smoothers/gsrb_3d.cuh"
#include "../smoothers/residual_3d.cuh"
#include "../transfer/prolong_3d.cuh"
#include "../transfer/restrict_3d.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace macroflow3d {
namespace multigrid {
namespace {

BCSpec periodic_bc() {
    const BCFace periodic{BCType::Periodic, real(0.0)};
    BCSpec bc;
    bc.xmin = bc.xmax = bc.ymin = bc.ymax = bc.zmin = bc.zmax = periodic;
    return bc;
}

void project(CudaContext& ctx, DeviceSpan<real> values, constraints::MeanZeroWorkspace& workspace) {
    constraints::MeanZeroProjector{}.project(ctx, values, workspace);
}

constexpr real grid_relative_tolerance = real(1e-12);

bool nearly_equal_positive(real lhs, real rhs) {
    return std::abs(lhs - rhs) <= grid_relative_tolerance * std::max(lhs, rhs);
}

bool is_valid_isotropic_grid(const Grid3D& grid) {
    if (grid.nx < 2 || grid.ny < 2 || grid.nz < 2 ||
        !std::isfinite(grid.dx) || !std::isfinite(grid.dy) || !std::isfinite(grid.dz) ||
        grid.dx <= real(0.0) || grid.dy <= real(0.0) || grid.dz <= real(0.0)) {
        return false;
    }
    return nearly_equal_positive(grid.dx, grid.dy) &&
           nearly_equal_positive(grid.dx, grid.dz);
}

void cycle(CudaContext& ctx, MGHierarchy& hierarchy, int level, const MGConfig& config,
           std::vector<constraints::MeanZeroWorkspace>& workspaces, const BCSpec& bc) {
    auto& fine = hierarchy.levels[level];
    auto& workspace = workspaces[static_cast<std::size_t>(level)];
    project(ctx, fine.b.span(), workspace);
    project(ctx, fine.x.span(), workspace);

    if (level == hierarchy.num_levels() - 1) {
        // Starting from zero at every coarse entry makes this fixed
        // red-black/black-red composition self-adjoint.
        gsrb_smooth_3d_ordered(ctx, fine.grid, fine.x.span(), fine.b.span(),
                               fine.coefficient.span(), config.coarse_solve_iters, bc,
                               GSRBColorOrder::RedBlack, PinSpec{false});
        project(ctx, fine.x.span(), workspace);
        gsrb_smooth_3d_ordered(ctx, fine.grid, fine.x.span(), fine.b.span(),
                               fine.coefficient.span(), config.coarse_solve_iters, bc,
                               GSRBColorOrder::BlackRed, PinSpec{false});
        project(ctx, fine.x.span(), workspace);
        return;
    }

    gsrb_smooth_3d_ordered(ctx, fine.grid, fine.x.span(), fine.b.span(), fine.coefficient.span(),
                           config.pre_smooth, bc, GSRBColorOrder::RedBlack, PinSpec{false});
    project(ctx, fine.x.span(), workspace);

    compute_residual_3d(ctx, fine.grid, fine.x.span(), fine.b.span(), fine.coefficient.span(),
                        fine.r.span(), bc, PinSpec{false});
    project(ctx, fine.r.span(), workspace);

    auto& coarse = hierarchy.levels[level + 1];
    auto& coarse_workspace = workspaces[static_cast<std::size_t>(level + 1)];
    restrict_3d(ctx, fine.grid, coarse.grid, fine.r.span(), coarse.b.span());
    project(ctx, coarse.b.span(), coarse_workspace);
    blas::fill(ctx, coarse.x.span(), real(0.0));
    project(ctx, coarse.x.span(), coarse_workspace);

    cycle(ctx, hierarchy, level + 1, config, workspaces, bc);

    project(ctx, coarse.x.span(), coarse_workspace);
    prolong_3d_add(ctx, coarse.grid, fine.grid, coarse.x.span(), fine.x.span());
    project(ctx, fine.x.span(), workspace);

    gsrb_smooth_3d_ordered(ctx, fine.grid, fine.x.span(), fine.b.span(), fine.coefficient.span(),
                           config.post_smooth, bc, GSRBColorOrder::BlackRed, PinSpec{false});
    project(ctx, fine.x.span(), workspace);
}

} // namespace

void validate_projected_positive_hierarchy(const MGHierarchy& hierarchy, const MGConfig& config) {
    if (hierarchy.num_levels() == 0) {
        throw std::invalid_argument("Projected positive MG requires at least one level");
    }
    if (config.num_levels != hierarchy.num_levels()) {
        throw std::invalid_argument("Projected positive MG config level count does not match hierarchy");
    }
    if (config.pre_smooth < 1 || config.post_smooth < 1 || config.pre_smooth != config.post_smooth ||
        config.coarse_solve_iters <= 0) {
        // Zero fine sweeps leave fine-grid modes unpreconditioned and can break SPD use in PCG.
        throw std::invalid_argument(
            "Projected positive MG requires equal positive pre/post sweeps and a positive coarse solve count");
    }
    for (int level = 0; level < hierarchy.num_levels(); ++level) {
        const auto& current = hierarchy.levels[level];
        if (!is_valid_isotropic_grid(current.grid) || current.grid.nx % 2 != 0 ||
            current.grid.ny % 2 != 0 || current.grid.nz % 2 != 0 ||
            current.x.size() != current.grid.num_cells() || current.b.size() != current.grid.num_cells() ||
            current.r.size() != current.grid.num_cells() ||
            current.coefficient.size() != current.grid.num_cells()) {
            throw std::invalid_argument(
                "Projected positive MG requires finite positive isotropic even-sized levels with matching buffers");
        }
        if (level + 1 < hierarchy.num_levels()) {
            const auto& coarse = hierarchy.levels[level + 1];
            if (current.grid.nx != 2 * coarse.grid.nx || current.grid.ny != 2 * coarse.grid.ny ||
                current.grid.nz != 2 * coarse.grid.nz || current.grid.nx % 2 != 0 ||
                current.grid.ny % 2 != 0 || current.grid.nz % 2 != 0 ||
                !nearly_equal_positive(coarse.grid.dx, real(2.0) * current.grid.dx) ||
                !nearly_equal_positive(coarse.grid.dy, real(2.0) * current.grid.dy) ||
                !nearly_equal_positive(coarse.grid.dz, real(2.0) * current.grid.dz)) {
                throw std::invalid_argument(
                    "Projected positive MG requires exact 2x2x2 dimensions and doubled spacing between levels");
            }
        }
    }
}

void projected_positive_v_cycle(CudaContext& ctx, MGHierarchy& hierarchy, const MGConfig& config,
                                std::vector<constraints::MeanZeroWorkspace>& workspaces) {
    if (workspaces.size() != hierarchy.levels.size()) {
        throw std::logic_error("Projected positive MG workspace count does not match hierarchy");
    }
    cycle(ctx, hierarchy, 0, config, workspaces, periodic_bc());
}

} // namespace multigrid
} // namespace macroflow3d
