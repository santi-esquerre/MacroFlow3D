#include "coefficient_hierarchy.cuh"

#include "../runtime/cuda_check.cuh"
#include <cmath>
#include <stdexcept>

namespace macroflow3d {
namespace multigrid {
namespace {

__global__ void coarsen_coefficient_geometric_kernel(
    real* __restrict__ coarse_coefficient, const real* __restrict__ fine_coefficient, int coarse_nx,
    int coarse_ny, int coarse_nz) {
    const int coarse_ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int coarse_iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (coarse_ix >= coarse_nx || coarse_iy >= coarse_ny)
        return;

    const int fine_nx = 2 * coarse_nx;
    const int fine_ny = 2 * coarse_ny;
    const int fine_plane = fine_nx * fine_ny;

    for (int coarse_iz = 0; coarse_iz < coarse_nz; ++coarse_iz) {
        const int fine_ix = 2 * coarse_ix;
        const int fine_iy = 2 * coarse_iy;
        const int fine_iz = 2 * coarse_iz;
        const int idx000 = fine_ix + fine_iy * fine_nx + fine_iz * fine_plane;
        const int idx001 = idx000 + fine_plane;

        const real log_sum = log(fine_coefficient[idx000]) + log(fine_coefficient[idx000 + 1]) +
                             log(fine_coefficient[idx000 + fine_nx]) +
                             log(fine_coefficient[idx000 + fine_nx + 1]) +
                             log(fine_coefficient[idx001]) + log(fine_coefficient[idx001 + 1]) +
                             log(fine_coefficient[idx001 + fine_nx]) +
                             log(fine_coefficient[idx001 + fine_nx + 1]);
        coarse_coefficient[coarse_ix + coarse_iy * coarse_nx +
                           coarse_iz * coarse_nx * coarse_ny] = exp(log_sum / real(8.0));
    }
}

bool is_isotropic(const Grid3D& grid) {
    if (!(std::isfinite(grid.dx) && std::isfinite(grid.dy) && std::isfinite(grid.dz)) ||
        grid.dx <= real(0.0) || grid.dy <= real(0.0) || grid.dz <= real(0.0)) {
        return false;
    }
    constexpr real tolerance = real(1e-12);
    return std::abs(grid.dx - grid.dy) <= tolerance * grid.dx &&
           std::abs(grid.dx - grid.dz) <= tolerance * grid.dx;
}

void validate_pair(const Grid3D& fine_grid, const Grid3D& coarse_grid) {
    if (!is_isotropic(fine_grid) || !is_isotropic(coarse_grid)) {
        throw std::invalid_argument("MG coefficient hierarchy requires positive isotropic grids");
    }
    if (fine_grid.nx != 2 * coarse_grid.nx || fine_grid.ny != 2 * coarse_grid.ny ||
        fine_grid.nz != 2 * coarse_grid.nz || coarse_grid.nx < 1 || coarse_grid.ny < 1 ||
        coarse_grid.nz < 1 || coarse_grid.dx != real(2.0) * fine_grid.dx ||
        coarse_grid.dy != real(2.0) * fine_grid.dy ||
        coarse_grid.dz != real(2.0) * fine_grid.dz) {
        throw std::invalid_argument("MG coefficient hierarchy requires exact 2x2x2 coarsening");
    }
}

} // namespace

void coarsen_coefficient_geometric(CudaContext& ctx, const Grid3D& coarse_grid,
                                   const Grid3D& fine_grid,
                                   DeviceSpan<const real> fine_coefficient,
                                   DeviceSpan<real> coarse_coefficient) {
    validate_pair(fine_grid, coarse_grid);
    if (fine_coefficient.size() != fine_grid.num_cells() ||
        coarse_coefficient.size() != coarse_grid.num_cells()) {
        throw std::invalid_argument("MG coefficient buffer size does not match its grid");
    }

    const dim3 block(16, 16);
    const dim3 grid((coarse_grid.nx + block.x - 1) / block.x,
                    (coarse_grid.ny + block.y - 1) / block.y);
    coarsen_coefficient_geometric_kernel<<<grid, block, 0, ctx.cuda_stream()>>>(
        coarse_coefficient.data(), fine_coefficient.data(), coarse_grid.nx, coarse_grid.ny,
        coarse_grid.nz);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void populate_coefficient_hierarchy(CudaContext& ctx, MGHierarchy& hierarchy,
                                    DeviceSpan<const real> fine_coefficient) {
    if (hierarchy.num_levels() == 0) {
        throw std::invalid_argument("Cannot populate an empty MG coefficient hierarchy");
    }
    if (fine_coefficient.size() != hierarchy.finest_grid().num_cells()) {
        throw std::invalid_argument("Fine coefficient size does not match the MG finest grid");
    }

    auto& finest = hierarchy.levels.front();
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(finest.coefficient.data(), fine_coefficient.data(),
                                           fine_coefficient.size() * sizeof(real),
                                           cudaMemcpyDeviceToDevice, ctx.cuda_stream()));

    for (int level = 0; level + 1 < hierarchy.num_levels(); ++level) {
        auto& fine = hierarchy.levels[level];
        auto& coarse = hierarchy.levels[level + 1];
        coarsen_coefficient_geometric(
            ctx, coarse.grid, fine.grid,
            DeviceSpan<const real>(fine.coefficient.data(), fine.coefficient.size()),
            DeviceSpan<real>(coarse.coefficient.data(), coarse.coefficient.size()));
    }
}

} // namespace multigrid
} // namespace macroflow3d
