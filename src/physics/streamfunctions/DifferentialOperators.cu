#include "DifferentialOperators.cuh"

#include "../../runtime/cuda_check.cuh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace macroflow3d {
namespace streamfunctions {
namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 65535;

__global__ void total_streamfunction_gradients_kernel(
    const real* u1, const real* u2, real* psi1_x, real* psi1_y, real* psi1_z, real* psi2_x,
    real* psi2_y, real* psi2_z, int nx, int ny, int nz, real dx, real dy, real dz,
    AffineGradient g1, AffineGradient g2) {
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                          static_cast<std::size_t>(nz);
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t y_stride = static_cast<std::size_t>(nx);
    const std::size_t z_stride = y_stride * static_cast<std::size_t>(ny);

    for (std::size_t index = start; index < n; index += stride) {
        const int i = static_cast<int>(index % static_cast<std::size_t>(nx));
        const int j = static_cast<int>((index / y_stride) % static_cast<std::size_t>(ny));
        const int k = static_cast<int>(index / z_stride);
        const std::size_t xm = i == 0 ? index + static_cast<std::size_t>(nx - 1) : index - 1;
        const std::size_t xp = i + 1 == nx ? index - static_cast<std::size_t>(nx - 1) : index + 1;
        const std::size_t ym = j == 0 ? index + static_cast<std::size_t>(ny - 1) * y_stride
                                      : index - y_stride;
        const std::size_t yp = j + 1 == ny ? index - static_cast<std::size_t>(ny - 1) * y_stride
                                           : index + y_stride;
        const std::size_t zm = k == 0 ? index + static_cast<std::size_t>(nz - 1) * z_stride
                                      : index - z_stride;
        const std::size_t zp = k + 1 == nz ? index - static_cast<std::size_t>(nz - 1) * z_stride
                                           : index + z_stride;

        psi1_x[index] = (u1[xp] - u1[xm]) / (real{2} * dx) + g1.x;
        psi1_y[index] = (u1[yp] - u1[ym]) / (real{2} * dy) + g1.y;
        psi1_z[index] = (u1[zp] - u1[zm]) / (real{2} * dz) + g1.z;
        psi2_x[index] = (u2[xp] - u2[xm]) / (real{2} * dx) + g2.x;
        psi2_y[index] = (u2[yp] - u2[ym]) / (real{2} * dy) + g2.y;
        psi2_z[index] = (u2[zp] - u2[zm]) / (real{2} * dz) + g2.z;
    }
}

std::size_t require_valid_grid(const Grid3D& grid) {
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
        throw std::invalid_argument("Total streamfunction gradients require positive grid extents");
    }
    if (!std::isfinite(grid.dx) || !std::isfinite(grid.dy) || !std::isfinite(grid.dz) ||
        grid.dx <= real{0} || grid.dy <= real{0} || grid.dz <= real{0}) {
        throw std::invalid_argument(
            "Total streamfunction gradients require finite positive direction-specific spacing");
    }

    const auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(real);
    const auto nx = static_cast<std::size_t>(grid.nx);
    const auto ny = static_cast<std::size_t>(grid.ny);
    const auto nz = static_cast<std::size_t>(grid.nz);
    if (nx > max_elements / ny || nx * ny > max_elements / nz) {
        throw std::invalid_argument(
            "Total streamfunction gradients require a grid whose storage size does not overflow");
    }
    return nx * ny * nz;
}

bool finite_gradient(const AffineGradient& gradient) {
    return std::isfinite(gradient.x) && std::isfinite(gradient.y) && std::isfinite(gradient.z);
}

bool overlaps(const void* a, std::size_t a_count, const void* b, std::size_t b_count) {
    const auto a_begin = reinterpret_cast<std::uintptr_t>(a);
    const auto b_begin = reinterpret_cast<std::uintptr_t>(b);
    const auto a_bytes = a_count * sizeof(real);
    const auto b_bytes = b_count * sizeof(real);
    if (a_begin <= b_begin) {
        return b_begin - a_begin < a_bytes;
    }
    return a_begin - b_begin < b_bytes;
}

void require_exact_nonoverlapping_spans(const PeriodicStreamfunctionFluctuations& fluctuations,
                                        const TotalStreamfunctionGradientOutput& output,
                                        std::size_t n) {
    const DeviceSpan<real> outputs[] = {output.psi1_x, output.psi1_y, output.psi1_z,
                                        output.psi2_x, output.psi2_y, output.psi2_z};
    if (fluctuations.u1.size() != n || fluctuations.u2.size() != n ||
        fluctuations.u1.data() == nullptr || fluctuations.u2.data() == nullptr) {
        throw std::invalid_argument(
            "Total streamfunction gradients require non-null fluctuation spans of exact grid size");
    }
    for (const auto span : outputs) {
        if (span.size() != n || span.data() == nullptr) {
            throw std::invalid_argument(
                "Total streamfunction gradients require non-null output spans of exact grid size");
        }
        if (overlaps(span.data(), n, fluctuations.u1.data(), n) ||
            overlaps(span.data(), n, fluctuations.u2.data(), n)) {
            throw std::invalid_argument(
                "Total streamfunction gradient outputs must not overlap fluctuation inputs");
        }
    }
    for (std::size_t first = 0; first < 6; ++first) {
        for (std::size_t second = first + 1; second < 6; ++second) {
            if (overlaps(outputs[first].data(), n, outputs[second].data(), n)) {
                throw std::invalid_argument(
                    "Total streamfunction gradient output spans must not overlap");
            }
        }
    }
}

} // namespace

void enqueue_total_streamfunction_gradients(
    CudaContext& ctx, const Grid3D& grid,
    const PeriodicStreamfunctionFluctuations& fluctuations, const AffineGauge& gauge,
    const TotalStreamfunctionGradientOutput& output) {
    const std::size_t n = require_valid_grid(grid);
    if (!finite_gradient(gauge.psi1_gradient) || !finite_gradient(gauge.psi2_gradient)) {
        throw std::invalid_argument("Total streamfunction gradients require finite affine gradients");
    }
    require_exact_nonoverlapping_spans(fluctuations, output, n);

    const std::size_t requested_blocks = (n + kBlockSize - 1) / kBlockSize;
    const int blocks = static_cast<int>(
        requested_blocks < static_cast<std::size_t>(kMaxBlocks) ? requested_blocks : kMaxBlocks);
    total_streamfunction_gradients_kernel<<<blocks, kBlockSize, 0, ctx.cuda_stream()>>>(
        fluctuations.u1.data(), fluctuations.u2.data(), output.psi1_x.data(), output.psi1_y.data(),
        output.psi1_z.data(), output.psi2_x.data(), output.psi2_y.data(), output.psi2_z.data(),
        grid.nx, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz, gauge.psi1_gradient,
        gauge.psi2_gradient);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace streamfunctions
} // namespace macroflow3d
