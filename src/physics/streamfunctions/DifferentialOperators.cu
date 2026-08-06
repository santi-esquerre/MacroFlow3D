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

__global__ void streamfunction_hessian_vector_b_kernel(
    const real* u1, const real* u2, const real* psi1_x, const real* psi1_y,
    const real* psi1_z, const real* psi2_x, const real* psi2_y, const real* psi2_z,
    real* hessian_psi2_times_grad_psi1_x, real* hessian_psi2_times_grad_psi1_y,
    real* hessian_psi2_times_grad_psi1_z, real* hessian_psi1_times_grad_psi2_x,
    real* hessian_psi1_times_grad_psi2_y, real* hessian_psi1_times_grad_psi2_z,
    real* b_x, real* b_y, real* b_z, int nx, int ny, int nz, real dx, real dy, real dz) {
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                          static_cast<std::size_t>(nz);
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t y_stride = static_cast<std::size_t>(nx);
    const std::size_t z_stride = y_stride * static_cast<std::size_t>(ny);
    const real inverse_dx2 = real{1} / (dx * dx);
    const real inverse_dy2 = real{1} / (dy * dy);
    const real inverse_dz2 = real{1} / (dz * dz);
    const real inverse_four_dxdy = real{1} / (real{4} * dx * dy);
    const real inverse_four_dxdz = real{1} / (real{4} * dx * dz);
    const real inverse_four_dydz = real{1} / (real{4} * dy * dz);

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

        // The twelve edge-diagonal neighbors complete the radius-one,
        // 19-point union stencil together with center and six axial points.
        const std::size_t im = static_cast<std::size_t>(i == 0 ? nx - 1 : i - 1);
        const std::size_t ip = static_cast<std::size_t>(i + 1 == nx ? 0 : i + 1);
        const std::size_t jm = static_cast<std::size_t>(j == 0 ? ny - 1 : j - 1);
        const std::size_t jp = static_cast<std::size_t>(j + 1 == ny ? 0 : j + 1);
        const std::size_t km = static_cast<std::size_t>(k == 0 ? nz - 1 : k - 1);
        const std::size_t kp = static_cast<std::size_t>(k + 1 == nz ? 0 : k + 1);
        const std::size_t k_offset = static_cast<std::size_t>(k) * z_stride;
        const std::size_t km_offset = km * z_stride;
        const std::size_t kp_offset = kp * z_stride;
        const std::size_t xmym = im + jm * y_stride + k_offset;
        const std::size_t xmyp = im + jp * y_stride + k_offset;
        const std::size_t xpym = ip + jm * y_stride + k_offset;
        const std::size_t xpyp = ip + jp * y_stride + k_offset;
        const std::size_t xmzm = im + static_cast<std::size_t>(j) * y_stride + km_offset;
        const std::size_t xmzp = im + static_cast<std::size_t>(j) * y_stride + kp_offset;
        const std::size_t xpzm = ip + static_cast<std::size_t>(j) * y_stride + km_offset;
        const std::size_t xpzp = ip + static_cast<std::size_t>(j) * y_stride + kp_offset;
        const std::size_t ymzm = static_cast<std::size_t>(i) + jm * y_stride + km_offset;
        const std::size_t ymzp = static_cast<std::size_t>(i) + jm * y_stride + kp_offset;
        const std::size_t ypzm = static_cast<std::size_t>(i) + jp * y_stride + km_offset;
        const std::size_t ypzp = static_cast<std::size_t>(i) + jp * y_stride + kp_offset;

        const real u1_center = u1[index];
        const real u2_center = u2[index];
        const real u1_xx = (u1[xp] - real{2} * u1_center + u1[xm]) * inverse_dx2;
        const real u1_yy = (u1[yp] - real{2} * u1_center + u1[ym]) * inverse_dy2;
        const real u1_zz = (u1[zp] - real{2} * u1_center + u1[zm]) * inverse_dz2;
        const real u1_xy = (u1[xpyp] - u1[xpym] - u1[xmyp] + u1[xmym]) * inverse_four_dxdy;
        const real u1_xz = (u1[xpzp] - u1[xpzm] - u1[xmzp] + u1[xmzm]) * inverse_four_dxdz;
        const real u1_yz = (u1[ypzp] - u1[ypzm] - u1[ymzp] + u1[ymzm]) * inverse_four_dydz;
        const real u2_xx = (u2[xp] - real{2} * u2_center + u2[xm]) * inverse_dx2;
        const real u2_yy = (u2[yp] - real{2} * u2_center + u2[ym]) * inverse_dy2;
        const real u2_zz = (u2[zp] - real{2} * u2_center + u2[zm]) * inverse_dz2;
        const real u2_xy = (u2[xpyp] - u2[xpym] - u2[xmyp] + u2[xmym]) * inverse_four_dxdy;
        const real u2_xz = (u2[xpzp] - u2[xpzm] - u2[xmzp] + u2[xmzm]) * inverse_four_dxdz;
        const real u2_yz = (u2[ypzp] - u2[ypzm] - u2[ymzp] + u2[ymzm]) * inverse_four_dydz;

        const real h2g1_x = u2_xx * psi1_x[index] + u2_xy * psi1_y[index] +
                            u2_xz * psi1_z[index];
        const real h2g1_y = u2_xy * psi1_x[index] + u2_yy * psi1_y[index] +
                            u2_yz * psi1_z[index];
        const real h2g1_z = u2_xz * psi1_x[index] + u2_yz * psi1_y[index] +
                            u2_zz * psi1_z[index];
        const real h1g2_x = u1_xx * psi2_x[index] + u1_xy * psi2_y[index] +
                            u1_xz * psi2_z[index];
        const real h1g2_y = u1_xy * psi2_x[index] + u1_yy * psi2_y[index] +
                            u1_yz * psi2_z[index];
        const real h1g2_z = u1_xz * psi2_x[index] + u1_yz * psi2_y[index] +
                            u1_zz * psi2_z[index];

        hessian_psi2_times_grad_psi1_x[index] = h2g1_x;
        hessian_psi2_times_grad_psi1_y[index] = h2g1_y;
        hessian_psi2_times_grad_psi1_z[index] = h2g1_z;
        hessian_psi1_times_grad_psi2_x[index] = h1g2_x;
        hessian_psi1_times_grad_psi2_y[index] = h1g2_y;
        hessian_psi1_times_grad_psi2_z[index] = h1g2_z;
        b_x[index] = h2g1_x - h1g2_x;
        b_y[index] = h2g1_y - h1g2_y;
        b_z[index] = h2g1_z - h1g2_z;
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

void require_exact_hessian_vector_b_spans(
    const PeriodicStreamfunctionFluctuations& fluctuations,
    const TotalStreamfunctionGradientView& total_gradients, const HessianVectorBOutput& output,
    std::size_t n) {
    const DeviceSpan<const real> inputs[] = {
        DeviceSpan<const real>(fluctuations.u1), DeviceSpan<const real>(fluctuations.u2),
        total_gradients.psi1_x, total_gradients.psi1_y, total_gradients.psi1_z,
        total_gradients.psi2_x, total_gradients.psi2_y, total_gradients.psi2_z};
    const DeviceSpan<real> outputs[] = {
        output.hessian_psi2_times_grad_psi1_x, output.hessian_psi2_times_grad_psi1_y,
        output.hessian_psi2_times_grad_psi1_z, output.hessian_psi1_times_grad_psi2_x,
        output.hessian_psi1_times_grad_psi2_y, output.hessian_psi1_times_grad_psi2_z,
        output.b_x, output.b_y, output.b_z};

    for (const auto input : inputs) {
        if (input.size() != n || input.data() == nullptr) {
            throw std::invalid_argument(
                "Streamfunction Hessian-vector B requires non-null inputs of exact grid size");
        }
    }
    for (const auto output_span : outputs) {
        if (output_span.size() != n || output_span.data() == nullptr) {
            throw std::invalid_argument(
                "Streamfunction Hessian-vector B requires non-null outputs of exact grid size");
        }
        for (const auto input : inputs) {
            if (overlaps(output_span.data(), n, input.data(), n)) {
                throw std::invalid_argument(
                    "Streamfunction Hessian-vector B outputs must not overlap any input");
            }
        }
    }
    for (std::size_t first = 0; first < 9; ++first) {
        for (std::size_t second = first + 1; second < 9; ++second) {
            if (overlaps(outputs[first].data(), n, outputs[second].data(), n)) {
                throw std::invalid_argument(
                    "Streamfunction Hessian-vector B output spans must not overlap");
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

void enqueue_streamfunction_hessian_vector_b(
    CudaContext& ctx, const Grid3D& grid,
    const PeriodicStreamfunctionFluctuations& fluctuations,
    const TotalStreamfunctionGradientView& total_gradients,
    const HessianVectorBOutput& output) {
    const std::size_t n = require_valid_grid(grid);
    require_exact_hessian_vector_b_spans(fluctuations, total_gradients, output, n);

    const std::size_t requested_blocks = (n + kBlockSize - 1) / kBlockSize;
    const int blocks = static_cast<int>(
        requested_blocks < static_cast<std::size_t>(kMaxBlocks) ? requested_blocks : kMaxBlocks);
    streamfunction_hessian_vector_b_kernel<<<blocks, kBlockSize, 0, ctx.cuda_stream()>>>(
        fluctuations.u1.data(), fluctuations.u2.data(), total_gradients.psi1_x.data(),
        total_gradients.psi1_y.data(), total_gradients.psi1_z.data(), total_gradients.psi2_x.data(),
        total_gradients.psi2_y.data(), total_gradients.psi2_z.data(),
        output.hessian_psi2_times_grad_psi1_x.data(),
        output.hessian_psi2_times_grad_psi1_y.data(),
        output.hessian_psi2_times_grad_psi1_z.data(),
        output.hessian_psi1_times_grad_psi2_x.data(),
        output.hessian_psi1_times_grad_psi2_y.data(),
        output.hessian_psi1_times_grad_psi2_z.data(), output.b_x.data(), output.b_y.data(),
        output.b_z.data(), grid.nx, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace streamfunctions
} // namespace macroflow3d
