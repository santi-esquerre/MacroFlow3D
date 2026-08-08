#include "affine_periodic_rhs.cuh"

#include "../../numerics/operators/face_coefficient.cuh"
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

__global__ void assemble_affine_rhs_kernel(const real* q, real* rhs1, real* rhs2, int nx, int ny,
                                            int nz, real h, AffineGradient g1, AffineGradient g2) {
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                          static_cast<std::size_t>(nz);
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t x_stride = 1;
    const std::size_t y_stride = static_cast<std::size_t>(nx);
    const std::size_t z_stride = y_stride * static_cast<std::size_t>(ny);

    for (std::size_t index = start; index < n; index += stride) {
        const int i = static_cast<int>(index % static_cast<std::size_t>(nx));
        const int j = static_cast<int>((index / y_stride) % static_cast<std::size_t>(ny));
        const int k = static_cast<int>(index / z_stride);
        const std::size_t xm = i == 0 ? index + static_cast<std::size_t>(nx - 1) * x_stride
                                      : index - x_stride;
        const std::size_t xp = i + 1 == nx ? index - static_cast<std::size_t>(nx - 1) * x_stride
                                           : index + x_stride;
        const std::size_t ym = j == 0 ? index + static_cast<std::size_t>(ny - 1) * y_stride
                                      : index - y_stride;
        const std::size_t yp = j + 1 == ny ? index - static_cast<std::size_t>(ny - 1) * y_stride
                                           : index + y_stride;
        const std::size_t zm = k == 0 ? index + static_cast<std::size_t>(nz - 1) * z_stride
                                      : index - z_stride;
        const std::size_t zp = k + 1 == nz ? index - static_cast<std::size_t>(nz - 1) * z_stride
                                           : index + z_stride;
        const real qc = q[index];
        const real dx = operators::harmonic_mean_positive_cell_coefficient(qc, q[xp]) -
                        operators::harmonic_mean_positive_cell_coefficient(qc, q[xm]);
        const real dy = operators::harmonic_mean_positive_cell_coefficient(qc, q[yp]) -
                        operators::harmonic_mean_positive_cell_coefficient(qc, q[ym]);
        const real dz = operators::harmonic_mean_positive_cell_coefficient(qc, q[zp]) -
                        operators::harmonic_mean_positive_cell_coefficient(qc, q[zm]);
        rhs1[index] = (dx * g1.x + dy * g1.y + dz * g1.z) / h;
        rhs2[index] = (dx * g2.x + dy * g2.y + dz * g2.z) / h;
    }
}

void require_valid_grid(const Grid3D& grid) {
    if (grid.nx < 2 || grid.ny < 2 || grid.nz < 2) {
        throw std::invalid_argument("Affine periodic RHS requires every grid dimension to be at least 2");
    }
    if (!std::isfinite(grid.dx) || !std::isfinite(grid.dy) || !std::isfinite(grid.dz) ||
        grid.dx <= real(0) || grid.dy <= real(0) || grid.dz <= real(0) || grid.dx != grid.dy ||
        grid.dx != grid.dz) {
        throw std::invalid_argument("Affine periodic RHS requires finite positive isotropic grid spacing");
    }
}

bool finite_gradient(const AffineGradient& gradient) {
    return std::isfinite(gradient.x) && std::isfinite(gradient.y) && std::isfinite(gradient.z);
}

bool overlaps(const void* a, std::size_t a_count, const void* b, std::size_t b_count) {
    const auto a_begin = reinterpret_cast<std::uintptr_t>(a);
    const auto b_begin = reinterpret_cast<std::uintptr_t>(b);
    const auto a_bytes = a_count * sizeof(real);
    const auto b_bytes = b_count * sizeof(real);
    return a_begin < b_begin + b_bytes && b_begin < a_begin + a_bytes;
}

void require_exact_nonoverlapping_spans(DeviceSpan<const real> q, DeviceSpan<real> rhs1,
                                        DeviceSpan<real> rhs2, std::size_t n) {
    if (q.size() != n || rhs1.size() != n || rhs2.size() != n || q.data() == nullptr ||
        rhs1.data() == nullptr || rhs2.data() == nullptr) {
        throw std::invalid_argument("Affine periodic RHS requires non-null q and RHS spans of exact grid size");
    }
    if (overlaps(q.data(), n, rhs1.data(), n) || overlaps(q.data(), n, rhs2.data(), n) ||
        overlaps(rhs1.data(), n, rhs2.data(), n)) {
        throw std::invalid_argument("Affine periodic RHS q, rhs1, and rhs2 spans must not overlap");
    }
}

} // namespace

void AffinePeriodicRhsWorkspace::prepare(std::size_t n) {
    mean_zero_.prepare(n);
    raw_means_.resize(2);
    projected_means_.resize(2);
}

bool AffinePeriodicRhsWorkspace::prepared_for(std::size_t n) const noexcept {
    return mean_zero_.prepared_for(n) && raw_means_.size() == 2 && projected_means_.size() == 2;
}

std::size_t AffinePeriodicRhsWorkspace::allocated_device_bytes() const noexcept {
    return mean_zero_.allocated_device_bytes() + raw_means_.capacity() * sizeof(real) +
           projected_means_.capacity() * sizeof(real);
}

std::size_t AffinePeriodicRhsWorkspace::estimate_device_bytes(std::size_t n) {
    // Mirrors prepare(): mean_zero_.prepare(n), then raw_means_/projected_means_
    // each resized to exactly 2 elements.
    return constraints::MeanZeroWorkspace::estimate_device_bytes(n) + 2 * sizeof(real) +
           2 * sizeof(real);
}

AffineRhsDeviceDiagnostics assemble_affine_periodic_rhs(
    CudaContext& ctx, const Grid3D& grid, DeviceSpan<const real> q, const AffineGauge& gauge,
    DeviceSpan<real> rhs1, DeviceSpan<real> rhs2, AffinePeriodicRhsWorkspace& workspace) {
    require_valid_grid(grid);
    const std::size_t n = grid.num_cells();
    if (!workspace.prepared_for(n)) {
        throw std::logic_error("Affine periodic RHS requires a workspace prepared for the exact grid size");
    }
    if (!finite_gradient(gauge.psi1_gradient) || !finite_gradient(gauge.psi2_gradient)) {
        throw std::invalid_argument("Affine periodic RHS requires finite affine gauge gradients");
    }
    require_exact_nonoverlapping_spans(q, rhs1, rhs2, n);

    const std::size_t requested_blocks = (n + kBlockSize - 1) / kBlockSize;
    const int blocks = static_cast<int>(
        requested_blocks < static_cast<std::size_t>(kMaxBlocks) ? requested_blocks : kMaxBlocks);
    assemble_affine_rhs_kernel<<<blocks, kBlockSize, 0, ctx.cuda_stream()>>>(
        q.data(), rhs1.data(), rhs2.data(), grid.nx, grid.ny, grid.nz, grid.dx,
        gauge.psi1_gradient, gauge.psi2_gradient);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    const constraints::MeanZeroProjector projector;
    const auto capture_mean = [&ctx](DeviceSpan<const real> source, DeviceSpan<real> destination) {
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(destination.data(), source.data(), sizeof(real),
                                               cudaMemcpyDeviceToDevice, ctx.cuda_stream()));
    };
    auto raw_means = workspace.raw_means_.span();
    auto projected_means = workspace.projected_means_.span();

    capture_mean(projector.mean_device(ctx, DeviceSpan<const real>(rhs1), workspace.mean_zero_),
                 DeviceSpan<real>(raw_means.data(), 1));
    projector.project(ctx, rhs1, workspace.mean_zero_);
    capture_mean(projector.mean_device(ctx, DeviceSpan<const real>(rhs1), workspace.mean_zero_),
                 DeviceSpan<real>(projected_means.data(), 1));

    capture_mean(projector.mean_device(ctx, DeviceSpan<const real>(rhs2), workspace.mean_zero_),
                 DeviceSpan<real>(raw_means.data() + 1, 1));
    projector.project(ctx, rhs2, workspace.mean_zero_);
    capture_mean(projector.mean_device(ctx, DeviceSpan<const real>(rhs2), workspace.mean_zero_),
                 DeviceSpan<real>(projected_means.data() + 1, 1));

    return {workspace.raw_means_.span(), workspace.projected_means_.span()};
}

AffineRhsHostDiagnostics synchronize_affine_rhs_diagnostics(
    CudaContext& ctx, const AffineRhsDeviceDiagnostics& diagnostics) {
    if (diagnostics.raw_means.size() != 2 || diagnostics.projected_means.size() != 2 ||
        diagnostics.raw_means.data() == nullptr || diagnostics.projected_means.data() == nullptr) {
        throw std::invalid_argument("Affine RHS diagnostics require exactly two non-null device means per kind");
    }
    AffineRhsHostDiagnostics result;
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(result.raw_means.data(), diagnostics.raw_means.data(),
                                           2 * sizeof(real), cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(result.projected_means.data(), diagnostics.projected_means.data(),
                                           2 * sizeof(real), cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();
    return result;
}

} // namespace streamfunctions
} // namespace macroflow3d
