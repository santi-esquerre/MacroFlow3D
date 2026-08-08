#include "MeanZeroProjector.cuh"

#include "../../runtime/cuda_check.cuh"
#include "../blas/scal.cuh"
#include "../blas/sum.cuh"

#include <algorithm>
#include <stdexcept>

namespace macroflow3d {
namespace constraints {
namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 65535;

void require_prepared(std::size_t n, const MeanZeroWorkspace& workspace) {
    if (!workspace.prepared_for(n)) {
        throw std::logic_error(
            "MeanZeroProjector requires a workspace prepared for the exact field size");
    }
}

__global__ void subtract_mean_kernel(real* x, const real* mean, std::size_t n) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const real value = mean[0];

    for (std::size_t i = index; i < n; i += stride) {
        x[i] -= value;
    }
}

} // namespace

void MeanZeroWorkspace::prepare(std::size_t n) {
    // Keep reduction ownership and exact-size metadata in one place.  The
    // wrapped preparation is the only permitted resize/query path.
    blas::prepare_sum_workspace(n, reduction_);
    prepared_n_ = n;
    prepared_ = true;
}

bool MeanZeroWorkspace::prepared_for(std::size_t n) const noexcept {
    return prepared_ && prepared_n_ == n && blas::sum_workspace_ready(n, reduction_);
}

std::size_t MeanZeroWorkspace::temporary_storage_capacity_bytes() const noexcept {
    return reduction_.temp.capacity();
}

std::size_t MeanZeroWorkspace::allocated_device_bytes() const noexcept {
    return blas::reduction_workspace_allocated_bytes(reduction_);
}

std::size_t MeanZeroWorkspace::estimate_device_bytes(std::size_t n) {
    // Mirrors prepare(): blas::prepare_sum_workspace(n, reduction_) resizes
    // reduction_.temp to blas::sum_workspace_bytes(n) and reduction_.d_scalar
    // to exactly one element (already guaranteed by ReductionWorkspace's
    // default constructor).
    return blas::sum_workspace_bytes(n) + sizeof(real);
}

DeviceSpan<const real> MeanZeroProjector::mean_device(CudaContext& ctx, DeviceSpan<const real> x,
                                                       MeanZeroWorkspace& workspace) const {
    require_prepared(x.size(), workspace);

    auto d_mean = workspace.reduction_.d_scalar.span();
    blas::sum_device(ctx, x, d_mean, workspace.reduction_);
    if (!x.empty()) {
        blas::scal(ctx, d_mean, real(1.0) / static_cast<real>(x.size()));
    }
    return DeviceSpan<const real>(d_mean);
}

void MeanZeroProjector::project(CudaContext& ctx, DeviceSpan<real> x,
                                MeanZeroWorkspace& workspace) const {
    require_prepared(x.size(), workspace);
    if (x.empty()) {
        return;
    }

    const auto d_mean = mean_device(ctx, DeviceSpan<const real>(x), workspace);
    const std::size_t requested_blocks = (x.size() + kBlockSize - 1) / kBlockSize;
    const int grid_size = static_cast<int>(
        std::min<std::size_t>(requested_blocks, static_cast<std::size_t>(kMaxBlocks)));
    subtract_mean_kernel<<<grid_size, kBlockSize, 0, ctx.cuda_stream()>>>(x.data(), d_mean.data(),
                                                                            x.size());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

real MeanZeroProjector::mean_host(CudaContext& ctx, DeviceSpan<const real> x,
                                  MeanZeroWorkspace& workspace) const {
    const auto d_mean = mean_device(ctx, x, workspace);
    real result = real(0.0);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(&result, d_mean.data(), sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();
    return result;
}

} // namespace constraints
} // namespace macroflow3d
