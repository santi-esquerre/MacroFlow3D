#include "../../runtime/cuda_check.cuh"
#include "sum.cuh"

#include <cub/device/device_reduce.cuh>

#include <climits>
#include <stdexcept>

namespace macroflow3d {
namespace blas {
namespace {

void require_sum_size(std::size_t n) {
    if (n > static_cast<std::size_t>(INT_MAX)) {
        throw std::overflow_error("sum supports at most INT_MAX input values");
    }
}

} // namespace

std::size_t sum_workspace_bytes(std::size_t n) {
    require_sum_size(n);
    if (n == 0) {
        return 0;
    }

    std::size_t bytes = 0;
    MACROFLOW3D_CUDA_CHECK(cub::DeviceReduce::Sum(
        nullptr, bytes, static_cast<const real*>(nullptr), static_cast<real*>(nullptr), n));
    return bytes;
}

void prepare_sum_workspace(std::size_t n, ReductionWorkspace& ws) {
    const std::size_t required_bytes = sum_workspace_bytes(n);

    // All possible allocation is deliberately confined to preparation.
    ws.ensure_scalar();
    ws.ensure_bytes(required_bytes);
    ws.sum_is_prepared = true;
    ws.sum_prepared_size = n;
    ws.sum_prepared_bytes = required_bytes;
}

bool sum_workspace_ready(std::size_t n, const ReductionWorkspace& ws) noexcept {
    return ws.sum_is_prepared && ws.sum_prepared_size == n && ws.d_scalar.size() >= 1 &&
           ws.temp.capacity() >= ws.sum_prepared_bytes;
}

void sum_device(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> d_result,
                ReductionWorkspace& ws) {
    if (d_result.size() < 1) {
        throw std::invalid_argument("sum_device requires at least one output scalar");
    }
    if (!sum_workspace_ready(x.size(), ws)) {
        throw std::logic_error("sum_device requires a workspace prepared for the exact input size");
    }

    if (x.size() == 0) {
        MACROFLOW3D_CUDA_CHECK(
            cudaMemsetAsync(d_result.data(), 0, sizeof(real), ctx.cuda_stream()));
        return;
    }

    std::size_t bytes = ws.sum_prepared_bytes;
    MACROFLOW3D_CUDA_CHECK(cub::DeviceReduce::Sum(ws.temp.data(), bytes, x.data(), d_result.data(),
                                                   x.size(), ctx.cuda_stream()));
}

real sum_host(CudaContext& ctx, DeviceSpan<const real> x, ReductionWorkspace& ws) {
    if (!sum_workspace_ready(x.size(), ws)) {
        throw std::logic_error("sum_host requires a workspace prepared for the exact input size");
    }

    sum_device(ctx, x, ws.d_scalar.span(), ws);

    real result;
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(&result, ws.d_scalar.data(), sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();
    return result;
}

} // namespace blas
} // namespace macroflow3d
