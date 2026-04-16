#include "../../runtime/cuda_check.cuh"
#include "nrm2.cuh"
#include <cassert>
#include <climits>

namespace macroflow3d {
namespace blas {

void nrm2_device(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> d_result,
                 ReductionWorkspace& ws) {

    if (x.size() == 0) {
        real zero = 0.0;
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_result.data(), &zero, sizeof(real),
                                               cudaMemcpyHostToDevice, ctx.cuda_stream()));
        return;
    }

    size_t n = x.size();
    assert(n <= INT_MAX && "nrm2: size exceeds cuBLAS int limit");

    // Use cuBLAS with device pointer mode (no host sync, no temp buffers)
    cublasHandle_t handle = ctx.cublas_handle();

    // Save current pointer mode and set to device
    cublasPointerMode_t old_mode;
    MACROFLOW3D_CUBLAS_CHECK(cublasGetPointerMode(handle, &old_mode));
    MACROFLOW3D_CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    // Compute L2 norm: result in device memory
    MACROFLOW3D_CUBLAS_CHECK(
        cublasDnrm2(handle, static_cast<int>(n), x.data(), 1, d_result.data()));

    // Restore pointer mode
    MACROFLOW3D_CUBLAS_CHECK(cublasSetPointerMode(handle, old_mode));
}

real nrm2_host(CudaContext& ctx, DeviceSpan<const real> x, ReductionWorkspace& ws) {

    ws.ensure_scalar();
    nrm2_device(ctx, x, ws.d_scalar.span(), ws);

    real result;
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(&result, ws.d_scalar.data(), sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();

    return result;
}

} // namespace blas
} // namespace macroflow3d
