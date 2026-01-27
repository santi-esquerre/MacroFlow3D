#include "dot.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cassert>
#include <climits>

namespace rwpt {
namespace blas {

void dot_device(CudaContext& ctx, 
                DeviceSpan<const real> x, 
                DeviceSpan<const real> y, 
                DeviceSpan<real> d_result, 
                ReductionWorkspace& ws) {
    
    if (x.size() == 0 || y.size() == 0) {
        real zero = 0.0;
        RWPT_CUDA_CHECK(cudaMemcpyAsync(d_result.data(), &zero, sizeof(real), 
                                         cudaMemcpyHostToDevice, ctx.cuda_stream()));
        return;
    }
    
    size_t n = (x.size() < y.size()) ? x.size() : y.size();
    assert(n <= INT_MAX && "dot: size exceeds cuBLAS int limit");
    
    // Use cuBLAS with device pointer mode (no host sync, no temp buffers)
    cublasHandle_t handle = ctx.cublas_handle();
    
    // Save current pointer mode and set to device
    cublasPointerMode_t old_mode;
    RWPT_CUBLAS_CHECK(cublasGetPointerMode(handle, &old_mode));
    RWPT_CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    
    // Compute dot product: result in device memory
    RWPT_CUBLAS_CHECK(cublasDdot(handle, static_cast<int>(n), 
                                  x.data(), 1, 
                                  y.data(), 1, 
                                  d_result.data()));
    
    // Restore pointer mode
    RWPT_CUBLAS_CHECK(cublasSetPointerMode(handle, old_mode));
}

real dot_host(CudaContext& ctx, 
              DeviceSpan<const real> x, 
              DeviceSpan<const real> y, 
              ReductionWorkspace& ws) {
    
    ws.ensure_scalar();
    dot_device(ctx, x, y, ws.d_scalar.span(), ws);
    
    real result;
    RWPT_CUDA_CHECK(cudaMemcpyAsync(&result, ws.d_scalar.data(), sizeof(real), 
                                     cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();
    
    return result;
}

} // namespace blas
} // namespace rwpt
