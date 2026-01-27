#include "scal.cuh"
#include "../../runtime/cuda_check.cuh"

namespace rwpt {
namespace blas {

__global__ void scal_kernel(real* x, size_t n, real a) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        x[i] = a * x[i];
    }
}

void scal(CudaContext& ctx, DeviceSpan<real> x, real a) {
    if (x.size() == 0) return;
    
    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((x.size() + block_size - 1) / block_size), max_blocks);
    
    scal_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        x.data(), x.size(), a
    );
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace blas
} // namespace rwpt
