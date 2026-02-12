#include "copy.cuh"
#include "../../runtime/cuda_check.cuh"

namespace macroflow3d {
namespace blas {

__global__ void copy_kernel(const real* x, real* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        y[i] = x[i];
    }
}

void copy(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) {
    if (x.size() == 0 || y.size() == 0) return;
    
    size_t n = (x.size() < y.size()) ? x.size() : y.size();
    
    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);
    
    copy_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        x.data(), y.data(), n
    );
    
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace blas
} // namespace macroflow3d
