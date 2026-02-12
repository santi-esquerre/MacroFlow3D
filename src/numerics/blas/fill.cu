#include "fill.cuh"
#include "../../runtime/cuda_check.cuh"

namespace macroflow3d {
namespace blas {

__global__ void fill_kernel(real* y, size_t n, real value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        y[i] = value;
    }
}

void fill(CudaContext& ctx, DeviceSpan<real> y, real value) {
    if (y.size() == 0) return;
    
    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((y.size() + block_size - 1) / block_size), max_blocks);
    
    fill_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        y.data(), y.size(), value
    );
    
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace blas
} // namespace macroflow3d
